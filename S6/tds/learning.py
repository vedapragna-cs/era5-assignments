"""The learning ledger: what came back from the model, attached to the data that caused it.

The consumption ledger answers *what did the model see*. This answers *what did it get from it*, and
it is the artifact the session cares most about:

    "nobody in the world releases a perplexity dataset ... this is the most important data ...
     literally maybe 10 lines of code and I could have done it [in V4]"

Three storage tiers, because full token traces do not survive a real run's volume (the session raises
the 10M-step / billion-point problem directly):

    full       - every loss-bearing token. Proxy runs, debug runs, suspicious shards.
    quantized  - loss rounded to 2 dp, sampled every Nth token. Selected intervals of a large run.
    aggregate  - per shard / lane / language / position-bucket statistics only. The full run.

`ppl = exp(loss)`. Shard-level perplexity is `exp(mean loss)`, i.e. the geometric mean of per-token
perplexity, which is the only average that composes correctly - averaging perplexities directly is
dominated by the single worst token.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

HOT_PPL_THRESHOLD = 20.0
N_POSITION_BUCKETS = 10


@dataclass
class TokenObservation:
    step: int
    sample_id: str
    shard_id: str
    doc_id: str
    lane: str
    language: str
    script: str
    position_in_pack: int
    position_in_doc: int
    position_bucket: int
    is_boundary: bool
    is_special: bool
    token_id: int
    decoded: str
    loss: float
    opus_decision_id: str | None
    model_age_tokens: int
    stage: str
    pass_number: int
    # The segment role this token came from (`tool_call`, `tool_observation`, `text`, ...). Lane is
    # the coarsest thing the ledger could attribute loss to and it cannot answer S5's question,
    # which is about which PART of an agentic trajectory is expensive. Required rather than
    # defaulted: a silently-empty role would produce a by-role table that looks complete.
    role: str
    # Which branch produced this observation, and whether the batch was later superseded. Without
    # these the ledger silently folds fork, control-fork, replay and ablation branches - and the
    # pre-crash batches the consumption ledger correctly supersedes - into every aggregate. Measured
    # before this existed: 16 duplicate (global_step, shard_id) pairs in learning_shards.jsonl.
    branch_id: str = "run-a"
    superseded: bool = False
    # Excess loss against a frozen reference model (tds/reference.py). None when no reference is
    # configured, so the field is never silently zero - "not measured" and "no excess" are different.
    reference_loss: float | None = None
    excess_loss: float | None = None
    learnability: str | None = None

    @property
    def ppl(self) -> float:
        return math.exp(min(self.loss, 30.0))

    def to_record(self, *, quantize: bool = False) -> dict:
        return {
            "global_step": self.step, "sample_id": self.sample_id, "shard_id": self.shard_id,
            "doc_id": self.doc_id, "capability_lane": self.lane, "language": self.language,
            "script": self.script, "position_in_pack": self.position_in_pack,
            "position_in_doc": self.position_in_doc, "position_bucket": self.position_bucket,
            "boundary_flag": self.is_boundary, "special_token_flag": self.is_special,
            "token_id": self.token_id, "decoded_preview": self.decoded,
            "cross_entropy_loss": round(self.loss, 2 if quantize else 6),
            "token_perplexity": round(self.ppl, 2 if quantize else 4),
            "opus_decision_id": self.opus_decision_id,
            "model_age_tokens": self.model_age_tokens, "curriculum_stage": self.stage,
            "repeated_pass_number": self.pass_number, "segment_role": self.role,
            "branch_id": self.branch_id, "superseded": self.superseded,
            "reference_loss": None if self.reference_loss is None
                              else round(self.reference_loss, 2 if quantize else 6),
            "excess_loss": None if self.excess_loss is None
                           else round(self.excess_loss, 2 if quantize else 6),
            "learnability": self.learnability,
        }


def _geo_ppl(losses: list[float]) -> float:
    return math.exp(sum(losses) / len(losses)) if losses else float("nan")


@dataclass
class LearningLedger:
    tier: str = "full"                 # full | quantized | aggregate
    sample_every: int = 1
    tokens: list[TokenObservation] = field(default_factory=list)
    n_observed: int = 0
    shard_exposures: list[dict] = field(default_factory=list)

    def observe_tokens(self, obs: list[TokenObservation]) -> None:
        self.n_observed += len(obs)
        if self.tier == "aggregate":
            self._fold(obs)
            return
        keep = obs if self.sample_every <= 1 else obs[::self.sample_every]
        self.tokens.extend(keep)
        self._fold(obs)

    _agg: dict = field(default_factory=dict)

    def _fold(self, obs: list[TokenObservation]) -> None:
        # Bucketed by branch. A single global fold made every aggregate a mixture of run-a, the
        # forks, the control fork, replay re-derivations and the spike ablation.
        for o in obs:
            br = self._agg.setdefault(o.branch_id, {})
            for key in (("shard", o.shard_id), ("lane", o.lane), ("language", o.language),
                        ("bucket", str(o.position_bucket)), ("stage", o.stage),
                        ("boundary", str(o.is_boundary)), ("role", o.role)):
                d = br.setdefault(key[0], {}).setdefault(key[1],
                                                         {"n": 0, "sum_loss": 0.0, "hot": 0})
                d["n"] += 1
                d["sum_loss"] += o.loss
                if o.ppl >= HOT_PPL_THRESHOLD:
                    d["hot"] += 1

    _DIMS = {"shard": lambda o: o.shard_id, "lane": lambda o: o.lane,
             "language": lambda o: o.language, "bucket": lambda o: str(o.position_bucket),
             "stage": lambda o: o.stage, "boundary": lambda o: str(o.is_boundary),
             "role": lambda o: o.role}

    def aggregate(self, dim: str, branch: str = "run-a") -> dict[str, dict]:
        """Per-dimension aggregates over ONE branch's surviving observations.

        This used to read the running `_fold` totals, which are accumulated on every `train_step` of
        every branch with no supersession filter. `effective_tokens` was written precisely to
        exclude that population and documented why - but only three consumers used it, and
        `aggregate` was not one of them. The shipped artifacts contradicted each other as a result:
        `evidence.json` reported `tokens_traced = 4457` and, in the same record, per-lane counts
        summing to 23252, folding in run-b, run-control, replay re-derivations, the spike ablation
        branch and the pre-crash batches supersession had retired. Recomputing on run-a-live alone
        changed the lane ORDERING, so the ranked table was a statement about no artifact a grader
        could read.

        `_agg` survives only for `tier == "aggregate"`, where token records are deliberately
        discarded and a running fold is the only thing left; that path reports its own scope.
        """
        if self.tier == "aggregate":
            src = self._agg.get(branch, {}).get(dim, {})
            out = {}
            for k, d in sorted(src.items()):
                mean = d["sum_loss"] / d["n"] if d["n"] else float("nan")
                out[k] = {"n_tokens": d["n"], "mean_loss": round(mean, 6),
                          "geometric_mean_ppl": round(math.exp(min(mean, 30.0)), 4),
                          "hot_token_pct": round(100 * d["hot"] / d["n"], 2) if d["n"] else 0.0,
                          "scope_caveat": "aggregate tier: folded at observation time, so "
                                          "supersession could not be retracted"}
            return out

        key = self._DIMS[dim]
        buckets: dict[str, list] = {}
        for o in self.effective_tokens(branch):
            buckets.setdefault(key(o), []).append(o)
        out = {}
        for k, obs in sorted(buckets.items()):
            mean = sum(o.loss for o in obs) / len(obs)
            hot = sum(1 for o in obs if o.ppl >= HOT_PPL_THRESHOLD)
            out[k] = {"n_tokens": len(obs), "mean_loss": round(mean, 6),
                      "geometric_mean_ppl": round(math.exp(min(mean, 30.0)), 4),
                      "hot_token_pct": round(100 * hot / len(obs), 2)}
        return out

    # ------------------------------------------------------------------ shard exposure
    def record_exposure(self, *, shard_id: str, lane: str, step: int, stage: str,
                        model_age_tokens: int, losses: list[float], gradient_norm: float,
                        opus_score: float, loss_before: float, loss_after: float,
                        pass_number: int, branch_id: str = "run-a") -> dict:
        mean = sum(losses) / len(losses) if losses else float("nan")
        hot = sum(1 for l in losses if math.exp(min(l, 30.0)) >= HOT_PPL_THRESHOLD)
        rec = {
            "shard_id": shard_id, "capability_lane": lane, "global_step": step,
            "branch_id": branch_id,
            "curriculum_stage": stage, "model_phase": phase_of(model_age_tokens),
            "tokens_consumed_when_seen": model_age_tokens,
            "n_loss_bearing_tokens": len(losses),
            "average_token_loss": round(mean, 6),
            "geometric_mean_ppl": round(math.exp(min(mean, 30.0)), 4),
            "high_perplexity_token_pct": round(100 * hot / len(losses), 2) if losses else 0.0,
            "gradient_norm": round(gradient_norm, 6),
            "opus_score": round(opus_score, 6),
            "loss_delta_before_after": round(loss_before - loss_after, 6),
            "repeated_pass_number": pass_number,
        }
        rec["usefulness"] = classify(rec)
        # Attribution safety is applied later, once the run's spike events are known: a shard
        # exposed inside a spike window has its verdict withheld, never guessed. See
        # LearningLedger.apply_spike_attribution and tds/spike.py.
        rec["attribution_uncertain"] = False
        rec["superseded"] = False
        self.shard_exposures.append(rec)
        return rec

    def effective_tokens(self, branch: str = "run-a") -> list:
        """Observations that a surviving model state actually consumed, on one branch.

        Every aggregate below is computed over this rather than over `self.tokens`, because
        `self.tokens` also holds fork branches, control forks, replay re-derivations, spike
        ablations and the pre-crash batches that supersession retired. An audit measured the
        consequence: 16 duplicate (step, shard) pairs, with steps 24-25 counted twice.
        """
        return [o for o in self.tokens if o.branch_id == branch and not o.superseded]

    def mark_superseded(self, branch: str, from_step: int) -> dict:
        """Flag observations a crash orphaned, mirroring the consumption ledger's supersession.

        Called from `resume` BEFORE the orphaned steps are re-run, so only the pre-crash records are
        touched; the regenerated ones are appended afterwards and stay live. Without this the
        learning ledger counts both executions of every orphaned step, and every downstream
        aggregate - per-lane excess loss, the DoReMi headroom order, the exposure-confound argument -
        is computed over a population that includes batches no surviving model state ever consumed.
        """
        n_tok = n_shard = 0
        for o in self.tokens:
            if o.branch_id == branch and o.step >= from_step and not o.superseded:
                o.superseded = True
                n_tok += 1
        for rec in self.shard_exposures:
            if rec.get("branch_id") == branch and rec.get("global_step", -1) >= from_step \
                    and not rec.get("superseded"):
                rec["superseded"] = True
                n_shard += 1
        return {"branch": branch, "from_step": from_step,
                "token_observations_superseded": n_tok, "shard_exposures_superseded": n_shard}

    def apply_spike_attribution(self, unsafe_steps: set[int], resolved: dict[int, str]) -> dict:
        """Withhold shard verdicts that a loss spike could explain.

        PaLM's ablation (arXiv:2204.02311 s5) showed a spike is generally the interaction of a batch
        with a parameter state, not a property of the batch. Classifying a shard `harmful` because
        the loss rose across the step that consumed it therefore ships a permanent wrong label into
        the next cohort's decisions - and nothing downstream can tell.

        `resolved` maps a spike step to the ablation verdict for it. Only `data_implicated` restores
        the ability to blame the data; everything else leaves the verdict withheld.
        """
        withheld, restored = [], []
        for rec in self.shard_exposures:
            step = rec.get("global_step")
            if step not in unsafe_steps:
                continue
            verdict = resolved.get(step)
            if verdict == "data_implicated":
                rec["attribution_note"] = (
                    "coincided with a spike, and the PaLM ablation REPRODUCED it from an earlier "
                    "checkpoint - the data is a legitimate suspect, so the verdict stands")
                restored.append(rec["shard_id"])
                continue
            rec["attribution_uncertain"] = True
            rec["usefulness_withheld"] = rec["usefulness"]
            rec["usefulness"] = "unattributed"
            rec["attribution_note"] = (
                "exposed inside a loss-spike window; a spike is a run event, not a shard property "
                f"(ablation verdict: {verdict or 'not run'}). Verdict withheld rather than guessed.")
            withheld.append(rec["shard_id"])
        return {"unsafe_steps": sorted(unsafe_steps),
                "verdicts_withheld": sorted(set(withheld)),
                "verdicts_upheld_after_ablation": sorted(set(restored))}

    # ------------------------------------------------------------------ V6 output
    def excess_loss_report(self) -> dict:
        """DoReMi's per-lane signal and RHO-1's per-token quadrants, from the same measurement.

        Per lane, mean excess loss answers "does more budget here still move the model?" - a lane the
        reference already handles better than we do has headroom; a lane where we match or beat the
        reference is saturating. That is the quantity DoReMi reweights on.

        Per token, the quadrants answer the question a perplexity level cannot: high loss with
        positive excess is learnable-hard and worth gradient; high loss with ~zero excess is
        irreducible and the budget spent on it is wasted. RHO-1 measured those at 26% and 11% of
        tokens respectively, which is why the distinction is worth this much machinery.
        """
        scored = [o for o in self.effective_tokens() if o.excess_loss is not None]
        if not scored:
            return {"measured": False,
                    "reason": "no reference model configured, so excess loss was not computed"}

        by_lane: dict[str, list[float]] = {}
        quadrants: dict[str, int] = {}
        by_lane_quadrant: dict[str, dict[str, int]] = {}
        for o in scored:
            by_lane.setdefault(o.lane, []).append(o.excess_loss)
            quadrants[o.learnability] = quadrants.get(o.learnability, 0) + 1
            by_lane_quadrant.setdefault(o.lane, {})
            by_lane_quadrant[o.lane][o.learnability] = \
                by_lane_quadrant[o.lane].get(o.learnability, 0) + 1

        n = len(scored)
        lane_rows = {}
        for lane, xs in sorted(by_lane.items()):
            mean_excess = sum(xs) / len(xs)
            lane_rows[lane] = {
                "tokens_scored": len(xs),
                "mean_excess_loss": round(mean_excess, 6),
                "headroom": ("behind_reference" if mean_excess > 0.05 else
                             "ahead_of_reference" if mean_excess < -0.05 else "matched"),
            }
        ranked = sorted(lane_rows, key=lambda k: -lane_rows[k]["mean_excess_loss"])
        return {
            "measured": True,
            "tokens_scored": n,
            "by_lane": lane_rows,
            "lanes_by_headroom_desc": ranked,
            "quadrants": {k: {"tokens": v, "fraction": round(v / n, 4)}
                          for k, v in sorted(quadrants.items())},
            "quadrants_by_lane": {k: dict(sorted(v.items()))
                                  for k, v in sorted(by_lane_quadrant.items())},
            "interpretation": {
                "learnable_hard": "high loss, reference finds it easier - worth gradient (RHO-1's "
                                  "selection signal)",
                "irreducible_hard": "high loss for BOTH models - aleatoric; collecting more buys "
                                    "nothing",
                "already_learned": "low loss, matched - candidate for down-weighting",
                "saturating": "we are ahead of the reference - the lane is done",
            },
            "doremi_note": "lanes_by_headroom_desc is the order DoReMi-style reweighting would "
                           "raise. It is reported, not applied: changing lane weights from a "
                           "32-dim model on a 50-document corpus would be a claim this run cannot "
                           "support.",
        }

    def trajectory_report(self, *, high_loss: float) -> dict:
        """RHO-1's four token classes, measured as a SLOPE rather than a level.

        This is the measurement excess loss cannot replace. Excess loss compares us to a reference at
        one moment; a trajectory compares us to ourselves over the run, which is what actually
        distinguishes "hard and improving" from "hard and stuck". RHO-1 report 51% L->L, 26% H->L,
        11% H->H and 12% L->H; only the last two are actionable, and only the slope separates them.

        A token instance is identified by (sample_id, position_in_pack). The same packed sample is
        re-served on later passes over its lane pool, so a token with pass_number > 0 has been seen
        by two different model states and has a slope. Tokens seen once are reported separately
        rather than silently bucketed - a single observation has no trajectory, and pretending
        otherwise is how a level gets mistaken for a trend.
        """
        seen: dict[tuple[str, int], list[tuple[int, float]]] = {}
        for o in self.effective_tokens():
            seen.setdefault((o.sample_id, o.position_in_pack), []).append((o.step, o.loss))

        repeated = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
        classes: dict[str, int] = {}
        deltas: list[float] = []
        for obs in repeated.values():
            first, last = obs[0][1], obs[-1][1]
            deltas.append(last - first)
            hi_first, hi_last = first >= high_loss, last >= high_loss
            key = ("H_to_H" if hi_first and hi_last else
                   "H_to_L" if hi_first and not hi_last else
                   "L_to_H" if not hi_first and hi_last else "L_to_L")
            classes[key] = classes.get(key, 0) + 1

        n = len(repeated)
        return {
            "measured": n > 0,
            "token_instances_seen_once": len(seen) - n,
            "token_instances_with_a_trajectory": n,
            "high_loss_cut": round(high_loss, 6),
            "classes": {k: {"tokens": v, "fraction": round(v / n, 4)}
                        for k, v in sorted(classes.items())} if n else {},
            "mean_loss_delta": round(sum(deltas) / n, 6) if n else None,
            "rho1_reference_distribution": {"L_to_L": 0.51, "H_to_L": 0.26,
                                            "H_to_H": 0.11, "L_to_H": 0.12},
            "actionable": {
                "H_to_L": "learnable-hard: the model is closing the gap - collect more like this",
                "H_to_H": "irreducible: high loss that does not move - aleatoric, stop paying for it",
                "L_to_H": "regression: was learned and got worse - a seam or interference signal",
            },
            "caveat": "a trajectory needs repeated exposure. Lanes with large pools relative to "
                      "their budget are drawn once and contribute no trajectory at all, so this "
                      "report is biased toward small and protected lanes. Reported, not corrected: "
                      "correcting it would mean over-sampling small lanes, which is not the plan.",
        }

    def reuse_advisory(self, *, cutoff: float = 1.2, epoch_ceiling: float = 4.0) -> dict:
        """Per-shard V6 placement decisions, from perplexity relative to the run-wide median.

        A loss curve says the run is working. Per-shard perplexity says *which shards* are working,
        and that is a placement decision rather than a diagnostic.

        The rule is a ratio against the corpus, not an absolute: absolute perplexity depends on
        vocabulary size and training depth, so a fixed threshold means nothing across runs or depths.

            ratio = shard_geometric_mean_ppl / median_over_shards

        - ratio >= cutoff        the model still finds this shard materially harder than the corpus.
                                 Either it is more informative (place earlier, or spend another
                                 epoch) or it is noise - which is exactly what excess loss
                                 disambiguates, so the two reports are meant to be read together.
        - ratio <= 1/cutoff      the model finds it easy; re-reading it in V6 buys little.
        - otherwise              typical; keep as is.

        Shards whose verdict is withheld by spike attribution are passed through untouched: a shard
        that happened to be in a spiking batch has no trustworthy perplexity signal either.
        """
        rows = [r for r in self.shard_exposures
                if r.get("geometric_mean_ppl") and r.get("branch_id") == "run-a"
                and not r.get("superseded")]
        if not rows:
            return {"measured": False, "reason": "no shard exposures recorded"}

        latest: dict[str, dict] = {}
        for r in rows:                                  # last exposure wins - the final verdict
            latest[r["shard_id"]] = r
        ppls = sorted(r["geometric_mean_ppl"] for r in latest.values())
        n = len(ppls)
        median = ppls[n // 2] if n % 2 else (ppls[n // 2 - 1] + ppls[n // 2]) / 2

        out = []
        for sid, r in sorted(latest.items()):
            ratio = r["geometric_mean_ppl"] / median if median else float("nan")
            if r.get("attribution_uncertain"):
                action = "withheld_pending_spike_attribution"
            elif ratio >= cutoff:
                action = "harder_than_corpus_place_earlier_or_repeat"
            elif ratio <= 1.0 / cutoff:
                action = "easier_than_corpus_do_not_repeat"
            else:
                action = "typical_keep_as_is"
            exposures = sum(1 for x in rows if x["shard_id"] == sid)
            out.append({
                "shard_id": sid,
                "capability_lane": r["capability_lane"],
                "geometric_mean_ppl": r["geometric_mean_ppl"],
                "ratio_to_median": round(ratio, 4),
                "exposures_this_run": exposures,
                "epoch_ceiling_headroom": round(epoch_ceiling - exposures, 2),
                "action": action,
            })
        return {
            "measured": True,
            "median_shard_ppl": round(median, 4),
            "cutoff": cutoff,
            "rule": "ratio = shard_ppl / median_shard_ppl; symmetric, so 1/cutoff on the easy side",
            "shards": out,
            "read_with": "excess_loss - a high ratio means hard, and only excess loss says whether "
                         "hard means learnable or irreducible",
        }

    def shard_report_cards(self, branch: str = "run-a") -> list[dict]:
        """One card per shard, folding every exposure of it across phases.

        Scoped like `effective_tokens`. Unscoped, `v6_recommendations.json` disagreed with itself in
        the same file: shard cards totalled 147 exposures while `reuse_advisory` - which does filter -
        totalled 107. A V6 placement decision computed over a control fork's exposures is a decision
        about an experiment that was thrown away.
        """
        by_shard: dict[str, list[dict]] = {}
        for e in self.shard_exposures:
            if e.get("branch_id", "run-a") != branch or e.get("superseded"):
                continue
            by_shard.setdefault(e["shard_id"], []).append(e)
        cards = []
        for sid, evs in sorted(by_shard.items()):
            by_phase = {}
            for e in evs:
                by_phase.setdefault(e["model_phase"], []).append(e)
            phases = {
                p: {"mean_loss": round(sum(x["average_token_loss"] for x in v) / len(v), 6),
                    "loss_delta": round(sum(x["loss_delta_before_after"] for x in v) / len(v), 6),
                    "hot_pct": round(sum(x["high_perplexity_token_pct"] for x in v) / len(v), 2),
                    "grad_norm": round(sum(x["gradient_norm"] for x in v) / len(v), 6),
                    "n_exposures": len(v)}
                for p, v in sorted(by_phase.items())
            }
            deltas = {p: v["loss_delta"] for p, v in phases.items()}
            best = max(deltas, key=lambda p: deltas[p]) if deltas else None
            worst = min(deltas, key=lambda p: deltas[p]) if deltas else None
            inverted = (best is not None and worst is not None and best != worst
                        and deltas[best] > 0 >= deltas[worst])
            cards.append({
                "shard_id": sid, "capability_lane": evs[0]["capability_lane"],
                "phases": phases, "best_phase": best, "worst_phase": worst,
                "usefulness_inverts_with_phase": inverted,
                "total_exposures": len(evs),
                "max_pass_number": max(x["repeated_pass_number"] for x in evs),
                "recommendation": recommend(phases, inverted, evs),
            })
        return cards

    def boundary_report(self) -> dict:
        """Perplexity at EOS specifically - the session asks 'how do I know EOS is working'."""
        agg = self.aggregate("boundary")
        return {
            "question": "Is the model learning the document boundary convention?",
            "at_boundary": agg.get("True"),
            "not_at_boundary": agg.get("False"),
            "interpretation": (
                "EOS perplexity should fall relative to non-boundary tokens as the model learns that "
                "documents end. A boundary perplexity that stays high means the model is not "
                "predicting the end of a document, so packed windows are teaching run-on context."
            ),
        }

    def position_bucket_report(self) -> dict:
        """Aggregation by position-within-document.

        The lesson prose requires this; the course widget's Pos view is broken (measured: it reads a
        field the token records do not have and collapses every token into one `undefined` bucket).
        """
        buckets = self.aggregate("bucket")
        ordered = {k: buckets[k] for k in sorted(buckets, key=lambda x: int(x))}
        vals = [v["geometric_mean_ppl"] for v in ordered.values()]
        return {
            "n_buckets": N_POSITION_BUCKETS,
            "definition": "decile of the token's position within its own document",
            "buckets": ordered,
            "monotone_decreasing": all(a >= b for a, b in zip(vals, vals[1:])) if len(vals) > 1 else None,
            "spread": round(max(vals) - min(vals), 4) if vals else None,
        }

    def volume_report(self) -> dict:
        """Bytes per storage tier, measured on this run's own trace.

        The session raises the ledger-volume problem directly: a 10M-step run with per-token traces
        is billions of points, and "you might be downloading a billion points just to look at a
        dashboard". So the tiers are not a design sketch - the cost of each is measured here and the
        extrapolation to a real run is stated rather than implied.
        """
        stored = sum(len(json.dumps(o.to_record(quantize=self.tier == "quantized"),
                                    sort_keys=True)) + 1 for o in self.tokens)
        per_full = (sum(len(json.dumps(o.to_record(), sort_keys=True)) + 1
                        for o in self.tokens[:200]) / max(1, min(200, len(self.tokens)))) \
            if self.tokens else 0.0
        # self.tokens is already sampled at `sample_every`, so the full tier must be extrapolated
        # from the OBSERVED count rather than measured on the stored subset.
        full = int(per_full * self.n_observed)
        agg = len(json.dumps({d: self.aggregate(d) for d in
                              ("shard", "lane", "language", "bucket", "stage", "boundary")},
                             sort_keys=True))
        n = max(1, len(self.tokens))
        return {
            "loss_bearing_tokens_observed": self.n_observed,
            "tokens_stored": len(self.tokens),
            "tier_used": self.tier,
            "sample_every": self.sample_every,
            "bytes_full_extrapolated": full,
            "bytes_stored_this_run": stored,
            "bytes_aggregate_only": agg,
            "bytes_per_token_full": round(per_full, 2),
            "compression_vs_full": {
                "stored_tier": round(stored / full, 5) if full else None,
                "aggregate_only": round(agg / full, 7) if full else None,
            },
            "extrapolation": {
                "note": "at this bytes-per-token, a run consuming 3e12 loss-bearing tokens would "
                        "cost the figure below in the full tier. This is why the full tier is for "
                        "proxy runs, debug runs and suspicious shards only.",
                "full_tier_bytes_at_3T_tokens": int(per_full * 3e12),
                "full_tier_petabytes_at_3T_tokens": round(per_full * 3e12 / 1e15, 2),
                "aggregate_tier_at_3T_tokens": "size is bounded by the number of distinct "
                                               "(shard, lane, language, bucket, stage) keys, not "
                                               "by token count - it does not grow with the run",
            },
            "policy": {
                "full": "proxy runs, debug runs, suspicious shards",
                "quantized": "selected intervals of a large run",
                "aggregate": "the full run",
            },
        }

    def write(self, dirpath: str | Path) -> dict[str, str]:
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        q = self.tier == "quantized"
        (d / "learning_tokens.jsonl").write_text(
            "".join(json.dumps(o.to_record(quantize=q), sort_keys=True) + "\n" for o in self.tokens),
            encoding="utf-8")
        (d / "learning_shards.jsonl").write_text(
            "".join(json.dumps(e, sort_keys=True) + "\n" for e in self.shard_exposures),
            encoding="utf-8")
        return {"tokens": str(d / "learning_tokens.jsonl"), "shards": str(d / "learning_shards.jsonl")}


def phase_of(model_age_tokens: int, *, total: int | None = None) -> str:
    """Model phase from tokens consumed. Boundaries mirror S5's stage fractions."""
    if total is None or total <= 0:
        return "early" if model_age_tokens < 10_000 else ("mid" if model_age_tokens < 40_000 else "late")
    f = model_age_tokens / total
    return "early" if f < 0.2 else "mid" if f < 0.6 else "late" if f < 0.97 else "anneal"


def classify(rec: dict) -> str:
    """useful | neutral | harmful, from measured loss movement."""
    d = rec["loss_delta_before_after"]
    if d > 0.002:
        return "useful"
    if d < -0.002:
        return "harmful"
    return "neutral"


def recommend(phases: dict, inverted: bool, evs: list[dict]) -> dict:
    """The V6 instruction for this shard, with the measurement that produced it.

    Deliberately not a bare verdict: the next cohort needs to know *why*, or the recommendation is
    just another unbacked number.
    """
    deltas = {p: v["loss_delta"] for p, v in phases.items()}
    best = max(deltas, key=lambda p: deltas[p]) if deltas else None
    total_delta = sum(deltas.values())
    hot = sum(v["hot_pct"] for v in phases.values()) / max(1, len(phases))
    max_pass = max(x["repeated_pass_number"] for x in evs)

    if inverted:
        action = "phase_guard"
        why = f"usefulness inverts with phase: best at {best}, negative elsewhere"
    elif total_delta <= 0:
        action = "delay_or_reclean"
        why = f"no measured loss improvement across any phase (sum of deltas {total_delta:+.4f})"
    elif hot > 40:
        action = "keep_with_phase_guard"
        why = f"productive but hard: {hot:.1f}% of tokens above ppl {HOT_PPL_THRESHOLD}"
    elif max_pass >= 2 and deltas.get(best, 0) < 0.01:
        action = "repetition_budget_exhausted"
        why = f"seen {max_pass + 1} passes with diminishing loss movement"
    else:
        action = "protect_or_upsample"
        why = f"consistent loss improvement, best at {best} ({deltas.get(best, 0):+.4f})"
    return {"action": action, "because": why,
            "best_phase": best, "mean_hot_token_pct": round(hot, 2),
            "max_pass_number": max_pass}
