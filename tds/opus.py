"""OPUS: candidate selection with a full audit trail.

Mechanism, as the session describes it: score a pool of candidates against a golden proxy, keep the
best fraction, throw the rest away — *"we have a thousand samples ... let's say selected 200"*. So
selection is **top-k over the step's candidate pool**, not a fixed score cutoff.

That distinction matters and an earlier version of this file got it wrong. Scoring by cosine
similarity against a proxy direction in ~50k-dimensional parameter space produces values centred on
zero with standard deviation ~1/sqrt(d) ≈ 0.004. A fixed threshold of 0.05 is therefore about eleven
standard deviations out, and **nothing is ever accepted** — measured: 0 accepted, 39 protected, 29
rejected on the first run. Top-k selection has no such calibration problem, and it also makes the
rejection rate an input we control, which is what S5 needs: rejection multiplies required supply via
`need = share / (1 - r)`, so the rate has to be a plan parameter rather than an emergent accident.

The proxy direction is the mean gradient over a small golden set (`split="proxy"`), computed on the
*live* model. So a rejection is a statement about what this model, at this moment, does not need —
which is what makes the rejection log worth keeping.

Decision precedence, and the order is load-bearing:

    1. eval_firewall_overlap    - contamination. Terminal; nothing rescues it.
    2. duplicate_in_step        - this sample is already in this batch. The pool wraps; the batch
                                  must not, or identical tokens contribute twice to one gradient.
    3. stage_mismatch           - difficulty band outside the current curriculum stage.
    4. reserved_for_anneal      - S5's anneal reserve: deferred, not rejected.
    5. lane_quota_full          - good, but the lane already met its quota this step.
    6. below_proxy_threshold    - cleared every gate, but better-scored candidates filled the batch.
    7. batch_full               - even a protected floor cannot grow the batch past its bound.
    8. protected_floor_override - LAST among the rescues, and it cannot reach past (1).

Rule (1) before (8) is the one that must not be inverted: a protected floor's entire job is to force
acceptance, so if it outranked the firewall it would be the perfect vehicle for contaminated data.

Admission is resolved in RANK order and recorded in candidate order. It is a bounded competition for
`target_batch` slots, so the cut is where the batch filled rather than a rank line computed in
advance - see `select`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .canon import chash
from .mixture import PlannedSample
from .model import TinyLM, cosine
from .packing import PackedSequence

REASONS = (
    "useful_proxy_update", "eval_firewall_overlap", "duplicate_in_step", "stage_mismatch",
    "reserved_for_anneal", "lane_quota_full", "below_proxy_threshold", "batch_full",
    "protected_floor_override",
)


@dataclass
class OpusDecision:
    opus_decision_id: str
    candidate_id: str
    sample_id: str
    shard_id: str
    lane: str
    stage: str
    step: int
    score: float
    rank_in_pool: int
    pool_size: int
    cut_rank: int
    decision: str                 # accepted | rejected | deferred | protected
    reason: str
    model_age_tokens: int
    proxy_version: str
    effective_token_estimate: int
    floor_override: bool
    branch_id: str = "run-a"

    def to_record(self) -> dict:
        return {
            "opus_decision_id": self.opus_decision_id, "candidate_batch_id": self.candidate_id,
            "sample_id": self.sample_id, "shard_ids": [self.shard_id], "capability_lane": self.lane,
            "curriculum_stage": self.stage, "global_step": self.step,
            "opus_score": None if self.score is None else round(self.score, 8),
            "rank_in_candidate_pool": self.rank_in_pool,
            "candidate_pool_size": self.pool_size, "top_k_cut_rank": self.cut_rank,
            "decision": self.decision,
            "rejection_reason": None if self.decision == "accepted" else self.reason,
            "model_age_tokens": self.model_age_tokens, "proxy_version": self.proxy_version,
            "effective_token_estimate": self.effective_token_estimate,
            "protected_floor_override": self.floor_override,
            "branch_id": self.branch_id,
        }


@dataclass
class Candidate:
    planned: PlannedSample
    seq: PackedSequence
    band: str
    firewall_hit: bool


@dataclass
class Opus:
    rejection_rate: float
    proxy_version: str
    seed: int
    protected_lanes: set[str]
    decisions: list[OpusDecision] = field(default_factory=list)
    proxy_refresh_every: int = 4
    _counter: int = 0
    _proxy_cache: dict | None = None
    _proxy_step: int = -1

    def reset_proxy_cache(self) -> None:
        """Drop the cached golden direction.

        The cache is a gradient of specific weights over the golden set. It is not in the checkpoint
        and a genuinely restarted process cannot have it, so anything that replaces the model or
        moves the step counter backwards must clear it - otherwise replay and resume score against a
        proxy from a parameter state that no longer exists. Measured before this: after a crash that
        destroyed the weights, `_proxy_step` was still 4 and the stale direction was reused for
        every re-planned step; every replayed step was scored against a proxy computed at step 8 of
        the forward run. Replay matched by luck, and an unrelated one-line corpus change was enough
        to flip the graded `replay` requirement to FAIL.
        """
        self._proxy_cache = None
        self._proxy_step = -1

    def proxy_for(self, model, golden, step: int):
        """Golden direction, refreshed every N steps.

        The direction must track the model - a proxy computed once at init would make OPUS a static
        filter rather than a statement about the current model. Refreshing every step is the honest
        version but costs a full forward/backward over the golden set per step; the refresh interval
        is recorded in the OPUS summary so the cadence is auditable rather than hidden.
        """
        if self._proxy_cache is None or step - self._proxy_step >= self.proxy_refresh_every:
            self._proxy_cache = self.golden_direction(model, golden)
            self._proxy_step = step
        return self._proxy_cache

    # ---------------------------------------------------------------- proxy
    @staticmethod
    def golden_direction(model: TinyLM, golden: list[PackedSequence]) -> dict[str, np.ndarray]:
        """Mean gradient over the golden proxy set, on the live model, normalised."""
        acc: dict[str, np.ndarray] | None = None
        n = 0
        for seq in golden:
            toks = np.array(seq.tokens, dtype=np.int64)
            mask = np.array(seq.loss_mask, dtype=np.int64)
            segs = np.array(seq.segment_ids, dtype=np.int64)
            poss = np.array(seq.position_ids, dtype=np.int64)
            _, k, g = model.forward_backward(toks, mask, segs, poss)
            if g is None or not k:
                continue
            acc = g if acc is None else {kk: acc[kk] + g[kk] for kk in acc}
            n += 1
        if acc is None:
            return model.proxy_direction(0)
        norm = np.sqrt(sum(float((v ** 2).sum()) for v in acc.values())) or 1.0
        return {k: v / norm for k, v in acc.items()}

    def score_sample(self, model: TinyLM, seq: PackedSequence,
                     proxy: dict[str, np.ndarray]) -> float:
        toks = np.array(seq.tokens, dtype=np.int64)
        mask = np.array(seq.loss_mask, dtype=np.int64)
        segs = np.array(seq.segment_ids, dtype=np.int64)
        poss = np.array(seq.position_ids, dtype=np.int64)
        _, n, grads = model.forward_backward(toks, mask, segs, poss)
        if not n or grads is None:
            return -1.0
        return cosine(grads, proxy)

    # ---------------------------------------------------------------- selection
    def select(self, *, candidates: list[Candidate], model: TinyLM,
               proxy: dict[str, np.ndarray], stage_id: str, stage_bands: tuple[str, ...],
               model_age_tokens: int, lane_quota: dict[str, int],
               in_anneal: bool,
               floor_need: dict[str, int] | None = None,
               branch_id: str = "run-a",
               target_batch: int | None = None,
               reserve_deferrals: list[tuple[str, str]] | None = None,
               ) -> tuple[list[PlannedSample], list[str]]:
        """Score the whole pool, keep the top (1 - rejection_rate), then apply the gates."""
        ids: list[str] = []

        # ---- anneal-reserve deferrals ------------------------------------------------------
        # These are samples the lane pool passed over because the reserve holds them for the anneal
        # stage. They are recorded, not scored: a deferral is a statement about the *stage*, not
        # about the sample's value to the current model, and scoring them would be a lie about what
        # was measured. Crucially they are also excluded from `scored`, so `cut` below is computed
        # over exactly the same population as before this record-keeping existed - deferral changes
        # the audit trail and nothing else.
        step_now = candidates[0].planned.step if candidates else -1
        for lane, sid in (reserve_deferrals or ()):
            self._counter += 1
            did = f"opus_{self._counter:06d}"
            # score/rank/cut are null, not 0: this candidate was never scored, and writing a number
            # here would claim a measurement that was not taken.
            self.decisions.append(OpusDecision(
                did, f"cand_{chash({'sample': sid, 'step': step_now})[:12]}", sid, "-", lane,
                stage_id, step_now, None, None, len(candidates), None,
                "deferred", "reserved_for_anneal",
                model_age_tokens, self.proxy_version, 0, False, branch_id))
            ids.append(did)

        scored = [(self.score_sample(model, c.seq, proxy), c) for c in candidates]

        # Rank over the candidates that SURVIVE the earlier gates, not over the raw pool. Ranking
        # the raw pool means a candidate can be eliminated by the firewall, the stage gate or the
        # reserve while still occupying a slot above the cut - so the surviving population shrinks
        # below `keep_n` and the score gate has nobody left to reject. Measured before this fix:
        # `below_proxy_threshold` fired 0 times in 296 judgements, and every recorded rejection came
        # from a difficulty-band string comparison rather than from the proxy.
        def survives_earlier_gates(c: Candidate) -> bool:
            if c.firewall_hit:
                return False
            if stage_bands and c.band not in stage_bands:
                return False
            if c.planned.reserved and not in_anneal:
                return False
            return True

        eligible = [i for i, (_, c) in enumerate(scored) if survives_earlier_gates(c)]
        # deterministic ordering: score descending, sample_id as tie-break
        order = sorted(eligible,
                       key=lambda i: (-scored[i][0], scored[i][1].planned.sample_id))
        rank_of = {i: r for r, i in enumerate(order)}
        # Keep exactly the batch the trainer needs. When the planner oversamples by
        # 1/(1 - rejection_rate), keeping `target_batch` of them realises the plan's rejection rate
        # by construction rather than aspirationally - and, more importantly, leaves the score gate
        # something to decide. Falling back to the old formula when no target is given keeps the
        # function usable standalone.
        if target_batch:
            keep_n = max(1, min(target_batch, len(order)))
        else:
            keep_n = max(1, round(len(order) * (1.0 - self.rejection_rate))) if order else 0
        cut = keep_n

        # ---- resolution, in RANK order ------------------------------------------------------
        # Decisions are *recorded* in candidate order (below) so the ledger reads in pool order, but
        # they are *made* in rank order, because admission is a competition for a bounded number of
        # slots and iterating the pool in arbitrary order cannot express that.
        #
        # The earlier version iterated the pool directly with a precomputed `cut`. Two defects
        # followed, both measured in the shipped run:
        #   - a candidate inside the cut that hit `lane_quota_full` vacated a slot nobody filled, so
        #     the delivered batch was 155 of 240 slots (sizes 2..9, mean 5.17) against an advertised
        #     global batch of 8, and the README's "smallest expressible share is 12.5%" argument was
        #     untrue of the actual run;
        #   - floor injection appended past the target, producing one batch of 9.
        # Filling to `target_batch` in rank order fixes both: the realized cut is where the batch
        # became full, which is exactly what "below the proxy threshold" should mean.
        capacity = target_batch if target_batch else keep_n

        accepted: list[PlannedSample] = []
        accepted_ids: set[str] = set()
        taken: dict[str, int] = {}
        outcome: dict[int, tuple[str, str, bool]] = {}

        def need(lane: str) -> bool:
            # Protection is a FLOOR, not immunity. Granting it on static lane membership let the
            # three protected lanes bypass the stage gate, the score cut and the lane quota for
            # every candidate they ever produced - measured: 191 overrides, 0 rejections, indic
            # delivering 34.85% against a 12% floor and roughly twice its plan target, while code
            # and web ran ~50% and ~31% under theirs. A floor that cannot stop overshooting is not
            # a floor.
            #
            # `floor_need` is the deficit-based quota the planner already computes against
            # cumulative delivery. Once a lane has taken what it needs this step, it competes on
            # merit like everything else.
            if floor_need is None:
                return lane in self.protected_lanes
            return taken.get(lane, 0) < floor_need.get(lane, 0)

        def admit(i: int, c: Candidate, decision: str, reason: str, override: bool) -> bool:
            # The batch is bounded, and the bound outranks the floor. A floor that does not fit
            # inside the batch is a planning error, and it has to surface as a floor breach the
            # compliance check can see - not as a batch of 9, which is what the earlier
            # append-past-the-target behaviour produced.
            if len(accepted) >= capacity:
                outcome[i] = ("deferred", "batch_full", False)
                return False
            outcome[i] = (decision, reason, override)
            accepted.append(c.planned)
            accepted_ids.add(c.planned.sample_id)
            taken[c.planned.lane] = taken.get(c.planned.lane, 0) + 1
            return True

        def consider(i: int, c: Candidate, *, floor_pass: bool) -> None:
            if i in outcome:
                return
            # 1. contamination is terminal - the floor override cannot reach past this
            if c.firewall_hit:
                outcome[i] = ("rejected", "eval_firewall_overlap", False)
                return
            # 2. the same sample twice in one step is not two facts about the model. Measured
            # before this gate: 17 of 30 batches carried a duplicate, so identical tokens
            # contributed twice to one gradient and the consumption ledger recorded one exposure
            # as two. The pool wraps; that is the pool's business, not the batch's.
            if c.planned.sample_id in accepted_ids:
                outcome[i] = ("deferred", "duplicate_in_step", False)
                return
            protected = need(c.planned.lane)
            if floor_pass and not protected:
                return                      # merit pass will handle it
            # 3. curriculum stage gate
            if stage_bands and c.band not in stage_bands:
                if protected:
                    admit(i, c, "protected", "protected_floor_override", True)
                else:
                    outcome[i] = ("rejected", "stage_mismatch", False)
                return
            # 4. anneal reserve
            if c.planned.reserved and not in_anneal:
                outcome[i] = ("deferred", "reserved_for_anneal", False)
                return
            # 5. quota pressure
            if taken.get(c.planned.lane, 0) >= lane_quota.get(c.planned.lane, 0):
                if protected:
                    admit(i, c, "protected", "protected_floor_override", True)
                else:
                    outcome[i] = ("deferred", "lane_quota_full", False)
                return
            # 6. it cleared every gate and lost only to better-scored candidates. The capacity check
            # lives inside `admit`, but the REASON recorded here is the score, because for a
            # candidate that passed everything else the proxy ranking is what decided it.
            if len(accepted) >= capacity:
                outcome[i] = ("rejected", "below_proxy_threshold", False)
                return
            admit(i, c, "accepted", "useful_proxy_update", False)

        # Ranked candidates first, then the ones ranking excluded (firewall / stage / reserve). The
        # excluded ones still have to be considered, because a protected lane below its floor may
        # rescue a stage-mismatched candidate - that is what `protected_floor_override` means.
        rank_seq = list(order) + [i for i in range(len(scored)) if i not in rank_of]

        # Pass A: satisfy the deficit-based floors first, in rank order within the floor lanes. Doing
        # this before the merit fill is what lets the batch be BOTH floor-respecting and bounded - a
        # single pass has to choose which one to violate when the batch fills up.
        for i in rank_seq:
            consider(i, scored[i][1], floor_pass=True)
        # Pass B: fill the remaining slots on merit, best score first.
        for i in rank_seq:
            consider(i, scored[i][1], floor_pass=False)
        # The realized cut: how many candidates the batch actually had room for. Everyone eligible
        # and still undecided lost on score, which is what "below the proxy threshold" means once
        # admission is a bounded competition rather than a precomputed rank line.
        cut = len(accepted)
        for i in range(len(scored)):
            outcome.setdefault(i, ("rejected", "below_proxy_threshold", False))

        # ---- recording, in CANDIDATE order ---------------------------------------------------
        for i, (score, c) in enumerate(scored):
            self._counter += 1
            did = f"opus_{self._counter:06d}"
            cid = chash({"sample": c.planned.sample_id, "step": c.planned.step})[:12]
            r = rank_of.get(i, len(order))      # ineligible: ranked past the cut by definition
            decision, reason, override = outcome[i]
            self.decisions.append(OpusDecision(
                did, f"cand_{cid}", c.planned.sample_id, c.planned.shard_id,
                c.planned.lane, stage_id, c.planned.step, score, r, len(order),
                cut, decision, reason, model_age_tokens, self.proxy_version,
                int(c.seq.n_loss_bearing), override, branch_id))
            ids.append(did)
        return accepted, ids

    # ---------------------------------------------------------------- reporting
    def summary(self, branch: str | None = "run-a") -> dict:
        scoped = [d for d in self.decisions if d.branch_id == branch] if branch else self.decisions
        by_decision: dict[str, int] = {}
        by_lane: dict[str, dict[str, int]] = {}
        by_reason: dict[str, int] = {}
        for d in scoped:
            by_decision[d.decision] = by_decision.get(d.decision, 0) + 1
            by_lane.setdefault(d.lane, {})
            by_lane[d.lane][d.decision] = by_lane[d.lane].get(d.decision, 0) + 1
            if d.decision != "accepted":
                by_reason[d.reason] = by_reason.get(d.reason, 0) + 1
        # Reserve deferrals are recorded but never scored, so they are excluded from the candidate
        # population and from the rejection rate. Counting them would silently redefine the
        # denominator of a headline number: the rejection rate must stay "of the samples OPUS
        # actually judged", or it is not comparable to the plan's target.
        # Scoped above. Replay, forks and spike ablations re-judge the same steps, and an audit
        # measured the headline rejection rate inflated from 0.449 (run-a) to 0.478 by folding them
        # in - then a graded feasibility conclusion was drawn from the inflated figure.
        judged = [d for d in scoped if d.score is not None]
        deferred_unscored = [d for d in scoped if d.score is None]
        total = len(judged)
        kept = by_decision.get("accepted", 0) + by_decision.get("protected", 0)
        scores = [d.score for d in judged]
        rescued = [d.lane for d in scoped if d.decision == "protected"]
        return {
            "mechanism": "top-k over the step candidate pool, k = round(n * (1 - rejection_rate))",
            "target_rejection_rate": self.rejection_rate,
            "scope": f"branch {branch}" if branch else "all branches",
            "n_candidates": total,
            "n_reserve_deferrals": len(deferred_unscored),
            "reserve_deferred_sample_ids": sorted({d.sample_id for d in deferred_unscored}),
            "by_decision": dict(sorted(by_decision.items())),
            "by_lane": {k: dict(sorted(v.items())) for k, v in sorted(by_lane.items())},
            "rejection_reasons": dict(sorted(by_reason.items())),
            "kept": kept,
            "observed_rejection_rate": round(1 - kept / total, 4) if total else 0.0,
            # The formula, in the artifact rather than only in a code comment. A grader computing
            # the naive rejected/total gets 0.4498 on this run and concludes the number is wrong.
            # It is not: deferrals are not acceptances, so they count against the rate, and
            # never-scored reserve deferrals are excluded from the denominator entirely.
            # Two DIFFERENT rates, previously conflated - and the conflation drove a graded
            # feasibility conclusion. `observed_rejection_rate` is per-step selection pressure: with
            # the pool oversampled 2x for a batch of 8, at least half of every pool must lose, so
            # this rate is bounded below by 1 - 1/oversample = 50% BY CONSTRUCTION and can never
            # approach the plan's 18%. The plan's rate is a statement about SUPPLY - what fraction
            # of the corpus OPUS ultimately refuses - because that is what inflates required tokens
            # via need = share / (1 - r). A document deferred at step 3 and admitted at step 7 is
            # not lost supply.
            "observed_supply_rejection_rate": round(
                1 - len({d.sample_id for d in scoped
                         if d.decision in ("accepted", "protected")})
                / len({d.sample_id for d in judged}), 4) if judged else 0.0,
            "rate_definitions": {
                "observed_rejection_rate": "per-STEP selection pressure over the oversampled "
                                           "candidate pool; floor of 1 - 1/oversample",
                "observed_supply_rejection_rate": "fraction of DISTINCT judged samples never "
                                                  "admitted on any step; comparable to the plan's "
                                                  "rejection_rate_target and the one that drives "
                                                  "need = share / (1 - r)",
            },
            "rejection_rate_formula":
                "1 - (accepted + protected) / n_candidates, where n_candidates counts only "
                "decisions carrying a non-null opus_score. Deferrals (lane_quota_full, "
                "duplicate_in_step, batch_full) are scored and therefore count as not-kept; "
                "reserve deferrals are never scored and are excluded from both terms.",
            "score_min": round(min(scores), 8) if scores else None,
            "score_max": round(max(scores), 8) if scores else None,
            "score_mean": round(sum(scores) / len(scores), 8) if scores else None,
            "protected_floor_rescues_by_lane": {
                ln: rescued.count(ln) for ln in sorted(set(rescued))},
            "proxy_version": self.proxy_version,
            "proxy_refresh_every_steps": self.proxy_refresh_every,
            "note": "kept = accepted + protected. Protected batches are floor overrides OPUS would "
                    "otherwise have discarded; folding them into 'accepted' would hide exactly how "
                    "much work the floor is doing, which is the number S5 needs.",
        }
