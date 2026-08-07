"""The OPUS selector: choosing which samples are worth training on.

The session's mechanism, implemented:

1. Hold a **golden proxy** -- a small set of samples representing what the model must
   become good at. Run them through the current model and record which parameters carry
   the largest gradient. That is the "this weight needs update" map.
2. For each candidate sample, run only its **first N tokens** through the same model and
   take the gradient over the same parameter block. N is small: the session sends the first
   512 of a 32,000-token sample, and the whole point is that the probe must be cheap
   relative to the training step it might replace.
3. Score the candidate by how well its gradient **aligns** with the proxy map. Accept the
   top fraction; reject the rest with a reason.

What makes this worth building rather than asserting is the fourth part: **the rejections
are the data**. "The selector considered the value to be low" is a judgement, and judgements
are auditable. If the mean gradient alignment of the *accepted* set falls round over round,
the pool is exhausted -- OPUS is "just selecting the bad quality data because the old data
is so bad it has to select something". That drift is computed in `selection_drift`.

Two overrides exist and both are recorded:

* **Protected flows** (Indic, agentic) bypass rejection entirely. S5 measured that an
  unprotected agentic lane goes to zero in 59 of 60 OPUS iterations -- so a selector that
  can starve a protected lane is a selector that has quietly overruled the plan.
* **Stage gate**: a sample whose difficulty band is outside the current stage's range is
  rejected as `stage_mismatch` before it is ever scored, because accepting it would breach
  the curriculum regardless of how good its gradient looks.

The `--with-api` path additionally asks Claude to rate sample quality and blends that with
the gradient score. It is off by default: `run_demo.py` must work with no API key, and a
selector whose decisions depend on a network call is not reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .determinism import content_hash
from .plan import LANE_SPEC
from .stream import Sample

# Decision outcomes
ACCEPT = "accept"
REJECT = "reject"
DEFER = "defer"

# Reason codes
R_ABOVE_THRESHOLD = "above_proxy_threshold"
R_BELOW_THRESHOLD = "below_proxy_threshold"
R_STAGE_MISMATCH = "stage_mismatch"
R_PROTECTED_OVERRIDE = "protected_flow_override"
R_DUPLICATE_SIGNATURE = "duplicate_gradient_signature"
R_DEFER_LOW_MARGIN = "deferred_near_threshold"

PROBE_TOKENS = 64           # the demo's analogue of the session's first-512-of-32k probe
PROXY_TOPK = 256            # parameters the golden proxy marks as "needs update"
DEFER_MARGIN = 0.02         # candidates within this of the cut are deferred, not rejected


# ---------------------------------------------------------------------------
# golden proxy
# ---------------------------------------------------------------------------


@dataclass
class GoldenProxy:
    """The map of which parameters the model most needs to move, and in which direction."""

    indices: np.ndarray        # int64 (k,) flat indices into the probed parameter
    direction: np.ndarray      # float32 (k,) unit-normalised gradient at those indices
    param_name: str
    n_gold_samples: int
    proxy_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "param_name": self.param_name,
            "top_k": int(self.indices.shape[0]),
            "n_gold_samples": self.n_gold_samples,
            "proxy_hash": self.proxy_hash[:12],
        }


def _probe_param(model) -> Tuple[str, Any]:
    """The parameter block gradients are measured over.

    The last block's first MLP projection: late enough to carry task-specific signal,
    single enough that a probe is one small backward pass rather than a full one.
    """
    name = "blocks.{}.mlp.0.weight".format(len(model.blocks) - 1)
    return name, model.blocks[-1].mlp[0].weight


def _grad_vector(model, batch, param, probe_tokens: int) -> np.ndarray:
    """Flat gradient of the mean loss over `param`, for one probe batch."""
    import torch

    from .model import masked_mean, token_losses

    ids = batch["input_ids"][:, :probe_tokens]
    pos = batch["position_ids"][:, :probe_tokens]
    lm = batch["loss_mask"][:, :probe_tokens]

    if lm.sum() == 0:
        return np.zeros(param.numel(), dtype=np.float32)

    logits = model(ids, pos, None)
    losses, valid = token_losses(logits, ids, lm)
    loss = masked_mean(losses, valid)
    (grad,) = torch.autograd.grad(loss, param, retain_graph=False, allow_unused=False)
    return grad.detach().reshape(-1).to(torch.float32).numpy()


def golden_set(registry, tokenizer, seq_len: int, *, n_shards: int = 4,
               n_sequences: int = 2) -> List[Any]:
    """Held-out reference samples the proxy is measured against.

    Drawn from the **validation** split, deliberately. The golden set defines what the model
    is supposed to become good at, and it is never trained on -- so taking it from training
    batches (which is the convenient thing to do) would make the proxy a map of the data the
    model is already consuming rather than of the capability being targeted. The firewall
    exists to keep validation out of the training path; this is the one sanctioned read of
    it, and it is read-only in both senses.
    """
    from .documents import SPLIT_VAL
    from .masks import LANE_ATTENTION, LANE_POSITION, materialize
    from .packing import LANE_POLICY, pack

    out = []
    keys = [k for k in sorted(registry.by_lane_split) if k[1] == SPLIT_VAL]
    for lane, split in keys[:n_shards]:
        for sid in registry.by_lane_split[(lane, split)][:1]:
            shard = registry.get(sid)
            plan = pack(shard, LANE_POLICY[lane], seq_len,
                        shuffle_key=("opus-golden", lane, sid))
            for seq in plan.sequences[:n_sequences]:
                out.append(materialize(seq, shard, tokenizer.pad_id,
                                       attention_policy=LANE_ATTENTION[lane],
                                       position_policy=LANE_POSITION[lane], lane=lane))
    return out


def build_golden_proxy(model, gold_batches: Sequence[Dict[str, Any]],
                       top_k: int = PROXY_TOPK,
                       probe_tokens: int = PROBE_TOKENS) -> GoldenProxy:
    """Compute the proxy map from the golden set.

    The golden set is *never trained on*. It exists only to say which parameters matter --
    the session's "we're not going to train on it, we're going to store this map".
    """
    import torch

    name, param = _probe_param(model)
    acc = np.zeros(param.numel(), dtype=np.float64)
    for batch in gold_batches:
        acc += _grad_vector(model, batch, param, probe_tokens).astype(np.float64)
    acc /= max(1, len(gold_batches))

    k = min(top_k, acc.shape[0])
    idx = np.argsort(-np.abs(acc))[:k].astype(np.int64)
    idx.sort()
    direction = acc[idx].astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm > 0:
        direction = direction / norm

    return GoldenProxy(
        indices=idx, direction=direction, param_name=name,
        n_gold_samples=len(gold_batches),
        proxy_hash=content_hash([name, idx.tolist(), np.round(direction, 6).tolist()]),
    )


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    sample_index: int
    lane: str
    shard_id: str
    doc_ids: List[str]
    difficulty_band: str
    outcome: str
    reason: str
    alignment: float
    gradient_norm: float
    protected: bool
    api_quality: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "sample_index": self.sample_index, "lane": self.lane,
            "shard_id": self.shard_id, "doc_ids": self.doc_ids,
            "difficulty_band": self.difficulty_band,
            "outcome": self.outcome, "reason": self.reason,
            "alignment": round(self.alignment, 6),
            "gradient_norm": round(self.gradient_norm, 6),
            "protected": self.protected,
        }
        if self.api_quality is not None:
            d["api_quality"] = round(self.api_quality, 4)
        return d


@dataclass
class SelectionRound:
    round_index: int
    step: int
    stage: str
    model_age_tokens: int
    accept_rate: float
    threshold: float
    decisions: List[Decision]
    proxy: GoldenProxy

    # ---- statistics ----------------------------------------------------

    def accepted(self) -> List[Decision]:
        return [d for d in self.decisions if d.outcome == ACCEPT]

    def rejected(self) -> List[Decision]:
        return [d for d in self.decisions if d.outcome == REJECT]

    def merit_accepted(self) -> List[Decision]:
        """Accepts that cleared the threshold on their own.

        Protected-flow overrides are excluded. They are admitted by policy regardless of
        score, so folding them into a quality statistic measures the floor rather than the
        pool -- and since an override's alignment can be negative, it does so in the
        direction that manufactures a degradation signal.
        """
        return [d for d in self.decisions
                if d.outcome == ACCEPT and d.reason == R_ABOVE_THRESHOLD]

    def stats(self) -> Dict[str, Any]:
        acc = self.accepted()
        rej = self.rejected()
        merit = self.merit_accepted()
        m_aligns = [d.alignment for d in merit]
        aligns = [d.alignment for d in acc]
        by_reason: Dict[str, int] = {}
        for d in self.decisions:
            by_reason[d.reason] = by_reason.get(d.reason, 0) + 1
        return {
            "round": self.round_index,
            "step": self.step,
            "stage": self.stage,
            "model_age_tokens": self.model_age_tokens,
            "candidates": len(self.decisions),
            "accepted": len(acc),
            "rejected": len(rej),
            "deferred": sum(1 for d in self.decisions if d.outcome == DEFER),
            "realized_accept_rate": round(len(acc) / max(1, len(self.decisions)), 4),
            "target_accept_rate": self.accept_rate,
            "threshold": round(self.threshold, 6),
            "accepted_alignment_mean": round(float(np.mean(aligns)), 6) if aligns else None,
            "accepted_alignment_min": round(float(np.min(aligns)), 6) if aligns else None,
            "accepted_alignment_max": round(float(np.max(aligns)), 6) if aligns else None,
            "merit_accepted": len(merit),
            "merit_alignment_mean": round(float(np.mean(m_aligns)), 6) if m_aligns else None,
            "merit_alignment_min": round(float(np.min(m_aligns)), 6) if m_aligns else None,
            "rejected_alignment_mean": (
                round(float(np.mean([d.alignment for d in rej])), 6) if rej else None
            ),
            "protected_overrides": sum(1 for d in self.decisions if d.protected),
            "reasons": dict(sorted(by_reason.items())),
            "proxy": self.proxy.to_dict(),
        }


# ---------------------------------------------------------------------------
# selector
# ---------------------------------------------------------------------------


class OpusSelector:
    def __init__(self, model, accept_rate: float = 0.25,
                 probe_tokens: int = PROBE_TOKENS,
                 use_api: bool = False, api_model: str = "claude-opus-5") -> None:
        self.model = model
        self.accept_rate = accept_rate
        self.probe_tokens = probe_tokens
        self.use_api = use_api
        self.api_model = api_model
        self.rounds: List[SelectionRound] = []
        self._seen_signatures: Dict[str, int] = {}
        self.api_calls = 0
        self.api_errors: List[str] = []

    # ---- scoring -------------------------------------------------------

    def _alignment(self, batch: Dict[str, Any], proxy: GoldenProxy) -> Tuple[float, float]:
        import torch

        _name, param = _probe_param(self.model)
        g = _grad_vector(self.model, batch, param, self.probe_tokens)
        sub = g[proxy.indices]
        norm = float(np.linalg.norm(sub))
        if norm == 0.0:
            return 0.0, 0.0
        return float(np.dot(sub / norm, proxy.direction)), float(np.linalg.norm(g))

    def _signature(self, batch: Dict[str, Any], proxy: GoldenProxy) -> str:
        """Coarse fingerprint of a candidate's gradient, for duplicate detection.

        Two samples whose probe gradients point the same way at this resolution teach the
        model the same thing; keeping both spends a step to learn something twice.
        """
        _name, param = _probe_param(self.model)
        g = _grad_vector(self.model, batch, param, self.probe_tokens)[proxy.indices]
        return content_hash(np.sign(np.round(g, 4)).astype(np.int8).tolist())

    # ---- the round -----------------------------------------------------

    def select(
        self,
        candidates: Sequence[Sample],
        proxy: GoldenProxy,
        *,
        step: int,
        stage: str,
        stage_bands: Sequence[str],
        model_age_tokens: int,
        to_batch,
    ) -> SelectionRound:
        """Score every candidate and rank-cut at the accept rate.

        Order of checks matters and is not arbitrary: the stage gate runs before scoring
        (a curriculum breach is not redeemable by a good gradient), protection runs before
        the threshold (a floor that a score can overrule is not a floor), and duplicate
        detection runs last (so a duplicate of an accepted sample is reported as a
        duplicate rather than as a low-alignment reject).
        """
        import torch

        scored: List[Tuple[int, Sample, float, float, str]] = []
        stage_rejects: List[Decision] = []

        for i, s in enumerate(candidates):
            if s.difficulty_band not in stage_bands:
                stage_rejects.append(Decision(
                    sample_index=i, lane=s.lane, shard_id=s.shard_id,
                    doc_ids=s.seq.doc_ids, difficulty_band=s.difficulty_band,
                    outcome=REJECT, reason=R_STAGE_MISMATCH,
                    alignment=0.0, gradient_norm=0.0, protected=False))
                continue
            batch = to_batch([s])
            align, gnorm = self._alignment(batch, proxy)
            sig = self._signature(batch, proxy)
            scored.append((i, s, align, gnorm, sig))

        # Rank cut. Ties broken by (lane, shard_id, index) so the cut is reproducible.
        ordered = sorted(scored, key=lambda t: (-t[2], t[1].lane, t[1].shard_id, t[0]))
        n_accept = max(1, int(round(len(ordered) * self.accept_rate))) if ordered else 0
        threshold = ordered[n_accept - 1][2] if n_accept and ordered else 0.0

        decisions: List[Decision] = list(stage_rejects)
        for rank, (i, s, align, gnorm, sig) in enumerate(ordered):
            protected = LANE_SPEC[s.lane].protected
            outcome, reason = REJECT, R_BELOW_THRESHOLD

            if rank < n_accept:
                outcome, reason = ACCEPT, R_ABOVE_THRESHOLD
            elif protected:
                # The whole point of a protected flow: it is admitted regardless of score.
                outcome, reason = ACCEPT, R_PROTECTED_OVERRIDE
            elif align >= threshold - DEFER_MARGIN:
                outcome, reason = DEFER, R_DEFER_LOW_MARGIN

            if outcome == ACCEPT and sig in self._seen_signatures:
                outcome, reason = REJECT, R_DUPLICATE_SIGNATURE
            elif outcome == ACCEPT:
                self._seen_signatures[sig] = step

            decisions.append(Decision(
                sample_index=i, lane=s.lane, shard_id=s.shard_id,
                doc_ids=s.seq.doc_ids, difficulty_band=s.difficulty_band,
                outcome=outcome, reason=reason, alignment=align,
                gradient_norm=gnorm, protected=protected and outcome == ACCEPT
                and reason == R_PROTECTED_OVERRIDE,
            ))

        if self.use_api:
            self._annotate_with_api(decisions, candidates)

        rnd = SelectionRound(
            round_index=len(self.rounds), step=step, stage=stage,
            model_age_tokens=model_age_tokens, accept_rate=self.accept_rate,
            threshold=threshold, decisions=sorted(decisions, key=lambda d: d.sample_index),
            proxy=proxy,
        )
        self.rounds.append(rnd)
        return rnd

    # ---- optional API scoring -----------------------------------------

    def _annotate_with_api(self, decisions: List[Decision], candidates: Sequence[Sample]) -> None:
        """Ask Claude to rate the accepted samples, and record the rating alongside.

        Advisory only -- it annotates decisions the gradient signal already made, and never
        changes an outcome. A selector whose accept/reject flips on a network call could not
        be replayed, which is the one property this whole system exists to preserve.
        """
        try:
            import anthropic
        except ImportError:
            self.api_errors.append("anthropic SDK not installed; --with-api ignored")
            self.use_api = False
            return

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            self.api_errors.append("no ANTHROPIC_API_KEY; --with-api ignored")
            self.use_api = False
            return

        client = anthropic.Anthropic()
        accepted = [d for d in decisions if d.outcome == ACCEPT][:4]
        for d in accepted:
            try:
                resp = client.messages.create(
                    model=self.api_model,
                    max_tokens=16,
                    system=("You rate pretraining data quality. Reply with a single number "
                            "from 0.0 to 1.0 and nothing else."),
                    messages=[{"role": "user", "content": (
                        "Lane: {}\nBand: {}\nDocuments: {}\nRate this sample's value as "
                        "pretraining data.".format(d.lane, d.difficulty_band,
                                                   ", ".join(d.doc_ids[:3])))}],
                )
                self.api_calls += 1
                if resp.stop_reason == "refusal":
                    self.api_errors.append("refusal on {}".format(d.shard_id))
                    continue
                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                d.api_quality = float(text.split()[0])
            except anthropic.RateLimitError:
                self.api_errors.append("rate limited; stopping API annotation")
                self.use_api = False
                return
            except Exception as exc:
                self.api_errors.append("{}: {}".format(type(exc).__name__, exc))

    # ---- drift ---------------------------------------------------------

    def selection_drift(self) -> Dict[str, Any]:
        """Is the pool getting worse?

        The session's diagnostic: if round 1 accepted samples averaging 0.9 alignment and
        round N accepts samples averaging 0.1, OPUS is not finding good data any more --
        it is picking the least bad of a bad pool, and the shard quality itself is the
        problem. This is the signal that cannot be obtained any other way than by watching
        the selector work over the course of a run.

        This is only readable because the proxy is **re-derived from the same held-out
        golden set at every round**, against the current weights. A proxy computed once and
        reused would go stale as the model moves, and its falling alignment would be
        measuring the proxy's age rather than the pool's quality -- the same number, an
        entirely different conclusion. `proxy_hash` differs per round for that reason.
        """
        rows = []
        for r in self.rounds:
            st = r.stats()
            rows.append({
                "round": r.round_index, "step": r.step, "stage": r.stage,
                "model_age_tokens": r.model_age_tokens,
                "merit_alignment_mean": st["merit_alignment_mean"],
                "merit_alignment_min": st["merit_alignment_min"],
                "accepted_alignment_mean": st["accepted_alignment_mean"],
                "rejected_alignment_mean": st["rejected_alignment_mean"],
                "protected_overrides": st["protected_overrides"],
                "realized_accept_rate": st["realized_accept_rate"],
            })
        means = [r["merit_alignment_mean"] for r in rows
                 if r["merit_alignment_mean"] is not None]
        verdict = "insufficient_rounds"
        delta = None
        if len(means) >= 2:
            delta = means[-1] - means[0]
            if delta < -0.25:
                verdict = ("pool_degrading: accepted-sample alignment has fallen sharply; "
                           "the selector is choosing the least bad of a bad pool")
            elif delta < -0.05:
                verdict = "pool_softening: accepted-sample alignment is drifting down"
            else:
                verdict = "pool_stable"
        return {
            "measured_over": ("threshold accepts only; protected-flow overrides are "
                              "excluded because they are admitted by policy, not by score"),
            "rounds": rows,
            "first_round_merit_mean": means[0] if means else None,
            "last_round_merit_mean": means[-1] if means else None,
            "delta": round(delta, 6) if delta is not None else None,
            "verdict": verdict,
            "rounds_measured": len(means),
            "confidence": ("low -- {} rounds at demo scale; the shape is demonstrated, the "
                           "verdict is not a claim about a real corpus".format(len(means))),
        }

    def report(self) -> Dict[str, Any]:
        return {
            "accept_rate_target": self.accept_rate,
            "probe_tokens": self.probe_tokens,
            "proxy_top_k": PROXY_TOPK,
            "rounds": [r.stats() for r in self.rounds],
            "drift": self.selection_drift(),
            "golden_set": {
                "source": "validation split, never trained on",
                "proxy_rebuilt_each_round": True,
                "proxy_hashes": [r.proxy.proxy_hash[:12] for r in self.rounds],
            },
            "unique_gradient_signatures": len(self._seen_signatures),
            "api_enabled": self.use_api,
            "api_calls": self.api_calls,
            "api_errors": self.api_errors,
            "note": ("Rejections are retained in logs/opus_trail.jsonl. The session's point "
                     "is that what was thrown away is as informative as what was kept."),
        }
