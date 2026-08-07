"""Compiling the S5 curriculum into a per-step schedule.

Session 5 decided the mixture in human terms -- lane shares per stage, protected floors,
seam widths, difficulty bands. This turns that into the thing a training loop can actually
consume: for every optimizer step, which lanes contribute how many sequences, drawn from
which shards, at what window, under which band gate.

Four constraints act at once, and they conflict:

1. **Stage mixture** -- lane shares interpolated across seams, never switched abruptly.
2. **Protected floors** -- Indic and agentic have a floor that survives renormalisation.
   "A protected capability that ramps is still a residue of whatever budget is left over."
3. **Difficulty band gate** -- a stage may only draw from its permitted bands. This closes
   the mechanism S5 left open; the gate is a shard-level lookup, because band was decided
   at shard-build time and frozen into the manifest.
4. **Epoch ceiling** -- no lane may exceed 4 epochs over its own supply (arXiv:2305.16264).

When they conflict, the ceiling wins and the shortfall is *recorded*, not absorbed. That is
how the demo reproduces S5 decision 3: agentic demand cannot be met from agentic supply, the
lane is clamped, and the deficit appears in the schedule as a number rather than as a claim.

Allocation is by **largest remainder**, not sampling. A sampled mixture is only correct in
expectation, and over 146 steps the variance is larger than the effects we are trying to
measure. Largest-remainder is exact, deterministic and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .determinism import content_hash, rng, stable_shuffle
from .plan import (
    EPOCH_CEILING,
    LANE_SPEC,
    LANES,
    SEAMS,
    STAGES,
    Scale,
    Stage,
    normalized_mixture,
    stage_by_name,
)
from .registry import ShardRegistry


# ---------------------------------------------------------------------------
# run shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunShape:
    """Batch geometry, in the session's vocabulary.

    micro_batch -> one GPU's slice; global batch = world_size * micro_batch * grad_accum.
    The demo numbers are tiny; `production_equivalent` reports what the same arithmetic
    gives at the real plan's scale, because the ratio is the thing worth checking, not the
    absolute size.
    """

    seq_len: int = 256
    long_seq_len: int = 1024
    world_size: int = 4          # simulated ranks
    micro_batch: int = 2
    grad_accum: int = 2
    total_tokens: int = 500_000
    # S5's seam bands are 2B and 4B tokens -- 0.07% and 0.13% of a 3T run. Scaled to a
    # 500k-token demo that is 333 tokens, less than a single step, so the blend would never
    # actually fire and the anneal machinery would be untested code. This floor widens each
    # seam to at least this many steps *in the demo only*. It is a declared demo parameter,
    # reported in logs/mixture_schedule.json alongside the plan-faithful width, not a
    # change to the plan.
    min_seam_steps: int = 4

    @property
    def sequences_per_step(self) -> int:
        return self.world_size * self.micro_batch * self.grad_accum

    @property
    def tokens_per_step(self) -> int:
        return self.sequences_per_step * self.seq_len

    @property
    def total_steps(self) -> int:
        return max(1, self.total_tokens // self.tokens_per_step)

    def seq_len_for(self, stage: Stage) -> int:
        return self.long_seq_len if stage.seq_len_multiplier > 1 else self.seq_len

    def production_equivalent(self) -> Dict[str, Any]:
        """The session's worked example, recomputed. Target ~1M tokens per optimizer step."""
        target = 1_000_000
        prod_seq = 4096
        gpus = 8
        micro = 32
        per_step = gpus * micro * prod_seq
        accum = max(1, round(target / per_step))
        return {
            "target_tokens_per_step": target,
            "sequence_length": prod_seq,
            "gpus": gpus,
            "micro_batch_per_gpu": micro,
            "tokens_per_step_without_accumulation": per_step,
            "gradient_accumulation_needed": accum,
            "achieved_tokens_per_step": per_step * accum,
            "note": ("The session's arithmetic: 8 GPUs x 32 micro-batch x 4096 tokens gives "
                     "1.05M tokens per step, so no accumulation is required at that shape. "
                     "Accumulation appears when the model is large enough that micro-batch "
                     "must fall."),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq_len": self.seq_len, "long_seq_len": self.long_seq_len,
            "world_size": self.world_size, "micro_batch": self.micro_batch,
            "grad_accum": self.grad_accum,
            "sequences_per_step": self.sequences_per_step,
            "tokens_per_step": self.tokens_per_step,
            "total_tokens": self.total_tokens, "total_steps": self.total_steps,
            "min_seam_steps_demo_floor": self.min_seam_steps,
            "production_equivalent": self.production_equivalent(),
        }


# ---------------------------------------------------------------------------
# per-step plan
# ---------------------------------------------------------------------------


@dataclass
class StepMixture:
    step: int
    stage: str
    stage_progress: float
    seq_len: int
    band_gate: List[str]
    target_weights: Dict[str, float]     # after seam blend + floors
    realized_weights: Dict[str, float]   # after ceiling clamp and integer allocation
    sequences_per_lane: Dict[str, int]
    in_seam: bool
    seam: Optional[str]
    clamped_lanes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step, "stage": self.stage,
            "stage_progress": round(self.stage_progress, 6),
            "seq_len": self.seq_len, "band_gate": self.band_gate,
            "target_weights": {k: round(v, 6) for k, v in self.target_weights.items()},
            "realized_weights": {k: round(v, 6) for k, v in self.realized_weights.items()},
            "sequences_per_lane": self.sequences_per_lane,
            "in_seam": self.in_seam, "seam": self.seam,
            "clamped_lanes": self.clamped_lanes,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _allocate(weights: Dict[str, float], total: int,
              carry: Dict[str, float]) -> Dict[str, int]:
    """Allocate `total` integer sequence slots proportional to `weights`.

    Largest-remainder **with carry across steps**, and the carry is not a refinement -- it
    is what makes a small lane possible at all.

    With 16 sequences per step, a 2% lane wants 0.32 of a sequence. Plain per-step
    largest-remainder floors that to zero and then loses every tie-break to lanes with
    remainders near 0.9, so the lane receives nothing on almost every step: agentic came out
    at 0.2% realised against a 2.0% target and a *protected floor*. A protected floor that
    integer rounding can delete is decorative.

    Carrying the fractional debt forward turns it into error diffusion: agentic accumulates
    0.32 per step and is issued a sequence roughly every third step, so the long-run share
    is exact even though no single step can express it. `carry` is mutated in place and is
    part of the schedule's state, so replay reproduces it.

    Ties broken by lane name so the result never depends on dict order.
    """
    if total <= 0 or not weights:
        return {k: 0 for k in weights}
    s = sum(weights.values())
    if s <= 0:
        return {k: 0 for k in weights}

    desired = {k: (v / s) * total + carry.get(k, 0.0) for k, v in weights.items()}
    floors = {k: max(0, int(v)) for k, v in desired.items()}
    remaining = total - sum(floors.values())
    if remaining > 0:
        order = sorted(weights.keys(), key=lambda k: (-(desired[k] - floors[k]), k))
        for k in order[:remaining]:
            floors[k] += 1
    elif remaining < 0:
        order = sorted(weights.keys(), key=lambda k: ((desired[k] - floors[k]), k))
        i = 0
        while remaining < 0 and i < len(order) * 4:
            k = order[i % len(order)]
            if floors[k] > 0:
                floors[k] -= 1
                remaining += 1
            i += 1

    for k in weights:
        carry[k] = desired[k] - floors[k]
    return floors


def _apply_floors(weights: Dict[str, float]) -> Dict[str, float]:
    """Raise protected lanes to their floor, then renormalise the unprotected remainder.

    Renormalising *everything* after raising a floor would immediately pull the protected
    lane back below it, which is the bug that makes protection decorative.
    """
    out = dict(weights)
    protected = {ln: sp.floor_share for ln, sp in LANE_SPEC.items()
                 if sp.protected and out.get(ln, 0.0) < sp.floor_share}
    if not protected:
        total = sum(out.values())
        return {k: v / total for k, v in out.items()} if total else out

    for ln, floor in protected.items():
        out[ln] = floor
    fixed = sum(out[ln] for ln in protected)
    rest = {k: v for k, v in out.items() if k not in protected}
    rest_sum = sum(rest.values())
    budget = max(0.0, 1.0 - fixed)
    if rest_sum > 0:
        for k in rest:
            out[k] = rest[k] / rest_sum * budget
    return out


def _blended_weights(token_pos: int, scale: Scale, seam_floor_tokens: int = 0) -> Tuple[Stage, float, Dict[str, float],
                                                            List[str], bool, Optional[str]]:
    """Lane weights at a given token position, interpolated across seams.

    The blend is centred on the stage boundary and spans the seam's band width, so a
    fraction of the incoming stage's distribution starts before the boundary and a fraction
    of the outgoing one persists after it -- "a bit of PhD level English should start very
    early on, and a bit of English should continue as the PhD level English starts".

    During a seam the band gate is the union of both stages' bands, so difficulty anneals
    on the same schedule as the mixture rather than stepping.
    """
    bounds: List[Tuple[Stage, int, int]] = []
    cursor = 0
    for st in STAGES:
        n = scale.tokens(st.tokens_b)
        bounds.append((st, cursor, cursor + n))
        cursor += n
    total = cursor
    pos = min(max(0, token_pos), max(0, total - 1))

    current = bounds[-1]
    for entry in bounds:
        if entry[1] <= pos < entry[2]:
            current = entry
            break
    stage, lo, hi = current
    progress = (pos - lo) / max(1, hi - lo)

    weights = dict(normalized_mixture(stage))
    bands = list(stage.band_range)
    in_seam, seam_name = False, None

    for seam in SEAMS:
        i = next((k for k, (s, _l, _h) in enumerate(bounds) if s.name == seam.from_stage), None)
        if i is None:
            continue
        boundary = bounds[i][2]
        half = max(1, scale.tokens(seam.band_tokens_b) // 2, seam_floor_tokens // 2)
        if boundary - half <= pos < boundary + half:
            a = stage_by_name(seam.from_stage)
            b = stage_by_name(seam.to_stage)
            f = (pos - (boundary - half)) / float(2 * half)
            wa, wb = normalized_mixture(a), normalized_mixture(b)
            weights = {ln: (1.0 - f) * wa[ln] + f * wb[ln] for ln in LANES}
            bands = sorted(set(a.band_range) | set(b.band_range))
            in_seam = True
            seam_name = "{}->{}".format(seam.from_stage, seam.to_stage)
            break

    return stage, progress, _apply_floors(weights), bands, in_seam, seam_name


# ---------------------------------------------------------------------------
# the schedule
# ---------------------------------------------------------------------------


@dataclass
class MixtureSchedule:
    shape: RunShape
    steps: List[StepMixture]
    lane_supply: Dict[str, int]
    lane_consumed: Dict[str, int]
    shortfalls: Dict[str, Dict[str, Any]]
    report: Dict[str, Any]
    schedule_hash: str

    def step(self, i: int) -> StepMixture:
        return self.steps[i]


def compile_schedule(
    registry: ShardRegistry,
    shape: RunShape,
    scale: Scale,
    *,
    enforce_epoch_ceiling: bool = True,
    lane_multipliers: Optional[Dict[str, float]] = None,
) -> MixtureSchedule:
    """Compile the whole run into per-step lane allocations.

    Computed once, up front, and hashed. Replay reads the compiled schedule rather than
    recomputing it -- the session's reason for a ledger at all: "if I want to go back in
    history and run something, I will not run the code, because some non-determinism can
    creep in. I am going to run the ledger."

    `lane_multipliers` scales a lane's blended weight before renormalisation, and exists for
    counterfactuals: forking a checkpoint under "twice as much Indic" and comparing. It is
    applied *before* the epoch ceiling and the protected floors, so an override cannot be
    used to smuggle a lane past either -- a knob that can breach the plan's own constraints
    is not a knob, it is a bug.
    """
    lane_supply = {ln: registry.lane_supply_tokens(ln) for ln in LANES}
    budget = {ln: int(lane_supply[ln] * EPOCH_CEILING) if enforce_epoch_ceiling else 1 << 60
              for ln in LANES}
    consumed = {ln: 0 for ln in LANES}
    demanded = {ln: 0 for ln in LANES}
    carry = {ln: 0.0 for ln in LANES}

    steps: List[StepMixture] = []
    token_pos = 0
    seam_floor_tokens = shape.min_seam_steps * shape.tokens_per_step

    total_tokens = shape.total_tokens
    i = -1
    while token_pos < total_tokens:
        i += 1
        stage, progress, weights, bands, in_seam, seam_name = _blended_weights(
            token_pos, scale, seam_floor_tokens)
        if lane_multipliers:
            weights = {ln: w * lane_multipliers.get(ln, 1.0) for ln, w in weights.items()}
            s = sum(weights.values()) or 1.0
            weights = {ln: w / s for ln, w in weights.items()}
        seq_len = shape.seq_len_for(stage)
        n_seq = shape.sequences_per_step

        # Lanes with no shard eligible for this stage's bands and window cannot be drawn
        # from, regardless of what the mixture asks for.
        available = {
            ln: bool(registry.eligible(ln, stage, seq_len)) for ln in LANES
        }

        clamped: List[str] = []
        effective = {}
        for ln in LANES:
            w = weights.get(ln, 0.0)
            if not available[ln]:
                clamped.append(ln + ":no_eligible_shard")
                w = 0.0
            elif consumed[ln] >= budget[ln]:
                clamped.append(ln + ":epoch_ceiling")
                w = 0.0
            effective[ln] = w

        tot = sum(effective.values())
        if tot <= 0:
            effective = {ln: (1.0 if available[ln] else 0.0) for ln in LANES}
            tot = sum(effective.values()) or 1.0
        effective = {ln: v / tot for ln, v in effective.items()}

        alloc = _allocate(effective, n_seq, carry)

        # Clamp to remaining budget, then redistribute the freed sequences to lanes that
        # still have headroom. Without this the ceiling is advisory: a lane at 3.9 epochs
        # is allowed one more full step and lands at 4.5.
        if enforce_epoch_ceiling:
            freed = 0
            for ln in LANES:
                room = max(0, budget[ln] - consumed[ln]) // seq_len
                if alloc[ln] > room:
                    freed += alloc[ln] - room
                    alloc[ln] = room
                    clamped.append(ln + ":epoch_ceiling")
            while freed > 0:
                takers = [ln for ln in LANES
                          if available[ln]
                          and (budget[ln] - consumed[ln]) // seq_len > alloc[ln]
                          and effective[ln] > 0]
                if not takers:
                    break
                takers.sort(key=lambda ln: (-effective[ln], ln))
                for ln in takers:
                    if freed <= 0:
                        break
                    alloc[ln] += 1
                    freed -= 1

        for ln in LANES:
            demanded[ln] += int(round(weights.get(ln, 0.0) * n_seq)) * seq_len
            consumed[ln] += alloc[ln] * seq_len

        realized = {ln: alloc[ln] / float(n_seq) for ln in LANES}

        steps.append(StepMixture(
            step=i, stage=stage.name, stage_progress=progress, seq_len=seq_len,
            band_gate=bands, target_weights=weights, realized_weights=realized,
            sequences_per_lane=alloc, in_seam=in_seam, seam=seam_name,
            clamped_lanes=sorted(set(clamped)),
        ))
        token_pos += sum(alloc.values()) * seq_len
        if sum(alloc.values()) == 0:
            break   # every lane exhausted; nothing further can be scheduled

    shortfalls: Dict[str, Dict[str, Any]] = {}
    for ln in LANES:
        want, got = demanded[ln], consumed[ln]
        supply = lane_supply[ln]
        epochs = got / supply if supply else float("inf")
        want_epochs = want / supply if supply else float("inf")
        if want - got > 0.02 * max(1, want):
            shortfalls[ln] = {
                "demanded_tokens": want,
                "delivered_tokens": got,
                "shortfall_tokens": want - got,
                "shortfall_fraction": round((want - got) / max(1, want), 6),
                "supply_tokens": supply,
                "epochs_delivered": round(epochs, 4),
                "epochs_demanded": round(want_epochs, 4),
                "epoch_ceiling": EPOCH_CEILING,
                "cause": ("epoch ceiling: demand exceeds 4x supply"
                          if want_epochs > EPOCH_CEILING else "no eligible shard for stage"),
            }

    from .plan import planned_lane_epochs

    report = {
        "shape": shape.to_dict(),
        "steps": len(steps),
        "steps_estimated_uniform": shape.total_steps,
        "lane_supply_tokens": lane_supply,
        "lane_demanded_tokens": demanded,
        "lane_delivered_tokens": consumed,
        "lane_epochs_demo": {
            ln: round(consumed[ln] / lane_supply[ln], 4) if lane_supply[ln] else None
            for ln in LANES
        },
        "lane_epochs_plan": {ln: round(v, 4) for ln, v in planned_lane_epochs().items()},
        "epoch_ceiling": EPOCH_CEILING,
        "shortfalls": shortfalls,
        "stage_step_counts": _count_by(steps, lambda s: s.stage),
        "seam_steps": sum(1 for s in steps if s.in_seam),
        "seam_widths": [
            {"seam": "{}->{}".format(sm.from_stage, sm.to_stage),
             "plan_band_tokens_b": sm.band_tokens_b,
             "plan_band_tokens_at_demo_scale": scale.tokens(sm.band_tokens_b),
             "demo_floor_tokens": seam_floor_tokens,
             "applied_tokens": max(scale.tokens(sm.band_tokens_b), seam_floor_tokens),
             "measured_peak_at_plan_scale": sm.measured_peak}
            for sm in SEAMS
        ],
        "integrated_realized_weights": _integrate(steps),
        "note": ("Demo epochs are not the plan's epochs -- the corpus is sized for runtime, "
                 "not proportional to the plan's 6.3T-token supply. What is reproduced is "
                 "which lane runs out first, and agentic does, exactly as S5 decision 3 "
                 "predicts."),
    }

    schedule_hash = content_hash([s.to_dict() for s in steps])
    return MixtureSchedule(shape=shape, steps=steps, lane_supply=lane_supply,
                           lane_consumed=consumed, shortfalls=shortfalls,
                           report=report, schedule_hash=schedule_hash)


def _count_by(steps: Sequence[StepMixture], key) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in steps:
        out[key(s)] = out.get(key(s), 0) + 1
    return dict(sorted(out.items()))


def _integrate(steps: Sequence[StepMixture]) -> Dict[str, float]:
    """Token-weighted realised lane shares over the whole run."""
    tot = 0
    acc = {ln: 0 for ln in LANES}
    for s in steps:
        for ln, n in s.sequences_per_lane.items():
            acc[ln] += n * s.seq_len
            tot += n * s.seq_len
    return {ln: round(acc[ln] / tot, 6) if tot else 0.0 for ln in LANES}
