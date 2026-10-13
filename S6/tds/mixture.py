"""Mixture schedule: S5's curriculum stages compiled into per-step lane quotas.

Three things this module must get right.

1. **Stage-scoped budgets.** Lane demand is computed against the *stage window*, never the run total.
   The course compiler gets this wrong: measured, its anneal profile computes demand against the full
   120B budget while its own timeline shows the anneal window as 18B - a 6.7x overstatement. S5 hit
   the same class of bug in its own session (~50x on anneal demand). Every stage here carries explicit
   `token_start`/`token_end` and demand is `(token_end - token_start) * share / (1 - rejection)`.

2. **Protected floors as mechanism, not preference.** S5 measured OPUS driving unprotected lanes to
   zero: Indic got 0 tokens in 60/60 iterations, agentic 0 in 59/60, and no proxy rebalancing rescued
   it. So floors are injected *after* selection and are invisible to the selector.

3. **A canonical global ordering.** The planner emits a flat, ordered list of samples per step. A
   sample's position in the run is one integer - the global sample index - and the rank that serves it
   is `index % world_size`, computed at load time. This is what lets a run crash on 4 ranks and resume
   on 8 without dropping or repeating anything; per-rank cursors cannot do that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .canon import chash, derive_rng
from .packing import PackedSequence


@dataclass(frozen=True)
class Stage:
    id: str
    name: str
    token_start: int
    token_end: int
    sequence_length: int
    mixture: dict[str, float]
    difficulty_bands: tuple[str, ...]

    @property
    def token_span(self) -> int:
        return self.token_end - self.token_start


@dataclass
class MixturePlan:
    plan_id: str
    run_tokens: int
    stages: list[Stage]
    protected_floors: dict[str, float]
    supply_tokens: dict[str, int]
    budget_share: dict[str, float]
    epoch_ceiling: float
    seams: list[dict]
    packing_policy: dict[str, str]
    anneal_reserve: dict[str, int]
    opus: dict
    scale_divisor: int = 1
    raw: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path, *, scale_divisor: int = 1) -> "MixturePlan":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        stages = [
            Stage(
                id=s["id"], name=s["name"],
                token_start=s["token_start"] // scale_divisor,
                token_end=s["token_end"] // scale_divisor,
                sequence_length=s["sequence_length"],
                mixture=dict(s["mixture"]),
                difficulty_bands=tuple(s["difficulty_bands"]),
            )
            for s in raw["stages"]
        ]
        return cls(
            plan_id=raw["plan_id"], run_tokens=raw["run_tokens"] // scale_divisor, stages=stages,
            protected_floors=dict(raw["protected_floors"]),
            supply_tokens=dict(raw["supply_tokens"]),
            budget_share=dict(raw["budget_share"]),
            epoch_ceiling=float(raw["epoch_ceiling"]),
            seams=list(raw["seams"]), packing_policy=dict(raw["packing_policy"]),
            anneal_reserve=dict(raw["anneal_reserve"]), opus=dict(raw["opus"]),
            scale_divisor=scale_divisor, raw=raw,
        )

    @property
    def lanes(self) -> list[str]:
        return sorted(self.budget_share)

    # ---------------------------------------------------------------- stages
    def stage_for_token(self, pos: int) -> Stage:
        for st in self.stages:
            if st.token_start <= pos < st.token_end:
                return st
        return self.stages[-1]

    def seam_weight(self, pos: int) -> tuple[Stage, Stage, float]:
        """Linear blend across a seam band. Returns (from_stage, to_stage, weight_of_to).

        S5 rule 2: blend every seam over >=2B tokens (measured peak gradient spike 2.8x, under the 3x
        stability threshold); the final seam gets 4B. Outside a band the weight is 0 or 1.
        """
        cur = self.stage_for_token(pos)
        for seam in self.seams:
            at = seam["at"] // self.scale_divisor
            band = max(1, seam["band_tokens"] // self.scale_divisor)
            lo = at - band // 2
            hi = at + band // 2
            if lo <= pos < hi:
                a = next(s for s in self.stages if s.id == seam["from"])
                b = next(s for s in self.stages if s.id == seam["to"])
                w = (pos - lo) / max(1, hi - lo)
                return a, b, w
        return cur, cur, 1.0

    def effective_mixture(self, pos: int) -> dict[str, float]:
        """Lane shares at a token position: seam-blended, then floored.

        The floors must survive normalisation. An earlier version renormalised the whole vector
        after raising protected lanes, which scaled those lanes straight back below their floor -
        a floor that is applied and then silently undone. So protected lanes are pinned and only
        the unprotected remainder is rescaled to absorb the difference.
        """
        a, b, w = self.seam_weight(pos)
        mix = {ln: (1 - w) * a.mixture.get(ln, 0.0) + w * b.mixture.get(ln, 0.0) for ln in self.lanes}

        protected = {ln: max(mix.get(ln, 0.0), f) for ln, f in self.protected_floors.items()}
        prot_total = sum(protected.values())
        free = {k: v for k, v in mix.items() if k not in self.protected_floors}
        free_total = sum(free.values())
        remaining = max(0.0, 1.0 - prot_total)

        out = dict(protected)
        if free_total > 0 and remaining > 0:
            for k, v in free.items():
                out[k] = v / free_total * remaining
        else:
            # protected lanes alone exceed the batch: scale them down proportionally and report
            # nothing for the free lanes. This is a plan error, not a runtime one.
            for k in free:
                out[k] = 0.0
            if prot_total > 1.0:
                out = {k: v / prot_total for k, v in out.items()}
        return out

    # ---------------------------------------------------------------- quotas
    @staticmethod
    def largest_remainder(shares: dict[str, float], n: int) -> dict[str, int]:
        """Integer allocation of n slots that sums to exactly n, deterministically.

        Floor everything, then hand out the remaining slots by largest fractional part, breaking ties
        by lane name so the result never depends on dict ordering.
        """
        raw = {k: v * n for k, v in shares.items()}
        base = {k: int(v) for k, v in raw.items()}
        left = n - sum(base.values())
        order = sorted(raw, key=lambda k: (-(raw[k] - base[k]), k))
        for i in range(left):
            base[order[i % len(order)]] += 1
        return base

    def lane_quota(self, pos: int, global_batch: int) -> dict[str, int]:
        return self.largest_remainder(self.effective_mixture(pos), global_batch)

    def floor_quota(self, global_batch: int, delivered: dict[str, int] | None = None,
                    total_delivered: int = 0) -> dict[str, int]:
        """Samples each protected lane must receive THIS step to keep its running share on floor.

        Per-batch floors do not work at realistic batch sizes. With a global batch of 8 the smallest
        expressible share is 12.5%, so a 2% agentic floor becomes 12.5% and three protected lanes
        lock 37.5% of every batch. Measured on an earlier build: agentic delivered **18.9% against a
        2% floor** and Indic 26.8% against 12%.

        That is not a toy artifact - it is the same failure S5 measured in the course's own always-on
        protected lane, where a *stated 8% design value delivered ~15% of trained tokens*. A floor
        expressed in per-batch integers is a floor specified in the wrong units.

        So the floor is enforced over the RUN, not the batch: a protected lane is injected only while
        its cumulative delivered share sits below the floor. Long-run delivery converges to the floor
        from above by at most one sample per lane, and the overshoot no longer scales with the number
        of protected lanes.
        """
        import math
        delivered = delivered or {}
        out: dict[str, int] = {}
        horizon = total_delivered + global_batch
        for ln, f in sorted(self.protected_floors.items()):
            required = math.ceil(f * horizon)
            out[ln] = max(0, min(global_batch, required - delivered.get(ln, 0)))
        return out

    # ---------------------------------------------------------------- feasibility
    def feasibility(self, rejection_rate: float | None = None) -> dict:
        """Per-lane demand vs supply at PRODUCTION scale, with the 4-epoch ceiling.

        Uses the unscaled plan: this is arithmetic about the real run, not about the demo. Demand is
        inflated by the OPUS rejection rate because rejection multiplies required supply -
        `need = budget * share / (1 - r)` - which is the single most consequential formula in S6.
        """
        r = self.opus["rejection_rate_target"] if rejection_rate is None else rejection_rate
        raw = self.raw
        rows = []
        for lane in sorted(raw["budget_share"]):
            per_stage = []
            demand = 0.0
            for s in raw["stages"]:
                span = s["token_end"] - s["token_start"]
                d = span * s["mixture"].get(lane, 0.0) / (1 - r)
                demand += d
                per_stage.append({"stage": s["id"], "window_tokens": span,
                                  "share": s["mixture"].get(lane, 0.0), "demand_tokens": round(d)})
            supply = raw["supply_tokens"][lane]
            # r is the lane's own repetition factor: how many times its supply must be traversed to
            # meet its demand. Numerically this is what "epochs" already measured; the rename is the
            # point. A single global epoch count implies every lane degrades from repetition at the
            # same rate, and the mixture scaling law (arXiv:2605.12715, 2,000+ runs) measures that it
            # does not - a lane with 5x less data tolerates r near 30 while a 5x larger one wants
            # r < 7. The ceiling is therefore expressed per lane.
            r_lane = demand / supply if supply else float("inf")
            ceiling = float(raw.get("per_lane_epoch_ceiling", {}).get(lane, raw["epoch_ceiling"]))
            rows.append({
                "lane": lane,
                "demand_tokens": round(demand),
                "supply_tokens": supply,
                "repetition_factor_r": round(r_lane, 3),
                "epochs": round(r_lane, 3),      # retained: same quantity, prior name
                "ceiling_applied": ceiling,
                "ceiling_source": ("per_lane" if lane in raw.get("per_lane_epoch_ceiling", {})
                                   else "global_default"),
                "within_ceiling": r_lane <= ceiling,
                "protected_floor": raw["protected_floors"].get(lane),
                "per_stage": per_stage,
            })
        return {
            "scope": "production 3T, arithmetic only - no tokens trained",
            "rejection_rate": r,
            "demand_rule": "need = stage_window * stage_share / (1 - rejection_rate), summed over stages",
            "epoch_ceiling": raw["epoch_ceiling"],
            "per_lane_epoch_ceiling": raw.get("per_lane_epoch_ceiling", {}),
            "ceiling_parameterisation": {
                "quantity": "repetition_factor_r = demand / supply, per lane",
                "why_per_lane": "arXiv:2605.12715 measures degradation as a function of each lane's "
                                "OWN repetition factor, not a global epoch count. Scarcer lanes "
                                "tolerate higher r (5x-smaller target -> r about 30; 5x-larger -> "
                                "r < 7).",
                "why_the_level_is_unchanged": "the same paper measures optimal r and optimal target "
                                              "weight both FALLING with model size (9.5% at 101M -> "
                                              "1.9% at 539M) and states that extrapolation beyond "
                                              "about 900M is open. We plan ~30B. Re-shaping the "
                                              "ceiling is free; raising it is an extrapolation the "
                                              "evidence does not support, so the level stays at "
                                              f"{raw['epoch_ceiling']} (arXiv:2305.16264).",
                "weakest_for_us": "their hardest extrapolation case is Swahili and their weakest "
                                  "predictions are on quality-tiered mixtures - i.e. exactly the "
                                  "Indic lane.",
            },
            "lanes": rows,
            "infeasible_lanes": [x["lane"] for x in rows if not x["within_ceiling"]],
        }

    def delivered_vs_target(self, batches: list[dict]) -> dict:
        """Compare the delivered lane mixture against the plan, weighted by where it was delivered.

        "Mixture compliance" checked only the protected floors, which is a lower bound on three of
        seven lanes and says nothing at all about the other four. An audit measured the gap that
        hid: total variation distance 0.275, indic at +101% of target, code at -53%, web absent for
        an entire stage - all while the floors-only check reported PASS.

        The target is stage-weighted rather than run-averaged, because the plan's whole structure is
        that shares change per stage; comparing against a run-average would score a correct
        curriculum as non-compliant and a flat mixture as perfect.
        """
        served: dict[str, int] = {}
        target_acc: dict[str, float] = {}
        n = 0
        for e in batches:
            stage = next((st for st in self.stages if st.id == e.get("curriculum_stage")), None)
            for lane, k in (e.get("mixture_lanes") or {}).items():
                served[lane] = served.get(lane, 0) + k
                n += k
            if stage:
                total_k = sum((e.get("mixture_lanes") or {}).values())
                for lane in self.lanes:
                    target_acc[lane] = target_acc.get(lane, 0.0) + \
                        stage.mixture.get(lane, 0.0) * total_k
        if not n:
            return {"measured": False, "reason": "no batches"}
        delivered = {ln: served.get(ln, 0) / n for ln in self.lanes}
        target = {ln: target_acc.get(ln, 0.0) / n for ln in self.lanes}
        tvd = 0.5 * sum(abs(delivered[ln] - target[ln]) for ln in self.lanes)
        rows = {ln: {"target": round(target[ln], 4), "delivered": round(delivered[ln], 4),
                     "abs_error": round(abs(delivered[ln] - target[ln]), 4),
                     "relative_error": (round((delivered[ln] - target[ln]) / target[ln], 3)
                                        if target[ln] > 1e-9 else None)}
                for ln in self.lanes}
        return {"measured": True, "samples": n, "by_lane": rows,
                "total_variation_distance": round(tvd, 4),
                "target_basis": "each stage's planned mixture, weighted by the samples actually "
                                "served in that stage",
                "note": "protected floors are a lower bound on three lanes; this is the only check "
                        "that constrains the other four, and the only one that can detect "
                        "OVER-delivery"}

    def dedup_sensitivity(self, rejection_rate: float | None = None) -> dict:
        """How much deduplication each lane's supply can absorb before it breaches its ceiling.

        S5's supply figures are **pre-dedup**: `S5/SUPPLY-AUDIT.md` s7 "Still not obtained" lists the
        Sangraha-Verified/IndicCorpV2 dedup rate and the judgment-corpus dedup rate as unmeasured.
        That is a stated gap, not an oversight - but it means every `within_ceiling` verdict in this
        plan is computed on an upper bound of the real supply.

        The published dedup literature makes the gap material rather than theoretical: near-duplicate
        removal on web corpora routinely removes tens of percent, and S5's own judgment-corpus
        p90/median ratio of 5.7 is a direct signal of boilerplate-heavy short documents.

        So rather than guess a rate, compute the *break-even*: the dedup fraction at which each lane
        crosses its own ceiling.

            r = demand / (supply * (1 - d))  <=  ceiling
            =>  d  <=  1 - demand / (supply * ceiling)

        A lane whose break-even is negative is already breached before any dedup at all.
        """
        feas = self.feasibility(rejection_rate)
        rows = []
        for row in feas["lanes"]:
            supply, demand = row["supply_tokens"], row["demand_tokens"]
            ceiling = row["ceiling_applied"]
            headroom = (1 - demand / (supply * ceiling)) if supply and ceiling else float("-inf")
            rows.append({
                "lane": row["lane"],
                "repetition_factor_r_at_published_supply": row["repetition_factor_r"],
                "max_tolerable_dedup_fraction": round(headroom, 4),
                "status": ("already_breached" if headroom < 0 else
                           "fragile" if headroom < 0.20 else "robust"),
            })
        return {
            "basis": "S5 supply figures are pre-dedup (S5/SUPPLY-AUDIT.md s7 lists the dedup rates "
                     "as still not obtained), so these are upper bounds on real supply",
            "rule": "d_max = 1 - demand / (supply * ceiling)",
            "lanes": sorted(rows, key=lambda r: r["max_tolerable_dedup_fraction"]),
            "consequence": "lanes marked fragile or already_breached cannot be signed off until the "
                           "dedup rate for their sources is measured. This is the cheapest "
                           "unblocking measurement in the whole plan.",
        }

    def max_feasible_rejection(self, lane: str, *, lo: float = 0.0, hi: float = 0.95) -> float:
        """Largest OPUS rejection rate this lane can absorb before breaching the epoch ceiling."""
        for _ in range(48):
            mid = (lo + hi) / 2
            row = next(x for x in self.feasibility(mid)["lanes"] if x["lane"] == lane)
            if row["within_ceiling"]:
                lo = mid
            else:
                hi = mid
        return round(lo, 4)

    def to_record(self) -> dict:
        return {
            "plan_id": self.plan_id, "scale_divisor": self.scale_divisor,
            "run_tokens_scaled": self.run_tokens,
            "stages": [{"id": s.id, "name": s.name, "token_start": s.token_start,
                        "token_end": s.token_end, "sequence_length": s.sequence_length,
                        "mixture": s.mixture, "difficulty_bands": list(s.difficulty_bands)}
                       for s in self.stages],
            "protected_floors": self.protected_floors,
            "packing_policy": self.packing_policy,
            "anneal_reserve": self.anneal_reserve,
            "plan_hash": chash(self.raw),
        }


# ==================================================================================================
# sample pools and the canonical global ordering
# ==================================================================================================

@dataclass
class PlannedSample:
    global_index: int
    step: int
    lane: str
    sample_id: str
    shard_id: str
    pool_index: int
    pass_number: int          # how many times this pool has wrapped: the repetition counter
    is_floor_injection: bool
    reserved: bool

    def to_record(self) -> dict:
        return {"global_index": self.global_index, "step": self.step, "lane": self.lane,
                "sample_id": self.sample_id, "shard_id": self.shard_id,
                "pool_index": self.pool_index, "pass_number": self.pass_number,
                "is_floor_injection": self.is_floor_injection, "reserved": self.reserved}


@dataclass
class LanePool:
    """Ordered, deterministically shuffled samples for one lane.

    `reserved` samples (S5's anneal reserve) are held out of the main-run portion of the pool and
    become drawable only once the scheduler enters the anneal stage.
    """
    lane: str
    samples: list[PackedSequence]
    shard_of: dict[str, str]
    order: list[int]
    reserved_ids: set[str] = field(default_factory=set)
    # doc-length reserve: sample_id -> the shortest stage sequence_length that can hold this
    # sample's source document without splitting it. A stage whose sequence length is below this
    # cannot draw the sample. See LENGTH RESERVE below.
    min_seq_len: dict[str, int] = field(default_factory=dict)
    band_of: dict[str, str] = field(default_factory=dict)
    length_reserve_relaxed: int = 0
    length_reserve_honoured: int = 0

    @classmethod
    def build(cls, lane: str, samples: list[PackedSequence], shard_of: dict[str, str],
              *, seed: int, reserved_ids: set[str] | None = None,
              difficulty_of: dict[str, str] | None = None,
              difficulty_ordered: bool = False,
              min_seq_len: dict[str, int] | None = None) -> "LanePool":
        idx = list(range(len(samples)))
        if difficulty_ordered and difficulty_of:
            # E6: order the lane internally by difficulty band, ties broken deterministically
            band_rank = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4, "B5": 5}
            idx.sort(key=lambda i: (band_rank.get(difficulty_of.get(samples[i].sample_id, "B2"), 2),
                                    samples[i].sample_id))
        else:
            rng = derive_rng(seed, "lane_order", lane)
            perm = rng.permutation(len(samples)).tolist() if samples else []
            idx = [int(i) for i in perm]
        return cls(lane=lane, samples=samples, shard_of=shard_of, order=idx,
                   reserved_ids=set(reserved_ids or ()), min_seq_len=dict(min_seq_len or {}),
                   band_of=dict(difficulty_of or {}))

    def draw(self, cursor: int, *, allow_reserved: bool,
             stage_seq_len: int | None = None,
             widest_seq_len: int | None = None,
             withheld: list[str] | None = None,
             stage_bands: tuple[str, ...] | None = None) -> tuple[PackedSequence, int, int]:
        """Return (sample, pool_index, pass_number), skipping anything this stage may not consume.

        AUDITABILITY OF THE SKIP
        ------------------------
        Skipping a reserved sample here is a *policy decision about training data*, and the original
        version of this method made it invisible: the sample simply never appeared downstream, so no
        ledger recorded that it was held back or why. That is the defect E1 rejected the mixed-lane
        shard arm for - a protected quantity you cannot audit is not protected, it is merely hoped
        for. `withheld` collects the anneal-reserved sample ids this call passed over, so the caller
        can turn each one into an OPUS `deferred / reserved_for_anneal` record.

        The skip itself is deliberately unchanged: the reserve must still keep these samples out of
        the pre-anneal stream. Only the record of it is new.

        LENGTH RESERVE
        --------------
        The session states the rule and the plan never carries it:

            "you cannot train on a big code at the beginning of the training when your context
             length is only 4096 ... you need to preserve that shard for future because in future
             you might train on a longer sequence."

        Long-form documents are the scarcest shape in the corpus - S5 budgets the long-context lane
        at 0.65 epochs assuming it arrives fresh at stage S4. But the default policy for general text
        is concatenate-and-chop, and S1-S3 run at 4k/8k, so a 40k-token document entering S1 is cut
        into ten short spans and consumed. The document survives; its value as long-context training
        data does not.

        This is the anneal reserve's problem on a different axis. The anneal reserve protects
        QUALITY; nothing in the plan protects LENGTH. So a sample whose source document needs a
        longer window than the current stage provides is undrawable until a stage that can hold it.
        """
        if not self.order:
            raise IndexError(f"lane pool {self.lane} is empty")
        n = len(self.order)
        probe = cursor
        for _ in range(n + 1):
            pos = probe % n
            i = self.order[pos]
            s = self.samples[i]
            # Prefer band-eligible samples for this stage. Without this the planner hands OPUS
            # candidates the curriculum gate will discard, the surviving population falls below the
            # top-k cut, and the proxy gate never rejects anything - measured: 0 score-based
            # rejections in 296 judgements. Protected lanes are exempt because their floor override
            # is *supposed* to reach past the stage gate, and skipping their ineligible samples
            # would silently delete that mechanism.
            blocked_band = (stage_bands is not None
                            and self.band_of.get(s.sample_id) is not None
                            and self.band_of.get(s.sample_id) not in stage_bands)
            blocked_reserve = (not allow_reserved) and s.sample_id in self.reserved_ids
            if blocked_reserve and withheld is not None and s.sample_id not in withheld:
                withheld.append(s.sample_id)
            need = self.min_seq_len.get(s.sample_id, 0)
            # Reserve only where reserving BUYS something: the document must not fit this stage's
            # window but must fit some later one. A document longer than every stage's window will
            # be split wherever it is used, so holding it back achieves nothing.
            blocked_length = (stage_seq_len is not None and widest_seq_len is not None
                              and stage_seq_len < need <= widest_seq_len)
            if not blocked_reserve and not blocked_length and not blocked_band:
                return s, i, probe // n
            probe += 1
        # Nothing drawable. A scheduler must not deadlock, so relax the LENGTH reserve (never the
        # anneal reserve) and record it: the relaxation count is the measurement of how badly this
        # corpus lacks short documents for its early stages.
        for probe2 in range(cursor, cursor + n):
            i = self.order[probe2 % n]
            s = self.samples[i]
            if allow_reserved or s.sample_id not in self.reserved_ids:
                self.length_reserve_relaxed += 1
                return s, i, probe2 // n
        # every sample reserved and reserve not allowed: fall back to the first
        i = self.order[cursor % n]
        return self.samples[i], i, cursor // n


@dataclass
class StreamPlanner:
    """Produces the canonical global sample ordering.

    `plan_step(step)` is a pure function of (plan, pools, seed, step). It emits *candidates*; OPUS then
    accepts, rejects, defers or floor-overrides them, and only accepted samples advance the ledger.
    Because OPUS scores against live model state, the ACCEPTED stream is not a pure function of step -
    which is precisely why replay reads the ledger instead of re-running the scheduler.
    """
    plan: MixturePlan
    pools: dict[str, LanePool]
    seed: int
    global_batch: int
    sequence_length: int
    _cursors: dict[str, int] = field(default_factory=dict)
    delivered: dict[str, int] = field(default_factory=dict)
    total_delivered: int = 0
    # (lane, sample_id) pairs withheld by the anneal reserve during the most recent plan_step.
    # Overwritten every call, never accumulated - see plan_step.
    last_reserve_deferrals: list[tuple[str, str]] = field(default_factory=list)
    # How many candidates to draw per delivered sample. OPUS cannot select from a pool the size of
    # the batch: with global_batch 8 and top-k, `keep_n = round(8 * 0.82) = 7`, so at most one
    # sample is ever cut and the maximum expressible rejection is 12.5% against a plan asking for
    # 18%. Measured consequence before this existed: the score gate fired 0 times in 296 judgements
    # and every recorded rejection came from a difficulty-band string comparison.
    #
    # The factor is not invented - it is the same 1/(1 - rejection_rate) the feasibility arithmetic
    # already uses to size supply (`need = window * share / (1 - r)`). The scheduler was not
    # honouring the assumption its own supply check was making.
    oversample: float = 1.0

    def __post_init__(self) -> None:
        self._cursors = {ln: 0 for ln in self.pools}
        if not self.delivered:
            self.delivered = {ln: 0 for ln in self.pools}

    def note_delivered(self, lanes: dict[str, int]) -> None:
        """Record what was actually SERVED, so floors track delivery rather than intent."""
        for ln, n in lanes.items():
            self.delivered[ln] = self.delivered.get(ln, 0) + n
            self.total_delivered += n

    def stage_seq_len(self, stage: "Stage") -> int:
        """The stage's own sequence length, mapped onto the demo's packing width.

        The plan's stages run at 4096/8192/32768. The demo packs at a single small width, so the
        ratio between stages is what carries over: a stage that is 1/8 the width of the widest stage
        may only draw documents that fit in 1/8 of the demo window.
        """
        widest = max(s.sequence_length for s in self.plan.stages)
        return max(1, round(self.sequence_length * stage.sequence_length / widest))

    def tokens_consumed(self, step: int) -> int:
        return step * self.global_batch * self.sequence_length

    def current_stage(self, step: int) -> Stage:
        return self.plan.stage_for_token(self.tokens_consumed(step))

    def plan_step(self, step: int, *, cursors: dict[str, int] | None = None,
                  start_index: int = 0) -> list[PlannedSample]:
        """Candidate global batch for `step`, in canonical order."""
        cur = dict(self._cursors if cursors is None else cursors)
        pos = self.tokens_consumed(step)
        stage = self.plan.stage_for_token(pos)
        in_anneal = stage.id == "anneal"
        # Draw enough candidates that the selector has something to reject.
        draw_batch = max(self.global_batch, int(round(self.global_batch * self.oversample)))
        quota = self.plan.lane_quota(pos, draw_batch)
        floors = {k: v for k, v in self.plan.floor_quota(
            self.global_batch, self.delivered, self.total_delivered).items() if v > 0}

        # protected floors are satisfied FIRST and are not negotiable
        for lane, need in floors.items():
            if lane in quota:
                quota[lane] = max(quota[lane], need)
        # trim the largest unprotected lane so the batch still sums to global_batch
        while sum(quota.values()) > draw_batch:
            donor = max((k for k in quota if k not in floors and quota[k] > 0),
                        key=lambda k: (quota[k], k), default=None)
            if donor is None:
                break
            quota[donor] -= 1
        while sum(quota.values()) < draw_batch:
            recv = max((k for k in quota if k not in floors), key=lambda k: (quota[k], k))
            quota[recv] += 1

        out: list[PlannedSample] = []
        # Anneal-reserved samples passed over while building this batch. They take no global index
        # and never enter the delivered stream - they exist so OPUS can record the deferral. Reset
        # per call so a speculative re-plan (resume, replay, exactly-once trace) cannot accumulate.
        withheld: list[tuple[str, str]] = []
        gi = start_index
        for lane in sorted(quota):
            pool = self.pools.get(lane)
            if pool is None or not pool.order:
                continue
            for _ in range(quota[lane]):
                before = pool.length_reserve_relaxed
                lane_withheld: list[str] = []
                s, pool_idx, passno = pool.draw(
                    cur[lane], allow_reserved=in_anneal,
                    stage_seq_len=self.stage_seq_len(stage),
                    widest_seq_len=self.sequence_length,
                    withheld=lane_withheld,
                    stage_bands=(None if lane in self.plan.protected_floors
                                 else stage.difficulty_bands))
                for sid in lane_withheld:
                    if (lane, sid) not in withheld:
                        withheld.append((lane, sid))
                if pool.length_reserve_relaxed == before:
                    pool.length_reserve_honoured += 1
                out.append(PlannedSample(
                    global_index=gi, step=step, lane=lane, sample_id=s.sample_id,
                    shard_id=pool.shard_of.get(s.sample_id, "?"),
                    pool_index=pool_idx, pass_number=passno,
                    is_floor_injection=lane in floors,
                    reserved=s.sample_id in pool.reserved_ids,
                ))
                cur[lane] += 1
                gi += 1
        self.last_reserve_deferrals = withheld
        if cursors is None:
            self._cursors = cur
        return out

    def peek_cursors(self) -> dict[str, int]:
        return dict(self._cursors)

    def set_cursors(self, cursors: dict[str, int]) -> None:
        self._cursors = dict(cursors)

    def loader_state(self) -> dict:
        return {"pool_cursors": dict(self._cursors), "delivered": dict(self.delivered),
                "total_delivered": self.total_delivered}

    def load_loader_state(self, st: dict) -> None:
        self._cursors = dict(st.get("pool_cursors", {}))
        self.delivered = dict(st.get("delivered", {}))
        self.total_delivered = int(st.get("total_delivered", 0))


def rank_of(global_index: int, world_size: int) -> int:
    """Which rank serves a given global sample index.

    A pure function of (index, world_size). This is the whole trick behind resuming on a different
    machine shape: position is one global integer, and the rank partition is recomputed on load
    rather than stored. A loader that persists 'rank r has consumed n samples' cannot do this.
    """
    return global_index % world_size


def partition(samples: list[PlannedSample], world_size: int) -> dict[int, list[PlannedSample]]:
    out: dict[int, list[PlannedSample]] = {r: [] for r in range(world_size)}
    for s in samples:
        out[rank_of(s.global_index, world_size)].append(s)
    return out
