"""The training plan, inherited from Session 5.

This module is the single source of truth for *what we decided*, expressed in the real
units the plan was written in: a 2,910B main run plus a 90B anneal, seven capability
lanes, five stages, seam bands measured in billions of tokens.

The demo does not use a different plan. It uses *this* plan divided by `Scale.factor`.
That matters: if the demo had its own hand-tuned numbers, the repo would be a toy that
resembles the plan rather than the plan executed small, and every ratio in the evidence
would be unfalsifiable.

Source: ERA5/S5 -- CURRICULUM.md (stage table, seam bands, difficulty bands),
DECISIONS.md (the twelve decisions), BENCHMARK-MAP.md (lane -> benchmark chains).
See docs/s5-inheritance.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# lanes
# ---------------------------------------------------------------------------

WEB = "general_web"
CODE = "code"
REASONING = "reasoning"
LONGCTX = "long_context"
INDIC = "indic"
STEM = "stem"
AGENTIC = "agentic"

LANES: List[str] = [WEB, CODE, REASONING, LONGCTX, INDIC, STEM, AGENTIC]


@dataclass(frozen=True)
class Lane:
    name: str
    budget_share: float          # S5 BUDGET-PROPOSAL final share of the whole run
    supply_tokens_b: float       # auditable unique supply, in billions
    benchmarks: Tuple[str, ...]  # S5 BENCHMARK-MAP
    protected: bool = False      # never allowed to fall below `floor_share`
    floor_share: float = 0.0
    # Decision 5 of the session: a code file cut mid-function is damaged. Lanes whose
    # documents must not be split carry this, and oversized documents are deferred to a
    # stage with a long enough window rather than chopped.
    forbid_midlane_split: bool = False
    # Structure-preserving packing: never concatenate two documents from this lane into
    # one sequence (agentic traces, SFT conversations).
    forbid_concatenation: bool = False


LANE_SPEC: Dict[str, Lane] = {
    WEB: Lane(
        name=WEB,
        budget_share=0.34,
        supply_tokens_b=4691.0,
        benchmarks=("MMLU", "common-sense (unmeasured)"),
    ),
    CODE: Lane(
        name=CODE,
        budget_share=0.24,
        supply_tokens_b=1103.0,
        benchmarks=("LiveCodeBench", "Aider Polyglot", "Codeforces"),
        forbid_midlane_split=True,
    ),
    REASONING: Lane(
        name=REASONING,
        budget_share=0.09,
        supply_tokens_b=85.1,
        benchmarks=("AIME 2024/2025", "GPQA Diamond", "HLE", "FrontierMath"),
        forbid_concatenation=True,
    ),
    LONGCTX: Lane(
        name=LONGCTX,
        budget_share=0.02,
        supply_tokens_b=100.0,
        benchmarks=("long-eval",),
        forbid_midlane_split=True,
    ),
    INDIC: Lane(
        name=INDIC,
        budget_share=0.17,
        supply_tokens_b=157.2,
        benchmarks=("MILU 10/10", "IndicGenBench 14/29"),
        protected=True,
        # S5 decision 11 + CURRICULUM.md: "Indic is deliberately flat at ~17% throughout.
        # A protected capability that ramps is still a residue of whatever budget is left
        # over; a flat lane is a commitment."
        floor_share=0.165,
    ),
    STEM: Lane(
        name=STEM,
        budget_share=0.12,
        supply_tokens_b=146.0,
        benchmarks=("GPQA", "AIME", "MMLU-science"),
    ),
    AGENTIC: Lane(
        name=AGENTIC,
        budget_share=0.02,
        supply_tokens_b=0.63,
        benchmarks=("BFCL v3", "tau-bench", "SWE-bench Verified"),
        protected=True,
        # S5 decision 3: 2% is a floor, not a preference. Measured: an unprotected
        # agentic lane goes to zero in 59 of 60 OPUS iterations.
        floor_share=0.01,
        forbid_concatenation=True,
    ),
}

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    name: str
    tokens_b: float                  # stage length in billions of tokens
    mixture: Dict[str, float]        # lane -> share within this stage, sums to 1.0
    band_range: Tuple[str, ...]      # permitted difficulty bands (S5 CURRICULUM.md)
    seq_len_multiplier: int = 1      # long-context stages widen the training window
    is_anneal: bool = False


# CURRICULUM.md stage table, verbatim. Verified to integrate to the lane budgets by
# `verify_plan()` below -- the constraint S5 notes the course widget never enforces.
STAGES: List[Stage] = [
    Stage(
        name="S1_seed",
        tokens_b=233.0,
        mixture={WEB: 0.561, CODE: 0.146, REASONING: 0.020, LONGCTX: 0.002,
                 INDIC: 0.172, STEM: 0.090, AGENTIC: 0.010},
        band_range=("B0", "B1"),
    ),
    Stage(
        name="S2_general",
        tokens_b=931.0,
        mixture={WEB: 0.453, CODE: 0.192, REASONING: 0.048, LONGCTX: 0.004,
                 INDIC: 0.169, STEM: 0.118, AGENTIC: 0.015},
        band_range=("B1", "B2"),
    ),
    Stage(
        name="S3_reasoning",
        tokens_b=1048.0,
        mixture={WEB: 0.281, CODE: 0.268, REASONING: 0.125, LONGCTX: 0.009,
                 INDIC: 0.169, STEM: 0.128, AGENTIC: 0.020},
        band_range=("B2", "B3", "B4"),
    ),
    Stage(
        name="S4_longctx",
        tokens_b=698.0,
        mixture={WEB: 0.204, CODE: 0.293, REASONING: 0.117, LONGCTX: 0.063,
                 INDIC: 0.172, STEM: 0.120, AGENTIC: 0.030},
        band_range=("B3", "B4", "B5"),
        seq_len_multiplier=4,
    ),
    Stage(
        name="anneal",
        tokens_b=90.0,
        mixture={WEB: 0.08, CODE: 0.20, REASONING: 0.18, LONGCTX: 0.08,
                 INDIC: 0.28, STEM: 0.10, AGENTIC: 0.08},
        band_range=("B4", "B5"),
        # Deliberately *not* widened. The anneal is a data-quality stage -- best data saved
        # for last -- while context extension is S4's job. Widening it too would make the
        # anneal a single step of the demo run, since it is only 3% of the budget.
        seq_len_multiplier=1,
        is_anneal=True,
    ),
]

MAIN_RUN_TOKENS_B = 2910.0
ANNEAL_TOKENS_B = 90.0
TOTAL_TOKENS_B = MAIN_RUN_TOKENS_B + ANNEAL_TOKENS_B

# ---------------------------------------------------------------------------
# seams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Seam:
    """A blended transition between two stages.

    S5 measured this rather than guessing: frozen embeddings spike 151x versus 8.0x
    unfrozen at identical sharpness, and with embeddings frozen *no* warmup band under 5B
    tokens brings the transition under the 3x stability threshold. Hence rule 1
    (embeddings stay trainable across every seam, enforced in the trainer) and rule 2
    (blend over >= 2B tokens).
    """

    from_stage: str
    to_stage: str
    band_tokens_b: float
    measured_peak: float


SEAMS: List[Seam] = [
    Seam("S1_seed", "S2_general", 2.0, 2.8),
    Seam("S2_general", "S3_reasoning", 2.0, 2.8),
    Seam("S3_reasoning", "S4_longctx", 2.0, 2.8),
    # Double-width: largest distribution shift in the run (web 20% -> 8%, Indic
    # 17% -> 28%) and it lands where a diverged run is most expensive to lose.
    Seam("S4_longctx", "anneal", 4.0, 2.0),
]

STABILITY_THRESHOLD = 3.0
EMBEDDINGS_MUST_REMAIN_TRAINABLE = True  # S5 seam rule 1, asserted in trainer.py

# ---------------------------------------------------------------------------
# difficulty bands  (closes the item S5 explicitly deferred to S6)
# ---------------------------------------------------------------------------

# CURRICULUM.md: "the mechanism for enforcing a difficulty band at sampling time
# (classifier? source tagging?) is not yet specified and belongs in the S6 data-stream
# work." Our answer: source tagging at shard-build time, enforced as a hard gate at
# sampling time. A document carries its band in the immutable manifest, so the gate is a
# lookup rather than a runtime classifier -- reproducible, and auditable after the fact.

DIFFICULTY_BANDS: List[str] = ["B0", "B1", "B2", "B3", "B4", "B5"]

BAND_LEVEL: Dict[str, str] = {
    "B0": "nursery",
    "B1": "grade-school",
    "B2": "high-school",
    "B3": "undergraduate",
    "B4": "graduate",
    "B5": "research/PhD",
}

# Reasoning-trace length bands. S5 caps L3 at 10%: measured HIGH -> ULTRA costs 2.15x the
# tokens for +4 accuracy points. Over-representing long traces teaches the model that more
# thinking is always better.
REASONING_LENGTH_BANDS: Dict[str, Tuple[int, int, float]] = {
    "L0": (0, 64, 0.30),
    "L1": (64, 256, 0.35),
    "L2": (256, 1024, 0.25),
    "L3": (1024, 8192, 0.10),
}

EPOCH_CEILING = 4.0  # arXiv:2305.16264, applied per lane

# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scale:
    """Maps the real plan onto a run that finishes while you watch it.

    Every absolute token figure in the plan is divided by `factor`. Shares, ratios, seam
    widths *as a fraction of stage length*, band gates and protected floors are all
    scale-invariant and pass through untouched.
    """

    factor: float

    def tokens(self, tokens_b: float) -> int:
        return max(1, int(round(tokens_b * 1e9 / self.factor)))

    @property
    def total_tokens(self) -> int:
        return self.tokens(TOTAL_TOKENS_B)


# 3e12 real tokens -> 6e5 demo tokens.
DEMO_SCALE = Scale(factor=5.0e6)

# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def stage_by_name(name: str) -> Stage:
    for st in STAGES:
        if st.name == name:
            return st
    raise KeyError(name)


def main_run_stages() -> List[Stage]:
    return [st for st in STAGES if not st.is_anneal]


def normalized_mixture(stage: Stage) -> Dict[str, float]:
    """Stage mixture renormalised to sum to exactly 1.0.

    S5's CURRICULUM.md publishes shares rounded to 0.1%, so three of the five stage rows
    sum to 1.001 or 0.999 as written. That is a display artifact, not a planning error,
    but it must not propagate into token accounting -- an 0.001 residue over 2,910B is
    ~3B tokens of unallocated budget. We renormalise here, deterministically, and
    `mixture_rounding_residue()` reports what was absorbed so the correction is visible
    rather than silent.
    """
    total = sum(stage.mixture.values())
    return {lane: share / total for lane, share in stage.mixture.items()}


def mixture_rounding_residue() -> Dict[str, float]:
    """Per-stage deviation of the published mixture from 1.0, before renormalisation."""
    return {st.name: sum(st.mixture.values()) - 1.0 for st in STAGES}


def integrated_lane_shares(include_anneal: bool = False) -> Dict[str, float]:
    """Token-weighted mean of each lane's share across stages.

    S5's constraint: a set of stage anchors that each sum to 100% can still spend a lane
    far above or below its allocation once weighted by stage length.

    Integrated over the **main run only** by default. The lane budget shares in
    BUDGET-PROPOSAL are shares of the 2,910B main run; the 90B anneal is budgeted
    separately with its own mixture (which is why Indic reads 28% and agentic 8% there --
    deliberate concentration, not a budget breach). Integrating the anneal in would make
    every lane appear to miss its target by roughly the anneal's weight.
    """
    stages = STAGES if include_anneal else main_run_stages()
    total = sum(st.tokens_b for st in stages)
    out: Dict[str, float] = {ln: 0.0 for ln in LANES}
    for st in stages:
        w = st.tokens_b / total
        for lane, share in normalized_mixture(st).items():
            out[lane] += w * share
    return out


def verify_plan(tolerance: float = 0.006) -> List[str]:
    """Return a list of problems with the plan. Empty list means the plan is coherent.

    Run at startup. A plan that does not satisfy its own constraints is not a plan, and
    finding that out at token 400B is expensive.
    """
    problems: List[str] = []

    for st in STAGES:
        total = sum(st.mixture.values())
        # 0.1%-rounded shares in the published table are expected; anything larger is a
        # transcription error and must fail loudly.
        if abs(total - 1.0) > 0.0015:
            problems.append(
                "stage {} mixture sums to {:.6f}, beyond display-rounding tolerance".format(
                    st.name, total
                )
            )
        missing = set(LANES) - set(st.mixture)
        if missing:
            problems.append("stage {} missing lanes {}".format(st.name, sorted(missing)))
        for band in st.band_range:
            if band not in DIFFICULTY_BANDS:
                problems.append("stage {} references unknown band {}".format(st.name, band))

    integrated = integrated_lane_shares()
    for lane, spec in LANE_SPEC.items():
        drift = integrated[lane] - spec.budget_share
        if abs(drift) > tolerance:
            problems.append(
                "lane {} integrates to {:.4f} but budget is {:.4f} (drift {:+.4f})".format(
                    lane, integrated[lane], spec.budget_share, drift
                )
            )

    total_stage_tokens = sum(st.tokens_b for st in STAGES)
    if abs(total_stage_tokens - TOTAL_TOKENS_B) > 1.0:
        problems.append(
            "stage lengths total {:.1f}B but run is {:.1f}B".format(
                total_stage_tokens, TOTAL_TOKENS_B
            )
        )

    # Protected floors must hold in every stage, not on average.
    for st in STAGES:
        norm = normalized_mixture(st)
        for lane, spec in LANE_SPEC.items():
            # Slack of one rounding unit: S1 agentic is published at exactly the 1.0%
            # floor, and renormalising a row that sums to 1.001 lands it at 0.999%.
            if spec.protected and norm[lane] < spec.floor_share - 0.0015:
                problems.append(
                    "protected lane {} is {:.3f} in stage {}, below floor {:.3f}".format(
                        lane, st.mixture[lane], st.name, spec.floor_share
                    )
                )

    names = [st.name for st in STAGES]
    for seam in SEAMS:
        if seam.from_stage not in names or seam.to_stage not in names:
            problems.append("seam {}->{} references unknown stage".format(
                seam.from_stage, seam.to_stage))
        if seam.measured_peak > STABILITY_THRESHOLD:
            problems.append(
                "seam {}->{} measured peak {:.1f}x exceeds {:.1f}x threshold".format(
                    seam.from_stage, seam.to_stage, seam.measured_peak, STABILITY_THRESHOLD
                )
            )

    lsum = sum(v[2] for v in REASONING_LENGTH_BANDS.values())
    if abs(lsum - 1.0) > 1e-9:
        problems.append("reasoning length bands sum to {:.4f}, not 1.0".format(lsum))

    return problems


def lane_demand_b() -> Dict[str, float]:
    """Absolute tokens each lane consumes: main-run share plus its anneal allocation."""
    integrated = integrated_lane_shares(include_anneal=False)
    anneal = stage_by_name("anneal")
    anneal_mix = normalized_mixture(anneal)
    return {
        lane: integrated[lane] * MAIN_RUN_TOKENS_B + anneal_mix[lane] * anneal.tokens_b
        for lane in LANES
    }


def planned_lane_epochs() -> Dict[str, float]:
    """Epochs each lane runs at under the real plan. S5's tight lanes should show up."""
    demand = lane_demand_b()
    out = {}
    for lane, spec in LANE_SPEC.items():
        out[lane] = (
            demand[lane] / spec.supply_tokens_b if spec.supply_tokens_b > 0 else float("inf")
        )
    return out


def plan_summary() -> Dict[str, object]:
    return {
        "run_tokens_b": TOTAL_TOKENS_B,
        "main_run_b": MAIN_RUN_TOKENS_B,
        "anneal_b": ANNEAL_TOKENS_B,
        "stages": [
            {
                "name": st.name,
                "tokens_b": st.tokens_b,
                "share_of_run": st.tokens_b / TOTAL_TOKENS_B,
                "mixture": st.mixture,
                "bands": list(st.band_range),
                "seq_len_multiplier": st.seq_len_multiplier,
                "is_anneal": st.is_anneal,
            }
            for st in STAGES
        ],
        "seams": [
            {
                "from": s.from_stage,
                "to": s.to_stage,
                "band_tokens_b": s.band_tokens_b,
                "measured_peak": s.measured_peak,
            }
            for s in SEAMS
        ],
        "integrated_lane_shares_main_run": integrated_lane_shares(include_anneal=False),
        "lane_budget_shares": {ln: sp.budget_share for ln, sp in LANE_SPEC.items()},
        "lane_demand_b": lane_demand_b(),
        "planned_lane_epochs": planned_lane_epochs(),
        "mixture_rounding_residue": mixture_rounding_residue(),
        "protected_lanes": [ln for ln, sp in LANE_SPEC.items() if sp.protected],
        "epoch_ceiling": EPOCH_CEILING,
        "source": "ERA5/S5 CURRICULUM.md + DECISIONS.md + BENCHMARK-MAP.md",
    }
