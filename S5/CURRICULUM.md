# ERA V5 — curriculum staging v0

The budget in [BUDGET-PROPOSAL.md](BUDGET-PROPOSAL.md) says *how much*. This says *when*.
Reproduce with `scripts/curriculum.py`.

## The constraint the course widget does not enforce

Per-stage mixtures must **integrate to the overall lane budget**. A set of stage anchors that each sum to
100% can still spend a lane far above or below its allocation once weighted by stage length — the course
curriculum widget interpolates between anchors without ever checking this.

We solve both constraints simultaneously by iterative proportional fitting:

1. every stage's mixture sums to 100%;
2. every lane's token-weighted mean across stages equals its budget share.

## Stage schedule — main run 2,910B, then a 90B anneal

| lane | S1 Seed (233B) | S2 General (931B) | S3 Reasoning (1,048B) | S4 Long-ctx (698B) | Anneal (90B) | budget | ∫ check |
|---|---|---|---|---|---|---|---|
| General web | 56.1% | 45.3% | 28.1% | 20.4% | 8% | 34% | ✓ |
| Code | 14.6% | 19.2% | 26.8% | 29.3% | 20% | 24% | ✓ |
| Reasoning | 2.0% | 4.8% | 12.5% | 11.7% | 18% | 9% | ✓ |
| Long-context | 0.2% | 0.4% | 0.9% | 6.3% | 8% | 2% | ✓ |
| **Indic** | **17.2%** | **16.9%** | **16.9%** | **17.2%** | **28%** | **17%** | ✓ |
| STEM | 9.0% | 11.8% | 12.8% | 12.0% | 10% | 12% | ✓ |
| Agentic | 1.0% | 1.5% | 2.0% | 3.0% | 8% | 2% | ✓ |

Shape rationale:

- **Web falls 56% → 20%**, matching the V4 trajectory (72 → 18) and Llama-3-style staging.
- **Reasoning enters late** (2% → 12.5%). The session is explicit that long traces poured into early
  pretraining do not produce a reasoning model; the base must exist first.
- **Long-context is essentially absent until S4** (0.2% → 6.3%). Consistent with arXiv:2402.10171: the
  capability is mostly acquired during general pretraining, so extension is a late, cheap operation.
- **Indic is deliberately flat at ~17% throughout.** It is the one lane that does not ramp. A protected
  capability that ramps is still a residue of whatever budget is left over; a flat lane is a commitment.
- **Agentic stays at its floor and concentrates in the anneal** (1% → 3%, then 8%). §3 of the budget shows
  this is not a preference but the only arithmetically feasible placement.

## Absolute spend, and where repetition bites

| lane | S1 | S2 | S3 | S4 | anneal | **total** | supply | epochs |
|---|---|---|---|---|---|---|---|---|
| web | 130.6 | 421.6 | 294.5 | 142.7 | 7.2 | 996.6 | 4,691 | 0.21 |
| code | 34.0 | 178.9 | 281.2 | 204.4 | 18.0 | 716.4 | 1,103 | 0.65 |
| reason | 4.5 | 44.8 | 130.7 | 81.9 | 16.2 | 278.1 | 85.1 | **3.27** |
| longctx | 0.4 | 4.2 | 9.3 | 44.3 | 7.2 | 65.4 | 100 | 0.65 |
| indic | 40.0 | 157.7 | 177.0 | 120.1 | 25.2 | 519.9 | **157.2** | **3.31** |
| stem | 21.0 | 110.2 | 134.1 | 84.0 | 9.0 | 358.2 | 146 | 2.45 |
| agentic | 2.3 | 13.9 | 20.8 | 21.2 | 7.2 | 65.4 | 0.63 | **104** |

All lanes sit under the 4-epoch ceiling (arXiv:2305.16264) except agentic, whose gap is the synthesis
programme sized in the budget proposal. **Indic at 3.31 and reasoning at 3.27 are the two tight lanes**
— if repetition evidence turns against either, they are the first shares to cut.

The Indic supply figure is 157.2B, not the 275.9B in the v0 inventory: the per-language audit (E2) found
a 12.8B English split inside "verified Indic" and a synthetic tier that is one corpus present twice, once
transliterated. See [INDIC-ALLOCATION.md](INDIC-ALLOCATION.md). This also means **the flat ~17% Indic
schedule is now near its repetition budget across every stage**, so the flat-lane commitment costs more
than it appeared to in v0 — it is still the right call, but it is no longer free.

## Seam stability — sized from measured data, not intuition

From the measured instability sweep (`SESSION-DIGEST.md` §13.3):

> Frozen embeddings **151×** spike vs **8.0×** unfrozen, at identical sharpness. With embeddings frozen,
> **no warmup band ≤5B tokens** brings the transition under the 3× stability threshold.

**Rule 1 — embeddings must remain trainable across every seam.** This dominates band width by ~19× and is
the actual lesson of the V4 incident, which is usually told as a mixture-shift story.
**Rule 2 — blend every seam over ≥2B tokens** (measured peak 2.8×, under the 3× threshold).

| seam | at | band | measured peak |
|---|---|---|---|
| S1 → S2 | ~233B | 2B | 2.8× |
| S2 → S3 | ~1,164B | 2B | 2.8× |
| S3 → S4 | ~2,212B | 2B | 2.8× |
| S4 → Anneal | ~2,910B | **4B** | 2.0× |

The final seam gets a double-width band because it is the largest distribution shift in the run
(web 20% → 8%, Indic 17% → 28%) and lands where a diverged run is most expensive to lose.

## Difficulty bands — a concrete example at each level

Each stage draws from a band range rather than a single band; the ladder advances with the stage.

| band | level | concrete example of the data | stages |
|---|---|---|---|
| **B0** | Nursery | simple children's narrative: *"The cat sat on the mat. It was warm."* — TinyStories-style synthetic | S1 |
| **B1** | Grade-school | Simple-Wikipedia articles; GSM8K-style word problems (*"Natalia sold clips to 48 friends…"*) | S1–S2 |
| **B2** | High-school | MMLU high-school subsets; introductory algebra and mechanics; well-commented beginner code | S2–S3 |
| **B3** | Undergraduate | undergraduate textbook chapters; LeetCode easy/medium with worked solutions; peS2o survey papers | S3–S4 |
| **B4** | Graduate | arXiv papers; AIME competition problems; GPQA-style graduate science; multi-file repository code | S3–S4 |
| **B5** | Research / PhD | proof-pile-2 formal proofs; FrontierMath-level problems; SWE-bench-style real repo patches with hidden tests | S4 + **anneal** |

B5 is deliberately concentrated in the anneal. It is the "best data saved for last" decision, and per the
session it only works if the reserve survives the main run.

## Reasoning-length bands — a concrete example at each level

Reasoning is **not one slot**; it is a distribution of trace lengths. Anchors are the measured traces from
the session widget on *"how many integers 1..1000 are divisible by 3 or 5?"* (answer 467).

| band | token budget | measured anchor | example of the trace | share of reasoning lane |
|---|---|---|---|---|
| **L0** low | ≤ 64 | 37 tok | *"Inclusion–exclusion: ⌊1000/3⌋+⌊1000/5⌋−⌊1000/15⌋ = 333+200−66."* | 30% |
| **L1** medium | 64–256 | 74 tok | counts each term separately, states why 15 is the overlap, then sums | 35% |
| **L2** high | 256–1,024 | 161 tok | derives each count with justification, then **verifies by complement** (2/3 × 4/5) | 25% |
| **L3** ultra | 1,024–8,192 | 346 tok | restates goal, plans, derives, checks by blocks of 15, re-checks endpoints, second density check | **10%** |

**Why L3 is capped at 10%.** Measured: HIGH → ULTRA costs **2.15× the tokens for +4 accuracy points**
(161→346 tok, 91%→95%) — the widget itself labels ULTRA *"wasted effort"*. Over-representing long traces
teaches the model that more thinking is always better, which is exactly the failure mode raised in the
session: because the model is rewarded for correctness, it will always spend maximum effort unless the
training distribution teaches it otherwise. The dial has to be *taught*, and teaching it requires the
short bands to dominate.

Each band must span mathematics, code and general problem solving, or the behaviour binds to one domain.

## Open

- Stage weights (8 / 32 / 36 / 24 of the main run) are a first pass and are not yet defended by experiment.
- Band-to-stage assignment above is a schedule, not a hard gate; the mechanism for enforcing a difficulty
  band at sampling time (classifier? source tagging?) is not yet specified and belongs in the S6 data-stream work.
