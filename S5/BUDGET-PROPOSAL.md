# ERA V5 — mixture budget proposal v0 (for review, not final)

Design point: **3T total tokens** (session stated 2.4–4T). Split **97% main run (2.91T) / 3% anneal (90B)**.
Anneal size follows OLMo 2, which sampled its 843B Dolmino pool into 50B / 100B / 300B anneal mixes.

Every epoch figure is bounded by **arXiv:2305.16264**: up to **4 epochs** of repeated data costs
negligible loss versus unique data; beyond that the value of added compute decays. So `epochs ≤ 4` is the
feasibility test, not a preference. Supply figures are the auditable per-row totals from the S5 inventory.

---

## 1. Main pretraining run — 2.91T tokens

| lane | share | demand | real supply | epochs | verdict |
|---|---|---|---|---|---|
| Code | 24% | 698B | 1,103B | 0.63 | covered |
| **Agentic** | **2%** | **58B** | **0.63B** | **93** | **synthesis programme — see §3** |
| Reasoning traces | 9% | 262B | 85B | 3.1 | repeat, under ceiling |
| Long-context | 2% | 58B | 100B | 0.58 | covered |
| Indic | 17% | 495B | **157B** | **3.2** | repeat, tight — see below |
| STEM / math | 12% | 349B | 146B | 2.4 | repeat, under ceiling — 250B claim resolved, E12 |
| General web | 34% | 989B | 4,691B | 0.21 | covered |

Changes from the course composer default (`24/2/6/6/16/12/34`):

- **Long-context 6% → 2%**, freeing **116B tokens**. arXiv:2402.10171 finds **0.5–5B tokens** of continued
  pretraining is enough to retrieve anywhere in a 128K context, because the capability is *"mostly already
  acquired through large-scale pretraining."* Even at 2% we allocate **58B — still 12× the upper bound.**
  This is the single largest defensible saving in the budget.
- **Reasoning 6% → 9%** (+87B). Llama 3 spends 25% on math+reasoning content; ours is 9% traces + 12% STEM
  = 21%, which lands in the same band. Stated this way so a reviewer does not read "9%" in isolation.
- **Indic 16% → 17%** (+29B). Justification in §2; the tier split matters far more than the headline.
- Composer default also fails at this scale on nothing *except* agentic — its long-context and STEM
  allocations are merely wasteful rather than infeasible.

## 2. Indic tier split — the headline number is not the claim

> **Corrected 2026-07-31 by the per-language audit (E2, `INDIC-ALLOCATION.md`).** Two rows in the
> original inventory did not survive being read language by language: **19.8% of "verified Indic" is
> English** (Sangraha's verified config has an `eng` split of 12.76B), and the **synthetic tier is one
> corpus transliterated, not two** — 14 languages × 2 scripts with row-for-row identical native and
> Latin splits, of which only the native half is scored by MILU/IndicGenBench. Usable Indic supply is
> **157B, not 276B**, and lane epochs are **3.2, not 1.8**. Still under the ceiling; no longer roomy.

Indic = 17% of 2.91T = **495B**. Supply by tier, corrected: A verified 51.5B · B unverified **41.9B** ·
C translated/parallel **~9B** · D synthetic 81.4B (native script).

| tier | proposed | demand | supply | epochs |
|---|---|---|---|---|
| A verified native | 28% | 138B | **51.5B** | **2.7** |
| B unverified crawl | 30% | 148B | **41.9B** | **3.5** |
| C translated / parallel | **2%** | 9.9B | **~9B** | **1.1** |
| D synthetic | 40% | 198B | **81.4B** | **2.4** |

**And the tier split is an average, not a specification.** Per-language verified share runs from
**5% (Assamese) to 46% (Bengali)**, so A28/B30/C2/D40 is achievable in aggregate and unachievable
language by language. Consequence for T2: evaluate synthetic-Indic quality **per language**, because
Assamese (95% synthetic) and Bengali (46% verified) are testing different hypotheses.

Per-language allocation within the lane — the mechanism, the funded/unfunded benchmark split, and why
the tail cannot be fixed by budget — is in **[INDIC-ALLOCATION.md](INDIC-ALLOCATION.md)**.

**The composer's default tier split is arithmetically impossible.** A40/B25/**C20**/D15 demands 99B of
translated/parallel text against **~9B** of real supply — **11 epochs**. Any plan that inherits the C20
default has already failed the "wishful accounting" test the rubric is built around.

Tier C was revised upward twice by audit and the finding survived both. Original: BPCC 3B + Samanantar 2B
= 5B (19.8 epochs). Corrected (E3/E6): **Samanantar is two directories inside BPCC**, 35.3GB of its
109.3GB, so listing them separately counted a third of the corpus twice — while BPCC itself is ~230M
bitext pairs, ~6.8B Indic-side, not 3B. Added (E5): **134,028 document-aligned Supreme Court judgments in
18 languages**, ~1.7–2.6B tokens, human-vetted and 100% parallel — the highest-quality tier-C data
available, against BPCC where only 2.2M of 230M pairs are human-translated.

Open definitional question for the reviewer: if tier C means *machine-translated content we generate*
rather than existing parallel corpora, supply is unbounded but the quality tier is unchanged. We treat C as
existing parallel corpora only, and route generated translation into D.

Synthetic at 40% of the Indic slice is **more conservative than raw supply** (synthetic is 59% of all
available Indic). It remains the most attackable number in the plan, which is why it is what the proxy
experiment in §5 tests.

> **Lane disjointness checked (E7, `scripts/multitag.py`).** S3 assumes documents carry multiple tags,
> which would make these tables wrong. Modelled: mean tags per token **m = 1.020**, so shares sum to
> 102% rather than 100%, and the overlap sits in code / long-context / agentic — lanes already at 0.63
> epochs or below. Indic and reasoning intersect nothing, because the web, STEM and reasoning corpora are
> all English. **The disjoint model is correct to within 2% and is retained.** Multi-tagging does,
> however, add an unfunded cross-lane requirement — see E7.

## 3. Agentic — the binding constraint, and why the anneal is the only feasible home

At the 4-epoch ceiling:

| placement | share | demand | unique needed | have | must synthesize |
|---|---|---|---|---|---|
| Main run | 2% | 58B | 14.6B | 0.63B | **13.9B = 22× all public agentic data** |
| **Anneal** | **8%** | **7.2B** | **1.8B** | **0.63B** | **1.2B = 2× public supply** |

This is the proposal's central finding. **The same 8% share is a fantasy in the main run and tractable in
the anneal**, purely because the anneal pool is 90B rather than 2.91T. The session's instruction to protect
agentic data for the anneal is therefore not pedagogy — at 3T it is the only arithmetically feasible
placement.

Consequence: **split the agentic lane in two**, because the inventory contains two different things:

- **Short function-calling** — Glaive 50M, ToolBench 80M, ToolACE 60M, Nexus 30M, xLAM 25M, Hermes 22M
  = **267M tokens**, high sample count, short. Cheap to synthesize from API schemas and verifiable by
  execution (the APIGen/ToolACE method). **This fills the 2% main-run lane and teaches call format.**
- **Long trajectories** — SWE-Gym 150M, SWE-smith 120M, OpenHands 90M = **360M tokens**, low sample count,
  very long. Expensive: needs real repos and execution environments. **This is the anneal reserve** and must
  not be spent early.

This split is the samples-vs-tokens lesson made operational, and it makes the synthesis commitment concrete
and sized rather than aspirational.

**Costing correction:** the S5 trajectory widget measures **356 supervised / 668 seen tokens = 53%**. Roughly
half of every agentic token carries no loss, so 58B of main-run agentic buys ~31B supervised tokens. Supply
must be sized in *seen* tokens; capability should be argued in *supervised* ones.

## 4. Anneal — 90B tokens, sized as its own pool

| lane | share | demand | supply | epochs |
|---|---|---|---|---|
| Indic | 28% | 25.2B | 276B | 0.09 |
| Reasoning | 18% | 16.2B | 85B | 0.19 |
| Code | 20% | 18.0B | 1,103B | 0.02 |
| Agentic | 8% | 7.2B | 0.63B | 11.5 → synthesize 1.2B |
| Long-context | 8% | 7.2B | 100B | 0.07 |
| STEM | 10% | 9.0B | 146B | 0.06 |
| General web | 8% | 7.2B | 4,691B | 0.00 |

Every lane except agentic is trivially covered, because the pool is small. **This is why the course
widget's anneal column is misleading** — it applies anneal shares to the full run size and reports demand
~30× too high, making the one affordable phase look impossible.

Reserve held back from the main run and spent only here: **Tier-A verified Indic**, **long agentic
trajectories**, and **long reasoning traces**. Per the session, if the selector consumes these early there
is nothing left to cool down on.

## 5. The proxy experiment — what would falsify this

**Protocol (Llama 3 annealing-as-evaluation / OLMo 2 microannealing), not a from-scratch 1B run.**
Meta state this is *more efficient than running scaling-law experiments for every small dataset*.

1. Train a 1B base to ~50% of its token budget on the §1 mixture.
2. For each candidate change, anneal LR linearly to 0 over a fixed small budget with
   **30% weight on the candidate / 70% on the default mix**.
3. Randomise data order and repeat where the signal is marginal (OLMo 2 averages 3 runs).

Three pre-registered tests, each with the metric that would refute the corresponding number:

| # | hypothesis | metric | refuted if |
|---|---|---|---|
| T1 | Long-context at 2% loses nothing vs 6% | long-eval retrieval accuracy at target context | 2% arm degrades beyond noise → restore budget, revisit §1 |
| T2 | 40% synthetic Indic is not worse than verified-only | MILU + IndicGenBench (chrF / exact-match) | synthetic-heavy arm underperforms verified-only → cut D share, raise A, accept a lower Indic total |
| T3 | Anneal-reserved agentic beats spending it early | SWE-bench Verified % resolved, BFCL accuracy | early-spend arm matches reserve arm → the reserve earns nothing and the floor is the whole story |

**Known limitation, stated before a reviewer raises it:** Llama 3 report that anneal gains were
**negligible at 405B** — *"our flagship model has strong in-context learning and reasoning capabilities and
does not require specific in-domain training samples."* Anneal-measured value shrinks as base capability
rises. Our target is far smaller so the signal should hold, but T1–T3 measure *anneal-stage* value and
their transferability to a much stronger base is unproven.

## 6. Scale sensitivity

Demands scale linearly, so the conclusions are stable across the stated 2.4–4T range. Agentic unique-token
requirement at the 4-epoch ceiling: **11.6B at 2.4T · 14.6B at 3T · 19.4B at 4T** — 18× to 31× all public
agentic data. No run scale in the course's stated range makes the main-run agentic lane fillable from real
data.

## 7. Open decisions needing a call

1. **Anneal fraction 3%.** OLMo 2 supports 50–300B for 7B–32B models; 90B is mid-range. Raising it to 5%
   (150B) makes every scarce lane more affordable but shortens the main run.
2. **Main-run agentic at the 2% floor** commits us to synthesizing ~13.9B tokens of function-calling data.
   That is a real production programme. The alternative is arguing the floor down, which contradicts the
   session's protected-floor design.
3. **Indic 17%** is 2× Llama 3's multilingual share (8%) and ≈ Nemotron-4's (15%), but those cover all
   languages with vast supply. Defensible on differentiator grounds; not defensible on precedent.
4. **STEM supply discrepancy.** Composer states 250B; auditable inventory rows total 146B
   (peS2o 42 + proof-pile-2 55 + D4 STEM 49). Used 146B throughout. Worth confirming which is intended.

---

## 8. The budget as funded share / trigger / fallback

Flat percentages hide which parts of a plan rest on data we hold and which rest on work not yet done.
Every share above is one of three things, and the plan should say which. Reproduce with
`scripts/conditional.py`.

### 8.1 What each share actually rests on

| lane | asked | max supportable at 4 epochs | status |
|---|---|---|---|
| General web | 34% | 645% | funded |
| Code | 24% | 152% | funded |
| Indic | 17% | 21.6% | funded |
| STEM | 12% | 20.1% | funded |
| Reasoning | 9% | 11.7% | funded |
| Long-context | 2% | 13.7% | funded |
| **Agentic** | **2%** | **0.087%** | **programme — 23× short** |

**The honest headline is not "100% allocated". It is: 98% of the main run is funded from supply we hold
today, and 2% rests entirely on a named synthesis programme.** One lane carries all the delivery risk in
the budget, and it is the lane the session's protected floor exists to defend.

### 8.2 The dependency nobody asks about

**78B of the reasoning lane's 85B is our own V4 AON corpus.** A public-only rebuild supports **1.0%, not
9%**. Across all lanes, losing the V4 corpora would open an **8.0% hole** in the run.

| lane | asked | with V4 | public only | fallback share |
|---|---|---|---|---|
| Reasoning | 9% | 85.1B | **7.1B** | **1.0%** |
| STEM | 12% | 146.0B | 97.0B | 12.0% |
| Code | 24% | 1,103.0B | 904.0B | 24.0% |
| Indic | 17% | 157.2B | 157.2B | 17.0% |

"We hold AON" is load-bearing and completely invisible in a flat percentage table. The court-judgment
ingest (~8B, CC BY 4.0) is the only thing in the audit that materially moves the public-rebuild floor.

### 8.3 Triggers and fallbacks

| lane | if | then | else |
|---|---|---|---|
| agentic | synthesis delivers 24.5M verified calls | holds 2% | **falls to 0.09%** |
| agentic | anneal delivers 153,600 trajectories | holds 8% of anneal | anneal agentic → 2.8% |
| indic | a gated language crosses **1.30B** | re-enters allocation automatically | stays gated; IndicGenBench stays 14/29 |
| stem | NCERT/SCERT/NPTEL/IGNOU agreement | Indic-STEM 3.4% → **15%** | lowered to 3.4%, capability conceded |
| stem | course confirms 250B | epochs 2.39 → 1.40 | stays at 146B |
| code | more diff-following data than CommitPackFT's 4B | upsample drops below 8× | 8× holds, 2 epochs |
| indic | **T2** — tier D holds on India-specific MILU subjects | D stays 40% | D cut, A raised, Indic total < 17% |
| longctx | **T1** — 2% loses nothing vs 6% | 2% holds, 116B stays reallocated | restore 6%, take 116B back |
| reason | court judgments ingested | public-rebuild floor doubles | public rebuild stays 7.1B |
| all | run scale moves to 4T | — | **Indic must fall to 15.7%** or breach the ceiling |

No trigger requires a redesign. Each moves one share, and the allocation machinery (IPF for stages,
content-weighted temperature for languages) is defined for any input.

### 8.4 Web is the adjustment variable, and should be named as one

Web is asked for 34% against 4,691B — **0.21 epochs**, and it could support 645% of the run alone. Every
fallback above is absorbed here and every trigger that fires draws from here.

**Web is not 34% because 34% is optimal. It is 34% because it is the only lane with the slack to absorb
change.** Saying so is more useful than defending 34% as a tuned value.

**The floor matters.** The session warns that cutting general web too far produces a model whose code
compiles and does not work — the capability with no clean benchmark. **We set a hard floor at 25%**,
below which fallbacks must come from somewhere else or the run is re-scoped.
