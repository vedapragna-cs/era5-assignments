# What S6 inherits from S5, and where each decision is enforced

S6 does not re-open the data mixture. S5 took twelve decisions and closed them; this system
implements the ones that have a runtime consequence and records the ones that do not. The
point of this document is that a plan which is not enforced anywhere is a document, not a
plan — so every inherited decision below names the file and the check that binds it.

## The plan in code

`ledger/plan.py` is S5's curriculum in real units: **2,910B main run + 90B anneal = 3,000B**.

`verify_plan()` re-derives the published lane budgets from the stage table and fails if they
disagree. It currently passes with zero problems, and getting there corrected a
misunderstanding on my side rather than an error in S5:

> The lane budget shares are shares of the **main run (2,910B)**, not of the 3,000B total.
> Web at 34% of 2,910B is 989.4B, and S5 publishes 989.3B. My first validator integrated
> over 3,000B, reported four failures, and was wrong on all four. The stage rows summing to
> 1.001 / 0.999 are the published table's 0.1% display rounding; `normalized_mixture()`
> handles it and `mixture_rounding_residue()` reports exactly what was absorbed.

With that fixed, all seven lanes reproduce S5's published epoch figures exactly.

## Decision by decision

| # | S5 decision | consequence in S6 | enforced at |
|---|---|---|---|
| 1 | run scale **3T** | `TOTAL_TOKENS_B = 2910 + 90`; the demo `Scale` divides it down, and shares/floors/seam widths are scale-invariant | `plan.py`, `Scale` |
| 2 | anneal **3% (90B)** | its own stage with its own mixture; **not** integrated into the main-run lane shares | `plan.py`, `integrated_lane_shares(include_anneal=False)` |
| 3 | agentic **2% floor**, fund synthesis | protected lane; floor survives integer allocation via the carry; first lane to approach the epoch ceiling | `mixture.py` `_allocate`, `LANE_SPEC[agentic].protected` |
| 4 | reasoning **English-internal** | corpus generates reasoning traces in English; no runtime branch | `corpus.py` `_gen_reasoning` |
| 5 | tail languages **gated at 1.30B** | supply-side; no runtime consequence at demo scale | recorded only |
| 6 | prioritise the NCERT/SCERT negotiation | acquisition decision; no runtime consequence | recorded only |
| 7 | execution environments **not scoped** | agentic traces are static; tool results are masked context, never supervised | `documents.py` `MASKED_ROLES` |
| 8 | romanized Indic **1%** (Dakshina) | `hi/Latn` present in the corpus and visible in the fertility table | `corpus.py`, `03_tokenizer.json` |
| 9 | CommitPackFT **8× upsample** | repetition is bounded by the shared 4-epoch ceiling, not by a per-source multiplier | `mixture.py` epoch clamp |
| 10 | IndicGenBench **14/29** | evaluation-side; follows #5 | recorded only |
| 11 | Indic **17%** | target share; realized 17.68% integrated over the demo run | `08_curriculum.json` |
| 12 | STEM supply **146B** | target share 12%; realized 11.99% | `08_curriculum.json` |

Five of the twelve are acquisition or evaluation decisions with no runtime surface. Saying
so explicitly is better than implying the system enforces more than it does.

## The seam rules

S5's seam rules carry into the trainer:

- **Rule 1 — the embedding table must be trainable at every seam.** `assert_embeddings_trainable()`
  runs before the first step and raises if `tok_emb` is frozen or absent from the optimizer.
  The failure it guards is silent: a frozen table still trains, the loss still falls, and any
  token whose distribution changed at the seam is stuck with its pre-seam representation.
  The damage only appears in evaluation on the new mixture.
- **Seam blending.** Lane weights interpolate across a band centred on the stage boundary
  rather than switching discretely. `_blended_weights` computes the blend; `logs/08_curriculum.json`
  records `in_seam` per step.

At demo scale the seams needed a floor to be visible at all: S5's 2B seam band is 333 demo
tokens, which is less than one step. `RunShape.min_seam_steps=4` is a **declared demo
concession**, not a silent rescaling — without it the blending code runs but never fires,
and a mechanism that never fires is untested.

## The benchmark map's supervision shape

S5's `BENCHMARK-MAP.md` measured agentic supervision at **356/668 = 53% yield** — assistant
turns are supervised *including failed calls*, tool observations are not. That distinction
is implemented in `documents.py`:

```python
SUPERVISED_ROLES = {text, response, assistant, tool_call, thinking}
MASKED_ROLES     = {prompt, system, user, tool_result}
```

and it is what makes the learning ledger's per-role table meaningful. The full run measures
`tool_call` perplexity at 68.7 against `response` at 12.5 — the model is learning to narrate
around the action faster than it is learning to act. That is only visible because the roles
survive tokenization, get run-length-encoded into the shard, and are expanded back out at
materialization; a lane-level average would report one number for both.

## The debt S6 was left, and repaid

`CURRICULUM.md` closes with:

> the mechanism for enforcing a difficulty band at sampling time (classifier? source
> tagging?) is not yet specified and belongs in the S6 data-stream work.

Answered in [`band-gating.md`](band-gating.md). Short version: neither — the band is resolved
into the shard key at build time, so the sampling-time gate is a dictionary lookup and
re-classification can never retroactively change what a past step is reconstructed to have
seen.

## What S6 sends back to S5's successor

`logs/v6_shard_advisory.json`. S5 planned the mixture from supply and demand estimates. S6
measures what the model actually learned per shard and turns that into placement
recommendations for the next run — promote, acquire more, investigate, or reduce. That is
the loop closing: S5 decided the plan from what was available, S6 reports what it was worth.
