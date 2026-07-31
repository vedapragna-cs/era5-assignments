# ERA V5 — agentic synthesis programme

The agentic lane cannot be filled from real data at any scale in the course's stated range. This sizes the
gap as a **production target in samples**, not a token wish. Reproduce with `scripts/agentic.py`.

## The lane is two different things

| shape | samples | tokens | mean | why it matters |
|---|---|---|---|---|
| **Short function-calling** | 458,000 | 267M | **583 tok** | cheap, schema-generable, execution-verifiable |
| **Long trajectories** | 38,400 | 360M | **9,375 tok** | expensive, needs real repos and a live environment |

Per-source spread is the samples-vs-tokens lesson in one table: **SWE-Gym is 2,400 samples but 62,500
tokens each**; **xLAM is 60,000 samples at 417 tokens each**. Sizing this lane on sample counts would
misjudge its training cost by two orders of magnitude.

Total public agentic supply: **627M tokens.**

## The gap, at the 4-epoch ceiling

| | main run (function-calling) | anneal (trajectories) |
|---|---|---|
| demand | 58.2B seen (**31.0B supervised** at 53% yield) | 7.2B seen (**3.8B supervised**) |
| unique needed at 4 epochs | 14.6B | 1.8B |
| have, of this shape | 0.27B | 0.36B |
| **must synthesize** | **14.3B = 53.5× existing** | **1.4B = 4.0× existing** |
| **production target** | **≈ 24.5M new calls** | **≈ 153,600 new trajectories** |

For scale: 153,600 trajectories is **5.9× what SWE-smith produced** (26,000). 24.5M calls is **54× the
entire public function-calling corpus**.

**This is the true price of the 2% protected floor**, and it should be stated as such rather than hidden
inside a percentage. The floor is not a small ask; it is an engineering programme. It is nonetheless the
right call, because the alternative — letting the selector set the lane — measurably drives it to zero
(0 tokens in 59/60 iterations, `SESSION-DIGEST.md` §13.2).

## Why the split makes the programme tractable

The 24.5M-sample number is large but it is the **cheap** shape: short, generated from API schemas, and
verifiable automatically. The expensive shape — long trajectories needing real repositories and execution
— is only **153,600 samples**, because it lives in the anneal where the pool is 90B rather than 2.91T.

Putting the expensive shape in the main run instead would demand ~1.5M trajectories. **The anneal
placement is what reduces the hard half of this programme by an order of magnitude.**

## Method

**Function-calling (main run) — APIGen / ToolACE pattern.**
1. Assemble an API pool from real REST/OpenAPI schemas; ToolACE's contribution is self-evolving synthesis
   over a pool of thousands of tools, which is what produces signature diversity beyond hand-written sets.
2. Sample a schema, generate a user intent, generate the call.
3. **Verify by execution**, then by format, then semantically — the three-stage filter is the reason
   xLAM/APIGen data is tier A/D rather than tier D.
4. Emit with the correct mask: green on the call only; schema and tool return grey.

**Trajectories (anneal) — SWE-smith / SWE-Gym pattern.**
1. Mine real repositories; synthesise task instances by injecting faults with known fixes.
2. Roll out an agent in a live execution environment, recording plan, calls, observations, failures,
   recoveries, final patch.
3. Keep only trajectories whose hidden tests pass — the verifier is the label.
4. **Retain failed-then-recovered trajectories deliberately.** Recovery is the taught behaviour: the
   failed call is supervised, the environment's error text is not.

## The QA gate is not optional

Unverified synthetic agentic data is **worse than no data**. The masking rule exists because applying loss
to tool observations teaches the model to invent tool results instead of calling the tool. Synthetic data
whose "tool returns" were never produced by a real tool has exactly that defect baked in — it is a
hallucinated environment presented as ground truth.

Hard gates, applied before any generated sample enters the corpus:

1. **Executed, not imagined.** Every tool return in a synthetic sample must come from a real invocation.
2. **Verifier-labelled.** Task success determined by tests or goal-state checks, never by a model's opinion.
3. **Masked correctly.** Only assistant-produced tokens carry loss. Audit the mask on a sample, since a
   masking bug here is silent and poisons the exact capability the lane exists to build.
4. **Deduplicated against benchmarks.** SWE-bench Verified draws from real GitHub issues; synthetic tasks
   mined from the same repositories can leak. Contamination check against all claimed benchmarks.

## Risks

- **Diversity collapse.** 24.5M samples from a self-evolving generator tend toward a narrow slice of the
  API space. Track distinct signatures and argument-shape entropy, not just sample count.
- **Verification cost.** Executing 24.5M calls is itself infrastructure. If the execution budget binds,
  the honest response is to cut the lane, not to ship unverified samples.
- **Benchmarks we are not claiming.** Terminal-Bench, WebArena and OSWorld need live shells, hosted sites
  and desktops. This programme does not deliver them; see `BENCHMARK-MAP.md`.
- **Scale-dependence of the anneal result.** If T3 refutes the reserve hypothesis, the anneal trajectory
  target collapses and the whole programme should be re-sized before, not after, it is built.

## Decision this forces

The 2% main-run floor and the 24.5M-call programme stand or fall together. Three options, in order of
preference:

1. **Fund it.** Accept the synthesis programme as a named deliverable with its own verification infra.
2. **Re-shape the lane.** Fill more of the main-run lane with trajectory data repeated to the 4-epoch
   ceiling, accepting far fewer unique samples and a narrower distribution.
3. **Argue the floor down.** Contradicts the session's protected-floor design and the measured evidence
   that an unprotected lane goes to zero. Not recommended without T3 results.
