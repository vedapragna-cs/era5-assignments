# What V5 still misses

> "you will catch up the points that I have missed … you yourself will see what does it take to train
> a model at the class of OpenAI and others."

V4 taught the course what it failed to capture. Those lessons are visibly folded into V5. This note is
the next iteration of that loop: **what is still missing from the V5 plan**, found by implementing it
end to end, ordered by what it costs to discover late.

The organizing principle is the same one behind the session's own regret about perplexity —
*"literally maybe 10 lines of code and I could have done it"*. Some things are **free before the run
and impossible after it**. Those come first.

---

## Part 1 — the V4 → V5 delta, as I read it

Stated first so the gaps below are calibrated against what has already been fixed, not repeated back.

| V4 missed | V5 includes |
|---|---|
| per-token perplexity never captured — *"I hate myself for not doing it"* | the learning ledger; token-level loss and ppl are a stated deliverable |
| OPUS decisions not logged, so rejections were unrecoverable | accepted / rejected / deferred / protected all recorded with reasons |
| data handling was manual — *"everything was manual, this time we're going to be taking a shard, run on it and update that server"* | the data ledger as a first-class service |
| embeddings frozen across a mixture seam → measured **151× gradient spike** vs 8.0× unfrozen | S5 rule 1: embeddings stay trainable across every seam |
| tokenizer not ready early enough to build an Indic curriculum | tokenizer frozen in S2, before mixture design |
| no way to reconstruct which data trained a checkpoint | consumption ledger + checkpoint→offset binding |

The pattern in every row is the same: **a cheap thing not written down during the run, wanted badly
afterwards.** Part 2 is the list of things that still have that shape.

---

## Part 2 — Tier A: unrecoverable. Not captured during the run means gone.

These cost almost nothing to add now and cannot be recovered later without re-running the model at the
same training state. This is exactly the V4 perplexity situation, one iteration on.

### A1. Perplexity without exposure is a mixture readout, not a difficulty map

The V6 payload is the plan's centrepiece: save per-token perplexity, use it to decide what V6 should
collect, protect, repeat, defer. **Measured on my run, that inference does not hold as specified.**

Indic received **43% of all traced tokens** — it is protected *and* has the largest pool — and came
out with the **lowest** perplexity: 166 against web's 398. That inverts the course's own widget, which
measures Indic hardest by a wide margin (125.9 vs 7.1 early).

Indic is not easy. It is simply the best-trained lane. **A lane ranking that does not control for
exposure ranks the mixture, not the data** — and the mixture is what you were going to use the ranking
to choose. The loop is circular unless exposure is divided out.

**Cost of missing it:** the V6 corpus decisions are derived from these numbers. If the correction
cannot be applied after the fact, the entire payload — the thing this session exists for — degrades to
a restatement of the schedule.

**Fix, before the run:** write `tokens_consumed_when_seen` and `repeated_pass_number` on every token
observation, and compare lanes at **matched tokens-seen**, not matched wall position. If those two
fields are not written during the run, the correction is impossible afterwards.

**The established correction is stronger than dividing out exposure.** DoReMi's formulation is
**excess loss** — loss relative to a reference model — and the data-mixing literature is explicit that
raw per-domain loss fails in both directions: *"using only the proxy model's loss (prefer hardest
domains) or only the negative reference loss (prefer easiest domains) both underperform the full
excess loss formulation."* Recent work adds the same warning about sample-level loss as a quality
proxy: *"high loss can indicate noise, difficult domain content, novel valuable information, or
short-text statistical instability — phenomena that are indistinguishable from a single scalar."*

So the V6 payload should carry **loss and a reference loss on the same tokens**, not loss alone. That
is one extra forward pass over a small reference model, and it is the difference between a difficulty
map and a scalar that cannot be interpreted. Add it now; it cannot be added later.

### A2. OPUS scores, not just OPUS decisions

The session describes the diagnostic itself: watch the *average gradient of selected batches* fall
over time, and when the top 25% starts scoring like the old bottom 25%, **the shard's quality is
exhausted** — *"it's telling you the quality is bad … it has to select something."*

That diagnostic needs the **score distribution per phase**, not the accept/reject label. A decision log
tells you what happened; only the scores tell you the pool is degrading, and they tell you *while
there is still time to act*.

**Cost of missing it:** you keep spending on an exhausted lane until a benchmark eventually notices,
which is tens of billions of tokens later.

**Fix:** record `opus_score`, `rank_in_candidate_pool` and `candidate_pool_size` per candidate. Cheap.
I record all three; the plan asks for the decision only.

### A3. Supervision yield is a corpus statistic being used as a constant

S5 measures **356/668 = 53%** on one reference trajectory and then sizes agentic supply in *seen*
tokens while arguing capability in *supervised* ones.

My packer applies the identical mask rule and measures **68.9%**, because my trajectories are shorter
with proportionally more assistant turns. The mask policy is the same; the yield is a property of the
data.

**Cost of missing it:** the agentic lane is already 22× short. Sizing it on a borrowed 53% when the
corpus yields something else puts the supervised-token budget out by tens of percent in the lane with
the least slack.

**Fix:** yield is a **manifest field per shard**, measured, not a plan constant.

### A4. The tokenizer's training document set

S2 freezes the tokenizer. S3/S4 define the eval firewall. Nothing connects them — and if the tokenizer
was trained on text that includes benchmark data, benchmark vocabulary is in the merge table. That is
a contamination channel which **survives every downstream document filter**, because you are no longer
filtering documents; you are shipping a tokenizer that has already seen the test set.

**This is not speculative — the signal is recoverable by an outsider.** *Data Mixture Inference: What
do BPE Tokenizers Reveal about their Training Data?* (NeurIPS 2024) shows that BPE merge **order** is
*"extremely sensitive to the proportion of different data categories"* and solves for a released
tokenizer's training mixture from the merge list alone. Their method surfaced a case where a tokenizer
implied 39.4% books *"even though books were not explicitly included in the training data."*

So a tokenizer trained on benchmark text does not merely risk contamination — it **carries a
recoverable fingerprint of that contamination**, readable by anyone with the tokenizer file. For a
model whose headline claim is Indic capability measured on MILU and IndicGenBench, that is a
reputational exposure as much as a correctness one.

**Cost of missing it:** unfixable without retraining the tokenizer, which invalidates every shard.
Found after the run, it invalidates the benchmark claims — and is independently checkable by a third
party.

**Fix:** train only on firewall-cleared text and record the document set in the tokenizer manifest, so
the exclusion is auditable rather than asserted.

---

## Part 3 — Tier B: silently corrupts. The run completes and you cannot tell what happened.

### B1. "Resume from where you left off" has three readings and they give different data

The checkpoint is at step N; the crash is at N+k. Batches N…N+k were committed and consumed by a model
state **that no longer exists**. Skip them? Re-serve them from the ledger? Re-plan them?

I shipped "re-serve from the ledger" because it seemed obviously right. **Measured: the stream diverged
from an uninterrupted run at global sample position 37 of 84** — no error, no warning, just different
data from the crash onward. Re-serving does not advance the planner's cursors, so the next
newly-planned step draws from the wrong pool position.

Only re-planning from the checkpoint is correct, with the ledger as the independent check.

**Cost of missing it:** every crash silently perturbs the stream. In a 45–60 day run there will be
many, and none of them are visible in the loss curve.

### B2. Nothing in the plan survives a change of machine shape

The plan assumes the same machine on resume. Over 45–60 days it will not be — you crash on 8 ranks and
what is free is 4, or 16. If loader position is per rank ("rank *r* consumed *n* samples"), resuming
at a different world size **duplicates or drops samples**, silently.

**This is a known production concern, and the state of the art has constraints worth knowing.**
Mosaic StreamingDataset provides *"deterministic sample ordering independent of the real number of
devices"*, which is exactly the property we want — but its resumption guarantee holds under stated
conditions: **the global batch size stays constant, the seed is unchanged, and the number of logical
nodes is maintained.** Elastic-training guidance is blunter still: *"when your job resumes with a
different world size, previously processed samples may be repeated or skipped, affecting training
quality."*

**Fix:** position is one global integer over a canonical ordering; the rank partition is *derived* at
load (`rank = global_index % world_size`), never persisted. Free to specify now, very expensive to
retrofit. This removes the logical-node constraint, and the global-batch constraint becomes explicit
rather than implicit — a change of global batch mid-run is a change of experiment and should be
recorded as a branch, not absorbed silently.

**And the test for it must be able to fail.** My first version re-partitioned the delivered list and
compared it against itself — true by construction. It reported PASS on a system that was broken.

### B3. A protected floor is specified in units that cannot be delivered

Floors are Indic 12%, agentic 2%, reasoning 6%, delivered by an always-on lane injecting a fixed
fraction of **every batch**. A per-batch floor has the batch's granularity: at global batch 8 the
smallest expressible share is 12.5%, so a **2% floor becomes 12.5%** and three protected lanes lock
37.5% of every batch. Measured on my first build: agentic **18.9% against a 2% floor**.

S5 already measured this shape — the course's always-on lane, *stated 8%, delivered ~15%* — and
recorded it as a widget quirk. It is not a quirk; it is what a per-batch floor does at any realistic
batch size.

**Fix:** enforce against *cumulative delivery*, not per batch. Mine delivers 2.65% for a 2% floor, and
the overshoot stops scaling with the number of protected lanes.

### B4. The anneal reserve has an intent but no mechanism

S5 holds back Tier-A Indic (25.2B), long agentic trajectories (360M) and long reasoning traces
(16.2B), with the stated reason *"the selector would consume it first."*

Nothing described stops it. OPUS selects greedily; if the reserved material is in the pool it will be
taken. "Reserved" is currently a property of a spreadsheet row, not something the scheduler can act on.

**Cost of missing it:** you reach the anneal and the reserve is gone — and S5's T3 experiment (does
reserving beat spending early?) becomes unrunnable, because there is no arm to compare.

**Fix:** undrawable at the pool level before the anneal stage, with the deferral recorded as a decision
(`reserved_for_anneal`) rather than as an absence, so the reserve's survival is provable.

---

## Part 4 — Tier C: foreseeable compute waste

### C1. Long-form documents get destroyed by early stages, and nothing prevents it

This one is stated in the session and then not carried into the plan:

> "you cannot train on a big code at the beginning of the training when your context length is only
> 4096 … **you need to preserve that shard for future** because in future you might train on a longer
> sequence."

Stage S4 needs 32k-token documents. They are the scarcest shape there is. But the default packing
policy for general text is concatenate-and-chop, and stages S1–S3 run at 4k/8k — so a 40k-token
document entering stage S1 is chopped into ten 4k spans and **consumed**. The document still exists;
its value as long-context training data does not, because S5 budgets long-context at 0.65 epochs
assuming it is fresh.

**This is the anneal-reserve problem again, for a different axis, and it is not in the plan at all.**
The anneal reserve protects *quality*. Nothing protects *length*.

**Cost of missing it:** you arrive at S4 — 2.2T tokens in, roughly 40 days — and discover the
long-context lane has already been spent at short context. There is no recovery that does not involve
re-collecting data.

**Fix:** a length reserve, enforced the same way as the anneal reserve. Any document above the *next*
stage's sequence length is undrawable by the current stage. This is a scheduler rule, and it costs
nothing except stating it.

**Implemented and measured here.** `LanePool.draw` blocks a sample when
`stage_window < doc_length <= widest_window` — reserving only where reserving buys something, since a
document longer than every window will be split wherever it is used. Measured on our corpus: 22 of 37
documents are held back from S1/S2, and **31.7% of draws could not honour the reserve** because the
lane had no short-enough document left. The scheduler relaxes rather than deadlocks and records the
relaxation. That rate is the measurement: in production it should be near zero, and if it is not, the
early stages are eating the long-context lane.

**Published anchor for the ratio.** Long-context extension work converges on roughly a **3:1
short-to-long ratio** during the extension phase to preserve short-context capability while extending
(4K → 32K → 256K staged curricula). S5's long-context lane runs 0.2% / 0.4% / 0.9% / 6.3% / 8% by
stage, so the 3:1 rule applies *within* S4 rather than across the run — worth stating explicitly,
because the natural misreading is to apply it globally and blow up the long-context budget.

### C2. The stage × band supply check does not exist

S5 maps difficulty bands B0–B5 onto stages (B0/B1 seed, B1/B2 general, B2–B4 reasoning, B3/B4
long-context, B5 anneal). The supply audit is **per lane**. The stage gate operates on **lane × band**.
Those are different tables, and only one of them has been checked.

I hit this immediately: the stage gate correctly rejected almost every candidate because the corpus was
mostly B2 while stage 1 admits B0/B1.

**Cost of missing it:** a stage starves mid-run, and the scheduler either stalls or silently relaxes
the curriculum. B5 is the most exposed — it is concentrated in the 90B anneal, so a shortfall there is
invisible in any whole-run supply figure.

**Fix:** feasibility per `(stage, lane, band)` — a ~20-cell table instead of a 7-row one.

### C3. The rejection rate is treated as a free knob; it is jointly constrained with the budget

*"What is the opus rejection rate we're going to go with? … We've not decided yet."* Elsewhere: reject
more to get better data.

Rejection **multiplies required supply**: `need = share ÷ (1 − r)`. Against S5's audited supply at 3T:

| rejection | lanes breaching the 4-epoch ceiling |
|---|---|
| 0% | agentic |
| 18% (S5's target) | agentic, indic |
| 45% | **every lane** |

"Reject more" is only available where supply has headroom, and S5's two tightest lanes (Indic 3.31
epochs, reasoning 3.27) have almost none. **Raising `r` is arithmetically equivalent to shortening the
run**, and nothing in the plan says so.

**Cost of missing it:** you set `r` high for quality, and discover 40% of the way in that a protected
lane is exhausted. That is the expensive discovery this note exists to prevent.

**Fix:** publish `r` with its per-lane consequence attached, and give each lane a
`max_feasible_rejection`. Mine computes it by bisection against the audited supply; it belongs in the
mixture document, not in the loader.

### C4. The 4-epoch ceiling is the wrong shape — but do not simply raise it

**What the plan says.** `epochs ≤ 4` for every lane, from Muennighoff et al., applied as a flat
lane-independent constraint. It is what makes Indic (3.31) and reasoning (3.27) "the tight lanes."

**What the evidence says**, after reading the relevant paper in full (Apple/ITU, 2,000+ runs, and the
setup is structurally ours — scarce target mixed with abundant generic, tested on German, French,
**Swahili**, and quality-tiered mixtures):

- **Degradation is governed by each lane's own repetition factor `r`, not by a global epoch count.**
  The same target weight is safe or harmful depending on supply; the same `r` degrades consistently.
- **Scarcer lanes want *higher* `r`.** Their 5×-smaller target optimum was **r ≈ 30**; the 5×-larger
  was **below 7**. Applying `≤4` to both agentic (0.63B) and web (4,691B) is the wrong shape.
- Mixture tolerance is genuinely higher than single-source — **15–20, up to 30** — because generic
  data supplies fresh tokens that never saturate and regularize the repeated target.

**But I have to correct myself here.** I previously wrote that the ceiling was "too conservative" and
that Indic and reasoning therefore had "comfortable headroom." The full read does not support that:
**optimal repetition and optimal target weight both fall with model size** — the weight needed to hit
optimal `r` drops from **9.5% at 101M parameters to 1.9% at 539M** — and the authors state plainly
that *"whether the fitted exponents extrapolate to multi-billion-parameter models remains an open
question."* We plan ~30B. Every scale trend in the paper runs against relaxing the ceiling for us.

Their hardest extrapolation case is **Swahili**, the closest analogue to our low-resource Indic
languages, and their weakest predictions are on **quality-tiered mixtures**, which is exactly what the
Indic lane is.

**What needs deciding:** re-parameterise the constraint from "epochs ≤ 4 per lane" to "per-lane
repetition factor `r_lane`" — free, and strictly better-shaped. Do **not** raise the level on this
paper's authority. Fit their law on our own proxy runs instead; S5's T-series already has the harness.

**One conclusion survives untouched:** agentic at 104 epochs is infeasible under any reading,
including r ≈ 30.

**And two things worth taking regardless.** Broad quality filters can beat more repetition — they
measure a crossover where a broader quality band overtakes a narrower one, which supports S5's Indic
tier broadening with a mechanism rather than an intuition. And among *multiple* constrained lanes,
weighting proportional to supply beats equal weighting when data is scarce; S5's agentic 2%-on-0.63B
is a deliberate departure from that, defensible as a capability requirement but worth naming as a
departure.

### C5. Two smaller ones

**Checkpoint cadence is costed in money but not in correctness.** *"Every $500, every hour"* is the
right instinct for spend. The interval also sets how many batches a crash can orphan (B1) and therefore
how much work resume redoes. Both numbers should be quoted together.

**"Immutable shard" has no stated identity rule.** The plan says shards are immutable and that
modifying one creates a new shard. It never says what the identity *is*. If it is anything other than
a hash over the token bytes, two shards can compare equal while differing and the lineage is
decorative.

---

## What I would add to the plan, in priority order

1. **Length reserve** (C1) — the only item here that can waste ~40 days of compute with no recovery.
   Implemented and measured in this submission; the rule is three lines in the scheduler.
2. **Reference loss recorded alongside perplexity** (A1) — free now, and without it the V6 payload
   does not support the inference it exists for. The literature is unambiguous that raw per-domain
   loss fails in both directions; excess loss against a reference is the established fix.
3. **Resume = re-plan, and position is global** (B1, B2) — silent corruption at every crash otherwise.
4. **Floors enforced against cumulative delivery** (B3) — already measured to overshoot ~1.9× in V5's
   own tooling.
5. **`max_feasible_rejection` per lane** (C3) — turns a free knob into a checked one.
6. **Per-lane repetition ceiling** (C4) — re-derive rather than inherit 4×; it currently decides
   which shares get cut, and it may be the wrong constraint.
7. **Stage × band feasibility** (C2), **reserve enforcement** (B4), **OPUS scores** (A2),
   **per-shard yield** (A3), **tokenizer provenance** (A4).

---

## Cross-reference

Three further items live in [`RESEARCH-ASSESSMENT.md`](RESEARCH-ASSESSMENT.md) because they come from
published work rather than from this build, and each challenges a plan decision rather than a gap:

- **The anneal is the one configuration the relevant paper measures as negative.** Read in full
  (Luo et al., ICLR 2026): the LR is an implicit importance weight, so decaying it to zero shrinks
  the update in the *signal* direction exactly when the best data arrives. Their mid-training table
  puts S5's design — uniform-ish ordering in the main run, best data in the late phase — at
  **−0.04 average / −0.31 core**; ordering *within* lanes as well measures **+1.20 / +2.00**. Their
  multi-domain algorithm delivers that without moving any lane share, validated at 3.2B over 886B
  tokens. It also means S5's **T3 is confounded**: it measures the reserve in the regime where the
  paper says it cannot pay off, so a null result would be uninterpretable.
- **Loss spikes are an optimizer–curvature event that specific data reliably triggers.** This is a
  correction to an earlier version of this note, which said flatly "an interaction, not a data
  property". Both halves are evidenced and the flat version was wrong:
  - PaLM §5 ran the ablation — the same batches from an earlier checkpoint did not spike — so a
    shard coinciding with a spike must **never** be auto-labelled `harmful`. That would poison the
    V6 payload with an artifact, and the ablation is the procedure for resolving it.
  - OLMo 2 §3.1 investigated their spiking batches and found long **repeated n-gram runs** heavily
    over-represented; filtering documents with a run of 32+ repeats of any 1–13 token span
    measurably flattened the gradient-norm trace. They are equally explicit that the relationship is
    non-deterministic — same sequence, spikes for a big model and not a small one, spikes under one
    ordering and not after a reshuffle.
  - The mechanism that reconciles them is the adaptive edge of stability (Cohen et al., via Wortsman
    et al. §4): whether a spike occurs depends on optimizer state and curvature, which degenerate
    data drives.
  **So V5 should filter the trigger at curation and still never blame a shard for coinciding with a
  spike.** Implemented and controlled — unlike spike *recovery*, which needs a model large enough to
  spike and remains untestable at demo scale.
- **Perplexity level cannot distinguish learnable-hard from irreducible-hard** (RHO-1, read in full).
  Measured token trajectories: 26% are genuinely being learned, but **11% are persistently high-loss
  from aleatoric uncertainty** — noise. They look identical in a snapshot and have opposite
  implications for what V6 should collect. The V6 payload needs the *trajectory*, which means
  re-scoring the same tokens at multiple checkpoints. Also cannot be reconstructed later.
- **The supply figures do not state a dedup basis.** At the 50–70% reduction typical of MinHash dedup,
  Indic at 157.2B could be 50–80B, moving the tightest lane through the ceiling under any reading.

## Appendix — widget defects

Separated because these are teaching-tool bugs, not plan gaps. Reproductions and sweep output in
`$HOME/shared/S6-extracted/`.

| widget | defect (measured) |
|---|---|
| shard manifest builder | admits at 87/100 with `pii_screen_status` **and** `dedup_status` missing |
| shard manifest builder | `content_hash` is `Math.random()`, regenerated per render |
| packing simulator | `greedy`, `best` and `structured` run the same search — identical in 10 of 12 cells |
| packing simulator | `concat_chop` never truncates a document |
| packing simulator | at ctx < longest doc: utilization **105–111%**, unused positions **−48** |
| padding lab | right / left / crop report identical metrics in all 24 cells (crop is a no-op loop) |
| perplexity heatmap | "Pos" view reads a missing field; all tokens collapse into one `undefined` bucket |
| throughput lab | 1 of 648 configs produces any loader wait; storage model ~100× under-scaled |
| mixture compiler | anneal demand computed against the full budget, not the 18B window — **6.7×** |

**Worth keeping:** the OPUS board checks `eval_firewall_overlap` *before* the protected-floor override
— the one ordering that must never be inverted. And the firewall's four-gate design is genuinely good:
disabling any single gate leaks nothing, and only overlap **and** derived together leak — and what
leaks is a *paraphrase*, which is the correct lesson about where firewalls are thin.
