# Training Data Execution System — ERA V5, Session 6

**Repository:** https://github.com/vedapragna-cs/era-v5-s6-data-ledger

> An earlier, independent implementation of this assignment is preserved on the
> [`v1-data-ledger`](https://github.com/vedapragna-cs/era-v5-s6-data-ledger/tree/v1-data-ledger)
> branch. It reaches the same submission structure by a different route — a PyTorch
> transformer, a larger corpus (102 steps, 122 shards), and a tighter mixture allocator
> (0.68 pp maximum lane deviation against 4.94 pp here). It has far weaker falsification,
> though: no independent verifier, no shipped token bytes, and four negative controls
> against the thirty-two here. This branch is the submission.

Executes a 3T-token mixture plan and **proves what it executed**.

Self-contained: **numpy is the only third-party dependency** (`pip install -r requirements.txt`;
Python 3.10+). No network, no downloads, no dependency on any other session's directory. The plan it
executes is checked in at [`plan/v5_plan.json`](plan/v5_plan.json) — it carries the lane shares,
curriculum stages, protected floors, audited supply and anneal reserve that this system consumes.
(Those numbers came from our S5 mixture-planning work, but nothing here reads that directory.)

```bash
python3 run_demo.py                  # ~35s, no network, no downloads
python3 run_demo.py --deep           # ~40s, lanes separate in perplexity (see caveat below)
python3 verify_evidence.py           # recompute every claim from artifacts alone
python3 tests/test_invariants.py     # 57 invariant tests
python3 make_dashboard.py            # static dashboard.html from the artifacts
python3 audit.py --step 7            # what trained this step
python3 audit.py --interval 0 6      # which shards influenced this range
python3 audit.py --shard v5_indic_train_002
python3 audit.py --branches          # divergence points and declared changes
```

Writes `submission_artifacts/` — `run.log`, `evidence.json`, `evidence.md`, `manifests/`,
`ledgers/`, `checkpoints/`, `performance.json`, `reports/`. **Exits non-zero if any invariant
fails**, so the demo cannot report success vacuously.

---

## What this is

S5 produced a 3T-token mixture plan. A plan is only worth having if the training system can execute
it accurately and prove afterwards what it did. This is that system:

```
documents → cleaning → dedup → PII screen → eval firewall → frozen tokenizer
  → immutable shards + manifests → admission gate → mixture schedule → OPUS selection
  → packing (masks, position ids) → batches → training
  → consumption ledger → learning ledger → checkpoint → crash → resume → replay → fork → audit
```

Two scales, **one code path**: the production plan (3T) is evaluated for feasibility as arithmetic;
the demo executes the identical scheduler objects with a scale divisor. `plan/v5_plan.json` holds
S5's numbers once and both read it.

## The three design decisions that matter

**1. Position is one global integer, and the rank partition is derived at load.**
A loader that persists "rank *r* has consumed *n* samples" duplicates or drops data when the machine
shape changes — and the loss curve will not tell you. Here the ledger offset is the position and
`rank = global_sample_index % world_size` is recomputed on restore. The demo crashes at 4 ranks and
resumes at **8**, then compares two independently produced runs element-wise.

> This test found a real bug. The first version compared a *re-partition of one list* against
> itself, which cannot fail. Written honestly, it exposed a divergence at global sample position 37:
> resume was re-serving orphaned batches from the ledger without advancing the planner's cursors, so
> the first newly-planned step drew from the wrong pool position. Resume now re-**plans** from the
> checkpoint and uses the ledger as the independent check. See `tds/exactly_once.py`.

**1b. The attention mask is enforced in the arithmetic, not merely validated.**
An earlier version of this system used a bigram model, on the reasoning that a transformer "would add
compute without changing a single ledger property". That was wrong about the one property this
submission argues hardest for. With no attention mechanism, `attention_crossings == 0` describes a
*mask array* — correct, but never exercised, because nothing existed that could have leaked.

The model now applies the mask inside 4-head self-attention: disallowed (query, key) pairs are
set to `-inf` before the softmax, so a document physically cannot read outside itself. That makes the
claim testable by perturbation instead of inspection:

| experiment | document 2's per-token losses |
|---|---|
| rewrite document 1, **segment mask on** | **bit-identical** |
| rewrite document 1, segment ids merged | move by 1.7e-03 |

The second row is the control. Without it, the first row would also pass on a model that simply never
looks across the boundary, and would prove nothing. Gradients are hand-written (no autograd), so they
are checked against finite differences in `test_hand_written_gradients_match_finite_differences` —
max relative error 3.2e-06.

**2. Protected floors are enforced over the run, not the batch.**
With a global batch of 8, the smallest expressible share is 12.5% — so a 2% agentic floor becomes
12.5%, and three protected lanes lock 37.5% of every batch. Measured on the first build: **agentic
delivered 18.9% against a 2% floor**. That is the same failure S5 measured in the course's own
always-on lane (stated 8%, delivered ~15%). Floors are now deficit-based against cumulative
delivery; agentic lands at **2.17%** against its 2% floor.

**3. Admission gates documents, then seals shards.**
A shard manifest reports the worst status among its members, so gating at shard level lets one
near-duplicate destroy an entire shard. Measured: every web shard was blocked and the **web lane —
34% of S5's budget — silently vanished**, which presented as a floor bug. Documents are now screened
first and exclusions recorded.

All three were found by running the thing, and all three are the class of bug that produces a
plausible-looking run with wrong data.

## Where this deliberately differs from the course widgets

Each widget was driven and swept before building; every difference is a measured defect, not taste.

| widget behaviour (measured) | what we do |
|---|---|
| Manifest gate admits a shard at 87/100 with `pii_screen_status` **and** `dedup_status` missing | all four S4 fields are hard blockers, plus a fifth for repeated n-grams |
| `content_hash` regenerated by `Math.random()` each render | SHA-256 over the shard's token bytes |
| "greedy", "best-fit" and "structure-preserving" all run the same best-fit search | six genuinely distinct policies (`test_greedy_and_best_fit_are_not_the_same_algorithm`) |
| "concat and chop" never truncates a document | ours truncates (`test_concat_chop_actually_truncates`) |
| utilization 105–111% with −48 "unused positions" at ctx < longest doc | `0 ≤ pad ≤ length` asserted per bin |
| position-bucket perplexity view reads a field that does not exist | implemented (`learning.position_bucket_report`) |
| 1 of 648 swept configs produces any GPU idle; storage model ~100× under-scaled | we measure real candidate-preparation vs step time (OPUS scoring dominates it, and the field says so) and do not model GPU idle |

## Answering a question the session left open

> *"do we make a shard in which we have 10% [indic] and 90% … or we keep the indic separately.
> That is one question that I've not thought about."*

Measured both arms (`reports/experiments.json`, E1):

| arm | Indic share by scheduler lane tag | by actual document content | auditable from the ledger? |
|---|---|---|---|
| **P — pure lanes** | 20.00% | 20.00% | **yes** |
| M — mixed into web shards | 16.39% | **44.26%** | **no** |

The decisive difference is not the share, it is that in arm M the ledger records the *carrier* lane,
so the protected lane's real delivery is invisible to every downstream audit — understated here by more than half. **A floor you cannot audit is not a floor.** Decision: pure-lane shards, floors injected
post-selection as whole batches.

## Two subsystems that exist because of the literature, not because the assignment asked

Both were added after a systematic read of the pretraining-data literature, and both are here on the
same terms as everything else: a measured effect, a negative control, and a stated limitation.

**Repeated-n-gram screening — the spike trigger.** OLMo 2 ([arXiv:2501.00656](https://arxiv.org/abs/2501.00656) §3.1)
investigated the batches where loss and gradient-norm spikes occurred and found long repeated n-gram
runs heavily over-represented; filtering them measurably flattened the gradient-norm trace. Their
rule — drop documents containing a run of 32+ repeats of any 1–13 token span — is adopted verbatim,
as is their second intervention, a **trainer-side loss mask** for anything that reaches the loader
anyway (`tds/packing.py::mask_repeated_ngrams`; tokens stay visible as context, only the loss is
masked, so span ids and ledger offsets are unaffected).

The planted case is deliberately **unique**, so shingle-Jaccard scores it perfectly clean: a
degenerate document is not a duplicate document, and no similarity threshold will ever find it.
Measured: `web-0008`, run of 60×1-gram, `dedup_status=unique`.

This does **not** overturn the PaLM reading elsewhere in these documents. OLMo 2 are explicit that
the relationship is non-deterministic — the same sequence spikes for a larger model but not a
smaller one, spikes under one data ordering but not after a reshuffle, and appears in non-spiking
batches too. The reconciliation is that spikes are an optimizer–curvature event (the adaptive edge of
stability, Cohen et al. via [arXiv:2309.14322](https://arxiv.org/abs/2309.14322) §4) that certain
degenerate data reliably triggers. So we filter the trigger **and** still never mark a shard harmful
merely because it coincided with a spike.

**N-gram containment — the decontamination standard.** Our overlap gate was shingle-Jaccard, which
measures *global* similarity between two documents. That structurally cannot see the ordinary shape
of contamination: one benchmark item quoted inside a long host article. GPT-3 filtered on 13-gram
overlap and the open replications use 8–13 token windows, so containment is now a sixth, independent
gate.

Measured on planted case `web-0102` — a long ordinary article quoting one GSM8K item verbatim:

| gate enabled alone | blocks it? |
|---|---|
| never_train, content_hash, canary, overlap, derived | no (Jaccard **0.0806** vs threshold 0.12) |
| **ngram_containment** | **yes** |

Adding it exposed a flaw in an existing planted case: `web-0100`, labelled a paraphrase, actually
reused 14 words of the benchmark verbatim, so containment caught it too and the "derived gate is
load-bearing" control was measuring the wrong thing. It has been rewritten as a genuine paraphrase,
and each contamination case now isolates exactly one gate. Stated limitation: exact n-gram matching
is defeated by paraphrase, translation and reformatting — it is a floor, not a ceiling, and semantic
retrieval is the named next layer.

## Packing: where our measurement meets the literature

`best_fit` is Best-Fit-Decreasing over chunk-then-pack, which is the algorithm in *Fewer Truncations
Improve Language Modeling* ([arXiv:2404.10830](https://arxiv.org/abs/2404.10830), ICML 2024). Our
own measurement reproduces their central claim on this corpus:

| policy | documents truncated / split | utilization | sequences |
|---|---|---|---|
| `concat_chop` | **74** truncated, **31** split | 0.978 | 44 |
| `best_fit` | **0** truncated, **15** split | 0.957 | 45 |

They measure what those truncations cost downstream at 7–13B: +4.7% reading comprehension, +16.8%
context following, +9.2% program synthesis, and up to −58.3% closed-domain hallucination. They also
report the packing overhead as ≤0.01% extra padding at billion-document scale; here it costs **one
extra sequence out of 44** to remove all 74 truncations. The overhead shrinks as the bin-packing
instance grows, which the paper predicts and which is why our tiny corpus is an upper bound on it. Their scale answer is a segment tree giving O(N log L) instead of our O(N²) linear scan;
named here rather than implemented, on the same terms as MinHash-LSH for dedup.

`web` still runs `concat_chop` in the plan, and that is now a **documented disagreement with the
evidence** rather than an unexamined default — the plan is S5's artifact and this system's job is to
execute and measure it, so the recommendation goes to `reports/v6_recommendations.json` instead of
being quietly applied. Segment-isolated attention masking follows Krell et al.
([arXiv:2107.02027](https://arxiv.org/abs/2107.02027)), whose cross-contamination result is the
reason `attention_crossings` must be 0 while `blocked_cross_segment_pairs` must be > 0.

## Three measurements the assignment did not ask for

**Excess loss against a frozen reference** (`tds/reference.py`). A perplexity level cannot tell you
whether a hard token is hard *in a way more training fixes*. RHO-1 measured 26% of tokens
learnable-hard and 11% irreducibly hard — identical in a snapshot, opposite implications for what V6
should collect. We train a reference on the golden proxy set (already firewalled out of training, so
it cannot leak), hold its architecture identical to the trained model so the difference isolates
*data*, and record `excess_loss = loss − reference_loss` per token.

Per lane it is DoReMi's reweighting signal. Measured: **agentic +0.1446** — the only lane still
behind the reference — then indic -0.4278, longctx -0.5688, code -0.8070, stem -0.9468, web -1.1651,
reasoning -1.5627.

**This inverts what the same table said one revision ago, and the reason is the optimizer, not the
data.** Under constant-LR SGD every lane came out positive (web +0.2572 down to indic +0.0497) and
this README read that as "every lane is `behind_reference`, so this is a ranking of headroom". With
AdamW the trained model overtakes the golden-set reference on six of seven lanes inside 30 steps,
and only agentic stays behind. Two earlier readings are retracted outright: that indic has the least
headroom of the seven (it now has the *most* of the six that are ahead), and the claim that this
corroborated the exposure confound by an independent route — it was corroborating the optimizer.

What survives is the mechanism and one result: excess loss separates *hard* from *hard in a way more
training fixes*, and agentic is the single lane where a model trained on the full mixture still does
worse than one trained on the proxy set alone. That is the lane V6 should collect more of, and it is
now a statement about one lane rather than a ranking of seven near-identical numbers.

**A loss spike is a run event, never a shard property** (`tds/spike.py`). The learning ledger marks a
shard `harmful` when loss rises across the step that consumed it. A spike does exactly that — and
PaLM's ablation showed a spike is generally the interaction of a batch with a parameter state, not a
property of the batch. So the naive version ships a permanent wrong label that nothing downstream can
detect.

Spikes are recorded in `ledgers/run_events.jsonl`; shards in a spike window have their verdict
**withheld**, and the withholding is resolved by running PaLM's own procedure — restart from an
earlier checkpoint, replay the identical batches, see whether it recurs. That experiment exists only
because replay is byte-exact.

> **It fired on the real run.** Step 29, gradient norm, z = 4.69. The ablation replayed it from an
> earlier checkpoint and it reproduced — but step 29 is the **anneal seam**, where lane composition
> changes. Reproducible *at a stage transition* means the mixture changed, not that the documents
> are bad, so the verdict is `seam_composition` and shard verdicts were withheld rather than
> blamed (6 withheld). The fix that implies is seam blending, not deleting data.

**Dedup break-even, not a binary ceiling verdict** (`mixture.dedup_sensitivity`). S5's supply figures
are pre-dedup by its own account (`S5/SUPPLY-AUDIT.md` §7 lists the dedup rates as still not
obtained — that file is the session-5 submission on the `hola` branch, not part of this
standalone S6 tree, and nothing here depends on it at runtime), so every `within_ceiling` verdict is computed on an upper bound. We compute
`d_max = 1 − demand/(supply × ceiling)` instead:

| lane | r | max tolerable dedup | status |
|---|---|---|---|
| agentic | 126.59 | −3064.7% | already breached |
| indic | 4.03 | **−0.8%** | already breached |
| reasoning | 3.99 | **+0.3%** | **fragile** |
| stem / code / longctx / web | 2.99 – 0.26 | +25% – +94% | robust |

`reasoning` reads `[OK]` in the feasibility table with a margin of three tenths of one percent. Three
lanes, not two, cannot be signed off until their sources' dedup rates are measured — the cheapest
unblocking measurement in the plan.

## Checkpoints refuse to load when they are wrong

A checkpoint that fails to load is an inconvenience. A checkpoint that loads while subtly wrong is a
silent disaster: the model resumes happily from the wrong position in the data stream and the loss
curve will not tell you. So `restore()` verifies before loading and **raises** rather than warns, and
nine corruption flavours are induced and required to be caught:

`weight_bit_flip`, `torn_write_truncated_weights`, `loader_cursor_moved`, `ledger_offset_moved`,
`step_counter_moved`, `tokenizer_identity_swapped`, `optimizer_moment_bit_flip`,
`optimizer_step_counter_moved`, `optimizer_moments_dropped` — all nine detected, and a separate
control proves `restore()` actually refuses rather than logging.

Three hashes, because they fail differently. `model_weight_hash` covers the weights;
`optimizer_hash` covers AdamW's moment buffers and step counter; `checkpoint_hash` covers step,
branch, ledger offset, loader cursors and tokenizer/plan identity. A corrupted loader cursor leaves
the model perfectly valid, and a dropped moment buffer leaves *both* the model and the cursor valid.

### Optimizer state you can actually lose

The last of those three is here because of a defect worth stating plainly. This system trained with
constant-learning-rate SGD, and its checkpoints recorded `optimizer_state = {"type": "sgd",
"lr": 0.35}` and `scheduler_state = {"policy": "constant"}`. Both are copies of the config. No
mutation of either could change what the next step computed, so **"resume restores the optimizer"
was a sentence with no check under it** — the same failure class as position ids that no arithmetic
read, one layer further down.

The optimizer is now AdamW (β=0.9/0.95, decoupled decay 0.01 exempting biases and both embedding
tables) with linear warmup to 6% of the run and cosine decay to 10% of peak, and global-norm
clipping at 1.0. Two moment buffers and a step counter are real state, so losing them is a real
event, and the drill is built so that losing them is invisible to every check that existed before:
restore twice from one checkpoint, drop the moments on the second, train three steps each. Measured
— the weight hash immediately after both restores is **identical**, and after three steps the two
trajectories **diverge**. A resume that forgets the optimizer passes every weight-hash comparison at
the restore instant and quietly re-warms Adam's bias correction over the following steps.

One finding from building the drill, because it is a property of Adam rather than of this code: with
a *constant* gradient the normalised step `m̂/√v̂` converges to `sign(g)` regardless of history, and
bias correction makes a fresh optimizer's first step equal to it exactly. A probe using a constant
gradient therefore reports "the moments are not load-bearing" and is wrong about why. The test uses
a varying gradient sequence and says so.

Changing the optimizer moved every number measured from a forward pass. The SGD rates (0.35 default,
1.2 deep) diverged outright under Adam — mean loss 14.3 nats against a uniform baseline of
ln(512)=6.24 — because Adam's per-parameter step is ~`lr` whatever the gradient scale. Re-swept at
default depth: 3e-3 → 5.951, 6e-3 → 5.703, 1e-2 → 5.668, 2e-2 → 5.491, 4e-2 → 6.020. The last
reaches the sweep's lowest minimum (3.348) and then climbs back above 6, which is overshoot rather
than learning, so **2e-2** is the default and `--deep` runs 5× the steps at half the rate.

## Throughput: the straggler a token counter cannot see

Attention is quadratic in segment length, so a 30+70 sequence costs ~16% more than 50+50 at the same
token count (MegaScale-Data, [arXiv:2504.09844](https://arxiv.org/abs/2504.09844)). Our rank
partition is `global_index % world_size`, which balances *tokens*.

Measured: **tokens differ by 0.0% across 4 ranks while attention cost differs by 17.3%.** The step
time is set by the slowest rank, so that is wasted GPU time per step — and perfectly invisible to any
monitor counting tokens.

Measured, not fixed, and the reason is recorded rather than glossed: a composition-aware partition
stops rank assignment being a pure function of the global index, which is the exact property that
makes resumption across a changed world size exact. Real design tension, not an oversight.

## Loss attributed per role, not only per lane

Every packed sequence carries a `role_codes` array beside its loss mask, and it lands on every token
observation. The mask alone cannot support this: supervised and context roles both collapse into it,
so `tool_call` and `text` are indistinguishable 1s and `user` and `tool_observation` indistinguishable
0s. Lane was therefore the finest attribution the ledger could make — and the question S5 left open
is not about lanes. It asks whether the *supervised part* of an agentic trajectory is the expensive
part, which is a per-role question.

Measured on run-a's surviving observations:

| role | geometric-mean ppl | tokens |
|---|---|---|
| `boundary` (EOS) | 37.82 | 41 |
| `tool_call` | 121.97 | 24 |
| `text` | 190.78 | 5,215 |
| `assistant_plan` | 336.32 | 48 |

Only supervised roles can appear, and that is the check rather than a caveat: context roles are read
by the model and never scored, so a context role carrying loss would mean the role array had drifted
out of step with the mask it explains. Two things are asserted independently — the table's token
total must equal the lane table's exactly (5,328 = 5,328), and `verify_evidence.py` recomputes every
perplexity from the shipped token records and requires no context role among them. The negative
control rolls `role_codes` by one position, which leaves every count and every length identical, and
requires the validator to reject it.

**At default depth these values do not answer the question.** `assistant_plan` at 48 tokens and
`tool_call` at 24 are far too thin to rank against `text` at 5,215, and the model sits near the
vocabulary floor anyway.

`--deep` trains long enough for the ordering to mean something, and it points the way S5 assumed:

| role | geometric-mean ppl | tokens | vs `text` |
|---|---|---|---|
| `boundary` (EOS) | 7.07 | 211 | 0.17x |
| `text` | 41.35 | 26,187 | — |
| `tool_call` | 70.76 | 114 | **1.71x** |
| `assistant_plan` | 95.52 | 228 | **2.31x** |

The two supervised agentic roles are the two hardest things the model is asked to emit, at roughly
1.7x and 2.3x plain text — which is the direction S5's 53% supervision-yield argument assumed and
had no measurement for. **Read it as a pointer, not a result.** 114 and 228 tokens are small
samples, they come from one lane whose exposure is 1.3%, and the exposure confound above applies to
roles exactly as it does to lanes. The deliverable here is a ledger that can be asked the question
at all; answering it needs a run with real agentic volume.

## What did not work

Reported because a submission that only shows successes has not been audited.

- **OPUS with a fixed score threshold rejected everything.** Cosine similarity in ~50k-dim parameter
  space is centred on 0 with σ≈0.004, so the plan's 0.05 threshold sat ~11σ out: 0 accepted, 39
  protected, 29 rejected. Replaced with top-k over the candidate pool, which is what the session
  actually describes and makes the rejection rate a controllable plan input.
- **The demo initially never left the last curriculum stage.** At the first scale divisor the whole
  scaled run was 1,000 tokens while one step consumes 1,024. The curriculum gate was working
  correctly; the scale was wrong.
- **E6 (Indic difficulty ordering) is inconclusive** and is reported as such. A 32-dim model over a
  few dozen documents cannot resolve a curriculum effect; the measured advantage is +2.4e-03, which
  is noise at this scale. The mechanism is implemented and ready to run where it could decide.
- **Floors initially undone by normalisation** — protected lanes were raised to their floor and then
  the whole vector renormalised, scaling them straight back under it.
- **The run never reached its own final curriculum stage.** Anneal opens at scaled token 29,100 =
  step 28.42; the demo ran 24 steps. Four of five stages executed, and the anneal reserve was
  therefore never released — every reserved document stayed withheld for the entire run, which makes
  a deferral indistinguishable from a permanent exclusion. Nothing asserted stage coverage, so
  nothing caught it. Found while trying to prove a deferral was honoured, not by inspection.
- **The post-resume advance trained past the plan's budget.** It was a hard-coded four steps; once
  the run was long enough to reach anneal, those four ran past `run_tokens`, where `stage_for_token`
  clamps and out-of-plan batches were silently attributed to the anneal stage. It flattered the very
  result being measured: with the overrun, every reserved document appeared to be admitted at anneal;
  plan-faithful, the honest number is 1 of 5. The advance is now bounded by the scheduler.
- **A duplicate-delivery scare that was not one.** After resume, six ledger records looked duplicated
  because none carried a `superseded` field. There is no such field: supersession is a separate
  append-only `batches_superseded` event carrying offsets. Checked properly, the effective stream is
  steps 0–29 with no duplicates. Recorded because the wrong check looked exactly like a real bug.

## The learning ledger's numbers, and what they do not show

At default depth the model is barely trained, so every lane's perplexity sits near the vocabulary
floor `exp(ln V) = 512` and the ordering is noise. `--deep` separates them properly — **36.8x
spread** over 147 steps, mean loss falling from 6.241 to a minimum of 1.89 — and the run reports
which regime it is in rather than letting the reader assume:

| lane | traced tokens | share | geometric-mean ppl |
|---|---|---|---|
| `longctx` | 1,063 | 4.0% | 733.57 |
| `agentic` | 342 | 1.3% | 86.43 |
| `stem` | 2,964 | 11.1% | 69.65 |
| `indic` | 3,336 | 12.5% | 65.58 |
| `web` | 8,856 | 33.1% | 35.50 |
| `code` | 7,136 | 26.7% | 27.40 |
| `reasoning` | 3,043 | 11.4% | 19.91 |

**Exposure and perplexity are still strongly related, and the run measures how strongly rather than
asserting it.** Traced-token share spans 1.3% (agentic) to 33.1% (web) — a 25x range in how much
each lane was trained on — and Spearman rank correlation between exposure and perplexity is
**−0.714**: more exposure, lower perplexity. So lane perplexity here measures **training volume at
least as much as intrinsic difficulty**, and a difficulty comparison would need equal exposure or a
per-token-seen normalisation. This run does neither.

What has changed with a real optimizer is that the automatic confound flag no longer trips. It fires
when the most-exposed lane is also the lowest-perplexity one; web is the most exposed but
`reasoning` (11.4%) now has the lowest perplexity, and `longctx` (4.0%) is worst by a factor of 8
over the next lane. Those two are the informative departures from the exposure trend, and they are
the shape you would expect: reasoning documents are short and formulaic, long-context ones are not.
**One rank correlation over seven lanes is not evidence for that reading**, and it is offered as the
next thing to test, not as a result.

*Two earlier versions of this section made stronger claims from the same measurement and neither
survived.* The first reported Indic receiving 49% of traced tokens and the lowest perplexity (138
against web's 435), calling it an inversion of expectation; both numbers were artefacts of protected
lanes having static immunity instead of deficit-gated floors. The second, after that fix, reported a
3.04x spread with Indic highest at 444 — measured under constant-LR SGD, which barely trained the
model, so the spread was mostly floor noise. Both are superseded by the table above. See
`EXPERIMENT_LOG` E10, E14 and E15.

We report the confound instead of fixing it, because fixing it means holding exposure equal across
lanes — which would stop the run from executing S5's mixture. Distorting the schedule to produce a
prettier number is the failure mode this whole submission is arguing against. See `EXPERIMENT_LOG` E7.

## Determinism: scope stated precisely

**Guaranteed exactly** — the data stream. Selection, ordering, packing, masks, position ids and all
content hashes are pure functions of `(plan, corpus, tokenizer, run_seed, global_step)`. No stateful
global RNG: every draw derives from `(seed, step, purpose)` via SHA-256, so worker count, world size
and machine cannot change it. Two full runs are compared and every artifact hash matches.

**Not claimed** — bitwise model state across hardware. BLAS reduction order varies. Within one
machine and thread count this is reproducible (`float64`, fixed reduction order) and the demo
asserts it, but that is a property of the configuration, not a portable guarantee. **The ledger does
not depend on model determinism** — which is precisely why replay restores a checkpoint rather than
recomputing one.

## Prior art

| system | taken | where we differ |
|---|---|---|
| Megatron-Core GPT dataset | plan as a pure function `step → sample ids` | no epoch concept: S5 runs lanes at 0.21–3.31 epochs, so repetition is a per-lane budget, not an epoch counter |
| Mosaic StreamingDataset | resumption must survive a changed worker count, and be tested | our stream is scheduler-generated (quotas, floors, OPUS), so position is recovered from the ledger rather than by arithmetic |
| NeMo Curator | dedup is a pre-shard admission property | exact + shingle-Jaccard is honest at this corpus size; MinHash-LSH is named as the scale answer rather than faked |
| WebDataset | shard as immutable object; multi-part agentic records | token arrays + sidecar manifests; tar matters for throughput we do not credibly measure here |
| Iceberg / Delta / LakeFS | append-only log, branches, divergence point | commits to *delivery position* in a stream rather than table state |

## Layout

```
run_demo.py              one command, full demonstration
plan/v5_plan.json        S5's numbers: stages, shares, floors, supply, reserve, seams
tds/
  canon.py               canonical hashing, hash chain, derived RNG
  corpus.py              7-lane corpus + planted PII / duplicate / contamination cases
  tokenizer.py           byte-level BPE, frozen and hashed (Indic-safe by construction)
  clean.py               cleaning hash, dedup, PII screen, language validation
  shards.py              immutable shards, manifests, admission gate
  firewall.py            eval/validation registry, six independent gates
  packing.py             six policies, loss/attention masks, EOS-reset positions
  mixture.py             stages → quotas, floors, seams, feasibility
  opus.py                top-k selection, accept/reject/defer/floor-override
  model.py               tiny numpy LM: 4-head masked self-attention, hand-written gradients
  ledger.py              consumption ledger, hash-chained, supersession
  reference.py           frozen reference model, excess loss, RHO-1 quadrants
  spike.py               spike detection, PaLM ablation, verdict withholding
  croissant.py           Croissant 1.1 provenance view of the manifests
  learning.py            per-token perplexity, trajectories, shard report cards, V6 recommendations
  trainer.py             loop, checkpoint, crash, resume, replay, fork, audit
  exactly_once.py        the delivery-equivalence comparison
  experiments.py         E1, E6, determinism
  controls.py            32 negative controls
tests/test_invariants.py 57 tests
```

## Auditing a run

`audit.py` answers the questions the session poses — *"which shards influenced the model between 5.4B
and 5.6B tokens?"*, *"which OPUS-selected batches appeared before a loss spike?"* — by reading
`submission_artifacts/` as data. It does not import the pipeline, so it answers the same way a grader
would. `make_dashboard.py` renders the same artifacts into one self-contained HTML file: a static
rendering, not a service (the session describes four servers and carves them out of the assignment).

## Ledger volume

Full per-token traces do not survive a real run. Measured on this run: **651 B per loss-bearing
token**, which extrapolates to **1.95 PB at 3T tokens**. So the ledger is tiered — full for proxy and
debug runs, quantized for selected intervals, aggregate for the full run — and the demo stores the
quantized tier with the tier costs reported in `reports/ledger_volume.json`. Pass `--full-trace` to
keep everything.

## Evidence

`evidence.json` contains no bare booleans: every requirement records the values compared, the
artifact paths and the record ids, and `passed` is computed from those values (`tds/evidence.py`
contains no literal `True`). Companion documents: [`../RUBRIC.md`](../RUBRIC.md) maps all 1,000
points to artifacts, [`../ARCHITECTURE.md`](../ARCHITECTURE.md) records the design decisions,
[`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md) holds the pre-registrations and results.
