# S6 — Training Data Execution System: architecture

The system that executes [S5's mixture plan](../S5/README.md) and proves what it executed.

Numbers are labelled **measured** (computed here), **stated** (asserted by course or S5), or
**stylized** (illustrative formula, no empirical backing).

---

## 1. The contract S6 inherits

S5 produced a plan. S6 must execute it *accurately, reproducibly and auditably*. Concretely, the
plan's obligations become code:

| S5 decision | S6 obligation |
|---|---|
| 3T run, 97% main (2.91T) / 3% anneal (90B) | stage schedule with token ranges, not percentages |
| 5 stages S1 Seed → S2 General → S3 Reasoning → S4 Long-ctx → Anneal | per-step lane quotas that integrate to the lane budget |
| protected floors Indic 12%, Agentic 2%, Reasoning 6% | always-on injection the selector cannot see |
| OPUS drives unprotected lanes to **0 tokens in 60/60 iterations** | floors enforced *after* selection, as mechanism |
| anneal reserve: Tier-A Indic 25.2B, long agentic 360M, long reasoning 16.2B | reserve is unspendable before the anneal stage |
| seam blending ≥2B tokens, 4B at the final seam | warmup interpolation across stage boundaries |
| 4-epoch repetition ceiling (arXiv:2305.16264) | supply feasibility check, per lane |
| agentic supervision yield **356/668 = 53%** | loss mask spec, verified against the packer |

## 2. Two scales, one code path

S5's plan is 3T tokens. The demo must run in seconds. Faking either is unacceptable, so:

- **Production scale (3T).** The mixture compiler runs S5's real numbers against S5's real supply
  and emits the feasibility verdict. No training happens here; it is an arithmetic artifact.
- **Demo scale.** The *same compiler code* runs with `scale_divisor` applied to the token budget.
  The stage boundaries, lane shares, floors, warmup bands and reserve rules are identical objects.

`plan/v5_plan.json` holds S5's numbers once. Both scales read it. A change to S5's plan changes both.

**Why this matters:** the widget-level failure this avoids is computing a phase's demand against the
whole-run budget. Measured in the S6 lesson widget: the anneal profile's demand is computed against the
full 120B budget while the timeline shows anneal as 15% of it — a **6.7× overstatement**. Our stages
carry explicit `token_start`/`token_end`, and lane demand is always computed against the *stage* window.

## 3. Pipeline

```
corpus documents
  → cleaning pipeline (hash) → dedup → PII screen → contamination scan
  → frozen tokenizer (hash) → token IDs
  → immutable shards + manifests → admission gate
  → eval firewall (train / validation / test permissions)
  → mixture compiler (stage → lane quotas → protected floors → reserve)
  → OPUS selector (accept / reject / defer / floor-override)
  → packer (policy per lane → loss mask, attention mask, position ids)
  → microbatches → global batch → optimizer step
  → consumption ledger (append-only, hash-chained)
  → learning ledger (token perplexity, gradient norm, loss delta, usefulness)
  → checkpoint (model+optimizer+scheduler+RNG+dataloader+ledger offset)
  → crash / resume / replay / fork / audit
  → evidence bundle
```

## 4. Determinism contract

Replay is only meaningful if the stream is reconstructible. Three rules:

1. **No stateful global RNG in the data path.** Every random decision derives its seed from
   `(run_seed, global_step, purpose)` via SHA-256. A worker restart, a different worker count, or a
   different machine cannot change the stream, because nothing depends on call order.
   This is the direct answer to *"why not just store the seed?"* — a seed only reproduces if every
   consumer draws in the same order, which is exactly what a distributed loader cannot guarantee.
2. **Content addressing everywhere.** Shards, batches, loss masks and ledger entries carry SHA-256
   hashes over canonical JSON (sorted keys, no whitespace). Equality is hash equality.
3. **The ledger is the authority, not the code.** Replay reads recorded `sample_ids` and re-packs
   them; it does not re-run the scheduler. If the scheduler changed, replay still reproduces history.

## 5. Admission gate — stricter than the course widget

**Decision (D2):** all four Session-4 fields are **hard blockers**: `cleaning_pipeline_hash`,
`dedup_status`, `pii_screen_status`, `eval_overlap_status` — plus `tokenizer_hash` and a safe license.

**Why we differ.** Measured on the course's shard-manifest widget: a shard missing PII screening scores
87/100 and reads "Admitted to registry"; same for missing dedup status. Only tokenizer hash, cleaning
hash, contamination and license actually block. The lesson quotes S4's contract as "cleaned,
deduplicated, PII-screened and contamination-scanned" and then makes half of it advisory. Since the
assignment is graded on the firewall being *executable*, an advisory PII gate is a defect, not a
simplification.

**Decision (D3):** `content_hash` is SHA-256 over the shard's token bytes. Measured on the same widget:
its `content_hash` is regenerated by `Math.random()` on every render — nudging a slider and returning to
the identical setting produced a different hash. A hash that is not a function of content cannot support
the immutability claim the same widget teaches.

## 6. Packing

Six policies, each *actually implemented* rather than aliased:

| policy | rule | used for |
|---|---|---|
| `pad_only` | one document per window | baseline / structured fallback |
| `concat_chop` | join with EOS, cut fixed windows, **documents genuinely split** | general web |
| `greedy` | **first** bin with room (order preserved) | code |
| `best_fit` | descending sort, tightest remaining gap | stem, reasoning |
| `structure_preserving` | never mix samples; pad the remainder | agentic |
| `long_context` | reserve full window for one long document | long-context |

**Why we differ from the widget.** Measured: the course packing simulator runs the *same best-fit
search* for greedy, best-fit and structure-preserving — only the pre-sort differs — so "greedy" and
"best-fit" were identical in 10 of 12 swept cells. Its `concat_chop` closes a bin *before* an
overflowing document, so nothing is ever truncated, which is the opposite of what "chop" means. And at
context < longest document it reports **utilization 105–111% with −48 "unused positions"** because total
padding is never floored at zero. Our packer truncates for real, first-fits for real, and asserts
`0 ≤ used ≤ capacity` per bin.

**Masks emitted per packed sequence:**
- `loss_mask` — per token. Pretraining: all real tokens. Agentic: assistant planning, tool calls and
  final answer including a *failed* call (S5: recovery is the taught behaviour); user turns and tool
  returns are context. Padding never loss-bearing.
- `attention_mask` — segment ids. Tokens attend only within their own document. This is what makes
  `concat_chop` safe: cross-document attention is blocked structurally, not just discouraged by EOS.
- `position_ids` — **reset at each document boundary** (`packed_reset_on_eos`, the policy the course
  ledger widget records).

## 7. OPUS

Score is a real proxy-alignment computation, not a random draw: cosine similarity between a candidate
batch's mean gradient direction and a golden-proxy direction, computed on the live model.

Decisions in the order the code applies them (`tds/opus.py::select`): `eval_firewall_overlap` →
`stage_mismatch` → `reserved_for_anneal` → `below_proxy_threshold` → `lane_quota_full` → accept,
with `protected_floor_override` able to rescue at every step **except** the first.

Reserve deferrals recorded by the lane pool are emitted ahead of this loop, because the pool passed
over those samples before they ever became candidates; the in-loop `reserved_for_anneal` branch
covers a reserved sample that still reaches scoring.

**Protected-floor override runs last and cannot rescue a firewall hit.** Measured on the course OPUS
widget, this precedence is correct there too, and it is the one ordering that must not be got wrong: a
protected lane rescuing a contaminated batch would defeat the firewall entirely.

**Rejections are retained.** S5's finding that unprotected lanes go to zero means the rejection log *is*
the evidence for the floor's existence. Every rejected candidate keeps its shard ids, lane, stage,
score, reason and effective-token estimate.

**Deferral is a record, not an absence** (D10). A reserved sample used to be skipped inside
`LanePool.draw`, so the document simply never appeared downstream and no artifact stated that it had
been withheld or why. That is the defect E1 rejected the mixed-lane shard arm for, one level down: a
protected quantity you cannot audit is not protected. The pool still performs the skip — the reserve
must keep those samples out of the pre-anneal stream — but now reports what it passed over, and OPUS
turns each one into a `deferred / reserved_for_anneal` record naming the document, the stage that
withheld it and the reason.

Deferrals are **recorded but never scored**, and are excluded from the candidate population. This is
load-bearing in two directions: the top-k cut is computed over the same population as before, so the
record-keeping cannot change what the run trains on (verified: every batch hash identical); and
`observed_rejection_rate` keeps its meaning, *of the samples OPUS judged*. A deferral says the stage
is wrong; a rejection says the sample lost to its peers. Folding the first into the second's
denominator would silently redefine a headline number — which it did, until `verify_evidence.py`
caught the inconsistency independently.

**A deferral only means something if it is later honoured**, so the run must reach the anneal stage —
which is what forced D11. See EXPERIMENT_LOG E8.

## 8. Ledgers

**Consumption ledger** — append-only, hash-chained (`prev_hash` → `entry_hash`). Any edit to history
breaks the chain and the verifier reports the first bad link. Records per S6 spec: run id, branch id,
global step, checkpoint id, rank, microbatch id, packed sample ids, shard ids, token span ids, loss mask
hash, attention/position policy, mixture lane, curriculum stage, tokenizer version, dataloader version,
OPUS decision id.

**Learning ledger** — the return path. Per token: loss, perplexity, lane, shard, position, mask flag.
Per shard: mean loss, high-perplexity cluster count, loss delta across exposure, gradient norm, model
phase, tokens consumed at exposure, and a `useful | neutral | harmful` classification.

**Why both.** The consumption ledger answers *what did the model see*. The learning ledger answers
*what did it get from it*. S5→V6 handoff needs the second; crash recovery needs the first.

## 9. Checkpoints

A checkpoint is not weights. It is `{model, optimizer, scheduler, rng_policy, dataloader_state,
ledger_offset, branch_id, stage}`. **A checkpoint without a data position is incomplete** — restoring
weights to step N while the data pointer sits at step M silently confounds every subsequent comparison.

Four recovery modes: `resume` (latest ckpt + latest offset), `replay` (old ckpt + recorded stream),
`fork` (old ckpt + new branch id, divergence point recorded), `audit` (reconstruct what trained an
interval).

## 10. Throughput

Measured from real wall-clock, not a formula. Reported: raw tokens/s, useful loss-bearing tokens/s,
accepted-after-OPUS tokens/s, packing utilization, loader wait, rejection rate by lane.

**We do not model GPU idle.** Measured on the course throughput widget: across 648 swept configurations
only **1** produced any loader wait, peaking at 2.6% idle, because its storage model is ~100×
under-scaled (a 1.05M-token step at 2 bytes/token is 2.48 MB against a bandwidth slider starting at
200 MB/s). Rather than reproduce an unreachable metric we measure what this system actually does:
loader time vs step time on real data.

## 11. Prior art — what we take, and where we differ

The lesson's own References name the systems a reviewer will compare this against. Engaging with them
is not decoration; each one solved a specific part of this problem and the design should say which.

| system | what it solved | what we take | where we differ |
|---|---|---|---|
| **Megatron-Core** GPT dataset | `document_index` / `sample_index` / `shuffle_index` — a planned sample lookup built once, so sample *k* of the run is computable without iterating | the three-index idea: the plan is a pure function `step → sample ids`, not a stateful iterator | Megatron's shuffle is a permutation over a fixed epoch. We have **no epoch** (S5 spends 3T tokens with lanes at 0.21–3.31 epochs), so our index is per-stage and lane-quota-driven, and repetition is an explicit per-lane budget rather than an epoch count |
| **Mosaic StreamingDataset** | mid-epoch deterministic resumption; the canonical statement that resumption must survive a changed worker count | the requirement itself, and that it is tested rather than asserted | Streaming reshards a fixed sample space across nodes. Our stream is generated by a *scheduler* (quotas + floors + OPUS), so we cannot recover position by arithmetic alone — we recover it from the ledger, and prove equivalence |
| **NVIDIA NeMo Curator** | exact / fuzzy / semantic dedup at corpus scale | dedup is a *pre-shard admission* property recorded in the manifest, not a training-time filter | at this corpus size exact + shingle-Jaccard is honest; MinHash-LSH is the scale answer and we say so rather than implementing a toy MinHash and implying it scales |
| **WebDataset** tar shards | sequential-read sharded format for samples made of multiple files | shard-as-immutable-object, and the agentic lane's multi-part records | we store token arrays + a sidecar manifest rather than tars; tars matter for throughput at scale, which we do not measure credibly here |
| **Iceberg / Delta / LakeFS** | transaction log, snapshot isolation, time travel, branching | the consumption ledger *is* a transaction log: append-only, hash-chained, with named branches and a divergence point | those systems give ACID over a table; we need ordering and exactly-once delivery over a *stream*, so our log commits to delivery position rather than table state |

**The one place we claim something the references do not give us:** Megatron and Streaming both make
the stream reproducible *given the same configuration*. Neither is designed to reproduce a stream when
the parallelism changes, because their position is expressed in per-rank terms. S5's plan makes that a
real event — the run is 45–60 days, the machine shape will change — so we express position in
**global sample-sequence terms** and rederive the rank partition on load. §12 is the test.

## 12. The invariant that actually matters

Manifests, masks and evidence files are bookkeeping. Two properties are load-bearing, and a reviewer
who has built one of these will go straight at them:

**(a) Exactly-once, order-preserving delivery across a restart with a different world size.**
The course asks only that resume produces the expected next batch on the same configuration. The
production event is: the run crashes on 4 ranks and comes back on 8, because that is the machine that
was free. A loader whose position is "rank *r* has consumed *n* samples" duplicates or drops samples
here, silently, and the loss curve will not tell you.

Our position is a single global integer, the **ledger offset**, over a canonical global ordering. Rank
assignment is a pure function `(global_index, world_size) → rank` applied at load time. The test
(`test_resume_across_world_size`) crashes at world size 4 and resumes at world size 8, then asserts on
the *union* of all ranks' samples:
- no sample id appears twice (exactly-once),
- no planned sample id is missing (no gaps),
- concatenated global order is identical to the uninterrupted run.

**(b) Stream identity under replay, verified by hash, not by inspection.**
Replay re-serves a historical interval and the batch content hashes must equal the originals. The
negative control is the point: replaying a *forked* branch must produce different hashes. A replay
check that cannot fail is not a check.

Everything else in this system exists to make these two provable.

## 13. Determinism: scope, stated precisely

Overclaiming determinism is the fastest way to lose a reviewer, so the boundary is explicit.

**Guaranteed exactly:** the *data stream*. Sample selection, ordering, packing, masks, position ids and
all content hashes are pure functions of `(plan, corpus, tokenizer, run_seed, global_step)`. These are
integer and byte operations. They are invariant to worker count, world size, machine and wall-clock.

**Not claimed:** bitwise-identical *model* state across hardware. Floating-point reduction order varies
with thread count, BLAS kernel and accelerator, so loss values can differ in the last bits. Within one
machine and one thread count this implementation is bitwise reproducible (`float64`, single-threaded,
fixed reduction order) and the demo asserts it — but it is a property of this configuration, not a
portable guarantee, and it is not what the ledger relies on.

**This distinction is the reason the ledger stores served sample ids rather than the RNG seed.** Data
determinism is achievable and we prove it; model determinism is not portable, so replay is defined on
the data stream and the model is restored from a checkpoint rather than recomputed.

## 14. Evidence discipline

**Every PASS names its artifact.** `evidence.json` contains no bare booleans. Each requirement records
the values compared, the artifact path and the record ids read. Not "crash recovery PASS" but
"expected `batch_hash=ab12…` at ledger offset 41; resumed batch produced `ab12…`;
`ledgers/consumption.jsonl#41`".

**Every invariant has a negative control that must fail.** A green suite is consistent with the check
being vacuous — which is exactly what happened during the S5 widget teardown, where a toggle-polarity
bug made a state check silently false-always and inverted a whole result set. `tds/controls.py` breaks
each invariant and asserts the verifier catches it: mutate a merge rule (tokenizer hash must mismatch),
flip a token byte (content hash must mismatch), drop `pii_screen_status` (must block), inject a test
shard (must block + emit a rejection event), mark a pad token loss-bearing (validator must error), let
attention cross a document boundary (isolation check must fail), disable floor injection (Indic must
fall below 12%), edit a historical ledger entry (chain must break at that link), resume from a wrong
offset (next-batch must mismatch), replay a forked branch (hashes must differ). **A control that passes
when it should fail is itself reported as a failure.**

**Nothing is claimed beyond its measurement.** Throughput here is a toy corpus on a CPU model in
`float64`. It is reported as this system's loader-time vs step-time ratio and packing utilization —
quantities that are real at any scale — and explicitly not as a tokens/sec prediction for 120B on
H100s. The 3T feasibility verdict is arithmetic over S5's supply table with no tokens trained; the demo
run trained real tokens at demo scale. The two are reported separately and never combined.

## 15. Accepted decisions

| # | decision | reverses if |
|---|---|---|
| **D1** | Stage-scoped budgets: lane demand computed against the stage window, never the run total | S5 restates shares as whole-run quantities |
| **D2** | All four S4 fields are hard admission blockers | the course clarifies that PII/dedup are advisory by intent |
| **D3** | `content_hash` = SHA-256 over shard token bytes | never |
| **D4** | Per-step derived RNG, no stateful global generator | a measured cost makes it impractical at scale |
| **D5** | Ledger authoritative for replay; scheduler not re-run | replay must validate scheduler changes, which is a different experiment (audit mode) |
| **D6** | Protected floors injected post-selection, invisible to OPUS | S5's 0/60 starvation measurement is overturned |
| **D7** | Firewall outranks protected floor | never |
| **D8** | Anneal reserve unspendable before the anneal stage | S5's T3 shows the reserve earns nothing |
| **D9** | Real per-token loss from a trained model, not synthetic numbers | none — the learning ledger is meaningless otherwise |
| **D10** | The anneal reserve is enforced by an OPUS `deferred` decision record, not by a silent skip inside the lane pool | never — an unrecorded withholding is unauditable by construction |
| **D11** | The demo run ends where the plan ends: step count is bounded by `tokens_consumed < run_tokens`, not by a constant | the plan gains an explicit over-run allowance |
| **D12** | Documents with a run of 32+ repeated n-grams (any 1–13 token span) are excluded at curation, and masked again in the trainer | OLMo 2's measured association between repeated n-grams and spikes fails to replicate |
| **D13** | Exact 13-gram containment against registered eval text is an independent firewall gate, not a tighter Jaccard threshold | a similarity measure is shown to catch quoted-passage contamination, which it structurally cannot |
| **D14** | Evidence-backed plan changes are *reported* in `v6_recommendations.json`, never silently applied | the plan becomes ours to change rather than S5's to execute |
| **D15** | A frozen reference model scores every traced token; the ledger records excess loss, not only loss | a cheaper signal is shown to separate learnable-hard from irreducible-hard |
| **D16** | A loss spike is a `run_event`; a shard exposed in a spike window has its verdict withheld until PaLM's ablation resolves it | never — a verdict from coincidence is a permanent wrong label |
| **D17** | The repetition ceiling is per-lane `r`; the *level* stays at 4.0 | the mixture scaling law is validated at our model scale (~30B), which it currently is not |
| **D18** | Feasibility reports the dedup break-even, not just a binary ceiling verdict | S5 measures its sources' dedup rates, at which point real supply replaces the upper bound |
| **D19** | The model applies the segment mask inside attention; isolation is proved by perturbation, not inspection | never — a mask that cannot leak is not evidence that the mask works |
| **D20** | `restore()` verifies checkpoint integrity and raises; six corruption flavours are induced and must be caught | never — a checkpoint that loads while wrong is worse than one that fails |
| **D21** | Every invariant's control must break the MECHANISM and observe the check fail — asserting non-vacuity in prose is not evidence | never |
| **D22** | The candidate pool is oversampled 2x so top-k selection has a surplus to reject from | the pool becomes large enough that scoring cost dominates |
| **D23** | Every reported aggregate is scoped to one branch with superseded batches excluded | never — mixing branches silently double-counts crash-orphaned steps |
| **D24** | The crash destroys in-memory state and resume reloads from disk, verified by a subprocess | never — an in-memory "crash" demonstrates nothing |
| **D25** | Position ids index a learned positional embedding inside `forward_backward` — they are a model input, not ledger bookkeeping | never — an array the arithmetic never reads cannot be validated except against a restatement of the rule that built it |
| **D26** | Selection is a bounded competition resolved in rank order: floor pass, then merit fill, capped at `target_batch`, with a sample admitted at most once per step | never — a duplicate in one batch trains identical tokens twice and records one exposure as two |
| **D27** | The eval-leak check is a content scan (exact hash + 13-gram containment) over the served text, implemented independently of the admission gate | never — registry membership is unsatisfiable once non-train splits are dropped upstream, and a predicate that cannot fail is not a check |
| **D28** | Two rejection rates are reported and never conflated: per-step selection pressure (floored by the oversample factor) and supply rejection (distinct samples never admitted, the one `need = share/(1−r)` uses) | never — the conflation drove a graded feasibility conclusion off a factor we chose ourselves |
| **D29** | Audit operations (`replay_interval`, `spike.ablate`) save and restore model state, planner delivery and the OPUS proxy — an audit that moves the run it measures is not an audit | never |
| **D30** | Shard token bytes ship with the bundle so `content_hash` is recomputable without re-running the build | never — a hash without its preimage is unverifiable |
| **D31** | Every load-bearing README figure is derived from `submission_artifacts/` and guarded by a drift test | never — in a submission whose thesis is reproducibility, a stale number is the most expensive error available |
