# The Data Ledger

ERA V5 Session 6. The accounting system that sits between raw documents and the training
loop — recording both what goes into the model and what comes back out.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_demo.py
.venv/bin/python -m pytest tests/ -q      # 38 tests, ~4s
```

One command. ~64 seconds. No API key, no network, no downloaded dataset. It runs the full
path — documents → tokenized shards → manifests → mixture schedule → packing → batches →
training → consumption ledger → learning ledger → checkpoint → crash → resume → replay →
audit — and writes the complete submission bundle.

**Last run: 14/14 requirements passed, 39 log checks PASS, 0 FAIL.**

```
python run_demo.py --fast           # ~24s, smaller token budget
python run_demo.py --with-api       # additionally annotate OPUS accepts with Claude
python run_demo.py --keep-weights   # keep checkpoint .pt blobs
```

## The submission bundle

```
submission_artifacts/
  run.log                   every event, its verdict, and the artifact that proves it
  evidence.json             machine-readable: each requirement, its predicate, its result
  evidence.md               the REQUIREMENT / RESULT / EVIDENCE table
  performance.json          throughput, packing utilisation, loss-bearing tokens/s
  run_summary.json          hashes, phase timings, model config
  execution_log.json        the run.log events, structured
  manifests/
    tokenizer.json          the frozen tokenizer, verified against every shard
    shard_index.json        registry hash, immutability proof, reload verification
    shards/*.json           one manifest per shard (122 of them)
  ledgers/
    consumption_ledger.json what went out, per step, with cursors either side
    learning_ledger.json    what came back, attributed to shard/doc/lane/band/role
    mixture_compliance.json planned versus actual shares, floors, epoch ceiling
    opus.json               every candidate decision, with reason codes
    opus_trail.jsonl        the same, one decision per line, rejections included
    crash_resume.json       the next-batch-exact proof
    replay.json             batch ids, token spans and hashes, original versus replay
    fork.json, audit.json, masks.json, packing.json, firewall.json, plan.json, corpus.json
    v6_shard_advisory.json  what to place differently next run
  checkpoints/
    checkpoint_index.json   ledger offsets, cursors, binding hashes, corruption drill
    step-NNNNNN/            meta.json + rng.json per checkpoint
```

`EVIDENCE.md` at the repo root is a long-form companion to `evidence.md`, generated from the
same artifacts so the two cannot disagree.

---

## The idea

A data loader answers *what should I feed the model next*. A data **ledger** answers that
and three more questions the loader never has to:

- *What did the model see at step 431?* — months later, without re-running the sampler.
- *What did it learn from it?* — per token, attributed back to the document that produced it.
- *Would I do it again?* — which shards earned their place, and which spent budget.

The second and third are the ones that don't exist in most training stacks. Training
normally keeps the scalar mean loss and throws away the per-token tensor it was computed
from. Keeping that tensor, and attributing it back through the packing plan to the document,
shard, lane and role that produced each token, is the artifact the session calls the one
nobody publishes. It is `submission_artifacts/ledgers/learning_ledger.json`.

## Architecture

Data flows down. Every arrow is a place something can be refused.

```
corpus.py        7 lanes of synthetic documents + planted pathologies
   |             (PII, restricted licence, exact dup, near dup, contamination canary)
   v
documents.py     Document / Segment / Provenance / Quality
   |             validate(): no restricted licences, no un-scrubbed PII, no untrainable docs
   v
tokenization.py  BPE fitted on the TRAIN SPLIT ONLY, then frozen and hashed
   |             a tokenizer that has seen held-out text has already leaked it
   v
firewall.py      admit -> exact dedup -> near dedup (MinHash) -> n-gram contamination scan
   |
   v
shards.py        immutable content-addressed shards; the id embeds the content hash
registry.py      cut per (lane, split, difficulty band); built twice and compared
   |
   +--> packing.py    6 policies; each lane uses the one its structure requires
   +--> masks.py      loss mask / attention mask / position ids, verified against each other
   |
   v
plan.py          the S5 curriculum in real units: 2,910B main + 90B anneal
mixture.py       compiled once into per-step lane allocations, then hashed
   |             seam blending, protected floors, 4-epoch ceiling, band gates
   v
stream.py        DataStream: cursors per (lane, stage, window); re-derives trainability
   |             from each manifest independently of any build-time decision
   v
model.py         TinyLM: pre-norm decoder, explicit (B,L,L) bool attention mask
trainer.py       AdamW + cosine LR; writes BOTH ledgers
   |
   +--> ConsumptionLedger   what went out: shards, documents, lane tokens, cursors
   +--> LearningLedger      what came back: per-token loss by shard/doc/lane/band/role
   |
   v
opus.py          golden-proxy gradient selector: accept ~25%, protected-flow override
checkpoint.py    atomic save of weights + optimizer + cursors + RNG + ledger offsets
replay.py        crash / resume / replay-from-ledger / fork / audit
advisory.py      the v6 shard re-placement recommendation
perf.py          throughput, packing utilisation, loss-bearing tokens/s
events.py        run.log -- structured events whose verdicts are computed, not passed in
evidence.py      evidence.json / evidence.md -- one predicate per requirement, run
                 against the artifacts on disk
report.py        EVIDENCE.md, the long-form companion
```

`determinism.py` sits under all of it: canonical JSON, content hashing, and counter-based
seed derivation. Nothing calls `random.seed()` and hopes.

---

## Design decisions

### 1. Seeds are derived from content, never carried

`derive_seed(*key)` hashes a canonical encoding of its arguments. Every shuffle, every pack
plan, every sampling decision derives its own generator from `(run_id, purpose, lane,
shard_id, epoch)`. Python's `random.seed(42)` is only stable inside one live interpreter;
a counter-based scheme is stable across processes, machines, and the eighteen months
between a run and the question about it.

The practical consequence is that replay does not need to store the RNG state of the
sampler. It stores the cursor, and the cursor plus the key regenerates the stream.

### 2. The shard id *is* the content hash

`shard-indic-train-0002-d073e546cd`. The trailing hash is derived from the tokens. A shard
whose content changed cannot keep its name, so "same code, different day, same shard" is
checkable rather than assertable — `build_registry(verify_immutability=True)` builds every
shard twice and compares. A single `set()` iteration or `time.time()` anywhere in the path
breaks it, silently, and this is the only thing that catches that.

### 3. A manifest is a precondition, not a description

`ShardManifest.is_trainable()` returns `(bool, reasons)`. Missing dedup status, missing
contamination status, a tokenizer hash that differs from the loader's, an unscrubbed PII
flag, a non-train split — any of these and the shard does not enter the stream. The loader
re-derives this independently at load time (`DataStream._admit_shard`) and deliberately
consults no decision made at build time. If the build-side firewall had a bug, the load-side
check is the one that has to catch it.

### 4. Packing policy is a property of the data, not of the run

Six policies. Which one a lane uses follows from what its documents are:

| lane | policy | why |
|---|---|---|
| general_web, indic | greedy | prose survives being split; padding is the only cost |
| code, stem | best_fit | fewer holes; a file split mid-function is still valid training data |
| reasoning, agentic | structure_preserving | a chain of thought cut in half teaches half a chain |
| long_context | long_context | one document per sequence, which is the point of the lane |

`ledgers/packing.json` runs *every* policy against the largest shard of every lane, at that
lane's own window. `concat_chop` wins on efficiency almost everywhere — it always will,
because it truncates. On the agentic shard it reaches 87% efficiency by cutting nine
documents mid-structure. That is what the chosen policy is refusing to buy.

The corollary: policies that refuse to split an atomic document report **zero sequences**
on a shard whose documents all exceed the window. That is the policy working, not failing.

### 5. Attention and position policy are per-lane, and both are logged

| policy | lanes | rationale |
|---|---|---|
| `causal_eos_only` | web, indic, long_context | the lecture's position; a leaked neighbour costs wasted signal |
| `causal_block_diagonal` | code, stem, reasoning, agentic | a packed neighbour here is *actively misleading* |
| `continuous` positions | web, indic, long_context | matches EOS-only attention |
| `reset_per_document` | code, stem, reasoning, agentic | matches block-diagonal attention |

`verify_masks()` asserts that a block-diagonal mask actually blocks — it checks the
materialized `(L, L)` boolean, rather than trusting the label. Position embeddings are
learned and absolute *deliberately*: it makes reset-per-document a one-line change and makes
the difference between the two policies visible in the arithmetic.

### 6. EOS is supervised; BOS is not emitted

EOS is the document boundary and the model has to predict it, so it carries loss. Its
perplexity is tracked as its own line in the learning ledger, because a model whose EOS
perplexity does not fall will bleed one document into the next at inference — and that is
invisible in the mean loss, where 550 EOS tokens sit among 350,000 others.

BOS is not emitted at all. The session's phrase is "waste of space": with EOS as a boundary,
a start marker carries no information the position ids don't already have.

### 7. Lane allocation carries error across steps

This is the least obvious decision and the one with the largest consequence.

With 16 sequences per step, a 2%-target lane wants 0.32 sequences. Largest-remainder
rounding floors that to 0 and loses every tie-break, so the agentic lane realized **0.2%
against a 2.0% protected floor** in the first version. A protected floor that integer
rounding can delete is decorative.

`_allocate` keeps a per-lane fractional carry across steps, so the deficit accumulates until
it buys a whole sequence. Agentic now realizes 1.63% of 2.0% at demo scale, limited by
supply rather than by arithmetic — and it is the lane that hits the epoch ceiling first,
which is exactly what S5 decision 3 predicted.

### 8. The epoch ceiling is enforced per step, not checked at the end

The 4-epoch repetition ceiling (arXiv:2305.16264) is clamped inside the allocation loop and
the freed sequences are redistributed to lanes with headroom. Checked afterwards instead,
a lane at 3.9 epochs is allowed one more full step and lands at 4.5. The ceiling has to bind
before the allocation, not after it.

### 9. Both ledgers, or neither

The consumption ledger records the cursor state on **both sides** of every step. `after` is
the resume point; `before` is what replay needs to regenerate the batch. Recording only one
makes the ledger resumable *or* replayable, not both.

Each row also carries a `batch_hash` over the actual token content. Replay checks the
regenerated batch against it — reproducing the shard list but not the tokens would reproduce
the bookkeeping and not the training.

### 10. Resume must repair the ledger, not just restore the weights

The failure this prevents only exists on disk. The ledger is appended to every step; the
weights are only checkpointed periodically. After a crash the ledger file contains rows for
steps whose weights were discarded, and a resume that keeps them claims the model consumed
tokens it has never seen — in the one artifact whose entire purpose is to be believed.

`truncate_ledgers` cuts the rows back to the checkpoint's recorded offsets and recomputes
the totals from the survivors rather than trusting the running counters. The drill in
`ledgers/crash_resume.json` shows 4 orphan rows repaired, weights bitwise identical, and the
final ledger matching the uninterrupted run row for row.

### 11. Checkpoints are bound to their data

Every checkpoint carries the tokenizer hash, registry hash, schedule hash and source-tree
hash. Loading one against a rebuilt shard set raises `CheckpointBindingError` rather than
resuming into a corpus the weights have never seen. The fork path passes
`strict_binding=False` on purpose — the schedule hash *is* what differs between branches —
and reports the drift so the difference stays visible.

### 12. The OPUS proxy is rebuilt every round, from held-out data

Two things that look like details and aren't.

The golden set comes from the **validation split**, never from the training batch. It
defines what the model should become good at; taking it from the data being consumed would
make the proxy a map of the training stream rather than of the target capability.

And the proxy is re-derived from that fixed golden set **at every round**, against current
weights. A proxy computed once goes stale as the model moves, and its decaying alignment
reads exactly like a degrading data pool — the same number, the opposite conclusion. The
first version of this made that mistake; `proxy_hashes` in `ledgers/opus.json` now differ
per round.

Relatedly, selection drift is measured over **threshold accepts only**. Protected-flow
overrides are admitted by policy regardless of score, and since their alignment can be
negative, folding them in manufactures a degradation signal out of the floor policy.

### 13. Rejections are kept

`ledgers/opus_trail.jsonl` holds every decision including the rejected ones, with a reason code
and the alignment that produced it. "The selector considered the value to be low" is a
judgement, and judgements are auditable. It is also the only record that a given sample was
ever considered.

### 14. The advisory recommends; it does not act

`ledgers/v6_shard_advisory.json` applies the 1.2 shard-perplexity reuse cutoff and emits four
distinct actions. A high ratio is ambiguous between *informative* and *broken*, so the
advisory separates them using loss movement across exposures — a shard that stays hard while
being re-read is usually damaged, not rich — and a shard already at the epoch ceiling gets
"acquire more like this", because more epochs are not available to it.

---

## What the run found

From the artifacts of the last full run (`submission_artifacts/run_summary.json`):

- **Indic fertility 7.78 tokens/word against English 2.06** (Telugu 8.20, Kannada 8.90) —
  a ~3.8× budget penalty on the same content, reproducing the session's Telugu warning at
  demo scale. The 1,024-token demo vocabulary makes this worse than S5's 2.0 target assumes;
  the direction is the finding, not the magnitude.
- **`tool_call` perplexity 68.7 against `response` 12.5** — on agentic data the model is
  learning to narrate around the action faster than it is learning to act. A lane-level
  average cannot show this; only the per-role attribution can.
- **EOS perplexity 1284 → 183** over the run, as the model learns where documents end.
- **Agentic is the first lane to approach the epoch ceiling**, exactly as S5 decision 3
  predicted when it set the 2% floor and funded synthesis.
- **The band-perplexity table runs backwards** (B0 1018, B5 11.4) — a curriculum confound,
  not a difficulty finding, and labelled as such in the artifact rather than presented as
  a result.

## Repository layout

```
run_demo.py               the single entry point
ledger/                   the system (see architecture above)
tests/                    38 tests for the invariants the demo asserts
docs/s5-inheritance.md    what S5 decided, and where each decision is enforced here
docs/band-gating.md       the difficulty-band mechanism S5 deferred to S6
submission_artifacts/     the generated bundle (committed)
artifacts/                shard payloads and the tokenizer (gitignored; rebuilt in ~2s)
EVIDENCE.md               long-form companion, generated from the bundle
```

## Limits

The corpus is synthetic and small (1,634 documents, ~326k shard tokens). The model is 1.05M
parameters. Neither is a claim about scale — they are sized so the ledger's behaviour is
visible in one run on a laptop.

What is genuinely demonstrated: the accounting is exact, the invariants are checked rather
than asserted, and every number in `EVIDENCE.md` traces to an artifact. What is *not*
demonstrated: that any of these numbers hold on a real corpus. The OPUS drift verdict in
particular is computed over four rounds and reports its own confidence as low.
