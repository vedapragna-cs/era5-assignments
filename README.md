# The Data Ledger

**Repository:** https://github.com/vedapragna-cs/era-v5-s6-data-ledger

When you train a language model, something has to decide which text to feed it next, in what
order, and in what proportions. That component is usually called a *data loader*, and it is
usually write-only: it makes thousands of decisions per run and keeps a record of almost
none of them.

This project builds the thing that should sit there instead — a **data ledger**. It does
everything a loader does, and it also keeps a complete, verifiable record of what it sent to
the model and what came back. Six months after a training run finishes, you can ask it *what
exactly did step 431 train on?* and get an exact answer, reconstructed from the record rather
than by guessing or re-running.

It is a working system, not a design document: it builds a corpus, trains a real (small)
transformer on it, deliberately crashes, resumes, replays history, forks an alternative
timeline, audits itself, and writes out the evidence for all of it.

---

## Quickstart

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python run_demo.py              # the full demonstration, ~64 seconds
.venv/bin/python -m pytest tests/ -q      # 40 tests, ~4 seconds
```

One command, start to finish. **No API key, no network access, no dataset download** — the
corpus is generated from a seed, so the whole thing runs offline on a laptop.

**Last run: 14/14 requirements passed, 39 checks PASS, 0 FAIL.**

Variants, none of them required:

```bash
python run_demo.py --fast           # ~24s, smaller token budget
python run_demo.py --with-api       # also ask Claude to rate selected samples (needs a key)
python run_demo.py --keep-weights   # keep the checkpoint weight files instead of deleting them
```

---

## The problem it solves

A training run makes an enormous number of small decisions, and afterwards nobody can
reconstruct them. Four questions come up repeatedly and are normally unanswerable:

| Question | Why it's normally unanswerable |
|---|---|
| *What did the model see at step 431?* | The loader shuffled with a random seed and kept no record. |
| *What did it learn from that?* | Training keeps one average loss number per step and discards the detail behind it. |
| *Did any test data leak into training?* | Nothing checked at the moment data was actually consumed. |
| *Did the crash on Tuesday corrupt anything?* | The restart "worked", so nobody looked closer. |

The ledger answers all four, and the second one is the interesting case. Training computes a
loss value for *every single token*, averages them into one number, and throws the rest away.
This system keeps that detail and traces each token's loss back to the document, file, data
category, and text role it came from. That is what makes it possible to say things like
*"the model is learning to describe tool calls but not to make them"* — which you cannot see
in an average.

---

## Vocabulary

The code uses a handful of terms consistently. They are worth five minutes because
everything below depends on them.

| Term | What it means here |
|---|---|
| **Document** | One piece of source text — a web page, a source file, a conversation. |
| **Token** | Text chopped into model-sized pieces. "Hello world" might be 3 tokens. |
| **Tokenizer** | The thing that does that chopping. Must never change mid-project. |
| **Shard** | A bundle of documents converted to tokens and frozen into an immutable file. The unit the trainer reads. |
| **Manifest** | A shard's passport: what's inside, which tokenizer made it, whether it passed safety checks. No valid manifest, no training. |
| **Lane** | A data category with its own budget. Here: `general_web`, `code`, `indic` (Indian languages), `stem`, `reasoning`, `agentic` (tool-use traces), `long_context`. |
| **Mixture** | How much of each lane the model sees. E.g. 34% web, 24% code, 17% Indic. |
| **Curriculum stage** | Training runs in phases that shift the mixture — easy and general early, harder and more specialised later. |
| **Seam** | The transition between two stages. Mixtures blend gradually rather than switching abruptly. |
| **Difficulty band** | A per-document difficulty label, B0 (easiest) to B5 (hardest). Early stages only draw from easy bands. |
| **Packing** | Fitting variable-length documents into fixed-size training windows. Like packing luggage; the strategy matters. |
| **Loss mask** | Which token positions the model is graded on. A question is context; only the answer is graded. |
| **Attention mask** | Which tokens each token is allowed to look at. Stops two unrelated documents packed together from bleeding into each other. |
| **Epoch ceiling** | The cap on how many times the same data may be reused — 4×, [per research](https://arxiv.org/abs/2305.16264) showing returns collapse beyond that. |
| **Protected floor** | A guaranteed minimum share for a lane that would otherwise be squeezed to zero. |
| **OPUS** | A sample-selection method: cheaply estimate whether a sample will teach the model anything useful, and skip it if not. Accepts ~25%. |
| **Perplexity** | Loss expressed as "how many options was the model effectively choosing between". Lower is better. |
| **Consumption ledger** | The record of what was sent *to* the model. |
| **Learning ledger** | The record of what came *back* — per-token loss, traced to its source. |

---

## What the demonstration actually does

`run_demo.py` runs this end to end and logs a verdict at each step:

1. **Generates a corpus** — 1,634 synthetic documents across 7 lanes, with deliberate
   problems planted in it: a document with unredacted personal data, one under a licence
   that forbids training, an exact duplicate pair, and a "canary" that secretly copies text
   from the test set.
2. **Trains and freezes a tokenizer** — fitted on the *training split only*, then hashed. A
   tokenizer fitted on held-out text has already seen it, which is a subtle form of cheating.
3. **Runs the safety firewall** — refuses the unsafe documents, removes duplicates, and scans
   for overlap with evaluation data. It catches all four planted problems.
4. **Builds 122 immutable shards** with manifests, then **builds every one a second time** and
   compares hashes, to prove the process is genuinely reproducible.
5. **Compiles the training schedule** — 102 steps with a shifting lane mixture, stage
   transitions, protected floors and reuse ceilings, all computed up front and hashed.
6. **Trains a small transformer** (1.05M parameters), writing both ledgers as it goes. Loss
   falls 6.94 → 2.35.
7. **Runs OPUS selection** at intervals, recording every accept, reject, deferral and
   protected-lane override *with the reason*.
8. **Saves checkpoints**, then deliberately corrupts copies of them to prove the corruption
   is detected rather than silently loaded.
9. **Simulates a crash**, resumes from a checkpoint, and proves the very next batch is
   *exactly* the batch the uninterrupted run would have produced.
10. **Replays history** — reconstructs batches from earlier in the run and verifies they
    match the originals down to the token.
11. **Forks** from a checkpoint into two alternative futures with different data mixtures,
    and compares them.
12. **Audits** the whole run, then writes the evidence bundle.

---

## What it produces

```
submission_artifacts/
  run.log                   every event, its PASS/FAIL verdict, and the file that proves it
  evidence.json             machine-readable: each requirement, its test, its result
  evidence.md               the same as a human-readable table
  performance.json          throughput and packing efficiency
  run_summary.json          hashes, timings, model configuration
  execution_log.json        run.log again, structured

  manifests/
    tokenizer.json          the frozen tokenizer, checked against every shard
    shard_index.json        registry hash, reproducibility proof, reload verification
    shards/*.json           one manifest per shard (122 files)

  ledgers/
    consumption_ledger.json what was sent to the model, step by step
    learning_ledger.json    what came back, traced to shard / document / lane / role
    mixture_compliance.json planned vs. actual data proportions
    opus.json               every selection decision with its reason
    opus_trail.jsonl        the same, one decision per line, rejections included
    crash_resume.json       the proof that resume produced the exact expected batch
    replay.json             original vs. replayed batch ids, spans and hashes
    fork.json, audit.json, masks.json, packing.json, firewall.json, plan.json, corpus.json
    v6_shard_advisory.json  recommendations for what to do differently next time

  checkpoints/
    checkpoint_index.json   ledger positions, cursors, binding hashes, corruption drill
    step-NNNNNN/            per-checkpoint metadata
```

Everything here is generated by the run. Nothing is written by hand — `evidence.md` is
produced by evaluating a test against each artifact on disk, so it cannot drift out of sync
with the run it describes. [`EVIDENCE.md`](EVIDENCE.md) at the repository root is a longer
narrative version built from the same files.

---

## How the code is organised

Data flows top to bottom. Every arrow is a point where something can be rejected.

```
corpus.py        7 lanes of synthetic documents + deliberately planted problems
   |
documents.py     the Document type; refuses restricted licences and unredacted personal data
   |
tokenization.py  BPE tokenizer fitted on the TRAINING SPLIT ONLY, then frozen and hashed
   |
firewall.py      admit -> remove exact duplicates -> remove near-duplicates -> scan for
   |             overlap with evaluation data
   v
shards.py        immutable shards whose filename contains a hash of their own contents
registry.py      shards split by (lane, train/test split, difficulty band); built twice
   |             and compared to prove reproducibility
   |
   +--> packing.py    6 strategies for fitting documents into fixed windows
   +--> masks.py      loss / attention / position masks, verified against each other
   |
plan.py          the target data mixture, in real units (3 trillion tokens)
mixture.py       compiled into a per-step schedule, then hashed: stage blending,
   |             protected floors, reuse ceilings, difficulty gates
   v
stream.py        the loader; tracks its exact position, and independently re-checks every
   |             shard's manifest rather than trusting build-time decisions
   v
model.py         a small transformer
trainer.py       the training loop; writes BOTH ledgers
   |
   +--> ConsumptionLedger   what went out
   +--> LearningLedger      what came back, per token
   |
   v
opus.py          sample selection, with reasons recorded for every decision
checkpoint.py    crash-safe saving of weights, optimizer, loader position and ledger offsets
replay.py        crash / resume / replay / fork / audit
advisory.py      recommendations for the next run
perf.py          throughput and packing efficiency
events.py        run.log — structured events whose verdicts are computed, never asserted
evidence.py      evidence.json and evidence.md — one test per requirement
report.py        the long-form EVIDENCE.md
```

Underneath all of it, `determinism.py` provides canonical JSON, content hashing, and
seed derivation. Nothing anywhere calls `random.seed()` and hopes for the best.

---

## Design decisions

Fourteen choices worth explaining. Each starts with the problem it addresses.

### Reproducibility

**1. Random seeds are derived from content, never stored.**
*Problem:* `random.seed(42)` gives the same sequence only within one running Python process.
Restart on a different machine and it can differ — so a replay can silently diverge.
*Solution:* every random decision derives its own generator by hashing a description of the
decision (`run_id`, purpose, lane, shard, epoch). Same inputs, same result, on any machine,
years later. Replay therefore never needs to store random state — just a position.

**2. A shard's filename contains a hash of its contents.**
E.g. `shard-indic-train-0002-d073e546cd`. A shard whose contents changed cannot keep its
name, so "same code, same data, next month" is *checkable* rather than merely claimed. The
build runs twice and compares. A single unordered `set()` iteration or stray timestamp
anywhere in the pipeline would break this — and nothing else would catch it.

**3. A manifest is a precondition, not a description.**
Missing duplicate-check status, missing contamination status, a tokenizer that doesn't match,
unredacted personal data, or a non-training split — any one and the shard never enters
training. The loader **re-derives this itself at load time** and deliberately ignores what
the build stage concluded. If the build-side check had a bug, the load-side check is the one
that has to catch it.

### Handling different kinds of data differently

**4. Packing strategy follows from what the data *is*.**

| Lane | Strategy | Why |
|---|---|---|
| `general_web`, `indic` | greedy | prose survives being split mid-way; wasted space is the only cost |
| `code`, `stem` | best-fit | minimise gaps; a source file split mid-function is still useful |
| `reasoning`, `agentic` | structure-preserving | a chain of reasoning cut in half teaches half a chain |
| `long_context` | one document per window, short ones turned away | a 200-token document in a 1024-token window wastes 80% of an expensive step and teaches nothing about long-range retrieval |

`ledgers/packing.json` runs *every* strategy against every lane so the trade-off is measured
rather than asserted. The truncating strategy always wins on raw efficiency — on the
tool-use lane it hits 87% by cutting nine documents mid-structure. That is precisely what the
chosen strategy is declining to buy.

A consequence worth knowing: a strategy that refuses to split documents reports **zero
sequences** when every document in a shard is longer than the window. That is the strategy
working correctly, not failing.

**5. Attention rules are also per-lane, and both choices are logged.**
When two documents share a training window, should the model be *allowed* to look across the
boundary?

- **Web, Indic, long-context:** allowed. The model learns not to, and the wasted signal is cheap.
- **Code, STEM, reasoning, tool-use:** blocked outright. Here a neighbouring document is
  actively misleading, and learning-not-to is too slow and too unreliable.

`verify_masks()` checks the actual mask matrix rather than trusting its label.

**6. The end-of-document marker is graded; the start marker doesn't exist.**
The model must learn where documents end, so the end marker carries loss and its perplexity
is tracked on its own line. If it doesn't improve, generated text will run one document into
the next — invisible in an average where ~550 end markers sit among 350,000 other tokens.
A start marker is omitted entirely: with an end marker present it carries no information the
position numbers don't already have.

### Keeping the mixture honest

**7. Fractional allocations carry over between steps.**
*This is the least obvious decision and had the largest consequence.* With 16 sequences per
step, the tool-use lane's 2% target works out to 0.32 sequences. Ordinary rounding takes that
to zero, every step, forever — so the lane came out at 0.2%, below even its 1% **protected
floor**. A floor that rounding can quietly delete is decoration. The allocator now carries the
fractional remainder forward until it accumulates into a whole sequence: that lane now lands
at 1.63%, comfortably above its floor and close to its 2% target.

Across all seven lanes the largest gap between planned and actual share is **0.68 percentage
points**, and both protected floors hold. The three lanes that fall short do so because
eligible data ran out, which the scheduler records explicitly in
`ledgers/mixture_compliance.json` rather than absorbing silently.

**8. The reuse ceiling is enforced during allocation, not checked afterwards.**
Checked at the end, a lane sitting at 3.9 epochs is granted one more full step and lands at
4.5 — over the limit, discovered too late. It is clamped inside the allocation loop instead,
and the freed capacity is redistributed to lanes with room. In this demo run no lane exceeds
2.1 epochs, so the clamp is not what limits it here. Two tests exercise it at a budget where
it *does* bind: one asserts six lanes press right up against the ceiling without crossing it,
the other disables the clamp at the same budget and confirms the run then overshoots — so
the first test cannot pass by accident.

### The two ledgers

**9. Record the loader's position on both sides of every step.**
The position *after* a step is where you resume from; the position *before* it is what you
need to regenerate that exact batch. Record only one and the ledger is resumable **or**
replayable, but not both. Each row also carries a hash of the actual tokens — reproducing the
file list but not the contents would reproduce the paperwork and not the training.

**10. Resuming after a crash must repair the ledger, not just restore the weights.**
This failure exists only on disk. The ledger is written every step; weights are saved
periodically. After a crash the ledger contains rows for steps whose weights were thrown
away — so a naive resume claims the model consumed data it has never seen, in the one file
whose entire purpose is to be trusted. The resume path truncates those rows and recomputes
totals from what survives. The drill in `ledgers/crash_resume.json` shows 4 orphan rows
repaired, weights identical bit for bit, and the final ledger matching the uninterrupted run
row for row.

**11. Checkpoints are bound to the data they were trained on.**
Each carries hashes of the tokenizer, the shard registry, the schedule and the source code.
Loading one against a rebuilt corpus raises an error instead of resuming into data the
weights have never seen. The fork feature deliberately relaxes this — a different schedule
*is* the experiment — and reports exactly what differs.

### Sample selection

**12. The selection reference is rebuilt every round, from held-out data.**
OPUS scores a candidate by comparing it against a reference map of "what the model currently
needs to learn". Two subtleties:

- That reference is built from the **validation split**, never from the training batch.
  Building it from data already being consumed would make it a map of the training stream
  rather than of the target capability.
- It is **rebuilt every round** against current weights. A reference computed once goes stale
  as the model moves, and its decaying scores look exactly like a degrading data pool — same
  number, opposite conclusion. An early version made this mistake.

Related: selection quality is measured over score-based accepts only. Protected lanes are
admitted regardless of score, so including them would manufacture a false decline out of the
protection policy.

**13. Rejections are kept.**
`ledgers/opus_trail.jsonl` records every decision, including everything thrown away, with a
reason code. "The selector judged this low value" is a *judgement*, and judgements should be
auditable. It is also the only surviving evidence that a given sample was ever considered.

**14. The advisory recommends; it never acts.**
`ledgers/v6_shard_advisory.json` flags shards the model still finds hard. But "hard" is
ambiguous between *rich* and *broken*, so the advisory distinguishes them by whether the loss
is still moving — a shard that stays hard while being re-read is usually damaged, not
informative — and issues one of four recommendations rather than changing anything itself.

---

## What the run found

Real results from the last full run, all traceable to files in `submission_artifacts/`:

- **Indian-language text costs ~3.8× more tokens than English** for the same content (7.78
  tokens per word vs 2.06; Telugu 8.20, Kannada 8.90). Since training budgets are counted in
  tokens, that is a direct tax on those languages. *The direction is the finding; the exact
  magnitude is inflated by the demo's small vocabulary.*
- **The model learns to describe tool use faster than to do it.** On tool-use data,
  perplexity on the actual tool calls is 68.7 against 12.5 on the surrounding prose. An
  average over that lane shows one number and hides this completely — it is visible only
  because loss is traced per text role.
- **End-of-document perplexity fell 1284 → 183**, so the model is genuinely learning where
  documents stop.
- **Scarce lanes survive only because they are protected.** The tool-use lane's 2% target
  rounds to zero sequences per step; without the carry-forward fix it collapsed to 0.2%,
  under its own 1% floor. With it, the run lands within 0.68 percentage points of plan on
  every lane and both protected floors hold. Three lanes still fall short — the scheduler
  records that eligible data ran out rather than hiding the gap.
- **The difficulty-band results run backwards** (easiest band worst, hardest band best).
  That is a side effect of the curriculum — easy material is only ever shown to an untrained
  model — not a statement about difficulty. It is labelled as such in the artifact rather
  than presented as a result.

---

## Repository layout

```
run_demo.py               the single entry point
ledger/                   the implementation
tests/                    40 tests covering the invariants
docs/s5-inheritance.md    the data-mixture plan, and where each part is enforced in code
docs/band-gating.md       how difficulty gating works, and why it happens at build time
submission_artifacts/     the generated output bundle (committed)
artifacts/                bulk token data (not committed; rebuilt in ~2 seconds)
EVIDENCE.md               long-form results write-up, generated from the bundle
```

---

## Limits

The corpus is synthetic and small (1,634 documents, ~326k tokens). The model is 1.05M
parameters — thousands of times smaller than a real one. Neither is a claim about scale;
both are sized so the ledger's behaviour is fully visible in a single run on a laptop.

**What is genuinely demonstrated:** the accounting is exact, the invariants are checked
rather than asserted, and every number in the evidence bundle traces to a file the run wrote.

**What is not:** that any of these *measurements* would hold on a real corpus. Where a result
is weakly supported, the artifact says so — the sample-selection trend, for instance, is
computed over four rounds and reports its own confidence as low.
