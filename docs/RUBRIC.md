# S6 rubric — 1,000 points, decomposed and mapped to evidence

The lesson states nine scoring areas and their weights. This decomposes each into concrete criteria,
names the artifact that proves each one, and tracks honest status. **Status is self-assessed and is
not a claim about the grade** — it is a checklist for what remains.

Grading is stated as three steps, and criterion design follows from step 3:

1. **Execute** — run the submitted command, regenerate `submission_artifacts/` from scratch.
2. **Verify evidence** — check `run.log`, `evidence.json`, `evidence.md` against the generated
   manifests, ledgers and reports.
3. **Inspect code** — confirm the evidence was *produced by the implementation*, not simulated or
   hardcoded.

Stated failure conditions, which are therefore the highest-value things to get right:

| stated failure | our guard |
|---|---|
| resume repeats or skips a batch → resume section fails | E2, `test_resume_exact_once`, plus the harder cross-world-size variant |
| replay produces different batch hashes → replay fails | `test_replay_hash_match`, with a fork negative control that must differ |
| evaluation data enters a loss-bearing batch → firewall fails | two-stage check: admission-time and per-batch, plus `test_eval_never_in_batch` |
| packing or throughput numbers cannot be reconstructed → no credit | every reported number recomputed from artifacts by `verify_evidence.py` |

Legend: ✅ done · 🔄 in progress · ⬜ not started

---

## 1. End-to-end execution — 150 pts

| # | criterion | pts | evidence | status |
|---|---|---:|---|---|
| 1.1 | One command runs the whole demonstration with no manual steps | 40 | `python run_demo.py` | ✅ |
| 1.2 | Runs offline from a clean clone; no network, no large download | 25 | vendored `tds/corpus.py`; optional `--fetch` only | ✅ |
| 1.3 | Full path executes in order: documents → shards → manifests → schedule → packing → batches → training → ledgers → checkpoint → crash → resume → replay → audit | 45 | `run.log` stage banners | ✅ |
| 1.4 | Deterministic: two runs produce identical artifact hashes | 25 | `evidence.json.determinism`, `test_two_runs_identical` | ✅ |
| 1.5 | Exits non-zero if any invariant fails (cannot "pass" silently) | 15 | `run_demo.py` exit code; forced-failure control | ✅ |

## 2. Shards, manifests and tokenizer integrity — 100 pts

| # | criterion | pts | evidence | status |
|---|---|---:|---|---|
| 2.1 | Tokenizer trained once, frozen, hashed; hash in every manifest | 20 | `manifests/tokenizer.json`, `tokenizer_hash` | ✅ |
| 2.2 | Tokenizer retrains bit-identically (deterministic tie-breaks) | 15 | `test_tokenizer_deterministic` | ✅ |
| 2.3 | `content_hash` is a function of shard token bytes | 15 | `shards.Shard.content_hash`; control: flip a byte | ✅ |
| 2.4 | Manifest carries all 14 stated fields | 15 | `manifests/shards/*.json` | ✅ |
| 2.5 | Admission gate blocks on any missing hard field | 20 | `admission.json`; control: drop `pii_screen_status` | ✅ |
| 2.6 | Lineage: `parent_shard_ids` present; modified shard = new id + new hash | 15 | `test_shard_immutability` | ✅ |

## 3. Packing, masks and batch correctness — 150 pts

| # | criterion | pts | evidence | status |
|---|---|---:|---|---|
| 3.1 | Six policies genuinely distinct (not aliased onto one search) | 30 | E3 table in `performance.json` | ✅ |
| 3.2 | Policy chosen per data type, justified | 15 | `plan/v5_plan.json.packing_policy` | ✅ |
| 3.3 | Loss mask correct: pad never loss-bearing; SFT/agentic roles per S5 | 35 | `validate_masks`, E4 yield ≈53% | ✅ |
| 3.4 | Attention mask isolates documents inside a pack (0 crossings) | 35 | `attention_crossings` == 0 for all packed sequences | ✅ |
| 3.5 | Position ids reset at document boundaries | 20 | `validate_masks`; policy `packed_reset_on_eos` | ✅ |
| 3.6 | Packing arithmetic sane at edges (no >100% utilization, no negative pad) | 15 | assertion in `pack`; control at ctx < longest doc | ✅ |

## 4. Mixture schedule, protected floors and OPUS — 150 pts

| # | criterion | pts | evidence | status |
|---|---|---:|---|---|
| 4.1 | S5 curriculum stages compiled into per-step lane quotas | 35 | `manifests/schedule.json` | ✅ |
| 4.2 | Stage-scoped budgets (demand vs the stage window, never the run total) | 20 | `schedule.json.stage_demand`; D1 | ✅ |
| 4.3 | Protected floors enforced and *measured* in the delivered stream | 35 | `evidence.json.mixture_compliance`; E1 | ✅ |
| 4.4 | OPUS accept / reject / defer / protected-floor override, all recorded | 30 | `ledgers/opus.jsonl` | ✅ |
| 4.5 | Firewall outranks the protected floor (override cannot rescue contamination) | 15 | `test_floor_cannot_rescue_eval` | ✅ |
| 4.6 | Supply feasibility at 3T incl. 4-epoch ceiling and rejection inflation | 15 | `manifests/feasibility.json` | ✅ |

## 5. Consumption and learning ledgers — 150 pts

| # | criterion | pts | evidence | status |
|---|---|---:|---|---|
| 5.1 | Consumption ledger records all 15 stated fields per batch | 35 | `ledgers/consumption.jsonl` | ✅ |
| 5.2 | Append-only and tamper-evident (hash chain) | 25 | `canon.HashChain.verify`; control: edit an entry | ✅ |
| 5.3 | Any global step reconstructs its packed samples, spans and shards | 25 | `audit.py --step N` | ✅ |
| 5.4 | Learning ledger: per-token loss and perplexity from a real model | 35 | `ledgers/learning_tokens.jsonl` | ✅ |
| 5.5 | Per-shard aggregates: mean loss, hot-token %, loss delta, gradient norm | 20 | `ledgers/learning_shards.jsonl` | ✅ |
| 5.6 | Usefulness classification feeding a V6 recommendation | 10 | `evidence.md` shard report card | ✅ |

## 6. Checkpoint, crash, resume, replay and fork — 150 pts

| # | criterion | pts | evidence | status |
|---|---|---:|---|---|
| 6.1 | Checkpoint binds model + optimizer + scheduler + RNG + dataloader state + ledger offset | 30 | `checkpoints/*.json` | ✅ |
| 6.2 | Deliberate crash, then resume | 20 | `run.log` crash banner | ✅ |
| 6.3 | **Resume serves exactly the expected next batch** (no skip, no repeat) | 35 | `test_resume_exact_once` | ✅ |
| 6.4 | **Exactly-once across a changed world size** (crash at 4 ranks, resume at 8) | 25 | E2, `test_resume_across_world_size` | ✅ |
| 6.5 | Replay reproduces batch ids, token spans and hashes | 25 | `test_replay_hash_match` | ✅ |
| 6.6 | Fork creates a new branch id and records the divergence point | 15 | `ledgers/branches.json` | ✅ |

## 7. Evaluation and validation firewall — 50 pts

| # | criterion | pts | evidence | status |
|---|---|---:|---|---|
| 7.1 | Registry of train / validation / test with distinct permissions | 15 | `manifests/eval_registry.json` | ✅ |
| 7.2 | Candidate blocked at admission, with a written rejection event | 15 | `firewall.rejections` | ✅ |
| 7.3 | Second check on every served batch (defence in depth) | 10 | `assert_no_eval_in_batch` | ✅ |
| 7.4 | Validation readable for eval but never gradient-bearing | 10 | `test_validation_never_gradient` | ✅ |

## 8. Throughput and packing efficiency — 50 pts

| # | criterion | pts | evidence | status |
|---|---|---:|---|---|
| 8.1 | Real measured wall-clock, not a formula | 15 | `performance.json` | ✅ |
| 8.2 | Useful loss-bearing tokens/sec separated from raw tokens/sec | 15 | `performance.json` | ✅ |
| 8.3 | Packing utilization per policy and per lane, reconstructible | 10 | E3 table | ✅ |
| 8.4 | Numbers reconstructible from artifacts by an independent verifier | 10 | `verify_evidence.py` — 15/15 checks | ✅ |

## 9. Tests, evidence quality and documentation — 50 pts

| # | criterion | pts | evidence | status |
|---|---|---:|---|---|
| 9.1 | Automated tests for every important invariant | 15 | `tests/` | ✅ |
| 9.2 | **Negative controls**: each invariant proven to fail when broken | 15 | `tds/controls.py` | ✅ |
| 9.3 | `evidence.json` + `evidence.md` generated, never hardcoded | 10 | `evidence.py`; every PASS carries compared values + record ids | ✅ |
| 9.4 | README explains architecture and design decisions concisely | 10 | `README.md` | ✅ |

---

# Extension work — beyond the stated 1,000

The instructor's framing of the assignment is explicit that the base spec is a floor, not a ceiling:

> "it's not going to be something that is as intense as we finally are looking for **but you will
> catch up the points that I have missed** and your agents and you will be able to come up with ideas
> and strategies."

So extension work is scoped to exactly three things: (a) questions he stated as **open**, (b) the
artifact he said he most regrets not producing, and (c) the parts of the real system he described in
the session and then explicitly carved out of the assignment. Volume is not extension work — the
project guidance is that padding loses marks — so each item below must change a number or answer a
question, not add pages.

## X1 — The V6 corpus-feedback artifact

The session's emotional centre, said twice: per-token perplexity plus shard outcomes is *"the most
important data"*, *"nobody in the world releases a perplexity dataset"*, and it was missed in V4
(*"literally maybe 10 lines of code and I could have done it"*).

| # | deliverable | evidence |
|---|---|---|
| X1.1 | `v6_recommendations.json`: per shard → protect / upsample / delay / re-clean / phase-guard, each with the measurement that produced it | generated from the learning ledger, not authored |
| X1.2 | Per-token perplexity retained with lane, script, position bucket, boundary flag, OPUS decision | `ledgers/learning_tokens.jsonl` |
| X1.3 | **Position-bucket aggregation** — the course widget promises it and its implementation is broken (reads a field that does not exist) | E5 |
| X1.4 | Phase-crossover detection: shards whose usefulness *inverts* between early and late | mirrors the measured `reason-proof-077` inversion |
| X1.5 | Perplexity at EOS specifically, as the session asks ("how do I know EOS is working") | `learning_boundaries.json` |

## X2 — Answer the questions the instructor left open

| # | open question (his words) | our answer | evidence |
|---|---|---|---|
| X2.1 | *"do we make a shard in which we have 10% [indic] and 90% ... or we keep the indic separately. That is one question that I've not thought about."* | measured both arms | **E1** |
| X2.2 | *"maybe there are some sentences that are easier to train on ... should we first train on it or should we focus on the simple news article"* — Indic intra-lane curriculum | difficulty-ordered Indic sub-curriculum (news → scientific → law), measured against flat ordering | **E6** |
| X2.3 | *"what is the opus rejection rate we're going to go with? We've not decided yet"* | rejection rate as a supply multiplier: `need = share / (1 - r)`; solve for the max feasible r per lane against S5 supply | `feasibility.json` |

## X3 — Production-hardness the spec does not require

| # | deliverable | why it is not padding |
|---|---|---|
| X3.1 | Exactly-once across a **changed world size** (crash at 4 ranks, resume at 8) | the spec's resume test is same-config; the real event is a different machine, and per-rank position silently breaks there |
| X3.2 | **Negative controls** — every invariant proven to FAIL when broken | a green suite is consistent with a vacuous check; the S5 teardown found exactly that bug class |
| X3.3 | Tamper-evident hash-chained ledger | "append-only" is otherwise an unenforced convention |
| X3.4 | Byte-level determinism proof: two full runs, identical artifact hashes | the spec says "I should get exactly what you got" — this measures it rather than asserting it |
| X3.5 | Explicit determinism *scope* statement (data stream exact; model bitwise only within one config) | overclaiming portable float determinism is the fastest way to lose a reviewer |

## X4 — The services the session described and the assignment carved out

He described a four-server architecture (training, dashboard, data-ledger, benchmark) and then said
*"you're not writing the service or the dashboard or the UI or the speed throughput"*. We do not build
servers. We build the **artifacts those servers would serve**, which is the part that carries
information:

| # | deliverable | scope limit |
|---|---|---|
| X4.1 | `dashboard.html` — self-contained, generated from artifacts, no network | a static rendering, not a service |
| X4.2 | Ledger query CLI: `audit.py --step N`, `--shard S`, `--interval A B` | answers the session's stated audit questions ("which shards influenced the model between 5.4B and 5.6B tokens") |
| X4.3 | Downsampling strategy for ledger volume | the session raises the 10M-step / billion-point problem; we state the tiering (full traces for proxy runs, quantized for intervals, aggregates for the full run) and implement the tiering |

## X5 — Corrections fed back to the course material

Six measured defects in the S6 widgets (E0). Each is reported with the reproduction, the measured
values, and what we did instead. This is the literal "catch up the points that I have missed" ask, and
it is the reason our admission gate, packer and throughput reporting differ from the widgets'.

---

## Where we do not claim credit

| not claimed | why |
|---|---|
| Throughput at production scale | measured on a CPU toy in float64; reported as loader-vs-step ratio and utilization only |
| MinHash-LSH dedup | exact + shingle-Jaccard is honest at this corpus size; the scale answer is named, not faked |
| Bitwise model determinism across hardware | float reduction order is not portable; only *data-stream* determinism is claimed |
| Running services, multi-GPU, or a real 120B loader | out of scope by the instructor's own statement; we generate artifacts, not servers |
| Any benchmark score | no benchmark is run; the model is a demonstration vehicle for the ledger |
