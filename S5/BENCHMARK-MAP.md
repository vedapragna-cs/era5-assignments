# ERA V5 — lane → benchmark → data-shape map

The rubric requires each lane tied back to the benchmarks it is meant to win. A lane share is only
defensible if the chain **lane → benchmark → required training-data shape → actual dataset → supply**
holds end to end. Where the chain breaks, that is stated rather than papered over.

`loss shape` uses the session's colour code: **green** = supervised, **grey** = masked context,
**violet** = reward only, no token loss.

---

## Code — 24% (716B)

| benchmark | what it scores | required training shape |
|---|---|---|
| LiveCodeBench | pass@1 on competition problems, rolling window to resist contamination | green: solution + reasoning · grey: problem statement, hidden tests · violet: tests pass |
| Aider Polyglot | % solved, **strict search/replace diff that must apply cleanly** (~225 exercises) | green: **the diff only** · grey: instructions, file contents, test output |
| Codeforces | ELO rating under contest constraints | green: submitted program + reasoning · grey: problem statement |

**Datasets:** The Stack v2 (900B, tier B) · CommitPack/CommitPackFT (4B, tier B) · D3 Code V4 (199B).

**The non-obvious mapping:** The Stack teaches code, but it does **not** teach editing. Aider scores a
search/replace diff that must apply cleanly, and Codeforces-style whole-file generation does not produce
that behaviour. **CommitPackFT is the only dataset in the inventory that teaches diff-following**, and it
is 4B tokens against the lane's 716B. It must be deliberately upsampled inside the code lane rather than
left to its natural 0.6% weight, or we fund a lane that wins LiveCodeBench and loses Aider.

## Agentic — 2% main + 8% anneal (65B)

The lane buys two different benchmark families that need two different data shapes. This is why §3 of the
budget splits it in two.

| benchmark | what it scores | fed by |
|---|---|---|
| BFCL v3 | AST-match accuracy on single/parallel/multi-turn function calls | **short function-calling** sub-lane |
| tau-bench / tau2 | task success + pass^k under a written policy across many turns | short function-calling + policy dialogue |
| SWE-bench Verified | % resolved (pass@1), 500 human-validated repo issues | **long trajectory** sub-lane |
| SWE-bench Live / Pro | same, on issues too new to have leaked | long trajectory |
| Terminal-Bench | task success in a real shell | long trajectory + execution environment |
| GAIA / BrowseComp / WebArena / OSWorld | multi-step tool orchestration, verifiable browsing, GUI goal-state | long trajectory + browser/desktop harness |

**Loss shape (identical across all of them, and the thing most often got wrong):**
green = assistant planning, tool calls, and final answer — **including a call that fails**, because
recovery is the behaviour being taught. grey = user turns, tool returns, repo files, shell output, page
and screen observations. violet = the verifier outcome.

Applying loss to tool observations teaches the model to **invent tool results instead of calling the
tool**. Measured supervision yield on the session's reference trajectory: **356 / 668 = 53%**.

**Datasets — short function-calling (267M):** Glaive 50M · ToolBench 80M · ToolACE 60M · Nexus 30M ·
xLAM 25M · Hermes 22M.
**Datasets — long trajectories (360M):** SWE-Gym 150M · SWE-smith 120M · OpenHands 90M.

**Where the chain breaks:** several of these benchmarks cannot be trained toward with static data at all.
Terminal-Bench, WebArena and OSWorld require a **live execution environment** — a real shell, a hosted
site, a desktop. The session states plainly that V4 never built this. No dataset in the inventory
substitutes for it. **We should not claim these benchmarks on the strength of a token budget**; they are
gated on infrastructure, and the plan says so rather than quietly counting them.

## Reasoning traces — 9% (278B)

| benchmark | what it scores | required shape |
|---|---|---|
| AIME 2024/2025 | accuracy, integer answers 0–999, no tools | green: CoT + final answer · grey: problem · violet (RLVR variant): checked integer |
| GPQA Diamond | accuracy on 198 google-proof graduate science MCQs | green: reasoning + chosen option · grey: question and options |
| HLE | accuracy across ~2,500 expert questions | green: derivation + short answer |
| FrontierMath | accuracy on research-level problems | green: full derivation + exact value |

**Datasets:** AON V4 (78B, tier A) · OpenThoughts2 (3B) · OpenMathReasoning (2B) · OpenR1-Math (1.6B,
long distilled traces) · NuminaMath (0.5B, curated problems).

**Honest accounting:** 78B of the 85B supply is our own V4 corpus. **Open, non-V4 reasoning supply is
~7.1B.** The lane is legitimate because we hold AON, but a reviewer asking "what could you rebuild from
public data?" gets 7.1B, not 85B. The lane also runs at **3.27 epochs**, the tightest real lane in the plan.

This lane feeds Sessions 17–18, where reasoning training actually happens. What we reserve here decides
what is possible there.

## STEM / math — 12% (358B)

Buys GPQA and AIME jointly with the reasoning lane, plus the science half of MMLU.
**Datasets:** proof-pile-2 (55B, formal + informal math) · peS2o (42B, open-access papers) · D4 STEM V4 (49B).

Supply is **146B by auditable inventory rows**, against the composer's stated 250B. We use 146B, which
puts the lane at 2.45 epochs. If 250B is correct the lane is comfortable; the discrepancy is unresolved
and flagged in TODO.

## Long-context — 2% (65B)

Buys `long-eval` (retrieval accuracy at target context).
**Datasets:** repo-packed code at 32K+ (60B) · book-length packed documents (40B).

Deliberately the smallest technical lane. arXiv:2402.10171 finds **0.5–5B tokens** suffice to retrieve
anywhere in a 128K context; 65B is 13× the upper bound. That paper also warns that **naively upsampling
long book documents is suboptimal** and balanced domain mixture matters more, which is why repo-packed
code carries the larger share here.

Constraint from the session: within a batch all sequences share one length, so this lane is staged
(4K → 8K → 16K → 32K+) rather than mixed, and it concentrates in S4 by construction.

## Indic — 17% (520B), 3.15 epochs on 157B usable supply

| benchmark | what it scores |
|---|---|
| MILU (AI4Bharat) | accuracy per language, multi-task understanding asked natively |
| IndicGenBench (Google) | **29 Indic languages, 13 scripts, 4 families**; cross-lingual summarization, MT, QA; ROUGE / chrF / exact-match |

Loss shape: green = the generated answer or summary only; grey = the native-language question or source
passage. The model is trained to *produce*, not to re-emit the prompt.

**Datasets by tier:** A Sangraha verified **51.5B** (64.3B less a 12.8B English split) · B Sangraha
unverified 24B + IndicCorpV2 20.9B · C BPCC + Samanantar + SC judgments **~9B** (Samanantar counted once — it is *inside* BPCC) · D Sangraha synthetic **81.4B**
native-script (the 162.7B figure double-counts a row-for-row Latin transliteration of the same corpus).

**The benchmark reaches further than the data — and now by a measured amount.** Auditing Sangraha
language by language (E2, **[INDIC-ALLOCATION.md](INDIC-ALLOCATION.md)**):

| benchmark | languages | funded | note |
|---|---|---|---|
| MILU | 11 (10 Indic + English) | **10/10** | weakest is Punjabi at 6.20B supply |
| IndicGenBench | 29 | **14/29** | 10 have **zero tokens**; 5 more are **supply-gated** (E10) |

IndicGenBench languages with **zero tokens**: Bhojpuri, Pashto, Awadhi, Haryanvi, Tibetan, Garhwali,
Chhattisgarhi, Rajasthani, Malvi, Marwari.
**Supply-gated** (below the 1.30B entry threshold, `INDIC-ALLOCATION.md` §13): Bodo, Konkani, Maithili,
Manipuri, Santali. These are not dropped — they carry an acquisition target and re-enter automatically on
crossing the threshold.

**We therefore claim MILU outright and IndicGenBench only on its 14-language funded subset.** Listing
IndicGenBench unqualified would claim an average over 15 languages we cannot fund — the same
wishful accounting this map exists to catch. Claiming 19/29 with five languages funded at ~30M tokens
would be a *weaker* position than 14/29 funded properly, because the five would not survive a check. Allocation across the funded languages is temperature
sampling at α = 0.72, water-filled against the per-language 4-epoch cap.

## General web — 34% (997B)

Buys MMLU, and nothing else directly. It is the largest lane and the one hardest to defend on benchmark
grounds alone, because its real product is **common sense** — the capability that has no clean benchmark
and whose absence, per the session, produces a model whose code compiles and does not work.

**Datasets:** DCLM-Baseline (2.6T, model-filtered from a 240T pool) · FineWeb-Edu (1.3T, edu-classifier
filtered) · D1/D2 V4 web (791B). Supply is 4.7T against 997B demand — the only lane with genuine slack.

That slack is the reason web is where we take budget *from* when a scarce lane needs it, and it is also
why cutting it too far is the classic failure the session warns about.

---

## Coverage summary

| lane | share | primary benchmarks | supply verdict |
|---|---|---|---|
| Code | 24% | LiveCodeBench, Aider, Codeforces | covered (0.65 ep) |
| Agentic | 2% + 8% anneal | BFCL, tau-bench, SWE-bench, GAIA, BrowseComp | **synthesis + infra gated** |
| Reasoning | 9% | AIME, GPQA, HLE, FrontierMath | 3.27 ep, 78B of 85B is V4 lineage |
| STEM | 12% | GPQA, AIME, MMLU-science | 2.45 ep, supply disputed |
| Long-context | 2% | long-eval | covered (0.65 ep) |
| Indic | 17% | MILU (10/10), IndicGenBench (**14/29**) | **3.15 ep**, content-weighted α\* = 0.59 |
| General web | 34% | MMLU (+ common sense, unmeasured) | covered (0.21 ep) |

Three benchmarks in the session's list are **not claimed** by this plan: Terminal-Bench, WebArena and
OSWorld. They require execution environments we have not built. Claiming them from a token budget alone
would be the same wishful accounting the plan exists to avoid.
