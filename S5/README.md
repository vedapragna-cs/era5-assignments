# ERA V5 — Data Mixture and Curriculum Plan

**A 3T-token mixture for an Indic-capable coding and agentic model.** Every share below is checked
against real supply; where a share exceeds supply, that is stated as a production programme with a target,
not absorbed into a percentage.

Numbers are labelled **measured** (computed or observed here), **stated** (asserted by the course or a
cited source), or **stylized** (a course widget's illustrative formula, no empirical backing).
Everything is reproducible from [`scripts/`](scripts/).

**Status: v1 — all twelve design decisions taken ([`DECISIONS.md`](DECISIONS.md)), each with a stated
reversal condition. Every share remains a hypothesis until T1–T5 run (§8).**

---

## 1. The plan

**3T total** (session states 2.4–4T) · **97% main run (2.91T) / 3% anneal (90B)**.
Anneal size follows OLMo 2, which sampled an 843B pool into 50B/100B/300B anneal mixes.

| lane | main | anneal | demand | supply | epochs | status |
|---|---|---|---|---|---|---|
| General web | 34% | 8% | 989B | 4,691B | 0.21 | funded |
| Code | 24% | 20% | 698B | 1,103B | 0.63 | funded |
| **Indic** | **17%** | **28%** | 495B | **157B** | **3.15** | funded, tightest real lane |
| STEM / math | 12% | 10% | 349B | 146B | 2.39 | funded |
| Reasoning traces | 9% | 18% | 262B | 85B | 3.08 | funded — 78B is V4 lineage |
| Long-context | 2% | 8% | 58B | 100B | 0.58 | funded |
| **Agentic** | **2%** | **8%** | 58B | **0.63B** | **92** | **programme — §4** |

Feasibility test throughout is **epochs ≤ 4** (arXiv:2305.16264: up to 4 epochs of repeated data costs
negligible loss; value decays beyond). It is a constraint, not a preference.

**Changes from the course composer default `24/2/6/6/16/12/34`:**

- **Long-context 6% → 2%**, freeing 116B. arXiv:2402.10171 finds **0.5–5B tokens** suffice to retrieve
  anywhere in a 128K context — the capability is *"mostly already acquired through large-scale
  pretraining."* Even at 2% we allocate 58B, still 12× the published upper bound. Largest defensible
  saving in the budget.
- **Reasoning 6% → 9%.** Llama 3 spends 25% on math+reasoning; ours is 9% traces + 12% STEM = 21%, the
  same band. Stated jointly so 9% is not read in isolation.
- **Indic 16% → 17%.** The tier split matters far more than the headline — §3.

### What each share actually rests on

| | |
|---|---|
| funded from supply we hold today | **98%** of the main run |
| rests on a named production programme | **2%** (agentic) |

**The honest headline is not "100% allocated."** All delivery risk in this budget sits in one lane — the
lane the session's protected floor exists to defend.

**A dependency that a flat table hides:** 78B of the reasoning lane's 85B is our own V4 AON corpus.
A public-only rebuild supports **1.0%, not 9%**; losing the V4 corpora opens an **8.0% hole** in the run.

### Protected floors the selector may not cross

OPUS optimises for a golden proxy and **measured, it drives unprotected lanes to zero** — Indic received
**0 tokens in 60/60 iterations** under the V4 English-heavy proxy, and agentic **0 in 59/60** even after
rebalancing the proxy. Floors are therefore mechanism, not policy. V4 protected Indic alone; **V5 extends
protection to Indic, agentic and reasoning.**

| lane | allocated | **floor** | rationale |
|---|---|---|---|
| Indic | 17% | **12%** | the session's stated floor, inherited |
| Agentic | 2% | **2%** | floor equals allocation — there is no slack to concede |
| Reasoning | 9% | **6%** | the composer's default share; we protect that level and leave our +3% increment contestable |

Delivered by an **always-on lane injecting a fixed fraction of every batch, invisible to the selector.**
**Stated design value 8%; measured simulated delivery ~15% of trained tokens** — we quote both, because
the widget's design value and its behaviour disagree and a floor specified in design units will be
overshot in practice.

### The anneal reserve, declared

Held back from the main run and spent only in the 90B cooldown:

| reserved | size | why it cannot be spent early |
|---|---|---|
| **Tier-A verified Indic** | 25.2B of the anneal | the only human-verified Indic; the selector would consume it first |
| **Long agentic trajectories** | 360M (SWE-Gym, SWE-smith, OpenHands) | the expensive shape; 7.2B anneal demand vs 58B if spent in the main run |
| **Long reasoning traces** | 16.2B of the anneal | B5-difficulty material, useful only once the base exists |

**T3 tests whether the reserve earns its keep.** If an early-spend arm matches it, the reserve earns
nothing and the floor alone carries the design.

**Web is the adjustment variable, not a tuned value.** At 0.21 epochs it could support 645% of the run
alone. Every fallback is absorbed there and every trigger draws from there. **Hard floor 25%** — the
session warns that cutting general web too far gives a model whose code compiles and does not work.

---

## 2. What auditing the inventory changed

The course inventory was taken as given, then checked source by source against live data. **Seven rows
did not survive**, five of them downward. Full audit: [`SUPPLY-AUDIT.md`](SUPPLY-AUDIT.md).

| claim as given | audited | how |
|---|---|---|
| Indic verified 64.3B | **51.5B** | 12.76B of it is an **English** split inside Sangraha's verified config |
| Indic synthetic 162.7B | **81.4B** native-script | native/Latin splits have **identical row counts** — one corpus transliterated |
| Tier C = BPCC 3B + Samanantar 2B | **Samanantar is *inside* BPCC** | two directories, 35.3GB of BPCC's 109.3GB |
| BPCC = 22M samples | **~230M bitext pairs** | ~6.8B Indic-side |
| translated judgments ~0.3B | **1.7–2.6B** | 134,028 documents measured, not 31,000 from a press figure |
| court corpus 37–142B | **~8B** | 122 PDFs measured; High Court orders median **392 words**, not the Supreme Court's 7,744 |
| IndicCorpV2 20.9B Indic | **~17.9B** | `data/en.txt` is the corpus's **largest file** at 39.74GB = 14.4% |

**Three of three major Indic corpora bundle English or English-derived content into a headline "Indic"
figure** — Sangraha's verified tier (12.76B), IndicCorpV2 (14.4%), and tier D's translated-English
provenance. That is a rule, not a coincidence: **assume an Indic corpus card includes English until the
per-language table says otherwise.**

**Consequence:** Indic lane epochs move **1.88 → 3.15**. The lane goes from comfortable to the tightest
real lane in the plan. Four of six corrections went down, which is the expected direction — inventories
are assembled from headline figures, and headline figures flatter.

---

## 3. Indic — the differentiator, and where it actually breaks

### The composer's default tier split is impossible

A40/B25/**C20**/D15 demands **99B** of translated/parallel text against **~9B** of real supply —
**11 epochs**. Any plan inheriting that default has already failed the wishful-accounting test.

| tier | proposed | demand | supply | epochs |
|---|---|---|---|---|
| A verified native | 28% | 138B | 51.5B | 2.7 |
| B unverified crawl | 30% | 148B | **41.9B** | **3.5** |
| C translated / parallel | **2%** | 9.9B | ~9B | 1.1 |
| D synthetic | 40% | 198B | 81.4B | 2.4 |

### Tier D is one encyclopedia rendered 28 ways

The largest row in the Indic inventory — 59% of all Indic supply — is, in AI4Bharat's own description,
*"Wikimedia English translated to 14 Indic languages and further romanized by transliteration."*

So it is **translated English, from one source (~4.5B words), rendered 14 languages × 2 scripts**. The
2.4-epoch figure counts token strings; the repetition ceiling is about repeated **content**.

**This makes T2 a sharper experiment.** MILU is India-centric by construction — state examinations, local
history, festivals, law. Translated English Wikipedia is the wrong content distribution for exactly those
subjects and roughly the right one for general science. **Prediction: tier D degrades MILU's
India-specific subjects while holding on its general ones.** Falsifiable per subject.

### Per-language allocation — because MILU is scored per language

A 17% lane silent on language distribution predicts neither MILU nor IndicGenBench.
Full method: [`INDIC-ALLOCATION.md`](INDIC-ALLOCATION.md).

- **Mechanism:** temperature sampling `p ∝ S^α` (XLM-R, arXiv:1911.02116), water-filled against a
  **per-language** cap of 4 × its own supply.
- **The cap is what binds.** α = 1 is exactly uniform-epoch allocation, so at 3.15 epochs there is under
  1 epoch of headroom and any lower α immediately pins languages to the ceiling.
- **Allocated over content, not tokens.** The same content costs **1.375 tokens/word in Hindi and 3.025
  in Malayalam** (MUTANT-Indic). A token-equal rule gives Malayalam 20% of Hindi's content. Correcting the
  unit gives **α\* = 0.59** and moves 14.2B from Hindi to the high-fertility languages.
- **It improves the unit and cannot fix the problem.** Malayalam reaches 29% of Hindi's content and stops
  — it hits the 4-epoch ceiling first. **High-fertility languages exhaust their repetition budget sooner**,
  so the languages most disadvantaged by a token rule are least able to absorb the correction.
  **Content parity is not reachable at this lane size.** We state that rather than imply otherwise.

**We differ from XLM-R's published α = 0.3 because the binding constraint differs**, not because we
dispute the finding: α = 0.3 pins 17 of 22 languages to the ceiling here.

### 14 funded, 8 gated

The supply distribution is not a smooth tail. It is 14 languages and then a **24× cliff** — Assamese
6.16B, then Sindhi 0.26B.

**Entry gate: ≥1.30B unique native-script tokens**, the level at which a language can reach 1% of the
lane without breaching 4 epochs. Below it, a language is an **acquisition target with an automatic
re-entry trigger**, not a 0.006% share doing nothing.

**Price, stated: IndicGenBench falls 19/29 → 14/29** (Bodo, Konkani, Maithili, Manipuri, Santali). MILU
unaffected at 10/10. **19/29 with five languages funded at ~30M tokens is a weaker claim than 14/29
funded properly** — a reviewer who checks Santali finds 7.7M tokens.

**The two halves need different kinds of "later."** The 14 funded languages grow *passively* as the web
grows. The 8 gated ones do not — almost nothing is written down in Santali or Dogri at volume, so their
supply grows **only if we fund collection**. Speech is the cheapest route and points straight at them:
**Rasa's single largest language is Bodo**; IndicVoices carries Manipuri.

### How the model reasons — decided, not deferred

**Indic question → English chain of thought → Indic answer** (Arm A). A plan that does not say how the
model reasons is not a plan, so this is a stated default that **T5 confirms or refutes**, not a question
left open.

**Why English-internal:** English-centred latent reasoning is the *default* for multilingual models
(Wendler et al., arXiv:2402.10588), so Arm A works with the grain. Decisively, **Arm A's translations are
verifiable and Arm B's are not** — A translates questions and short answers, frequently exact-match
checkable; B translates free-form reasoning chains, where a fluent mistranslation is indistinguishable
from a correct one and errors compound across steps. Cost is secondary (16.4B vs 27.8B translated tokens).

**Sizing:** 10% of the reasoning lane (27.8B), drawn from within it — no other lane moves. Needs 28% of
our 78B AON corpus as source.

**Honest limitation:** this slice is a **re-presentation of AON content, not new content**. The English CoT
is unchanged and only question and answer are translated, so **it does not relieve the reasoning lane's
3.08 epochs** — it changes the form of some of those repetitions.

**Against it:** MILU is India-centric, so reasoning about Indian law and festivals in English risks losing
the grounding the model exists to have. That is exactly what T5's per-subject cut measures, and arm C
(no cross-lingual data) is included because both arms could be worse than nothing.

---

## 4. Agentic — the only lane not funded by data

At the 4-epoch ceiling, the same 8% share is a fantasy in the main run and tractable in the anneal:

| placement | share | demand | unique needed | have | must synthesize |
|---|---|---|---|---|---|
| Main run | 2% | 58B | 14.6B | 0.63B | **13.9B = 22× all public agentic data** |
| **Anneal** | **8%** | **7.2B** | **1.8B** | 0.63B | **1.2B = 2× public supply** |

**The anneal is not pedagogy — at 3T it is the only arithmetically feasible placement.** Sized in samples
rather than tokens ([`AGENTIC-SYNTHESIS.md`](AGENTIC-SYNTHESIS.md)): **~24.5M new function calls** (main
run, cheap shape — schema-generable, execution-verifiable) and **~153,600 new trajectories** (anneal,
expensive shape) — the latter being **5.9× what SWE-smith produced**.

Placing the expensive shape in the main run instead would demand ~1.5M trajectories. **The anneal
placement reduces the hard half of this programme by an order of magnitude.**

**Why not just let the selector allocate it?** Measured: an unprotected lane goes to **zero** — 0 tokens
in 59/60 OPUS iterations — and **no proxy choice rescues it**. Agentic ranks last under a balanced proxy
and second-last under an English one. Only a protected floor works.

**QA gate is not optional.** Unverified synthetic agentic data is worse than none: applying loss to tool
observations teaches the model to invent tool results instead of calling the tool. Every tool return must
come from a real invocation; the verifier is the label; masks are audited.

---

## 5. Curriculum

Full schedule: [`CURRICULUM.md`](CURRICULUM.md).

**Stage mixtures must integrate to the lane budget** — a constraint the course widget does not enforce.
Anchors that each sum to 100% can still spend a lane far above its allocation once weighted by stage
length. Solved by iterative proportional fitting over both constraints.

| lane | S1 Seed 233B | S2 General 931B | S3 Reasoning 1,048B | S4 Long-ctx 698B | Anneal 90B |
|---|---|---|---|---|---|
| General web | 56.1% | 45.3% | 28.1% | 20.4% | 8% |
| Code | 14.6% | 19.2% | 26.8% | 29.3% | 20% |
| Reasoning | 2.0% | 4.8% | 12.5% | 11.7% | 18% |
| **Indic** | **17.2%** | **16.9%** | **16.9%** | **17.2%** | **28%** |
| STEM | 9.0% | 11.8% | 12.8% | 12.0% | 10% |
| Long-context | 0.2% | 0.4% | 0.9% | 6.3% | 8% |
| Agentic | 1.0% | 1.5% | 2.0% | 3.0% | 8% |

- **Reasoning enters late** (2% → 12.5%): long traces poured into early pretraining do not produce a
  reasoning model; the base must exist first.
- **Indic is deliberately flat.** A protected capability that ramps is still a residue of leftover budget.
  A flat lane is a commitment — and post-audit it now runs near its repetition budget throughout, so the
  commitment is no longer free.
- **Long-context is absent until S4**, consistent with the capability being acquired during general
  pretraining.

**Seam stability — sized from measurement, not intuition.** Measured: frozen embeddings give a **151×**
gradient spike vs **8.0×** unfrozen at identical sharpness, and with embeddings frozen **no warmup band
≤5B tokens** brings a transition under the 3× threshold.

> **Rule 1: embeddings stay trainable across every seam.** This dominates band width by ~19× and is the
> actual lesson of the V4 incident, usually told as a mixture-shift story.
> **Rule 2: blend every seam over ≥2B tokens** (measured peak 2.8×). Final seam gets 4B — largest shift,
> and where a diverged run is most expensive to lose.

**Reasoning-length bands — a concrete example at each level.** Reasoning is a *distribution of trace
lengths*, not one slot. Anchors are the measured widget traces on *"how many integers 1..1000 are
divisible by 3 or 5?"* (answer 467).

| band | budget | measured | example of the trace | share |
|---|---|---|---|---|
| **L0** low | ≤64 tok | 37 tok | *"Inclusion–exclusion: ⌊1000/3⌋+⌊1000/5⌋−⌊1000/15⌋ = 333+200−66."* | 30% |
| **L1** medium | 64–256 | 74 tok | counts each term separately, states why 15 is the overlap, then sums | 35% |
| **L2** high | 256–1,024 | 161 tok | derives each count with justification, then **verifies by complement** (2/3 × 4/5) | 25% |
| **L3** ultra | 1,024–8,192 | 346 tok | restates goal, plans, derives, re-checks by blocks of 15, second density check | **10%** |

**Why L3 is capped at 10%.** Measured: HIGH→ULTRA costs **2.15× the tokens for +4 accuracy points**
(161→346 tok, 91%→95%) — the widget itself labels ULTRA *"wasted effort"*. Over-representing long traces
teaches that more thinking is always better, the exact failure the session warns about. **The dial has to
be taught, and teaching it requires the short bands to dominate.** Each band must span mathematics, code
and general problem solving, or the behaviour binds to one domain.

**Difficulty bands — a concrete example at each level.** Each stage draws from a band range; the ladder
advances with the stage.

| band | level | concrete example of the data | stages |
|---|---|---|---|
| **B0** | Nursery | *"The cat sat on the mat. It was warm."* — TinyStories-style synthetic | S1 |
| **B1** | Grade-school | Simple-Wikipedia articles; GSM8K word problems (*"Natalia sold clips to 48 friends…"*) | S1–S2 |
| **B2** | High-school | MMLU high-school subsets; introductory algebra and mechanics; well-commented beginner code | S2–S3 |
| **B3** | Undergraduate | undergraduate textbook chapters; LeetCode easy/medium with worked solutions; peS2o survey papers | S3–S4 |
| **B4** | Graduate | arXiv papers; AIME problems; GPQA-style graduate science; multi-file repository code | S3–S4 |
| **B5** | Research / PhD | proof-pile-2 formal proofs; FrontierMath-level problems; SWE-bench-style repo patches with hidden tests | S4 + **anneal** |

**B5 concentrates in the anneal** — the "best data saved for last" decision, which only works if the
reserve survives the main run.

---

## 6. Benchmarks — what we claim, and what we do not

Full chain (lane → benchmark → required data shape → dataset → supply):
[`BENCHMARK-MAP.md`](BENCHMARK-MAP.md).

| lane | benchmarks | verdict |
|---|---|---|
| Code | LiveCodeBench, Aider Polyglot, Codeforces | covered |
| Agentic | BFCL v3, tau-bench, SWE-bench Verified, GAIA, BrowseComp | synthesis-gated |
| Reasoning | AIME, GPQA Diamond, HLE, FrontierMath | 3.08 ep; 78B of 85B is V4 |
| STEM | GPQA, AIME, MMLU-science | 2.39 ep; supply disputed |
| Indic | **MILU 10/10**, **IndicGenBench 14/29** | see §3 |
| Long-context | long-eval | covered — repo-packed code at 32K+ (60B) + book-length packed documents (40B) |
| Indic (romanized) | **Dakshina** (transliteration, 12 langs) | 1% sub-lane, 0.06 ep |
| General web | MMLU (+ common sense, unmeasured) | covered |

**Romanized Indic is scoped, not ignored.** The ~81B Latin-script half of the synthetic tier buys a real
capability — reading how Indians actually type — but *generating* romanized text matters little for a
coding model, so the need is input robustness, not fluency. **1% of the Indic lane (~5B) claiming
Dakshina** (LREC 2020, CC BY-SA 4.0, 12 languages, 10 of them in our funded 14). Not GLUECoS: word-level
classification only, and Hindi is its sole Indic language. The motivating evidence is Indi-RomCoM's
finding that *"LLMs consistently underperform on RCM instructions, with performance degrading as
code-mixing density increases"* — a documented weakness on the exact input form Indian users produce. We
cite the finding, not the benchmark: 4 languages, unreviewed preprint.

**Not claimed: Terminal-Bench, WebArena, OSWorld.** They need a live shell, hosted sites and desktops
that V4 never built. No token budget buys them, and claiming them from a budget would be the same wishful
accounting this plan exists to avoid.

**The non-obvious mapping in the code lane:** The Stack teaches code but does **not** teach editing. Aider
scores a search/replace diff that must apply cleanly. **CommitPackFT is the only dataset in the inventory
that teaches diff-following**, at 4B against a 716B lane — 0.6% at natural weight. **Upsampled 8×** to
~4.5% and 2 epochs, or we fund a lane that wins LiveCodeBench and loses Aider.

**Loss masking, identical across all agentic benchmarks and the thing most often got wrong:** green =
assistant planning, tool calls and final answer — *including a call that fails*, because recovery is the
taught behaviour. grey = user turns, tool returns, repo files, shell output. violet = verifier outcome.
Measured supervision yield on the reference trajectory: **356/668 = 53%**, so supply must be sized in
*seen* tokens and capability argued in *supervised* ones.

---

## 7. Structural checks that found nothing — and are reported anyway

Three objections were raised against the budget and none moved a number. Reporting them is the point: a
reviewer is entitled to assume they were never run.

- **Multi-tagged lanes** ([`multitag.py`](scripts/multitag.py)). S3 assumes documents carry multiple tags,
  which would make every table wrong. **Measured mean tags per token m = 1.020** — shares could sum to
  102%, not 130%. The overlap sits in code/long-context/agentic, already at ≤0.63 epochs. **Indic and
  reasoning intersect nothing by corpus construction** (web = DCLM/FineWeb, STEM = peS2o/proof-pile-2,
  reasoning = AON — all English). Robust: reaching 2.5 epochs on Indic would need **28% of the entire STEM
  corpus to be Indic-language**. Multi-tagging is the most natural way to make an infeasible budget look
  feasible, so **m is reported**.
- **India-first as a lane** ([`CROSS-LANE.md`](CROSS-LANE.md)). S3 gives it 9%; open English-language
  India-first supply is **8.46B = 31 epochs**. As a *coverage tag* it is over-delivered — 9.7% even
  assuming only half the Indic lane is India-first context — and S3's "≥55% non-English" rule passes at
  88–94%. **The lane is unfundable and the tag is over-delivered**; S3 and S5 never actually disagreed.
- **Cross-lane requirements.** S3's *"≥15% of STEM should be Indic-language"* needs 52.4B against ~11.7B —
  **4.5× short, supply supports 3.4%.** Resolved by **sourcing, not reallocation**: NCERT/SCERT/NPTEL/IGNOU
  are Indian-curriculum STEM in Indian languages and are gated. **This is the plan's strongest case for
  prioritising one specific negotiation.**

**A general finding worth carrying:** for Indic data the binding constraint is repeatedly **provenance,
not volume**. Tier D is translated Wikipedia; IndicAlign is 99.8% derived or translated; the verified tier
contained English. Every corpus that looks large is large because something was translated, and the
natively-authored fraction runs two to three orders of magnitude smaller. **A plan that sizes Indic
capability in tokens will keep concluding it is funded when it is not.**

---

## 8. What would refute this

Every number above is a hypothesis until these run. Protocol throughout is Llama 3 annealing-as-evaluation
/ OLMo 2 microannealing — LR linear to 0 with 30% candidate / 70% default mix — **not** from-scratch runs.
Full pre-registration: [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

**Two scales, with different jobs.** **1B screens** — all five hypotheses, cheap, randomised data order,
repeated where the signal is marginal (OLMo 2 averages 3 runs). **3B confirms** — only the hypotheses that
survive 1B, because the failure mode we most fear is scale-dependence: Llama 3 report anneal gains were
negligible at 405B, so an effect measured small must be re-measured larger before it sets a full-scale
budget. A hypothesis that survives 1B and fails 3B has told us the cheaper experiment was measuring the
wrong thing.

| # | hypothesis | metric | refuted if |
|---|---|---|---|
| **T1** | long-context at 2% loses nothing vs 6% | long-eval retrieval at target context | 2% arm degrades beyond noise → restore 6%, return 116B |
| **T2** | tier D at 40% is not worse than verified-only | **MILU per subject**, India-specific vs general | India-specific subjects degrade → differently-sourced synthetic, not less |
| **T3** | reserving agentic for the anneal beats spending it early | SWE-bench Verified, BFCL v3 | early-spend matches reserve → the reserve earns nothing |
| **T4** | content-weighted α beats proportional | MILU **mean and minimum** | proportional matches, **or** the mean rises via Hindi alone |
| **T5** | English-internal vs native reasoning vs neither | MILU by subject + **script detection on CoT and answer** | see below |

**T5 is the one genuinely open design decision.** Nothing in the inventory has the target shape (Indic
question → reasoning → Indic answer), so both arms must be synthesized: **Arm A** (English CoT) needs
16.4B translated tokens, **Arm B** (native CoT) 27.8B. The asymmetry that matters is not the 1.69× — it is
that **A's translations are exact-match checkable and B's are free-form reasoning chains where a fluent
mistranslation is indistinguishable from a correct one.** Arm C (no cross-lingual data) is included
because both arms could be worse than nothing.

**Known limitation, stated before a reviewer raises it:** Llama 3 report anneal gains were **negligible at
405B** — *"our flagship model ... does not require specific in-domain training samples."* T1–T5 measure
anneal-stage value; transferability to a much stronger base is unproven. Likewise, latent-language
behaviour depends on scale, so a null T5 at 1B is weaker evidence than a positive one.

---

## 9. Decisions taken, and what would reverse them

**All twelve are decided** — full record in [`DECISIONS.md`](DECISIONS.md). A decision without a reversal
condition is a preference, so each carries one. The ones that change the most:

| decision | reverses if |
|---|---|
| run scale **3T** | 4T is chosen — Indic then breaches the ceiling and must fall to ~15.7% |
| anneal **3% (90B)** | trajectory synthesis proves cheaper than costed, making 5% affordable |
| agentic **2% floor, programme funded** | **T3** shows the anneal reserve earns nothing |
| reasoning **English-internal (Arm A)** | **T5** — Arm B wins on India-specific MILU subjects |
| tail languages **gated at 1.30B** | a language crosses the threshold — **automatic, no redesign** |
| **prioritise** the NCERT/SCERT/NPTEL/IGNOU negotiation | it fails; Indic-STEM drops to 3.4% |
| execution environments **not scoped** | V4 infrastructure delivers a live harness |
| CommitPackFT **8× upsample** | more diff-following data than its 4B arrives |
| Indic **17%** | **T2** refutes tier D, or run scale moves to 4T |

Two were closed by measurement rather than judgement: **romanized Indic** (E13) and the **STEM supply
figure** (E12).
- ~~STEM supply discrepancy~~ — **resolved (E12).** The inventory has **no STEM band**: `peS2o` and
  `proof-pile-2` are tagged `general`, in a band literally named *"General web & STEM"*. The composer
  invents a seventh lane and splits that band. Our 146B is the sum of its three STEM-content rows, and
  4,837 − 146 = **4,691B**, exactly the web figure we use. The two numbers were never measuring the same
  thing.

---

## 10. Reproducing

| script | what it produces |
|---|---|
| [`budget.py`](scripts/budget.py) | lane budgets and epoch arithmetic |
| [`curriculum.py`](scripts/curriculum.py) | IPF solver for stage mixtures under the budget-integral constraint |
| [`indic.py`](scripts/indic.py) · [`indic_fertility.py`](scripts/indic_fertility.py) · [`indic_gated.py`](scripts/indic_gated.py) | per-language allocation, fertility adjustment, supply gate |
| [`agentic.py`](scripts/agentic.py) | synthesis programme sized in samples |
| [`supply_audit.py`](scripts/supply_audit.py) | live queries against HuggingFace datasets-server |
| [`judgments.py`](scripts/judgments.py) | measures the AWS court corpora directly from S3 |
| [`multitag.py`](scripts/multitag.py) · [`indiafirst.py`](scripts/indiafirst.py) · [`conditional.py`](scripts/conditional.py) | structural checks |
| [`xlingual.py`](scripts/xlingual.py) | T5 arm costing |
| [`explore*.py`](scripts/) · [`render_all.py`](scripts/render_all.py) | lesson-widget teardown harness |

**Supporting documents:** [BUDGET-PROPOSAL](BUDGET-PROPOSAL.md) · [CURRICULUM](CURRICULUM.md) ·
[BENCHMARK-MAP](BENCHMARK-MAP.md) · [AGENTIC-SYNTHESIS](AGENTIC-SYNTHESIS.md) ·
[INDIC-ALLOCATION](INDIC-ALLOCATION.md) · [SUPPLY-AUDIT](SUPPLY-AUDIT.md) · [CROSS-LANE](CROSS-LANE.md) ·
[DECISIONS](DECISIONS.md) · [EXPERIMENT_LOG](EXPERIMENT_LOG.md) · [RESEARCH-NOTES](RESEARCH-NOTES.md) ·
[SESSION-DIGEST](SESSION-DIGEST.md) · [TODO](TODO.md)

Course material (lesson page, transcript, widget sources) is deliberately **not** committed here; this
repository holds our own analysis and reproducible scripts.
