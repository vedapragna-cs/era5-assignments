# S5 experiment log

Append-only. Entries are written **before** execution and results appended after.
Never rewrite a historical measurement; supersede it with a new dated entry.

---

## E0 — Widget teardown (COMPLETE, 2026-07-30)

**Question.** What does the session actually assert, numerically, as opposed to what was said aloud?

**Method.** Extracted the 9 lesson widgets from the `.webarchive`, then drove each in Chromium via
Playwright and swept its parameter space. Harness in `scripts/`; full results in `SESSION-DIGEST.md` §13.

**Results (measured).**
- Feasibility matrix over preset × run size: a **1T run is the only scale at which the course's own
  pretrain preset is nearly feasible** from real supply; agentic reads "must synthesize" at *every* scale.
- OPUS starvation is **total, not gradual**: with the V4 English-heavy proxy and protection off, Indic got
  **0 tokens in 60/60 iterations**.
- **Rebalancing the proxy does not rescue agentic.** Under a balanced proxy Indic recovers to ~32% but
  agentic stays at 0.41% (0 in 59/60). Dot products explain it: agentic ranks **last** under the balanced
  proxy and second-last under the English one. Only a protected floor works.
- Mixture-shift instability is dominated by **frozen embeddings, not the shift itself**: 151× vs 8.0× at
  identical sharpness. With embeddings frozen, no warmup band available in the widget (≤5B tokens) brings
  the spike under the 3× stability threshold.
- Agentic supervision yield: **356 supervised / 668 seen tokens = 53%**.
- Reasoning effort: HIGH→ULTRA costs **2.15× tokens for +4 accuracy points** (161→346 tok, 91%→95%).

**Failures / corrections.** First sweep of the instability widget was reported with **inverted** frozen-
embedding labels: the state check used class `on` where the widget uses `s5ms-on`, so it read false
always. Caught by re-deriving against the widget's documented default before publishing. Table in the
digest is the corrected one.

**Widget defects found (do not quote these columns).**
1. The anneal supply column computes `demand = share × full run size`, overstating anneal demand ~30×.
2. The "8% always-on lane" delivers ~15% of trained tokens in simulation.
3. The effective-token curve claims 3.1× at 90% keep — stylized, defensible only near its 40% anchor.

---

## E1 — Budget arithmetic (COMPLETE, 2026-07-30)

**Question.** At the course's stated 2.4–4T scale, which lane budgets are reachable from real supply?

**Method.** `scripts/budget.py`. Feasibility test is `epochs ≤ 4`, from arXiv:2305.16264 (≤4 epochs of
repeated data costs negligible loss vs unique; value decays beyond).

**Results (measured).** Full tables in `BUDGET-PROPOSAL.md`.
- Agentic at 2% of a 2.91T main run needs **14.6B unique tokens at the 4-epoch ceiling vs 0.63B available
  = 22× all public agentic data.** The same 8% share inside a 90B anneal pool needs 1.8B unique = **2×**.
  **The anneal is the only arithmetically feasible home for the agentic lane.**
- The composer's default Indic tier split is **impossible**: C (translated) at 20% demands 99B against
  **5B** of real parallel corpora = 19.8 epochs.
- Long-context at 6% allocates 175B where the literature requires **0.5–5B** (arXiv:2402.10171).
  Cutting to 2% frees **116B tokens** and is still 12× the published upper bound.

---

## E2 — Indic supply audited per language (COMPLETE, 2026-07-31)

**Question.** The lane is 17%. Of *which languages*? MILU is scored per language and IndicGenBench spans
29, so a lane share silent on language distribution predicts neither score.

**Method.** `scripts/indic.py` over the AI4Bharat Sangraha per-language token table. That table's totals
(64,306.1M verified / 24,307.7M unverified / 162,707.9M synthetic) reproduce the S5 inventory rows
exactly, so each per-language row is auditable against a number the session already uses. Allocation is
temperature sampling `p_i ∝ S_i^α` (XLM-R, arXiv:1911.02116) water-filled against a per-language cap of
4 × supply (arXiv:2305.16264). Full write-up in `INDIC-ALLOCATION.md`.

**Results (measured).**
- **19.8% of "Tier A verified Indic" is English.** `eng` is a split in Sangraha's verified config at
  12,759.9M tokens. Verified *Indic* is **51.5B, not 64.3B**.
- **The synthetic tier is one corpus transliterated, not two.** 28 splits = 14 languages × 2 scripts, and
  each native/Latin pair has an identical row count (`hin_Deva` = `hin_Latn` = 5,775,143 rows). Only the
  native half is scored by MILU/IndicGenBench: **~81.4B, not 162.7B**.
- Combined effect: **usable Indic supply 157.2B, not 275.9B → lane epochs 1.88 → 3.31.** Still under the
  ceiling; the lane goes from comfortable to second-tightest in the plan.
- **10 of IndicGenBench's 29 languages have zero tokens** in the inventory (Bhojpuri, Pashto, Awadhi,
  Haryanvi, Tibetan, Garhwali, Chhattisgarhi, Rajasthani, Malvi, Marwari). **MILU is 10/10 fundable.**
- 8 languages hold under 1B tokens; **combined 292.7M = 0.19% of supply**. At their full 4-epoch ceiling
  they absorb 1,171M = **0.225% of the lane** — the tail is a supply problem, not an allocation one.
- **α = 1 is exactly uniform-epoch allocation** (every language at 3.31). With only 0.69 epochs of
  headroom, **α* = 0.72** is the largest temperature at which the weakest MILU language (Punjabi) already
  reaches its cap; every lower α buys MILU nothing and pins more languages at 4 epochs.
- Per-language verified share spans **5% (Assamese) to 46% (Bengali)** — a single global tier split is an
  average, not a specification.

**Consequences for existing documents.** Indic epoch figures in `BUDGET-PROPOSAL.md` §1/§2,
`CURRICULUM.md` and `BENCHMARK-MAP.md` corrected; the unqualified IndicGenBench claim narrowed to its
19-language funded subset.

**Limitation.** IndicCorpV2 (20.9B) and BPCC + Samanantar (5.0B) publish no per-language token table.
They are counted in lane supply (raising it to 183.1B, 2.84 epochs) but not allocated by language.

---

## T1 — Long-context at 2% loses nothing versus 6% (PRE-REGISTERED, not yet run)

**Hypothesis.** Reducing the long-context lane from 6% to 2% costs no measurable long-context capability,
because the ability is mostly acquired during general pretraining (arXiv:2402.10171).

**Method.** 1B base trained to ~50% of budget on the §1 mixture. Two anneal arms via the Llama 3
protocol (LR linear to 0, 30% candidate / 70% default mix). Randomise data order; repeat if marginal.

**Metric.** long-eval retrieval accuracy at target context.

**Refuted if.** The 2% arm degrades beyond run-to-run noise → restore budget and revisit the §1 reallocation.

**Result.** _pending_

---

## T2 — 40% synthetic Indic is not worse than verified-only (PRE-REGISTERED, not yet run)

**Hypothesis.** An Indic slice at A28/B30/C2/D40 performs comparably to a verified-heavy slice, making the
17% Indic total reachable without exhausting the 64B verified pool.

**Rationale.** This is the most attackable number in the plan: 40% of the Indic lane is synthetic. Raw
supply is 59% synthetic, so 40% is already conservative — but conservative is not evidence.

**Metric.** MILU accuracy per language; IndicGenBench (chrF / exact-match).

**Refuted if.** The synthetic-heavy arm underperforms verified-only → cut D, raise A, and accept a lower
Indic total rather than defending an unbacked share.

**Result.** _pending_

---

## T3 — Reserving agentic data for the anneal beats spending it early (PRE-REGISTERED, not yet run)

**Hypothesis.** Holding the long agentic trajectories (SWE-Gym / SWE-smith / OpenHands, 360M tokens) for
the anneal yields better agentic capability than consuming them during the main run.

**Metric.** SWE-bench Verified % resolved (pass@1); BFCL v3 accuracy.

**Refuted if.** The early-spend arm matches the reserve arm → the anneal reserve earns nothing and the
protected floor alone carries the design. This would remove a core claim of the plan, so it is worth
running early.

**Known limitation, recorded up front.** Llama 3 report anneal gains were **negligible at 405B**
("strong in-context learning ... does not require specific in-domain training samples"). T1–T3 measure
*anneal-stage* value; transferability to a much stronger base is unproven and must be stated, not assumed.

**Result.** _pending_

---

## T4 — Temperature α = 0.72 beats proportional on per-language MILU (PRE-REGISTERED, not yet run)

**Hypothesis.** Allocating the Indic lane by `p_i ∝ S_i^0.72` (water-filled at 4 epochs) beats
proportional allocation (α = 1) on MILU, by moving ~22B from Hindi to the mid-resource languages.

**Rationale.** MILU is a per-language average, so it rewards lifting the weakest language, not the
strongest. Proportional allocation gives every language identical epochs and therefore lets absolute
token count — which spans 29.8B (Hindi) to 6.16B (Assamese) — set per-language capability.

**Method.** Same anneal-as-evaluation protocol as T1–T3. Two arms differing *only* in Indic per-language
weights: α = 1.0 versus α = 0.72. 30% candidate / 70% default mix, LR linear to 0.

**Metric.** MILU accuracy over the 10 Indic languages — reported as **both the mean and the minimum**.

**Refuted if.** Proportional matches or beats α = 0.72 on the mean → the mid-tier gain does not pay for
Hindi's loss; revert to α = 1, which has the added benefit of leaving every language 0.69 epochs of
headroom. A mean improvement driven by Hindi alone also counts as refutation: the rule was designed to
raise the minimum, and a rule that raises the average by improving the strongest language has not done
the thing it was designed to do.

**Result.** _pending_

---

## E3 — Supply corrections from S3/S4 prior work (COMPLETE, 2026-07-31)

**Question.** Our own S3 source map and S4 cleaning measurements predate this plan. Do they change any
number in it?

**Method.** Read `S3/assignment/INDIA_FIRST_DATA_SOURCE_MAP.md` and the S4 deployed report
(https://vedapragna-cs.github.io/era5-s4-data-cleaning/), then verified each claim against the primary
source (HF dataset cards, AWS Open Data registry, Article 348).

**Results.**

1. **Samanantar is double-counted in tier C.** The S5 inventory lists BPCC (3B) and Samanantar (2B) as
   separate rows. Samanantar is a *component of BPCC-Mined* (19.4M pairs inside its 104.4M "existing
   data" block). Tier C is not 5B of distinct data.

2. **BPCC is ~10× larger than the inventory states.** Inventory: 22M samples / 3B tokens. Actual:
   ~230M bitext pairs (BPCC-Mined ~228M, BPCC-Human ~2.2M), plus ~800M back-translation pairs. Indic-side
   real supply ≈ 7B. **The C20 finding survives**: 99B demand against ~7B is 14 epochs, still impossible.
   The published margin corrects from 19.8× to ~14×. Note only 2.2M of 230M pairs are human-translated,
   so tier C is not quality-uniform either.

3. **Curated archives yield 6.7× more usable text than raw web (measured, S4).** Identical 8-stage
   pipeline: Indian Supreme Court judgments **90.6% retention** (27.0M of 29.8M words, 3,848 docs,
   18 languages) versus Common Crawl CC-MAIN-2025-13 **13.5%** (17.3M of 128.4M words). Measured unit:
   **7,744 words/judgment**.
   S4's own framing is the honest one: curated corpora are the *output* of a cleaning pipeline, so 90.6%
   means they **arrive** clean, not that our pipeline adds value.

4. **The court corpus is a reasoning/long-context supply line, not an Indic one.** The AWS High Court
   release is **15.9M judgment PDFs, ~1TB, 25 courts, CC BY 4.0**, anonymous S3 bulk. But **Article
   348(2)** provides that decrees, judgments and orders of a High Court **shall be in English**, even in
   the four states where Hindi is authorised for proceedings (Rajasthan 1950, UP 1969, MP 1971,
   Bihar 1972). The corpus is constitutionally English; regional-language versions exist only as
   after-the-fact SUVAS-cell translations.
   - Size, **extrapolated not measured**: 15.9M × 2,000–7,744 words × ~1.15 tok/word = **37–142B tokens**.
     Against public non-V4 reasoning supply of 7.1B that is **5–20×**.
   - Translated subset (eSCR, ~31,000 judgments, 72% Hindi) ≈ **0.3B tokens** — a 4% uplift to tier C.
     Structurally the same case as CommitPackFT in the code lane: best-in-kind, too small at natural
     weight, so upsample within the tier rather than resize the tier.

5. **S3 asserts a mechanism this plan does not model.** *"Documents receive multiple tags, so a Tamil
   physics lesson satisfies both Tamil and STEM targets even though it is sampled only once."* The S5
   budget treats lanes as disjoint. Overlapping tags would relieve the tightest lane (Indic, 3.31 epochs).
   Unresolved and potentially the largest correction outstanding.

6. **S3 and S5 disagree on the anneal.** S3: 4T budget, Indic 20%, **7.5% anneal**. S5: 3T, 17%, **3%**.
   A reviewer reading both will ask; the reconciliation is owed.

7. **Token counts are not comparable across languages.** S3's MUTANT-Indic fertility ceilings run
   **1.15 tokens/word (English) to 3.75 (Manipuri, Malayalam, Sanskrit, Santali)**. A token-proportional
   per-language allocation therefore under-serves high-fertility languages *in content terms*. E2's α
   allocation is in the wrong unit and should be fertility-adjusted.

**Not obtained.** Per-language token tables for IndicCorpV2. Dedup status of IndicCorpV2 against Sangraha.

**Caveats on (4).** S4's 90.6% was measured on already-extracted Supreme Court text and **does not
transfer** to a 1TB eCourts PDF collection, much of which is scanned and needs OCR. Corpus-scale dedup
over 15.9M judgments sharing cause titles, recitals and citation strings will also cut far harder than it
did over 3,848 documents. Both push the usable figure toward the low end of the 37–142B range.

---

## E4 — The High Court corpus, measured (COMPLETE, 2026-07-31)

**Question.** E3 sized the AWS High Court corpus at 37–142B tokens by applying S4's measured 7,744
words/document — a Supreme Court figure — to 15.9M High Court documents. Is that extrapolation sound?

**It is not.** Corrected below by direct measurement of the live bucket. Reproduce with
`scripts/judgments.py`.

**Method.** `arn:aws:s3:::indian-high-court-judgments` (ap-south-1, CC BY 4.0, anonymous). Enumerated
the parquet metadata layout, then downloaded and text-extracted **122 judgment PDFs** across **20 courts
and 11 years (1995–2025)**. Sampling starts each listing at a random letter inside the bench partition,
because taking the first keys alphabetically biases toward one CNR block.

**Corpus shape (measured).** 1,493 parquet metadata partitions, ~8.0GB of metadata alone, partitioned
`year=/court=/bench=`, **25 courts, years 1950–2026**. PDFs live under `data/pdf/` on the same layout.

**Results (measured, 122 documents).**

| | |
|---|---|
| mean words/document | **781** |
| median words/document | **392** |
| p90 / max | 2,248 / 8,355 |
| mean words/page | 183 |
| extractable text layer | **97.5%** (3/122 without) |
| mean Devanagari | **0.17%** — only 2/122 documents above 5% |
| bytes per word | 145.6 |

**Corpus size — two independent estimators.**

| estimator | words | tokens (× 1.15) |
|---|---|---|
| 15.9M docs × median 392 | 6.2B | **7.2B** |
| 15.9M docs × mean 781 | 12.4B | 14.3B |
| 1TB ÷ 145.6 bytes/word | 6.9B | **7.9B** |

The sample's mean PDF is 113,761 bytes against a corpus mean of 62,893 (1TB / 15.9M), so the sample
skews large and the mean-based figure is an upper bound. The byte-based estimator does not depend on
document count and agrees with the median-based one.

**BEST ESTIMATE: ~7–14B tokens, centre ~8B, before corpus-scale dedup.**

**Correction to E3.** The published 37–142B was **5–18× too high**. Consequence: against public non-V4
reasoning supply of 7.1B, the High Court corpus is roughly **1–2×**, not the "5–20×" claimed in E3. It
is a real and licensed addition to the thinnest lane in the plan; it is not transformative, and E3's
framing of it must not be quoted.

**Three findings that were not assumptions.**

1. **Scanned pages carry an OCR text layer.** 97.5% of sampled PDFs yield extractable text with no OCR
   step of our own. E3 flagged OCR as a cost risk; **measurement withdraws that risk.**
2. **Article 348(2) is confirmed empirically, not just legally.** Devanagari averages 0.17% across the
   sample, and Allahabad, Patna and Rajasthan — the Hindi-authorised courts — return 2.63%, 0.00% and
   0.00%. The corpus is English in practice as well as in law. **It is not Indic supply.**
3. **The `pdf_exists` metadata column is unreliable.** It reads False on **267,214/267,214** rows
   sampled across Patna, Rajasthan, Allahabad and Bombay 2024, while the corresponding PDFs are present
   under `data/pdf/`. Any supply count built on that column would read zero.

**Limitation.** 122 documents against 15.9M. The two estimators agree, but a definitive figure needs a
larger draw and, more importantly, corpus-scale dedup — 15.9M orders share cause titles, recitals and
citation strings, and the p90/median ratio of 5.7 says the distribution is dominated by short orders
whose boilerplate fraction is highest.

---

## E5 — The Supreme Court regional corpus (COMPLETE, 2026-07-31)

**Question.** Does the judgment data contain translations, and how many?

**Found by looking rather than assuming.** The **High Court** bucket has no language dimension — bench
folders are geographic (Gwalior/Indore/Jabalpur, Jaipur/Jodhpur, Calcutta original/appellate side),
enumerated across all 25 courts. The **Supreme Court** bucket splits
`data/pdf/year=YYYY/{english,regional}/`, with the language in the filename suffix
(`2024_10_108_125_HIN.pdf`).

**Results (measured, full enumeration of 177,545 objects, 73.86GB).**

| | |
|---|---|
| regional-language judgments | **134,028** |
| English judgments | 43,517 |
| languages | **18** |
| **document-aligned to an English counterpart of the same stem** | **134,019 / 134,028 = 100.0%** |

| lang | docs | share | | lang | docs | share |
|---|---|---|---|---|---|---|
| Punjabi | 42,068 | 31.4% | | Marathi | 3,731 | 2.8% |
| Hindi | 41,678 | 31.1% | | Kannada | 2,270 | 1.7% |
| Bengali | 13,007 | 9.7% | | Odia | 2,223 | 1.7% |
| Gujarati | 7,324 | 5.5% | | Assamese | 2,111 | 1.6% |
| Tamil | 5,702 | 4.3% | | Nepali | 203 | 0.2% |
| Malayalam | 5,140 | 3.8% | | Sanskrit | 60 | — |
| Telugu | 4,511 | 3.4% | | Konkani 18 · Garo 17 · Khasi 12 · Kashmiri 1 | 48 | — |
| Urdu | 3,952 | 2.9% | | | | |

**Text measurement (40 regional PDFs across 12 languages).** Mean **9,665 words/document**, median
**6,175**; **38/40 yield extractable text**; native-script dominance **82–99%** — genuinely native script,
not romanised.

**Size.** 134,028 × 6,175 median words ≈ 0.83B words → at Indic fertility ~2.0 tok/word ≈ **1.7B tokens**;
at the 9,665 mean, **~2.6B**. Call it **1.7–2.6B tokens of native-script Indic, fully parallel to English.**

**Correction.** This plan previously sized translated judgments at **~0.3B** from a press figure of
~31,000 eSCR judgments. Measured is **134,028 documents and ~6–9× the tokens**, because both the document
count (4.3×) and the words per document (6,175 vs the 392 median measured for High Court orders) were
wrong.

**Why it matters more than its size.**

1. **Tier C rises ~25–37%** (≈7B → ~9B), and the addition is the highest-quality kind: professionally
   vetted judicial translation, **document-aligned**, not mined bitext. Only 2.2M of BPCC's 230M pairs are
   human-translated; this is 134K aligned documents.
2. **It is document-level parallel, not sentence bitext.** English counterparts average ~10,800 words in
   the sample, so this supports document-level translation and cross-lingual summarisation — the
   IndicGenBench task shapes — rather than only sentence-level MT.
3. **Punjabi is the largest language at 31.4%.** Punjabi is the *binding constraint* in the E2 allocation
   — the weakest MILU language, pinned at its 4-epoch cap, and the language that sets α* = 0.72. ~42K
   Punjabi documents ≈ 0.4B tokens is a ~7% uplift to precisely the language that determines the
   allocation.
4. **Garo appears.** 17 documents, but Garo is absent from Sangraha *and* IndicCorpV2. Konkani and Khasi
   likewise appear here.

**Still does not rescue C20.** Tier C at ~9B against a 99B C20 demand is 11 epochs. C2 stands.

**Caveat.** 40 documents measured against 134,028. Byte sizes per language vary ~12× per document
(Urdu 1.4MB/doc vs Kannada 116KB/doc), which suggests a mix of born-digital and scanned originals; the
2/40 that yielded no text are the visible edge of that.

---

## E6 — Structural sweep of every dataset (COMPLETE, 2026-07-31)

**Why.** E5 found 134,028 parallel judgments only because the folder structure was inspected rather than
assumed. That is a process failure, not a lucky find: the previous audit queried *aggregate sizes* and
never enumerated *layout*. Repeated here for every dataset in the inventory.

**Method.** HF `api/datasets/{id}?blobs=true` for file-level sizes, `?full=true` for tree shape; S3
`list-type=2&delimiter=/` walks for the buckets.

### Finding 1 — the Samanantar double-count is now proven, not inferred

BPCC is **109.3GB across 17 components**, and two of them are Samanantar:

| component | files | GB | | component | files | GB |
|---|---|---|---|---|---|---|
| additional | 15 | 46.8 | | bpcc-seed-latest | 22 | 0.76 |
| **samanantar_v2** | 27 | **28.9** | | bpcc-seed-v2 | 22 | 0.73 |
| nllb_filtered | 35 | 22.5 | | ilci | 16 | 0.49 |
| **samanantar_v0.3_filtered** | 22 | **6.4** | | bpcc-seed-v1 / wiki | 88 | 0.53 |
| comparable | 29 | 2.1 | | daily / massive / nllb_seed | 31 | 0.05 |

**Samanantar is 35.3GB = 32% of BPCC.** Listing BPCC and Samanantar as separate tier-C rows counted a
third of the corpus twice. At ~8 bytes/token, BPCC is ~13.7B tokens both-sides ≈ **~6.8B Indic-side**,
Samanantar included and not re-added.

### Finding 2 — BPCC *does* cover the tail, in kind but not in volume

**25 language pairs**, including every tail language I had recorded as having no parallel data:

| language | pair | MB | | language | pair | MB |
|---|---|---|---|---|---|---|
| Sindhi | snd_Arab | 40.2 | | Bodo | brx_Deva | 0.4 |
| Kashmiri | kas_Deva | 3.7 | | Santali | **sat_Olck** | 0.4 |
| Kashmiri | kas_Arab | 2.6 | | Manipuri | mni_Mtei | 0.4 |
| Maithili | mai_Deva | 1.7 | | Dogri | doi_Deva | 0.4 |
| | | | | Konkani | gom_Deva | 0.3 |

Corrects "BPCC and Samanantar are the only parallel corpora and the tail has none" — the tail **is**
covered, at 0.3–1.7MB per language (~40K–200K tokens). Coverage without volume. **The conclusion that the
tail is a supply problem stands, but the reason is now precise: parallel data exists for every one of
them, in quantities three orders of magnitude below use.**

### Finding 3 — the tail is fragmented across scripts, which halves a corpus that is already negligible

BPCC carries **Kashmiri in both Arabic and Devanagari, Manipuri in both Bengali and Meetei Mayek, Sindhi
in both Arabic and Devanagari**, and Santali in Ol Chiki. Sangraha and IndicCorpV2 each pick *one* script
per language, and not always the same one. So tail supply is not merely tiny, it is **split across
incompatible scripts**, and a script choice must be made per language before any of it can be pooled.
This was invisible at the aggregate-size level.

### Finding 4 — Indic instruction data exists and this plan has no lane for it

`ai4bharat/indic-align` — **17.5GB ≈ 2.2B tokens**, absent from the budget entirely:

| sub-collection | GB | | sub-collection | GB |
|---|---|---|---|---|
| wiki_chat | 10.6 | | oasst | 0.47 |
| indowordnet | 5.3 | | dolly | 0.16 |
| indicsharellama | 0.92 | | anudesh | 0.04 |

The budget covers pretraining and an anneal. The session's lifecycle also has **SFT (<1%)**, and for a
3T run <1% is up to ~30B tokens — against which 2.2B of native Indic instruction data is the *only*
supply of its kind we have found. Not costed anywhere in the plan.

### Finding 5 — Aksharantar is transliteration, and it covers the tail too

**729MB across 21 languages**, including brx 1.0MB, doi 0.1MB, kas 1.2MB, mni 0.3MB, mai 7.1MB,
kok 17.4MB. This is the data anchor for the romanised sub-lane left open in `INDIC-ALLOCATION.md` §10 —
and it is **transliteration (script conversion), not translation (meaning conversion)**, so it buys
script handling and code-mixed input, never cross-lingual reasoning.

### Finding 6 — speech corpora are the largest untapped tail source

`ai4bharat/Rasa` (1,319 files) and `ai4bharat/IndicVoices` (1,501 files) are organised **by language**,
and Rasa's largest single language is **Bodo (125 files)** — a language with 1.5M tokens of text in
Sangraha. Maithili, Konkani and Sanskrit are likewise well represented. Speech is not text tokens, but
**ASR transcription is a supply route to exactly the languages where text does not exist**, and it is the
only route found so far that is not rate-limited by what was already written down.

### Also confirmed

Sangraha's layout matches the token table: synthetic 1,792 files = 28 dirs × 64 shards, verified 538,
unverified 212. The High Court bucket has **no language dimension** — bench folders are geographic across
all 25 courts, verified by enumeration.

**Process change.** Enumerate layout before querying size, for every source. Aggregate size answers "how
much" and hides "of what" — which is where four of the six findings above were sitting.

---

## T5 — Should reasoning happen in English or natively? (PRE-REGISTERED, not yet run)

**Why this exists.** The plan maps the Indic lane to MILU and IndicGenBench, both scored
answer-in-native, and never specifies whether the *reasoning* is Indic. There is **no data anywhere in
the inventory with the target shape** — Indic question → reasoning → Indic answer. IndicAlign is
monolingual instruction data, not cross-lingual traces. So the shape has to be synthesized, and which
shape to synthesize is an open design decision, not a detail. Costing in `scripts/xlingual.py`.

**Background.** English-centred internal reasoning is the *default* for multilingual models, not
something we engineer: Wendler et al., *Do Llamas work in English?* (arXiv:2402.10588), and the
cross-lingual collapse literature, find models translate into English representations internally. Two
documented failure modes: chains-of-thought drift toward the dominant pretraining language and drag the
**output** with them, and an entry-stage translation error propagates through everything downstream.

### Arms

- **A — English-internal.** Indic question, **English chain of thought**, Indic answer.
- **B — native.** Indic question, **Indic chain of thought**, Indic answer.
- **C — control.** No cross-lingual data. Without it neither arm can be shown to help at all.

### Cost (measured arithmetic, not estimates)

Fertility is the S3 MUTANT-Indic ceiling weighted by the α = 0.72 allocation: **1.858 tokens/word Indic
against 1.150 English**, so the same content costs **1.616×** more tokens in Indic. CoT length is the
lane-weighted mean of the measured L0–L3 bands = **112 tokens**; an English example is Q 80 + CoT 112 +
A 20 = **212 tokens**.

| per example | Arm A | Arm B | B/A |
|---|---|---|---|
| trained tokens | 273 | 342 | **1.25×** |
| of which translated | 162 | 342 | **2.12×** |
| translated share | 59% | 100% | |

At a cross-lingual slice of **10% of the reasoning lane = 27.8B tokens**:

| at scale | Arm A | Arm B | B/A |
|---|---|---|---|
| examples needed | 102M | 81M | 0.80× |
| tokens to translate **and verify** | **16.4B** | **27.8B** | **1.69×** |
| English source drawn from AON | 21.5B (28%) | 17.2B (22%) | |

Both arms are **affordable from AON alone** — neither needs more than 28% of our 78B English reasoning
corpus, and translation quality can be calibrated against the 134,028 document-aligned human-vetted
judgment pairs (E5) plus BPCC's 2.2M human pairs.

### The cost gap is not what decides this

Arm B costs 1.69× the translation volume, but the binding problem is **verifiability**. Arm A translates
questions and short answers — frequently integers or multiple-choice options, checkable by exact match,
the same *verifier-is-the-label* rule the agentic programme runs on. Arm B translates free-form reasoning
chains, where a **fluent mistranslation is indistinguishable from a correct one** and errors compound
across steps. Arm B is not 1.69× harder; it is qualitatively harder to trust.

### Metrics

1. MILU accuracy over the 10 Indic languages — **mean and minimum**.
2. **Split by MILU's India-specific versus general subjects.** This is the T2 cut and it is the crux: if
   English-internal reasoning loses culturally grounded knowledge, it shows up here and nowhere else.
3. **Answer-language fidelity** — does a Hindi question get a Hindi answer? Measurable by script detection.
4. **Reasoning-language** — what language is the emitted CoT actually in? Also script detection.
5. IndicGenBench (chrF / exact-match) on the 19-language funded subset.

### What refutes what

| outcome | reading |
|---|---|
| B beats A on MILU **India-specific** subjects beyond noise | native reasoning earns its 1.69×; adopt B despite the verification problem |
| A ≈ B on all metrics **and** B's emitted Indic CoT does not raise metric 4 | **cross-lingual collapse** — B pays 1.69× for a veneer; adopt A |
| A shows answer-language drift (English answers to Indic questions) | A is refuted regardless of accuracy; the model must answer in the asked language |
| Neither A nor B beats **C** | cross-lingual synthesis is unfunded work; drop the slice and return 27.8B to the lane |

That last row is why C exists. Both arms could be worse than doing nothing, and the plan should be able
to find that out for the price of a third arm.

### Protocol

Same anneal-as-evaluation as T1–T4: 1B base to ~50% of budget, three arms differing **only** in the
cross-lingual slice, 30% candidate / 70% default, LR linear to 0. Randomise data order; repeat where
marginal. Arms are matched on **trained tokens, not example count** — Arm B gets 81M examples to Arm A's
102M, which is the honest equal-compute comparison.

**Known limitation.** Latent-language behaviour is a property of scale and of pretraining-mixture
dominance. A 1B proxy may not exhibit the English-centred pathway the same way a production model does,
so a null result at 1B is weaker evidence than a positive one. Stated up front rather than discovered
afterwards.

**Result.** _pending_

---

## E7 — Multi-tagged lanes (COMPLETE, 2026-07-31)

**Question.** S3 asserts a mechanism this plan never modelled: *"Documents receive multiple tags, so a
Tamil physics lesson satisfies both Tamil and STEM targets even though it is sampled only once."* Every
S5 budget table assumes lanes are **disjoint**. If they are not, the tables are wrong. Model in
`scripts/multitag.py`.

**The arithmetic.** Under multi-tagging, lane shares need not sum to 100%; they sum to **m × 100%**,
where m is the mean number of tags per token. Distinct tokens required for the same coverage is
`T × Σshares / m`. **The relief is exactly m.** m is not a modelling choice — it is a property of the
corpus, and it must be measured before any share is raised on the strength of it.

**Result: m = 1.020.** Total unique supply 6,282.9B, total tokens carrying two or more tags 126.8B.
Multi-tagging would let shares sum to **102.0%**, not the 130% the mechanism implies at first reading.

| pair | overlap | shared | basis |
|---|---|---|---|
| long-context × code | 60.0% | 60.0B | **inventory** — 60B of the 100B lane *is* repo-packed code |
| long-context × web | 40.0% | 40.0B | **inventory** — the rest is book-length packed prose |
| agentic × code | 50.0% | 0.3B | **inventory** — trajectories are patches and repo files |
| STEM × reasoning | 15.0% | 12.8B | estimate |
| **Indic × STEM** | **8.0%** | **11.7B** | **estimate** |
| Indic × code | 1.0% | 1.6B | estimate |
| Indic × reasoning | 0.5% | 0.4B | estimate |
| Indic × web | 0.0% | 0.0B | **inventory** — the web lane is DCLM/FineWeb, English |

**Why it does not help.** The overlap concentrates in exactly the wrong lanes.

| lane | demand | disjoint ep | tag-inclusive ep | change |
|---|---|---|---|---|
| Indic | 494.7B | 3.15 | 2.90 | −0.25 |
| reasoning | 261.9B | 3.08 | 2.66 | −0.41 |
| STEM | 349.2B | 2.39 | 2.05 | −0.34 |
| long-context | 58.2B | 0.58 | **0.29** | −0.29 |
| code | 698.4B | 0.63 | 0.60 | −0.03 |
| agentic | 58.2B | 92.4 | 61.6 | still infeasible |

**Indic and reasoning do not intersect anything, by corpus construction.** The web lane is DCLM and
FineWeb-Edu (English), the STEM lane is peS2o and proof-pile-2 (English papers and English mathematics),
the reasoning lane is AON (English). Sangraha intersects none of them. Meanwhile the lanes that *do*
overlap heavily — code, long-context, agentic — already sit at 0.63 epochs or below, where relief buys
nothing.

**Robustness.** The Indic × STEM figure is an estimate, and the conclusion depends on it, so §5 of the
script inverts the question: to move Indic from 3.15 to **2.5** epochs, **28% of the entire STEM corpus
would have to be Indic-language**; to reach 2.0, **62%**. peS2o and proof-pile-2 are English. The
conclusion survives the estimate being wrong by 3×.

**The finding that actually matters: multi-tagging ADDS an unfunded requirement.** A disjoint model
cannot express a cross-lane constraint, so it never checked S3's own:

> *"At least 15% of STEM data should be Indic-language."*

STEM demand 349B × 15% = **52.4B of Indic-tagged STEM** against ~11.7B available at the estimated 8%
overlap — a **4.5× shortfall**, requiring **36% of all Indic supply to be STEM content**. For crawl-derived
Indic text that is implausible.

S3's companion requirement — *"at least 55% of India-first data should be non-English"* — cannot be
checked at all, because S5 has no India-first lane. S3 allocates it 9%. That is a structural gap between
the two plans, not a number to adjust.

**Conclusions.**

1. **The disjoint model is retained.** It is not an approximation we are stuck with; it is correct to
   within 2% for this inventory. The S5 budget tables stand.
2. **Multi-tagging is a tagging and evaluation mechanism here, not a budget mechanism.** It is the right
   way to *express* cross-lane requirements and to route a document to more than one target — it is not a
   source of extra tokens.
3. **State this explicitly in the submission.** Multi-tagging is the most natural way to make an
   infeasible budget look feasible, since every scarce token can be counted twice. A reviewer is entitled
   to assume it has been used that way unless m is reported. **m = 1.020.**
4. Long-context is *more* over-budgeted than previously stated: tag-inclusive supply is 200B and the lane
   runs at **0.29 epochs**.

**What would settle the estimates.** Classify a sample of Sangraha with an Indic-capable STEM/subject
classifier to measure Indic × STEM directly. S3 warns that English-trained filters must never be the
final gate for Indic data, so this needs a native-labelled quality set per language — which makes it a
real task, not a script.

---

## E8 — Fertility-adjusted allocation and the script decision (COMPLETE, 2026-07-31)

**Question.** E2 allocated the Indic lane in tokens. S3's MUTANT-Indic fertility ceilings say the same
content costs 1.375 tokens/word in Hindi and 3.025 in Malayalam. Is the allocation in the wrong unit?

**Yes.** Under the E2 allocation Malayalam receives **45% of Hindi's tokens and 20% of its content**.
MILU scores comprehension, not token counts. Reproduce with `scripts/indic_fertility.py`.

**Method.** Allocate over content — `content_i ∝ C_i^α` where `C_i = S_i / f_i`, hence
`tokens_i ∝ C_i^α · f_i` — water-filled against the same 4×supply **token** cap. The cap is
fertility-invariant (`tokens ≤ 4S` and `content ≤ 4S/f` are the same constraint), so only the shape moves.

**Results (measured).** **α\* falls 0.72 → 0.59.** Hindi −14.2B, Bengali −4.9B, Urdu −2.0B fund gains to
Malayalam +4.7B, Telugu +3.6B, Tamil +3.3B, Kannada +2.4B, Sanskrit +2.3B. No language breaches the cap.

**The finding that matters.** Malayalam's content rises from **20% to 29%** of Hindi's and stops, because
it hits the 4-epoch ceiling first. Generally: **high-fertility languages need more tokens per unit of
content and therefore exhaust their repetition budget sooner** — the languages most disadvantaged by a
token-based rule are the least able to absorb the correction. **10 of 22 languages sit pinned at 4
epochs** afterwards. **Content parity is not reachable at this lane size**, and a plan that allocated by
content while implying parity would claim something the epoch ceiling forbids.

**Script decision.** Rule: (a) use the script the claimed benchmark scores in; (b) else the script with
more supply; (c) report what is stranded rather than pooling it.

| language | scripts | IndicGenBench | chosen | basis |
|---|---|---|---|---|
| Manipuri | Beng / Mtei | Meetei Mayek | **Mtei** | benchmark |
| Santali | Ol Chiki | Ol Chiki | **Olck** | benchmark |
| Kashmiri | Arab / Deva | not covered | **Arab** | supply |
| Sindhi | Arab / Deva | not covered | **Arab** | supply |

Manipuri is the live case: BPCC carries both scripts, IndicCorpV2 only Mtei, and the benchmark decides
it — stranding the Bengali-script material. Kashmiri and Sindhi are outside IndicGenBench's 29, so the
choice is **unvalidated by any benchmark we claim**, which is how it is recorded.

**Limitation.** Fertility figures are S3 group midpoints from MUTANT-Indic — **stated, not measured
here**, and they are *prose* ceilings. S3 itself warns that words are the wrong unit for code, LaTeX and
tool traces, so this adjustment applies to the Indic prose lane and not beyond it. Measuring per-language
fertility against our own tokenizer would settle it and belongs with the S2 tokenizer work.

---

## E9 — Cross-lane requirements: India-first, Indic-STEM, Indic SFT (COMPLETE, 2026-07-31)

**Question.** S3 states three requirements that cut across lanes and S5 cannot express any of them. Do
they change the budget? Full treatment in `CROSS-LANE.md`, reproduce with `scripts/indiafirst.py`.

**Result 1 — India-first is a tag, not a lane.** At S3's 9% share it demands 262B against **8.46B** of
open English-language India-first supply = **31 epochs**. It is the second unfundable lane in the plan
after agentic, and it comes from our own prior work, which assigned the share without a supply check.
Fundable at 4 epochs: **1.16%**. Everything else in the source map is gated and carries no number.

But India-first is a *property* of documents already counted elsewhere — a judgment is India-first and
reasoning; a Tamil news article is India-first and Indic. As a coverage tag it is **over-delivered**:
18.2% of trained tokens at φ = 1, **9.7% at φ = 0.5**, where φ is the fraction of Indic text about Indian
context. S3's "≥55% non-English" rule passes at **88–94%**.

**S3 and S5 never disagreed.** S3 modelled it as a lane because S3 assumed multi-tagging; a disjoint
budget cannot book the same content twice. Adopted: **no India-first lane; a ≥9% coverage target**
delivered by the Indic lane plus an explicit ~1.2% English-India slice inside the reasoning lane, and
**φ must be measured** before the claim is unconditional.

**Result 2 — the Indic-STEM requirement is genuinely unfunded.** 15% of a 349.2B STEM lane = **52.4B**
against ~11.7B available. **4.5× short; supply supports 3.4%.** Chosen resolution: **source it** —
NCERT/SCERT/NPTEL/IGNOU are Indian-curriculum STEM in Indian languages and are gated — with lowering the
requirement to 3.4% as the stated fallback. This is the plan's strongest case for prioritising **one
specific negotiation**, and it is how a gated source should enter a budget: as a programme dependency
with a number attached to what it unlocks, never as supply.

**Result 3 — Indic SFT: volume covered 4.4×, provenance 48× short.** IndicAlign is 17.5GB ≈ 2.19B tokens
against S3's 0.5B target. But S3 also requires half of Indic instructions to be **natively authored**, and
only `anudesh` (0.042GB) is — **0.2% of the corpus**. Against 0.25B needed we hold ~0.005B.

**The general finding.** For Indic data the binding constraint is repeatedly **provenance, not volume**.
Tier D is translated Wikipedia; IndicAlign is 99.8% derived or translated; the verified tier contained
English. Every corpus that looks large is large because something was translated or derived, and the
natively-authored fraction is consistently two to three orders of magnitude smaller. **A plan that sizes
Indic capability in tokens will keep concluding it is funded when it is not.**

**Effect on the budget: none.** Main-run shares stand at 34/24/17/12/9/2/2. Three structural objections
were raised and none moved a number — the claims around the budget changed, not the budget.

**Limitation.** φ is unmeasured. The India-first coverage claim is conditional on φ ≥ 0.5 and needs a
classified sample of the Indic lane; S3 warns English-trained classifiers must not be the final gate for
Indic data, so this needs native-labelled sets per language — the same dependency as the Indic × STEM
overlap measurement in E7.

---

## E10 — Gate the tail instead of funding it (DECIDED, 2026-07-31)

**Question raised.** "Drop the tail languages if we can't do a reasonable job. We will get more data
later, so what we need is a plan that works for us — or do you think data decides the plan?"

**The answer that governs this entry.** Data decides what we may **claim**. It does not decide what we
**build for**. Every lane in this plan was sized to a July-2026 snapshot of public supply as though that
were a physical constant — but the opposite move, "more data will arrive", is the wishful accounting the
rubric grades against **unless the plan states how much, from where, and what happens if it does not**.
So lanes become **conditional**: funded share, trigger, fallback. Reproduce with `scripts/indic_gated.py`.

**The threshold is read off the data, not chosen.** Ranking supply shows the distribution is not a smooth
tail — it is 14 languages and then a **24× cliff**: Punjabi 6.20B, **Assamese 6.16B → Sindhi 0.26B**,
Maithili 0.01B.

**Entry gate: ≥1.30B unique native-script tokens** — the level at which a language can reach 1% of the
lane without breaching 4 epochs. All 14 above the cliff clear it (min 6.16B); none of the 8 below come
within 5×.

| gated | supply | gap to gate | IndicGenBench | route |
|---|---|---|---|---|
| Sindhi | 258.2M | 5× | not scored | text / partnership |
| Maithili | 14.6M | 89× | scored | Rasa + IndicVoices |
| Konkani | 10.1M | 129× | scored | Rasa |
| Manipuri | 7.4M | 176× | scored | IndicVoices |
| Bodo | 1.5M | 866× | scored | **Rasa's largest language** |
| Kashmiri | 0.5M | 2,600× | not scored | text / partnership |
| Santali | 0.3M | 4,332× | scored | text / partnership |
| Dogri | 0.06M | 21,662× | not scored | text / partnership |

**Cost, stated plainly: IndicGenBench falls 19/29 → 14/29.** We stop claiming Bodo, Konkani, Maithili,
Manipuri, Santali. MILU unaffected at 10/10. Freed budget is 1.17B = **0.225% of the lane**, which is not
the reason. **19/29 with five languages funded at ~30M tokens is a weaker claim than 14/29 funded
properly** — a reviewer who checks Santali finds 7.7M tokens behind a claimed benchmark language.

**The trigger is what makes this a plan rather than a snapshot.** A gated language re-enters
automatically on crossing 1.30B. No redesign: the allocation rule is defined for any number of languages,
so adding one is a re-run.

**The distinction that matters most.** The two halves need **different kinds of "later"**:

- **passive** — the 14 funded languages grow as the web grows;
- **active** — the 8 gated ones do not. Almost nothing is being written down in Santali or Dogri at
  volume. Their supply grows **only if we fund collection**, and speech is the cheapest route: Rasa's
  single largest language is Bodo and IndicVoices carries Manipuri.

Treating those two as the same kind of "later" is how a plan ends up waiting for data that will never
arrive on its own.

**A 0.006% share was never a commitment to Manipuri. A threshold is.**

---

## E11 — The budget restated as funded / trigger / fallback (COMPLETE, 2026-07-31)

**Why.** E10 made the Indic tail conditional. The same logic applies to the whole budget: flat
percentages hide which shares rest on data we hold and which rest on work not yet done. Written into
`BUDGET-PROPOSAL.md` §8, reproduce with `scripts/conditional.py`.

**Result 1 — the headline changes.** Comparing each requested share against what its supply supports at
4 epochs: **six of seven lanes are funded from supply we hold today.** Only agentic is not — 2% requested
against 0.087% supportable, **23× short**.

> **98% of the main run is funded; 2% rests entirely on a named synthesis programme.**

That is a materially different statement from "100% allocated", and it locates all the delivery risk in
the budget in one lane — the lane the session's protected floor exists to defend.

**Result 2 — the V4 dependency was invisible.** 78B of the reasoning lane's 85B is our own AON corpus, so
a public-only rebuild supports **1.0%, not 9%**. Losing the V4 corpora would open an **8.0% hole** in the
run. "We hold AON" is load-bearing and does not appear anywhere in a flat percentage table. The
court-judgment ingest (~8B, CC BY 4.0) is the only item in the whole audit that materially moves the
public-rebuild floor.

**Result 3 — web is the adjustment variable and should be named as one.** 34% against 4,691B is 0.21
epochs; it could support 645% of the run alone. Every fallback is absorbed here and every trigger draws
from here. **Web is not 34% because 34% is optimal — it is 34% because it is the only lane with slack.**
Stating that is more useful than defending 34% as tuned. **Hard floor set at 25%**, below which fallbacks
must come from elsewhere, because the session warns that cutting web too far gives a model whose code
compiles and does not work.

**Nine triggers recorded**, each moving exactly one share. None requires a redesign: the allocation
machinery (IPF across stages, content-weighted temperature across languages) is defined for any input, so
a trigger firing is a re-run.

**Effect on the numbers: none.** As with E7 and E9, the restructure changed what the budget *says*, not
what it allocates. Three passes now have left the shares at 34/24/17/12/9/2/2.

---

## E12 — Two "open questions" that were only unfinished (COMPLETE, 2026-07-31)

### The STEM supply "discrepancy" is a category error, not a conflict

**The plan has been carrying this as the one number where we differ from the course material without
knowing why.** Resolved by reading the inventory widget's own band structure.

**There is no STEM band.** The inventory has **six** groups — `agentic, code, general, indic, long,
reason` — and `peS2o` and `proof-pile-2` are both tagged **`g:'general'`**. The widget's band is literally
named *"General web & STEM"*, and reasoning's is *"Reasoning & math"*.

| band | total | rows |
|---|---|---|
| general | **4,837B** | DCLM 2,600 · FineWeb-Edu 1,300 · D2 Web 627 · D1 Web 164 · **proof-pile-2 55** · **D4 STEM 49** · **peS2o 42** |

The composer invents a **seventh** lane called STEM and assigns it 250B by splitting that 4,837B band.
**No dataset row is tagged STEM anywhere.** Our figure is the sum of the three STEM-content rows inside
`general`: 55 + 49 + 42 = **146B exactly**. And **4,837 − 146 = 4,691B**, which is precisely the web
supply figure this plan uses.

**So the decomposition is confirmed rather than disputed.** The composer's 250B was never a supply figure;
it is a lane-level assumption about how to split a combined band. Our 146B is the auditable subset, and
the two numbers were never measuring the same thing. *(Noted in passing: the composer's own split —
web 4,500 + STEM 250 = 4,750B — does not reconcile with the band's 4,837B either, losing 87B.)*

**This removes an open item and a stated weakness from the submission.**

### IndicCorpV2 per-language, past the conversion cap — and a third English split

E3 could not get per-language sizes for the 11 major languages because the HF datasets-server converts
only 37.6% and pins large splits at ~5GB. **The raw file listing bypasses it entirely.**

| | GB | | | GB |
|---|---|---|---|---|
| **en** | **39.74** | | sa | 3.11 |
| hi (3 files) | 79.98 | | or | 2.18 |
| ml | 26.32 | | as | 1.18 |
| kn | 19.58 | | gom | 0.53 |
| bn | 15.99 | | kha | 0.27 |
| te | 15.76 | | mai | 0.20 |
| ne | 14.77 | | sd | 0.11 |
| mr | 14.55 | | sat | 0.06 |
| gu | 14.13 | | bd | 0.05 |
| ta | 11.79 | | mni | 0.01 |
| pa | 9.48 | | dg, ks | ~0.00 |
| ur | 5.59 | | **total** | **275.39** |

**`data/en.txt` is the single largest file in the corpus at 39.74GB = 14.4%.** This is the **third**
corpus in the audit with a large English component counted as Indic supply, after Sangraha's verified
tier (12.76B) and tier D's translated-English provenance.

**Correction:** IndicCorpV2 Indic-only is ~**17.9B**, not 20.9B. Tier B falls **44.9B → 41.9B**, and its
epochs rise 3.3 → **3.53**. Better byte-per-token calibration from real file sizes: **11.3**, replacing
the 8.0 extrapolated in E6 — which lowers the tail estimates slightly (Konkani 67.7M → 47M) and **leaves
the gate decision unchanged**, since every gated language remains orders of magnitude below 1.30B.

**The pattern is now a finding, not a coincidence.** Three of three major Indic corpora bundle English or
English-derived content into a headline "Indic" figure. **Any Indic supply number taken from a corpus
card should be assumed to include English until the per-language table is checked.**

---

## E13 — Romanized Indic scoped; judgment dedup caveat withdrawn (DECIDED, 2026-07-31)

### Romanized Indic: a 1% sub-lane, claiming Dakshina

The ~81B Latin-script half of Sangraha's synthetic tier was excluded from the Indic lane because MILU and
IndicGenBench score native script. It has been sitting **neither claimed nor disclaimed** — the one
position that cannot be defended. Resolved.

**A benchmark exists.** Candidates checked:

| benchmark | covers | verdict |
|---|---|---|
| **Dakshina** (Google, LREC 2020, **CC BY-SA 4.0**) | 12 South Asian languages; ~300K word pairs + ~120K sentence pairs; single-word transliteration, full-sentence transliteration, LM of native and romanized text | **claimed** |
| GLUECoS (Microsoft, ACL 2020) | 11 datasets, 6 tasks, En-Hi and En-Es | **not claimed** — word-level classification (POS, NER); its own critics note it cannot evaluate high-level reasoning, and only Hindi is Indic |
| Indi-RomCoM (arXiv:2606.30790) | **4 Indic languages**, 7 instruction-following tasks, 3 controlled code-mixing intensity levels | **noted, not claimed** — preprint (29 Jun 2026), 4 languages only |

Dakshina covers 12 languages; **10 are in our funded 14** (Sindhi is gated, Sinhala is out of scope).

**Indi-RomCoM's headline result is the argument for doing this at all:** *"LLMs consistently underperform
on RCM instructions, with performance degrading as code-mixing density increases."* That is a documented
weakness in existing models on the exact input form Indian users produce — a differentiator, not a
box-tick. We do not claim the benchmark (4 languages, unreviewed preprint), but the finding is why the
sub-lane earns its 1%.

**The capability is asymmetric, and that decides the size.** Reading romanized Indic input matters
enormously — it is how a large share of Indian users type. *Generating* romanized output matters far less
for a coding and agentic model, whose output is code plus an explanation in English or native script.
**Input robustness needs far less data than generation fluency**, so 81B (2.8% of the whole run) for one
benchmark is indefensible.

**Decision: 1% of the Indic lane ≈ 5B tokens**, drawn from **Aksharantar** (729MB, 21 languages, includes
the tail) for explicit transliteration mapping plus a slice of Sangraha's Latin half.

- **No new budget.** It comes out of the existing 17%, so no other lane moves.
- **Epoch check: 5B against ~81B available = 0.06 epochs** — the only Indic sub-lane with real slack.
- **Effect on native-script Indic:** demand 495B → 490B, epochs **3.15 → 3.12**. Marginally *easier*.

### The judgment dedup caveat was over-stated — withdrawn

E4 and E5 both carried a warning that corpus-scale dedup "will cut far harder" on the judgment corpora
than it did over a 3,848-document sample. **That framing was wrong**, and our own S4 measurement already
answered it: the S4 pipeline **includes MinHash/LSH** and returned **90.6% retention** on Supreme Court
judgments against 13.5% on Common Crawl.

Court judgments are **distinct legal documents**. There is no near-duplicate-document problem of the kind
web crawl has, where one article is syndicated across fifty sites, so document-level dedup does not
collapse the corpus — and S4 measured exactly that.

**What remains true is a weaker and different claim:** judgments share cause titles, standard recitals and
citation strings, so informational density per token is lower than the raw count implies. But that
boilerplate **is** legal register — a capability the reasoning lane wants, not noise to be stripped. It
does not reduce the usable token count.

**Net: the ~8B High Court estimate stands without a dedup discount.** The E4/E5 caveat is superseded.
