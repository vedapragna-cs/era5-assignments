# ERA V5 — supply audit

The lane budgets were built on the course's dataset inventory. Auditing that inventory against the
sources themselves has now corrected **five** of its rows, four of them in the direction that makes the
plan smaller. This document is the audit, source by source, with every number labelled by how it was
obtained.

Reproduce with `scripts/supply_audit.py` (live queries) and `scripts/judgments.py` (corpus measurement).

**Labels.** `measured` — we queried or computed it here · `stated` — the source publishes it and we could
not independently verify · `gated` — real but unobtainable without an MoU, so it carries a rights tier and
**no number** · `withdrawn` — a figure this plan published and has since disproved.

---

## 1. What auditing changed

| # | claim as published | audited | direction |
|---|---|---|---|
| 1 | Indic verified supply 64.3B | **51.5B** — 12.76B of it is an English split | ↓ |
| 2 | Indic synthetic supply 162.7B | **81.4B** native-script; the rest is a 1:1 transliteration | ↓ |
| 3 | Tier C = BPCC 3B + Samanantar 2B | **Samanantar is a component of BPCC** — double-counted | ↓ |
| 4 | BPCC = 22M samples / 3B tokens | **~230M bitext pairs**, ~7B Indic-side | ↑ |
| 5 | Court judgments = 37–142B tokens | **~8B** measured | ↓ (**withdrawn**) |

Four of five went down. That is the expected direction: inventories are assembled from headline figures,
and headline figures are the ones that flatter.

## 2. Sangraha Synthetic is one encyclopedia rendered 28 ways

The single largest row in the Indic inventory — 162.7B tokens, 59% of all Indic supply, and 40% of our
proposed lane — is described by AI4Bharat as *"Wikimedia English translated to 14 Indic languages and
further romanized from 14 languages by transliteration to English."* **(stated)**

Three consequences, and they compound:

1. **It is translated English, not native Indic.** The content distribution is English Wikipedia's, which
   is global and Western-weighted.
2. **It is one source.** English Wikipedia is ~4.5B words (2024). The 162.7B-token tier renders that
   single corpus **28 ways** (14 languages × 2 scripts). Its distinct informational content is one
   encyclopedia. **(measured: the synthetic config's native and Latin splits have identical row counts —
   `hin_Deva` and `hin_Latn` are both 5,775,143.)**
3. **The epoch figure hides this.** Tier D at 40% of the lane demands 208B against 81.4B native-script
   supply, which reads as a comfortable 2.4 epochs. In *content* terms it is far heavier repetition of a
   single corpus, and the repetition ceiling in arXiv:2305.16264 is about repeated **content**, not
   repeated token strings.

**This sharpens T2 into a much better experiment.** MILU is India-centric by construction — regional and
state examinations, local history, arts, festivals, law. Translated English Wikipedia is precisely the
wrong content distribution for those subjects and roughly the right one for MMLU-style general science.
So the prediction is not "synthetic is worse" but **"tier D underperforms on MILU's India-specific
subjects while holding up on its general ones."** That is falsifiable per-subject, and if it holds, the
fix is not less synthetic data but *differently sourced* synthetic data.

## 3. The tail, re-measured

IndicCorpV2's per-language table has never been published. It is recoverable from the HF datasets-server
size API, with one trap: the conversion is **partial (37.6%)** and **11 splits are pinned at a ~5GB cap**,
so their sizes are artefacts. The **12 splits below the cap are complete — and they are exactly the tail
languages.** Byte-to-token calibration: extrapolated full size 167GB against the published 20.9B tokens
gives **8.0 bytes/token**. **(measured)**

| language | Sangraha | IndicCorpV2 | combined | uplift | max @ 4 epochs |
|---|---|---|---|---|---|
| Sindhi | 258.2M | 14.0M | 272.2M | 1.1× | 1,089M |
| Konkani | 10.1M | 67.7M | 77.8M | **7.7×** | 311M |
| Maithili | 14.6M | 25.2M | 39.8M | 2.7× | 159M |
| **Khasi** | — | 34.8M | 34.8M | **NEW** | 139M |
| Manipuri | 7.4M | 1.2M | 8.6M | 1.2× | 34M |
| Santali | 0.3M | 7.4M | 7.7M | **25.7×** | 31M |
| Bodo | 1.5M | 5.7M | 7.2M | 4.8× | 29M |
| Kashmiri | 0.5M | 0.1M | 0.6M | 1.2× | 2.4M |
| Dogri | 0.06M | 0.2M | 0.2M | 4.0× | 1.0M |
| **total** | **292.7M** | **156.3M** | **449.0M** | **1.5×** | **1.80B** |

Santali gains 25.7× and Konkani 7.7×, and **Khasi appears at all** — it is absent from Sangraha entirely.

**The conclusion is unchanged.** At its full 4-epoch ceiling the entire tail absorbs **1.80B = 0.345% of
the Indic lane**, up from 0.225%. Measurement made the tail 1.5× larger and left it microscopic. The tail
is a supply problem; no allocation rule reaches it.

**On the double-count risk:** Sangraha Unverified is drawn from "existing multilingual corpora" and could
in principle re-contain IndicCorp — but it has splits for **only the 14 major languages**, so none of the
tail above can be double-counted there. Residual overlap risk is against Sangraha *Verified*, which was
built by a different route (human-verified sites, PDF OCR, video transcription) than IndicCorp's crawl.

## 4. Court judgments — measured, and reclassified

Full measurement in `EXPERIMENT_LOG.md` E4. Summary: **15.9M documents, ~1TB, 25 courts, 1950–2026,
CC BY 4.0, anonymous bulk access.** Sampling 122 PDFs across 20 courts and 11 years gives mean **781
words/document**, median **392**. Two independent estimators — document count × median, and 1TB ÷ 145.6
bytes-per-word — agree on **~7–8B tokens**. **(measured)**

- **Not Indic supply.** Article 348(2) requires High Court judgments to be in English even where Hindi is
  authorised for proceedings, and measurement confirms it: Devanagari averages **0.17%**, with the
  Hindi-authorised courts returning 2.63% (Allahabad), 0.00% (Patna), 0.00% (Rajasthan).
- **Reasoning and long-context supply, in English.** Against public non-V4 reasoning supply of 7.1B this
  is ~1–2×. Meaningful, not transformative — and the "5–20×" this plan previously published is
  **withdrawn**.
- **No OCR cost.** 97.5% of sampled PDFs carry an extractable text layer. A previously-flagged risk,
  withdrawn by measurement.
- **Do not use the `pdf_exists` column.** It reads False on all 267,214 rows sampled while the PDFs are
  present under `data/pdf/`.

Translated judgments (eSCR, ~31,000, 72% Hindi) are ~0.3B tokens — the best tier-C data available and 4%
of the tier. Same structural case as CommitPackFT in the code lane: **upsample within the tier, do not
resize the tier around it.**

## 5. Source-by-source status

**Measured.** Sangraha (per-language, per-tier) · IndicCorpV2 (per-language, tail exact) · AWS High Court
judgments · AWS Supreme Court judgments (bucket layout confirmed; corpus measured in S4 at 3,848 docs,
7,744 words/doc, 90.6% cleaning retention).

**Stated, not independently verified.** BPCC ~230M bitext pairs (~7B Indic-side; only 2.2M human-translated) ·
Aksharantar transliteration pairs · IndicCorpV2 total 20.9B · eSCR ~31,000 translated judgments ·
Sansad Digital Library 1.45M items / 5M pages · National Archives 190K publications · KrishiKosh 45M
pages / 240K items / 108K theses · OpenSansad Lok Sabha QA 100K–1M records · BharatSchemes ~2K records ·
India Biodiversity Portal 58K species / 1.62M observations · OpenAlex (CC0 metadata; full text
item-specific).

*Neither BPCC nor Aksharantar is auto-converted on the HF datasets-server, so neither returns a size
through the route that worked for IndicCorpV2. Their figures remain stated.*

**Gated — rights tier recorded, no number claimed.** India Code · e-Gazette and state gazettes · Sansad
Digital Library bulk · Digital Sansad debates · PIB archive · state assembly sites and NeVA · ministry
annual reports · Census of India · RBI DBIE (research use with acknowledgement; training terms need
explicit agreement) · SEBI · ICMR/NCDC/CDSCO · KrishiKosh/ICAR · IMD · National Archives · Manuscripts
Mission · NCERT/SCERT/IGNOU/NPTEL · Indian Kanoon, DAKSH, CivicDataLab, OpenNyAI, SCC Online, Manupatra.

These are **programme dependencies, not supply.** A comprehensive proposal names them and states that no
budget line depends on them until the agreement exists. That is the difference between a plan and a wish.

**Open-licence, immediately usable, not yet sized.** data.gov.in GODL items (HMIS, AGMARKNET daily mandi
prices, CPCB real-time AQI, 1,400+ agriculture APIs) · CAG audit reports · Ashoka TCPD Question Hour ·
data.gov.in Lok Sabha verbatim debates · KanoonGPT (Apache 2.0) · Project Madurai · DIKSHA's CC BY / CC BY-SA
subset. These are mostly **tabular or tool-shaped rather than prose**, so their value is agentic and
tool-use training data rather than pretraining tokens, and they should be sized in *tasks*, not tokens.

## 6. What this does to the plan

- **Indic lane.** Usable supply 157.2B stands; the tail rises to 449M and stays immaterial. The lane's
  real weakness is not volume but **tier D's content collapse** (§2).
- **Reasoning lane.** Gains ~8B of licensed English legal reasoning — roughly doubling public non-V4
  supply, against a lane running at 3.27 epochs.
- **Tier C.** Loses the Samanantar double-count, gains BPCC's true scale, gains 0.3B of translated
  judgments. Net ~7B. **C20 remains impossible** (99B demand = 14 epochs); C2 stands.
- **Long-context.** Judgments are natively long and need no packing.

## 7. Still not obtained

- Per-language sizes for IndicCorpV2's **11 major languages** — blocked by the 5GB conversion cap. Would
  need the raw dataset.
- Dedup rate between Sangraha Verified and IndicCorpV2 for the eight overlapping tail languages.
- Corpus-scale dedup rate for the judgment corpus, where a p90/median ratio of 5.7 says short,
  boilerplate-heavy orders dominate.
- BPCC and Aksharantar per-language pair counts.
