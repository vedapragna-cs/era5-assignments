# S5 ledger

## Done
- [x] Extract lesson widgets from `.webarchive`; establish session ground truth → `SESSION-DIGEST.md`
- [x] Drive all 9 widgets, sweep parameter spaces, record measured results → digest §13, log E0
- [x] Research pass: proxy-run methodology + published mixture anchors → `RESEARCH-NOTES.md`
- [x] Budget arithmetic at 2.4–4T against real supply → `BUDGET-PROPOSAL.md`, log E1
- [x] Pre-register T1–T3 with refutation criteria → `EXPERIMENT_LOG.md`

## Decisions — all taken
- [x] **All twelve decided** → `DECISIONS.md`, each with a stated reversal condition. Two closed by
      measurement rather than judgement: romanized Indic (E13) and the STEM supply figure (E12).

- [x] Curriculum staging under the budget-integral constraint; seam bands sized from measured spike data → `CURRICULUM.md`
- [x] Difficulty bands B0–B5 and reasoning-length bands with a concrete example per level → `CURRICULUM.md`

- [x] Per-lane benchmark mapping, incl. required training-data shape per benchmark → `BENCHMARK-MAP.md`

- [x] Agentic synthesis programme sized in samples, with method and QA gate → `AGENTIC-SYNTHESIS.md`

- [x] Per-language allocation inside the Indic lane, audited against the Sangraha per-language table
      → `INDIC-ALLOCATION.md`, log E2, pre-registered T4. Corrected Indic supply 276B → 157B
      (English split + transliteration double-count) and epochs 1.88 → 3.31 in the budget, curriculum
      and benchmark map.

- [x] Audit the S3 source map against live supply → `SUPPLY-AUDIT.md`, logs E3/E4.
      Corrected 5 inventory rows; measured the High Court corpus at ~8B (not 37-142B);
      recovered IndicCorpV2's per-language tail; identified tier D as translated English Wikipedia.

- [x] Cross-lane requirements resolved → `CROSS-LANE.md`, log E9. India-first is a tag not a lane
      (9% = 31 epochs; as coverage it is over-delivered). Indic-STEM 15% is unfunded — sourcing
      dependency on NCERT/SCERT/NPTEL/IGNOU. Indic SFT volume covered 4.4x, provenance 48x short.
- [x] Fertility-adjusted allocation and tail script decision → `INDIC-ALLOCATION.md` §11–12, log E8

## Next
- [ ] **Measure φ** — fraction of the Indic lane that is India-first context. The coverage claim is
      conditional on φ ≥ 0.5 until then. Same dependency as the Indic × STEM measurement.
- [x] Reasoning language decided: **Arm A (English-internal)**, on verifiability. T5 confirms or refutes.
- [ ] Run T1–T5; append results; revise budget from what they say
- [ ] **Re-cut T2 per MILU subject**, not per language: tier D is translated English Wikipedia, so
      the prediction is India-specific subjects degrade while general ones hold
- [ ] Measure Indic x STEM overlap directly (classify a Sangraha sample; needs native-labelled sets)
- [x] Fold into the submission README — `README.md` is now the standalone plan, not an index

## Blocked / external
- [ ] Plans are reviewed only once the team meets the **data-gating threshold** — cleaning continues
      toward the cumulative target, aimed at the lanes the supply check shows starved
