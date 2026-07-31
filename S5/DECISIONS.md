# ERA V5 — decisions taken

**All twelve are decided.** This is the record: what was chosen, the evidence behind it, and — for each —
**what would reverse it**. A decision without a reversal condition is a preference, so every entry carries
one.

Two were closed by measurement rather than judgement (#8 romanized Indic, #12 STEM supply) and are marked
**resolved** rather than decided.

| # | decision | reverses if |
|---|---|---|
| 1 | run scale **3T** | 4T is chosen — Indic then breaches the ceiling and must fall to ~15.7% |
| 2 | anneal **3% (90B)** | agentic trajectory synthesis proves cheaper than costed, making 5% affordable |
| 3 | agentic **2% floor, fund the programme** | T3 shows the anneal reserve earns nothing |
| 4 | reasoning **English-internal (Arm A)** | **T5** — Arm B wins on India-specific MILU subjects |
| 5 | tail languages **gated at 1.30B** | a gated language crosses the threshold — automatic |
| 6 | **prioritise** the NCERT/SCERT/NPTEL/IGNOU negotiation | it fails; requirement drops to 3.4% |
| 7 | execution environments **not scoped** | V4 infrastructure work delivers a live harness |
| 8 | romanized Indic **1% claiming Dakshina** | resolved by measurement (E13) |
| 9 | CommitPackFT **8× upsample** | more diff-following data arrives |
| 10 | IndicGenBench **14/29** | follows #5 automatically |
| 11 | Indic **17%** | T2 refutes tier D, or run scale moves to 4T |
| 12 | STEM supply **146B** | resolved by measurement (E12) — not a discrepancy |

---

## Tier 1 — changes every table

### 1. Run scale — recommend **3T**

- **Evidence.** Session states 2.4–4T. Demands scale linearly, so no *conclusion* changes across the
  range: agentic needs 11.6B / 14.6B / 19.4B unique tokens at 2.4T / 3T / 4T, which is 18–31× public
  supply at every point.
- **What it changes.** Every absolute number. Nothing relative.
- **Why 3T.** Midpoint; 100 tokens/parameter puts it near a 30B model, and S3 assumed 4T for 40B.
- **If you pick 4T:** Indic epochs go 3.15 → 4.33, **breaching the 4-epoch ceiling**. At 4T the Indic
  share must fall to ~15.7% or the lane is infeasible. This is the one place run scale is not neutral.

### 2. Anneal fraction — recommend **3% (90B)**

- **Evidence.** OLMo 2 sampled its 843B Dolmino pool into 50B / 100B / 300B anneal mixes for 7B–32B
  models, so 90B is mid-range and precedented. **Our own S3 says 7.5%** — 2.5× higher, unjustified there.
- **What it changes.** The anneal is the only feasible home for agentic (8% of 90B needs 1.8B unique vs
  14.6B in the main run). A larger anneal makes every scarce lane cheaper and shortens the main run.
- **Tension to be aware of:** at 5% (150B) the agentic anneal demand rises to 12B seen / 3B unique,
  which **doubles the trajectory synthesis target** from 153,600 to ~307,000. Bigger anneal is not free.

---

## Tier 2 — commits us to a production programme

### 3. Main-run agentic at the 2% floor — recommend **fund it**

- **Evidence.** 2% of 2.91T = 58B against 0.63B public supply. At the 4-epoch ceiling that is 14.6B
  unique = **22× all public agentic data**, i.e. **~24.5M synthesized function calls**. Measured: an
  unprotected lane goes to **zero** (0 tokens in 59/60 OPUS iterations), and **no proxy choice rescues
  it** — agentic ranks last under a balanced proxy and second-last under an English one.
- **Options.** (a) fund the synthesis programme; (b) re-shape the lane to trajectory data repeated to the
  ceiling, accepting a much narrower distribution; (c) argue the floor down — contradicted by the
  measurement above.
- **Why (a).** The cheap shape (short calls, schema-generable, execution-verifiable) is what fills the
  main-run lane; the expensive shape sits in the anneal at only 153,600 trajectories.

### 4. Reasoning language — **DECIDED: Arm A (English-internal), T5 to confirm**

- **Evidence.** Nothing in the inventory has the target shape (Indic question → reasoning → Indic
  answer). Both arms are affordable from AON alone (≤28% of our 78B). **Arm A** (English-internal CoT)
  needs 16.4B translated tokens; **Arm B** (native CoT) needs 27.8B — 1.69×.
- **The real asymmetry is not cost.** Arm A translates questions and short answers, often exact-match
  checkable. Arm B translates free-form reasoning chains, where a fluent mistranslation is
  indistinguishable from a correct one. **B is qualitatively harder to trust, not 1.69× harder.**
- **Against A:** MILU is India-centric by construction; reasoning about Indian law and festivals in
  English risks losing the grounding the model exists to have.
- **Decision: Arm A is the default**, on verifiability. A plan that does not say how the model reasons is
  not a plan, and T5 exists to confirm or refute a stated default rather than to choose from scratch.
- **Sizing:** the cross-lingual slice is **10% of the reasoning lane (27.8B)**, drawn from within it — no
  other lane moves. Requires 16.4B translated tokens from 28% of AON.
- **Honest limitation:** this slice is a *re-presentation of AON content*, not new content. The English CoT
  is unchanged and only question and answer are translated, so **it does not relieve the reasoning lane's
  3.08 epochs** — it changes the form of some of those repetitions.
- **T5 still runs**, with arm C (no cross-lingual data), because both arms could be worse than nothing.

### 5. Tail-language acquisition — **DECIDED: gate them** (E10)

- **Evidence.** After adding IndicCorpV2, the eight sub-1B languages hold **449M tokens** combined; at
  their full 4-epoch ceiling they absorb **0.345% of the lane**. No allocation rule reaches them.
  BPCC covers all of them but at 0.3–1.7MB each. **Rasa and IndicVoices are organised by language and
  Rasa's largest is Bodo** — ASR is the only supply route not limited by what was already written down.
- **Decision.** Gated, not funded. Entry threshold **1.30B unique tokens** — the level at which a
  language can reach 1% of the lane without breaching 4 epochs. The threshold sits at a **24× cliff**
  between Assamese (6.16B) and Sindhi (0.26B), so it is read off the distribution rather than chosen.
- **Cost:** IndicGenBench claim falls **19/29 → 14/29**. MILU unaffected at 10/10.
- **Why:** a 0.006% share was never a commitment to Manipuri. They move to the acquisition programme
  with a target and an automatic re-entry trigger. See `INDIC-ALLOCATION.md` §13.

### 6. NCERT / SCERT / NPTEL / IGNOU negotiation — recommend **prioritise it**

- **Evidence.** S3 requires ≥15% of STEM to be Indic-language = **52.4B**; supply supports **3.4%**
  (~11.7B). **4.5× short.** These four are Indian-curriculum STEM in Indian languages, and gated.
- **What it changes.** A single agreement is the difference between a 3.4% and a 15% Indic-STEM share.
  No budget reallocation substitutes for it, and synthesis would mean translated science — which runs
  into the tier-D finding that translated content teaches the source culture's distribution.
- **Fallback if not pursued:** lower the requirement to 3.4% and state the capability is conceded.

---

## Tier 3 — what we claim

### 7. Execution environments — recommend **do not scope in**

- **Evidence.** Terminal-Bench, WebArena and OSWorld need a live shell, hosted sites and desktops. V4
  never built them; no dataset substitutes.
- **What it changes.** Scoping them in adds infrastructure, not tokens, and the budget cannot buy them.
  Currently **not claimed**, which is the honest position.

### 8. Romanized Indic — **DECIDED: 1% sub-lane claiming Dakshina** (E13)

- **Evidence.** Sangraha's synthetic tier is 14 languages × 2 scripts with **identical row counts** —
  the Latin half is transliteration of the same documents. Excluded from the lane because MILU and
  IndicGenBench score native script. **Aksharantar (729MB, 21 languages) is the data anchor** if scoped.
- **Decision.** 1% of the Indic lane (~5B) from Aksharantar plus a slice of the Latin half, claiming
  **Dakshina** (LREC 2020, CC BY-SA 4.0, 12 languages, 10 in our funded 14). No new budget; native-script
  Indic eases 3.15 → 3.12 epochs. **Not** GLUECoS — word-level classification, Hindi its only Indic
  language. Indi-RomCoM noted as emerging but a 2026 preprint.
- **Why 1% and not 81B.** The capability is asymmetric: reading romanized input matters, generating it
  does not for a coding model. Input robustness needs far less data than fluency.

### 9. CommitPackFT upsample factor — recommend **8×, to ~5% of the code lane**

- **Evidence.** CommitPackFT (4B) is the **only** dataset in the inventory that teaches diff-following.
  Aider Polyglot scores a search/replace diff that must apply cleanly; whole-file generation does not
  produce that behaviour. At natural weight it is **0.6% of the 716B code lane**.
- **What it changes.** 8× puts it at 32B = 4.5% of the lane and **2 epochs** — under the ceiling. Without
  an explicit factor we fund a lane that wins LiveCodeBench and loses Aider.

---

## Tier 4 — ratifications (evidence decides these; they need agreement)

### 10. IndicGenBench claimed on **14 of 29 languages**

Ten have **zero tokens** anywhere in the inventory: Bhojpuri, Pashto, Awadhi, Haryanvi, Tibetan,
Garhwali, Chhattisgarhi, Rajasthani, Malvi, Marwari. Five more — Bodo, Konkani, Maithili, Manipuri,
Santali — are **supply-gated** under #5. MILU stays **10/10**.

### 11. Indic at **17%**

2× Llama 3's multilingual share and ≈ Nemotron-4's — but both cover all languages with vast supply.
Defensible on differentiator grounds, not on precedent, and that is how it is stated. At 3T it runs at
**3.15 epochs**, the second-tightest lane. **At 4T it breaches the ceiling** — see #1.

### 12. STEM supply — **use 146B, flag the discrepancy**

The course composer states 250B; auditable inventory rows total **146B** (peS2o 42 + proof-pile-2 55 +
D4 STEM V4 49). The 104B gap is unexplained. We use 146B throughout, which puts the lane at 2.39 epochs;
at 250B it is 1.40 and comfortable. **Worth asking the course which is intended** — it is the one
remaining number where we differ from the material without knowing why.

---

## Not decisions — measurements still owed

These are logged as work, not choices: **φ** (fraction of the Indic lane that is India-first context,
the coverage claim is conditional on φ ≥ 0.5), **Indic × STEM overlap** (E7 uses an 8% estimate),
**corpus-scale dedup** for the judgment corpora, **per-language fertility** against our own tokenizer,
and **T1–T5** themselves.
