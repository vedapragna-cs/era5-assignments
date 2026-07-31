# ERA V5 — requirements that cut across lanes

Three gaps between our S3 proposal and this plan turned out to be one gap seen from three sides:
**S5 has no way to express a requirement that cuts across lanes.** S3 states three such requirements and
S5 could not check any of them. Reproduce with `scripts/indiafirst.py`.

| S3 requirement | S5 could state it? | verdict |
|---|---|---|
| "India-first domains" at **9%** | no such lane | **unfundable as a lane, over-delivered as a tag** |
| "at least **15%** of STEM data should be Indic-language" | no cross-lane syntax | **genuinely unfunded — 4.5× short** |
| "at least **55%** of India-first data should be non-English" | no India-first lane | **passes, 88–94%** |
| Indic SFT (IndicAlign, 17.5GB) | outside the 97/3 budget | **volume fine, provenance 48× short** |

---

## 1. India-first is a tag, not a lane

**As a lane it is the second unfundable lane in the plan, after agentic** — and it comes from our own
prior work, which assigned 9% without a supply check.

Open, English-language India-first supply, measured or published:

| source | tokens |
|---|---|
| High Court judgments (E4, **measured**) | 8.00B |
| Supreme Court English, 43,517 docs (E5, **measured**) | 0.39B |
| OpenSansad Lok Sabha QA (CC BY) | 0.05B |
| India Biodiversity, Project Madurai, etc. | 0.02B |
| **total** | **8.46B** |

*KanoonGPT's 17M rows are a derivative of the same court sources and are deliberately **excluded** rather
than added. That is the Samanantar mistake, and the plan makes it once.*

**9% of 2.91T = 262B against 8.46B = 31 epochs.** Fundable share at the 4-epoch ceiling: **1.16%.**

Everything else in the S3 source map — India Code, e-Gazette, Sansad Digital Library, Digital Sansad,
PIB, state assemblies, Census, RBI, CAG, KrishiKosh — is **gated**. None of it carries a number, so none
of it can carry a budget share.

**But India-first is not a corpus.** It is a *property* of documents already counted in other lanes: a
court judgment is India-first **and** reasoning; a Tamil news article is India-first **and** Indic. E7
measured mean tags per token at m = 1.020 and kept the disjoint model for the *budget* — this is the case
where the tag view is nonetheless the right one, because the content is genuinely dual-purpose.

Coverage, where φ is the fraction of Indic-language text that is about Indian context:

| φ | Indic contribution | English slice | coverage | % of run | non-English | vs S3's 9% |
|---|---|---|---|---|---|---|
| 1.00 | 495B | 33.8B | 529B | **18.2%** | 94% | pass |
| 0.75 | 371B | 33.8B | 405B | 13.9% | 92% | pass |
| 0.50 | 247B | 33.8B | 281B | **9.7%** | 88% | pass |
| 0.25 | 124B | 33.8B | 158B | 5.4% | 79% | fail |

Even at φ = 0.5 — conservative, since text written in Indian languages is overwhelmingly about Indian
life — coverage clears S3's own target, and the **55% non-English rule passes at every φ by a wide
margin**.

**S3 and S5 do not actually disagree.** S3 modelled India-first as a lane because S3 assumed
multi-tagging; in a disjoint budget the same content cannot be booked twice. **The lane is unfundable and
the tag is over-delivered.** What is genuinely scarce is the **English-language** India-first component at
8.46B unique tokens — carried almost entirely by court judgments, and carved out explicitly inside the
reasoning lane rather than given a lane of its own.

**Decision.** No India-first lane. An India-first **coverage target of ≥9% of trained tokens**, delivered
by the Indic lane plus an explicit ~1.2% English-India slice, and **reported as a measured coverage
figure rather than assumed**. φ must be measured by classifying a sample of the Indic lane; until then
the claim is conditional on φ ≥ 0.5.

## 2. The requirement that is genuinely unfunded

> *"At least 15% of STEM data should be Indic-language."*

STEM demand 349.2B × 15% = **52.4B** of Indic-language STEM against **~11.7B** available (E7's 8%
Indic × STEM overlap). **4.5× short. Supply supports 3.4%, not 15%.**

Three options. The plan picks one rather than restating the rule:

- **(a) Lower it to 3.4%** and say why. Honest, and concedes the capability.
- **(b) Synthesize 41B of Indic STEM.** Translated science — which runs directly into the tier-D finding
  that translated content teaches **the source culture's distribution**, not India's. For STEM that
  matters less than for MILU's history and festivals, but Indian-curriculum framing is exactly what would
  be lost.
- **(c) Source it.** NCERT, SCERT, NPTEL and IGNOU are Indian-curriculum STEM in Indian languages, and
  they are **gated**.

**We choose (c), with (a) as the stated fallback.** This is the strongest argument in the plan for
prioritising one specific negotiation: a single NCERT/SCERT/NPTEL/IGNOU agreement is the difference
between a 3.4% and a 15% Indic-STEM share, and no amount of budget reallocation substitutes for it. The
plan therefore names it as a **programme dependency with a number attached to what it would unlock** —
which is the only honest way a gated source enters a budget.

## 3. Indic SFT — the volume is fine and the provenance is not

SFT is **<1% of the lifecycle** and sits outside this budget's 97/3 pretrain/anneal split, so it needs
naming and sizing, not a share. S3's plan: **2.5B SFT tokens, 20% native Indic assistance = 0.5B needed.**

`ai4bharat/indic-align` — **17.5GB ≈ 2.19B tokens**:

| sub-collection | GB | provenance |
|---|---|---|
| wiki_chat | 10.60 | derived from Wikipedia |
| indowordnet | 5.30 | derived from a lexical database |
| indicsharellama | 0.92 | derived |
| oasst | 0.47 | translated |
| dolly | 0.16 | translated |
| **anudesh** | **0.042** | **natively collected** |

**Covered 4.4× by volume.** But S3 also requires *"at least half of Indic instructions authored natively,
not translated"* — and natively-authored is **0.042GB of 17.5GB = 0.2%**. Against 0.25B needed we hold
~0.005B: **48× short.**

Same pattern as tier D, and worth stating as a general finding: **for Indic data the binding constraint
is repeatedly provenance, not volume.** Every corpus that looks large is large because something was
translated or derived; the natively-authored fraction is consistently two to three orders of magnitude
smaller. A plan that sizes Indic capability in tokens will keep concluding it is funded when it is not.

**Decision.** Indic SFT is named as a phase with a **0.5B target**, of which **0.25B must be natively
authored** — recorded as an acquisition commitment of the same kind as the agentic synthesis programme,
not as a dataset we already hold.

## 4. What changed in the lane table

**Nothing.** That is the result. The main-run shares stand at 34/24/17/12/9/2/2, because:

- India-first resolves to a tag with a coverage target, not a share;
- the Indic-STEM requirement resolves to a sourcing dependency, not a reallocation;
- SFT sits outside the pretraining budget entirely.

Three structural objections were raised against the budget and none of them moved a number in it. The
budget survived; the **claims around it** changed, which is where the work was.
