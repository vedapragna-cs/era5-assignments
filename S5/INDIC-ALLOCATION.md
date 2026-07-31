# ERA V5 — per-language allocation inside the Indic lane

`BUDGET-PROPOSAL.md` sets the Indic lane at 17% and splits it by provenance tier.
It does not say **which languages**. MILU is scored *per language* and IndicGenBench spans
29 of them, so a lane share that is silent on language distribution does not predict either score.
This closes that gap. Reproduce with `scripts/indic.py`.

Supply is the AI4Bharat **Sangraha per-language token table**, whose totals
(64,306.1M verified / 24,307.7M unverified / 162,707.9M synthetic) reproduce the S5 inventory rows
exactly — so every per-language row below is auditable against a number the session already uses.

---

## 1. Two corrections the per-language table forces

Reading the table by language, rather than by its total, breaks two numbers currently in the plan.

**(a) 19.8% of "Tier A verified Indic" is English.** Sangraha's verified config has 23 splits, and
`eng` is one of them at **12,759.9M tokens** — the second-largest split in the entire tier.
Verified *Indic* is **51.5B, not 64.3B**. Counting English toward an Indic lane is precisely the
wishful accounting this plan exists to avoid. Routed to the web lane, which has 3.7T of slack and
does not notice.

**(b) The synthetic tier is one corpus transliterated, not two corpora.** The synthetic config has
28 splits = 14 languages × 2 scripts, and each native/Latin pair has an **identical row count**
(`hin_Deva` and `hin_Latn` are both 5,775,143 rows; `tam_Taml` and `tam_Latn` both 5,533,119).
It is the same documents, romanized. MILU and IndicGenBench are scored in **native script**, so only
the native half buys those benchmarks: **~81.4B, not 162.7B.**

The romanized half is not worthless — code-mixed Latin-script Indic is how a large share of Indian
users actually type, and it is a real capability. But it is a *different* capability from the one
the Indic lane was justified by, and it must not be counted twice.

| | claimed in v0 | corrected |
|---|---|---|
| Indic supply | 275.9B | **157.2B** (native-script, English removed) |
| Lane epochs at 520B | 1.88 | **3.31** |

Still under the 4-epoch ceiling, but the lane goes from *comfortable* to *the second-tightest in the
plan*, behind reasoning at 3.27. Adding IndicCorpV2 (20.9B) and BPCC + Samanantar (5.0B) brings
usable supply to 183.1B and epochs to **2.84**; those two have no published per-language table, so
they are counted in the lane total but not allocated by language here.

## 2. Supply is concentrated, and the tail is not thin — it is absent

| | |
|---|---|
| Hindi + Bengali | **33.6%** of all usable Indic supply |
| 14 languages with synthetic coverage | 99.8% of supply |
| 8 languages below 1B tokens | **292.7M combined = 0.19% of supply** |

Those 8 — Sindhi, Maithili, Konkani, Manipuri, Bodo, Kashmiri, Santali, Dogri — are all
8th-Schedule or IndicGenBench languages. Dogri has **0.06M tokens**. Santali has **0.3M**.

## 3. What the benchmarks actually reach

| benchmark | languages | funded | verdict |
|---|---|---|---|
| **MILU** | 11 (10 Indic + English) | **10/10** | fully fundable; weakest is Punjabi at 6.20B supply |
| **IndicGenBench** | 29 | **19/29** | **10 languages have zero tokens** |

IndicGenBench languages with **no supply anywhere in the inventory**:
Bhojpuri, Pashto (medium-resource) · Awadhi, Haryanvi, Tibetan, Garhwali, Chhattisgarhi,
Rajasthani, Malvi, Marwari (low-resource).

**This changes what `BENCHMARK-MAP.md` may claim.** MILU is a benchmark this budget can win.
IndicGenBench is scored as an average over 29 languages, and we can fund 19 — so a plan that lists
IndicGenBench without qualification is claiming 10 languages it has no data for. We claim
**IndicGenBench on its 19-language funded subset**, and report the excluded 10 explicitly.

Note the mismatch runs both ways: Sindhi and Kashmiri have Sangraha supply but are not in
IndicGenBench's 29 at all.

## 4. Allocation mechanism

Two constraints, one lever.

**Lever — temperature sampling.** Allocate `p_i ∝ S_i^α`, the standard multilingual-pretraining
control (XLM-R, arXiv:1911.02116, which found **α = 0.3** best overall; higher α favours
high-resource languages, lower α favours low-resource ones).

**Constraint — the epoch ceiling binds per language, not just per lane.** No language may exceed
`4 × its own supply` (arXiv:2305.16264). So the allocation is temperature sampling **water-filled**:
spread the budget by `S_i^α`, clip anything over its cap, redistribute the excess to the uncapped,
repeat.

The cap is what makes this interesting. **α = 1 (proportional) is exactly uniform-epoch allocation**
— every language lands at 3.31 epochs, the lane figure. Since the lane already runs at 3.31, there
is only 0.69 epochs of headroom, so any α < 1 immediately pins languages to the ceiling.

## 5. Where the lever stops working

| α | Hindi share | weakest MILU lang | languages pinned at 4 epochs |
|---|---|---|---|
| 1.00 | 18.9% | 20.5B | 0 |
| 0.85 | 16.6% | 22.8B | 8 |
| **0.72** | **14.8%** | **24.8B** | **10** |
| 0.70 | 14.5% | 24.8B | 10 |
| 0.50 | 12.4% | 24.8B | 14 |
| 0.30 | 10.8% | 24.8B | 17 |

**α* = 0.72.** Punjabi, the weakest MILU language, has 6.20B of supply, so it can absorb at most
24.8B. It reaches that ceiling at α = 0.72. **Every α below 0.72 buys MILU nothing** — the weakest
language is already maxed — and costs progressively more languages their headroom, pinning them at
exactly the repetition limit where the 4-epoch paper says value starts decaying.

XLM-R's α = 0.3 is therefore the wrong import here. It is tuned for a corpus with orders of
magnitude of dynamic range and slack; ours has 0.69 epochs of slack. **We differ from the published
recipe because the binding constraint is different**, not because we disagree with it.

## 6. Proposed allocation — α = 0.72

| language | supply | allocation | share of lane | epochs | MILU |
|---|---|---|---|---|---|
| Hindi | 29.8B | 76.8B | 14.78% | 2.58 | ✓ |
| Bengali | 23.1B | 64.1B | 12.32% | 2.77 | ✓ |
| Tamil | 11.4B | 38.6B | 7.42% | 3.38 | ✓ |
| Gujarati | 10.7B | 36.8B | 7.08% | 3.44 | ✓ |
| Telugu | 10.3B | 35.8B | 6.89% | 3.47 | ✓ |
| Malayalam | 9.84B | 34.6B | 6.66% | 3.52 | ✓ |
| Urdu | 9.69B | 34.3B | 6.59% | 3.53 | |
| Marathi | 8.89B | 32.2B | 6.19% | 3.62 | ✓ |
| Kannada | 8.21B | 30.4B | 5.85% | 3.70 | ✓ |
| Sanskrit | 8.12B | 30.2B | 5.80% | 3.72 | |
| Nepali | 7.60B | 28.8B | 5.53% | 3.78 | |
| Odia | 6.87B | 26.7B | 5.14% | 3.89 | ✓ |
| Punjabi | 6.20B | 24.8B | 4.77% | **4.00** | ✓ |
| Assamese | 6.16B | 24.6B | 4.74% | **4.00** | |
| Sindhi | 0.26B | 1.03B | 0.20% | **4.00** | |
| Maithili | 14.6M | 58M | 0.011% | **4.00** | |
| Konkani | 10.1M | 40M | 0.008% | **4.00** | |
| Manipuri | 7.4M | 30M | 0.006% | **4.00** | |
| Bodo | 1.5M | 6.0M | 0.001% | **4.00** | |
| Kashmiri | 0.5M | 2.0M | 0.000% | **4.00** | |
| Santali | 0.3M | 1.2M | 0.000% | **4.00** | |
| Dogri | 0.06M | 0.2M | 0.000% | **4.00** | |
| **total** | **157.2B** | **519.9B** | 100% | 3.31 | |

Versus proportional, this moves ~22B from Hindi (18.93% → 14.78%) to the mid-tier: Punjabi
3.94% → 4.77%, Assamese 3.92% → 4.74%, Odia 4.37% → 5.14%.

## 7. The tail is not an allocation problem

Every one of the 8 sub-1B languages is at its 4-epoch cap in the table above, which is the most any
allocation rule is permitted to give it. Together that is **1,171M tokens = 0.225% of the Indic
lane.** There is no share of the budget that fixes this, because the constraint is supply, not
allocation:

| language | supply | max at 4 epochs | as % of lane |
|---|---|---|---|
| Sindhi | 258.2M | 1,032.8M | 0.199% |
| Maithili | 14.6M | 58.4M | 0.011% |
| Konkani | 10.1M | 40.4M | 0.008% |
| Manipuri | 7.4M | 29.6M | 0.006% |
| Bodo | 1.5M | 6.0M | 0.001% |
| Kashmiri | 0.5M | 2.0M | 0.0004% |
| Santali | 0.3M | 1.2M | 0.0002% |
| Dogri | 0.06M | 0.2M | 0.00004% |

**Stating this is the point.** A plan that assigns the Indic lane a share and stops has implicitly
claimed 22 languages. This one claims 14 at scale, 8 at token level, and 10 IndicGenBench languages
not at all. The honest response to the tail is a **data acquisition programme**, not a budget line —
and it belongs in the same category as the agentic synthesis programme, sized in documents to
collect rather than tokens to spend.

## 8. Tier mix is not a free per-language choice

`BUDGET-PROPOSAL.md` §2 proposes a global tier split (A 28 / B 30 / C 2 / D 40). Per language, the
available mix is fixed by what exists:

- **Assamese is 5% verified** — 95% of its usable supply is synthetic.
- **Bengali is 46% verified.** Hindi 42%. Sanskrit 16%, Odia 17%, Punjabi 17%.

If each language draws its tiers in proportion to what it has, the realized global Tier-A share is
**31.4%**, close to the proposed 28% — but that aggregate hides a 5%–46% spread. **The global tier
split is an average, not a specification**, and for the synthetic-heavy languages the plan is
committing to a much lower-quality mix than the headline implies. T2 (synthetic-Indic quality)
should therefore be evaluated **per language**, not on the MILU average, because Assamese and
Bengali are testing very different hypotheses.

## 9. What would falsify this — T4

| hypothesis | metric | refuted if |
|---|---|---|
| α = 0.72 beats proportional (α = 1) on the MILU per-language average | MILU accuracy averaged over the 10 Indic languages, and the min across them | proportional matches or beats it → the mid-tier gain is not worth Hindi's loss, revert to α = 1 |

Run under the same anneal-as-evaluation protocol as T1–T3: two candidate mixes differing only in
Indic per-language weights, 30% candidate / 70% default. Report both the mean **and the minimum**
across languages — a rule that lifts the average by improving Hindi has not done the thing it was
designed to do.

## 11. Fertility-adjusted allocation (E8) — supersedes §6

§6 allocates in **tokens**. S3's MUTANT-Indic ceilings say that is the wrong unit: the same content
costs **1.375 tokens/word in Hindi and 3.025 in Malayalam**. Under the §6 allocation Malayalam receives
45% of Hindi's tokens and **20% of its content**.

Allocating over content instead — `content_i ∝ C_i^α` where `C_i = S_i / f_i`, so `tokens_i ∝ C_i^α·f_i`,
water-filled against the same token cap of 4×supply:

| language | fertility | §6 tokens | adjusted | Δ | epochs |
|---|---|---|---|---|---|
| Hindi | 1.375 | 76.8B | **62.6B** | **−14.2** | 2.10 |
| Bengali | 1.725 | 64.1B | 59.2B | −4.9 | 2.56 |
| Malayalam | 3.025 | 34.6B | **39.4B** | **+4.7** | 4.00 |
| Telugu | 2.050 | 35.8B | 39.5B | +3.6 | 3.83 |
| Tamil | 2.050 | 38.6B | 41.9B | +3.3 | 3.67 |
| Kannada | 2.050 | 30.4B | 32.8B | +2.4 | 4.00 |
| Sanskrit | 3.025 | 30.2B | 32.5B | +2.3 | 4.00 |
| Urdu | 1.375 | 34.3B | 32.3B | −2.0 | 3.33 |

**α\* falls from 0.72 to 0.59** in content space. Hindi funds the shift almost entirely.

**It improves the unit and cannot fix the problem.** Malayalam's content rises from **20% to 29%** of
Hindi's and stops there, because it hits the 4-epoch ceiling first. This is a general property worth
stating: **high-fertility languages need more tokens per unit of content, so they exhaust their
repetition budget sooner.** The languages most disadvantaged by a token-based rule are the ones least
able to absorb the correction. **10 of 22 languages are pinned at 4 epochs** after adjustment, against 10
before — the headroom does not survive.

Content parity is not reachable at this lane size. Saying so is the point; a plan that allocated by
content and implied parity would be claiming something the epoch ceiling forbids.

## 12. Script decision for the fragmented tail (E6 → E8)

Rule, in order: **(a)** use the script the claimed benchmark scores in; **(b)** failing that, the script
with more supply; **(c)** report the supply stranded in the unchosen script rather than pooling it.

| language | scripts present | IndicGenBench | chosen | basis |
|---|---|---|---|---|
| Manipuri | Bengali / Meetei Mayek | **Meetei Mayek** | Mtei | benchmark |
| Santali | Ol Chiki | **Ol Chiki** | Olck | benchmark |
| Kashmiri | Perso-Arabic / Devanagari | not covered | Arab | supply |
| Sindhi | Perso-Arabic / Devanagari | not covered | Arab | supply |

**Manipuri is the live case.** IndicGenBench scores Meetei Mayek, BPCC carries *both* `mni_Beng` and
`mni_Mtei`, and IndicCorpV2 carries only `mni_Mtei`. Choosing Mtei is right for the benchmark and strands
the Bengali-script material. **Kashmiri and Sindhi are not in IndicGenBench's 29**, so nothing forces a
script; we take Perso-Arabic on supply and record that the choice is **unvalidated by any benchmark we
claim**.

## 13. Supply gate — 14 funded, 8 gated (supersedes §7)

§7 gave all 22 languages their full 4-epoch ceiling, which meant funding Santali at 31M tokens and Dogri
at 1M. That is not a commitment to those languages; it is the shape of a plan that lets today's supply
set tomorrow's design.

**The threshold is in the data, not chosen.** Ranking supply reveals the distribution is not a smooth
tail — it is 14 languages and then a cliff:

| rank | language | supply | ratio to next |
|---|---|---|---|
| 13 | Punjabi | 6.20B | 1.0× |
| **14** | **Assamese** | **6.16B** | **23.8×** |
| 15 | Sindhi | 0.26B | 17.7× |
| 16 | Maithili | 0.01B | |

**Entry gate: a language is funded when it can reach 1% of the lane (5.2B) without breaching 4 epochs —
i.e. holds ≥1.30B unique native-script tokens.** All 14 above the cliff clear it (min 6.16B); none of the
8 below come within 5×.

| gated | supply | gap to gate | IndicGenBench | supply route |
|---|---|---|---|---|
| Sindhi | 258.2M | 5× | not scored | text / partnership |
| Maithili | 14.6M | 89× | scored | Rasa + IndicVoices |
| Konkani | 10.1M | 129× | scored | Rasa |
| Manipuri | 7.4M | 176× | scored | IndicVoices |
| Bodo | 1.5M | 866× | scored | **Rasa's largest language** (125 files) |
| Kashmiri | 0.5M | 2,600× | not scored | text / partnership |
| Santali | 0.3M | 4,332× | scored | text / partnership |
| Dogri | 0.06M | 21,662× | not scored | text / partnership |

**The price, stated plainly: the IndicGenBench claim falls from 19/29 to 14/29.** We stop claiming Bodo,
Konkani, Maithili, Manipuri and Santali. MILU is unaffected at 10/10. This is worth paying — 19/29 where
five languages are funded at 30M tokens is a *weaker* claim than 14/29 funded properly, because a
reviewer who checks Santali finds 7.7M tokens behind a claimed benchmark language.

Freed budget is **1.17B = 0.225% of the lane**. That is not why we do it.

**The trigger is what makes this a plan rather than a snapshot.** A gated language re-enters
automatically on crossing 1.30B unique tokens. No redesign is needed: the allocation rule
(content-weighted temperature at α\*, water-filled against 4 epochs) is defined for any number of
languages, so adding one is a re-run.

**The two halves need different kinds of "later".** The 14 funded languages grow *passively* as the web
grows. The 8 gated ones do not — almost nothing is being written down in Santali or Dogri at volume, so
their supply grows **only if we fund collection**. Speech is the cheapest route and it points exactly
here: Rasa's single largest language is Bodo, and IndicVoices carries Manipuri.

So the gated set is not dropped. It moves from the budget, where a 0.006% share was doing nothing, to the
acquisition programme, where it has a target and a trigger. **A share of 0.006% was never a commitment to
Manipuri. A threshold is.**

## 10. Open

- **Romanized Indic.** ~81B of transliterated text is excluded here. It should either be scoped as
  its own sub-lane with its own benchmark, or dropped. Currently it is neither.
- IndicCorpV2 and BPCC/Samanantar are counted in lane supply but not allocated per language for lack
  of a published per-language table.
- Script-level allocation within a language (Sanskrit in Devanagari vs romanized; Punjabi in
  Gurmukhi vs Shahmukhi) is not addressed.
