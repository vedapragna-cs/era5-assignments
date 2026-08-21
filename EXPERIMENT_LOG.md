# S8 experiment log

Chronological. Failures and dead ends are recorded, not tidied away.

## E0 — extraction (2026-08-21)
Pulled the S8 material read-only from the portal API with the session cookie: sessions list,
`/api/lessons/cms8lmsrhwu9k6b284dx` (markdownContent, 63,961 bytes, 1,532 lines), the assignment
record, all 17 widgets, and the transcript. Artifacts in `$HOME/shared/S8-extracted/`.

**Unexpected:** the widget iframe list is in the lesson markdown itself, so Playwright was not
needed to enumerate widgets — the S7 handoff had assumed it would be. Saved a dependency.

**Spec finding:** `parts` is a single `{type:"link", points:1000}` field and `rubricJson` is `[]`.
Question 2 is worth another 1000 points and **has no field to answer it in**, and the required
GitHub repo has no field either. Recorded as G1–G3 in our internal rubric notes. Response: `allowMultiple:true`
plus `requireCaption:true` means several captioned links can be submitted, and the Q2 answer lives
at a route inside the app rather than in a text box that does not exist.

## D1 — arXiv v1 dates for 37 candidate papers
**Question.** Can every mechanism's date be resolved from a primary source rather than recall?

**Method.** One batched `export.arxiv.org/api/query?id_list=...` call; `published` is the v1
submission timestamp. Response committed at `data/arxiv-response.xml`.

**Result.** 37/37 returned. 27 of them are used by the final 28 cards (standard attention and
sinusoidal encoding share arXiv 1706.03762).

**Failure worth recording:** the first attempt used `http://export.arxiv.org` and silently returned
**0 bytes** with exit 0 from curl — the XML parse was the only thing that surfaced it. `https`
worked. A date pipeline that had swallowed that error would have produced an empty timeline with no
warning.

## D2 — DroPE identification
**Question.** What is DroPE and when did it launch?

**Result — the assignment's own warning, reproduced.** Two real papers share the acronym:

| | arXiv | v1 | field |
|---|---|---|---|
| DRoPE | 2503.15029 | 2025-03-19 | autonomous-driving trajectory modelling |
| DroPE | 2512.12167 | 2025-12-13 | dropping positional embeddings to extend LLM context |

The exact-string match is the wrong paper and is nine months earlier. Read both abstracts before
choosing. Logged as correction C1.

**Second finding (C2).** Lesson section 9 and widget `s8_widget_8_drope` both state that the record
"does not establish the exact DroPE algorithm." arXiv 2512.12167, published eight months before the
lesson, states it in its abstract. Identification of the V4 cookbook's DroPE with this paper is
**inference** — the lesson names no paper — and is labelled as such on the corrections page.

## D3 — NTK-aware scaling: a date we could not pin
**Question.** Exact launch date of NTK-aware scaled RoPE.

**Attempts and outcomes.**
1. `reddit.com/.../14lz7j5/.json` → **HTTP 403**. Failed.
2. `old.reddit.com` same path → **HTTP 403**. Failed.
3. YaRN bibliography via ar5iv → succeeded, but gives year only (`bloc97 [2023a]`). Confirms the
   post is the origin and that YaRN's first author Bowen Peng *is* u/bloc97, making the citation
   effectively primary.
4. GitHub API → `jquesnelle/scaled-rope` created **2023-06-26T13:27:56Z**; bloc97 PR #1 opened
   **2023-07-07T20:40:33Z**.

**Resolution.** Dated 2023-06-26, status `bounded`, rendered as *date bounded, not exact*. The
commonly repeated 2023-06-30 is plausible and is **not** asserted. Because the band is 11 days wide,
the card's ordering against Position Interpolation (2023-06-27) is inside the uncertainty and the
findings page claims simultaneity, not priority.

## D4 — conflict hunt across widget JS / lesson prose / transcript
**Method.** Regex for any four-digit year across the lesson markdown; numeric claims cross-checked
against widget JS and recomputed by hand.

**Results.**
- Lesson markdown contains **zero** dates (C4). The session teaches 18 mechanisms and dates none of
  them, so there was no course number to copy or contradict.
- Transcript line 38 dates the Transformer to 2018 (C5). Actual v1 2017-06-12. Logged as an ASR
  artifact — the speaker self-corrects to "17" in the same sentence — not as an instructor error.
- **Negative control (C6):** recomputed the lesson's KV-cache arithmetic,
  `2 × 48 × 8 × 128 × 32768 × 2 = 6,442,450,944 B = 6.442 GB`, and `× 8 users = 51.54 GB`. The
  lesson states 6.44 GB and 51.54 GB. **Exact.** DroPE widget's `262144 / 8192 = 32×` also exact.
  Reported because a hunt that only ever finds errors is a bias, not a hunt.

## D5 — timeline analysis (the Question 2 answer)
**Question.** What does date order show that a list cannot? Computed in `analyse.py`, nothing
hand-entered.

**Headline result — measured.** Lag from arXiv v1 to first well-known deployment, in publication
order: MQA 881 d → sliding window 1278 d → RoPE 678 d → ALiBi 255 d → FlashAttention 27 d →
GQA 57 d → NTK-aware 59 d → YaRN 0 d → MLA 0 d. This result exists only because each row carries
**two** dates, which was the design decision taken before building.
*(Original entry read "Monotone apart from ALiBi." — superseded by V2 below.)*

**Second result.** Four unrelated idea→use gaps land within 90 days of each other: MQA→GQA 1293 d,
Longformer→Mistral 1278 d, sliding-window→attention-sinks 1267 d, delta-rule→parallel-DeltaNet
1204 d. Observation on four points; no mechanism claimed for the specific number.

**Prediction check.** The lesson predicts the field wants "exactness, then memory, then length, then
memory again," and that sparsity returns. The family-by-year spread confirms the sparsity return
(2019, 2019, 2020, *three-year hole*, 2023, 2024, 2025) and adds what the prediction omits: it
returned inverted — hand-designed and bolted-on in 2019-20, natively trained and hardware-aligned in
NSA 2025.

## V1 — falsifiability controls
Both checks were deliberately broken to prove they can fail.

| control | injected fault | expected | observed |
|---|---|---|---|
| `verify_dates.py` | RoPE date changed 2021-04-20 → 2021-04-21 | non-zero exit, `MISMATCH` | exit 1, `rope 2021-04-21 2021-04-20 MISMATCH` |
| `build.py` | card with five advantages and empty `gives_up` | build fails, no HTML written | exit 1, `'gives_up' is empty` |

Both restored afterwards; re-ran clean (0 failures, 28 cards, 17/17 coverage).

## Known limitations
- "Shipped" dates are a judgement about *first widely-known deployment*. Each names its specific
  artifact so the call can be disputed row by row, but it is not a mechanical criterion the way
  arXiv v1 is.
- The DroPE ↔ V4-cookbook identification (C2) is inference, not established.
- The 1250-day clustering in D5 is four data points.
- Q2-3's claim that FlashAttention caused the approximation wave to stop is argued from adjacency,
  not proved. The competing reading — the approximations simply lost on merit — is stated alongside.


## V2 — correction to D5, caught by plotting it
**Superseded claim.** D5 as first written, and the first published version of finding Q2-1, said
the lag sequence was "monotone apart from ALiBi."

**That is wrong**, and rendering the chart is what caught it. The sequence is
881 → 1278 → 678 → 255 → 27 → 57 → 59 → 0 → 0, which has **three** upward steps:

| step | from → to | direction |
|---|---|---|
| MQA → sliding window | 881 → 1278 | **up** |
| FlashAttention → GQA | 27 → 57 | **up** |
| GQA → NTK-aware | 57 → 59 | **up** |

ALiBi (255 d) is not an exception at all — it is lower than RoPE (678 d) before it, which is
exactly the direction the trend predicts. The claim named the one point that *fits*.

**Corrected reading, now published:** not monotone. Sliding window is the high point, and the 2023
cluster (FlashAttention 27, GQA 57, NTK-aware 59) is flat rather than still falling. What is
unambiguous is the change of unit — years before 2022, weeks from mid-2022, zero by 2024.

**Why this is worth recording rather than quietly fixing.** The assignment's warning is about
confident sentences that are wrong, and this was one of ours: it survived being written into the
data file, the README and a commit message, and was only caught when the numbers were put on an
axis where a rising segment is impossible to miss. Step 7 of the visualization procedure — render it
and look at it — is the control that fired. Superseded text is left in place above with a pointer
here, per the no-rewriting-measurements rule.


## D6 — auditing the lesson against the assignment's own list
**Question.** Is the 17-item "what to cover, at minimum" list the same set as what the session
actually taught? We had been treating it as the specification.

**Why we asked.** The transcript, not the lesson, states the grading rule:

> "It'll make something but you don't know it missed something and if it did miss something I'm
> going to score you zero. And how will you know it missed something? By actually studying it."

A miss is a **zero condition**, not a deduction, and the binding set is "all the attention mechanism
that we have covered" — the session, not the enumeration.

**Method.** Walked all 18 lesson sections against the list, then counted term frequency in the
lesson markdown as a cross-check on importance.

**Result — the list is short by two.**

| lesson section | mechanism | on the list |
|---|---|---|
| 13, *Schedules across depth* | hybrid depth schedules, `DDDGDDDG` | no |
| 14, *State that outlives the window* | cross-chunk O(1) state, the V4 Memory Stream | no |

Frequency: `schedule` 10 occurrences, `Memory Stream` 4 — against **1 each** for ALiBi, MLA and
attention sinks, all three of which are on the list. Term frequency is a crude proxy, but a
10-to-1 ratio is not noise.

**Action.** Four cards added, each dated from a primary source and verified through the same script
as every other date: hybrid depth schedules → Griffin (2402.19427, 2024-02-29), with Jamba
(2024-03-28) and Samba (2024-06-11) inside fifteen weeks; and the cross-chunk lineage the lesson
leaves implicit — Transformer-XL (1901.02860, 2019-01-09), Compressive Transformer (1911.05507,
2019-11-13) and RMT (2207.06881, 2022-07-14), the closest published relative of the Memory Stream.
Card count 28 → 32; verified dates 27 → 31. Logged as correction C8 and finding Q2-8.

**What this says about our own method.** We wrote an internal rubric from the assignment text and treated
its list as the spec for the whole build. That was the wrong primary source: the spec is the
session. The failure mode the instructor described — an agent that produces something plausible and
silently misses what was taught — is the one we were inside of, and only the transcript's grading
sentence prompted the audit that caught it.


## D7 — strict re-read of the instructions against the built artifact
**Question.** Does the artifact satisfy every instruction, checked against the HTML rather than
against our own rubric?

**Method.** Parsed `index.html` and `data/mechanisms.json` and tested each stated instruction
mechanically instead of re-reading our own ticks.

**Four gaps found, all now closed.**

1. **Our internal rubric contained a false tick.** Criterion 1.1 claimed "coverage table in app + README".
   Only the README had one; `index.html` contained no coverage table and no count. This is the same
   failure mode we documented in others — a claim that was true of intent and not of the artifact.
   Fixed: `index.html` now opens with "Coverage, stated as a count", 17 rows mapping each required
   item to the card covering it. The tick is now true.
2. **Five cards had no `antecedents`**, against the instruction that everything after standard
   attention "should then be presented as a response to some limitation of what came before".
   Fixed for sparse-transformer, MQA, linear attention and FlashAttention. Bahdanau correctly keeps
   none: it is genuinely first.
3. **Two added cards had no `note`**, against his standard that an added mechanism must state "where
   it belongs in the timeline". Fixed for Bahdanau and Luong.
4. **His predicted arc had never been tested.** See D8.

## D8 — testing the instructor's prediction
**Question.** The assignment predicts the ordering a correct timeline will show. Does our dated
table agree?

**Operationalisation.** His claim is about priorities *returning*, so per family we take the paper
ending that family's longest internal gap.

| family | first | longest internal gap | returns |
|---|---|---|---|
| foundation | 2014-09-01 | 665 d | 2017-06-12 |
| position | 2017-05-08 | **1408 d** | **2021-04-20** |
| cache | 2019-11-06 | **1293 d** | **2023-05-22** |
| sparsity | 2019-04-23 | 1267 d | 2023-09-29 |
| recurrent | 2019-01-09 | 507 d | 2022-07-14 (modern wave 2023-12-01) |
| scaling | 2023-06-26 | 835 d | 2025-12-13 |

**Result.** Five of seven stages hold in the order he gives. **One pair is inverted and it is not
marginal:** he places "cheaper decoding memory" before "better position handling"; the data puts
position 25 months earlier (2021-04-20 vs 2023-05-22). The inversion survives every rule we tried —
return date, first appearance (2017-05 vs 2019-11) and median (2021-04 vs 2023-05).

**Why the memory runs the other way, and why it vindicates the two-date design.** MQA, the first
cache mechanism, *was* published early — 2019-11-06 — and then sat unused for 1,293 days. Date the
cache era by publication and his ordering is right; date it by when the field acted and it is not.
That is exactly the distinction the shipped-date column exists to expose.

**Reported as ambiguous, not as an error:** recurrent-versus-sparsity. Sparsity returns 2023-09-29,
Mamba lands 2023-12-01 — two months apart, and if RMT (2022-07-14) is the recurrent return then his
order holds. We do not claim a two-month result.


## D9 — commissioned independent review, and what it found
**Question.** We had audited our own work twice and declared it compliant. Would reviewers who were
not told our conclusions agree?

**Method.** Three reviews run in parallel, none given our findings: (1) re-fetch every date from the
arXiv API without using our `verify_dates.py`, since that script is the thing under test; (2) grade
the *rendered HTML* against the assignment text and audit all 18 lesson sections for missed
mechanisms; (3) check every mechanism description against its source paper.

**Result: they falsified our headline finding and nine further claims.** This is the most important
entry in this log, because everything below had already survived two of our own audits.

| # | our claim | what is actually true |
|---|---|---|
| 1 | Q2-1: publish-to-ship lag "collapsed from 3.5 years to zero" | The column mixed third-party adoption with authors' own same-day code drops. The collapse was substantially definitional. Two dates were also wrong. **Retracted.** |
| 2 | GQA: "32 query heads and 8 KV heads means 4 groups" | 8 groups of 4. Groups = KV heads. The card contradicted itself one clause later. |
| 3 | PI extends "up to 32x" | The abstract says 32,768 tokens; from LLaMA's 2,048 that is **16x**. We read a token count as a multiplier. |
| 4 | FlashAttention is "bit-for-bit" exact | Mathematically exact, but tiling reorders FP accumulation, so not bit-identical. |
| 5 | Differential Transformer costs "roughly double the score computation" | The paper halves head count precisely to stay FLOP-matched. |
| 6 | Differential Transformer is "the direct descendant of the attention-sink diagnosis" | **Fabricated.** 2410.05258 never cites StreamingLLM and never uses "sink". Attention *noise* and attention *sinks* are different phenomena. |
| 7 | Q2-7: "neither author had the other's result" | ALiBi benchmarks against rotary explicitly and cites GPT-J popularising it. |
| 8 | MLA's 93.3% is "versus its own dense baseline" | DeepSeek 67B already used GQA-8. It is a saving on top of GQA. |
| 9 | Transformer-XL buys its context "at no extra training memory" | Stop-gradient removes gradient memory; the cached activations are real. |
| 10 | Q2-9 graded recurrent-vs-sparsity "ambiguous" | We failed to apply our own rule. The recurrent family's longest gap ends at RMT (2022-07-14), so the instructor's ordering **holds**. Six of seven stages, not five. |

**Coverage gaps they found that our own two audits missed:** the V4 Memory Stream had no card — we
had diagnosed it as a taught-but-unlisted mechanism in C8 and then left it as prose inside two other
cards, with no date, no timeline position and none of the six required fields. And required item 16
was mapped to NSA alone, when lesson §12 actually describes DeepSeek's **low-rank indexer** and never
says NSA. Partial RoPE, which the lesson flags as "the implementation detail people often miss", was
absent entirely.

**Rebuilt Q2-1.** With a single consistent definition of "shipped" — first public release of weights
or reference code, read from the GitHub/HuggingFace API — the sequence by publication order is
1295, -10, 50, 319, -8, 50, 59, 675, 0, 0, 0. **There is no trend in time.** What predicts the lag is
what the mechanism costs: everything that waited had a price (MQA degrades quality, 1295 d; attention
sinks fixes a bug you only meet after streaming millions of tokens, 675 d; ALiBi gives up retrieval,
319 d), and everything free or author-released shipped within two months or before its own paper.
That is the assignment's own thesis — every mechanism is a trade — measured rather than asserted.
The chart was rebuilt as a sorted comparison, because plotting lag against time implied a trend the
data does not contain.

**The lesson for our method.** Every one of these survived our own review because we were checking
the artifact against our notes rather than against the sources. The two checks that did work were
mechanical and adversarial: re-fetching from an API, and rendering a chart that made a false shape
visible. Self-audit found formatting and coverage problems; only outside review found wrong facts.


## D10 — second review round: the fixes were also wrong
**Question.** Round one found eleven errors. Are the fixes correct?

**Result.** Three more reviews found **six further defects, two of them worse than anything in round
one**, because they were failures of the correction process itself rather than of the original work.

1. **A card passed validation and never rendered.** `build.py` buckets cards into era ranges that
   stopped at 2025; the Memory Stream card is dated 2026-08-15, so it fell outside every bucket and
   was silently dropped. The build still printed `validation passed: 34 cards` while the page showed
   33, and the timeline SVG drew a dot linking to an anchor that did not exist. **The validator was
   checking the data, not the artifact.** Fixed with a fail-closed assertion that every validated
   card must appear in the rendered output; the control confirms it fires.
2. **A logged fix had never been applied.** Correction C9 listed the Transformer-XL activation-memory
   error as fixed and said "all corrected in place". The string was still there verbatim — the
   `replace()` had not matched. A log that claims a fix which is not in the artifact is worse than
   the original error, because it stops anyone looking again.
3. **The DSA card described a mechanism DeepSeek did not build.** We wrote it from lesson section 12:
   a *low-rank* indexer ranking *block summaries*. DeepSeek's own report calls it a **lightning
   indexer** selecting the **top-2048 individual tokens** per query, and explicitly contrasts itself
   with block granularity. "low-rank" appears nowhere in it. Logged as C11 — a genuine
   lesson-versus-primary-source conflict, which the project's precedence rule resolves in favour of
   the primary source.
4. **The shipped rule was violated by five of its own rows.** Round one's retraction promised
   "'shipped' now means one thing throughout". It did not: ALiBi, RoPE, MLA, sliding window and
   FlashAttention all had *authors' own releases predating their papers*, and we had credited later
   third parties. MQA's row used Falcon-40B when SantaCoder (2022-12-02) is 18 months earlier.
5. **MLA's 93.3% includes 6-bit KV quantization.** Architecture alone gives 82.2%; adding 6-bit
   against bf16 reproduces 93.34%, the published figure to two decimals. We had flagged the GQA
   baseline and the layer-count difference and missed the larger confound.
6. **Q2-9 miscounted its own table twice** — "six of seven" then "ambiguous" on a row the corrected
   rule resolves. It is five of seven.

**What the corrected data says.** Applying one release rule to every row separates the field in three
with no overlap: seven mechanisms shipped at or before their own paper because the authors released
the code (RoPE -29 d, MLA -15, sliding window -10, FlashAttention -8, ALiBi -3, YaRN 0, DSA 0); two
waited about two months where adoption was nearly free (GQA 57, NTK-aware 59); two waited years where
it cost something (attention sinks 675, MQA 1122). No trend in time within any group. This is the
third version of that finding and the first that survives its own rule.

**Process change.** Added `check_consistency.py`, which fails on stale counts, dead anchors,
unrendered cards, antecedent dates disagreeing with the referenced card, shipped rows missing the
field the analysis needs, and raw markdown leaking into rendered text. Six of round two's findings
were mechanically detectable and none of our prose audits had caught them.

**The standing lesson, now with two rounds of evidence.** Self-audit catches formatting, coverage and
structure. It does not catch wrong facts, and it does not catch fixes that silently failed to apply.
Both rounds of outside review had a high hit rate against work we had already declared clean.


## D11 — third review round, and the negative result
**Result.** A third review found six more, including two that collapsed the replacement finding.

- `mit-han-lab/streaming-llm` was created **2023-09-29T17:45:40Z**, the same day as the arXiv v1
  (GitHub API). We had dated attention sinks' release to gpt-oss (2025-08-04) and called the 675-day
  gap evidence of adoption cost. Wrong twice over: it violates our own rule, and gpt-oss's *learned
  per-head sink logit* is a different mechanism from the published keep-four-tokens trick.
- NTK-aware was classed as a 59-day third-party adoption via Code Llama, while `jquesnelle/scaled-rope`
  — the repository whose creation timestamp we use to *date the mechanism* — is its reference
  implementation. You cannot date a mechanism by a repo and then decline to count that repo as its
  release.

**With those corrected, the classification collapses.** Nine of eleven rows are at or before their
own paper. The `cost` class had shrunk to n = 1.

**So Q2-1 is now a negative result, and that is what we publish.** The shipped column measures
*publishing practice*, not adoption: if you write the paper, you push the repo. The two exceptions
are both Google papers published without weights, and their waits differ by twenty times — GQA 57
days, MQA 1,122. Cost plausibly governs that difference, but on two points from one lab, stated as
an observation after two retractions for overreaching on precisely this claim.

The design still earned its place: a single-date timeline would have let all three wrong versions
stand. It was the second column that made each failure visible.

**Also fixed this round:** NSA's 11.6x decode figure is the paper's Table 4 *expected* speedup
derived from memory-access volume, not a kernel timing — relabelled **stylized**, and the measured
9.0x/6.0x split into their own row. Differential Transformer does not need a custom kernel (the
paper reuses stock FlashAttention with two calls). Griffin's antecedent is the LRU (2303.06349), not
Mamba, which Griffin only benchmarks against — the same fabrication class as the one removed in D9.
Ring Attention's antecedent is Blockwise Parallel Transformer, named in its own title. Q2-9's
recurrent gap had silently become 614 days when the Memory Stream card was added, so the rule now
excludes gaps that end at the present. C4's "zero regex matches" was false — there is one match, a
path component in an image URL — though its conclusion held.

**Three rounds, 23 findings.** The rate is not falling as fast as it should, which is the argument
for the mechanical checks now in `check_consistency.py` rather than another prose audit.
