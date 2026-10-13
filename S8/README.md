# How attention works now — a chronological timeline

**ERA V5, Session 8: Modern Attention Variants.**

Every attention mechanism from the session, ordered by **when it was launched** — not by when it
was taught and not grouped into a taxonomy. 34 mechanisms: all **17** items on the
required list, plus **13** we added. Two of those additions are mechanisms the session
**taught** that the minimum list does not name. Each is written as the answer to a problem that
existed on its date, with what it buys, what it gives up, and when you would actually choose it.

- **Timeline** → `index.html`
- **What the timeline shows (Question 2)** → `findings.html`
- **Corrections and limitations** → `corrections.html`

## Where the dates come from

This is the part the assignment says is easiest to get wrong and easiest to check, so here is the
whole thing, checkable.

**Convention.** The sort key is the **arXiv v1 submission date** — the day the idea became public,
not the conference publication date and not the day it became popular. Where a mechanism shipped in
a well-known model at a materially different time, the shipped date is carried as a second column
rather than replacing the first. Two mechanisms that share a paper share a date, and that is not an
error: standard attention and sinusoidal encoding are both *Attention Is All You Need*, 2017-06-12.

**Primary source.** Every arXiv date was fetched from the arXiv API (`export.arxiv.org/api/query`),
whose `published` field is the v1 submission timestamp. Nothing was taken from memory, from a
secondary summary, or from a model's recall.

**Reproduce it.** The check is a script, not a claim:

```bash
python3 verify_dates.py     # re-fetch every arXiv v1 date, diff against stored, exit 1 on drift
python3 analyse.py          # recompute every number quoted on the findings page
python3 build.py            # validate all cards, regenerate the three HTML pages
```

Last run: **31 of 34 dates re-verified against the arXiv API, 3 with no paper by design, 0 failures** (2026-08-21T22:44:01Z). Raw API response is committed at `data/arxiv-response.xml`, and the diff at
`data/date-verification.json`.

**Claims we got wrong and corrected.** Three independent reviews found **twenty-three** errors across
three versions of this site that we had already audited ourselves and believed clean — including in
the headline finding, which has now been retracted and rebuilt twice. Everything is corrected in
place with the retraction visible; nothing was quietly swapped. The headline finding is now a **negative result** — the thing it set out to measure cannot be
measured that way. Details in `corrections.html` C9 and C11, and in `EXPERIMENT_LOG.md` D9/D10/D11.

**One date we could not pin.** NTK-aware scaled RoPE has no paper. It is a Reddit post by u/bloc97,
and reddit.com returns HTTP 403 from our build environment, so we could not read the post's
timestamp directly. It is bounded instead — `github.com/jquesnelle/scaled-rope` was created
2023-06-26T13:27:56Z and bloc97's PR #1 opened 2023-07-07T20:40:33Z, both from the GitHub API, and
YaRN's bibliography cites the post as `bloc97 [2023a]`. It is rendered on the timeline as *date
bounded, not exact*. The commonly repeated 2023-06-30 is plausible and we do not assert it.

### Every date, with its source

| v1 date | mechanism | shipped | primary source | status |
|---|---|---|---|---|
| 2014-09-01 | Additive attention (Bahdanau) *(added)* | — | [arXiv 1409.0473](https://arxiv.org/abs/1409.0473) | verified |
| 2015-08-17 | Multiplicative / dot-product attention (Luong) *(added)* | — | [arXiv 1508.04025](https://arxiv.org/abs/1508.04025) | verified |
| 2016-11-07 | Absolute learned position embeddings | — | [arXiv 1611.02344](https://arxiv.org/abs/1611.02344) | verified |
| 2017-06-12 | Sinusoidal position encoding | — | [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) | verified |
| 2017-06-12 | Scaled dot-product attention | — | [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) | verified |
| 2019-01-09 | Segment-level recurrence (Transformer-XL) *(added)* | — | [arXiv 1901.02860](https://arxiv.org/abs/1901.02860) | verified |
| 2019-04-23 | Sparse attention (strided / fixed patterns) | — | [arXiv 1904.10509](https://arxiv.org/abs/1904.10509) | verified |
| 2019-11-06 | Multi-query attention (MQA) | 2022-12-02 | [arXiv 1911.02150](https://arxiv.org/abs/1911.02150) | verified |
| 2019-11-13 | Compressive memory *(added)* | — | [arXiv 1911.05507](https://arxiv.org/abs/1911.05507) | verified |
| 2019-12-25 | Top-k / explicit sparse attention | — | [arXiv 1912.11637](https://arxiv.org/abs/1912.11637) | verified |
| 2020-04-10 | Sliding-window attention | 2020-03-31 | [arXiv 2004.05150](https://arxiv.org/abs/2004.05150) | verified |
| 2020-06-29 | Linear attention | — | [arXiv 2006.16236](https://arxiv.org/abs/2006.16236) | verified |
| 2021-02-22 | The delta rule (fast-weight programmers) | — | [arXiv 2102.11174](https://arxiv.org/abs/2102.11174) | verified |
| 2021-04-20 | RoPE (rotary position embedding) | 2021-03-22 | [arXiv 2104.09864](https://arxiv.org/abs/2104.09864) | verified |
| 2021-08-27 | ALiBi (attention with linear biases) | 2021-08-24 | [arXiv 2108.12409](https://arxiv.org/abs/2108.12409) | verified |
| 2022-05-27 | FlashAttention *(added)* | 2022-05-19 | [arXiv 2205.14135](https://arxiv.org/abs/2205.14135) | verified |
| 2022-07-14 | Recurrent memory tokens (RMT) *(added)* | — | [arXiv 2207.06881](https://arxiv.org/abs/2207.06881) | verified |
| 2022-12-20 | xPos (length-extrapolatable Transformer) *(added)* | — | [arXiv 2212.10554](https://arxiv.org/abs/2212.10554) | verified |
| 2023-05-22 | Grouped-query attention (GQA) | 2023-07-18 \* | [arXiv 2305.13245](https://arxiv.org/abs/2305.13245) | verified |
| 2023-06-26 | NTK-aware scaled RoPE | 2023-08-24 | [no paper](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) | **bounded** |
| 2023-06-27 | Position Interpolation (PI) *(added)* | — | [arXiv 2306.15595](https://arxiv.org/abs/2306.15595) | verified |
| 2023-08-31 | YaRN | 2023-08-31 | [arXiv 2309.00071](https://arxiv.org/abs/2309.00071) | verified |
| 2023-09-29 | Attention sinks / StreamingLLM | 2025-08-04 | [arXiv 2309.17453](https://arxiv.org/abs/2309.17453) | verified |
| 2023-10-03 | Ring Attention *(added)* | — | [arXiv 2310.01889](https://arxiv.org/abs/2310.01889) | verified |
| 2023-12-01 | Mamba (selective state space) *(added)* | — | [arXiv 2312.00752](https://arxiv.org/abs/2312.00752) | verified |
| 2024-02-29 | Hybrid depth schedules (interleaved layer types) *(added)* | 2024-03-28 | [arXiv 2402.19427](https://arxiv.org/abs/2402.19427) | verified |
| 2024-05-07 | Multi-head latent attention (MLA) | 2024-04-22 | [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) | verified |
| 2024-06-10 | DeltaNet, parallelised | — | [arXiv 2406.06484](https://arxiv.org/abs/2406.06484) | verified |
| 2024-10-07 | Differential Transformer *(added)* | — | [arXiv 2410.05258](https://arxiv.org/abs/2410.05258) | verified |
| 2024-12-09 | Gated DeltaNet | — | [arXiv 2412.06464](https://arxiv.org/abs/2412.06464) | verified |
| 2025-02-16 | Native Sparse Attention (NSA) | — | [arXiv 2502.11089](https://arxiv.org/abs/2502.11089) | verified |
| 2025-09-29 | DeepSeek Sparse Attention (the lightning indexer) | 2025-09-29 | [no paper](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) | **bounded** |
| 2025-12-13 | DroPE (dropping positional embeddings) | — | [arXiv 2512.12167](https://arxiv.org/abs/2512.12167) | verified |
| 2026-08-15 | Memory Stream (cross-chunk state, LightningLM V4) *(added)* | — | [no paper](https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms8msrhwu8ss0yxe9zg/lesson) | **bounded** |

## Coverage

All **17** required items are covered, mapped card-by-card in a "Coverage, stated as a count" table
at the top of the app. 34 cards in total, plus 13 we added — two of which the session
**taught** but the minimum list omits (hybrid depth schedules, and the V4 Memory Stream).

## What the checks actually check

Two of them, each with a paired control that makes it fail — a check that only ever passes is
indistinguishable from no check.

| check | what it enforces | negative control | result |
|---|---|---|---|
| `verify_dates.py` | every stored date matches the arXiv API | corrupt RoPE's date by one day | exits 1, reports `MISMATCH` |
| `build.py` | no card may have an empty `gives_up` | inject a card with five advantages and no cost | build fails, page not emitted |
| `build.py` | every validated card must actually render | put a card outside every era bucket | build fails, names the invisible card |
| `check_consistency.py` | pages must not contradict the data or each other | drop a `shipped` row's class | exits 1, names the silently-excluded row |

The third and fourth exist because review found bugs of exactly those shapes: a card that passed
validation and was then silently dropped from the page by an era-bucket range, and counts in prose
that had drifted from the data across five pages.

The second exists because the assignment says it directly: *"If your page contains a method with
five advantages and no meaningful downside, go back and investigate it again."* Here that is a build
error rather than a style note.

## Corrections, including to ourselves

Detail and sources in `corrections.html` — ten entries. The headline ones:

1. **We commissioned an independent review and it falsified our own headline finding.** We had
   claimed the publish-to-ship lag "collapsed from 3.5 years to zero". The column mixed third-party
   adoption with authors' own same-day code releases, so the collapse was substantially an artefact
   of switching definitions, and two dates were wrong. Retracted in place, not quietly swapped. The
   rebuilt finding is better: adoption lag tracks **what the mechanism costs you**, not the calendar.
   Nine further errors of ours were found and fixed the same way — see C9.
2. **The assignment's minimum list is shorter than the session.** Lesson §13 teaches hybrid depth
   schedules (`DDDGDDDG`) and §14 teaches cross-chunk O(1) state (the V4 Memory Stream); neither is
   on the 17-item list. §12 teaches DeepSeek's **low-rank indexer**, which the list names only
   obliquely — we had mapped item 16 to NSA alone, and the lesson never says NSA.
3. **DroPE is an acronym collision.** `DRoPE` (2503.15029, 2025-03-19) is a self-driving trajectory
   paper. The intended DroPE is 2512.12167, v1 **2025-12-13**. The wrong paper is the better string
   match and nine months earlier.
4. **The lesson says DroPE's mechanism is not established; it was published eight months earlier.**
   arXiv 2512.12167 states it in the abstract: drop positional embeddings after pretraining, then a
   short recalibration phase.
5. **The lesson contains no dates** — zero four-digit years in 1,532 lines.
6. **Two transcript slips**, both read as ASR artifacts rather than instructor errors: the
   Transformer dated to 2018, and "GSA"/"MSA" inside the assignment sentence itself.
7. **We could not find the Session 7 correction** the assignment refers to, after searching four
   places. Reported as a non-finding rather than guessed at.
8. **Negative control:** we recomputed the lesson's KV-cache arithmetic and it is exactly right
   (6.442 GB; 51.54 GB at eight users). Reported because a hunt that only finds errors is a bias.

## Build

No build step is required to view the site — `index.html` is committed and opens from the file
system with no server. `build.py` regenerates it from `data/*.json`, which is the single source of
truth. The pages are readable with JavaScript disabled; JS only adds a family filter, and the order
stays chronological either way.

## Contents

See `MANIFEST.md` for the file list with SHA-256 for each.
