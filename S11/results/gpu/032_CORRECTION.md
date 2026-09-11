# Correction: job 032 ran at width 512, not width 256

**Raised 2026-09-11, before any 032 number entered `README.md`. Found by diffing the staged
script against job 021 rather than by reading its output.**

## The error

`032a_s11_lrbatch_04_16.py` and `032b_s11_lrbatch_32.py` were cloned from **job 022** (the
width-**512** LR sweep), not from job 021 (width 256). Line 40 reads `WIDTH = 512`. The
`extra=` dict passed to `run()` hard-codes `width=256`, so **every row of
`032a_s11_lrbatch_04_16.json` records `"width": 256` for a model that was 512 wide.** The
docstring's pre-registration also says width 256.

## The evidence that settles it

`input_params` is the dense input-embedding parameter count, `NV × D_MODEL`, and it is
computed at runtime rather than declared:

| | |
|---|---:|
| `61070 × 256` | 15,633,920 |
| `61070 × 512` | **31,267,840** |
| recorded in every 032a row | **31,267,840** |
| recorded in every 022 row (width 512) | **31,267,840** |

All ten 032a rows carry 31,267,840. The runs were width 512.

A second, independent check: at `lr = 6e-4`, 032a measures 6.1087 (batch 4) and 5.5427
(batch 16). Job 022 — width 512, batch 8, same LR — measures 5.8060, which lies between
them, as a batch-8 curve must. Job 021 (width **256**, batch 8, `lr = 6e-4`) measures
5.9810, which does **not** sit where a width-256 batch-8 point would relative to these two.

## What it costs, and what it does not

The batch axis is **not** confounded: batches 4 and 16 ran in the same job at the same
width, and 032b runs batch 32 from the same script. Every point in the series is width 512.
So the measured quantity is *"how the optimal LR moves with batch size at width 512,
depth 4"* — which is the question C1 needs answered. Only the label was wrong.

Two consequences:

1. **The width-256 batch-8 optimum (2.37e-3) is not the batch-8 anchor for this series**
   and must not be mixed in. The correct anchor is **job 022** — width 512, depth 4,
   batch 8, run 2026-09-10 as an independent job on the grid `{3e-4 … 8e-3}`, optimum
   interpolated at **1.25e-3**. That it was measured separately, before this series was
   conceived, makes it a stronger anchor than a point run inside the same job.
2. The two grids differ — 022 uses `{3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3}`, 032 uses
   `{6e-4, 1.2e-3, 2.4e-3, 4.8e-3, 9.6e-3}`. Both are 2× log grids, offset from each
   other. Minima are therefore compared by **parabolic interpolation in log-LR**, not by
   grid argmin, so the offset does not bias the comparison. Grid argmin alone would
   quantise every optimum to its own grid and is not used.

## What was *not* done

The archived JSON is left byte-identical to what the runner produced. Historical
measurements are not rewritten. `code/verify.py` asserts
`input_params == 61070 * 512` on every 032 row, so any consumer that reads `width` as 256
fails the verifier rather than passing quietly.

## Why this was possible

Every job in this queue is a clone of a previous job with the varying constant substituted.
This is the **second** time the substitution was missed: jobs 024a/b/c were cloned from 022
and left at depth 4 (caught before the runner reached them). The class of bug is "the
recorded metadata and the executed constant come from different places." The verifier check
above closes it for width by tying the label to a runtime-computed quantity.
