# E10 — capacity-matched control (DESIGN, NOT YET RUN)

Purpose: E7's control was confounded (rand4 trained 2,560 rows vs char4's 479, a 5.3x gap).
E10 asks the question E7 was supposed to ask: **at matched trained capacity, does index
structure help, hurt, or do nothing?** Design is under review before any GPU time is spent.

## Measured facts E10 must respect

Live rows (rows of W receiving gradient from >=1 token with train freq >=1), n=60,779 types:

| arm | D | live rows | trained params |
|---|---:|---:|---:|
| byte | 8192 | 1,855 (22.6%) | 474,880 |
| char2 | 6144 | 1,305 | 334,080 |
| char3 | 2816 | 547 | 140,032 |
| char4 | 2560 | 479 (18.7%) | 122,624 |
| rand4 | 2560 | 2,560 (100%) | 655,360 |
| hashbyte4 | 2560 | 2,415 | 618,240 |
| hashchar4 | 2560 | 2,520 (98.4%) | 645,120 |

Spearman(live rows, val loss) = -1.0000 over all seven arms.

## Proposed arms (d_model 256, 4L, 3000 steps, seq 256, batch 8, LR 3e-4, seeds {0,1} unless changed)

| id | arm | construction | live rows | pairs with |
|---|---|---|---:|---|
| A1 | byte8192 | published grid, unchanged | 1,855 | baseline (already have: 6.1484 / 6.1557) |
| A2 | byte1855 | same grid, dead rows removed, D=1855 | 1,855 | A1 — pruning must be FREE; if A2 != A1 the harness is wrong |
| A3 | hashchar1855 | hashchar4 hashed into D=1855 | ~1,855 | A1 — structure vs hash at matched capacity |
| A4 | char4 | unchanged | 479 | baseline (already have: 6.3214 / 6.3250) |
| A5 | rand479 | rand4 with indices drawn from 479 slots | 479 | A4 — **the control E7 should have been** |
| A6 | dense | nn.Embedding, raw (as in E6) | 61,070 | baseline (already have) |
| A7 | dense_zn | nn.Embedding + per-token z-norm across d_model | 61,070 | A6 — **the normalisation control E6 never had** |

A2 is a harness assertion, not a result. A5 and A7 are the two experiments that can each
independently kill a published claim of ours.

## Pre-registered hypotheses

- **H10a** — A5 (rand479) will NOT beat A4 (char4) by more than the 0.0115 seed spread.
  If refuted (rand479 still clearly wins), index structure is genuinely anti-useful at
  matched capacity and E7's conclusion is rehabilitated.
- **H10b** — A3 (hashchar1855) will NOT beat A1 (byte8192) by more than the seed spread.
  If refuted, E8's codec claim survives in corrected form: equal capacity, 4.4x smaller
  allocated table, no loss.
- **H10c** — A7 (dense_zn) will beat A6 (dense) by enough to consume a material fraction of
  E6's -3.97% "byte beats dense" gap. If refuted, E6's replication stands.
- **H10d** — A2 == A1 to 4 dp. Any deviation invalidates the harness, not the hypothesis.

**Pre-committed:** if H10a and H10b both hold (both gaps within seed spread), the codec
recommendation stays withdrawn permanently and the contribution is the capacity diagnostic
(rho = -1.000) plus E1, not a codec.

## Known open questions I want the review to settle before the run

1. **Is "live row count" the right matched quantity?** It ignores the *collision pattern* —
   which tokens share which rows, weighted by frequency. Reviewer proposed frequency-weighted
   effective rank of S^T diag(f) S (rand4 1143.4, hashchar4 188.0, byte 113.5, char4 29.4).
   Should E10 match effective rank instead of / in addition to live rows? If effective rank is
   the real variable, matching live rows is another confounded control and E10 fails the same
   way E7 did.
2. **LR.** Single LR 3e-4 across arms whose trained capacity now differs by design. Does E10
   need a per-arm 3-point sweep (1e-4/3e-4/1e-3, compare each at its own optimum)? That is 3x
   the cost. If yes, which arms genuinely need it vs which can share.
3. **Codec seed n=1.** Every hash/random arm uses one draw. For a "random beats structured"
   claim n=1 in the randomness is the whole ballgame. How many codec seeds are the minimum
   worth running — and is that more valuable than any other arm here?
4. **Val is 49.5/19.0/31.5 en/hi/ml, not 1:1:1.** Rebuild val to true 1:1:1, or keep it and
   report per-language only? Rebuilding changes the denominator on every prior number.
5. **Metric.** Aggregate NLL is dominated by 960 types = 57.7% of occurrences. Move headline to
   macro-average NLL over token *types*, and/or held-out-type NLL (hold N mid-frequency types
   out of training entirely)? The held-out design tests the property codecs actually exist for,
   which rand4 structurally cannot do.
6. **Budget.** ~8 min/run on the M3 Pro. 5 new arms x 2 seeds ~= 80 min. A 3-point LR sweep
   takes it to ~4h; 5 codec seeds adds more. Given ~4h as a soft ceiling before we should
   instead be writing up, what is the highest-value subset?

## Scale caveat that no amount of M3 time fixes

arXiv:2605.09751 reaches parity for randomly-recoded fixed codes at 17B tokens (375x our
budget). Whatever E10 finds at 6.65M tokens and ~0.9 epochs is a statement about learning
speed in an undertrained regime, and must be labelled as one.
