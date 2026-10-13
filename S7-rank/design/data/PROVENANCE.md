# design/data — derived artifacts and where they come from

Committed so the analysis scripts run from a clean clone. Small and derived;
the token stream itself (26 MB) is **not** committed — only what the analyses actually read.

| file | bytes | derived from | how |
|---|---|---|---|
| `codec.npz` | 1.2 MB | `theschoolofai/BrahmicTokenizer-131K` | for each of the 61,070 train-observed token ids: `byte_idx` (raw UTF-8 byte values, 32-wide, −1 pad), `blk_idx`/`off_idx` (Unicode block and in-block offset per character), `nbytes`, `nchars` |
| `train_freq.npy` | 488 KB | the training token stream | `np.bincount(train)`; 6,650,880 tokens, 60,779 of 61,070 types observed |
| `meta.json` | 4 KB | run config | `vocab=61070`, `vocab_full=131072`, `P=32`, `NB=64`, `langs=[en,hi,ml]`, `D_byte=8192`, `D_char=6144` |

**Vocabulary is restricted to observed tokens.** Unobserved rows receive no gradient, so
excluding them is exact rather than an approximation, and it is what makes `rank(S)` a statement
about trained capacity rather than allocated capacity.

**Corpus provenance (D4).** Byte-range **prefixes** of `ai4bharat/IndicCorpV2`, not random samples —
any domain skew in the head of each file is inherited. Evaluation is FLORES-200 devtest.
