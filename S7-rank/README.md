# The 32-byte window is not the constraint. Rank is.

**ERA V5 · Session 7 · Problem #3** — *"Today Kronecker is limiting to presenting 32 positions for
every word… That's a waste of space. What can we do? How can it be dynamic and doesn't force us to
crop a word?"*

This is one problem, not a mix. Everything below tests **#3** and nothing else. Where a result also
bears on #5 (invertibility) it is marked and scoped, never claimed.

---

## TL;DR

The assignment asks how to make the 32-position window **dynamic**. We measured the window and found
it is **not the binding constraint**, so widening it buys nothing:

- Of the 8,192 cells in the published `d_c=256 × d_p=32` grid, **1,855 are reachable** by any token
  in the corpus and **≈200 are effective** — **2.4% of allocation** carries the signal. *(measured)*
- Widening the window from 32 to 64 bytes on the shipped `BrahmicTokenizer-131K` is a **literal
  no-op**: identical reachable cells, identical rank, identical collisions, identical floor, because
  **no token in the vocabulary exceeds 32 bytes**. *(measured)*

But the window *does* bind — one step upstream, at tokenizer construction. Because the tokenizer was
**built** under the cap:

- At V=128,000 the 32-byte cap costs **Tamil +9.08%, Malayalam +8.53%, Kannada +6.17%,
  Hindi +0.49%, English +0.00%** of fertility, and the gap **grows monotonically with vocabulary
  size** with no sign of saturation. *(measured, pre-registered falsifier survived)*

So the honest answer to #3 is: **don't make the window dynamic — the window is slack at inference and
binding at construction. The quantity that actually governs quality is the rank of the codec's
feature matrix**, which we show predicts every arm we trained, and which we have not seen reported
for this family of methods.

**Three of our own headline claims were killed by controls we designed against ourselves.** That
sequence is the contribution, and §6 reports it in full.

---

## 0. Claim ledger

Every claim we make, what it rests on, and what would kill it. **Static** claims need no trained
model and do not inherit the toy-scale caveat; **trained** claims do, and are the ones §8 stresses.

**Read the `kind` column first.** The seven **static** claims need no trained model and carry the
weight of this submission. The **trained** claims all sit at one small scale with an unresolved
scale objection (§8) and effect sizes quoted as descriptive ratios, not tests — treat them as
directional.

| # | claim | evidence | kind | falsifier | status |
|---|---|---|---|---|---|
| C1 | 8,192 allocated cells → 1,855 reachable → ≈200 effective (2.4%) | `verify.py` B, C | **static** | a reachable-cell count above 1,855 on this corpus | **holds** |
| C2 | `byte64` is a no-op on the shipped tokenizer | `e17e_anchors.json` | **static** | a vocabulary token longer than 32 bytes | **holds** |
| C3 | the cap costs ta +9.08% / ml +8.53% / en +0.00% fertility, growing with V | `e13b_results.json`, 60 BPE trainings | **static** | capped ≈ uncapped within noise out to 256K | **holds** (pre-registered) |
| C4 | the cost in C3 is the cap, not a pre-tokenizer artifact | `e20b_pretok_behavior.json` | **static** | BrahmicTokenizer showing a `\p{L}+`-style Indic floor | **holds** (1.07×) |
| C5 | Falcon's `Digits`-after-`ByteLevel` hits Kannada alone | Unicode layout → measurement | **static** | any other Brahmic script equally hit | **holds** (kn 19.6 vs ml 8.4) |
| C6 | at V=1M a byte codec head leaves 9.35% of ids inseparable, costing 0.00209 bpb | `e17d_collision_cost.json` | **static** | — it is an exact bound | **holds** |
| C7 | a phoneme basis cuts that to 3 ids at 237× head compression | `e17d`, `e17c` | **static** | — | **holds**, untrained |
| C8 | `rank(S)` orders arms that differ in rank; ρ = −1.0000 held out | `e22_stats.py` | trained | any inversion between arms differing in rank | **holds** |
| C9 | `rank(S)` is *silent* on rank-matched arms | three matched pairs | trained | — this is the limit, not the claim | **holds** |
| C10 | byte structure is worth +0.1337 nats at exactly matched rank and load | `010_e14.json` + audit | trained | `byte_shuf` ≈ `byte` within seeds | **holds** |
| C11 | permuting position-bound bigram alignment *helps* by 0.0383 nats | `012_e19.json` + audit | trained | `bigram_pos_shuf` ≈ `bigram_pos` | **sign holds**; **16% is a measured rank leak** (`CORRECTION C4`), mechanism **unexplained** |
| C12 | `bigram_pos` beats the published codec by 0.1133 nats at 2× params | `012_e19.json` | trained | — | **holds**; gain is capacity, not conjunction |
| C13 | the dense control beats every codec by 0.20–0.49 nats | `010_e14.json` | trained | a codec beating dense | **holds** (unmatched params) |
| C14 | `1/√L` in the published Eq. 1 is provably inert | `verify.py` F | **static** | — algebra | **holds** |
| C15 | our reimplementation matches the published reference to 4.2e-06 | `e23_fidelity.json` | **static** | any divergence beyond float32 | **holds** |
| — | *do the trained effects survive scale?* | §8, in flight | trained | — | **OPEN.** 3× training moves the baseline 0.504 nats, 4× our largest effect |

### Withdrawn — claims we published and then killed

| withdrawn claim | why it died | killed by |
|---|---|---|
| "hashchar4 beats the byte codec at 3.2× fewer parameters" | 3.2× was *allocated*; in trained rows hashchar4 is **1.36× larger** | our own live-row audit |
| "byte structure contributes ≤0%, it is worse than random" | the control changed capacity (2,560 vs 479 trained rows), so the conclusion did not follow | E14's permutation null (C10) |
| "code overlap predicts loss, ρ = +0.900" | n=5, post-hoc, p=0.083, and it needed a patch that was the right variable all along | rank(S) (C8) |
| "rank(S) is *the* mechanism" | its own pre-registered falsifier fired at 0% rank mismatch | E14 (C10) |
| "conjunctions repair the linearity ceiling" | the gain survived; the explanation did not | E18/E19 controls (C11) |
| "`rank(E) ≤ rank(S)` caps the table" | true but **non-binding** at `d_model=256` | self-review (`CORRECTION C3`) |
| "E21's 3,000-step checkpoint replicates the 3,000-step runs" | cosine LR schedule makes them different experiments | first E21 run (`CORRECTION C2`) |

---

## 1. Mechanism, stated precisely enough to reimplement

The published codec (arXiv:2605.29459, Eq. 1) is

```
κ(b) = (1/√L) · Σ_p  c_{b_p} ⊗ p_p          d_c = 256, d_p = 32, D = 8192
e(v) = zn(κ(b_v)) · W                        W ∈ R^{D × d_model}, the only learned part
```

Strip the notation. `c_{b_p} ⊗ p_p` with one-hot `c` and `p` **is a one-hot in `D`**. So `κ(b)` is a
multi-hot indicator of the `(byte, position)` cells a token occupies, and the embedding table is

> **`E = S W`**, where `S` is fixed before training and never learned.

For the published byte codec `S` is binary, because a cell is `byte × 32 + position` and is unique
per slot. In general it is non-negative **integer**: `KronEmb` computes `W[idx].sum(1)`, so a cell
occurring twice in one token contributes twice. That distinction is not pedantic — it changed a rank
we had published (`CORRECTION C1`).

Three consequences follow immediately. We have not found any of them stated in the published
description we work from.

| # | consequence | why it matters |
|---|---|---|
| **M1** | `E = S W` ranges over a **linear subspace of dimension `rank(S) · d_model`** | the embedding table's independent knobs are fixed before training. The published codec gets **443,648** of them — **2.8%** of a dense table's freedom — against **2,097,408 allocated parameters**, so `W` is **4.7× redundant with itself** |
| **M2** | the map is **linear in the features**, so `emb("ab") = emb(a@1) + emb(b@2)` exactly | no conjunction is representable at **any** training budget |
| **M3** | `D_allocated` is an upper bound, not a capacity | the axis the parameter-efficiency argument lives on may not be reached |

*(The weaker `rank(E) ≤ rank(S)` is true but **non-binding** here: `rank(E) ≤ min(rank(S), d_model)`
and every arm has `rank(S) > d_model = 256`, so `rank(E) = 256` throughout. We state the subspace
dimension instead, which is the quantity that actually varies. See `CORRECTION C3`.)*

**The imposed structure, named.** Two tokens share representation mass exactly when they share a
byte at the same absolute offset. That is a **Smolensky tensor-product representation with absolute
position binding** — and it is equivariant to nothing. `cat`/`scat` have cosine **0.000**.

**What would make this wrong.** If `rank(S)` failed to order arms by loss, M1 would be inert. If
permuting a code at fixed rank cost nothing, the structure would carry no information beyond capacity.
We ran both. **The first held for arms that differ in rank and is silent — provably — on arms matched
in rank** (§5). **The second held for the byte family and reversed for a redundant one** (§6.3).

**Reversal condition for the whole design.** If a codec arm beats a dense table **at matched trained
parameters**, the "a structured table is worth its cost" premise is live and we would drop this
framing. **That matched comparison has not been run** — it needs a rank-limited dense table, which we
did not build. What we have is weaker and pointed the same way: the dense control beats every codec
arm by 0.20–0.49 nats while having 3.7–24× *more* input parameters (§4).

---

## 2. How we prove it, as a method — stated before the results

Every **trained comparison** below was written as a hypothesis with a **named refuting metric and a
numeric threshold, committed to git before the run that tested it**. Commit hashes are in
`EXPERIMENT_LOG.md`; the pre-registration is in the job file's docstring, committed with the job and
not written afterwards. The static analyses (§3.1, §3.4, §9) are measurements rather than hypothesis
tests, and are labelled as such — except §3.3 and §3.4's Kannada prediction, which *were*
pre-registered and are marked where they appear.

**The control we rely on, and why it is the right one.** A random-index codec (`rand4`) is *not* a
control for structure, because randomising indices changes three things at once — alignment, rank,
**and row-load uniformity** (Gini 0.20 vs 0.89 for every structured codec; §6.3). Our control is a
**frequency-stratified permutation null**: permute *which token receives which code*, within
(frequency-quantile × slot-count) strata. This is a row permutation of `S`, so

- rank is preserved **exactly** — *when the whole row is permuted* (E14). E19 permutes only one
  block of each row and does **not** preserve rank; measured leak +4.46%, worth 16% of its effect
  (`CORRECTION C4`). We state which control is which rather than claiming exactness for both.
- live rows, trained parameters, total gradient touches and the full row-load distribution are
  preserved **by construction**,
- **only** the alignment between code overlap and string overlap is destroyed.

Every job **prints its own audit** — live rows before/after, total touches before/after, and the
load ratio at six quantiles — so the control's tightness is *reported*, not asserted. Measured:
`480 → 480`, `1,416,040 → 1,416,040`, load ratio `1.0` at every quantile.

**Labels.** Every number is exactly one of **measured** (we computed it), **stated** (the course or a
paper asserts it), **stylized** (an illustrative formula with no empirical backing). We never blur them.

**Is our reimplementation faithful?** We built the codec from the paper's description, so this has to
be checked rather than assumed. Against the reference implementation on PyPI
(`kronecker-embeddings` 0.1.1, Apache 2.0), over 3,000 sampled tokens:
**max abs difference 4.211e-06, cosine ≥ 0.9999997** — float32 precision. The reference's own
docstring gives the storage convention as `index = byte_value * pos_dim + pos`, which is exactly what
every `S` in this repo is built on.

**Our one deviation, quantified.** We z-normalise *after* projection across `d_model`; the paper and
reference z-normalise *before* projection across `D`. The resulting embeddings differ at cosine
**min 0.9679 / mean 0.9978** — genuinely a different map, so our trained arms replicate a **variant**,
and we say so. It does **not** touch the rank results: z-norm is a per-row rescale plus a rank-1
subtraction, and measured, rank is **identical** (1,060 → 1,060).

---

## 3. The window is not the constraint — measured

### 3.1 What the published grid actually uses

`design/verify.py` recomputes all of these from `design/data/` and **exits non-zero** on disagreement.

| quantity | value | |
|---|---:|---|
| allocated cells `d_c × d_p` | 8,192 | *stated* (paper) |
| allocated parameters `(D+1) × 256` | 2,097,408 | *measured* |
| **reachable** cells (any token in a 131,072 vocabulary) | **2,789** (34.0%) | *measured* |
| **reachable** cells (61,070 tokens observed in training) | **1,855** (22.6%) | *measured* |
| **rank(S)** over the 60,779 trained types | **1,733** | *measured* |
| **effective** rows, `exp(H)` of the frequency-weighted row load | **≈200** | *measured* |
| **effective ÷ allocated** | **2.4%** | *measured* |

Two independent statistics agree on the order of the last line: row-load entropy `exp(H) = 200.4`,
and the frequency-weighted Gram effective rank `= 105.4`. The design allocates 8,192 rows and
concentrates its training signal on a couple of hundred.

**60,779 live types share 1,733 independent directions — 35.1 types per degree of freedom.
59,046 of 60,779 token embeddings are forced linear combinations of the others before training starts.
The whole table is confined to a subspace of dimension 1,733 × 256 = 443,648, which is 2.8% of the
freedom a dense table would have.**

### 3.2 Widening the window is a no-op on the shipped tokenizer

Measured on `theschoolofai/BrahmicTokenizer-131K`, on **raw bytes** (§7.2 explains why that matters):

| codec | D | reachable | rank | collisions | unigram floor (bpb) |
|---|---:|---:|---:|---:|---:|
| byte32 *(published)* | 8,192 | 2,789 | 2,457 | **0** | 0.1226 |
| **byte64** | 16,384 | **2,789** | **2,457** | **0** | **0.1226** |

Identical in every column, because **no token in the vocabulary exceeds 32 bytes**. There is nothing
for a wider window to hold. *(measured)*

### 3.3 …but the window binds at construction, and only for Indic

The tokenizer was **built** under the cap, so the cost was paid earlier. We trained 60 BPE
vocabularies, capped at 31 characters vs uncapped, on IndicCorpV2, and evaluated on FLORES devtest.

**Pre-registered before the run** — *"if capped and uncapped Malayalam curves stay within noise out
to 256K, the cap is slack even asymptotically, the architecture→tokenizer claim collapses, and we
drop it and say so."*

capped − uncapped fertility (tokens/word):

| V | en | hi | kn | ta | ml |
|---:|---:|---:|---:|---:|---:|
| 4,000 | 0.0000 | 0.0005 | 0.0005 | 0.0123 | 0.0034 |
| 32,000 | 0.0000 | 0.0028 | 0.0390 | 0.0692 | 0.0605 |
| **128,000** | **0.0000** | 0.0058 | 0.0920 | 0.1312 | **0.1402** |

As a percentage penalty at V=128,000: **ta +9.08%, ml +8.53%, kn +6.17%, hi +0.49%, en +0.00%.**
English is **exactly** 0.0000 at all six vocabulary sizes. Every Indic language diverges, and the gap
**grows monotonically** at every step with no saturation. The falsifier did not fire. *(measured)*

The mechanism is E1: the 32-**byte** window holds ~26 Latin characters but exactly
**⌊31/3⌋ = 10** characters for 7 of 9 Brahmic scripts — a **2.60×** asymmetry, invariant under `d_p`
(we swept `d_p` = 16/32/48/64/128/256; the ratio stays 3.0–3.2×). *(measured)*

### 3.4 The control §3.3 needs: is that fertility cost the cap, or the pre-tokenizer?

BPE can never merge across a pre-tokenizer boundary, so a regex that splits on `\p{L}+` shreds Indic
words **before BPE runs** — Indic vowel signs are Unicode category **Mark**, not Letter. If
BrahmicTokenizer had that defect, §3.3 would be measuring a regex, not the cap. We drove twelve
shipped pre-tokenizers and counted what they emit, per whitespace-word, on FLORES devtest:

| tokenizer | en | ml | kn | worst-Indic ÷ en |
|---|---:|---:|---:|---:|
| tiiuae/falcon-7b | 1.176 | 8.500 | **19.721** | **16.77×** |
| gpt2 · EleutherAI/gpt-neox-20b | 1.161 | 7.287 | 5.987 | 6.28× |
| Qwen/Qwen2.5-0.5B | 1.175 | 4.563 | 3.992 | 3.88× |
| **theschoolofai/BrahmicTokenizer-131K** | 1.139 | **1.222** | 1.195 | **1.07×** |
| gemma-2-2b · Mistral-7B · sarvam-1 · nllb-200 · xlm-roberta | 1.000 | 1.000 | 1.000 | 1.00× |

**The control passes: BrahmicTokenizer is in the clean group at 1.07×**, so §3.3's fertility cost is
the cap and not a regex artifact. *(measured)*

Two findings fall out that are not ours to have expected. The floor **binds** — Falcon's Kannada
fertility is **92.9% pre-tokenizer**, so BPE recovers 7% of it. And Falcon carries a *second*,
distinct defect: its `Digits` splitter runs **after** `ByteLevel`, where byte `0xB2` → `'²'` and
`0xB3` → `'³'`, which are superscript **digits**. Kannada is U+0C80–U+0CFF, whose UTF-8 second byte
is exactly `0xB2`/`0xB3`. We derived from the Unicode layout that **Kannada alone** among the nine
Brahmic scripts should be hit, then checked: kn **19.571** against next-worst ml **8.433**. Prediction
confirmed. *(measured)*

A static scan of `tokenizer.json` reports `\p{M}=False` for GPT-2 and is worthless — `ByteLevel`'s
regex is compiled into Rust and never serialized. Only the behavioural test measures this.

---

## 4. Experimental design

**Setup, identical across every arm.** d_model 256, 4 layers, 4 heads, seq 256, batch 8, AdamW
lr 3e-4 cosine, 61,070-token vocabulary, 6.65M training tokens, **dense output head throughout**.
Corpora are byte-range **prefixes** of IndicCorpV2 (not random samples); validation is a held-out
slice, **49.5% en / 19.0% hi / 31.5% ml**.

| rubric requirement | what we ran |
|---|---|
| **dense-embedding control** | `dense_zn` — a dense table with the *same per-token z-norm* every codec arm gets. **The normalisation control the original comparison never had.** |
| **matched trainable parameters** | both allocated *and* trained parameter counts printed per arm; the permutation controls match **exactly**, with a printed audit |
| **≥3 seeds** | 3 seeds for the hardened arms (§8), 2 for the exploratory sweep; spread reported everywhere, never hidden |
| **≥2 scales** | 3,000 and 9,000 steps, six-point trajectory. **Complete for the `byte` baseline only** at submission time (§8); the rest was still running and is not claimed. |
| **independent negative controls** | the permutation null, run on three separate feature families, each with a printed audit proving capacity is preserved |
| **tuned baseline** | ⚠ **not yet demonstrated.** Every arm used `lr=3e-4` because the first script did. A sweep is queued with a committed threshold (§8.2); until it returns, §4.1's margins are **upper bounds**. |

**A note on how effect sizes are quoted below.** "18× the seed spread" is a *descriptive* ratio
against the observed range of 2–3 runs. **It is not a significance test**, and a range is not a
standard error. **Every trained effect size in this README rests on that descriptive ratio.** The
apparatus to replace it — per-token loss vectors and a paired block bootstrap over 204,544 validation
tokens, respecting the autocorrelation of contiguous text — is written and committed
(`design/e21_paired.py`), and the runs feeding it had not finished at submission (§8).

### 4.1 All arms, 3,000 steps

Two parameter columns, because they differ by up to 4.7× and only the second is a capacity (M1/M3).
**allocated** = `(D+1)·d_model`, what a parameter-efficiency claim would quote. **free** =
`rank(S)·d_model`, the dimension of the subspace the embedding table is actually confined to.

| arm | D | allocated | **free = rank·d** | rank(S) | val loss | note |
|---|---:|---:|---:|---:|---:|---|
| `dense_zn` | — | 15,633,920 | 15,559,424 | 60,779 | **5.8326** | dense control — **beats every codec** |
| `bigram_pos_shuf` | 16,384 | 4,194,560 | 1,447,168 | 5,653 | 6.0004 | ⚠ **not a codec** (§7.1) |
| `rand4` | 2,560 | 655,616 | 655,360 | 2,560 | 6.0328 | ⚠ **not a codec** — keys on token id |
| **`bigram_pos`** | 16,384 | 4,194,560 | 1,447,168 | 5,653 | **6.0388** | **best deployable codec** |
| `bigram_free` | 16,384 | 4,194,560 | 1,135,104 | 4,434 | 6.0723 | |
| `hashchar4` | 2,560 | 655,616 | 609,792 | 2,382 | 6.1220 | |
| `hashbyte4` | 2,560 | 655,616 | 443,648 | 1,733 | 6.1426 | |
| **`byte`** *(published)* | 8,192 | **2,097,408** | **443,648** | 1,733 | **6.1521** | baseline — **4.7× redundant** |
| `char2` | 6,144 | 1,573,120 | 306,944 | 1,199 | 6.1768 | |
| `byte_shuf` | 8,192 | 2,097,408 | 443,648 | **1,733** | 6.2858 | permutation control for `byte` |
| `char4` | 2,560 | 655,616 | 89,344 | 349 | 6.3232 | |
| `char4_shuf` | 2,560 | 655,616 | 89,344 | **349** | 6.5577 | permutation control for `char4` |

Read the allocated column and `byte` looks 3.2× larger than `hashbyte4`; read the free column and
they are **identical**, which is why their losses differ by 0.0095 nats — inside the seed spread (§5).
That single row is the whole argument for reporting rank.

**Adding position-bound byte-pair cells to the published codec improves val loss by 0.1133 nats
(15.5× the byte seed spread) at 2× the input parameters** — and its own control shows that gain is
capacity, not the conjunction (§6.2). That is the only codec improvement we claim.

---

## 5. rank(S) predicts every arm; allocated parameters predict nothing

| predictor | Spearman vs val loss |
|---|---:|
| **rank(S)** | **−1.0000** on the 7 arms it was built on |
| **rank(S)** | **−1.0000** on the 5 arms measured *afterwards* — a genuine held-out test |
| rank(S) | −0.9301 pooled over all 12 |
| **`D_allocated`** — the axis the parameter-efficiency argument lives on | **+0.0714** |

The pooled figure is *lower* than either subset. That is not a paradox: pooling introduces the
**exactly-rank-matched pairs**, where rank must predict a tie and does not get one. That is the
honest limit of the predictor and §6.1 is what it costs.

`loss ≈ 7.2221 − 0.3240·log₁₀(rank)`, **R² = 0.7801**, residual SD 0.0783 nats over 12 arms.

**A natural experiment that was already in the data.** `hashbyte4` and the published `byte` grid have
**identical rank, 1,733**, while differing 3.2× in D (2,560 vs 8,192), in construction (hash vs
byte×position grid), and in reachable cells (2,415 vs 1,855). Their losses differ by **0.0095 nats —
inside the seed spread.** Two arms matched on rank and on nothing else land on top of each other.
This was not designed; it fell out of a run made for another purpose.

**Rank also beat the predictor we published before it.** "Live rows" and rank agree on 6 of 7 arms
and disagree on exactly one: live rows says `hashbyte4` has 30% more capacity than `byte` (2,415 vs
1,855) and should win; rank says hashing a rank-1,733 space into 2,560 buckets **cannot add
information** and predicts a tie. Measured: a tie. Rank was right where live rows was wrong.

**Why arms have the ranks they have** — one rule, three operations:

| operation | effect on rank | predicted | measured |
|---|---|---|---|
| **factorise** a symbol into sub-symbols | product feature → sum of marginals; collapses | strict decrease | 1,199 → 444 → 349 ✓ |
| **hash** into `D′ > rank` | random column-merge; preserves | `= 1,733` | **1,733** ✓ |
| **randomise** per token | independent features; saturates | `= D` | **2,560** ✓ |

---

## 6. Three of our own claims, killed by our own controls

This is the part we most want read. Each refutation was produced by a control we designed, wrote a
threshold for, and committed **before** the run.

### 6.1 "Byte structure is not load-bearing" — **overturned**

We had published, from a random-index control: *"at this scale the byte/character structure is not
load-bearing for language-modelling loss. It is worse than random."* That control changed **capacity**
(2,560 trained rows vs 479, a 5.3× gap), so the conclusion did not follow.

The permutation null is the control that experiment needed. At **exactly** matched rank and load:

| pair | rank (both) | Δ = loss(permuted) − loss(structured) |
|---|---:|---:|
| `char4` vs `char4_shuf` | 349 | **+0.2345** (65× seed spread) |
| `byte` vs `byte_shuf` | 1,733 | **+0.1337** (18× seed spread) |

**Byte structure is worth 0.134 nats at fixed capacity.** `rand4` beat `byte` because it had 48% more
rank and a near-uniform row load — not because structure is useless. Our own published claim was wrong.

### 6.2 "rank(S) is the mechanism" — **downgraded, on its own pre-registered falsifier**

We wrote: *"if rank is the whole story, the permuted arms must land on their parents. If they do not,
rank is necessary but not sufficient and this entry is downgraded to a bound."* Rank was matched at
**0%**, not 5%, and the arms differ by 18–65 seed spreads. **The falsifier fired.** rank survives as a
capacity **ceiling** and as an ordering statistic; it does not survive as *the* mechanism.

Corrected decomposition, both measured:

| effect | comparison | Δ |
|---|---|---:|
| **capacity** | `rand4` vs `byte` (rank 2,560 vs 1,733) | −0.1193 nats |
| **structure** | `byte` vs `byte_shuf` (rank matched exactly) | −0.1337 nats |

Comparable. Neither dominates.

### 6.3 "Conjunctions repair the linearity ceiling" — **refuted twice, the second time by a sign reversal**

M2 says no conjunction is representable at any budget, so we added adjacent-byte bigram cells. It
improved on the published codec — and then its own control killed the explanation:

| family | position | Δ_perm | reading |
|---|---|---:|---|
| **byte** | bound | **+0.1337** (18.3× spread) | structure is load-bearing |
| bigram | **free** | +0.0008 (0.1× spread) | structure is **inert** |
| **bigram** | **bound** | **−0.0383** (4.3× spread) | structure is **harmful** |

Two hypotheses were tested and both died. *Hash quality*: dead — the vocabulary contains only **3,596
distinct adjacent byte pairs** and the hash touches 2,806 buckets where an ideal random hash touches
2,911. *Position binding*: dead, in the **opposite** direction from the prediction — permuting the
position-bound bigram assignment **improves** loss.

**We proposed a mechanism, measured it, and it failed.** The proposal was that the bigram family is
**redundant given** the byte family — adding no information but adding ties. Redundancy is measurable
without training, as `1 − ΔR_F/R_F`. Measured, the bigram family contributes **3,920 of its own 4,220
directions** on top of byte: redundancy **0.0711**, i.e. nearly non-redundant. Every skip-gram and
whole-token variant scores 0.0002–0.243. **The mechanism is refuted and the phenomenon is
unexplained.** We report it that way rather than keeping a story the data does not support.

**And the control leaks.** E19's permutation shuffles only the **bigram block** of each row, keeping
the byte block fixed — so unlike E14's, it is **not** a whole-row permutation and does **not**
preserve rank. Measured: 5,650 → **5,902** (+4.46%). Using §5's fitted slope, that gain predicts
−0.0061 nats of the observed −0.0383, so **≈16% of E19's effect is a capacity leak and ≈84% is not**.
The job's audit reported live rows and touches, which *are* preserved, and never reported rank — which
is how it stayed invisible. Recorded as `CORRECTION C4`.

**So the exact-rank claim belongs to E14 alone.** `byte` vs `byte_shuf` is a true whole-row
permutation with rank measured identical at 1,733, and that is the row this README leans on. Per our
pre-commitment, we stop proposing conjunction repairs.

### 6.4 A verifier that found a real error in our own numbers

`design/verify.py` recomputes 26 headline quantities from committed data and exits non-zero on
disagreement. **On its first execution it disagreed with our own published rank** for `bigram_pos`:
5,653 recomputed against 5,668 logged. Cause: two conventions for building `S` — "accumulate" (a cell
occurring twice in a token counts twice) versus "dedup" (binary indicator). `KronEmb.forward`
computes `W[idx].sum(1)`, so **accumulate is the convention that matches the trained model.**

Corrected: `bigram_free` 4,459 → **4,434**; `bigram_pos` 5,668 → **5,653**. Byte/char arms are
unaffected (a cell is `symbol×P + position`, unique per slot). **No conclusion moves** — R² 0.7798 →
0.7801, pooled Spearman unchanged, and E19's central claim is untouched because a permutation
permutes *rows* of `S` and cannot change rank under either convention. Recorded as `CORRECTION C1`,
appended rather than edited over. A verifier that only confirms what you already believe is doing
nothing; this one paid for itself on run one.

---

## 7. What this does not show

### 7.1 The leading arms are not codecs

`bigram_pos_shuf` (6.0004) and `rand4` (6.0328) are the two best non-dense arms, and **neither is a
deployable codec.** `rand4` keys on token **id**; `bigram_pos_shuf`'s assignment is a permutation
table. Neither mapping is computable from a token's bytes, so neither has OOV, spelling, or typo
behaviour — the properties the technique exists for. **They are capacity controls.** Reporting either
as a codec win would repeat exactly the error that killed our first headline (§6.1).

### 7.2 Four wrong token representations before one was right

Recorded because the corrections were the work:

1. mixed decoded/byte-level per token — inconsistent across ids;
2. `tk.decode([i])` throughout — **lossy**: 658 BrahmicTokenizer ids decode to `'�'` and 396 to
   `' �'`, because they are partial UTF-8 fragments. They are *distinct byte strings* a byte codec
   separates perfectly. **This would have shipped a fabricated 1.1% irreducible-collision claim
   against the instructor's tokenizer;**
3. counting those 1,496 as irreducible — on raw bytes the artifact has **0**;
4. **raw bytes via the ByteLevel inverse alphabet** — correct, and used for every anchor number here.

Separately, an invalid earlier run of the cap experiment used a `\p{L}+` pre-tokenizer, which shreds
Indic words *before BPE runs* and pinned every curve at a regex floor. It is preserved as
`e13_BROKEN_gpt2regex.json` and superseded, not deleted.

### 7.3 Scope and known limits

| limit | consequence |
|---|---|
| **one corpus family**, IndicCorpV2 **prefixes** (not random samples) | any domain skew in each file's head is inherited |
| 6.65M training tokens, ~0.9 epochs | **arXiv:2605.09751 finds randomly-recoded codes reach parity at 17B tokens — 375× our budget.** The best-calibrated prior that our effects are undertrained-regime artifacts. §8 shows tripling the training moves the baseline **0.504 nats — 4× our largest structural effect** — and the curve has not flattened. **This objection is live and unresolved.** |
| **dense output head on every arm** | our own §9 argues the head is where the parameters are; we did **not** train a tied head. E17's head results are static bounds only. |
| val split n=1, codec draw n=1 | the two variance components a "random beats structured" claim most needs are unsampled by seeds; we address the first with a block bootstrap over val tokens |
| unigram floor | a **diagnostic**, not a proven bound on LM loss. The **collision cost is** a hard bound. |
| redundancy mechanism (§6.3) | inferred from three points; the sweep that would measure it is not run |
| load Gini | ignores AdamW's per-parameter adaptivity, which plausibly compresses load-skew effects — untested |

### 7.4 Where we do not claim credit

| not ours | who |
|---|---|
| `rand4` is the **hashing trick** | Weinberger et al. 2009 |
| `hashchar4` is **CANINE**'s hashed character scheme | Clark et al., TACL 2022 |
| "matches the full table at a fraction of the parameters" | **Hash Embeddings** and **Bloom Embeddings**, both 2017 |
| Kronecker/TT embedding factorisation | **Tensorized Embedding Layers** (2019), **KroneckerBERT** (2021) — five years prior, uncited by arXiv:2605.29459 and by us until review |
| K=4 hashes | spaCy's default |
| fixed random codes reach parity at scale | arXiv:2605.09751 |
| the head/tail lookup-vs-composition crossover at 16–32 occurrences | Schick & Schütze 2019; Bhatia et al. 2016 |
| production scale, any benchmark score, a real V5 vocabulary | nobody here — we have 11 CPU cores and one M3 Pro |

---

## 8. Scale: an in-flight test, reported as in-flight

The strongest objection to every trained number above is that 3,000 steps is an undertrained regime,
and arXiv:2605.09751 already found that randomly-recoded fixed codes reach **parity** with structured
ones at 17B tokens — **375× our budget**. So this is the objection most worth testing, and we
pre-registered it and started it.

**It had not finished when this was submitted. We report what completed and claim nothing about the
rest.**

### 8.1 What completed: the published codec at 3× the training

| arm | steps | seed 0 | seed 1 | spread |
|---|---:|---:|---:|---:|
| `byte` | 9,000 | 5.6485 | 5.6477 | **0.0008** |
| `byte` | 3,000 *(published table)* | — | — | mean 6.1521, spread 0.0073 |

Six-point trajectory, seed 0 → seed 1:

| step | 1,000 | 2,000 | 3,000 | 4,500 | 6,000 | 9,000 |
|---|---:|---:|---:|---:|---:|---:|
| val loss | 6.4451 / 6.4571 | 6.1546 / 6.1559 | 5.9831 / 5.9954 | 5.8271 / 5.8317 | 5.7306 / 5.7322 | **5.6485 / 5.6477** |

**The one thing this already establishes, and it cuts against us.** Tripling the training improves the
baseline by **0.504 nats** — more than **4×** the largest structural effect we report (0.1337). The
loss curve has not flattened by 9,000 steps. **We are deep in the undertrained regime that the parity
result describes**, and no result in this README should be read as a statement about converged models.
*(measured)*

### 8.2 What did not complete, and what it was pre-registered to test

The permuted arms (`byte_shuf`, `bigram_pos`, `bigram_pos_shuf`), the reference arms, the LR sweep and
E25's exact-rank control were still running. So:

- **`Δ_perm` at 9,000 steps is not reported and not claimed.** Whether byte structure's +0.1337 nats
  shrinks, holds, or grows with training is **open**.
- **The baseline is not yet shown to be tuned.** Every arm used `lr=3e-4` because that is what the
  first script used. The sweep (1e-4 / 3e-4 / 6e-4 / 1e-3 on the byte codec *and* the dense control)
  is queued with a committed threshold: **if any rate beats 3e-4 by more than the byte seed spread
  (0.0073 nats), the baseline was handicapped and every "arm X beats byte" figure here is inflated.**
  Until it returns, treat §4.1's margins as **upper bounds**.

**These experiments were run; some are still pending.** Each was pre-registered in its job docstring
— hypotheses and refutation conditions written down before the run, not after — and the two whose
scripts are included here are [`design/018_e21f_lrsweep.py`](design/018_e21f_lrsweep.py) and
[`design/019_e25_fullshuf.py`](design/019_e25_fullshuf.py), where those pre-registrations can be read
in full. The training jobs themselves ran on local hardware and their runner scripts and raw result
JSONs are not part of this repository; what is committed is the derived data in
[`design/data/`](design/data/), which [`design/verify.py`](design/verify.py) recomputes every static
claim from. Results from the pending runs will be appended to `EXPERIMENT_LOG.md` whether or not they
are favourable. This section is the honest state of the work, not a placeholder for numbers we expect
to like.

---

## 9. Bearing on Problem #5, scoped and not claimed

A codec output head computes `logits_v = ⟨φ(v), W h⟩`, so head parameters are `d·D`, **independent of
V**. Three exact quantities, no trained model, on a real 6.58M-type distribution sampled at three sizes:

| V | rank(S) | ids sharing a code with another | *excess* ids over one-per-group | collision cost (hard bound) |
|---:|---:|---:|---:|---:|
| 50,257 | 1,472 | 551 (1.10%) | 336 (0.67%) | 0.00014 bpb |
| 131,072 | 1,998 | 3,623 (2.76%) | 2,303 (1.76%) | 0.00039 bpb |
| **1,000,000** | 3,559 | **93,460 (9.35%)** in 25,584 groups | 67,876 (6.79%) | **0.00209 bpb** |

*(Two distinct quantities, reported separately because they answer different questions: how many
tokens are ambiguous, and how many ids would have to be given up to remove the ambiguity.)*

**V grows 20×, rank grows 2.4×, collisions grow 202×.** So *"a vocab of 1M without any issues"* is
**false for decodability** — 9.35% of the vocabulary is provably inseparable, receiving identical
logits in every context — while **nearly true for loss**, because collisions land on rare tokens.
The collision cost is a **hard bound**: identical rows of `S` force identical probabilities, so
`Σ_G Σ_{v∈G} p_v log(p_v / (p_G/|G|))` is exact, with no optimisation and no model. Both halves
belong in the claim.

The fix is measured. A **phoneme basis** — `codepoint & 0x7F`, which Unicode makes script-invariant
across all nine Brahmic scripts (**76 of 124 populated offsets carry an identical phoneme label in
every script**; KA, MA, VOWEL SIGN AA and VIRAMA are exact in all nine) — reduces collisions at
V=1,000,000 from **67,876 to 3**, beating a 64-byte window's 201 at **25% smaller D**:

| codec | head params @ V=1M, d=768 | vs dense | inseparable tokens |
|---|---:|---:|---:|
| dense | 768,000,000 | 1.0× | 0 |
| byte32 | 2,931,456 | 262× | 67,876 |
| byte64 | 4,333,056 | 177× | 201 |
| **phon32** | **3,242,496** | **237×** | **3** |

*This is a static analysis.* We did not train a tied head, and we do not claim a quality result for one.

---

## 10. Reproducibility

```bash
git clone <this repo> && cd S7
python -m venv .venv && .venv/bin/pip install numpy scipy tokenizers pyarrow
.venv/bin/python design/verify.py        # exits 0 only if all 26 checks pass
```

`verify.py` **recomputes** each quantity from `design/data/` rather than reading it back from a
results file, and **exits non-zero** on any disagreement, so it cannot pass vacuously. It found a
real error in our own numbers on its first run (§6.4).

```
A. corpus and vocabulary ........ 6 checks   train tokens, types, cap honoured, 0 tokens over 32 bytes
B. rank(S) ...................... 10 checks  byte/char2/char4/bigram_pos rank and reachable cells
C. row-load uniformity .......... 3 checks   Gini 0.8887, effective rows 200.4, effective/allocated 2.4%
D. Unicode phoneme alignment .... 3 checks   124 populated offsets, 76 identical across all 9 scripts
E. rank-matched pairs ........... 3 checks   the model-free evidence of §6.1
F. 1/sqrt(L) is inert ........... 1 check    measured relative effect 4.56e-05
                                  ---------
                                  26 checks, exit 0
```

**Artifact inventory:** 37 files, SHA-256 (first 16 hex) in `design/MANIFEST.json`. Key entries:

| file | bytes | sha256[:16] |
|---|---:|---|
| `design/verify.py` | 6,352 | `6b45f8b380ccf36e` |
| `design/data/codec.npz` | 1,200,852 | `c610e19f6a1a0d0e` |
| `design/data/train_freq.npy` | 488,688 | `4b881a24d8e194e9` |
| `design/data/meta.json` | 247 | `1d432d62b8d50713` |
| `design/e22_stats.py` | 5,288 | `1e28feb7dd8bdd1e` |
| `design/e17_rank.py` | 6,446 | `252a60203f888cfc` |
| `design/e21_paired.py` | 4,243 | `958c369b74087cdb` |

Determinism scope: `verify.py` and all `design/*.py` analyses are exactly reproducible (fixed seeds,
no GPU). **Trained losses are not bitwise reproducible** — they were produced on Apple MPS, where
reduction order is not guaranteed; seed-to-seed spread is reported for every arm precisely because
of this.

---

## 11. What we would do next, and what would reverse this

| next | why |
|---|---|
| **redundancy sweep** — byte-pairs → skip-grams → whole-token hashes | the one measurement that would turn §6.3's mechanism from a three-point inference into a result |
| **train a tied codec head** | §9 is static; our own argument says the head is where the parameters are, and we have not tested it |
| **frequency-gated hybrid** (lookup for the head, codec for the tail) | capacity is currently allocated by **spelling length**, which correlates with frequency at only **ρ_s = +0.13**; 53% of live types receive **1.6%** of slot-uses |

**Reversal conditions, stated concretely.**

- If a **deployable** codec beats the dense control at matched trained parameters, the framing here
  is wrong and the structured table is worth its cost. Currently `dense_zn` leads by 0.20–0.49 nats.
- If `Δ_perm` for `byte` goes to zero with more training (§8), the structure claim is a learning-speed
  artifact and arXiv:2605.09751 explains us entirely.
- If a wider window ever changes reachable cells or rank on a real tokenizer, §3.2 is wrong — but it
  would first require a vocabulary containing tokens longer than 32 bytes, which this one does not.

---

## Appendix — every number's source

**measured** here means we computed it from data in this repo; **stated** means a paper or the course
asserts it; **stylized** means an illustrative formula with no empirical backing. No number in this
README is stylized. Full provenance, pre-registrations, refutations and superseded entries are in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md); durable decisions in [`ARCHITECTURE.md`](ARCHITECTURE.md);
prior-work positioning in [`SOTA-MAP.md`](SOTA-MAP.md).
