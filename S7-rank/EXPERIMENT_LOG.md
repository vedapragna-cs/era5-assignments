# S7 experiment log

Every number is labelled **measured** (we computed it), **stated** (course or paper asserts it), or
**stylized** (a widget's illustrative formula, no empirical backing). Historical entries are never
rewritten; corrections are appended and name the entry they supersede.

**What this repository contains.** The training jobs referenced below (`002_…` through `019_…`) were
run on local hardware — Apple M3 / M3 Pro, MPS. Their runner scripts and raw `results/*.json` outputs
are not committed here; the two pending jobs whose pre-registrations matter most are the exception and
ship as `design/018_e21f_lrsweep.py` and `design/019_e25_fullshuf.py`. What *is* committed is the
derived data in `design/data/`, and `design/verify.py` recomputes all 31 static claims from it and
exits non-zero on any disagreement — so every claim that needs no trained model can be checked from a
clean clone. Claims that rest on the training runs are labelled as such, and **some of those runs are
still pending**; §8 of the README states which, and what each was pre-registered to test.

---

## E1 — Does BrahmicTokenizer-131K actually honour its 32-byte ceiling?

**Date** 2026-08-21 · **Artifacts** `theschoolofai/BrahmicTokenizer-131K` (HF, Apache 2.0)

**Why.** arXiv:2605.29379 §structural properties claims a *"maximum token byte length of 32 UTF-8
bytes"*, said to be *"relevant for byte-pooled embedding architectures"* — i.e. the tokenizer was
constrained to fit the Kronecker codec's `d_p = 32` window. The transcript says the same thing in his
own words. Both are **stated**. Nobody has checked it.

**H1a** Some tokens exceed 32 bytes (the constraint is aspirational). → **REFUTED**
**H1b** The byte ceiling converts to a *character* ceiling that differs by script. → **CONFIRMED**

**Method.** Load all 131,072 vocabulary entries, invert the byte-level BPE unicode mapping to recover
raw bytes, measure length distribution, classify by Unicode block of the decoded text.

**Measured.**
- max token length **exactly 32 bytes**, **0 tokens above**. All 131,072 mapped, none unmappable.
- Only 11 tokens reach 32 bytes; only ~340 reach 24 or more.
- Character ceiling by script: Latin **~26 chars** (max 26 bytes); every Indic script **10 chars**
  (max 31 bytes = 1 leading-space byte + 10 × 3-byte characters).
- Longest Devanagari tokens, all exactly 10 characters, and there are only **6** of 4,143:
  ` विद्यार्थी` ` राष्ट्रपति` ` महाराष्ट्र` ` महत्वपूर्ण` ` खिलाड़ियों` ` अधिकारियों`
- Devanagari tokens by character count: 1:264 2:908 3:1222 4:853 5:453 6:225 7:123 8:63 9:26 10:6
- Vocabulary composition: Latin **108,200 (82.55%)**, nine Indic scripts **15,535 (11.85%)**.
- His own in-class collision example अंतर्राष्ट्रीयकरण is 17 chars / **51 bytes** → 5 tokens; its
  near-twin अंतर्राष्ट्रीयता is 16 chars / 48 bytes → 5 tokens. Neither can ever be one token.

**Reading.** The stated property holds. The cost is a **2.6× asymmetry in maximum token length**
between Latin and Indic, imposed by the embedding and paid at the tokenizer.

---

## E2 — Does the 32-byte window impose a fertility floor above 1.0?

**Date** 2026-08-21 · **Corpus** FLORES-200 devtest, 12 languages, 14.9k–25.9k words each

**Why.** Lesson §4 states fertility *"decays toward a floor of one token per word, so a language's
reducible headroom is what extra vocabulary can actually buy"* (**stated**), and widget 4 computes
`fert = 1 + (f₀−1)·(32768/V)^α` (**stylized**), which tends to exactly 1 as V grows. If no token may
exceed 32 bytes, that floor cannot be 1 for scripts whose words routinely exceed 32 bytes.

**H2a** The floor exceeds 1.0 for Indic and equals 1.0 for English. → **CONFIRMED**
**H2b** The window is a materially binding constraint on fertility today. → **REFUTED**

**Method.** Segment into words (whitespace split, strip leading/trailing P/S categories, preserve
combining marks). `floor_cap(w)` greedily packs characters into ≤`cap`-byte chunks, first chunk
reserving 1 byte for the leading space — the minimum tokens for that word at **any** vocabulary size.
`actual` = `len(tok.encode(" "+w))`.

**Measured.**

| lang | words | ch/w | B/w | %words >31B | floor32 | floor48 | floor64 | actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eng_Latn | 21887 | 4.95 | 4.95 | 0.00 | 1.0000 | 1.0000 | 1.0000 | 1.114 |
| pan_Guru | 25891 | 4.08 | 12.12 | 0.40 | 1.0040 | 1.0000 | 1.0000 | 2.493 |
| hin_Deva | 25591 | 4.06 | 12.07 | 1.18 | 1.0118 | 1.0004 | 1.0000 | 1.532 |
| guj_Gujr | 20853 | 4.97 | 14.78 | 2.83 | 1.0284 | 1.0012 | 1.0000 | 2.147 |
| asm_Beng | 19198 | 5.46 | 16.24 | 4.40 | 1.0440 | 1.0016 | 1.0001 | 2.622 |
| ben_Beng | 19445 | 5.54 | 16.51 | 5.17 | 1.0518 | 1.0025 | 1.0001 | 2.180 |
| ory_Orya | 19498 | 5.83 | 17.34 | 5.60 | 1.0562 | 1.0029 | 1.0002 | 4.097 |
| mar_Deva | 19012 | 5.90 | 17.60 | 7.22 | 1.0724 | 1.0043 | 1.0001 | 2.409 |
| tel_Telu | 16906 | 6.70 | 19.78 | 11.61 | 1.1162 | 1.0101 | 1.0001 | 2.844 |
| kan_Knda | 16059 | 7.48 | 22.26 | 19.86 | 1.2000 | 1.0265 | 1.0009 | 3.064 |
| tam_Taml | 16730 | 8.08 | 24.07 | 22.82 | 1.2322 | 1.0402 | 1.0023 | 2.944 |
| mal_Mlym | 14903 | 8.89 | 26.50 | **31.48** | **1.3295** | 1.0798 | 1.0108 | 3.298 |

**Pipeline control.** Our measured Odia fertility 4.097 against his published 4.13 (**stated**,
arXiv:2605.29379) — independent path, agrees. The measurement is trusted to ~1%.

**Reading — and the part that refutes our own hypothesis.**
- H2a confirmed: English floor is exactly 1.0000 (no English word in FLORES exceeds 31 bytes);
  every Indic floor is above it, Malayalam **1.3295**. The lesson's and widget 4's universal floor of
  1.0 is **false for Indic**. Corrected law: `fert_L(V) = floor_L + (f₀_L − floor_L)·(32768/V)^α`.
- H2b **refuted**: `actual` sits far above `floor32` everywhere. Decomposing the headroom —
  Malayalam 1.968 buyable with vocabulary vs 0.330 forbidden by the window (6:1); Odia 3.040 vs
  0.056 (54:1). **Vocabulary, not the window, is the binding constraint today.**
- Widening is not free: `d_p=48` → 99,483,648 projection params (+50%, still 10.7× under dense);
  `d_p=64` → 132,644,864 (+100%, 8.0× under dense). At `d_p=32` it is 66,322,432 (16.0× under dense).

**Consequence for the plan.** The window bounds the *return on all future Indic vocabulary
investment* rather than charging a large bill today. Since vocabulary is what binds, E3 moves to the
allocation question, carrying `floor_L` in as a constraint.

**Caveats.** FLORES devtest is ~1000 translated sentences per language; translationese may bias
compound length. Word segmentation is whitespace-based, which is defensible for these scripts but not
for the unsegmented ones. Neither is corrected for here.

---

## E4 — Vocabulary allocation across scripts (PLANNED, not yet run)

*Renumbered from E3 when E3 below was executed first. No measurements existed in the E3 placeholder.*

Hypotheses and refutation criteria are recorded **before** execution, per the working agreement.
Pre-committed falsifiers:
- optimal allocation within a few % of actual in total-token terms → the retrofit is near-optimal,
  report and stop;
- marginal-value curves flat enough that allocation barely moves the objective → same;
- our retrained BPE fails to reproduce his published per-language fertility → pipeline is wrong and
  no allocation conclusion is safe until it is fixed.

---

## E3 — What does a byte-decoding output head cost in Indic?

**Date** 2026-08-21 · **Corpus** IndicCorpV2 prefixes, 40k lines/language, 0.43M–1.15M words each
· **Vocabulary** BrahmicTokenizer-131K

**Why.** The instructor reports a working "K2" reversible head: output head 38,597,376 → 2,439,554
params (15.8x, 93.7% removed), total 81.27M → 46.69M, val bpb 1.7135 → 1.7089, top-1 22.29% → 22.20%
(all **stated**). `50257 x 768 = 38,597,376` confirms GPT-2 vocab at `d_model=768` (**measured**
arithmetic). The evaluation is English. His own paper lists *"Multilingual evaluation"* as future
work. A head that emits bytes does work proportional to bytes, and Indic bytes behave differently.

**H3a** A byte-decoding head amplifies the existing Indic:English cost ratio. → **CONFIRMED**
**H3b** The amplification is largest for the worst-served languages. → **REFUTED — it is inverted.**

**Method.** Corpus-frequency-weighted. `denseR` = ratio of tokens/word to English (decisions a
one-per-token head makes); `byteR` = ratio of bytes/word to English (byte slots a byte head fills);
`amplification = byteR / denseR`, which reduces exactly to the bytes-per-token ratio.

**Measured.**

| lang | words | B/word | tok/word | B/token | %multibyte chars | denseR | byteR | amplification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 771,145 | 4.95 | 1.194 | 4.15 | 0.1 | 1.00 | 1.00 | 1.00 |
| hi | 1,148,497 | 11.94 | 1.497 | 7.97 | 97.9 | 1.25 | 2.41 | **1.92** |
| ml | 572,668 | 27.20 | 3.415 | 7.97 | 98.7 | 2.86 | 5.50 | **1.92** |
| ta | 667,057 | 23.58 | 2.986 | 7.90 | 97.9 | 2.50 | 4.76 | **1.90** |
| bn | 914,901 | 16.01 | 2.197 | 7.29 | 98.9 | 1.84 | 3.23 | 1.76 |
| kn | 519,694 | 21.22 | 2.948 | 7.20 | 98.6 | 2.47 | 4.29 | 1.74 |
| mr | 827,098 | 17.24 | 2.400 | 7.19 | 99.0 | 2.01 | 3.48 | 1.73 |
| te | 692,081 | 20.46 | 2.913 | 7.02 | 98.6 | 2.44 | 4.13 | 1.69 |
| gu | 827,452 | 14.59 | 2.131 | 6.85 | 98.4 | 1.78 | 2.95 | 1.65 |
| as | 804,288 | 16.45 | 2.632 | 6.25 | 99.1 | 2.20 | 3.32 | 1.51 |
| pa | 1,052,254 | 11.79 | 2.485 | 4.74 | 98.2 | 2.08 | 2.38 | 1.14 |
| or | 430,780 | 16.47 | 4.026 | 4.09 | 99.0 | 3.37 | 3.33 | **0.99** |

Exact-token decode probability under **independent** per-byte prediction, `p^(B/token)`:

| p | en | hi | ml | ta | or |
|---|---:|---:|---:|---:|---:|
| 0.99 | 0.959 | 0.923 | 0.923 | 0.924 | 0.960 |
| 0.95 | 0.808 | 0.664 | 0.665 | 0.667 | 0.811 |
| 0.90 | 0.646 | 0.432 | 0.432 | 0.435 | 0.650 |

**Reading.**
- H3a confirmed: for Hindi, Malayalam and Tamil a byte head does **1.9x** more work per unit of
  meaning than a dense head does, *on top of* the fertility disadvantage already there.
- **H3b refuted, and the inversion is the interesting result.** Amplification is not worst for the
  worst-served languages — it tracks bytes-per-token, so it is worst for the languages the tokenizer
  serves *best*. **Hindi has the best Indic fertility (1.497 tok/word, denseR 1.25) and the joint-worst
  amplification (1.92).** Odia has the worst fertility (4.026) and amplification **0.99 — none at
  all**, because at 4.09 bytes/token the tokenizer has already shredded it to near-byte fragments,
  so a byte head costs it nothing extra. A byte-structured head therefore claws back precisely the
  gains a good Indic tokenizer just bought.
- 97.9–99.1% of Indic characters are multi-byte versus 0.1% for English, so nearly every Indic
  character is a 3-byte sequence in which one wrong byte corrupts the character and can produce
  invalid UTF-8 (cf. arXiv:2511.05578).

**Caveats — three, and they bound the claim.**
1. This measures the *structure* of byte decoding, **not his K2 head**, whose code is not released.
   Nothing here is a claim about his implementation.
2. The `p^(B/token)` table assumes **independent** per-byte prediction, i.e. a parallel head. An
   autoregressive byte decoder has dependent predictions and does substantially better; this is
   therefore an upper bound on the harm, applying to the parallel case only.
3. Corpora are byte-range **prefixes** of IndicCorpV2, not random samples (ARCHITECTURE D4).

---

## E5 — A character-basis Kronecker codec that is cheaper *and* wider

**Date** 2026-08-21 · **Code** `assignment/charkron.py` · **Artifacts** BrahmicTokenizer-131K

**Why.** E3 showed the Indic penalty of a byte-decoding head reduces exactly to the bytes-per-token
ratio. So the fix is not a better head, it is a better basis. UTF-8 is a variable-length encoding
built for ASCII backward compatibility; the codec inherited that asymmetry into its axes, charging
3 bytes per Indic character and 1 per Latin character. A character basis removes it — but the naive
version is catastrophic, since a one-hot over ~150,000 Unicode codepoints gives D ≈ 4.8M.

**The construction.** Factor the codepoint as `(block, offset)` with 128-wide blocks — an alignment
Unicode already provides, since Devanagari is U+0900–097F, Telugu U+0C00–0C7F and so on. Then

```
kappa(t) = Σ_p [ block(c_p) ⊗ e_p ]  +  Σ_p [ offset(c_p) ⊗ e_p ]
```

A **sum of two factored terms, not a three-way product**. Position indexes both, so `(block_p,
offset_p)` is recoverable and the codepoint is `block_p·128 + offset_p`. Two block ids are reserved
so byte-level BPE fragments that are not valid UTF-8 alone round-trip as raw bytes **at zero extra
dimension cost**, using 2 of the 24 unused block slots.

**H5a** A character basis costs more than the byte grid. → **REFUTED — it costs 25% less.**
**H5b** It removes the script-dependent fertility floor of E2. → **CONFIRMED**

**Measured.**

| | D | projection @ d=8096 | vs byte grid | window |
|---|---:|---:|---:|---|
| byte grid `256 × 32` | 8192 | 66,322,432 | — | 32 bytes = **10 Indic chars** |
| **character basis** `64×32 + 128×32` | **6144** | **49,741,824** | **−25.0%** | **32 chars, every script** |

- **Round-trip: 131,072 / 131,072 exact. 0 mismatches, 0 tokens over 32 characters.** Includes all
  1,306 non-UTF-8 byte fragments via the reserved blocks. 40 distinct blocks are used; the axis is
  padded to 64.
- **The floor of E2 collapses.** Minimum tokens/word under a 32-character window, corpus-weighted:

| lang | floor (byte, E2) | floor (char, E5) | %words >32 chars |
|---|---:|---:|---:|
| mal_Mlym | **1.3295** | **1.0005** | 0.049 |
| tam_Taml | 1.2322 | 1.0003 | 0.021 |
| kan_Knda | 1.2000 | 1.0001 | 0.006 |
| tel_Telu | 1.1162 | 1.0001 | 0.005 |
| hin_Deva | 1.0118 | 1.0000 | 0.001 |
| eng_Latn | 1.0000 | 1.0000 | 0.001 |

- **The head amplification of E3 inverts.** Per-token symbols a reversible head must emit:

| lang | byte amplification | char amplification | improvement |
|---|---:|---:|---:|
| hi | 1.92 | **0.65** | 2.95× |
| ml | 1.92 | **0.65** | 2.97× |
| ta | 1.90 | 0.65 | 2.95× |
| or | 0.99 | 0.33 | 2.97× |

The improvement is uniformly 2.95–2.98× across every Indic script, which is simply the ~3 bytes per
character that UTF-8 charges them. On a character basis every Indic language costs **less** per token
than English, because its tokens are shorter in characters (2.31–2.69 vs English 4.14).

**Robustness.** The 64-wide block axis holds because only 40 blocks appear in this vocabulary. A more
script-diverse vocabulary needing up to 128 blocks gives `128×32 + 128×32 = 8192` — **exactly the
byte grid**. The construction therefore ties in the worst case and never loses.

**Loop closed.** BrahmicTokenizer-131K carries a 32-byte maximum token length *because the codec
needs it* (arXiv:2605.29379, and the transcript in his own words). A character basis removes that
requirement, freeing the tokenizer to build Indic tokens up to 32 characters where it is currently
capped at 10 (E1). The fertility that would buy is unmeasured and is the natural next experiment.

**What this does NOT show — four things.**
1. **No language model was trained.** This is a codec-level result: parameter count, coverage,
   exact round-trip, and the fertility/head costs derived from them. Whether it *trains* as well is
   untested and is the obvious next step.
2. The head numbers assume a reversible head's cost scales with emitted symbols. That is structural.
   **It is not a claim about his K2 implementation**, whose code is unreleased.
3. **Prior art boundary.** The instructor's published index describes a cohort submission replacing
   UTF-8 byte slots with codepoint slots. The idea of a codepoint basis is therefore not ours. What
   is ours is the two-factor sum that makes D *smaller* than the byte grid rather than ~600× larger,
   the reserved-block byte fallback, and the verified round-trip. We have not read that repository.
4. Corpora remain byte-range prefixes (ARCHITECTURE D4); the reversal check is still outstanding.

---

## E6 — Does the character basis actually train? (hypotheses recorded before results)

**Date** 2026-08-21 · **Job** `002_e6_codec_arms.py` · **Device** Apple M3, MPS,
torch 2.8.0 · **Status at time of writing: RUNNING, no results seen.**

**Why.** E5 is arithmetic and round-trip: 25% fewer codec parameters, a 3.2× deeper window, an exact
131,072/131,072 reconstruction. None of that shows the basis *learns* as well. This is the test that
decides whether E5 is a result or a favourable accounting identity.

**Setup.** Only the input path differs; body and output head are identical across arms.
d_model 256, 4 layers, 4 heads, seq 256, batch 8, 3000 steps, AdamW lr 3e-4, warmup 150, cosine,
grad clip 1.0, seeds {0,1}. Corpus is a shuffled 1:1:1 chunk-interleave of English, Hindi and
Malayalam from IndicCorpV2 prefixes — 6.65M train tokens, 204,800 val. Vocabulary restricted to the
61,070 tokens observed (46.6% of 131,072); unused rows receive no gradient, so the restriction is
exact rather than an approximation. Metric is val loss and bits-per-byte, reported per language.

- **dense** — `nn.Embedding(V, d)`, the control arm the lesson names for V5.
- **byte** — `κ = (1/√L) Σ_p byte(b_p) ⊗ e_p`, `d_p = 32`, per arXiv:2605.29459 Eq. 1.
- **char** — `κ = (1/√L)[Σ_p blk(c_p) ⊗ e_p + Σ_p off(c_p) ⊗ e_p]`, ours, per E5.

**Hypotheses, with refutation criteria fixed in advance.**

| | hypothesis | refuted if |
|---|---|---|
| **H6a** | char matches or beats byte on overall val loss, at fewer input parameters | char's val loss exceeds byte's by more than the seed spread |
| **H6b** | char's advantage over byte concentrates in Indic, not English | the char−byte bpb gap is not larger for `ml`/`hi` than for `en` |
| **H6c** | both Kronecker arms trail dense on overall loss, since dense has ~15× more input parameters | either Kronecker arm beats dense — which would be a stronger result than we are claiming |

**Committed in advance:** if H6a is refuted, E5 is an accounting result only and the README must say
so in those words. A 25% parameter saving that costs quality is not a contribution, and we will not
reframe it as one.

**Known limits of this run.** Two seeds, not three. One scale, so no trend. ~0.9 epochs, so all arms
are far from converged and this measures early-training behaviour rather than final quality. The
output head is dense and identical in every arm, so nothing here speaks to K2-style byte decoding.

### E6 methods note — profiling the M3 Pro before trusting any timing

**Date** 2026-08-21 · **Job** `004_profile.py` · Apple M3 Pro, 18-core GPU,
6.4 TFLOPS fp32 spec, 150 GB/s unified bandwidth, torch 2.8.0 MPS

A first attempt at E6 ran at **≥1.3 s/step**, about 3% of the machine. Rather than guess, we
profiled. Three of the four things we were about to "fix" turned out not to be the problem.

| measured | result | consequence |
|---|---|---|
| raw GEMM | fp32 **5.26 TFLOPS** (82% of spec); fp16 5.87 (**+11% only**) | fp16 not worth a numerical difference between arms |
| output head 2048×256→61,070 fwd+bwd | **116.4 ms** fp32 / 79.1 ms fp16, 500 MB logits | this *is* the step; ~1.65 TFLOPS, so `log_softmax` is bandwidth-bound, not the matmul |
| chunked cross-entropy | 2048→117.7 ms · 512→123.3 · 256→**140.7** | **chunking is worse.** The Cut-Cross-Entropy-style fix we were about to implement would have slowed it down |
| batch assembly | host slice+copy 0.23 ms vs on-device gather 0.08 ms | **irrelevant.** Our host-copy hypothesis was wrong by three orders of magnitude |
| attention, 4 layers | MHA 6.53 ms vs SDPA 5.21 ms | small but free; kept |
| `torch.unique` (both Kronecker arms) | 0.42 ms | free; the codec path is not a cost |

Also from the literature and load-bearing here: **bfloat16 is the wrong dtype on MPS** — it runs
through unoptimised kernels and can be up to 10× slower than float16, inverting the usual CUDA
advice. And `addcmul_`/`addcdiv_`, which are AdamW's internals, can silently yield zeros or NaNs on
non-contiguous MPS tensors.

**Configuration adopted:** fp32 throughout (fp16's 11% is not worth introducing a numerical
difference between the arms this experiment is comparing), SDPA causal attention, unchunked loss,
plain host batching. Realised rate **169–171 ms/step**, against ~130 ms predicted from the profile;
the remainder is AdamW and grad-norm clipping over 34M parameters. The original 1.3 s/step did not
reproduce.

Sources: arXiv:2411.09009 (Cut Cross-Entropy) · arXiv:2501.14925 (profiling Apple Silicon for ML
training) · HuggingFace Apple Silicon training docs · pytorch/pytorch#139386 (MPS bf16 autocast).

### E6 results — all three hypotheses refuted

**Completed** 2026-08-21 · `results/005_e6.json` · 6 runs, 3 arms × 2 seeds × 3000 steps,
~8.5 min each at 160–187 ms/step

| arm | input params | total | val loss | seed spread | bpb | en | hi | ml |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 15,633,920 | 34,492,928 | 6.4064 | 0.0034 | 1.4683 | 2.2492 | 1.0921 | 0.9947 |
| **byte** | 2,097,408 | 20,956,416 | **6.1521** | 0.0073 | **1.4100** | 2.1804 | **1.0432** | **0.9400** |
| char | **1,573,120** | 20,432,128 | 6.1768 | 0.0045 | 1.4156 | **2.1795** | 1.0484 | 0.9519 |

**H6c — REFUTED, upward.** We predicted both Kronecker arms would trail dense. Byte **beats** dense
by **0.2543 nats (−3.97%)** using **7.5× fewer input parameters** and 39% fewer total. Seed spread is
0.003–0.007, so the gap is ~35× the noise. This independently replicates the core claim of
arXiv:2605.29459 (−2.5 ± 0.2% vs a BPE-tied baseline at GPT-2 124M) at a much smaller scale on a
trilingual corpus. It was not one of our hypotheses, which makes it a useful check that the harness
is sound.

**H6a — REFUTED.** char is **+0.0247 nats (+0.40%) worse** than byte, and the gap exceeds the largest
seed spread (0.0073). Per the commitment recorded before the run: **E5 is an accounting result.** The
25% parameter saving, the 3.2× deeper window and the exact 131,072/131,072 round-trip are all real
and stand as facts about the codec, but they do not buy quality. char is 0.40% worse at 25% fewer
parameters — a trade, not a win, and we will not present it as one.

**H6b — REFUTED AND INVERTED, and this is the finding.** We predicted char's advantage would
concentrate in Indic. The opposite happened, monotonically:

| lang | Δbpb (char − byte) | byte active slots/token | char active slots/token | char/byte |
|---|---:|---:|---:|---:|
| en | **−0.0010** (char better) | 4.56 | 9.09 | **1.99** |
| hi | +0.0052 | 8.18 | 6.39 | 0.78 |
| ml | **+0.0120** (char worst) | 7.87 | 5.79 | **0.74** |

The ordering of the quality change matches the ordering of **active codec slots per token** exactly.
The byte arm activates `nbytes` entries per token; the char arm activates `2 × nchars` (block +
offset). For English that is 4.56 → 9.09, nearly double. For Malayalam it is 7.87 → 5.79, a quarter
fewer. Every language that gains slots improves; every language that loses slots degrades; the one
that loses most degrades most.

**So the premise of E3 and E5 was half wrong.** We treated UTF-8's 3-bytes-per-character expansion
for Indic as pure waste — a tax to be removed. It is also **3× more active codec parameters per
Indic token**, and the model demonstrably uses that capacity. Compressing the basis to characters
removes the waste *and* the capacity together, and at this scale the capacity is worth more.

**What survives from E3 and E5, and what does not.**
- Survives: the exact round-trip (131,072/131,072); the parameter arithmetic (D 8192 → 6144); the
  fertility-floor collapse (Malayalam 1.3295 → 1.0005) as a statement about what a 32-character
  window covers; E3's *structural* claim about byte-decoding heads, which E6 does not test because
  every arm here uses an identical dense head.
- Does not survive: any claim that a character basis trains better, and specifically that it helps
  Indic. Measured, it hurts Indic and helps English.

**The legitimate follow-up, stated as such and not as a rescue.** The comparison was not
parameter-matched: char had 25% fewer input parameters. A matched arm exists — at `P = 43` character
positions, `43×64 + 43×128 = 8256 ≈ D_byte = 8192`, giving equal parameters *and* 43 characters of
coverage. Whether char matches byte there is unknown and untested. Until it is run, the honest
position is that the character basis is cheaper and wider but slightly worse, and that the byte
basis's apparent inefficiency on Indic is partly doing real work.

**Throughput note (job 006).** Both optimisations we proposed were wrong. Batch scaling buys almost
nothing — 12,260 tok/s at batch 8 versus 13,003 at batch 64, **+6%** — because the head scales
linearly with tokens. And **fp16 autocast is slower**, not faster: 11,826 tok/s at batch 8 and 12,363
at batch 32, below their fp32 counterparts. The workload sits at ~1.5 TFLOPS against a 5.26 TFLOPS
ceiling and no configuration change we tested moves it, because it is bandwidth-bound on the
61,070-way softmax. Our earlier estimates of +26% and +22% were both wrong.

---

## E7 — Structure or slots? (hypotheses recorded before results)

**Date** 2026-08-21 · **Job** `007_e7_slots.py` · **Status at time of writing:
QUEUED, no results seen.** 5 arms × 2 seeds × 3000 steps, same harness and data as E6.

**Why.** E6 found quality tracked *active codec slots per token* rather than anything about bytes or
characters. If that is the real mechanism, two things follow that E6 could not test.

**The arms.** All character factorisations reconstruct the codepoint exactly (verified over the
whole vocabulary before the run: `char3 True, char4 True`).

| arm | construction | D | input params | slots/token |
|---|---|---:|---:|---|
| byte | `byte(256) ⊗ pos` | 8192 | 2,097,152 | `nbytes` |
| char2 | `blk(64) + off(128)`, each `⊗ pos` | 6144 | 1,572,864 | `2·nchars` |
| char3 | `blk(64) + off»3(16) + off&7(8)` | 2816 | 720,896 | `3·nchars` |
| char4 | `blk(64) + off»5(4) + (off»2)&7(8) + off&3(4)` | 2560 | 655,360 | `4·nchars` |
| **rand4** | **identical D and slot count to char4, indices randomised** | 2560 | 655,360 | `4·nchars` |

Corpus-weighted slots per token — note char4 gives Indic *more* than the byte grid does:

| lang | byte | char2 | char3 | char4 |
|---|---:|---:|---:|---:|
| en | 4.56 | 9.09 | 13.64 | 18.18 |
| hi | 8.18 | 6.39 | 9.58 | 12.78 |
| ml | 7.87 | 5.79 | 8.68 | 11.58 |

**Hypotheses, refutation criteria fixed in advance.**

| | hypothesis | refuted if |
|---|---|---|
| **H7a** | quality improves monotonically with slots: char4 > char3 > char2 | the ordering does not hold across both seeds |
| **H7b** | char4 matches or beats byte despite **69% fewer** input parameters | char4's val loss exceeds byte's by more than the seed spread |
| **H7c** | **rand4 is materially worse than char4** — the byte/character structure contributes beyond raw slot count | rand4 ≈ char4 |

**H7c is the one that matters and it is aimed at ourselves.** If randomised indices with identical D
and identical slot count perform as well as the structured codec, then the byte/character structure
is not load-bearing and Kronecker embeddings are, at this scale, a deterministic **sparse random
projection**. That would undercut the premise of the technique — including arXiv:2605.29459's, and
including E5's. **Committed in advance: if H7c is refuted we report it prominently and in those
terms, not as a footnote.**

**Known limits.** Two seeds, one scale, ~0.9 epochs. rand4 remains deterministic per token, so it
preserves the property that a token always maps to the same code; only the *structure* of the
mapping is destroyed.

### E7 results — all three refuted, and the random control wins

**Completed** 2026-08-21 · `results/007_e7.json` · 10 runs, 5 arms × 2 seeds × 3000 steps

| arm | D | input params | slots/tok | val loss | bpb | en | hi | ml |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| byte | 8192 | 2,097,408 | 6.47 | 6.1521 | 1.4100 | 2.1804 | 1.0432 | 0.9400 |
| char2 | 6144 | 1,573,120 | 11.59 | 6.1768 | 1.4156 | 2.1795 | 1.0484 | 0.9519 |
| char3 | 2816 | 721,152 | 17.39 | 6.2971 | 1.4432 | 2.1986 | 1.0715 | 0.9900 |
| char4 | 2560 | 655,616 | 23.19 | 6.3232 | 1.4492 | 2.2007 | 1.0799 | 0.9980 |
| **rand4** | 2560 | **655,616** | 23.19 | **6.0328** | **1.3826** | **2.1702** | **1.0077** | **0.9022** |

**H7a — REFUTED, backwards.** More slots made it *worse*, monotonically: char2 → char3 → char4 is
6.1768 → 6.2971 → 6.3232. The slot-capacity mechanism proposed in E6 is **wrong**.

**H7b — REFUTED.** char4 trails byte by 0.171 nats.

**H7c — REFUTED, decisively and in the direction we said would be most damaging.** rand4 does not
merely match char4; it **beats char4 by 0.290 nats** and beats the full byte grid by **0.119 nats at
3.2× fewer parameters**. It is the best arm in the experiment. rand4 vs char4 is airtight —
identical D, identical slot count, identical parameter count, only the index structure differs — so
the result is not confounded by capacity.

**Per the commitment recorded in 8ccd000: at this scale the byte/character structure is not
load-bearing for language-modelling loss. It is worse than random.**

**Mechanism, measured rather than asserted.** Pairwise cosine similarity between token codes,
400 sampled tokens per arm (all 400 codes distinct in every arm, so this is angular separation and
not collision):

| arm | max sub-factor cardinality | mean cos | p99 cos | val loss |
|---|---:|---:|---:|---:|
| rand4 | 2560 | **0.0083** | 0.0833 | **6.0328** |
| byte | 256 | 0.0828 | 0.4364 | 6.1521 |
| char3 | 64 | 0.3240 | 0.6667 | 6.2971 |
| char2 | 128 | 0.3598 | 0.6682 | 6.1768 |
| char4 | 64 | 0.3818 | 0.7250 | 6.3232 |

**Code overlap predicts loss: Spearman ρ = +0.900; Pearson +0.848 (mean cos), +0.903 (p99 cos).**
Low-cardinality sub-factors force many characters through the same few code directions, so tokens
become less separable; random indices spread them almost orthogonally. char2 vs char3 is the one
inversion, and parameter count (1.57M vs 721K) is the plausible second factor — so overlap is the
dominant predictor, not the only one.

**What this does and does not say about arXiv:2605.29459.**
- It does **not** refute the paper. Our scale is a toy (d=256, 4 layers, 6.65M tokens, ~0.9 epochs)
  against its GPT-2 124M / 2.5B tokens.
- It does say the **−2.5% validation-loss result needs a different explanation than byte structure**,
  because at our scale that advantage survives when the byte structure is destroyed and in fact grows.
  E6 independently replicated the direction of that result (−3.97% vs dense), so the effect is real;
  the attribution is what is in question.
- The paper's **spelling, OOV and typo claims are untouched**. rand4 keys its indices on token id, so
  it has none of those properties. Those are separate claims and this experiment does not test them.

**Constructive consequence — the next experiment.** If separation is what buys loss and byte
structure is what buys OOV/typo robustness, they are not in conflict: a **hybrid** code that keeps the
byte grid and adds a high-cardinality random component should get both. That is a concrete, testable
design and it is the natural E8.

**Limits.** Two seeds, one scale, ~0.9 epochs, one corpus, dense output head throughout. rand4 is
deterministic per token but keyed by token id, so it is a parameter-efficient random sparse embedding
rather than a codec, and it cannot be computed for an unseen string.

---

## E8 — Spelling structure and separation, decoupled (hypotheses before results)

**Date** 2026-08-21 · **Job** `008_e8_hash.py` · **Status: QUEUED, no results seen.**

**Why.** E7 found code overlap predicts loss (ρ = +0.900) and a random code wins. But building the
hashed codecs exposed why that cannot be the whole story: **overlap between byte-derived codes *is*
spelling similarity.** Two tokens sharing a byte at a position necessarily share code mass. So
"minimise overlap" and "preserve spelling structure" are the same quantity with opposite signs, and
E7's conclusion was under-specified.

**The resolution — split overlap into two parts, measured before this run** (400 tokens; "related"
= shares a 4-byte prefix; selectivity = related-cos / random-cos):

| arm | related cos | random cos *(spurious)* | **selectivity** | input params |
|---|---:|---:|---:|---:|
| byte | 0.5823 | 0.0735 | **7.9×** | 2,097,408 |
| **hashbyte4** | 0.5862 | 0.0804 | **7.3×** | **655,616** |
| hashchar4 | 0.5363 | 0.0859 | 6.2× | 655,616 |
| char4 | 0.7165 | 0.3739 | 1.9× | 655,616 |
| rand4 | 0.0127 | 0.0093 | 1.4× | 655,616 |

This reinterprets every prior arm. char4's related-cos of 0.72 looked high but is **fake** — its
unrelated pairs sit at 0.37, so spurious overlap swamps the signal, and its selectivity is 1.9×.
rand4 has no spelling structure at all (1.4×). Only byte and the hashed codecs are selective.

**The construction.** hashbyte4/hashchar4 are CountSketch-style: each `(symbol, position)` pair is
hashed to K=4 slots in D=2560. The code stays a **pure function of the token's bytes**, so unseen
strings work and spelling similarity survives — unlike rand4, which is keyed on token id and
therefore cannot be computed for a string it has not seen.

**Hypotheses, refutation criteria fixed in advance.**

| | hypothesis | refuted if |
|---|---|---|
| **H8a** | hashbyte4 matches byte's val loss at **3.2× fewer parameters** (equal spurious overlap, equal selectivity) | hashbyte4's loss exceeds byte's by more than the seed spread |
| **H8b** | hashchar4 ≈ hashbyte4 — once spurious overlap is removed, byte-vs-character keying does not matter for loss, so the 32-**character** window comes free | the two differ by more than the seed spread |
| **H8c** | rand4 still beats both, and that gap **is the price of spelling structure**, measurable in nats | rand4 does not beat them, which would mean selectivity is free |

**What each outcome means.** If H8a holds, we have a strictly better input codec for K2: same
spelling behaviour, same quality, 3.2× fewer parameters. If H8b also holds, characters can be used
instead of bytes, which restores the 32-character window and removes the Indic truncation ceiling
measured in E1/E2 at no cost. H8c quantifies, for the first time as far as we can tell, what
byte-structured embeddings *pay* for their spelling property.

**Limits unchanged.** Two seeds, one scale, ~0.9 epochs, dense output head, one corpus.

### E8 results — a codec that beats the published one at a third of the parameters

**Completed** 2026-08-21 · `results/008_e8.json` · 8 runs, 4 arms × 2 seeds × 3000 steps

| arm | input params | val loss | seed spread | bpb | en | hi | ml |
|---|---:|---:|---:|---:|---:|---:|---:|
| byte *(published, d_p=32)* | 2,097,408 | 6.1521 | 0.0073 | 1.4100 | 2.1804 | 1.0432 | 0.9400 |
| hashbyte4 | 655,616 | 6.1426 | 0.0115 | 1.4078 | 2.1765 | 1.0420 | 0.9388 |
| **hashchar4** | **655,616** | **6.1220** | 0.0084 | **1.4030** | **2.1734** | **1.0343** | **0.9345** |
| rand4 | 655,616 | 6.0328 | 0.0029 | 1.3826 | 2.1702 | 1.0077 | 0.9022 |

Max seed spread across arms: **0.0115**.

**H8a — CONFIRMED.** hashbyte4 is −0.0094 nats vs byte, inside the seed spread, at **3.2× fewer
input parameters**. Equal quality, a third of the parameters.

**H8b — REFUTED, favourably.** We predicted hashchar4 ≈ hashbyte4. It is **better** by 0.0206 nats,
which exceeds the spread. Characters are not merely free, they win. Recorded as a refutation because
that is what it is: the prediction was equivalence and the observation is a difference.

**H8c — CONFIRMED.** rand4 still leads. **The price of spelling structure is 0.0892 nats**
(hashchar4 − rand4). As far as we can tell this quantity has not previously been measured.

**The headline.** hashchar4 versus the published byte codec:

| | published byte | hashchar4 | delta |
|---|---:|---:|---|
| val loss | 6.1521 | 6.1220 | **−0.0301 nats (−0.49%)**, 2.6× seed spread |
| input params | 2,097,408 | 655,616 | **3.2× fewer** |
| en / hi / ml bpb | 2.1804 / 1.0432 / 0.9400 | 2.1734 / 1.0343 / 0.9345 | better on **all three** |
| window | 32 **bytes** = 10 Indic chars | 32 **characters**, every script | ceiling removed |
| Malayalam fertility floor | 1.3295 | 1.0005 | E2's floor collapses |
| spelling selectivity | 7.9× | 6.2× | **−21%, a real cost** |

At V5's reference shape (V=131,072, d=8,096) that is **66,322,432 → 20,725,760** input parameters,
1.06 GB → 0.33 GB of AdamW state.

**The honest qualifier.** hashchar4 is **not strictly dominant**. It gives up ~21% of spelling
selectivity (7.9× → 6.2×). Loss, parameters and window all improve; the spelling property degrades.
Whether that degradation matters depends entirely on whether selectivity is load-bearing for the
downstream properties the published work actually claims — typo robustness, OOV handling,
morphological generalisation — **none of which this experiment tests.** That is E9 and it is the
condition on which this result should be sold or withdrawn.

**Construction, for reproduction.** Each `(codepoint, character-position)` pair is hashed to K=4
slots in D=2560 by a 64-bit mix (`x ^= x>>33; x *= 0xff51afd7ed558ccd; x ^= x>>33;
x *= 0xc4ceb9fe1a85ec53; x ^= x>>33`), summed, scaled by `1/√L`, then z-normalised — the same
normalisation as arXiv:2605.29459 Eq. 1. The code remains a pure function of the token's text, so
unseen strings are embeddable and spelling similarity survives.

---

## E9 — Is spelling selectivity load-bearing? (hypotheses before results)

**Date** 2026-08-21 · **Job** `009_e9_probes.py` · **Status: QUEUED, no results seen.**
Same 4 arms, 2 seeds, 3000 steps.

**Why.** E8's hashchar4 wins on loss, parameters and window but gives up **21% of spelling
selectivity** (7.9× → 6.2×). That trade is only acceptable if selectivity does not buy the downstream
properties the published work actually claims. E9 tests two of them.

**Probe A — rare-tail NLL by train-frequency bucket.** A token seen 1–9 times receives almost no
direct gradient, so decent NLL there means it borrowed from slots shared with other tokens — which is
precisely what structure is supposed to provide. Crucially, **58% of val occurrences are freq-1000+
tokens**, so any rare-tail effect is invisible in aggregate loss. This is also the *"token-bucket NLL
analysis: per-frequency-bucket training dynamics"* listed verbatim as future work in arXiv:2605.29459.

| bucket | tokens | val occurrences |
|---|---:|---:|
| 0 | 291 | 346 |
| 1–9 | 3,380 | 4,103 |
| 10–99 | 10,401 | 19,732 |
| 100–999 | 6,819 | 62,524 |
| 1000+ | 960 | 118,095 |

Bucket 0 is underpowered (346 occurrences, 332 of them English) and will be reported but not relied on.

**Probe B — typo stability.** 4,000 edit-distance-1 token pairs built by delete-1 signature hashing,
both members seen ≥20× in training, spanning all three scripts (` fat`/` Nat`, ` निक`/` शिक`,
` നമ്മ`/` തമ്മ`). Swap the last context token for its neighbour and measure whether the next-token
top-1 survives. Same shape as the paper's 55.5% vs 47.3% claim — **with rand4 as the control it did
not have.**

**Hypotheses, refutation criteria fixed in advance.**

| | hypothesis | refuted if |
|---|---|---|
| **H9a** | rand4's aggregate win **shrinks or reverses** in the rare buckets — structure buys the tail | rand4's advantage is flat or grows as frequency falls |
| **H9b** | typo stability tracks **selectivity**: byte (7.9×) > hashbyte4 (7.3×) > hashchar4 (6.2×) ≫ rand4 (1.4×) | the ordering does not follow selectivity |
| **H9c** | hashchar4's 21% selectivity loss costs **little** typo stability relative to byte | hashchar4 falls far closer to rand4 than to byte |

**What each outcome decides.** If H9a and H9b hold, the story is complete and constructive: aggregate
loss is the wrong metric for judging a codec, structure buys the rare tail and robustness, and
hashchar4 is a good point on that trade. **If H9c is refuted — if hashchar4's typo stability collapses
toward rand4 — then E8's codec is not a safe replacement and we say so and withdraw the
recommendation.** That is the pre-committed condition.

---

## R1 — Adversarial review lands. E7/E8 headline is confounded by trained capacity.

**Date:** 2026-08-21. Independent SOTA reviewer briefed to refute. Three claims landed.
I re-derived all three from our own artifacts before accepting them; all reproduce exactly.

### R1.1 The parameter accounting was allocated rows, not trained rows — measured

Counted rows of `W` that receive gradient from at least one token with train frequency ≥ 1
(`liverows.py`, from `data/codec.npz`, `data/hash_idx.npz`, `data/train.npy`; 60,779 of 61,070
types occur in train):

| arm | D | allocated params | **live rows** | live % | **trained params** |
|---|---:|---:|---:|---:|---:|
| byte (published) | 8192 | 2,097,408 | 1,855 | 22.6% | 474,880 |
| char2 | 6144 | 1,573,120 | 1,305 | 21.2% | 334,080 |
| char3 | 2816 | 721,152 | 547 | 19.4% | 140,032 |
| char4 | 2560 | 655,616 | 479 | 18.7% | 122,624 |
| rand4 | 2560 | 655,616 | 2,560 | 100.0% | 655,360 |
| hashbyte4 | 2560 | 655,616 | 2,415 | 94.3% | 618,240 |
| hashchar4 | 2560 | 655,616 | 2,520 | 98.4% | 645,120 |

Two different true statements were blurred into one false one:

- **True:** hashchar4 allocates 3.20× fewer rows than the published byte grid (655,616 vs 2,097,408).
- **True:** hashchar4 *trains* **1.36× more** parameters than the byte grid (645,120 vs 474,880).
- **False, and what we wrote:** that hashchar4 matched/beat the byte grid *because* it was smaller.
  Trained capacity went **up**, not down. The allocated number cannot explain the quality result.

The byte grid's `d_p=32` position window is **77.4% dead on this corpus** — which is E1/E2's own
finding restated in parameter terms. We compared against a baseline we had already shown to be
mostly unused, then credited ourselves the unused part.

### R1.2 E7's "airtight control" is not a control — measured

E7 claimed "rand4 vs char4: identical D, identical slot count, identical parameter count, only the
index structure differs." The first two hold. The third is false: **479 trained rows vs 2,560, a
5.3× capacity gap.** Randomising the indices did not remove structure at constant capacity — it
*created* capacity by spreading tokens across 2,080 rows the structured code never touches.

**E7's conclusion ("byte structure contributes ≤0% of the LM advantage") does not follow from E7.**

### R1.3 A simpler predictor beats our overlap mechanism — measured

Spearman(live rows, val loss), our own numbers:

| arm | live rows | val loss (mean of 2 seeds) |
|---|---:|---:|
| rand4 | 2560 | 6.0328 |
| hashchar4 | 2520 | 6.1220 |
| hashbyte4 | 2415 | 6.1426 |
| byte | 1855 | 6.1521 |
| char2 | 1305 | 6.1768 |
| char3 | 547 | 6.2971 |
| char4 | 479 | 6.3232 |

- E7 arms only (n=5): **ρ = −1.0000** (perfect), Pearson(log live, loss) = −0.977
- E7+E8 (n=7): **ρ = −1.0000** (perfect), Pearson(log live, loss) = −0.947

Our published predictor was code overlap at ρ = +0.900 on n=5 **with an inversion we patched by
appealing to parameter count as "a plausible second factor."** That patch was the right variable
all along. One capacity variable orders all seven arms with no inversion and no residual.
**Code overlap is a proxy for trained capacity, not an independent mechanism.**

Statistical note the reviewer is right about: n=5, ρ=0.900 ⇒ exact permutation p = 5/120 =
**0.042 one-sided / 0.083 two-sided**, on a predictor chosen after seeing the ordering. Not leadable.

### R1.4 `1/√L` is provably inert — in our code and in both papers

z-normalisation is scale-invariant, so a per-token positive scalar applied before it cancels exactly.
Measured: max relative difference between `z(κ)` and `z(κ/√L)` = **3.5×10⁻⁴**, entirely the `eps=1e-5`
term. The paper justifies the factor as *"variance-preserving normalization … expected squared L2 norm
of κ(b) is approximately 1 regardless of L"* and then z-normalises, which cancels it.
This is a clean correction to Eq. 1 that costs nothing to state and applies to the published work too.

### R1.5 Our codec is not the paper's Eq. 1

`KronEmb.forward` computes `W[idx].sum(1)` (projection fused) and z-normalises **across d_model,
after projection**. The paper z-normalises κ(b) **across its D coordinates, before** the `Linear(D, d_model)`.
Not equivalent: the paper's ordering leaves a length-dependent rank-1 bias term, ours does not.
E6–E8 replicate a *variant*, not arXiv:2605.29459.

Worse: this per-token z-norm is applied to **every Kronecker arm and to none of the dense arm**
(`nn.Embedding` output used raw). **E6's "byte beats dense −3.97%, independently replicating the paper"
is confounded by an input normalisation only one side has.**

### R1.6 The val language mix is not 1:1:1 — measured

Documented as a 1:1:1 chunk interleave. Actual, from `data/val_lang.npy` (204,800 tokens):

| lang | tokens | share |
|---|---:|---:|
| en | 101,376 | 49.5% |
| hi | 38,912 | 19.0% |
| ml | 64,512 | 31.5% |

English also carries by far the highest loss (6.89 vs hi 5.91, ml 5.13, from `009_e9.json`), so our
headline aggregate val loss is **approximately an English metric** on a corpus whose whole motivation
is Indic. Every aggregate loss number in E6/E7/E8 inherits this.

### R1.7 Leakage: hunted and refuted, not a concern

The reviewer measured it directly rather than assuming: verbatim val n-grams also in train =
1.09% at n=8, 0.17% at n=16, 0.08% at n=32, **0.00% at n≥64**. Best-matching train chunk shares a
mean of 2.37 rare types per 1024-token val chunk. Absurd upper bound ≈ 0.005 nats against a
0.119-nat effect. **Leakage is not the explanation.** Recorded so we stop paying attention to it.

### R1.8 Prior art we missed — the core idea is not novel

- rand4 **is the hashing trick**, K=4 (Weinberger et al., ICML 2009; Ganchev & Dredze 2008).
- **Hash Embeddings** (Svenstrup et al., NeurIPS 2017, arXiv:1709.03933) — k vectors per token from a
  shared pool, "at least the same level of performance … at a fraction of the parameters." Our rand4
  result, published 2017, with importance weights we do not have.
- **Bloom Embeddings** (Serrà & Karatzoglou, RecSys 2017) — multi-hash beats the full table, up to +12%.
- **CANINE** (Clark et al., TACL 2022) — codepoint → K=8 hashes into per-function tables. **This is hashchar4.**
- **spaCy MultiHashEmbed** (arXiv:2212.09255) — K=4 is *spaCy's default*, not our design choice.
- **Tensorized Embedding Layers** (2019), **KroneckerBERT** (2021) — Kronecker factorisation of the
  embedding matrix predates arXiv:2605.29459 by five years; neither it nor we cited them.
- **arXiv:2605.09751** — 32L, 17B tokens, fixed *and randomly recoded* input codes reach **parity**
  (2.39 / 2.36 / 2.44, within a 4.8% seed band). At 375× our token budget the random-code advantage is
  **zero**. Best-calibrated prior on where our headline goes at scale.

### What survives

- **E1** — 32-byte cap honoured exactly; Indic tokens cap at 10 chars vs Latin ~26, a 2.6× asymmetry.
  No training run, fully verifiable, novel in this context. **This is now our strongest result.**
- **E2/E5** as stated, with E2 restricted to the artifact (see R1.9).
- **R1.4** — the `1/√L` correction, which applies to the published paper.
- Paired-by-seed statistics are *better* than we claimed: seed effect is common-mode, so
  hashchar4−byte = −0.0306/−0.0295, t(1), p≈0.012. But we sampled **zero** of the two variance
  components that matter for a "random beats structured" claim: **codec draw (n=1)** and **val split (n=1)**.

### R1.9 Claim 5 was a category error

The lesson's "floor of one token per word" describes what *vocabulary scaling* asymptotes to for an
unconstrained tokenizer. The 32-byte cap is a property of one artifact. Refuting the former with the
latter is a category error — especially since our own H2b found the cap **non-binding today**
(Malayalam actual 3.298 vs floor 1.3295, ~6:1 headroom to vocabulary rather than window).
Defensible form: the arithmetic statement about this tokenizer, on FLORES devtest, not a refutation
of the course.

### Status

**E7 and E8 headline claims are withdrawn pending a capacity-matched control (E10).**
E9 was still running when this landed (2/8: byte s0 6.1484 / s1 6.1557, typo-keep 27.8% / 23.7%).
E9 tests typo stability, which is downstream of a claim that is now confounded — but its Probe A
buckets are 96% English (bucket 0 = 332 en / 5 hi / 9 ml), so Probe A cannot answer its question
regardless.

---

## E11 — Full-vocabulary audit of the tokenizer/embedding pair (CPU only, no training)

All 131,072 tokens decoded from `theschoolofai/BrahmicTokenizer-131K` via the GPT-2 byte-level map;
script assigned by Unicode name of alphabetic characters; "content" = token string minus the
leading-space marker. Scripts: `full_scripts.npy`, lengths: `full_lens.npy`.

### E11.1 E1 confirmed at full vocabulary, and sharpened — measured

131,072/131,072 decode. Max token length **exactly 32 bytes, 0 above**. Confirms E1's structural check
on the whole vocabulary rather than the 61,070 observed subset.

The earlier "Indic caps at 10 chars" is correct for **content** characters. Raw string length reaches 11
because the leading-space marker counts as a character. Stated precisely:

| script | n | bytes/char | cap (chars) | longest observed | at ceiling? |
|---|---:|---:|---:|---:|:--:|
| LATIN | 108,171 | 1 | 31 | 26 | **no**, 5 spare |
| DEVANAGARI | 4,002 | 3 | 10 | 10 | YES |
| BENGALI | 2,149 | 3 | 10 | 10 | YES |
| MALAYALAM | 1,783 | 3 | 10 | 10 | YES |
| GUJARATI | 1,720 | 3 | 10 | 10 | YES |
| KANNADA | 1,415 | 3 | 10 | 10 | YES |
| TELUGU | 1,380 | 3 | 10 | 10 | YES |
| TAMIL | 1,047 | 3 | 10 | 10 | YES |
| ORIYA | 645 | 3 | 10 | 9 | no |
| GURMUKHI | 580 | 3 | 10 | 7 | no |

Longest content token: Latin 26 vs Brahmic 10 = **2.60×**, confirming E1's ratio under a clean definition.
7 of 9 Brahmic scripts sit *exactly* at the arithmetic ceiling ⌊31/3⌋ = 10; the two that do not
(Oriya, Gurmukhi) are the two smallest token populations, i.e. vocabulary-bound rather than cap-bound.

### E11.2 The cap is NOT actively clipping — measured, and this refutes a framing I was about to adopt

If the 32-byte cap were binding, token-length histograms would pile up against it. They do not.
Share of each script's tokens at each content length:

| script | 7 | 8 | 9 | 10 | 11+ | n(10)/n(9) |
|---|---:|---:|---:|---:|---:|---:|
| LATIN | 10.9% | 7.8% | 5.2% | 3.2% | 3.9% | 0.62 |
| DEVANAGARI | 3.1% | 1.6% | 0.6% | **0.1%** | 0.0% | 0.23 |
| MALAYALAM | 3.1% | 2.7% | 1.0% | **0.6%** | 0.0% | 0.59 |
| TAMIL | 2.9% | 1.3% | 0.7% | 0.3% | 0.0% | 0.43 |
| TELUGU | 2.0% | 1.5% | 0.9% | 0.1% | 0.0% | 0.15 |

Every Brahmic script is still **decaying** at the ceiling — 0.1–0.6% of tokens sit there. There is no
spike. **The cap costs approximately nothing today.** It is a latent ceiling, not an active constraint.

This is consistent with H2b (vocabulary binds, not the window) and it means any "the byte window is
unfair to Indic" claim must be stated as *headroom*, not as a present cost. Recorded because I was
about to build a framing on the opposite assumption.

### E11.3 Fertility floor under a byte cap vs a character cap — measured

FLORES devtest, whitespace words, cap of 31 content bytes/chars (32 minus the leading-space marker).
Floor = mean over words of ⌈size / cap⌉ — the minimum tokens/word achievable at *any* vocabulary size.

| lang | words | actual | floor (byte-32) | floor (char-32) | % words structurally multi-token (byte) | (char) |
|---|---:|---:|---:|---:|---:|---:|
| mal_Mlym | 14,930 | 3.4999 | **1.3301** | 1.0003 | **31.53%** | 0.03% |
| tam_Taml | 16,775 | 3.1387 | 1.2322 | 1.0000 | 22.82% | 0.00% |
| kan_Knda | 16,100 | 3.2551 | 1.1999 | 1.0000 | 19.86% | 0.00% |
| tel_Telu | 16,938 | 3.0401 | 1.1168 | 1.0000 | 11.66% | 0.00% |
| mar_Deva | 19,046 | 2.5596 | 1.0727 | 1.0000 | 7.25% | 0.00% |
| ory_Orya | 19,620 | 4.1454 | 1.0598 | 1.0000 | 5.96% | 0.00% |
| ben_Beng | 19,506 | 2.3342 | 1.0532 | 1.0000 | 5.31% | 0.00% |
| hin_Deva | 25,643 | 1.6483 | 1.0120 | 1.0000 | 1.20% | 0.00% |
| eng_Latn | 21,901 | 1.2393 | **1.0000** | 1.0000 | 0.00% | 0.00% |

Reproduces E2's Malayalam floor (1.3301 here vs 1.3295 logged; difference is the 31-vs-32 cap
convention — E2's number should be restated with the convention named). **A character position basis
takes every floor to 1.0000.** The gain is largest for Dravidian scripts (agglutinative, long words)
and exactly zero for English.

Caveat that must travel with this: actual fertility (Malayalam 3.50) is ~2.6× above the floor (1.33),
so removing the floor changes nothing today. Per E11.2 the ceiling is not being pressed against.
**The character basis buys headroom at 25% less dimension (E5: D 6144 vs 8192), not a present win.**

### E11.4 What actually binds: vocabulary allocation — measured

| allocation | tokens | share |
|---|---:|---:|
| **LATIN** | **108,171** | **82.5%** |
| all 9 Brahmic scripts combined | 14,721 | 11.2% |
| non-alphabetic | 6,819 | 5.2% |
| partial UTF-8 fragments | 1,306 | 1.0% |
| Devanagari | 4,002 | 3.05% |
| Bengali | 2,149 | 1.64% |
| Malayalam | 1,783 | 1.36% |
| Tamil | 1,047 | 0.80% |
| Gurmukhi | 580 | 0.44% |

**Latin : Brahmic = 7.3 : 1** in an artifact named *BrahmicTokenizer-131K*. Malayalam receives 1.36%
of the vocabulary and pays 3.50 tokens/word; English receives the bulk and pays 1.24.

This is the mechanism behind H2b, quantified: the window is not what is costing Indic — the
vocabulary split is. The tokenizer is a retrofit of `o200k_base`, so the Latin mass is inherited from
the donor; the Brahmic capacity *added* by the retrofit is 11.2%.

### Consequence for the direction of this project

The codec's parameter cost is `(D+1)·d` — **independent of V**. A dense table costs `V·d`. Since the
measured fix for Indic fertility is *more vocabulary*, and vocabulary is free under a codec but linear
under a dense table, the defensible case for compositional embeddings in this setting is
**decoupling vocabulary size from parameter count**, not the −2.5% val-loss claim. That argument needs
the codec only to be *comparable* to dense, which survives the capacity critique in §R1 — it does not
need the codec to win.

---

## E12 — The byte basis cannot equalize scripts at any position budget (derivation + measurement)

Motivated by the paper's own prescribed mitigation, arXiv:2605.29459 Limitations, **stated**:

> "Multilingual tokenizers (especially SentencePiece with Indic scripts) have the highest truncation
> rates. **Increasing `d_p` at the cost of `D` is the obvious mitigation.**"

E12 tests whether that mitigation can work. `dpsweep.py`, FLORES devtest, whitespace words.

### E12.1 Maximum representable token length, in characters — derived, bound not observation

| `d_p` | `D_byte` = 256·d_p | Latin (1 B/char) | Brahmic (3 B/char) | **ratio** | char basis, all scripts |
|---:|---:|---:|---:|---:|---:|
| 16 | 4,096 | 16 | 5 | **3.20** | 16 |
| 32 | 8,192 | 32 | 10 | **3.20** | 32 |
| 48 | 12,288 | 48 | 16 | **3.00** | 48 |
| 64 | 16,384 | 64 | 21 | **3.05** | 64 |
| 128 | 32,768 | 128 | 42 | **3.05** | 128 |
| 256 | 65,536 | 256 | 85 | **3.01** | 256 |

**The ratio is invariant.** It is exactly bytes-per-character and does not depend on `d_p`. Spending
dimension on the position axis buys both scripts more length in the same proportion and closes
nothing. **The paper's stated mitigation cannot work at any `d_p`.** A character position basis is
scale-free in the script: every script gets the full `d_p`.

### E12.2 Fertility floor vs position budget — measured

Floor = mean over FLORES devtest words of ⌈size / d_p⌉ (no leading-space adjustment here; see caveat).

| `d_p` | `D_byte` | `D_char` | eng byte | eng char | hin byte | mal **byte** | mal **char** | tam byte | tam char |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4,096 | 3,072 | 1.0006 | 1.0005 | 1.2281 | **2.1611** | 1.0650 | 2.0065 | 1.0328 |
| 32 | 8,192 | 6,144 | 1.0000 | 1.0000 | 1.0119 | **1.3255** | **1.0001** | 1.2300 | 1.0000 |
| 48 | 12,288 | 9,216 | 1.0000 | 1.0000 | 1.0003 | 1.0640 | 1.0000 | 1.0326 | 1.0000 |
| 64 | 16,384 | 12,288 | 1.0000 | 1.0000 | 1.0000 | 1.0109 | 1.0000 | 1.0023 | 1.0000 |
| 96 | 24,576 | 18,432 | 1.0000 | 1.0000 | 1.0000 | 1.0001 | 1.0000 | 1.0000 | 1.0000 |

**The headline number of this session.** To drive the Malayalam floor to ≈1.01 on a **byte** basis
costs `d_p=64`, `D=16,384`. The **character** basis reaches a *better* floor (1.0001) at `d_p=32`,
`D=6,144` — **2.67× less dimension for a strictly better floor**, against an architecture whose
entire pitch is parameter efficiency. English is 1.0000 on both bases at every `d_p ≥ 32`, so the
saving is paid for entirely by the scripts that were being penalised.

### E12.3 Three ratios that must never be blurred

A reviewer will collapse these; label every use.

| quantity | value | kind |
|---|---:|---|
| bound, no leading space: 32 vs ⌊32/3⌋=10 | **3.20×** | derived |
| bound, with leading-space marker: 31 vs ⌊31/3⌋=10 | **3.10×** | derived |
| observed longest token in *this* vocabulary: Latin 26 vs Brahmic 10 | **2.60×** | measured |

E11.1 reports the measured 2.60×. Earlier S7 notes used 3.2× for the bound. Both are correct for
different objects; the write-up must state which and report the other alongside.

### E12.4 Status of the claim after the novelty check

Confirmed unclaimed in the literature (see `SOTA-MAP.md` for citations):
1. **Attributing tokenizer unfairness to an embedding architecture's positional budget.** Every
   causal paper in this space attributes premiums to data mix, vocabulary size, pre-tokenization,
   BPE-vs-Unigram, byte premium or morphology. There is no published architecture→tokenizer arrow.
2. **A fertility floor derived from a structural cap, independent of vocabulary size.** Arnett et al.
   (NeurIPS 2025) is the near-miss: an *empirically fitted* per-language power-law asymptote from
   BPE runs at 8K–262K vocab, with no closed form and no identified cause. Ours is derived in closed
   form and computable without training anything. **Different epistemic object — say so explicitly
   or a reviewer collapses them.**
3. **The character-position basis as the fix, at lower `D`.** The paper enumerates alternative
   factorizations (byte-bigrams, byte triples, position-aware n-grams); character-position is absent.

Already taken, and must be conceded loudly, not claimed:
- Tokenizers are unfair across scripts (Petrov NeurIPS 2023; Ahia EMNLP 2023; Rust ACL 2021).
- UTF-8 gives Brahmic 3 bytes/char and this inflates sequences (MYTE §2, ACL 2024, verbatim).
- Characters give better parity than bytes (**MYTE concedes this explicitly**; it rejects characters
  only because the Unicode alphabet is 149,878 symbols — we must state that we re-index the
  *position* axis and keep the byte alphabet, or a MYTE reader assumes we missed it).
- Codepoints as a basis (CANINE TACL 2022 — but its stated justification is *ergonomic*, not fairness).
- No vocabulary size equalizes languages (Arnett NeurIPS 2025).

**Precedent worth one sentence:** AllenNLP's ELMo `ELMoCharacterMapper` uses `max_word_length = 50`
with ids 0–255 as UTF-8 bytes — the identical pattern (compositional embedding, fixed budget, indexed
in bytes). No published analysis of its script-differential effect. This shows a recurring
architecture pattern rather than one preprint's bug.

### E12.5 The objection that decides whether this is publishable

**"The constraint is slack, so the harm is fictional."** Built from both papers' own numbers:
the Kronecker paper measures ≤0.18% of tokens truncated at `d_p=32`; BrahmicTokenizer reports
measured Malayalam fertility 3.47 (we measure 3.50); our floor is 1.3255. The tokenizer operates at
**2.6× above its own floor**, and E11.2 found no pile-up at the cap. We have identified a wall the
system never touches.

**This objection is currently unanswered and it is fatal if it stays that way.** The answer is not
rhetorical, it is E13: sweep vocabulary size and show the Malayalam floor is the *terminal asymptote*
every tokenizer under this architecture converges to, while English converges to 1.0000. That
converts "3.50 is far from 1.33, who cares" into "1.33 is where Malayalam terminates and 1.00 is
where English terminates, under this embedding, forever." Use Arnett et al.'s power-law methodology
so the comparison is like-for-like, and state that our asymptote has a closed form theirs does not.

### Caveats
- Floors here use no leading-space adjustment (E11.3 used 31); the two differ in the 4th decimal
  (mal 1.3255 vs 1.3301). Pick one convention for the write-up and name it.
- Whitespace word segmentation is a poor unit for agglutinative Dravidian text and likely understates
  true fertility.
- "3 bytes/char" is modal, not universal: Brahmic combining marks and ZWJ/ZWNJ vary, and the ratio
  table uses the modal value.

---

## E13 — Does the byte cap bind asymptotically? (pre-registered, CPU)

**Why.** E12.5 identified the objection that decides whether the architecture→tokenizer claim is
publishable: *the constraint is slack*. Malayalam runs at 3.50 tokens/word against a 1.3255 floor,
the paper measures ≤0.18% truncation, and E11.2 found no pile-up at the cap. If the cap never binds
at any reachable vocabulary size, we have described a wall nobody touches and the claim dies.

**Design.** Monolingual BPE (HF `tokenizers`, ByteLevel, `add_prefix_space=True`), matched 125.8 MB
training budget per language, over a vocabulary ladder 4K → 256K. Two arms per (language, vocab):

- **capped** — `max_token_length=32`, reproducing the BrahmicTokenizer/Kronecker constraint.
- **uncapped** — `max_token_length` unbounded. **This is the counterfactual the whole claim needs.**

Languages: `en` (floor 1.0000), `hi` (1.0119), `kn` (1.1991), `ta` (1.2300), `ml` (1.3255) — chosen
to span the measured floor range. Fertility measured on FLORES devtest (parallel, so evaluation is
content-matched across languages). Method follows Arnett et al. (NeurIPS 2025) so the asymptote
comparison is like-for-like.

### Pre-registered hypotheses

- **H13a** — For **English**, capped and uncapped curves are identical at every vocabulary size. The
  cap can never bind for a 1-byte/char script below `d_p=32` characters.
- **H13b** — For **Malayalam**, capped and uncapped curves **diverge, and the gap grows with
  vocabulary size**. The cap becomes binding as vocabulary grows, even though it is slack today.
- **H13c** — The capped Malayalam curve **asymptotes at or near the derived floor 1.3255**, while the
  uncapped curve continues below it.
- **H13d** — Divergence onset orders by floor: `ml` before `ta` before `kn` before `hi`, and never for `en`.

### Pre-committed refutation

**If H13b is refuted — if capped and uncapped Malayalam curves stay within noise out to 256K — then
the cap is slack even asymptotically, the architecture→tokenizer claim collapses, and we drop it and
say so.** In that case the surviving contributions are E1, the `1/√L` correction, the capacity
diagnostic, the bucket decomposition and the power analysis, none of which depend on E13.

### Controls
- Requested vocabulary is not always reachable: a 20 MB Malayalam sample yielded 21,352 of a
  requested 32,000. **Achieved vocabulary size is recorded per run**; any point where achieved <
  requested is data-limited, not cap-limited, and is excluded from asymptote fits.
- Both arms see byte-identical training data within a language, so the capped/uncapped contrast has
  no data confound.
- Training budget is matched in **bytes**, so Indic languages see ~3× less *content* than English.
  This is a confound for cross-language comparison of achieved fertility, **not** for the
  capped-vs-uncapped contrast and **not** for the floors (derived from parallel FLORES). Labelled.

---

## E14 — Frequency-stratified permutation null: was the structure load-bearing? (pre-registered)

**Why the previous two controls failed.** E7 used `rand4` (random indices at matched allocated `D`):
5.3× trained-capacity gap. E10 proposed `rand479` (random indices at matched *live rows*): identical
live rows (479) and identical trained parameters (122,624), but **effective rank 332.9 vs char4's
29.4 — an 11.3× mismatch**. Matching any scalar summary of the code leaves the claim "equivalent
except for structure" unfalsifiable, because structure *is* what produces the collision pattern.

**The design.** Keep the code matrix; permute **which token receives which code**, within strata of
(train-frequency quantile × slot count), 400 strata. The code multiset is preserved exactly, so live
rows, trained parameters, row-load distribution and total gradient-touches are identical *by
construction*. The only thing destroyed is the alignment between code overlap and string overlap.

**Control verified locally before spending GPU time** (`010_e14_perm.py` `audit()`; numbers from a
CPU dry run over `codec.npz` + `train.npy`):

| arm | live rows | total gradient-touches | tokens receiving a different code |
|---|---|---|---:|
| byte, draws 1/2/3 | 1,859 → **1,859** (exact) | 394,906 → **394,906** (exact) | 96.3–96.4% |
| char4, draws 1/2/3 | 480 → **480** (exact) | 1,416,040 → **1,416,040** (exact) | 97.1–97.3% |

Residual effective rank (reviewer-measured): byte_shuf **+0.7%**, char4_shuf **+3.4%**, against
rand479's **+1032%** — and signed *against* the structure hypothesis, so a structured win is
conservative. Stratification matters: 20 strata leaves +14.6% residual on char4, 100 leaves +10%,
400 leaves +3.4%.

**This also dissolves the learning-rate objection (R1/P2).** Row loads match at every quantile
(byte q10–q99 ratios 1.000/1.000/0.985/1.013/1.051/1.008), so the two arms see the same number of
updates on the same load profile. A single LR is correct here and a sweep would be wasted compute.

### Pre-registered hypotheses

- **H14a** — `byte_shuf` matches `byte` within the seed spread (0.0073 for byte) across all three
  permutation draws → byte structure is **not** load-bearing for LM loss at this scale.
- **H14b** — `char4_shuf` matches `char4` within its seed spread (0.0036) → character structure is
  not load-bearing either.
- **H14c** — `dense_zn` beats `dense` by a material fraction of E6's −3.97% gap → E6's "byte beats
  dense, replicating the paper" was partly a **normalisation artifact**, since that per-token z-norm
  is applied to every Kronecker arm and to no dense arm.

**REFUTATION, and it cuts toward the paper rather than away from it:** if `byte_shuf` is clearly
worse than `byte` across all three draws, **structure IS load-bearing at fixed capacity**, E7's
conclusion is wrong for a second independent reason, and the published codec's premise is
rehabilitated on evidence the paper never produced. Both outcomes are reportable; neither is a
result we are rooting for.

### Novelty
The hashing lineage (Weinberger 2009; Hash Embeddings 2017; Bloom Embeddings 2017; CANINE 2022;
spaCy multi-hash 2022) compares hashed codes against full tables. **Nobody shuffles the
code-to-token map at fixed collision structure and asks whether the structure was load-bearing.**
It is a small methodological contribution and it is precisely the control arXiv:2605.29459 lacks.

### Cut from E10, with reasons
- **`rand479`** — 11.3× effective-rank mismatch; it is the confound we were trying to escape.
- **`hashchar1855`** — serves a claim already withdrawn, and is the one arm that genuinely needs a
  3-point LR sweep, so it costs 3× and settles least.
- **`byte1855`** — and it carried a real bug. `torch.manual_seed` seeds one global stream, and
  `KronEmb.__init__` draws `(D+1)×256` normals *before* the transformer body is built from the same
  stream: D=8192 consumes 2,097,408 draws, D=1855 consumes 475,136. **The bodies would have received
  different initialisations, so H10d would have "failed" for a reason unrelated to pruning.**
  Pruning is provable analytically instead — dead rows are never gathered, contribute zero gradient,
  and never reach the output.

### Free results recovered from the killed E9 (no GPU)
Bucket decomposition of `rand4 − byte`, seed 0, from `009_e9.partial-2of8.json` + `007_e7.json`:

| train-freq bucket | n | % of occurrences | Δ (rand4−byte) | **% of total gap** |
|---|---:|---:|---:|---:|
| 0 | 344 | 0.2 | −0.2502 | 0.4% |
| 1–9 | 4,092 | 2.0 | −0.1106 | 1.9% |
| 10–99 | 19,684 | 9.6 | **+0.0630** | **−5.2%** |
| 100–999 | 62,458 | 30.5 | −0.0864 | 22.5% |
| **1000+** | 117,966 | **57.7** | −0.1632 | **80.4%** |

**80.4% of rand4's entire advantage comes from the 960 types with ≥1000 updates, and rand4 loses in
the 10–99 bucket.** Type-macro-averaged the gap shrinks −0.1171 → −0.0246, a **4.8× reduction**.
Aggregate val loss was measuring head-of-distribution addressability, not codec quality.

**Power analysis of the published typo claim (no GPU).** arXiv:2605.29459 reports 55.5% vs 47.3% on
**110 prompt pairs**: SE ≈ 4.74 pp each, difference SE ≈ 6.72 pp → 8.2 ± 6.7 pp, **z = 1.22,
p = 0.22 — not significant**, before any seed variance. Our harness measures a **4.07 pp seed
spread** on 1,500 pairs (byte 27.80 / 23.73), i.e. seed variance ≈ 3.6× binomial. The paper's
robustness claim is underpowered by roughly an order of magnitude in pairs.

**Val mix does not need rebuilding.** Re-weighting the three languages equally, from per-language
losses we already have, leaves the **rank order identical for all seven arms** (rand4 6.0328 →
5.8339; byte 6.1521 → 5.9804; char4 6.3232 → 6.1767). Rebuilding would cost every prior number and
change no conclusion. Fix the documentation, always report per-language, publish the equal-weighted
column as a derived robustness check.

---

## E15 — The code is left-anchored: absolute position binding is shift-brittle (CPU)

**Hypothesis I formed and then partly refuted.** `κ(b) = Σ_p c_{b_p} ⊗ p_p` is a Smolensky tensor-product
representation with **absolute** position binding. TPRs are known not to generalise across position
shifts. I predicted this would explain the codec's failure to help Indic specifically, because Indic
fertility 3.50 fragments words into pieces whose shared morphemes land at varying within-token offsets.

### E15.1 A single prefix insertion annihilates the code — measured

Code cosine = |A∩B| / √(|A||B|) over (byte, position) sets:

| a | b | shared cells | cosine | kind |
|---|---|---:|---:|---|
| cat | cats | 3 | 0.866 | suffix |
| **cat** | **scat** | **0** | **0.000** | prefix shift |
| walk | walked | 4 | 0.816 | suffix |
| **walk** | **rewalk** | **0** | **0.000** | prefix shift |
| running | runnin | 6 | 0.926 | suffix |
| running | prerunning | 1 | 0.120 | prefix shift |
| ത്ത | ത്തി | 9 | 0.866 | suffix |
| ത്ത | വത്ത | 4 | 0.385 | prefix shift |

**The code is left-anchored.** It generalises over suffix variation (0.82–0.93) and is annihilated by
prefix insertion (0.000). This is structural, not empirical — position is bound absolutely from the
token start.

### E15.2 How often it bites, at a script-fair threshold — measured

Sampled token pairs from the full vocabulary, tokens of 3–12 characters, requiring a longest common
substring of **≥3 characters** (byte threshold scaled per script: 3 for Latin, 9 for Brahmic — the
first pass used a flat 4-byte threshold, which is 4 Latin characters but only 1.3 Brahmic ones and
inflated Indic hit counts ~135×; that comparison was invalid and is superseded here).

| script | hits | aligned n | cos | shifted n | cos | % shifted | **overlap lost** |
|---|---:|---:|---:|---:|---:|---:|---:|
| LATIN | 500 | 178 | 0.482 | 322 | 0.091 | 64.4% | **81.1%** |
| DEVANAGARI | 275 | 161 | 0.762 | 114 | 0.204 | 41.5% | **73.2%** |
| MALAYALAM | 847 | 278 | 0.765 | 569 | 0.191 | 67.2% | **75.0%** |
| TAMIL | 997 | 290 | 0.799 | 707 | 0.199 | 70.9% | **75.1%** |
| KANNADA | 457 | 173 | 0.771 | 284 | 0.174 | 62.1% | **77.4%** |
| TELUGU | 432 | 206 | 0.764 | 226 | 0.182 | 52.3% | **76.2%** |
| BENGALI | 320 | 170 | 0.770 | 150 | 0.187 | 46.9% | **75.7%** |

**Finding that stands:** 42–71% of vocabulary token pairs sharing a ≥3-character substring share it at
a *different* offset, and misalignment destroys **73–81% of the code overlap** those tokens would
otherwise have. The codec therefore captures only about a quarter of the spelling similarity actually
present in the vocabulary.

**My Indic hypothesis is REFUTED.** The penalty is essentially uniform across scripts (73–81%), and
**Latin loses the most (81.1%)**, not least. This is a general property of absolute-position binding,
not an Indic-specific mechanism. Recorded because I designed the measurement expecting the opposite.

A secondary observation runs *in Indic's favour* and should not be suppressed: aligned Brahmic pairs
reach cosine 0.76–0.80 against Latin's 0.482, because a 3-character Brahmic substring is 9 bytes and
therefore a larger fraction of a short token's code.

### E15.3 What it implies

The fix is a **shift-covariant binding** in place of absolute position — character n-gram sets
(shift-invariant by construction, cf. fastText), circular convolution / holographic reduced
representations (Plate), or relative/Fourier position where a shift is a phase rotation. This is a
concrete, mechanistically-motivated redesign of Eq. 1 rather than a hyperparameter change, and it is
falsifiable: if a shift-covariant code does not beat the absolute-position code at matched capacity
under the E14 permutation protocol, the brittleness does not cost LM loss.

### E15.4 Prior art that reframes the whole direction (novelty check, held-out-type line)

The "compositional embeddings tie on the head and win the tail" shape is **published**, with the
coverage confound already removed:

- **Bhatia et al. 2016**, Table 1 (verified from PDF): on in-vocabulary items, word2vec 51.4 vs
  SumEmbed 45.9 vs VarEmbed 51.3 (WordSim-353); Rare Words in-vocab 33.6 / 37.3 / **44.1**.
  Critically, **pure composition loses the head (45.9 vs 51.4); only the hybrid ties it (51.3).**
- **Avraham & Goldberg**, Table 2 (verified): n-gram composition 0.712 / 0.767 vs lookup 0.707 / 0.675
  on common/rare — a tie on the head, a win on the tail. Their conclusion, verbatim: *"This suggests
  using different compositions for common and rare words."*
- **Schick & Schütze (AAAI 2020)**, verbatim: *"one can simply replace BERT's embeddings for rare
  words using AM while keeping its original embeddings for frequent words."* NAACL 2019 puts the
  crossover near 16–32 occurrences.
- Counter-example to handle: **Charagram** Table 6 wins the zero-OOV bin (73.8 vs 71.4).
  **Vania & Lopez**'s reversal is language-specific — Czech shows none.

**The strategic consequence.** The field's standing prescription is **frequency-gated hybridization,
not pure substitution.** arXiv:2605.29459 is a pure-substitution proposal, and so were all of our arms.
A reviewer will ask why we did not gate. Our own bucket decomposition is the crossover measurement:
80.4% of the random-code advantage sits in the ≥1000-occurrence bucket, and the sign reverses at
10–99. **A hybrid arm — lookup rows for the frequent head, codec for the tail — is the obvious next
design and converts the standing objection into a result.**

Caveat before that becomes a claim: the existing bucket numbers compare `rand4` (2,560 live rows)
against `byte` (1,855), so part of the head advantage is capacity, not code type. **E14's permutation
arms give the capacity-matched version of exactly this table**, and the hybrid should be designed off
those numbers, not these.

---

## E16 — Rank of the feature matrix is the whole mechanism (measured)

**Completed** 2026-08-21 · `design/rankall.py` · no GPU, 75 s CPU

**Question.** §R1 replaced our overlap predictor with "live rows" (ρ = −1.0000) but never said *why*
live rows should matter. Live rows is a count of touched columns; it is not a capacity. First
principles: `κ` is **linear** in a fixed sparse binary feature map. Write `S ∈ {0,1}^{V×D}` for the
token→cell indicator. Then the entire embedding table is `E = S W`, so

> **rank(E) ≤ rank(S).** The number of degrees of freedom in a 131,072-row embedding table is
> capped by the rank of a matrix that is fixed before training and never learned.

That is the quantity live rows was proxying. Measured directly on the 60,779 types occurring in
train (`codec_local.npz`, `hash_local.npz`, `train_local.npy`; eigendecomposition of `SᵀS` and of the
frequency-weighted `Sᵀdiag(f)S`, rank tolerance `λ_max·1e-9`):

| arm | D_alloc | reachable | **rank(S)** | eff_rank (freq-wtd) | val loss |
|---|---:|---:|---:|---:|---:|
| **rand4** | 2560 | 2560 | **2560** | 1563.1 | **6.0328** |
| hashchar4 | 2560 | 2520 | 2382 | 189.9 | 6.1220 |
| hashbyte4 | 2560 | 2415 | **1733** | 97.3 | 6.1426 |
| byte *(published)* | 8192 | 1855 | **1733** | 105.4 | 6.1521 |
| char2 | 6144 | 1305 | 1199 | 55.8 | 6.1768 |
| char3 | 2816 | 547 | 444 | 41.8 | 6.2971 |
| char4 | 2560 | 479 | 349 | 29.1 | 6.3232 |

| predictor | Spearman vs val loss (n=7) |
|---|---:|
| rank(S) | **−1.0000** |
| live rows | −1.0000 |
| D_reachable | −1.0000 |
| eff_rank (freq-weighted) | −0.9643 |
| **D_allocated** — the paper's parameter-efficiency axis | **+0.0714** |

**The natural experiment inside the existing data.** `hashbyte4` and the published `byte` grid have
**identical rank, 1733**, while differing in D (2560 vs 8192, 3.2×), in construction (hash vs
byte×position grid) and in reachable columns (2415 vs 1855). Their val losses are 6.1426 and 6.1521,
a gap of **0.0095 nats — inside the seed spread** (0.0115 / 0.0073). Two arms matched on rank and on
nothing else land on top of each other. This was not designed; it was already in `results/008_e8.json`.

**Consequences, stated plainly.**

1. **`D_allocated` does not predict quality (ρ = +0.0714).** The axis the published parameter-count
   argument is built on is uncorrelated with the outcome across our seven arms. Our own withdrawn
   "3.2× fewer parameters" headline was on that axis too.
2. **The mechanism behind E7 is not "random beats structure"; it is that randomisation raised rank
   from 349 to 2560 (7.3×).** Structure per se is not penalised — it is penalised *when it collapses
   rank*, which byte×position structure does because the grid is 66.0% unreachable.
3. **E6/D5's amendment gets a mechanism it previously lacked.** char2 is 0.40% worse than byte, and
   the reason is rank 1199 vs 1733 (−31%), not anything about characters versus bytes.
4. **The published design's real capacity is 1,733 directions for 60,779 live types — 35.1 types per
   degree of freedom.** 59,046 of 60,779 token embeddings are forced linear combinations of the
   others, before any training happens. Frequency-weighted, the effective rank is **105.4**.

**What would refute this.** Two arms matched on rank(S) to within 5% that differ in val loss by more
than the seed spread. E14's frequency-stratified permutation is exactly that test: permuting which
token receives which code preserves the multiset of columns and therefore holds rank almost fixed. If
rank is the whole story, **E14's permuted arms must land on their parents.** If they do not, rank is
necessary but not sufficient and this entry is downgraded to a bound.

**Label.** measured. Superseded by nothing; supersedes §R1's "live rows" framing, which was correct
in ordering but had no mechanism attached.

### E16.1 Capacity is allocated with almost no regard to frequency — measured

Slots per token `L` = capped byte length. Against train frequency, over 60,779 live types:

- Pearson corr(log₁₀ f, L) = **+0.2460**; Spearman = **+0.1333**.

| freq bucket | types | mean L | share of all slot-uses |
|---|---:|---:|---:|
| 1–9 | 32,205 | 5.87 | **1.6%** |
| 10–99 | 20,658 | 6.64 | 10.3% |
| 100–999 | 6,956 | 8.69 | 42.5% |
| ≥1000 | 960 | 6.80 | 45.6% |

53.0% of live types receive 1.6% of the gradient signal. The codec gives every token a budget set by
its **spelling length**, which carries almost no information about how much capacity it needs
(ρ_s = +0.13). This is the quantitative case for a frequency-gated allocation and it is measured on
our own corpus, not imported from Schick & Schütze.

### E16.2 Unicode Brahmic blocks are phoneme-aligned — measured

`unicodedata` over the nine Brahmic blocks (Devanagari U+0900 … Malayalam U+0D00), comparing the
character name at each in-block offset with the script prefix stripped:

- 124 offsets are populated in ≥2 scripts.
- **76 of 124 (61.3%)** carry an *identical* phoneme label across every script that populates them.
- 98 of 124 (79.0%) collapse to ≤2 distinct labels.
- Every core unit is exactly aligned: offset 0x15 = LETTER KA in all 9; 0x2E = LETTER MA in all 9;
  0x3E = VOWEL SIGN AA in all 9; 0x4D = SIGN VIRAMA in all 9; 0x05 = LETTER A in all 9.

So `codepoint & 0x7F` is a **script-invariant phoneme identifier** across the nine Indic scripts, by
Unicode's deliberate ISCII-derived layout. E5 already computes exactly this quantity and calls it
`off_idx`, then throws the invariance away by concatenating it with `blk`. Under the byte codec the
invariance is destroyed outright: Devanagari KA is `E0 A4 95` and Malayalam KA is `E0 B4 95`, which
share one of three byte cells and none of the position-bound ones.

**Label.** measured (Unicode property, reproducible from `unicodedata` alone).

### E16.3 Why each arm has the rank it has — factorisation destroys rank, hashing preserves it

The rank table is not seven unrelated numbers. Given `E = S W`, every arm is a **linear
transformation of the byte×position grid's column space**, and rank behaves accordingly:

| operation | effect on rank | arms | predicted | measured |
|---|---|---|---|---|
| **factorise** a symbol into k sub-symbols (`char` → `blk ⊕ off` → finer) | replaces a product feature with a **sum of marginals**; rank collapses | char2/3/4 | strict decrease | 1199 → 444 → 349 ✓ |
| **hash** the same pairs into D′ > rank (CountSketch, K=4) | random column-merge; preserves rank a.s. when `D′ > rank` | hashbyte4 | **= byte = 1733** | **1733** ✓ |
| **randomise** the index per token | creates independent features; rank saturates at D | rand4 | = D = 2560 | **2560** ✓ |

`hashchar4` = 2382 is consistent once the construction is read: it hashes the **undecomposed**
`(character, position)` pair, not `char2`'s factorised `(blk,pos) ⊕ (off,pos)`. An undecomposed
symbol carries more distinct features than its factorisation, so 2382 > 1199 is expected, not
anomalous.

**This discriminates rank from live rows, which §R1 could not.** The two predictors disagree on
exactly one pair:

- live rows: hashbyte4 (2,415) should beat byte (1,855) by **+30% capacity**.
- rank: hashbyte4 (1733) should **tie** byte (1733) — hashing a rank-1733 space into 2560 buckets
  cannot add information.
- measured: 6.1426 vs 6.1521 = **0.0095 nats, inside the seed spread** (0.0115 / 0.0073). **Tie.**

Rank made the correct prediction where live rows made the wrong one. Live rows is a count of touched
columns; rank is the dimension of what those columns span, and only the second is a capacity.

**A defensible remnant of the withdrawn headline.** We cannot claim hashing improves quality. We
*can* claim, with both a derivation and a measurement, that **hashing the published byte grid
preserves its rank exactly (1733 → 1733) while cutting allocated D by 3.2× (8192 → 2560), at a
measured cost of 0.0095 nats, inside the seed spread.** That is a compression result about the
published design, not a quality result, and it is the form that survives review.

**Cheap falsifier, not yet run.** Hash `char2`'s features into `D′ = 1280` (> its rank 1199). The
theory says loss must match char2's 6.1768 within the seed spread while allocating 4.8× less than the
byte grid. If it lands materially worse, "hashing preserves rank" is false at this scale and E16.3
falls.

---

## E13b — the 32-byte cap binds, and only on Indic (H13b CONFIRMED)

**Completed** 2026-08-21 · `design/e13b.py` → `e13b_results.json` · 60 BPE trainings, CPU only

Supersedes the run recorded as `e13_BROKEN_gpt2regex.json`, which was invalid: it used
`pre_tokenizers.ByteLevel()`, whose GPT-2 regex splits on `\p{L}+`. Indic vowel signs are Unicode
category **Mark**, not Letter, so a Malayalam word was shredded into pieces *before BPE ran* and every
curve was pinned at a regex floor. Measured pre-tokenizer pieces/word, GPT-2 regex vs mark-aware:
en **1.1527 / 1.1593** (0.99×), hi **3.2956 / 1.1165** (2.95×), ml **7.3404 / 1.2608** (**5.82×**).
BrahmicTokenizer's own `tokenizer.json` regex *does* include `\p{M}` — it is the o200k pattern — so
the defect was ours, not the artifact's. E13b uses the artifact's exact pre-tokenizer.

**capped(31 chars) − uncapped fertility**, tokens/word, FLORES devtest:

| V | en | hi | kn | ta | ml |
|---:|---:|---:|---:|---:|---:|
| 4,000 | 0.0000 | 0.0005 | 0.0005 | 0.0123 | 0.0034 |
| 8,000 | 0.0000 | 0.0012 | 0.0080 | 0.0238 | 0.0144 |
| 16,000 | 0.0000 | 0.0009 | 0.0188 | 0.0439 | 0.0338 |
| 32,000 | 0.0000 | 0.0028 | 0.0390 | 0.0692 | 0.0605 |
| 64,000 | 0.0000 | 0.0047 | 0.0663 | 0.1001 | 0.1009 |
| **128,000** | **0.0000** | 0.0058 | 0.0920 | 0.1312 | **0.1402** |

At V=128,000, as a penalty on fertility: **ta +9.08%, ml +8.53%, kn +6.17%, hi +0.49%, en +0.00%.**

- **H13a — CONFIRMED, exactly.** English is 0.0000 at every vocabulary size, to four decimals.
  Uncapped English BPE does learn pieces up to 128 chars; none of them occur in FLORES devtest.
- **H13b — CONFIRMED.** Every Indic language diverges and **the gap grows monotonically with V** at
  every step, with no sign of saturation at 128,000. The cap costs more as the vocabulary grows, not
  less, so this is not an artifact of a small-vocabulary regime.
- **H13c — not evaluated.** Asymptote untested; the curve has not turned over by 128,000.
- **H13d — CONFIRMED.** Divergence orders **ml (0.1402) > ta (0.1312) > kn (0.0920) > hi (0.0058)
  > en (0.0000)** on absolute delta. By percentage the top two swap (ta +9.08% > ml +8.53%); both
  orderings are reported because they answer different questions.

**What this establishes.** The pre-commitment in §E13 was: *"if capped and uncapped Malayalam curves
stay within noise out to 256K, the architecture→tokenizer claim collapses and we drop it."* They do
not. The embedding's 32-byte window is a **binding constraint on the tokenizer**, and it is paid
entirely by Indic: at V=128,000 it costs Tamil 9.08% and Malayalam 8.53% of fertility, and English
nothing at all. This is the causal link E1 could only assert — E1 measured the window's asymmetry in
characters, E13b measures what that asymmetry costs downstream in tokens per word.

**Limits.** One corpus family (IndicCorpV2 prefixes, D4), one eval set (FLORES devtest), BPE only, cap
applied at 31 characters as a proxy for 32 bytes minus the boundary marker. Fertility is not loss;
E13b does not show that the fertility penalty converts to a proportional quality penalty.

**Label.** measured.

---

## E14 — the permutation null fires E16's falsifier: structure IS load-bearing

**Completed** 2026-08-21 · `010_e14_perm.py` → `results/010_e14.json` · 8 runs, M3 Pro MPS

Permute **which token receives which code**, within (train-frequency quantile × slot count) strata.
The code multiset is preserved exactly, so live rows, trained parameters, row-load distribution and
total gradient-touches are identical by construction. **Rank is preserved exactly, not
approximately** — permuting the assignment is a row permutation of `S`, and row permutation cannot
change `rank(S)`. The only thing destroyed is the alignment between code overlap and string overlap.

| arm | draws | val loss (mean) | spread | live rows | parent | **delta** | vs parent seed spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| byte_shuf | 3 | 6.2858 | 0.0040 | 1,859 (byte: 1,855) | byte 6.1521 | **+0.1337** | **18.3×** |
| char4_shuf | 3 | 6.5577 | 0.0154 | 480 (char4: 479) | char4 6.3232 | **+0.2345** | **65.1×** |
| dense_zn | 2 seeds | 5.8326 | 0.0077 | — (15,633,920 params) | — | — | — |

Audit printed by the job itself: for char4_shuf, live rows 480 → 480, total touches
1,416,040 → 1,416,040, load ratio at q[10,25,50,75,90,99] = [1.0]×6. The control is tight and the
tightness is **reported, not asserted**.

### H14a and H14b — both REFUTED, and in the damaging direction

**H14a** predicted byte_shuf ≈ byte, i.e. byte structure not load-bearing. It is off by 18.3 seed
spreads. **H14b** likewise, off by 65.1. Destroying only the code→string alignment, at exactly
matched rank and exactly matched capacity, costs **0.134 nats (byte)** and **0.235 nats (char4)**.

### This refutes E16 as a sufficient explanation — logged 2 h earlier, today

E16's committed falsifier was: *"Two arms matched on rank(S) to within 5% that differ in val loss by
more than the seed spread… If rank is the whole story, E14's permuted arms must land on their
parents. If they do not, rank is necessary but not sufficient and this entry is downgraded to a
bound."* Rank is matched at **0%**, not 5%, and the arms differ by 18–65 seed spreads. **The
falsifier fired. E16 is downgraded as committed.**

What survives of E16, and what does not:

- **Survives.** The derivation `E = S W ⇒ rank(E) ≤ rank(S)` is algebra and is untouched. rank is a
  genuine **capacity ceiling**, it ordered the seven arms at ρ = −1.0000, and it made the correct
  call on the one pair where live rows made the wrong one (hashbyte4 ≡ byte, predicted tie, observed
  0.0095 nats). The E17 collision and floor results are bounds on what the head *cannot* represent
  and do not depend on E16's causal claim at all.
- **Does not survive.** "Structure is penalised only when it collapses rank", and the implied
  reading that rank is the whole mechanism. Both are false.

**The corrected decomposition, measured.** The two effects are comparable, and neither dominates:

| effect | comparison | rank | delta |
|---|---|---|---|
| **capacity** | rand4 vs byte | 2,560 vs 1,733 | **−0.1193 nats** |
| **structure** | byte vs byte_shuf | 1,733 vs 1,733 | **−0.1337 nats** |

### And it overturns E7 / §R1 in the opposite direction

§R1 accepted, from E7, that *"at this scale the byte/character structure is not load-bearing for
language-modelling loss. It is worse than random."* E7's control changed capacity (rand4 trained
2,560 rows against char4's 479) so it could not support that conclusion — §R1 said so. E14 is the
capacity-matched control E7 needed, and it says the **opposite**: byte structure is worth 0.134 nats
at fixed capacity. rand4 beat byte because it had 48% more rank, not because structure is useless.

**Both of our headline positions this session were wrong, in opposite directions, and each was
killed by a control we designed and pre-registered against ourselves.** That sequence — E7 →
§R1 → E16 → E14 — is the result, and it is reported as the result.

### H14c — dense_zn, the normalisation control E6 never had

`dense_zn` applies the same per-token z-norm every Kronecker arm gets, which no previous dense arm
did. Measured 5.8326 (2 seeds, spread 0.0077), at 15,633,920 input parameters:

| vs | codec loss | dense_zn better by |
|---|---:|---:|
| char4 | 6.3232 | 0.4906 nats |
| byte (published) | 6.1521 | **0.3195 nats (5.2%)** |
| hashchar4 | 6.1220 | 0.2894 nats |
| rand4 | 6.0328 | 0.2002 nats |

A dense table beats **every** codec arm we have run, including the best one, by 0.20–0.49 nats — at
7.5× the input parameters of the byte codec. Any claim that the Kronecker input path is free at this
scale is not supported by our own data.

**Limits.** 3 permutation draws at one model seed for the shuffled arms (permutation variance
sampled, init variance not); 2 seeds for dense_zn; 3000 steps, one corpus, one scale, dense output
head throughout. The permutation is stratified on frequency and slot count only — any residual
correlate of those two is not controlled.

**Label.** measured.

---

## E17 — what a headless (codec) output head can and cannot represent at V = 50K / 131K / 1M

**Completed** 2026-08-21 · `design/e17_rank.py`, `e17_run.py`, `e17b_controls.py`, `e17c_final.py`,
`e17d_collision_cost.py`, `e17e_anchors.py` · CPU only, no GPU

**Motivation.** Assignment P5: *"Kronecker is forward deterministic… How do I make a reverse of this?
If we can do this, then we can get rid of the final head as well! Then we can have a vocab of 1M as
well without any issues!"* A codec head computes `logits_v = ⟨φ(v), W h⟩`, so head parameters are
`d·D`, independent of V. Writing `S ∈ {0,1}^{V×D}` for the stacked codes, the achievable logit
vectors are exactly **col-space(S)**. Three exact consequences follow with no trained model:

1. **rank(S)** — at most `rank(S)` independent logit patterns over V tokens. Capped by D, *independent of V*.
2. **collisions** — identical rows of `S` receive identical logits in every context, so they can
   never be separated. This bounds "100% decodable" directly.
3. **unigram floor** — `min_u CE(softmax(S u), p) − H(p)`, convex, solved to optimality.
   Converged: 400 → 4,000 L-BFGS iterations moves it **0.00029 nats (0.04%)**, so `conv=False` is the
   strict gradient criterion, not an unconverged objective.

### E17.1 Scaling family — one real distribution, three sizes, no extrapolation

Vocabulary = the top-V word types from 12 IndicCorpV2 prefixes (6,580,849 distinct types,
83,101,504 tokens), with **measured** frequencies. Same distribution at every V, so the
V-dependence is measured rather than fitted.

| V | codec | D_alloc | rank | codec-induced collisions | floor (nats) | floor (bpb) |
|---:|---|---:|---:|---:|---:|---:|
| 50,257 | byte32 | 8,192 | 1,472 | 336 (0.67%) | 0.8075 | 0.0911 |
| 131,072 | byte32 | 8,192 | 1,998 | 2,303 (1.76%) | 1.0939 | 0.1163 |
| **1,000,000** | **byte32** | 8,192 | 3,559 | **67,876 (6.79%)** | 1.8798 | 0.1835 |
| 1,000,000 | byte64 | 16,384 | 5,030 | 201 (0.02%) | 1.8769 | 0.1832 |
| 1,000,000 | **bigram32** | 16,384 | **7,188** | 67,876 | **1.7266** | **0.1685** |
| 1,000,000 | phon32 | 12,304 | 3,759 | **3** | 1.8834 | 0.1838 |
| 1,000,000 | char32 | 282,688 | 2,772 | 2 | 1.9312 | 0.1885 |

**V grows 20×; rank grows 2.4×; collisions grow 202×.** That is the shape of the answer.

### E17.2 The real artifacts, measured on RAW BYTES

| vocab | codec | D_alloc | D_reach | rank | collisions | floor (bpb) | H (bpb) |
|---|---|---:|---:|---:|---:|---:|---:|
| **brahmic131K** | byte32 *(published)* | 8,192 | 2,789 | 2,457 | **0** | 0.1226 | 1.8539 |
| brahmic131K | byte64 | 16,384 | 2,789 | 2,457 | 0 | 0.1226 | 1.8539 |
| brahmic131K | **bigram32** | 16,384 | **7,063** | **6,716** | 0 | **0.0705** | 1.8539 |
| gpt2_50257 | byte32 | 8,192 | 1,737 | 1,301 | 10 | 0.4399 | 2.4701 |
| gpt2_50257 | byte64 | 16,384 | 1,914 | 1,311 | 1 | 0.4399 | 2.4701 |
| gpt2_50257 | **bigram32** | 16,384 | **3,967** | **3,386** | 10 | **0.3329** | 2.4701 |

`D_reach = 2,789` for brahmic131K/byte32 reproduces `occ2.py`'s independently-measured 2,789
reachable cells exactly.

**Two findings on the shipped artifact, both direct answers to P3:**

- **`byte64` is a pure no-op on BrahmicTokenizer.** D_reach, rank, collisions and floor are
  *identical* to `byte32`, because no token in the vocabulary exceeds 32 bytes. Widening the window
  changes literally nothing. This does not contradict E13b — the tokenizer was **built** under the
  cap, paying 8.53% Malayalam fertility to stay inside it, so the window binds at construction time
  and is slack at inference. "Make the window dynamic" is answering the wrong end of that trade.
- **Adjacent-byte conjunctions are worth far more than window depth.** At *identical* D = 16,384,
  bigram32 gives **2.73× the rank** of byte64 and cuts the unigram floor **42.5%** (0.1226 → 0.0705
  bpb) on BrahmicTokenizer, and **24.3%** on GPT-2. This is axis C: `κ` is linear in `φ`, so
  `emb("ab") = emb(a@1) + emb(b@2)` exactly, and no conjunction is representable at any budget.

### E17.3 Collision cost — a hard bound, unlike the floor

If two tokens share a row of `S` the model must give them equal probability in every context, so
`Σ_G Σ_{v∈G} p_v log(p_v/(p_G/|G|))` is a genuine lower bound on the loss increase — exact, no
optimisation, no model.

| V | codec | groups | ids involved | % of V | cost (nats) | cost (bpb) |
|---:|---|---:|---:|---:|---:|---:|
| 1,000,000 | **byte32** | 25,584 | 93,460 | **9.35%** | 0.0214 | **0.00209** |
| 1,000,000 | byte64 | 154 | 355 | 0.036% | 1.79e-5 | 1.75e-6 |
| 1,000,000 | **phon32** | **3** | 6 | 0.0006% | 1.93e-5 | 1.88e-6 |

**The honest reading, and it cuts both ways.** At V=1M the published byte32 codec leaves **9.35% of
the vocabulary provably indistinguishable**, so *"a vocab of 1M without any issues"* is false as
stated for decodability. But the **loss** cost is only **0.00209 bpb**, because collisions fall on
rare tokens. So the claim fails on the property it is actually about (invertibility) and very nearly
survives on the metric people will quote (bpb). Both halves must be reported.

`phon32` — the E16.2 phoneme basis, `cp & 0x7F` shared across all nine Brahmic scripts — reduces
collisions at V=1M from 67,876 to **3**, beating byte64's 201 at **25% smaller D**. On the 131K BPE
artifact it is roughly a wash (floor 0.1245 vs byte32's 0.1226), so its advantage is a large-V,
word-like-vocabulary effect, not a general one.

### E17.4 Two controls, both of which changed a conclusion

- **C1 — same-string floor.** Tokens sharing a surface string are inseparable by *any* surface
  codec and must be subtracted before blaming a codec. GPT-2 has **333** such ids by decoded string,
  so its apparent 343 collisions are really **10** codec-induced.
- **C2 — index-order confound, and it is enormous.** An index-factored head keys on the integer id.
  With frequency-sorted ids its floor at V=1M is **0.0942 nats**; with ids shuffled, **2.6224** — a
  **+2.53 nat** penalty. Frequency-sorting is a free and legitimate design choice, but it means an
  index-factored head's quality is set by an arbitrary convention while a surface codec's is not.
  On BrahmicTokenizer, whose ids are BPE-merge-ordered, factoredx2's floor is **2.1258** against
  byte32's 0.5672 — nearly 4× worse. **Index-factored heads are fragile in a way surface codecs are not.**

### E17.5 Four wrong token representations, each caught by a check

Recorded because the corrections were the work, not an aside:

1. mixed decoded/byte-level per token — inconsistent across ids;
2. `tk.decode([i])` throughout — **lossy**: 658 BrahmicTokenizer ids decode to `'�'` and 396 to
   `' �'`, because they are partial UTF-8 fragments. They are *distinct byte strings* a byte codec
   separates perfectly; the collapse was the decoder's, not the codec's;
3. counting those 1,496 as "irreducible" — they are not; on raw bytes the artifact has **0** duplicates;
4. **raw bytes via the ByteLevel inverse alphabet** — correct, and used for all E17.2 numbers.

Correction 2 alone would have produced a fabricated 1.1% irreducible-collision claim against the
instructor's tokenizer.

**Caveat on char32/phon32 collisions.** Both decode bytes→str with `errors="replace"`, so they
collapse partial UTF-8 fragments and show 1,141/1,149 collisions on brahmic131K. E5's design avoids
this with two reserved block ids; that fix is **not** implemented here, so those two collision counts
are pessimistic relative to E5 and should not be quoted against it.

**Limits.** Unigram only — the floor is the deficit on constant prediction, and is a diagnostic, not
a proven bound on LM loss (unlike E17.3's collision cost, which is). Word-type vocabularies are a
stand-in for a 1M BPE vocabulary, which we did not train. One corpus family.

**Label.** measured.

---

## E18 — bigram conjunctions: the gain is real, the explanation was wrong (H18d REFUTED)

**Completed** 2026-08-21 · `011_e18_bigram.py` → `results/011_e18.json` · 4 runs, M3 Pro MPS

Codec = published byte cells (256×32) **plus** position-free hashed adjacent-byte bigrams (8,192
buckets). D = 16,384. Rank measured **before** running: byte32 1,733 → bigram **4,459** (2.57×, and
1.74× rand4's 2,560), so H18b was a quantitative point prediction, not a direction.

| arm | runs | val loss | spread | input params |
|---|---:|---:|---:|---:|
| bigram | 2 seeds | **6.0723** | 0.0082 | 4,194,560 |
| bigram_shuf | 2 draws | **6.0732** | 0.0003 | 4,194,560 |
| *byte32 (baseline)* | 2 seeds | 6.1521 | 0.0073 | 2,097,408 |
| *rand4 (baseline)* | 2 seeds | 6.0328 | 0.0029 | 655,616 |
| *dense_zn (baseline)* | 2 seeds | 5.8326 | 0.0077 | 15,633,920 |

Audit: bigram_shuf live rows 4,675 → 4,675, touches 728,742 → 728,742. Capacity preserved exactly.

| | threshold | result | verdict |
|---|---|---|---|
| **H18a** | < 6.1448 | 6.0723 | **PASS** — beats byte32 by 0.0798 nats, 10.9× seed spread |
| **H18b** | < 6.0328 | 6.0723 | **FAIL** — does not beat rand4, despite 74% more rank |
| **H18c** | < 6.0722 | 6.0723 | **FAIL by 0.0001 nats** — closes **24.98%** of the byte→dense gap against a 25% threshold |
| **H18d** | bigram < bigram_shuf − spread | **−0.0008 nats, 0.1× spread** | **REFUTED** |

H18c is recorded as a failure. It missed by one ten-thousandth of a nat and it is not rounded up;
that is the point of naming the threshold in advance.

### H18d is the result, and it refutes the axis-C argument I proposed

Committed in advance: *"if bigram and bigram_shuf agree within the seed spread, then adjacent-byte
conjunctions carry no usable information at this scale, the axis-C argument is wrong, and we report
that the linearity ceiling is not the binding defect."* They agree at **0.1× the spread**.

**The whole 0.0798-nat gain over byte32 is capacity (rank 1,733 → 4,459), not conjunction.**

The contrast with E14 is sharp, and it is the same control applied to two different cell families in
the same codec:

| permuted family | capacity held | cost of destroying the alignment |
|---|---|---:|
| byte cells (E14) | exactly | **+0.1337 nats** (18.3× spread) |
| bigram cells (E18) | exactly | **+0.0008 nats** (0.1× spread) |

**Position-bound single-byte identity carries information; position-free byte-pair identity carries
none beyond the rows it adds.** That is a specific, measured statement, and it was not predictable
from E14 — I expected the opposite.

**Candidate mechanisms, none tested, offered as hypotheses only.** (a) The bigram hash is heavily
collided — only 2,816 of 8,192 buckets are touched — so the "structure" may already be scrambled
before the permutation touches it. (b) Making bigrams position-free to gain shift-invariance may
have removed exactly the positional binding that made the byte family informative. (c) 3,000 steps
at d_model=256 may be too little to exploit conjunctions. Distinguishing (a) from (b) is one cheap
run: a **position-bound** bigram family with the same bucket count.

**What survives.** bigram is now the best *structured* codec we have run (6.0723 vs byte 6.1521),
and it is the second-best arm overall behind rand4 (6.0328) — which remains unbeaten by every
structured codec, at 6.4× fewer input parameters than bigram. rand4 is still not a codec: it keys on
token id, so it has no OOV, spelling or typo behaviour.

**Limits.** 2 seeds / 2 draws, 3,000 steps, one corpus, one scale, dense output head. Only the
bigram family was permuted; byte cells were left intact, so this isolates the bigram contribution
and says nothing new about the byte family beyond E14.

**Label.** measured.

---

## E20 — pre-tokenizer mark-awareness across shipped tokenizers

**Completed** 2026-08-21 · `design/e20_pretok.py`, `e20b_behavior.py`, `e20c_endtoend.py` · CPU, no GPU

**Why this is a question at all.** BPE can never merge across a pre-tokenizer boundary. If the
pre-tokenizer regex splits on `\p{L}+`, Indic vowel signs — Unicode category **Mc/Mn (Mark)**, not
Letter — become separate pieces *before BPE runs*, and no vocabulary can repair it. This is invisible
in every end-to-end fertility comparison, because everyone measures the tokenizer as a whole. We hit
it ourselves: it invalidated E13 and cost Malayalam 5.82×.

FLORES devtest is 36–40% Mark characters by character count for every Brahmic script, and **0.0%**
for English (measured). So the exposure is entirely one-sided before any tokenizer is involved.

### E20.1 A static regex scan is not sufficient — recorded because it nearly ended the audit

Reading `tokenizer.json` and testing each pre-tokenizer pattern for `\p{M}` reports **`\p{M}=False`
for GPT-2** — which looks like a clean bill of health and is meaningless. `ByteLevel`'s GPT-2 regex
is compiled into the Rust implementation and never appears in the serialized config. Only a
**behavioural** test — drive the pre-tokenizer, count what it emits — measures the real thing.

### E20.2 Pre-tokenizer floor, pieces per whitespace-word, FLORES devtest (8,000 words/lang)

The pre-tokenizer alone, before BPE. 1.000 = words left intact.

| tokenizer | en | hi | ta | ml | kn | bn | worst-Indic ÷ en |
|---|---:|---:|---:|---:|---:|---:|---:|
| **tiiuae/falcon-7b** | 1.176 | 4.052 | 7.633 | 8.500 | **19.721** | 5.649 | **16.77×** |
| **gpt2** | 1.161 | 3.329 | 6.701 | 7.287 | 5.987 | 4.676 | **6.28×** |
| **EleutherAI/gpt-neox-20b** | 1.161 | 3.329 | 6.701 | 7.287 | 5.987 | 4.676 | **6.28×** |
| **Qwen/Qwen2.5-0.5B** | 1.175 | 2.540 | 4.303 | 4.563 | 3.992 | 3.170 | **3.88×** |
| bigscience/bloom | 1.122 | 1.092 | 1.163 | 1.167 | 1.143 | 1.116 | 1.04× |
| bert-base-multilingual-cased | 1.174 | 1.116 | 1.211 | 1.226 | 1.172 | 1.173 | 1.04× |
| ai4bharat/IndicBERTv2 | 1.168 | 1.111 | 1.190 | 1.204 | 1.168 | 1.147 | 1.03× |
| **theschoolofai/BrahmicTokenizer-131K** | 1.139 | 1.106 | 1.186 | **1.222** | 1.195 | 1.139 | **1.07×** |
| google/gemma-2-2b · mistralai/Mistral-7B-v0.1 · sarvamai/sarvam-1 · facebook/nllb-200 · xlm-roberta-base | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.00×** |

**The course's own tokenizer is in the clean group (1.07×).** The defect is real but it is not his.

### E20.3 The floor binds — end-to-end validation

A floor only matters if BPE cannot recover it. Measured end-to-end fertility on the same words:

| tokenizer | lang | pre-tok floor | end-to-end | **floor as % of total** |
|---|---|---:|---:|---:|
| falcon-7b | **kn** | 19.571 | 21.069 | **92.9%** |
| falcon-7b | hi | 4.074 | 7.701 | 52.9% |
| gpt2 | ml | 7.247 | 26.032 | 27.8% |
| Qwen2.5 | ml | 4.546 | 11.497 | 39.5% |
| gemma-2-2b | ml | 1.000 | 5.716 | 17.5% |
| sarvam-1 | ml | 1.000 | 2.871 | 34.8% |

**Falcon's Kannada fertility is 92.9% pre-tokenizer.** BPE recovers 7%. For gemma and sarvam the
floor is 1.000, so all of their fertility is vocabulary — improvable by design. That is the
distinction that matters: **a pre-tokenizer floor is architecturally locked; a vocabulary cost is not.**

### E20.4 Falcon's mechanism, isolated stage by stage — and a confirmed prediction

Falcon's pre-tokenizer is `Punctuation → ByteLevel(use_regex=True) → Digits → Split`. On the Kannada
word `ಇಲಿಗಳು` (6 characters):

| after stage | pieces |
|---|---:|
| 1. Punctuation | 1 |
| 2. ByteLevel | 4 ← the `\p{L}+` vs `\p{M}` bug, same as GPT-2 |
| 3. **Digits** | **14** ← a second, different bug |
| 4. Split | 14 |

**Why `Digits` fragments Kannada, and only Kannada.** `Digits` runs *after* `ByteLevel`, so it sees
the byte-level alphabet, where byte `0xB2` → `'²'` and `0xB3` → `'³'` — **superscript digits**.
Kannada is U+0C80–U+0CFF, whose UTF-8 second byte is exactly `0xB2`/`0xB3`. The splitter is matching
Kannada's UTF-8 encoding as if it were digits.

Derived from Unicode alone, **before** looking at the per-language numbers:

| script | UTF-8 second byte → byte-level char | digit-like? |
|---|---|---|
| Devanagari | 0xA4→`¤`, 0xA5→`¥` | no |
| Bengali | 0xA6→`¦`, 0xA7→`§` | no |
| Tamil | 0xAE→`®`, 0xAF→`¯` | no |
| Telugu | 0xB0→`°`, 0xB1→`±` | no |
| **Kannada** | **0xB2→`²`, 0xB3→`³`** | **YES** |
| Malayalam | 0xB4→`´`, 0xB5→`µ` | no |

Prediction: Kannada alone should be catastrophically fragmented. Measured floors: kn **19.571**,
ml 8.433, ta 7.633, bn 5.649, hi 4.074. **Kannada is 2.3× the next worst.** The prediction was
derived from the Unicode layout and confirmed against independently measured data.

### What this buys the V5 plan

- Two shipped-tokenizer defect classes, both Indic-specific, both upstream of everything S7 studies:
  **(i)** `\p{L}+` pre-tokenizers (GPT-2, NeoX, Qwen, Falcon) impose an irreducible 2.5–7.3× floor;
  **(ii)** byte-level-artifact splitting (Falcon `Digits`) adds a further 2.3× on Kannada alone.
- **Any fertility comparison against a GPT-2-lineage tokenizer on Indic text is measuring the
  pre-tokenizer, not the vocabulary.** arXiv:2605.29459 blames Gemma-3's Indic coverage on
  "word pieces that exceed 32 bytes"; the floor decomposition above is the check that claim needs,
  and Gemma's floor is **1.000**, so for Gemma that particular explanation is not pre-tokenizer-driven.
- The published `\p{M}`-aware o200k-style regex is the fix, and BrahmicTokenizer already uses it.

**Limits.** FLORES devtest only; whitespace-word denominator (not morphological words); 6 languages;
`Tokenizer.from_pretrained` snapshot as of 2026-08-21; gated repos (mt5, Llama-3.1, aya-23) not
measured. Pieces/word is not loss.

**Label.** measured.

---

## E19 — position binding is not the answer either; redundant structure is *harmful* (H19b REFUTED by reversal)

**Completed** 2026-08-21 · `012_e19_bigrampos.py` → `results/012_e19.json` · 6 runs, M3 Pro MPS

Identical to E18 in every respect but one term in the hash:
`E18 cell = hash(bᵢ, bᵢ₊₁) % 8192` (position-**free**) vs `E19 cell = hash(bᵢ, bᵢ₊₁, i) % 8192`
(position-**bound**). Same D = 16,384, same byte family, same bucket count, same hash constants.

| arm | runs | val loss | spread |
|---|---:|---:|---:|
| bigram_pos | 3 seeds | **6.0388** | 0.0090 |
| bigram_pos_shuf | 2 draws | **6.0004** | 0.0017 |

Audit: live rows 6,165 → 6,165, touches 728,742 → 728,742, load ratio 1.0 at every quantile.
Permutation preserves rank exactly.

| | threshold | result | verdict |
|---|---|---|---|
| **H19a** | < 6.0641 | 6.0388 | **PASS** — beats bigram_free (6.0723) by 0.0335 |
| **H19b** | Δ_perm > +0.02 | **−0.0383 (−4.3× spread)** | **REFUTED, and the sign is reversed** |
| **H19c** | < 6.0328 | 6.0388 | **FAIL** — rand4 still unbeaten by a deployable codec, by 0.0060 |

### The three permutation experiments together

Same control, three feature families, three different answers:

| family | position | Δ_perm = loss(shuf) − loss(base) | reading |
|---|---|---:|---|
| **byte** (E14) | bound | **+0.1337** (18.3× spread) | structure is load-bearing |
| bigram (E18) | free | +0.0008 (0.1× spread) | structure is inert |
| **bigram** (E19) | bound | **−0.0383** (4.3× spread) | **structure is harmful** |

**Position binding does not explain the E14/E18 split.** It was the only surviving hypothesis after
E18 killed the hash-collision explanation, and it is now refuted in the *opposite* direction from
the prediction. Committed in advance: *"if Δ_perm is again within the seed spread… we report that,
and we stop proposing conjunction repairs."* It was not within the spread — it reversed — so the
conclusion is stronger than the one we pre-committed to, and we stop proposing conjunction repairs.

### What distinguishes the families, offered as the mechanism the data supports

The byte family is the token's **primary** encoding: `(byte, position)` cells are what identify the
token at all. The bigram family is **redundant given it** — a byte-pair is almost determined by the
two byte cells already present. At exactly matched rank and capacity:

- **primary structure** (byte identity × position) — worth **+0.134 nats**;
- **redundant structure** (byte-pair identity, position-free) — worth **0.000 nats**;
- **redundant structure, position-bound** — worth **−0.038 nats**, i.e. it *costs*.

A redundant conjunction feature adds almost no information but does add **ties**: every pair of
tokens sharing that `(byte-pair, position)` is forced to share a row. Randomising the assignment
converts constraint into free independent capacity. Rank counts rows; it does not distinguish a row
that separates tokens from a row that binds them together. **That is the variable neither rank nor
live rows captures, and E19 is the first arm where the distinction is visible in the loss.**

**This is a hypothesis with one supporting experiment, not an established mechanism.** The direct
test is a redundancy sweep — conjunction features of increasing independence from the byte family
(byte-pairs → skip-grams → whole-token hashes) — which we have not run.

### Standings, and an important caveat about the leader

| arm | val loss | deployable codec? |
|---|---:|---|
| dense_zn | 5.8326 | n/a — dense table, 15.6M params |
| **bigram_pos_shuf** | **6.0004** | **no** — permuted assignment is not computable from the string |
| rand4 | 6.0328 | **no** — keys on token id |
| **bigram_pos** | **6.0388** | **yes — best deployable codec, beats published byte by 0.1133 nats** |
| bigram_free | 6.0723 | yes |
| byte *(published)* | 6.1521 | yes |

`bigram_pos_shuf` is the best non-dense arm we have run, and it is **not a codec**: like rand4, its
mapping cannot be computed from a token's bytes, so it has no OOV, spelling or typo behaviour. It is
a capacity control. Reporting it as a codec win would repeat exactly the error §R1 caught.

The defensible statement is narrower and still worth having: **adding position-bound byte-pair cells
to the published codec improves val loss by 0.1133 nats (15.5× the byte seed spread) at 2× the input
parameters — and its own control shows the improvement is capacity, not the conjunction.**

**Limits.** 3 seeds / 2 draws, 3,000 steps, one corpus, one scale, dense output head. Only the
bigram family was permuted. The redundancy mechanism is inferred from three points, not measured.

**Label.** measured.

---

## E22 — statistical hardening: out-of-sample test of rank, and the third variable

**Completed** 2026-08-21 · `design/e22_stats.py` · CPU only

Three objections a reviewer raises against §R1/E16, and what the data now says.

### E22.1 The rank predictor was fitted and evaluated on the same 7 arms — now it has a held-out set

E18 and E19 added five arms **after** E16 was written, so their ordering is a genuine out-of-sample
test that could not have been tuned:

| set | arms | Spearman(rank, val loss) |
|---|---|---:|
| in-sample (E16's original 7) | byte, char2, char3, char4, rand4, hashbyte4, hashchar4 | **−1.0000** |
| **held out** (measured afterwards) | char4_shuf, byte_shuf, bigram_free, bigram_pos, bigram_pos_shuf | **−1.0000** |
| all 12 pooled | | **−0.9301** |

The held-out ordering is perfect, and the pooled figure is *lower* than either subset — which is not
a paradox but the signature of the E16 downgrade: pooling introduces the **exactly-rank-matched
pairs**, where rank must predict a tie and does not get one.

### E22.2 How much does rank actually explain? — 78%, and the residual is structure

`loss ≈ a + b·log₁₀(rank)` over 12 arms at 3,000 steps: **a = 7.2209, b = −0.3236, R² = 0.7798**,
residual SD **0.0784 nats**.

| arm | rank | loss | predicted | residual | code |
|---|---:|---:|---:|---:|---|
| rand4 | 2,560 | 6.0328 | 6.1180 | **−0.0852** | randomised |
| char4 | 349 | 6.3232 | 6.3980 | −0.0748 | intact |
| bigram_pos_shuf | 5,668 | 6.0004 | 6.0062 | −0.0058 | permuted |
| bigram_pos | 5,668 | 6.0388 | 6.0062 | +0.0326 | intact |
| **byte_shuf** | 1,733 | 6.2858 | 6.1728 | **+0.1130** | permuted |
| **char4_shuf** | 349 | 6.5577 | 6.3980 | **+0.1597** | permuted |

The two largest positive residuals are both permuted arms. Adding a structure-intact indicator lifts
R² 0.7798 → **0.8243** with a coefficient of **−0.0683 nats**.

**But that indicator is mis-specified, and the table says so.** rand4 is coded "randomised" yet has
the most *negative* residual, while byte_shuf and char4_shuf are also non-structured and have the
most *positive* ones. A single binary cannot be right. The reason is E22.3.

### E22.3 The third variable: row-load uniformity — measured

rand4 differs from the structured codecs in **three** ways, not one. Frequency-weighted gradient
touches per live row:

| arm | live rows | load Gini | effective rows `exp(H)` | effective ÷ live |
|---|---:|---:|---:|---:|
| **byte** *(published)* | 1,855 | 0.8887 | **200.4** | **10.8%** |
| char2 | 1,305 | 0.8957 | 117.3 | 9.0% |
| char4 | 479 | 0.8460 | 95.5 | 19.9% |
| bigram_pos | 6,150 | 0.8806 | 667.6 | 10.9% |
| **rand4** | 2,560 | **0.2009** | **2,338.5** | **91.4%** |

Gini 0 = uniform load, 1 = all signal on one row. **Every structured codec sits at Gini ≈ 0.85–0.90;
rand4 sits at 0.20.** rand4 is uniform by construction, so it changed rank *and* load uniformity *and*
alignment simultaneously. The permutation controls (E14/E18/E19) hold rank and load **exactly** fixed
— the audits print load ratio 1.0 at every quantile — and vary alignment alone. That is precisely why
they are the right control and rand4 never was, and it now has a number attached rather than an argument.

**A sharper statement of the published design's capacity than E16 had.** The byte codec allocates
8,192 rows, of which 1,855 are live, of which **≈200 are effective**. That is **2.4% of allocation**
carrying the signal. Two independent measures agree on the order: load entropy `exp(H) = 200.4`, and
E16's frequency-weighted Gram effective rank `= 105.4`.

### E22.4 The model-free evidence — three exactly-rank-matched pairs

No regression, no covariates; rank and load are identical within each pair by construction:

| pair | rank (both) | Δ = loss(permuted) − loss(structured) |
|---|---:|---:|
| char4 vs char4_shuf | 349 | **+0.2345** |
| byte vs byte_shuf | 1,733 | **+0.1337** |
| bigram_pos vs bigram_pos_shuf | 5,668 | **−0.0384** |

Two positive, one negative. **"Structure helps" is not a general law** — it holds for the primary
byte/character family and reverses for the redundant bigram family, which is E19's conclusion stated
without a model.

**Limits.** 12 arms, one scale (3,000 steps), one corpus, 2–3 seeds each; the regression is
descriptive, not causal, and n=12 cannot support more than two covariates. Load Gini is computed on
train frequencies over live rows and ignores the optimiser's per-row adaptivity (AdamW normalises
per-parameter, which plausibly compresses the effect of load skew — untested).

**Label.** measured.

---

## CORRECTION C1 — two rank conventions were mixed; the verifier caught it on its first run

**Found** 2026-08-21 by `design/verify.py`, which recomputes headline numbers from
`design/data/` and exits non-zero on disagreement. It disagreed on `rank(S)` for `bigram_pos`:
recomputed **5,653**, logged **5,668**.

**Cause.** Two conventions for building `S` were used in different scripts:

- **accumulate** — a cell occurring twice in one token gets entry 2. Used by `rankall.py` (E16)
  and by `verify.py`.
- **dedup** — `S` is a binary indicator, `sorted(set(cells))`. Used by `e17_rank.py` (E17) and by
  the ad-hoc script that produced E18/E19's pre-registered rank predictions.

**Which is correct.** `KronEmb.forward` computes `W[idx].sum(1)` — it **sums** over the 32/63 slots,
so a repeated cell genuinely contributes twice. **Accumulate is the convention that matches the
trained model.** Dedup describes a model we did not train.

**Which numbers move.** Only arms containing a **position-free** feature family, where the same cell
can recur within one token. For `byte`/`char`, a cell is `symbol×P + position` and is unique per slot,
so the two conventions agree exactly.

| arm | dedup (as logged) | **accumulate (correct)** | error |
|---|---:|---:|---:|
| byte, char2, char3, char4, hashbyte4, hashchar4, rand4 | — | **unchanged** | 0 |
| bigram_free (E18) | 4,459 | **4,434** | −0.56% |
| bigram_pos, bigram_pos_shuf (E19) | 5,668 | **5,653** | −0.26% |

**What it changes: nothing that was concluded.** Re-fitting E22 with corrected ranks:

| | dedup (as logged) | accumulate (correct) |
|---|---|---|
| `loss ≈ a + b·log₁₀(rank)` | a=7.2209, b=−0.3236 | a=7.2221, b=−0.3240 |
| R² | 0.7798 | **0.7801** |
| residual SD | 0.0784 | 0.0783 |
| pooled Spearman | −0.9301 | **−0.9301** |

- **E19's central claim is untouched**: `bigram_pos` and `bigram_pos_shuf` still share an *identical*
  rank (5,653 = 5,653), because a permutation permutes rows of `S` and cannot change its rank under
  either convention. The 0.0384-nat gap at identical rank stands.
- E18's "2.57× byte32's rank" becomes **2.56×**; "1.74× rand4" becomes **1.73×**.
- E19's "27% more rank than bigram_free" becomes **27.5%**.

**Superseded values.** E16.3, E17.2, E18 and E19 as written quote the dedup ranks for the bigram
arms; read 4,434 and 5,653. **E17's `bigram32` and `phon32` rows use dedup throughout** (`phon32`
also has a position-free family — the 16-way script tag — so it is affected too); their ranks are
high by well under 1%, and no ordering in the E17 tables changes. E17's `byte32`/`byte64`/`char32`
rows are unaffected.

**Process note, which is the reason to record this at all.** The verifier was written to satisfy a
rubric line about exiting non-zero on invariant failure. It found a real inconsistency in our own
published numbers on its first execution, in a quantity we had used for a *pre-registered
prediction*. A verifier that only ever confirms what you already believe is not doing anything;
this one paid for itself immediately.

**Label.** measured.

---

## CORRECTION C2 — E21's intermediate checkpoints are a trajectory, not replications

**Found** 2026-08-21, on E21's first completed run, by the replication check built into the analysis.

`013_e21a_byte.py`'s docstring claimed: *"The 3,000-step point also REPLICATES the existing
E7/E14/E19 numbers on an independent set of seeds."* **That is wrong, and the first run showed it:**
`byte` seed 0 reads **5.9831** at step 3,000 of a 9,000-step run, against the published **6.1521**
from a 3,000-step run — a gap of **−0.1690 nats**.

**Cause.** The LR schedule is a cosine over `STEPS`, so the same step number sees a different
learning rate depending on the run length:

| step | LR in a 3,000-step run | LR in a 9,000-step run | ratio |
|---:|---:|---:|---:|
| 1,000 | 2.325e-04 | 2.919e-04 | 1.26× |
| 2,000 | 9.750e-05 | 2.684e-04 | 2.75× |
| 2,999 | 3.000e-05 | 2.325e-04 | **7.75×** |

At step 3,000 the 3,000-step run has fully annealed to `0.1·LR`; the 9,000-step run has annealed only
to `cos(π/3)`. They are not the same experiment at that point. The −0.169 nats is a **schedule
effect**, not a discrepancy between the arms.

**What remains valid, and it is the part E21 was for.**

1. **Cross-arm comparison at a matched checkpoint is exact.** Every arm in jobs 013–016 uses the
   identical schedule and identical step count, so `Δ_perm(step)` — the quantity that answers "does
   the structure advantage shrink with training?" — is a clean comparison at every checkpoint.
2. **The two-scale comparison is between fully-annealed endpoints:** 3,000 steps (published table)
   and 9,000 steps (E21). Both are annealed to `0.1·LR` at their own horizon, so those two points are
   comparable to each other.
3. **The intermediate checkpoints are a within-schedule trajectory.** They show the *shape* of
   `Δ_perm` against training, which is exactly what the arXiv:2605.09751 parity objection is about.

**What is now not available.** A same-schedule replication of the 3,000-step numbers on fresh seeds.
That would need separate 3,000-step runs, which we did not queue. The published 3,000-step table
therefore still rests on its original 2 seeds per arm, and this is stated as a limit rather than
papered over.

**Process note.** The replication check that caught this was three lines in the analysis, comparing a
new checkpoint against an already-published number. It cost nothing and it prevented a claim of
replication that the data does not support.

**Label.** measured.

---

## CORRECTION C3 — "rank(E) ≤ rank(S)" is true but non-binding; the right statement is the dimension of the family

**Found** 2026-08-21 during an adversarial self-review of the README, before submission.

E16 and the README's mechanism section state the consequence of `E = S W` as **`rank(E) ≤ rank(S)`**,
and call it the reason a 131,072-row table's degrees of freedom are capped. The inequality is true.
**It is also non-binding at our scale, and would be the first thing a reviewer attacks.**

`rank(E) ≤ min(rank(S), d_model)`. With `d_model = 256` and **every** arm having `rank(S) > 256`
(the smallest is char4 at 349), the binding term is always `d_model`. Verified directly: for a
4,000-token submatrix with `rank(S) = 639`, a random `W` gives **`rank(E) = 256`**, not 639.

**The correct statement, which is what rank(S) actually measures.** `E = S W` ranges over a *linear
subspace* of `R^{V×d}`, and that subspace has dimension

> **`rank(S) · d_model`** — the number of independent knobs the embedding table has.

That is the quantity that varies across arms, and it is what correlates with loss:

| arm | rank(S) | free parameters `rank·d` | as % of a dense table's freedom |
|---|---:|---:|---:|
| char4 | 349 | 89,344 | **0.6%** |
| char2 | 1,199 | 306,944 | 2.0% |
| **byte** *(published)* | 1,733 | **443,648** | **2.8%** |
| hashchar4 | 2,382 | 609,792 | 3.9% |
| rand4 | 2,560 | 655,360 | 4.2% |
| bigram_free | 4,434 | 1,135,104 | 7.3% |
| bigram_pos | 5,653 | 1,447,168 | 9.3% |
| dense | 61,070 | 15,633,920 | 100% |

**The published codec confines the embedding table to a subspace with 2.8% of the freedom of a dense
table** — and *that* is the sentence the paper's parameter-efficiency argument should be answering.
It also states the allocated-vs-effective gap sharply: 2,097,408 allocated parameters against
**443,648** independent directions, a factor of **4.7×** of pure redundancy inside `W` itself.

**Does anything empirical change?** No. `rank(S)` is unchanged, every ordering is unchanged, E22's
regression is unchanged (it regresses on `log₁₀ rank`, and `rank·d` differs by a constant factor, so
the fit is identical up to the intercept). Only the *mechanism sentence* changes — from a bound that
does not bite to the quantity that does.

**Why this is recorded rather than quietly edited.** The claim as written would have survived a
casual read and failed a careful one. It is exactly the class of error this session has been
generating and catching: a true statement doing work it cannot support.

**Label.** measured (the `rank(E) = 256` check is empirical; the subspace-dimension identity is algebra).

---

## E23 — is our reimplementation faithful to the published reference?

**Completed** 2026-08-21 · `design/e23_fidelity.py` → `e23_fidelity.json` · CPU

S7 reimplements the codec from the paper's description (ARCHITECTURE D2) and has never been checked
against the reference implementation, which ships on PyPI as `kronecker-embeddings` (Apache 2.0,
v0.1.1). **That is the first thing a reviewer asks**, and until now we could not answer it.

### E23.1 The codec matches to float32 precision

3,000 tokens sampled from `BrahmicTokenizer-131K`, encoded to raw bytes, compared against
`kronecker_embeddings.kronecker_codec(..., length_normalize=True, z_normalize=True)`:

| | |
|---|---:|
| max abs difference | **4.211e-06** |
| cosine similarity, min / mean | **0.9999997479** / 0.9999999720 |

The reference's own docstring states the storage convention as `index = byte_value * pos_dim + pos`
— **exactly** the convention every `S`-matrix in S7 is built on. Index layout, `1/√L` and z-norm all
agree.

**A convention bug in the check itself, worth recording.** The first run reported max abs error
**5.5e-03** at cosine 0.9999997 — a scale discrepancy, not a direction one. Cause: `torch.std`
defaults to **ddof=1** (sample) and `numpy.std` to **ddof=0** (population); over `D=8192` the ratio
is `√(8192/8191) = 1.000061`, which is invisible in cosine and shows up in the largest components.
The reference and our `KronEmb` both use torch, so ddof=1 is the correct convention and the
discrepancy was in the verification script, not in either implementation. Fixed: 5.5e-03 → 4.2e-06.

### E23.2 Our one deviation, now quantified rather than asserted

ARCHITECTURE D2 records that we z-normalise **after** projection across `d_model`, while the paper
and the reference z-normalise **before** projection across `D`. Measured, with a shared random `W`:

- resulting embeddings differ at cosine **min 0.9679, mean 0.9978**.

So the two are genuinely different maps — S7's training runs replicate a **variant**, exactly as D2
states — but they are far closer than the deviation's description suggests. This is now a number
instead of a caveat.

### E23.3 The deviation does not invalidate the rank analysis

`rank(S)` is computed on the codec **before** either normalisation, so in principle neither placement
touches it. Checked anyway, since z-norm subtracts a per-row mean:

| | rank |
|---|---:|
| raw `S` (with `1/√L` only) | 1,060 |
| z-normalised `S` | **1,060** (difference **+0**) |

z-norm is `x ↦ (x − mean(x))/std(x)`: a per-row rescale, which is rank-preserving, plus subtraction
of a rank-1 all-ones component, so rank can move by at most 1 and here moves by 0. **Every rank
number in S7 is valid for the published variant as well as ours.**

**Limits.** 3,000 of 131,072 tokens; reference v0.1.1 as of 2026-08-21; the comparison covers the
codec function, not the reference's `KroneckerEmbedding` module, its training loop, or any
hyperparameter.

**Label.** measured.

---

## E24 / CORRECTION C4 — the redundancy mechanism is refuted, and E19's control leaks 16%

**Completed** 2026-08-21 · `design/e24_redundancy.py`, `e24b_spectrum.py` · CPU

E19 proposed that a conjunction family **redundant given the byte family** adds no information but
adds ties, so permuting it helps. That was inferred from three points, and redundancy was never
measured. It is measurable without training.

### E24.1 The redundancy hypothesis is REFUTED

`redundancy = 1 − ΔR_F / R_F`, where `R_F` is the family's own rank and `ΔR_F = rank(byte+F) −
rank(byte)` is what it adds on top of the byte family. 1.0 means the family adds nothing new.

| family | rank(F) | rank(byte+F) | ΔR | **redundancy** |
|---|---:|---:|---:|---:|
| bigram gap-1 **position-bound** | 4,220 | 5,653 | 3,920 | **0.0711** |
| bigram gap-1 position-free | 2,784 | 4,434 | 2,701 | **0.0298** |
| skip gap-2 position-bound | 3,696 | 5,020 | 3,287 | 0.1107 |
| skip gap-4 position-bound | 3,028 | 4,330 | 2,597 | 0.1423 |
| skip gap-8 position-bound | 1,958 | 3,215 | 1,482 | 0.2431 |
| whole-token hash | 8,191 | 9,922 | 8,189 | 0.0002 |

**Every one of these families is nearly non-redundant** — the bigram family contributes 3,920 of its
own 4,220 directions on top of byte. "Redundant given the byte family" is simply **false** as a
description of it, so E19's proposed mechanism does not survive its first measurement. The word was
wrong; the phenomenon it was invented to explain is still unexplained.

*(The one ordering that does survive: position-bound has 2.4× the redundancy of position-free
(0.0711 vs 0.0298), and position-bound is the harmful one. Two points. Not a result.)*

### E24.2 CORRECTION C4 — E19's control is **not** rank-matched, unlike E14's

E19 and the README state that its permutation preserves rank exactly, "because a permutation permutes
rows of `S`". **That is true of E14 and false of E19**, and the difference is in the control's own code:

- **E14**: `shuf(idx, D, seed)` with `cols=None` → `new[ix] = idx[perm]`. A **whole-row** permutation.
  Rank preserved exactly — measured, byte 1,733 and byte_shuf 1,733.
- **E19**: `shuf(idx, D, seed, cols=BIGCOLS)` → only the **bigram block** of each row is permuted,
  with the byte block held fixed. This is **not** a row permutation of `S`. Recombining one token's
  byte block with another's bigram block creates novel combinations that span more dimensions.

Measured: `bigram_pos` rank **5,650**, `bigram_pos_shuf` rank **5,902 / 5,900** — **+4.46%**. The
job's audit was not wrong; it reported live rows and touches, which *are* preserved. It simply did
not report rank, so the leak was invisible.

**How much of E19's effect does the leak explain?** Using E22's fitted slope (−0.3240 nats per decade
of rank):

| | |
|---|---:|
| predicted from the rank gain alone | **−0.0061 nats** |
| observed `bigram_pos_shuf` − `bigram_pos` | **−0.0383 nats** |
| **share explained by the leak** | **16%** |
| share remaining | **84%** |

The frequency-weighted effective rank also improves, 221.4 → 298.8 (**+35.0%**), and the Gram
condition number falls 9.81e8 → 1.41e8. Both are side effects the control was intended to exclude.

**What E19 can still claim.** Permuting the position-bound bigram assignment improves loss by
0.0383 nats, of which **≈16% is a measured capacity leak and ≈84% is not**. So the *sign* stands —
destroying that alignment does not hurt and plausibly helps — but the clean statement "at exactly
matched rank" belongs to **E14 only**. E19's contrast with E14 is weakened, not eliminated: E14
remains an exact control showing +0.1337 nats, and E19 shows the opposite sign with a small
quantified leak.

**The fix, not run.** Permute whole rows of the *combined* code (byte block and bigram block
together), which is a true row permutation and preserves rank exactly. That measures something
slightly different — it destroys byte alignment too — so the clean design is to permute the bigram
block **within a rank-matched constraint**, which we have not implemented.

**Label.** measured.

**Process note.** This was found by measuring the mechanism E19 asserted, rather than by re-reading
E19. Two claims died: the redundancy story, and "exactly matched rank" for E19's control. Both had
been committed, and both are corrected here rather than edited away.
