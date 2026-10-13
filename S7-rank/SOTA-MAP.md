# S7 SOTA map — surveyed 2026-08-21

## 0. Supply corrections (both of my earlier claims were wrong)

| earlier claim | actual |
|---|---|
| "The released Kronecker module is not distributed." | **False.** `pip install kronecker-embeddings`, Apache 2.0, `theschoolofai/kronecker-embeddings` — core impl, 4 probe scripts, 3 integration examples, tests, CI. API: `KroneckerEmbedding(vocab_size, d_model, tokenizer, pos_dim, mode="dynamic"|"cached")`. |
| "No real vocabulary available." | **Partly false.** V5's own vocabulary doesn't exist, but the paper evaluates against six real tokenizers (incl. Gemma-3 multilingual). Real multilingual tokenizers are obtainable; only "the V5 vocabulary" specifically is not. |

**The paper**: Rohan Shravan, *Kronecker Embeddings: Byte-Level Structured Token Representations for
Parameter-Efficient Language Models*, arXiv:2605.29459, 28 May 2026.

Codec, verbatim Eq. 1: `κ(b) = (1/√L) Σ_{p=1..L} c_{b_p} ⊗ p_p`, `d_c = 256`, `D = 256·d_p`,
production `d_p = 32`, then per-token z-normalisation. Results: **−2.5 ± 0.2% val loss vs BPE-tied**
at GPT-2 124M / 2.5B FineWeb-Edu (3 seeds), **~1.43× fewer steps** to convergence, typo top-1
preservation **55.5% vs 47.3%**. `d_p=32` covers ≥99.82–99.86% of tokens across six tokenizers.

**This contradicts the transcript**, where he says he *"was not able to finally prove this"* and
*"couldn't publish it"*. The lesson prose is right that it was published; the transcript's
inconclusiveness refers to the unproven *extensions*, not the core result. Precedence rule applies.

## 1. The author's own open list — quote it in the README

Limitations (verbatim): byte-similar/semantically-distant clustering (`compute`/`commute`);
position-aware encoding weakens on suffix-only families; **weight tying inapplicable**; **truncation
discards bytes for tokens longer than `d_p`**; two operational variants; OOD byte sequences; **no
benchmark against modern byte-level / character-aware baselines**; `D` may need to grow with `d_model`.

Future work (verbatim): Scale · Three-arm comparison (untied-BPE) · **Multilingual evaluation** ·
Long-context · Downstream tasks · Layered probe at 1B+ · **Token-bucket NLL analysis: per-frequency-
bucket training dynamics** · Downstream eval · **Multi-modality** · **Output-side Kronecker and
unbounded effective vocabulary**.

Mapping to the assignment: future-work item 9 **is** problem 2; item 10 **is** problem 5; item 3
(multilingual) is where problem 3 bites. The paper confirms it contains **no discussion of Zipfian
training dynamics, rare-token optimisation, or frequency-dependent behaviour**.

## 2. Per-problem SOTA and contribution room

### P1 — mathematical structure · room: LOW
- **Adele-space embeddings**, arXiv:2606.14108 (Jun 2026): real + p-adic components, exact
  coordinate-wise `A(q1+q2)=A(q1)⊕A(q2)`, `A(q1·q2)=A(q1)⊗A(q2)`; first perfect accuracy on Weaving
  Pattern. This *is* problem 1, solved, two months ago.
- **BitTokens**, arXiv:2510.06824: IEEE-754 single-token numbers.
- Abacus embeddings (*Transformers Can Do Arithmetic with the Right Embeddings*).
- Verdict: we would be reproducing June-2026 work with less compute.

### P2 — multimodal · room: LOW (for us)
- MM-Tokenizer / VQ-RVQ family; **Fusion Embedding**, arXiv:2607.18666; EntangleCodec 2606.02739;
  AR-Omni 2601.17761; Gemini Embedding 2 (commercial, 2026).
- Verdict: industrial-scale field; CPU-only cannot contribute.

### P3 — dynamic byte budget · room: **MEDIUM-HIGH**
- **BLT**, arXiv:2412.09871 (ACL 2025): entropy-based dynamic byte patches, FLOP-controlled to 8B/4T.
- MrT5 2410.20771 (dynamic token merging); Length-MAX 2511.20849; entropy pre-tokenisation 2506.15889.
- **Indic-specific, all mid-2026**: *The Tokenizer Tax* arXiv:2607.24276 — Indian languages pay
  **8.0× vs English under cl100k, 13.0× for Malayalam**, effective context down to **12%**;
  **IndicSuperTokenizer** (OpenReview CSrGFB070m) — +39.5% fertility over LLaMA-4, +18% over Sutra;
  **BHARATI** arXiv:2607.23319; ML-Embed 2605.15081.
- Gap: every one of these is about **tokenizer fertility**. None is about the **codec window inside a
  Kronecker embedding**. The author's own limitation ("truncation discards bytes beyond `d_p`") and
  future-work item ("multilingual evaluation") are both unclosed.

### P4 — Fourier · room: MEDIUM-LOW
- Mature VSA/HRR literature: circular-convolution binding, bundling, FHRR/GHRR, *Learning with HRR*.
  The anagram/order failure is textbook binding-vs-bundling — rediscovering it is not a contribution.
- Fourier features: learnable Fourier features 2106.02795; geometry of positional encodings 2604.05217.
- Verdict: the honest version of P4 is "this is HRR", which is a good README and a small result.

### P5 — invertibility · room: MEDIUM, but the ground shifted
- **Language Models are Injective and Hence Invertible**, arXiv:2510.15511, **ICLR 2026** + **SipIt**
  (github giorgosnikolaou/SIPIT): transformer LMs are almost-surely injective, proven and confirmed
  over billions of collision tests on six models with **zero collisions**; SipIt reconstructs exact
  input text from hidden activations in **linear time**.
- Embedding inversion via conditional masked diffusion, arXiv:2602.11047: 81.3% token accuracy,
  8 forward passes, 78M model.
- **UTF-8 Plumbing**, arXiv:2511.05578: byte-level tokenisers *unavoidably* let LMs emit ill-formed
  UTF-8 — a hard obstacle for any byte-grid output head.
- SpeLLM 2507.16323 (character-level multi-head decoding); Cut Your Losses 2411.09009; adaptive/
  hierarchical softmax.
- Consequence: "is it invertible?" is answered — yes, provably. The live question is decode **cost**
  and whether an LM can *emit* a valid code, which is where UTF-8 Plumbing bites.

## 3. The frequency lens — and its honest novelty boundary

The lesson's §2 emotional centre (Zipf → rare rows starve → Indic starves) has a strong 2026 literature:

- **TIDE**, arXiv:2605.06216 (May 2026): bounds cumulative squared gradient norm as
  `E[Σ‖∇_{e_v}L‖²] ≤ τ·f_v·B·T·G²`; hapax ≈ **1,660** expected updates vs common ≈ **1.66×10⁹** —
  **six orders of magnitude**, matching the lesson's "five or six". Fixes it by *adding* K memory
  blocks (more parameters).
- **Leviathan**, arXiv:2601.22040 (Jan 2026): *already ran the frequency-bucket experiment* for
  parameter-shared codes. At 400M, vs Tied: **+0.68 nats (49% ppl reduction) for tokens seen 1–9
  times**, 0.50 for 10–99, 0.25 for 100–999, 0.08 for 1k–10k. Explicitly: *"the benefit of parameter
  sharing should be largest for rare tokens."*
- Also: *Rare Tokens Degenerate All Tokens* 2109.03127; *Exploiting Vocabulary Frequency Imbalance*
  2508.15390.

**So the mechanism is NOT novel.** Parameter sharing helps the rare tail, quantified at 400M in
January 2026. Claiming it as our discovery would be false and a reviewer would find Leviathan in one
search. What remains genuinely open:

1. It has never been measured **for the Kronecker byte codec** — the author lists exactly this as
   future work ("token-bucket NLL analysis").
2. Nobody has measured **the interaction**: the shared projection rescues the rare tail, while the
   `d_p=32` truncation destroys precisely the rare *long* tokens. In Indic these are the same tokens —
   3 bytes/char, conjuncts costing 9 bytes each, and an 8.0–13.0× tokenizer tax. Two mechanisms
   pulling opposite directions on one population, and the net sign is unmeasured.

That interaction is the contribution. It lands on **problem 3**, it is on the author's open list twice
over, it is India-first, and it is CPU-measurable because the effect is a *ranking over frequency
buckets*, not an absolute benchmark number.

## 4. Falsifiers for the proposed claim

- If Kronecker's per-bucket NLL advantage over dense is **flat across frequency deciles**, the sharing
  story is dead (Leviathan predicts a decaying gradient; a flat result refutes transfer to this codec).
- If, after removing truncated tokens from the evaluation, the Indic gap **does not close**, then
  truncation is not the mechanism and our diagnosis is wrong.
- If a widened window (`d_p=48/64`) buys **no bucket-wise improvement** at matched parameters, the
  proposed fix is not a fix.
- If collisions at `d_p=32` on a real multilingual tokenizer are **below ~0.2%** as the paper's
  99.82% coverage implies, the problem is too small to matter and we should say so and stop.

## 5. Sources
arXiv: 2605.29459 · 2412.09871 · 2410.20771 · 2511.20849 · 2511.05578 · 2506.15889 · 2601.22040 ·
2605.06216 · 2109.03127 · 2508.15390 · 2510.15511 · 2602.11047 · 2606.14108 · 2510.06824 ·
2607.18666 · 2607.24276 · 2607.23319 · 2605.15081 · 2604.05217 · 2106.02795 · 2507.16323 · 2411.09009
Repos: theschoolofai/kronecker-embeddings · giorgosnikolaou/SIPIT
OpenReview: IndicSuperTokenizer CSrGFB070m · Injectivity 0kHbD6ad07 · Learning with HRR RX6PrcpXP-
