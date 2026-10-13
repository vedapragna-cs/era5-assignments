# S7 architecture and decisions

Durable decisions only. Each names the evidence that produced it and what would reverse it.

## D1 — Problem choice: #3, the byte budget, widened to vocabulary allocation

**Decision.** Of the five assignment problems we take **#3** (*"Kronecker is limiting to presenting 32
positions for every word… how can it be dynamic"*), reframed as: given a fixed 131,072-slot budget and
a byte-window constraint, how should slots be allocated across scripts, and how close is the shipped
retrofit?

**Why not the others.**
- **#1 mathematical structure** — superseded. Adele-space embeddings (arXiv:2606.14108, Jun 2026)
  already give exact coordinate-wise `A(q₁+q₂)=A(q₁)⊕A(q₂)` and `A(q₁·q₂)=A(q₁)⊗A(q₂)`; plus
  BitTokens (2510.06824) and Abacus. We would be reproducing two-month-old work with less compute.
- **#2 multimodal** — out of compute reach. MM-Tokenizer/VQ-RVQ, Fusion Embedding (2607.18666),
  Gemini Embedding 2. This box has no GPU (aarch64 Linux guest on an M3; Metal is macOS-only and
  cannot be exposed to a Linux guest).
- **#4 Fourier** — mature VSA/HRR literature, where the anagram failure of order-free wave sums is the
  textbook binding-vs-bundling distinction. Decisively: **he attempted it himself and abandoned it**
  (*"my first idea was Fourier… I couldn't solve that so I ended up with Kronecker"*), so he holds
  unpublished negative results we cannot see.
- **#5 invertibility** — highest ceiling and he is publicly stuck, but the ground moved: *Language
  Models are Injective and Hence Invertible* (arXiv:2510.15511, ICLR 2026) proves almost-sure
  injectivity with zero collisions across billions of tests, and SipIt inverts exactly in linear time.
  What remains is decode cost, against UTF-8 Plumbing's (2511.05578) result that byte-level tokenisers
  *unavoidably* permit ill-formed UTF-8. A real bet with a real chance of yielding only a negative.

**Reverses if.** The allocation optimum turns out within a few percent of the shipped allocation, or
marginal-value curves are flat enough that allocation barely moves the objective. Either outcome makes
#5 the better use of remaining time.

## D2 — Evaluate against his artifacts, not proxies

**Decision.** Vocabulary is `theschoolofai/BrahmicTokenizer-131K`; codec is the published
`κ(b) = (1/√L) Σ_p c_{b_p} ⊗ p_p` with `d_c=256`, `d_p=32`, z-normalised (arXiv:2605.29459 Eq. 1);
reference implementation is `pip install kronecker-embeddings` (Apache 2.0).

**Why.** Auditing his tokenizer against his codec tests his stated design on his own artifacts.
Gemma-3 was the alternative — his paper benchmarks it and blames "Indic word pieces that exceed 32
bytes" for its coverage drop — but it is a proxy for a vocabulary we can obtain directly.

**Consequence.** The earlier supply concern is void: the V5 vocabulary class *does* exist publicly.
Only V5's own future corpus does not.

## D3 — The fertility floor is a constraint, not a headline

**Decision.** `floor32_L` (E2) enters the allocation optimisation as a lower bound on
`fertility_L(V_L)`, rather than being reported as a standalone finding.

**Why.** E2 confirmed the floor exceeds 1.0 for every Indic script (Malayalam 1.3295) and equals
exactly 1.0 for English, refuting the universal floor of 1.0 that lesson §4 states and widget 4's
formula implies. But E2 also **refuted our own H2b**: `actual` fertility sits far above `floor32`
everywhere, so vocabulary — not the window — is what binds today. The floor's honest role is to cap
the marginal value of slots for exactly the scripts that need them most, which is an interaction
neither result shows alone.

## D4 — Corpus: IndicCorpV2 prefixes, FLORES for evaluation

**Decision.** BPE training corpora are byte-range prefixes of `ai4bharat/IndicCorpV2` (~120 MB per
language, stored gzipped); evaluation is FLORES-200 devtest.

**Why.** IndicCorpV2 ships one plain-text file per language, which is the cleanest input for BPE
training. Full files are 1–40 GB against 2.8 GB of free disk here, so prefixes are a supply-driven
compromise and must be declared as such: **the corpora are prefixes, not samples**, so any
domain skew in the file head is inherited. FLORES is gated; access is via the user's HF token, stored
box-private and never written to `shared`.

**Reverses if.** Fertility measured on a prefix differs materially from fertility on a random sample
of the same file — testable by re-fetching a mid-file range and comparing.

## D5 — The codec basis is characters, not bytes

**Decision.** Our proposed variant replaces `c_byte ⊗ p_bytepos` with
`(block ⊗ p_charpos) ⊕ (offset ⊗ p_charpos)`, 128-wide Unicode blocks, 32 character positions, two
reserved block ids for non-UTF-8 byte fragments.

**Why.** E3 reduced the Indic penalty of a byte-decoding head exactly to the bytes-per-token ratio,
so the basis is the lever, not the head. E5 measured the result: D 8192 → 6144 (**−25% parameters**),
window 32 bytes → 32 characters (**3.2× deeper for 3-byte scripts**), fertility floor 1.3295 → 1.0005
for Malayalam, head amplification 1.92 → 0.65 for Hindi, and a **131,072/131,072 exact round-trip**.
Cheaper and wider at once, which is why H5a was refuted rather than confirmed.

**Reverses if.** A trained comparison shows the character basis learns materially worse at matched
parameters — untested, and the reason E6 is a training experiment rather than more analysis.

## D5 amendment — the character basis is cheaper and wider, but not better

**Trigger.** D5's stated reversal condition was "a trained comparison shows the character basis learns
materially worse at matched parameters". E6 is not parameter-matched (char used 25% fewer), so the
condition is not strictly met — but the result is close enough that the decision must be amended
rather than left standing.

**Amended.** The character basis remains a real codec result: exact round-trip over the whole
vocabulary, 25% fewer parameters, 3.2× deeper window, fertility floor removed. It is **not** a quality
win. Measured, it is 0.40% worse than the byte basis overall and worse specifically on Indic
(Malayalam +1.28% bpb), because active codec slots per token — the capacity the basis gives each
token — fall for Indic (7.87 → 5.79) while rising for English (4.56 → 9.09).

**Consequence for the write-up.** The claim we can defend is narrow and must stay narrow: a character
basis buys coverage and parameters at a small quality cost, and UTF-8's byte expansion for Indic is
partly capacity rather than purely waste. The parameter-matched arm at `P = 43` is the experiment that
would settle it, and it has not been run.

## D6 — The contribution is the control and the mechanism, not the codec

**Decision.** The deliverable is the negative result plus its mechanism, not a proposed codec.

**Why.** E7's random-index control beats every structured arm, so we cannot offer a codec as an
improvement. What we can offer, and what nobody in the published work appears to have run, is: a
randomised-index control at matched D and matched slot count; the finding that it wins; and a measured
predictor of quality across codecs (pairwise code overlap, Spearman +0.900) that explains the whole
ordering including our own failures.

**Consequence.** E5's character basis is reported as a refuted proposal with its arithmetic intact,
not as a result. D5 stands amended: the character basis is cheaper, wider, exactly invertible, and
worse — and E7 shows why, which D5 could not.
