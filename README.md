# Invertible Kronecker Embeddings

**ERA V5, Session 7 — Problem 5.** Kronecker embeddings, run backwards.

The Kronecker codec turns out to be exactly invertible. That means the output
head of a language model can be the input projection transposed, which deletes
the output matrix entirely and unties model size from vocabulary size. This
repository implements that, trains four models to test it, and reports what it
costs.

**[→ interactive page](docs/index.html)** · [`results/evidence.json`](results/evidence.json) · [`results/run.log`](results/run.log)

### The result in four lines

- **Tying works.** Replacing the output head with `Wᵀ` costs **0.052 bits/byte**
  (1.4%) and removes **99.6% of the output parameters** — the whole
  2,097,152-weight matrix, leaving an 8,192-element bias. On tokens withheld from
  training it costs nothing at all.
- **The byte head is genuinely worse than the baseline**, by 1.70 bits/byte. Only
  **0.41** of that is the price of ranging over every byte string instead of a
  fixed list. The other **1.30** is a real modelling weakness — the 32 byte
  positions are predicted independently — and it is the larger half.
- **It degrades far less on unseen vocabulary.** Going from trained-on to
  never-trained-on tokens costs the baseline **+2.82** bits/byte and the tied byte
  head **+1.29**. First-byte accuracy there is 54.7% against the baseline's 42.6%.
- **The session's nearest-neighbour formulation fails**, but not for the reason I
  expected — see [below](#why-not-nearest-neighbour).

---

## The problem, for someone who hasn't seen the session

A language model has a vocabulary — a fixed list of, say, 50,000 word-pieces.
Two big matrices are built from that list:

- an **input embedding**, which turns token #4,213 into a vector, and
- an **output head**, which turns the model's final vector into a score for each
  of the 50,000 tokens.

Both grow linearly with the vocabulary. At 1M tokens and 768 dimensions, the
output head alone is 768 million parameters. Worse, the model is *locked* to that
list: it can never emit a word not on it. Misspellings, rare names, and new
identifiers get chopped into fragments.

**Kronecker embeddings** fix the input side. Instead of storing a row per token,
a token is treated as a byte string and built from two small tables — one for
byte values, one for byte positions:

```
κ(b) = (1/√L) Σ_p  c_{b_p} ⊗ p_p
```

With 256 possible byte values and a 32-byte window, every token lands in the same
`D = 256 × 32 = 8192`-dimensional space, no matter how many tokens exist. The
input table stops growing.

The output head still does, though. And the standard trick for that — **weight
tying**, reusing the input embedding transposed as the output head — doesn't
apply, because the codec maps into `D = 8192` while the model works in
`d_model` dimensions, and those aren't the same space. The V1 paper says as much.

## The idea

The obstruction is real but it's aimed at the wrong target. Tying doesn't need
the *codec* to be transposable; it needs the *learned projection* to be. The
pipeline is:

```
bytes ──κ──> 8192-dim codec ──W──> d_model ──[ transformer ]──> d_model ──?──> next token
             (fixed, no params)   (learned)                              the only variable
```

`W` is a real `8192 × d_model` learned matrix, and `Wᵀ` maps `d_model` back to
8192. So point the output at **codec space instead of vocabulary space**: read
the 8192 numbers as a `32 × 256` grid and train it as 32 independent 256-way
classifications — "which byte is at position 0", "which byte is at position 1",
and so on, with a `PAD` class marking the end.

Two things follow.

1. **The output head can be `Wᵀ`.** Its size no longer depends on the vocabulary
   — it doesn't depend on anything, because it isn't a separate parameter at all.
2. **The model can emit any byte string.** Its range is no longer the vocabulary;
   it's every string the 32-byte window can hold.

Crucially the *decoder is a classifier, not a lookup.* The session's phrasing —
predict the Kronecker vector, then find the nearest token — sounds equivalent and
isn't. Nearest neighbour is an `argmin` over a fixed table, so it can only ever
return something already in that table, which throws away the entire benefit.
See [Why not nearest neighbour](#why-not-nearest-neighbour).

## What's here

Four models, **identical in every respect except the output head**. Same trunk,
same Kronecker input, same data, same schedule, same seeds:

| arm | output head | output params at \|V\| = 16,384 |
|---|---|---|
| `vocab` | standard `d_model × \|V\|` softmax — the baseline | 4,194,304 |
| `kron` | `d_model × D`, read as a 32 × 256 byte grid | 2,097,152 |
| `kron_tied` | **`Wᵀ` — no weight of its own, only a bias** | 8,192 |
| `regress` | predict the codec vector under MSE, decode by nearest neighbour | 2,097,152 |

`regress` is the session's literal formulation. It's included because
reproducing its failure is evidence, not decoration.

### Comparing them honestly

Byte cross-entropy and token cross-entropy are different units; putting them in
one table would be meaningless. So every arm is scored in **bits per byte** — the
negative log-probability it assigns to the *correct next token*, divided by that
token's length in bytes. For `vocab` that's one softmax term. For the byte heads
it's the sum over the byte positions, which under the head's independence
assumption is exactly the log-probability of that byte string.

Two consequences worth stating plainly:

- **`regress` has no bits-per-byte at all.** An MSE head defines no normalised
  distribution over tokens, so a likelihood does not exist for it. It is reported
  as N/A rather than given a substitute number that would look comparable.
- **The byte heads pay a real tax.** Their distribution ranges over *all* byte
  strings, so they must spend probability mass ruling out nonsense that `vocab`
  gets to ignore for free. That belongs in the number, not in a footnote — so
  it's also measured separately, by renormalising over the real vocabulary and
  rescoring. The difference between the two is the price of vocabulary-freedom.

## Results

<!-- RESULTS -->

3 seeds per arm, 2,000 steps at batch 16, sequence length 256 — 1.15 epochs over 7,101,125 training tokens. Mean ± sd across seeds.

| arm | output params | bits/byte | vs. baseline | exact token acc. |
|---|---:|---:|---:|---:|
| `vocab` | 4,194,304 | **1.995 ± 0.005** | 0.000 | 0.2375 |
| `kron` | 2,097,152 | **3.698 ± 0.004** | +1.704 | 0.1617 |
| `kron_tied` | 8,192 | **3.750 ± 0.020** | +1.756 | 0.1540 |
| `regress` | 2,097,152 | *undefined* | — | 0.0624 |

**Tying costs 0.052 bits/byte**, against a seed-to-seed spread of 0.020. It removes 2,088,960 output parameters — the entire 2,097,152-weight output matrix, leaving a 8,192-element bias vector.

**Where `kron`'s 1.704 bit/byte gap to the baseline comes from.** Renormalising its distribution over just the 16,384 real tokens gives 3.292 bits/byte. So 0.406 of the gap is the price of ranging over every byte string rather than a fixed list — the capability being bought — and the remaining **1.297** is not. That larger part is the head predicting all 32 byte positions independently from one hidden state: knowing a token starts with `th` tells it nothing about position 2.

<!-- /RESULTS -->

## The held-out vocabulary test

Matching the baseline on ordinary text only shows the byte head keeps up. The
actual claim of P5 is *vocabulary-freedom*, and that needs its own test.

A set of token ids is withheld from **every arm's training loss**. The withheld
tokens still appear in the context, so every arm's shared Kronecker input sees
their bytes — they are simply never a prediction target. At evaluation, all arms
are scored on exactly those positions.

That isolates the head and nothing else. The `vocab` head's output row for a
withheld token receives gradient only as a *negative* example, pushed down by
softmax normalisation every time some other token is correct — so it should end
up at or below chance. The byte heads were never asked for the token either, but
they were asked for every one of its bytes in other words. Whether they compose
is the whole question.

<!-- HELDOUT -->

| arm | bits/byte, seen | bits/byte, held out | penalty |
|---|---:|---:|---:|
| `vocab` | 1.957 | 4.777 | +2.821 |
| `kron` | 3.680 | 5.032 | +1.352 |
| `kron_tied` | 3.733 | 5.027 | +1.294 |

373 token ids withheld, covering 71,068 training positions (1.00%) and scored on 3,719 validation positions. Examples: `q`, `AR`, ` te`, ` gra`, `ents`, ` pos`, ` pres`, `val`, `IO`, `that`.

<!-- /HELDOUT -->

### One prediction at a time

`demo.py` re-runs the same test on every held-out position and asks a blunter
question: not "was the token right" but "how close was the byte string".

<!-- DEMO -->

| arm | exact | mean edit distance | first byte correct |
|---|---:|---:|---:|
| `vocab` | 0/3595 | 4.624 | 42.6% |
| `kron` | 4/3595 | 4.436 | 55.9% |
| `kron_tied` | 2/3595 | 4.356 | 54.7% |

Exact reproduction of a never-trained token is out of reach for every arm at this scale — 4 and 2 successes in 3,595 are not a capability claim, and they are reported as the small numbers they are. The signal is in the partial credit: the byte head gets the **first byte** right 54.7% of the time against the baseline's 42.6%, and lands closer by edit distance. It composes part of the spelling; it just rarely finishes it.

Note these two measures are defined for *every* arm, including the baseline: whatever token `vocab` selects is ultimately a byte string too, so it can be scored the same way. Nothing here is a metric the byte head gets to grade itself on.

<!-- /DEMO -->

## Why not nearest neighbour?

Two measurements, in [`analysis.py`](analysis.py). **One of them came out against
the design argument and is reported anyway.**

**Noise.** The convenient claim is that nearest-neighbour decoding is *fragile* —
that codec vectors are so crowded a small prediction error lands on the wrong
token. Measured, that's false. Under equal signal energy, nearest neighbour
decodes *better* than per-position argmax at every noise level tested, and the
codec space isn't pathologically crowded either (median gap between best and
second-best cosine: 0.052). The equal-energy framing is also the wrong yardstick,
since a classifier's logit scale is free while an MSE head's target norm is pinned
at 1 by the codec. Either way, fragility is not the argument.

**Reachability.** This is the argument, and it's structural rather than
statistical. Nearest neighbour is an `argmin` over a table of |V| entries, so its
range *is* those |V| strings. Tested on 15 real misspellings, proper nouns, code
identifiers, and multi-byte strings — `netwrok`, `Kubernetes`, `get_event_loop`,
`café`, `日本語` — nearest neighbour can emit **0 of 15**; per-position argmax can
emit **15 of 15**. A decoder that cannot leave the vocabulary cannot deliver a
vocabulary-free model.

Add the fact that an MSE head has no likelihood — it cannot be scored in bits per
byte and cannot be sampled from — and the classifier formulation is the only one
that delivers what P5 asks for.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest tests/ -q     # 55 tests
.venv/bin/python analysis.py             # -> results/analysis.json   (~1 min)
.venv/bin/python run.py                  # -> results/evidence.json   (~2 h on an M-series GPU)
.venv/bin/python site.py                 # -> docs/index.html
```

`python run.py --smoke` runs the whole pipeline in a couple of minutes to check
it works. It stamps `"smoke": true` into the evidence, and the generated page
refuses to present smoke numbers as results.

The corpus ships with the repository (`data/corpus.txt.gz`, 8.7 MB compressed)
so nothing needs the network. `run.py` verifies it against the SHA-256 in
`data/corpus_manifest.json` before training and refuses to start on a mismatch.
`python -m kron2.corpus --force` rebuilds it from source, which does need network.

## Layout

```
kron2/codec.py     the invertible codec: encode, decode, layout, round trip
kron2/corpus.py    corpus construction and integrity checking
kron2/data.py      byte-level BPE, byte tables, batching, held-out selection
kron2/model.py     the trunk and the four heads
kron2/train.py     training, evaluation, bits-per-byte, decoders
analysis.py        codec-only measurements; no trained model involved
run.py             the experiment; writes results/evidence.json and run.log
site.py            builds docs/index.html from the artefacts
tests/             55 tests
```

Every number in the README and on the page comes from `results/evidence.json` or
`results/analysis.json`. None is typed in by hand.

## Limitations

- **The byte positions are predicted independently, and this is the single
  biggest cost.** Given the hidden state, position 3 doesn't know what position 2
  chose. For a token like ` through`, knowing the first byte massively constrains
  the rest, and this head cannot use that. Measured at **1.30 bits/byte** — the
  majority of the 1.70 gap to the baseline, and larger than the 0.41 that
  vocabulary-freedom actually costs. Conditioning the positions on each other
  would fix it and would also destroy the single-matmul property that makes tying
  possible, so it is a real cost of the design rather than an implementation gap.
  If this work has an obvious next step, it is here.
- **32 bytes is a hard ceiling.** Longer tokens are truncated (12 of 16,384 in
  this vocabulary). Multi-byte scripts spend the window three to four times
  faster than ASCII.
- **`PAD` is byte 0.** A token containing a literal `0x00` would be
  indistinguishable from a shorter one. Such a token exists in the vocabulary,
  because byte-level BPE seeds all 256 bytes, but it never occurs in the corpus —
  checked at load time rather than assumed.
- **One learning rate for all four arms.** `3e-4` everywhere, unswept. That is
  neutral rather than fair: the arms have very differently sized output layers,
  and a per-arm sweep could move the comparison either way. Tuning each arm
  properly would have cost four times the compute, so it wasn't done, and the
  numbers should be read as "under one shared schedule" rather than "at each
  arm's best".
- **Tying changes the optimisation, not just the parameter count.** In
  `kron_tied` the projection receives gradient from both the input and the output
  path, so that one tensor takes a larger effective step than it does in `kron`.
  That is inherent to tying rather than a bug, but it means `kron` and
  `kron_tied` do not differ *only* in parameter count.
- **Scale.** This is a ~9M-parameter model on 27 MB of text for roughly one
  epoch. The parameter-scaling argument is arithmetic and holds at any size; the
  quality comparison is measured only here.
