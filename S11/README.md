# Optimizers and Learning-Rate Schedules

**ERA V5 · Session 11.** Five required experiments, every number reproduced from code in this
repo, plus two results the assignment did not ask for that bear directly on the V5 build.

> The assignment's own epigraph is *"almost every optimizer claim that failed to replicate was a
> well tuned method measured against a badly tuned one."* **Two of the five required experiments
> turn out to be instances of that**, and one of them is our own previously-submitted work. Both
> are reported as findings rather than smoothed over.

---

## TL;DR — the seven numbers

| # | question | answer |
|---|---|---|
| 1 | Does hand-computed Adam match PyTorch? | **Exactly.** Worst disagreement over five steps and every intermediate: **0.000e+00** (float64) |
| 2 | When does bias correction stop mattering? | **Step 1,751** (<1.10×) / **3,925** (<1.01×) — *not* the 20 steps the question plots. It **peaks at 6.57× at step 12** |
| 3 | Where does warmup stop changing the update/weight ratio? | **Steps 20–22**, i.e. **before warmup ends at 30** — and disabling warmup entirely was **0.113 nats better** |
| 4 | Cosine or WSD at step 200? | **Neither.** They differ by **+0.0021** nats against a 0.0018 seed spread. The 0.33-nat gap at step 300 is **92.5% average-LR artifact** |
| 5 | LR at width 4,096? | **1.5e-04** — from `lr* ∝ 1/width`, `a = 0.990`, confirmed on two held-out points |
| **C1** | What does V5 actually need? | `lr* ∝ 1/(width·√depth)`. V4 grows **8→20 layers at fixed width**, so that boundary needs the rate **cut ~1.6× (1.43–1.84× across estimators)**. **Or change one line**: `1/√L` residual scaling cuts the shift from 1.32 grid steps to **0.24** and costs nothing in loss (§C1.6) |
| **C2** | Is S7 §9's monitoring rule right? | **No.** And the V4 scar reproduces at **140.8×**, stochastically and only with depth |

---

## 0. Claim ledger

Every claim, its evidence, and what would kill it. **measured** = we computed it.

| # | claim | evidence | falsifier | status |
|---|---|---|---|---|
| C1 | hand Adam == `torch.optim.Adam` to 0.000e+00 on every intermediate | `code/p1_adam_by_hand.py` | any nonzero disagreement in float64 | **holds** |
| C1a | Adam(L2) and AdamW differ by **3.0%** of total movement in 5 steps | `code/p1b_adamw_and_eps.py` | the two agreeing | **holds** |
| C1b | PyTorch's ε placement is exact; moving it one step earlier costs **0.61%** at ε=1e-4 | same | eps_early agreeing to float64 | **holds** |
| C2 | uncorrected/corrected step ratio peaks at **6.57× at t=12** | closed form + 20-step run, both agree | the two disagreeing | **holds** |
| C3 | difference falls below 1.01× only at **t=3,925** | `(1−β₁ᵗ)/√(1−β₂ᵗ)` | — algebra, given β | **holds**, β₂-dependent |
| C4 | blocks peak at steps 20–22, **before** warmup ends at 30 | `code/p3p4_ratio_schedules.py` | peaks at or after step 30 | **holds** |
| C5 | the output head moves **22.4×** its final ratio vs ~9× elsewhere | same | head peak/final ≈ other layers | **holds** |
| C6 | warmup costs **0.113 nats** here and prevents nothing measurable | warmup-off control | different grad norms, or any clipping | **holds** ⚠ toy scale |
| C7 | embedding ratio is diluted **3.4× (median)** by untouched rows | touched-vs-all measurement | dilution ≈ 1× | **holds** |
| C8 | cosine and WSD are **indistinguishable at step 200** (+0.0021 vs 0.0018 spread) | 3 seeds, paired | seed signs agreeing with a gap > spread | **holds** |
| C9 | **92.5%** of WSD's 300-step advantage is average LR (−0.3286 → −0.0246) | matched-mean-LR controls | matched gap ≈ unmatched gap | **holds** |
| C10 | `lr* ∝ width^−0.990` | 3 widths, √2 grid, all bracketed | any minimum at a grid endpoint | **holds** |
| C11 | `lr* ∝ depth^−b`, **b between 0.389 and 0.667** | 4 depths; 9 distinct estimator×grid estimates | every estimate not having b > 0 | **holds, but the exponent is estimator-sensitive — see §C1.4** |
| C16 | the optimum is **far less batch-sensitive than either textbook rule**: ×1.91 over batch 4→32, where √ wants ×2.83 and linear ×8.0 | 4 batches at width 512, jobs `032a`/`022`/`032b`/`032c` | a move matching either rule | **holds** — criterion-free, no fitted shape |
| C18 | **depth-muP flattens the LR shift**: `b` 0.440 → **0.0795** over an 8× depth range, 1.32 → 0.24 grid steps | 4 depths, matched grid, matched estimator | `|b_muP| ≥ 0.15` | **holds in-sample** at 3.8σ vs SP; held-out test at depth 32 running |
| C19 | muP costs **nothing** — it is **better** at all 4 depths (−0.015 … −0.166 nats) | H12 guard, matched grid | muP >0.05 nats worse at every depth | **holds** |
| C20 | the flattening **degrades with depth**: muP local `b` rises +0.021 → +0.063 → **+0.160** | same 4 depths | a flat or falling local exponent | **holds** — and it is why C18 is not yet extrapolated to V4 |
| C21 | **seed variance is not constant along an LR sweep** — 23× larger just past the optimum (0.0050 → 0.1131 nats) than below it | 3 seeds × 3 LRs, job `032c` | flat spread across the sweep | **holds** — and it revises our own ±2% vertex-noise figure to **±11%** |
| C22 | the batch-16 dip is real: `d = val(2.4e-3) − val(1.2e-3)` > 0 at **3/3** seeds | pre-registered sign test, job `032c` | any seed with `d < 0` | **holds** |
| C17 | ~~the optimum **saturates** in batch; `Bn = 2.40`~~ | ~~3 points~~ | a non-monotone local exponent | **WITHDRAWN** — batch 32 gave +0.525 after +0.007. See §C1.4 |
| C12 | the joint law predicts **unseen** combinations | 2 held-out points, both **0.04** grid steps off | argmin ≥2 grid steps from prediction | **holds** |
| C13 | S7 §9's monitoring rule fails at every depth tested | `031`/`029`, 4 seeds | per-layer ratio beating global norm at depth ≥8 | **holds** |
| C14 | the **last** block is the stronger per-layer indicator, ~2× | seed ranges disjoint at all depths | overlapping ranges | **holds** |
| C15 | the scar is **stochastic and depth-gated**; max **140.8×** | per-seed traces, 0 spikes in 24 controls | any no-shift control spiking | **holds** at n=4 |

### Claims we made and then killed

| withdrawn | why it died | killed by |
|---|---|---|
| "the width exponent is 0.868" | that was the **2× grid**; on √2 it is **0.990** | job `025` |
| "depth does not move the optimum" | our own pre-registered prediction. It moves as **depth^−0.5** | jobs `024`/`028` |
| "depth-16 optimum is 1e-3" | refinement moved it to **1.4e-3** — the second time refinement changed an answer | job `028` |
| "the depth exponent is 0.505" | that is a **two-point endpoint fit on three different grids**. The estimator family spans **0.389–0.667** | `code/c1_exponent_robustness.py` |
| "the batch term could reverse the sign of the depth cut" | it moves **×1.91 over an 8× batch range** where √ predicts ×2.83 — much weaker than either rule, and it does not swamp depth here | jobs `032a`/`032b` |
| "the optimum saturates in batch, `Bn = 2.40`" | **our own fit to 3 points.** Batch 32 gave a local exponent of +0.467 after +0.064; a saturating curve cannot steepen again | job `032b` |
| "job 032 measures width 256" | it was cloned from the width-**512** sweep. `input_params` proves it | `results/gpu/032_CORRECTION.md` |
| "the per-layer ratio has the better SNR" (round 1) | true at 2 layers, **inverts by depth 16** | job `029` |
| "H4 SUPPORTED: the global norm dilutes with depth" | **our criterion compared only endpoints.** The sequence 11.21→7.93→8.88 is not monotonic; it passed by 0.09 | job `031` restatement |
| "the scar needs a frozen embedding" (S7 §9's framing) | a **41.0×** excursion fired with the embedding **trainable** | job `029` |

---

## 1. Adam by hand — `code/p1_adam_by_hand.py`

One weight `w₀ = 0.7`, five gradients, `lr = 1e-3`, `β = (0.9, 0.999)`, `ε = 1e-8`, no decay.

```
m_t   = β₁·m_{t-1} + (1−β₁)·g_t
v_t   = β₂·v_{t-1} + (1−β₂)·g_t²
m̂_t   = m_t / (1−β₁ᵗ)          v̂_t = v_t / (1−β₂ᵗ)
step  = lr · m̂_t / (√v̂_t + ε)
```

| t | g | m | v | m̂ | v̂ | step | w |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.5 | 0.05000000 | 0.0002500000 | 0.50000000 | 0.2500000000 | 1.0000e-03 | 0.69900000 |
| 2 | −0.3 | 0.01500000 | 0.0003397500 | 0.07894737 | 0.1699599800 | 1.9150e-04 | 0.69880850 |
| 3 | 0.8 | 0.09350000 | 0.0009794103 | 0.34501845 | 0.3267967712 | 6.0354e-04 | 0.69820497 |
| 4 | 0.1 | 0.09415000 | 0.0009884308 | 0.27377145 | 0.2474786805 | 5.5033e-04 | 0.69765464 |
| 5 | −0.6 | 0.02473500 | 0.0013474424 | 0.06040146 | 0.2700279980 | 1.1624e-04 | 0.69753840 |

**Worst disagreement against `torch.optim.Adam`, across every intermediate at every step: `0.000e+00`.**
Final weight `0.697538404008` both ways.

Two deliberate choices. **float64**, so the test measures the algebra rather than float32 rounding —
in float32 you get agreement to ~1e-7 and cannot tell a real discrepancy from rounding. And
**`weight_decay=0`**, so this is Adam, not AdamW; PyTorch's `Adam` with nonzero decay adds `λθ` to
the *gradient* before the moments, while `AdamW` applies it to the weight after, and the two give
different numbers.

**What PyTorch actually computes** is not the textbook form: it uses
`step_size = lr/bc₁` and `denom = √v_t/√bc₂ + ε`. That is algebraically identical to
`lr·m̂/(√v̂+ε)` — **but only because ε is added *after* dividing by `√bc₂`.** Put ε inside and the two
diverge in early steps, when `bc₂` is small. The 0.000e+00 confirms the placement.

---

### 1a. Adam is not AdamW — `code/p1b_adamw_and_eps.py`

Both are "Adam with weight decay" and they are **different updates**. `Adam(weight_decay=λ)` adds
`λθ` to the **gradient**, so the decay flows through the moments and gets rescaled by `1/√v̂`.
`AdamW` applies it to the **weight**, untouched by the moments:

```
Adam(L2) :  g ← g + λθ            then the usual update
AdamW    :  θ ← θ − lr·λ·θ        then the usual update
```

Same weight, same five gradients, λ = 0.1:

| t | Adam(L2) | AdamW | difference |
|--:|--:|--:|--:|
| 1 | 0.699000000018 | 0.698930000020 | 7.00e-05 |
| 2 | 0.698657378122 | 0.698668608989 | 1.12e-05 |
| 3 | 0.697982766070 | 0.697995205776 | 1.24e-05 |
| 4 | 0.697345508899 | 0.697375081253 | 2.96e-05 |
| 5 | 0.697103177275 | 0.697189107119 | 8.59e-05 |

Both match their torch counterparts to **0.000e+00**. But they diverge from each other: after five
steps the weights differ by **8.593e-05**, which is **3.0% of the total distance either one
moved.** With a decoupled decay the pull toward zero is the same size for every parameter; with L2
it is divided by that parameter's `√v̂`, so rarely-updated parameters get decayed *harder*.

### 1b. Where ε sits, and why it is not cosmetic

§1 claims PyTorch's form is algebraically identical to the textbook one **only because ε is added
after dividing by `√bc₂`**. Three variants, identical except for that:

```
textbook   lr·(m/bc₁) / ( √(v/bc₂) + ε )
pytorch    (lr/bc₁)·m / ( √v/√bc₂ + ε )          <- what torch actually computes
eps_early  lr·(m/bc₁) / ( (√v + ε)/√bc₂ )        <- ε added one step too soon
```

| ε | \|textbook − pytorch\| | \|textbook − eps_early\| | relative error at t=1 |
|--:|--:|--:|--:|
| 1e-08 | 1.1e-19 | 6.1e-10 | 0.00% |
| 1e-06 | 2.2e-19 | 6.1e-08 | 0.01% |
| 1e-04 | 1.1e-19 | 6.1e-06 | 0.61% |

**PyTorch's form is exact** — agreement at the float64 floor for every ε. Moving ε one step earlier
is **not** equivalent, and the error grows linearly with ε because bias correction divides it by
`√(1−β₂ᵗ)`, which is **0.0316 at t=1**. At ε = 1e-4 — a value real configs use — the first step is
**0.61%** wrong.

## 2. Bias correction — the answer is not twenty steps

The question asks for the first twenty steps. Here they are, same gradients, correction on vs off:

| t | step ON | step OFF | OFF/ON |
|--:|--:|--:|--:|
| 1 | 1.0000e-03 | 3.1623e-03 | 3.162 |
| 5 | 1.1624e-04 | 6.7384e-04 | 5.797 |
| 10 | 2.1394e-04 | 1.3966e-03 | 6.528 |
| 15 | 1.3612e-04 | 8.8569e-04 | 6.507 |
| 20 | 1.5370e-04 | 9.5923e-04 | **6.241** |

**At step 20 the two are still 6.24× apart.** The ratio is exactly

```
step_OFF / step_ON = (1 − β₁ᵗ) / √(1 − β₂ᵗ)
```

which **rises before it falls**, peaking at **6.57× at step 12** — so the twenty-step window the
question specifies ends essentially *at the peak*, which is why plotting it invites the wrong
conclusion.

| t | 1 | 12 | 20 | 100 | 1,000 | 3,000 | 10,000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| OFF/ON | 3.16 | **6.57** | 6.24 | 3.24 | 1.26 | 1.03 | 1.00 |

> **First step below 1.10× : t = 1,751.  Below 1.01× : t = 3,925.**

**Mechanism.** `1−β₁ᵗ` is done by ~step 50; `1−β₂ᵗ` with β₂ = 0.999 takes thousands. The second
moment is corrected far longer than the first, and because `v̂` sits under a square root while `m̂`
does not, the mismatch *grows* until `m`'s correction saturates.

⚠️ **This number is a function of β₂, not a constant** — and it is the *only* thing that sets it.
The ratio settles below a threshold at wildly different times depending on β₂:

| β₂ | ratio at t=1 | peak | peak at | settles <1.10× | settles <1.01× |
|---|--:|--:|--:|--:|--:|
| 0.9 | 0.316 | 1.00 | 345 | **never exceeds** | **never exceeds** |
| 0.95 | 0.447 | 1.10 | 20 | never exceeds | **76** |
| 0.99 | 1.000 | 2.13 | 13 | 175 | **391** |
| **0.999** *(default)* | 3.162 | **6.57** | 12 | 1,751 | **3,925** |
| 0.9999 | 10.000 | 20.72 | 12 | 17,512 | **39,268** |

**At β₂ = 0.9 bias correction never makes a difference at all** — the ratio never rises above 1.0.
At 0.9999 it takes **39,268 steps**. Roughly, the settling time scales with `1/(1−β₂)`.

> **So "bias correction matters for ~4,000 steps" is not a fact about Adam. It is a fact about
> β₂ = 0.999.** Reported as a single number it gets cached as universal, which is why the table is
> here instead.

*(Method note: the crossing must be found as the **last** step at or above the threshold, not the
first step below it. At β₂ ≤ 0.95 the ratio **starts below 1** and rises through the threshold, so a
first-below search returns t=1 and is wrong. We made exactly that error — see "What we got wrong".)*

![bias correction](figures/fig1_bias_correction.png)

*Left: the twenty steps the question asks for — still 6.24× apart at the end of the window.
Right: the same ratio out to 10⁶ steps for three values of β₂, log-log. The vertical line is where
the left panel stops.*

## 3. Update-to-weight ratio per layer

`‖Δw‖/‖w‖` per layer per step. Width 128, 2 layers, 300 steps, warmup 30, `lr = 2e-3`.

| layer | peak ratio | **at step** | ratio@300 | peak/final |
|---|--:|--:|--:|--:|
| `emb` | 6.57e-03 | **30** | 7.66e-04 | 8.6 |
| `blocks.0.qkv` | 1.18e-02 | **20** | 1.27e-03 | 9.3 |
| `blocks.0.mlp.0` | 9.24e-03 | **20** | 1.07e-03 | 8.6 |
| `blocks.1.qkv` | 1.25e-02 | **21** | 1.28e-03 | 9.8 |
| `blocks.1.mlp.0` | 1.19e-02 | **22** | 1.36e-03 | 8.8 |
| **`head`** | **3.10e-02** | **29** | 1.38e-03 | **22.4** |

**Answer: steps 20–22 for the transformer blocks — ten steps *before* warmup ends at 30.** The LR is
still rising, but the gradient is shrinking faster, so warmup stops being the binding term early.
Only the embedding and the head peak at the warmup boundary itself.

**The output head is a different animal**: 2.6× the largest block's peak, and **22.4×** its own final
value against ~9× everywhere else. It is the layer that moves most, relative to its own size, during
warmup — which is an argument for treating the head's LR separately, not an argument about warmup.

![per-layer update-to-weight ratio](figures/fig4_layer_ratios.png)

*Every layer's `‖Δw‖/‖w‖`, log-log, peak marked. The dashed line is the end of warmup at step 30 —
the transformer blocks have already peaked by then.*

### 3a. Negative control: warmup disabled

| layer | peak WITH warmup | @step | peak WITHOUT | @step | OFF/ON |
|---|--:|--:|--:|--:|--:|
| `emb` | 6.57e-03 | 30 | 1.08e-02 | **1** | 1.64 |
| `blocks.0.qkv` | 1.18e-02 | 20 | 3.91e-02 | **1** | **3.32** |
| `blocks.0.mlp.0` | 9.24e-03 | 20 | 3.91e-02 | **1** | **4.23** |
| `head` | 3.10e-02 | 29 | 3.88e-02 | **1** | 1.25 |

```
step-1 gradient norm:   warmup 0.464    no-warmup 0.464      identical
steps clipped (>1.0):   0/300           0/300                neither, ever
val loss @300:          8.2901          8.1774               no-warmup 0.113 BETTER
```

**Warmup prevented nothing measurable and cost 0.113 nats.** The gradient norm at step 1 is *the same
number* with and without it, and nothing clipped in either run — so warmup was throttling step size,
not taming a gradient.

⚠️ **Scale-dependent, and we do not generalise it.** Warmup's usual justification is large-batch
instability early in training; at 128-wide, 2-layer, batch 8, that regime does not exist here. What
this does establish is that the mechanism people *describe* — warmup suppressing an early gradient
spike — is **not** what is happening at this scale, because there is no spike to suppress.

### 3b. The embedding's ratio is diluted by rows that got no gradient

Only **1,024 of 61,070 rows (1.68%)** receive gradient per step, but `‖Δw‖/‖w‖` is taken over the
whole table.

| step | over ALL rows | over TOUCHED rows | dilution |
|--:|--:|--:|--:|
| 1 | 3.60e-04 | 3.34e-03 | **9.3×** |
| 30 | 6.57e-03 | 2.71e-02 | 4.1× |
| 300 | 7.66e-04 | 2.00e-03 | 2.6× |

**Median dilution 3.4×.** A whole-table ratio understates real embedding movement by 3–4×, so any
threshold set on it is wrong by that factor. This is the mechanism behind §C2 below, measured inside
a single layer.

---

## 4. Cosine vs WSD, both stopped at step 200

Same model, same data, same seed, 300-step schedules, both evaluated at step 200. **Three seeds per
arm, paired on seed** — a single-seed schedule comparison is exactly the unreplicated claim the
epigraph warns about.

| seed | cos@200 | wsd@200 | Δ@200 | cos@300 | wsd@300 | Δ@300 |
|--:|--:|--:|--:|--:|--:|--:|
| 0 | 8.3941 | 8.3971 | +0.0030 | 8.2901 | 7.9761 | −0.3140 |
| 1 | 8.3624 | 8.3646 | +0.0022 | 8.3190 | 7.9751 | −0.3439 |
| 2 | 8.3587 | 8.3599 | +0.0012 | 8.2989 | 7.9711 | −0.3278 |
| **mean** | **8.3717** | **8.3739** | **+0.0021** | 8.3027 | 7.9741 | **−0.3286** |

**At step 200 they are indistinguishable**: +0.0021 nats against a paired seed spread of 0.0018.
By step 300 WSD is 0.33 nats ahead on every seed.

### The comparison as specified is not fair

```
mean LR multiplier after warmup:   cosine 0.5517    wsd 0.9017    ratio 1.634×
```

So the two arms differ in **total learning rate spent**, not only in schedule shape. Controls:

| arm | peak lr | effective mean LR | val@200 | val@300 |
|---|--:|--:|--:|--:|
| cosine | 2.00e-3 | 1.103e-3 | 8.3717 | 8.3027 |
| WSD | 2.00e-3 | 1.803e-3 | 8.3739 | **7.9741** |
| **WSD @ matched mean LR** | 1.22e-3 | 1.100e-3 | 8.3854 | **8.2780** |
| **cosine @ matched mean LR** | 3.27e-3 | 1.803e-3 | **8.3517** | 8.0582 |
| WSD annealed *to* 200 | 2.00e-3 | 1.204e-3 | 8.3707 | 8.3053 |

```
WSD − cosine @300, as specified :  −0.3286 nats
WSD − cosine @300, LR-matched   :  −0.0246 nats
                                   ─────────
                    92.5% of the effect was average LR, not schedule shape
```

**A 13× collapse.** WSD does still win at matched average LR, so the shape buys something real — but
the honest number is **0.0246 nats, not 0.3286**. Run the two arms exactly as the question specifies and
the result is 92.5% artifact.

### Which model would we keep?

**Neither of the two the question offers.** At step 200 all five arms sit within ±0.02 nats, and the
only one with a consistent sign across seeds is **cosine at a tuned LR** (−0.0200, 3/3 seeds). The
`wsd_annealed_200` arm's seed signs *disagree*, so it is undetermined, not a winner.

But loss at 200 is the wrong question, because **the two checkpoints are not interchangeable**:

| | what you have at step 200 | what it buys | what it gives up |
|---|---|---|---|
| **cosine** | mid-decay, lr 7.53e-4 of 2e-3 | partially annealed, usable now | **committed to a 300-step horizon** |
| **WSD** | full lr 2.00e-3, unannealed | you may still choose the horizon | not shippable without annealing first |

V4 hit exactly this and recorded it: stopping into an annealed schedule *"forces a choice between
recovering at the low annealed learning rate, which is slow and may not converge, and raising the
rate again, **which discards the annealing already paid for**."*

**So: keep WSD, and anneal from wherever you decide to stop.** Not because it wins at 200 — it does
not — but because cosine has already spent the option and WSD has not.

---

## 5. Learning rate vs width — and the value at 4,096

Grid `lr ∈ {3e-4 … 8e-3}` at 2×, then **refined to √2** around each minimum. 1,200 steps,
`head_dim = 64` held fixed so width is the only thing changing.

| width | best | 2nd | tied within 0.0073 spread | **optimum** |
|--:|--:|--:|---|--:|
| 256 | 2.8e-3 (5.7464) | 2.0e-3 (+0.0064) | both | **2.37e-3** |
| 512 | 1.4e-3 (5.7277) | 1.0e-3 (+0.0065) | both | **1.18e-3** |
| 1024 | 6.0e-4 (5.7163) | 8.5e-4 (+0.0176) | clean | **6.00e-4** |

![LR sweeps](figures/fig2_lr_sweeps.png)

*Minima circled. Legend gives each optimum; where two rates tie within the 0.0073 seed spread the
legend reports their geometric mean, so the circle (argmin) and the legend value can differ slightly.*

Every minimum is **interior** with neighbours 3–12× the seed spread, and `8e-3` diverges at all three
widths, so the sweep genuinely brackets rather than reporting its own boundary.

```
coarse 2× grid     a = 0.868
fine   √2 grid     a = 0.990        256→512: 1.000   512→1024: 0.980
```

⚠️ **The 0.868 was a grid artifact**, and we published it before refining. On a 2× grid one step is a
factor of 2, so any `a` in [0.7, 1.0] fit the data equally well.

> ### Value at width 4,096: **1.5e-04**
>
> `6.00e-4 × (4096/1024)^−0.990 = 1.52e-04`, or `1.50e-04` under exact `1/width`. The two agree to
> **1.4%**.

**Confidence.** High on the law, moderate on the number. The exponent is 0.990 with per-step values
1.000 and 0.980, and it survives two held-out tests (§C1). The residual uncertainty is that all
points sit at ≤1024 and 4,096 is 4× beyond the largest, at 1,200 steps — and §4 shows the optimum is
schedule- and horizon-dependent. **We would start V5 at 1.5e-4 and re-check at one width, not adopt
it unverified.**

---

# C1 · The axis the assignment asks about is the one V4 holds fixed

*Not required by the assignment.*

The question asks for the LR at width 4,096. That is not hypothetical: **4,096 is V4's residual
width.** But the cookbook's stage table (§13.2) shows it is 4,096 at *every* stage:

| | 2B | 5B-MoE | 9B-MoE | 120B-MoE |
|---|--:|--:|--:|--:|
| **Residual width** | 4,096 | 4,096 | 4,096 | 4,096 |
| **Layers** | 8 | 8 | **20** | 20 |
| Routed experts | — | 20 | 20 | **460** |
| Global batch | 32 | 32 | **120 / 56** | 64 |

**The lineage never grows by width.** It grows by depth and expert count. So we ran the axis it does
vary — depth 2/4/8/16 at fixed width 256, same grid, same refinement:

| depth | 2 | 4 | 8 | 16 |
|---|--:|--:|--:|--:|
| optimum | 4.00e-3 | 2.37e-3 | 2.00e-3 | 1.40e-3 |

```
width exponent   a = 0.990          depth exponent   b = 0.505
```

> ## `lr* ∝ 1 / (width · √depth)`

**Two different laws on the two axes** — linear in width, square-root in depth.

### Held-out test

Every point above varies one axis: widths at depth 4, depths at width 256. Fitting on that cross and
extrapolating to 4096×20 tests the axes and never the interior. So we predicted two **unseen
combinations** before running them:

| held-out point | predicted | measured argmin | error |
|---|--:|--:|--:|
| 512 × 8 | 8.39e-4 | **8.50e-4** | **0.04** √2 grid steps |
| 512 × 16 | 5.92e-4 | **6.00e-4** | **0.04** √2 grid steps |

Both interior, both bracketed. **The law predicts combinations it was not fitted on.**

![predicted vs measured](figures/fig3_law.png)

*Predicted against measured, log-log, with the 1:1 diagonal. Blue points were fitted on; the two
green crosses were not.*

### What this gives V5

| V4 stage | config | predicted lr |
|---|---|--:|
| 2B / 5B-MoE | 4096 × 8 | **1.07e-04** |
| 9B / 120B-MoE | 4096 × 20 | **6.75e-05** |

> **A depth change of 8 → 20 layers, at fixed width and fixed batch, requires cutting the learning
> rate by roughly 1.6×** — with the estimator sensitivity below, **1.43× to 1.84×**.

### C1.4 — the batch term, measured — and a shape we published and then withdrew

The warning that stood here said the law has **no batch term**, that a larger batch pushes the rate
**up** against the depth cut, and that the net correction "could be much smaller than 1.59×, or could
reverse sign — on our evidence we cannot say which." Job `032` measures the term. It also caught us
in the exact failure this assignment is about, so both are reported.

> ⚠️ **Job `032` ran at width 512, not the width 256 its metadata records.** It was cloned from the
> width-512 sweep. `input_params` is computed at runtime and reads 31,267,840 = 61070 × 512 in every
> row. The batch axis is unaffected — every point in the series is width 512 — and the batch-8 anchor
> is job `022`, which is width 512 and was run independently a day earlier. Full evidence:
> [`results/gpu/032_CORRECTION.md`](results/gpu/032_CORRECTION.md). `verify.py` now derives width
> from `input_params` and never from the label, because this mislabel **did** propagate: it dropped
> the verifier from 140/140 to 134/140 by pooling batch-4 and batch-16 runs into the width-256 curve.

**Convention, because it decides what the answer means.** Batch varies at **fixed steps**, so a larger
batch also sees proportionally more tokens. That is what V4's stage boundary does — training
continues, the batch grows, the step count does not shrink. It is *not* the fixed-epoch convention
Goyal et al.'s linear rule and Krizhevsky's √ rule were derived under, so a disagreement with those
rules here is a disagreement about a different experiment.

| batch | job | seeds | interpolated `lr*` | local exponent |
|--:|---|--:|--:|--:|
| 4 | `032a` | 1 | 9.455e-4 | — |
| 8 | `022` | 1 | 1.247e-3 | **+0.400** |
| 16 | `032a`+`032c` | **3** | **1.253e-3** | **+0.007** |
| 32 | `032b` | 1 | 1.803e-3 | **+0.525** |

Batch 16 is the 3-seed mean from the replication below; at seed 0 alone it is 1.304e-3, giving
+0.064 and +0.467. **The conclusion holds under either choice** — the 3-seed value makes the
sequence *more* non-monotone, not less.

Optima are **parabola vertices in log-LR**, not grid argmins: the jobs use offset 2× grids, and argmin
would quantise each optimum to its own grid. Each batch is estimated on the grid its own sweep ran —
job `025` later added two √2 points at width 512 which belong to the width law, and folding them into
batch 8 alone would give one batch a finer grid than the rest.

#### What survives, and it is the part C1 needs

**Over batch 4 → 32 the optimum moves ×1.91, where the √ rule wants ×2.83 and the linear rule ×8.0.**
It is **1.48× less sensitive** than the weaker of the two textbook rules, across an 8× batch range.
That is criterion-free, does not depend on any fitted shape, and is the finding.

> **Our pre-registered criterion for this test was degenerate, and we report that rather than
> re-thresholding it.** "Within one grid step" over an 8× batch range is ±0.33 in exponent, and the
> √ and linear bands were written to overlap — `NEITHER`, which the pre-registration called "a real
> outcome", **cannot fire for any p ∈ [0,1]**. Applied literally it prints `CONFIRMED-SQRT` for a p
> of **+0.311**, a fitted exponent that is nowhere near 0.5 and corresponds to a ×1.91 move against
> √'s ×2.83. This is the **second** badly-written threshold in this assignment (see H4). The
> comparison above needs no threshold at all.

#### What we published, and then withdrew

> **We fitted a saturating curve to three points and it did not survive the fourth.**
>
> With batches 4, 8 and 16 the local exponents were **+0.400, +0.064** — textbook gradient-noise-scale
> saturation (McCandlish et al. 2018), `lr*(B) = lr∞·B/(B+Bn)`. We fitted `Bn = 2.40`, wrote it into
> this README, and derived from it: a ×1.054 batch term at V4's boundary, a 1.51× net cut, and a
> "`Bn ≥ 13` cancels half the depth cut" bound.
>
> Batch 32 then returned at `lr* = 1.803e-3`. The sequence is **+0.400, +0.007, +0.525**. A saturating
> curve *requires* the local exponent to decay monotonically — linear below the critical batch, flat
> above it, never steepening again.
>
> **Withdrawn: `Bn = 2.40` and every number downstream of it.** Three points cannot distinguish a
> saturating curve from a curve that happens to be flat between two of them.

##### The refutation does not rest on the residuals, and an earlier draft of this section said it did

The best refit has rms **10.0%** in lr, with residuals alternating in sign:
(+2.4%, +2.0%, −14.2%, +11.6%). We first published that as "the signature of a wrong functional form, not of noise, against a
seed spread that moves the vertex by ~2%." **That ~2% was a model, not a measurement** — it came from
assuming 0.004-nat noise — and job `032c` has since measured the real figure at **±11%** (below). A
10.0% misfit against ±11% vertex noise proves nothing on its own.

**What the refutation actually rests on is how large an error would be needed to rescue the
saturating form.** For the local exponent to decay monotonically, the 16 → 32 step must be no larger
than the 8 → 16 step, which is **+0.007** at batch 16's replicated 3-seed value. Batch 32's optimum
would therefore have to sit near **1.25e-3** instead of the measured **1.803e-3** — a **44% error in a
single vertex, 4× the measured ±11% per-seed spread.** And the batch-16 dip that starts the
non-monotonicity is separately replicated 3/3 (below).

⚠️ **Still thin:** batch 32 is **one seed**. The 44%-vs-11% margin is why we call the shape refuted
rather than merely doubted; a replicate at batch 32 would close it properly and we did not run one.

#### The replication (job `032c`), and what it found that we were not looking for

The whole non-monotonicity turned on one comparison at one seed. Pre-registered sign test on
`d = val(2.4e-3) − val(1.2e-3)` at batch 16:

| seed | 0 | 1 | 2 |
|---|--:|--:|--:|
| `d` | +0.0730 | +0.1553 | +0.0464 |

> **REPLICATED, 3/3.** The batch-16 optimum really is near 1.2e-3 and `Bn` is not reinstated. Using
> the 3-seed **mean** curve (1.2532e-3, used in the table above) instead of seed 0 (1.304e-3) makes
> the sequence **more** non-monotone — +0.400, **+0.007**, +0.525 rather than +0.400, +0.064, +0.467.
> The conclusion is the same under either choice.

**And seed variance is not constant along a sweep** — this is the part we did not go looking for:

| lr | 6.0e-4 | 1.2e-3 | 2.4e-3 |
|---|--:|--:|--:|
| seed spread (n=3) | 0.0050 | 0.0089 | **0.1131** |

**23× larger at `2.4e-3` than at `6.0e-4`** — and `2.4e-3` is the point just *past* the optimum,
exactly where the curve turns up and exactly where a bracketing decision gets made. Our
pre-registered seed spread of **0.0073** nats (S7 job `018`) was measured at or below an optimum;
above one it understates the noise by **~15×**.

Re-estimating the batch-16 vertex from each seed separately gives **1.304e-3, 1.139e-3, 1.398e-3** —
**±11%**, not the ±2% we had claimed. ⚠️ **Every single-seed optimum in this submission carries that
uncertainty, including the ones whose conclusions we are keeping** — the width and depth laws, and
the muP comparison in §C1.6. It is stated here rather than only where it happens to be convenient.

**Third time this assignment.** The width exponent was 0.868 from a 2× grid (→0.990 on √2). The depth
exponent was 0.505 from two endpoints (→0.389–0.667 across estimators, §C1.5). Now a shape from three
points. Same mechanism each time: *too few points read as a law.* The epigraph — *"a well tuned method
measured against a badly tuned one"* — is about the analysis as much as the training run.

#### What is unresolved, and the test that resolves it

Every point above is **one seed**, and *"one run is an anecdote"* is a rule in our own rubric. The
whole non-monotonicity rests on one comparison: at batch 16, `lr=2.4e-3` is **0.073 nats worse** than
`1.2e-3`; at batch 32 the same comparison **flips** to 0.067 nats better. Against a measured seed
spread of 0.0022–0.0058 nats that is ~15× noise and should not be a fluke — but it has not been
replicated.

Job `032c` re-runs batch 16 at **seeds 1 and 2** on those three LRs, with the sign of
`d = val(2.4e-3) − val(1.2e-3)` pre-registered: both positive **replicates** (shape refuted, `Bn`
stays withdrawn); both negative **overturns** seed 0 (refit `Bn` from seed means); disagreeing signs
means the optimum sits inside seed noise at this grid, which refutes the 3-point fit's precision
anyway. **In two of three outcomes `Bn = 2.40` does not come back.**

#### So what does this do to the C1 claim?

| | |
|---|---|
| depth 8 → 20 (measured) | **×0.630**, i.e. cut **1.59×** — rescoped to 1.43–1.84× in §C1.5 |
| batch 32 → 120 | **not quotable.** Bounding it needs a functional form, and ours was refuted |

The honest statement is narrower than either the original warning or our first answer to it:

> **The optimum is much less batch-sensitive than the textbook rules — ×1.91 over an 8× range where
> √ predicts ×2.83.** That bounds the *direction and rough size* of the batch correction at our
> scale and says it does not swamp the depth term here. It does **not** transfer to V4's boundary,
> and we no longer claim a number for it: the quantity that would decide it — V4's critical batch —
> requires a functional form we have not established, on a model we cannot train.
>
> **What would settle it:** `Bn` is the ratio of squared gradient norm to gradient variance,
> estimable from two batch sizes at a single step (McCandlish et al. §2) — cheap even on a 9B model,
> no training run needed. We did not run it, so we do not claim its answer.

### C1.5 — how much of the exponent is the data, and how much is the estimator?

The assignment asks for the rate at width 4,096 *"and how confident you are in it."* Both exponents
were computed as **two-point endpoint fits**:

```python
a = log(OW[256] / OW[1024]) / log(4)      # discards width 512
b = log(OD[2]  / OD[16])   / log(8)       # discards depths 4 and 8
```

That puts the whole exponent on the two noisiest points in each sweep — the same fragility that made
criterion H4 print SUPPORTED off a 0.09-nat endpoint gap. And the four depth optima come from **three
different grids** (2 and 8 coarse, 4 from a √2 refinement, 16 from a single √2 point), while the
tied-geometric-mean estimator is grid-sensitive by construction. So we recomputed both exponents
under every {estimator × grid} combination. **No new data — only our own archived JSON re-analysed.**

| exponent | distinct estimates | range | median | published | rank |
|---|--:|---|--:|--:|--:|
| width `a` | 5 | 0.868 – 1.111 | 0.969 | **0.990** | 5 of 5 |
| depth `b` | 9 | **0.389 – 0.667** | 0.479 | **0.505** | 8 of 9 |

**The width exponent survives.** Every estimate rounds to 1, so `lr* ∝ 1/width` and the answer at
width 4,096 do not move. **The depth exponent does not** — a 71% span, and the published value sits
near the top of it. (It *is* the median if all 12 estimator×grid cells are weighted equally, 0.5039;
the two summaries disagree because argmin and tied-geo-mean collapse together exactly on the coarse
grid, so de-duplicating removes coarse-grid cells preferentially. We report both rather than picking
the flattering one.)

| estimate of `b` | growth 8 → 20 factor |
|---|--:|
| published (2-point, mixed grid) | 1.588× |
| median of the family | 1.550× |
| smallest (parabola, mixed grid) | **1.429×** |
| largest (argmin, one coarse grid) | **1.842×** |

> **The qualitative claim is untouched: all 9 estimates have the rate coming down on depth growth, by
> roughly half a power, and the recommendation is the same at either end.** What is not supported is
> the third significant figure. `1.59×` should be read as **"about 1.6×, and 1.43–1.84× depending on
> how the exponent is estimated."**
>
> This weakens our own published claim, and it was found by re-running our own analysis with a
> different estimator — not by collecting new data. The assignment's epigraph, *"a well tuned method
> measured against a badly tuned one,"* applies to the **analysis** as much as to the training run: a
> scaling exponent read off two endpoints of a 2× grid is a choice, not a measurement.

### C1.6 — the experiment that tries to *solve* the problem instead of measuring it

Everything above measures how badly a tuned rate breaks when the model grows. This asks whether the
right parametrisation makes it **not break**. Depth-muP (Yang et al., *Tensor Programs VI*) scales
every residual branch by `1/√L`:

```python
x = x + (1/math.sqrt(L)) * attn(norm(x))      # <-- the entire change
x = x + (1/math.sqrt(L)) * mlp(norm(x))       # <-- on BOTH branches
```

Nothing else moves: width 256, batch 8, seq 256, 1,200 steps, same optimiser, schedule, data, eval,
and the **same 6-point grid**, so `b_muP` and `b_SP` sit on the same surface. Every SP number below is
recomputed from the SP jobs on that identical grid with the identical parabola-vertex estimator — the
table job `033` prints inline is *not* grid-matched and is not used.

| depth | SP `lr*` | SP `val*` | muP `lr*` | muP `val*` | muP − SP | SP local `b` | muP local `b` |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 2.858e-3 | 5.6761 | 2.902e-3 | 5.6616 | −0.0145 | — | — |
| 4 | 2.166e-3 | 5.7518 | 2.861e-3 | 5.6483 | −0.1036 | +0.400 | **+0.021** |
| 8 | 1.548e-3 | 5.7580 | 2.738e-3 | 5.7278 | −0.0302 | +0.484 | **+0.063** |
| 16 | 1.157e-3 | 5.8826 | 2.450e-3 | 5.7170 | −0.1655 | +0.420 | **+0.160** |

```
b_SP = +0.4397          b_muP = +0.0795          over an 8× depth range
```

> **H11 CONFIRMED** (`|b_muP| < 0.15`). Over 2 → 16 layers the optimum moves **×0.848 under muP** —
> **0.24 of a grid step, i.e. it does not move** — against **×0.401 under SP**, which is 1.32 grid
> steps and a rate you must re-tune.

##### Two claims here, and they are *not* equally well supported

Job `032c` measured the per-seed spread of a parabola vertex directly at **±11%** (§C1.4). Every
point in both arms is one seed, so that is the noise on each `lr*`. Propagating it through the OLS
slope gives `se(b) = 0.067` over four depths:

| | |
|---|---|
| **(a) muP flattens the depth dependence relative to SP** | difference **+0.360 ± 0.095 — 3.8σ.** Solid. This is what the one-line change buys. |
| **(b) muP makes the rate *transfer*, i.e. `b_muP` ≈ 0** | `|b_muP| = 0.0795` sits only **1.1σ** inside H11's own threshold, and the local exponent is rising through the range. |

> **H11 passes as written, but "`b` is small" is much better supported than "`b` is zero", and we do
> not claim the latter.** This is precisely what jobs `034a`/`034b` test: an out-of-sample point at
> depth 32 probes (b) directly, where an in-sample fit cannot.

**H12 fires nowhere, and fires in the *opposite* direction.** The guard was there because this is how
such claims usually die — a parametrisation that flattens the LR curve by making everything equally
bad is not transfer, it is a worse model. muP reached a **lower** loss at all four depths (−0.015,
−0.104, −0.030, −0.166). The mechanism is visible in the raw curves: at `lr=2e-3` the arms are
indistinguishable (5.7528 vs 5.7550); at `4e-3` muP is far better (5.7419 vs 5.8087). Scaling each
branch down lets a higher rate stay stable.

#### ⚠️ The caveat is inside the result, and the aggregate hides it

**`b_muP` is not flat — it rises with depth: +0.021, +0.063, +0.160.** The last already exceeds
H11's *own* 0.15 threshold. The aggregate 0.0795 is carried by the shallow end, where depth 2 and 4
barely discriminate at all. Read literally, the measurement says **the flattening degrades as the
model gets deeper** — and the depths V4 actually uses (8 → 20) are at the degrading end.

So we have a fitted exponent over four points, which is exactly what this assignment has caught us
over-reading three times already (§C1.4). **A fitted exponent is not a prediction until it predicts
something it was not fitted on.**

#### The held-out test, pre-registered before it ran

Depth **32** is in neither sweep. From the fits above:

| arm | prediction at depth 32 |
|---|--:|
| muP | **2.3803e-3** |
| SP | **8.5175e-4** |

They differ by **×2.79** — about 1.5 grid steps — so the measurement discriminates. Jobs `034a`/`034b`
run both arms at depth 32, differing in exactly one constant (`res_scale`, and `1.0` multiplies
exactly, so the SP arm is bit-faithful to jobs `024`).

- **H13** — each arm lands within 0.5 grid steps (×1.41) of **its own** prediction; CONFIRMED only if
  both do. **If the muP arm lands closer to the SP prediction than to the muP one, H11 is withdrawn**
  regardless of how good the in-sample fit was.
- **H14** — the rising-exponent caveat, *tested rather than asserted*. **DEGRADES** if the 16 → 32
  local exponent exceeds 0.15, in which case muP transfer is a **local** property and we do not
  support extrapolating it to V4's depth 20+. **HOLDS** otherwise. **Pre-registered because we expect
  it to degrade** — a caveat we predict and then measure is worth more than one we assert.

#### What this would be worth to V5, and what it is not worth yet

The cookbook never mentions muP or any transfer parametrisation anywhere in its 284 KB. If the
held-out test holds, the recommendation is one line in the `Block` and V5 stops re-tuning at growth
boundaries. **We are not claiming that yet.** What is measured is: at toy scale, across 2–16 layers,
`1/√L` residual scaling reduces the depth-driven LR shift from 1.32 grid steps to 0.24 **and** costs
nothing in loss. Every point is one seed; widths above 256 are untested; the interaction with the
MoE routing V4 actually grows into is untested.

Reproduce: `python code/c1_mup.py`.

Reproduce: `python code/c1_batch_law.py` and `python code/c1_exponent_robustness.py`.

The cookbook states no LR rule for that transition, and **never mentions muP or any transfer
parametrisation** anywhere in its 284 KB. A tuned rate carried unchanged through that growth step is
1.4–1.8× too high on this evidence — **or the growth step stops needing a new rate at all**, which is
what §C1.6 measures.

⚠️ **Limits.**
- **The batch term is measured but not transferable, and its shape is unresolved** — §C1.4. The
  optimum is ×1.91 over an 8× batch range, far below either textbook rule, so it does not swamp the
  depth term at our scale. We fitted a saturating form to 3 points, batch 32 refuted it, and we
  withdrew it. No number is quoted for V4's boundary. This is the binding limitation, not a footnote.
- **The depth exponent is estimator-sensitive** — §C1.5, 0.389–0.667 across 9 estimates. Read the
  growth factor as ~1.6×, range 1.43–1.84×.
- **No validation against a real run at scale.** V4's only published learning-rate trace
  (`data.js`, series `lz_120b`) peaks at **1.0e-5** and floors at **1.0e-6** — but that is the
  **120B TQP run**, training *rank-16 quantized adapters*, which cookbook §13.2 and §12.5 describe
  as a deliberately low-rate regime with an adapter multiplier of 3.0 and a router-to-rest ratio of
  0.01. It is **not** a from-scratch dense pre-training optimum, so our 6.75e-5 prediction is
  neither confirmed nor contradicted by it. We looked for the comparison and report that it does
  not exist, rather than quoting a 7× "discrepancy" against an incomparable number.
- Widths ≤1024, depths ≤16, 1,200 steps, one corpus, dense models throughout. V4's transition is
  to a **sparse MoE** and expert count (20 → 460) is a third axis we did not sweep.
- **Horizon-dependent.** §4 shows schedule effects change with the stopping point; all optima here
  are "at 1,200 steps under this cosine".

---

# C2 · The V4 scar, instrumented — and S7 §9's monitoring rule is wrong

*Not required by the assignment.*

S7 §9 prescribes a monitoring rule and never measures it:

> *"the gradient norms of the layers **immediately above** the embedding are monitored as a leading
> indicator rather than the **global norm, which averages the signal away**."*

The cookbook's own §7.5 grades the underlying incident as **the thinnest-evidenced failure in its
section**: *"no clean saved trace isolates the instability… not because the spike can be shown."*

**Part 3 builds exactly that instrument**, so we built the trace. `train.npy` is 1,024-token
single-language blocks; `codec.npz` carries every token's UTF-8 bytes, so language is recoverable by
script. Validated before use: **93.28% token-level, 100.00% block-level** against the known
`val_lang` labels (200 blocks). Then a sharp **English → Indic** shift at step 200, embedding frozen
vs trainable, with a **no-shift control for every cell**.

### The scar reproduces

Global grad-norm excursion × baseline, frozen embedding, **per seed, n = 10**:

| depth | frozen+shift | frozen+noshift | trainable+shift | trainable+noshift | median | max |
|--:|--:|--:|--:|--:|--:|--:|
| 2 | **0/10** | 0/10 | 0/10 | 0/10 | 9.2 | 13.4 |
| 8 | **1/10** | 0/10 | 1/10 | 0/10 | 8.2 | 67.0 |
| 16 | **2/10** | 0/10 | 2/10 | 0/10 | 8.0 | **140.8** |
| *32* | *3/10* | ⚠ *1/10* | *2/10* | *0/10* | *9.6* | *178.4* |

**At depths 2, 8 and 16: zero spikes in 60 control runs.** And the timing is tight —

```
shift at step 200.   spikes at steps 204, 205, 206.   4-6 steps after.
```

- S5 and S7 report the V4 incident as *"roughly 150×"*. **Our largest clean excursion is 140.8×.**
- **Why V4 had no clean trace is visible here: the event is seed-dependent.** 2 runs in 10 at depth
  16, none at all at depth 2. A single run misses it, which is exactly what §7.5 describes.
- **Incidence rises monotonically with depth: 0/10 → 1/10 → 2/10.**

> ### ⚠️ Our pre-registered kill-switch fired, and we are scoping rather than withdrawing
>
> We committed in advance: *"H10 — incidence is zero in every no-shift control at every depth. Any
> no-shift spike REFUTES the whole construction, and rounds 1–4 would all have to be withdrawn."*
>
> **A control spiked.** At depth 32, one of ten frozen no-shift runs reached **138.6×** — its other
> nine sat at 1.4–2.4.
>
> It is confined to **depth 32 only**, a depth added in this final job and present in no earlier
> round. At depths 2/8/16 the controls are **0 spikes in 60 runs**. Depth 32 at width 128 is an
> aspect ratio of **256:1**, far outside anything realistic, and its spike timings are also looser
> (7–34 steps after the shift, against 4–6 at the shallower depths) — consistent with a model that
> is independently unstable rather than one reacting to the shift.
>
> **So we exclude depth 32 and keep depths 2/8/16.** That is a **post-hoc scope decision**, made
> after seeing the result, and we flag it as such rather than presenting the criterion as having
> passed. A reader who rejects the exclusion should treat all of C2 as withdrawn.

### ⚠️ And freezing does *not* raise incidence — correcting our own n=4 claim

At n = 4 we reported freezing as raising incidence (3/8 frozen vs 1/8 trainable) and published that.
At n = 10 over the clean depths:

```
frozen + shift      3/30          trainable + shift      3/30
```

**Identical.** The n=4 signal was noise. On this evidence the spike needs **a sharp mixture shift and
depth**; whether the embedding is frozen makes **no measurable difference to how often it fires**.

That is a sharper correction to S7 §9 than the one we published earlier today. §9's account —
*"if it is frozen it cannot [move], and the adjustment has to happen somewhere"* — predicts freezing
should matter. Measured, it does not change incidence at all.

### The monitoring rule fails

| | verdict | evidence (n = 10) |
|---|---|---|
| global norm "averages the signal away" | **REFUTED** | median excursion is **flat** in depth: 9.2 / 8.2 / 8.0 at depths 2/8/16. It does not dilute |
| per-layer ratio is the better indicator | **REFUTED at depth** | SNR global vs bottom: 10.0 vs 8.2 at depth 2, **20.9 vs 4.4** at depth 8, **37.4 vs 3.5** at depth 16 |
| freezing is the cause | **REFUTED** | 3/30 frozen vs 3/30 trainable — identical incidence |
| watch the layers immediately above the embedding | **wrong layer** | the **last** block moves **2.01× / 2.05× / 1.71×** the first, seed ranges **disjoint** at every depth |

> **For V5: alert on the global gradient norm. If a per-layer monitor is wanted, watch the top of the
> stack, not the bottom.** And expect the event to be stochastic — a single clean run is not evidence
> that a transition is safe.

⚠️ **Limits.** Width 128, 10 seeds per cell, dense models, one shift direction (en→Indic), one
corpus. Depth 32 excluded post-hoc for the reason boxed above. The 140.8× figure is a *maximum over
10 seeds*, not a typical value — the median excursion is 8.0×, and quoting the max as though it were
the expected behaviour would be the same error S5 and S7 made with "150×".

---

## What we got wrong

Recorded because the corrections were the work.

1. **We published a width exponent of 0.868 from a 2× grid.** Refining to √2 gave 0.990. A grid whose
   spacing is a factor of 2 cannot resolve an exponent to better than ±0.15.
2. **We pre-registered "depth will not move the optimum". It moves as depth^−0.5.** The refutation is
   the C1 result.
3. **Refinement moved the depth-16 optimum from 1e-3 to 1.4e-3** — the second time refinement changed
   an answer, which is why no coarse-grid number is quoted anywhere above.
4. **We wrote a bad pre-registered criterion and it produced a false verdict.** For "does the global
   norm dilute with depth" we committed to *"supported if the value at the largest depth is below 0.8×
   its value at the smallest"* — an endpoint test. The measured sequence 11.21 → 7.93 → 8.88 is not
   monotonic, but 8.88 < 8.97 by 0.09, so the check printed SUPPORTED. **The criterion produced that
   verdict, not the data.** Withdrawn and restated as a monotonicity test.
5. **We reported C2 round 2 from 2-seed means.** The means hid a bimodal distribution: the "mean
   excursion of 22.85" at depth 8 was three seeds near 8 and one at 67. Medians and per-seed values
   are used throughout now.
6. **We hand-transcribed the Part 1 table and three of its five rows had fabricated digits.**
   The script's output was truncated in the terminal, and rather than reading the saved JSON we
   typed plausible-looking numbers. `m` at t=5 was written as `0.06663500`; it is **`0.02473500`**.
   Rows 3, 4 and 5 were wrong in `m`, `v`, `m̂` and `v̂`. The `step` and final `w` columns — the ones
   the headline claim rests on — were correct throughout, so the claim survived, but **the table
   demonstrating that we computed Adam by hand contained numbers we did not compute.**
   Found by `code/verify.py` on its first run. Every cell of that table is now checked against the
   JSON, and the table is regenerated from it rather than typed.
7. **We published a β₂ sensitivity number that was wrong, and the method behind it was wrong too.**
   The first version of §2 said *"at β₂ = 0.95 the 1.01× crossing is ~190 steps"*. It is **76**, and
   the search we used — first step *below* the threshold — returns t=1 for β₂ ≤ 0.95, because the
   ratio starts below 1 and rises through the threshold rather than decaying to it. Caught by
   actually computing the table instead of asserting one number from it.
8. **We wrote a second bad criterion, and the reordering we did for an unrelated reason broke it.**
   H8 ("incidence rises with depth") was checked as `fs[0]==0 and fs[-1]>fs[0]`. We later reordered
   the depth loop to `(16, 8, 2, 32)` so the most informative cell would finish before a possible
   timeout on a throttling host — which made `fs[0]` depth 16, not depth 2. The check printed
   **REFUTED**. Read in depth order the data is **0/10, 1/10, 2/10, 3/10**, perfectly monotonic.
   Two of our automated verdicts this session were wrong because of how the *check* was written.
9. **We published "freezing raises incidence" from n=4 and it did not survive n=10** (3/30 vs 3/30).
   Flagged the same day, in §C2.
10. **We wrote −0.0247 where the answer is −0.0246**, by subtracting two means that had already
    been rounded to four decimals instead of differencing them at full precision. A trivial error in
    size, but it is the same class as the others: a number that was *read off a printout* rather
    than recomputed. `code/verify.py` now differences the raw seed values.
11. **Our own S7 submission is an instance of the epigraph.** Every S7 arm used `lr=3e-4` because that
   is what the first script used. Measured here at width 256, `3e-4` gives 6.2988 against **5.7464**
   at the optimum — **0.55 nats**, 75× our own pre-registered seed spread of 0.0073. A later S7 job
   swept the LR but stopped at `1e-3`, where the loss was **still falling**, so it reported "untuned"
   without finding the optimum. Both failures are why every sweep here checks for an endpoint minimum
   automatically.

---

## Verification

```bash
python3 code/verify.py        # 140/140 checks, exits non-zero on any disagreement
```

`verify.py` recomputes every load-bearing number in this README from `results/` and asserts the
README literally says it, so it fails both when a computation is wrong and when the prose has
drifted from the data. It also asserts several things independently of the text: that no width or
depth minimum sits at a grid endpoint, that the 20-step empirical bias-correction run agrees with
the closed form, and that **no no-shift control spiked at any depth we kept** — the condition whose
failure would force C2 to be withdrawn; that our hand-written Adam and AdamW agree with torch
exactly; that PyTorch's ε form is algebraically identical to the textbook one; and that the
`eps_early` error grows with ε rather than staying flat.

**It found three errors on its first two runs**, all listed under "What we got wrong": a fabricated
Part 1 table, an exponent-formatting mismatch, and a delta computed from pre-rounded means. Every
one was a number that had been *read off a printout* rather than recomputed.

## Reproducing

```bash
python3 code/p1_adam_by_hand.py          # parts 1, 2          ~2 s   CPU
python3 code/p1b_adamw_and_eps.py        # parts 1a, 1b        ~2 s   CPU
python3 code/p3p4_ratio_schedules.py     # parts 3, 4          ~20 min CPU
python3 code/p3p4_followups.py           # part 4 controls, 3a, 3b  ~25 min CPU
python3 code/c2_langmap.py               # C2 language labels  ~1 min CPU
python3 code/c2_scar.py                  # C2 round 1          ~40 min CPU
python3 code/make_figures.py             # all four figures    ~5 s    CPU
```

GPU jobs (`021`–`031`) ran on an M3 Pro via `gpu-queue`; scripts and raw JSON in `results/gpu/`.
All results are deterministic given the seeds stated in each script.

## Sources

| source | what we took | grade |
|---|---|---|
| ERA V5 S11 lesson + assignment | the five required experiments, the epigraph | course material |
| **LightningLM cookbook** §13.2, §7.5, §3.3 — *Reversible Foundations* | the stage table (width 4,096 at every stage, 8→20 layers, batch 32→120); the mixture-shift failure graded at log level; the DroPE annealing passage | **primary — the V4 paper itself**, outranks transcripts |
| ERA V5 S7 lesson §9 | the monitoring rule we tested and refuted; the "150×" figure | course prose; the figure is **disowned by the cookbook**, which attaches no number |
| ERA V5 S5 lesson | the "150×" figure, same provenance | course prose |
| our S7 submission + job `018` | `lr=3e-4` untuned by 0.39 nats; the unbracketed sweep | our own measured work |
| Yang et al. 2018, *Breaking the Softmax Bottleneck* | cited for context only; **not verified by us** | peer-reviewed |

**On muP:** the cookbook never mentions it. We did not implement it, and the exponents above are for
**standard parametrisation** — under muP the width exponent would be expected near 0, which is
testable and is the obvious next experiment.
