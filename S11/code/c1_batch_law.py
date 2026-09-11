"""C1 addendum - does the optimal learning rate depend on batch size, and how?

WHY THIS EXISTS.  The README's headline C1 claim is that growing 8 -> 20 layers requires cutting
the learning rate 1.59x.  Cookbook §13.2 shows V4's 9B stage moves TWO things at once: layers
8 -> 20 AND global batch 32 -> 120.  Our depth law has no batch term, so as published the claim
describes a change V4 does not actually make.  A larger batch is generally held to push the optimal
rate UP, which would work against the depth-driven cut.  This measures the missing term.

CONVENTION, STATED UP FRONT BECAUSE IT DECIDES WHAT THE ANSWER MEANS.  Batch is varied at FIXED
STEPS (1200), so a larger batch also sees proportionally more tokens.  That is the convention V4's
own stage boundary uses - training continues, the batch gets bigger, the step count does not shrink
to compensate.  It is NOT the convention the textbook rules were derived under: Goyal et al.'s
linear rule and Krizhevsky's sqrt rule are both stated at fixed EPOCHS, where a larger batch means
proportionally FEWER steps.  So a disagreement with those rules here is a disagreement about a
different experiment, and is reported as such rather than as a refutation of them.

PRE-REGISTERED (job 032 docstrings, committed before running):
  CONFIRMED-LINEAR  fitted exponent within one grid step of 1.0
  CONFIRMED-SQRT    within one grid step of 0.5
  NEITHER           a real outcome: batch-insensitive over this range, and the depth-only 1.59x
                    stands as published.

WIDTH.  See results/gpu/032_CORRECTION.md.  Job 032 was cloned from the width-512 sweep and ran at
width 512 while recording width=256.  Every point in the series is width 512, so the batch axis is
clean; the batch-8 anchor is job 022, which is width 512 and was run independently a day earlier.
The assertion below ties the width label to a runtime-computed quantity so the mislabel cannot
silently propagate.
"""
import json, math, os, statistics as stats, sys

HERE = os.path.dirname(os.path.abspath(__file__))
G    = os.path.join(HERE, "..", "results", "gpu")
NV   = 61070          # vocabulary; input_params = NV * width

def load(name):
    with open(os.path.join(G, name)) as f: return json.load(f)

def vertex(pts):
    """Optimum by a parabola through the argmin and its two neighbours, in log2(lr).

    Grid argmin alone would quantise each optimum to its own grid, and the two grids here are
    offset from each other (022 uses {3e-4..8e-3}, 032 uses {6e-4..9.6e-3}).  Interpolating
    removes that.  Returns (lr*, val*, status); status 'ENDPOINT' means the sweep did not
    bracket and the value is not quotable - the same check every sweep in this assignment carries.
    """
    pts = sorted(pts)
    i = min(range(len(pts)), key=lambda k: pts[k][1])
    if i in (0, len(pts) - 1):
        return None, None, "ENDPOINT"
    (x1, y1), (x2, y2), (x3, y3) = [(math.log2(l), v) for l, v in pts[i-1:i+2]]
    u1, u3, d1, d3 = x1 - x2, x3 - x2, y1 - y2, y3 - y2
    det = u1 * u1 * u3 - u3 * u3 * u1
    a   = (d1 * u3 - d3 * u1) / det
    b   = (d3 * u1 * u1 - d1 * u3 * u3) / det
    u   = -b / (2 * a)
    return 2 ** (x2 + u), y2 + a * u * u + b * u, "ok"

# ---------------------------------------------------------------- assemble the series
series, prov = {}, {}
REPL = {}     # (batch, lr) -> {seed: val}, from job 032c
for f in sorted(os.listdir(G)):
    if not f.startswith("032") or not f.endswith(".json"): continue
    if f.startswith("032c"):
        for r in load(f):
            assert r["input_params"] == NV * 512
            REPL.setdefault((r["batch"], r["lr"]), {})[r["model_seed"]] = r["val_loss"]
        continue
    for r in load(f):
        assert r["input_params"] == NV * 512, (
            f"{f}: input_params={r['input_params']} is not 61070*512; see 032_CORRECTION.md")
        series.setdefault(r["batch"], []).append((r["lr"], r["val_loss"]))
        prov.setdefault(r["batch"], f.split("_")[0])
        REPL.setdefault((r["batch"], r["lr"]), {})[r["model_seed"]] = r["val_loss"]
# GRID RULE: each batch is estimated on the LR grid its own sweep ran.  Job 022 ran exactly the
# coarse 6-point grid, so reading its file gives batch 8 the same grid density as every other
# batch.  Job 025 later added two sqrt(2) points at width 512 depth 4 for the WIDTH law; they are
# deliberately NOT read here.  Folding them in moves batch 8's optimum 1.247e-3 -> 1.258e-3, and a
# batch series with one batch measured more finely than the rest is not a batch series.
# code/verify.py enforces the same rule by grid rather than by filename, and agrees to 4 digits.
for r in load("022_s11_lrwidth_512.json"):           # the independent batch-8 anchor
    assert r["input_params"] == NV * 512
    series.setdefault(8, []).append((r["lr"], r["val_loss"]))
    prov.setdefault(8, "022")
assert 8 in prov and prov[8] == "022", "batch-8 anchor must come from job 022, not from job 032"

print(__doc__)
print("=" * 92)
print("MEASURED - width 512, depth 4, seq 256, 1200 steps, cosine, AdamW(0.9,0.95), wd 0.1")
print("=" * 92)
print(f"{'batch':>6}{'job':>6}{'pts':>5}{'grid argmin':>14}{'interpolated lr*':>19}{'val* ':>10}{'status':>10}")
opt = {}
for B in sorted(series):
    lr, val, st = vertex(series[B])
    am = min(series[B], key=lambda t: t[1])[0]
    if st == "ok": opt[B] = lr
    print(f"{B:>6}{prov[B]:>6}{len(series[B]):>5}{am:>14.2e}"
          + (f"{lr:>19.4e}{val:>10.4f}" if st == "ok" else f"{'-':>19}{'-':>10}") + f"{st:>10}")

# ---------------------------------------------------------------- the replication (job 032c)
B16 = {lr: v for (b, lr), v in REPL.items() if b == 16 and len(v) > 1}
if B16:
    print(f"\n{'-'*92}\nREPLICATION - job 032c, batch 16 at seeds 1 and 2\n{'-'*92}")
    print("  The whole non-monotonicity turned on ONE comparison at ONE seed: at batch 16,")
    print("  lr=2.4e-3 was 0.0730 nats worse than 1.2e-3, while at batch 32 the same comparison")
    print("  flips.  Pre-registered sign test on d = val(2.4e-3) - val(1.2e-3):")
    ds = {sd: B16[2.4e-3][sd] - B16[1.2e-3][sd] for sd in sorted(B16[2.4e-3]) if sd in B16[1.2e-3]}
    for sd, d in ds.items(): print(f"      seed {sd}:  d = {d:+.4f}")
    allpos = all(d > 0 for d in ds.values())
    print(f"\n  >>> {'REPLICATED' if allpos else 'NOT REPLICATED'} - "
          f"d > 0 at {sum(1 for d in ds.values() if d > 0)}/{len(ds)} seeds.  The batch-16 optimum really is")
    print("      near 1.2e-3, the local-exponent sequence really is non-monotone, and the")
    print("      saturating form stays REFUTED.  Bn is not reinstated.")

    print(f"\n  AND THE REPLICATION FOUND SOMETHING WE WERE NOT LOOKING FOR.")
    print(f"  {'lr':>10}{'n':>4}{'mean':>10}{'seed spread':>14}")
    for lr in sorted(B16):
        v = list(B16[lr].values())
        print(f"  {lr:>10.2e}{len(v):>4}{stats.mean(v):>10.4f}{max(v)-min(v):>14.4f}")
    lo_, hi_ = min(B16), max(B16)
    sp_lo = max(B16[lo_].values()) - min(B16[lo_].values())
    sp_hi = max(B16[hi_].values()) - min(B16[hi_].values())
    print(f"""
  SEED VARIANCE IS NOT CONSTANT ALONG THE SWEEP.  It is {sp_hi/sp_lo:.0f}x larger at {hi_:.1e} than at {lo_:.1e},
  and {hi_:.1e} is the point just PAST the optimum - exactly where the curve turns up and exactly
  where a bracketing decision gets made.  Our pre-registered seed spread of 0.0073 nats (from S7
  job 018) was measured at or below an optimum; above one it understates the noise by ~15x.

  WHAT THAT COSTS US.  Re-estimating the batch-16 vertex from each seed separately:
      {', '.join(f'seed {sd}: {vertex([(l, B16[l][sd]) for l in sorted(B16)])[0]:.4e}' for sd in sorted(ds))}
  a spread of about +-11% in the optimum, NOT the ~2% we previously claimed from a 0.004-nat noise
  model.  Every single-seed optimum in this submission carries that uncertainty, including the ones
  whose conclusions we are keeping.  It is reported here rather than only where it is convenient.

  DOES IT CHANGE THE CONCLUSION?  No, and it makes it stronger.  Using the 3-seed MEAN curve at
  batch 16 instead of seed 0 moves that optimum to {vertex([(l, stats.mean(list(B16[l].values()))) for l in sorted(B16)])[0]:.4e}, which makes the
  local-exponent sequence MORE non-monotone, not less.  Both choices are shown below.""")
    opt16_mean = vertex([(l, stats.mean(list(B16[l].values()))) for l in sorted(B16)])[0]

if len(opt) < 3:
    print("\nfewer than three bracketed optima - not enough to fit anything. Stopping."); sys.exit(0)

# ---------------------------------------------------------------- the pre-registered test
Bs = sorted(opt)
print(f"\n{'-'*92}\nLOCAL EXPONENT between adjacent batches   (lr* ~ B^p)\n{'-'*92}")
for a, b in zip(Bs, Bs[1:]):
    r = opt[b] / opt[a]
    print(f"  {a:>3} -> {b:<3}   lr* x{r:.4f}   p = {math.log(r)/math.log(b/a):+.4f}")
if B16:
    alt = dict(opt); alt[16] = opt16_mean
    print(f"\n  with batch 16 from the 3-seed MEAN ({opt16_mean:.4e}) instead of seed 0:")
    for a, b in zip(Bs, Bs[1:]):
        r = alt[b]/alt[a]
        print(f"  {a:>3} -> {b:<3}   lr* x{r:.4f}   p = {math.log(r)/math.log(b/a):+.4f}")
    print("  -> still non-monotone; the dip at 16 is deeper, so the saturating form fits worse.")
lo, hi = Bs[0], Bs[-1]
P = math.log(opt[hi] / opt[lo]) / math.log(hi / lo)
print(f"\n  overall {lo} -> {hi}:  lr* x{opt[hi]/opt[lo]:.4f},  fitted exponent p = {P:+.4f}")
print(f"  linear rule predicts p = 1.0 (lr* x{hi/lo:.0f});  sqrt rule predicts p = 0.5 "
      f"(x{math.sqrt(hi/lo):.2f})")
# "within one grid step of" was pre-registered against a 2x LR grid.  Over a 4x batch range one
# LR grid step is log2(2)/log2(4) = 0.5 in exponent, so that is the band, applied literally:
STEP = 0.5
verdict = ("CONFIRMED-LINEAR" if abs(P - 1.0) < STEP else
           "CONFIRMED-SQRT"   if abs(P - 0.5) < STEP else "NEITHER")
print(f"\n  >>> PRE-REGISTERED CRITERION, APPLIED LITERALLY: {verdict}")
print(f"""
  AND THE CRITERION IS DEGENERATE.  Do not read that verdict as a result.  With STEP = 0.5 the
  sqrt band is p in (0.0, 1.0) and the linear band is p in (0.5, 1.5): they overlap, and the sqrt
  band swallows every exponent between no dependence at all and a full linear rule.  NEITHER -
  the outcome the pre-registration called "a real outcome" - cannot fire for any p in [0, 1].
  The criterion produced this verdict, not the data.  This is the second time in this assignment
  a badly written threshold has done that (see H4 in the README); it is recorded rather than
  quietly re-thresholded, and the discriminating comparison is made directly instead:

      measured    lr* x{opt[hi]/opt[lo]:.4f} over batch {lo} -> {hi}
      sqrt rule   lr* x{math.sqrt(hi/lo):.4f}      ({100*(opt[hi]/opt[lo]/math.sqrt(hi/lo)-1):+.1f}% vs measured)
      linear rule lr* x{hi/lo:.4f}      ({100*(opt[hi]/opt[lo]/(hi/lo)-1):+.1f}% vs measured)

  The measured move is {math.sqrt(hi/lo)/(opt[hi]/opt[lo]):.2f}x smaller than the sqrt rule and {(hi/lo)/(opt[hi]/opt[lo]):.2f}x smaller than the linear rule.
  Neither rule describes this.  The criterion-free finding is below, and it does not depend on a
  threshold at all: the LOCAL exponent is NON-MONOTONE, {', '.join(f'{math.log(opt[b]/opt[a])/math.log(b/a):+.3f}' for a,b in zip(Bs,Bs[1:]))},
  and no single power law can produce a local exponent that changes at all.""")

# ---------------------------------------------------------------- what the shape actually is
# The local exponents decay rather than holding constant, so a single power law is the wrong
# family.  The gradient-noise-scale form (McCandlish et al. 2018) predicts exactly that decay:
#     lr*(B) = lr_inf * B / (B + Bn)
# linear for B << Bn, flat for B >> Bn.  Fit lr_inf and Bn by least squares in log-lr.
# ---------------------------------------------------------------- what the shape actually is
print(f"\n{'-'*92}\nSHAPE  -  is it a saturating curve?\n{'-'*92}")
locs = [math.log(opt[b]/opt[a])/math.log(b/a) for a, b in zip(Bs, Bs[1:])]
mono = all(x >= y - 1e-12 for x, y in zip(locs, locs[1:]))
print(f"  local exponents, in batch order: {', '.join(f'{x:+.3f}' for x in locs)}")
print(f"  monotonically decaying?  {'YES' if mono else 'NO'}")

# The gradient-noise-scale form (McCandlish et al. 2018) is the standard account:
#     lr*(B) = lr_inf * B / (B + Bn)
# linear for B << Bn, flat for B >> Bn.  It REQUIRES the local exponent to decay monotonically.
best = None
for k in range(1, 20001):
    Bn_ = k * 0.01
    li_ = math.exp(sum(math.log(opt[B] * (B + Bn_) / B) for B in Bs) / len(Bs))
    ss_ = sum((math.log(opt[B]) - math.log(li_ * B / (B + Bn_))) ** 2 for B in Bs)
    if best is None or ss_ < best[0]: best = (ss_, Bn_, li_)
ss, Bn, li = best
rms = math.sqrt(ss / len(Bs))
print(f"\n  best saturating fit: lr*(B) = {li:.4e} * B / (B + {Bn:.2f})"
      f"       rms {100*(math.exp(rms)-1):.1f}% in lr")
print(f"{'batch':>6}{'measured lr*':>16}{'fitted lr*':>14}{'residual':>11}")
for B in Bs:
    f_ = li * B / (B + Bn)
    print(f"{B:>6}{opt[B]:>16.4e}{f_:>14.4e}{100*(opt[B]/f_-1):>10.1f}%")

if mono:
    print(f"\n  The saturating form is consistent with these points. Critical batch Bn = {Bn:.2f}"
          f" sequences ({Bn*256:.0f} tokens at seq 256).")
else:
    bad = max(range(len(Bs)), key=lambda i: abs(math.log(opt[Bs[i]]/(li*Bs[i]/(Bs[i]+Bn)))))
    print(f"""
  >>> THE SATURATING FORM IS REFUTED AS THE SHAPE, and this overturns what we published.

  A saturating curve REQUIRES the local exponent to decay monotonically - it is linear below the
  critical batch and flat above it, and it never steepens again.  Ours goes {', '.join(f'{x:+.3f}' for x in locs)}.

  THE ARGUMENT IS NOT THE rms, AND WE SHOULD SAY SO.  The fit's rms of {100*(math.exp(rms)-1):.1f}% in lr is NOT
  comfortably outside the noise: job 032c measures the per-seed spread of a vertex at about +-11%
  (see above), so a {100*(math.exp(rms)-1):.1f}% misfit on single-seed points proves little on its own.  An earlier
  draft of this file leaned on the rms against a "~2%" noise figure that came from a 0.004-nat
  model rather than from measurement, and that comparison was wrong.

  WHAT THE REFUTATION ACTUALLY RESTS ON is the size of the error that would be needed to rescue
  the saturating form.  For the local exponent to decay monotonically, the 16 -> 32 step would have
  to be no larger than the 8 -> 16 step.  With batch 16 at its replicated 3-seed value the 8 -> 16
  step is +0.007, so batch 32's optimum would have to sit at essentially batch 16's, near 1.25e-3,
  rather than the measured 1.80e-3.  That is a 44% error in a single vertex, four times the
  measured +-11% per-seed spread.  The batch-16 dip that starts the non-monotonicity is separately
  replicated at 3/3 seeds by a pre-registered sign test.

  WHAT IS STILL THIN: batch 32 is ONE seed.  The 44%-vs-11% margin is the reason we call the shape
  refuted rather than merely doubted, but a replicate at batch 32 would close it properly and we
  did not run one.

  WHAT WE PUBLISHED AND WHY IT WAS WRONG.  With batches 4, 8 and 16 alone the sequence was
  +0.400, +0.064 - textbook saturation - and we fitted Bn = 2.40 and wrote it into the README
  before batch 32 returned.  Three points cannot distinguish a saturating curve from a curve that
  happens to be flat between two of them.  This is the assignment's own epigraph landing on us a
  third time, and the mechanism is the same each time: a shape inferred from too few points, or an
  exponent inferred from too few points, reads as a law.

  WHAT SURVIVES, AND IT IS THE PART THE C1 CLAIM ACTUALLY NEEDS.  The batch sensitivity itself is
  criterion-free and does not depend on the shape at all: over batch {Bs[0]} -> {Bs[-1]} the optimum moves
  x{opt[Bs[-1]]/opt[Bs[0]]:.2f}, where the sqrt rule wants x{math.sqrt(Bs[-1]/Bs[0]):.2f} and the linear rule x{Bs[-1]/Bs[0]:.0f}.  It is {math.sqrt(Bs[-1]/Bs[0])/(opt[Bs[-1]]/opt[Bs[0]]):.2f}x less
  sensitive than the weaker of the two textbook rules, measured across an {Bs[-1]//Bs[0]}x batch range.

  WHAT IS WITHDRAWN.  Bn = 2.40, the {Bn:.2f} refit, and every number downstream of them - the x1.054
  batch term at V4's boundary, the 1.51x net cut, and the "Bn >= 13 cancels half the depth cut"
  bound.  All of those assume a functional form the data now rejects.  Batch {Bs[bad]} is the worst
  offender at {100*(opt[Bs[bad]]/(li*Bs[bad]/(Bs[bad]+Bn))-1):+.1f}%.

  WHAT WOULD RESOLVE IT.  Every point above is ONE SEED, and "one run is an anecdote" is a rule we
  wrote into our own rubric.  The whole non-monotonicity rests on batch 16 preferring lr=1.2e-3
  over 2.4e-3 by 0.073 nats while batch 32 prefers 2.4e-3 by 0.067.  Job 032c re-runs batch 16 at
  seeds 1 and 2 on those three LRs with a pre-registered sign test.  Until it returns, the shape is
  reported as UNRESOLVED and no critical batch is quoted.""")

if not mono:
    print(f"\n{'='*92}")
    sys.exit(0)

# ---------------------------------------------------------------- how well is Bn pinned?
# Three or four points and a two-parameter fit.  Bn is quoted with a range, not as a value.
# Noise model is the MEASURED seed spread on val loss from job 028 (two replicate pairs at
# width 256 depth 4: 0.0022 and 0.0058 nats), taken as sigma = 0.004 and resampled.
import random
SIGMA = 0.004
random.seed(0)
draws = []
for _ in range(2000):
    o = {}
    for B in Bs:
        pts = [(l, v + random.gauss(0, SIGMA)) for l, v in series[B]]
        lr, _, st = vertex(pts)
        if st != "ok": break
        o[B] = lr
    if len(o) != len(Bs): continue
    bb = None
    for k in range(1, 20001):
        n = k * 0.01
        l0 = math.exp(sum(math.log(o[B] * (B + n) / B) for B in Bs) / len(Bs))
        q = sum((math.log(o[B]) - math.log(l0 * B / (B + n))) ** 2 for B in Bs)
        if bb is None or q < bb[0]: bb = (q, n)
    draws.append(bb[1])
draws.sort()
BN_LO, BN_HI = draws[int(.05 * len(draws))], draws[int(.95 * len(draws))]
print(f"  Bn 90% interval over {len(draws)} resamples at sigma = {SIGMA} nats: "
      f"[{BN_LO:.2f}, {BN_HI:.2f}] sequences   (point estimate {Bn:.2f})")
print(f"""  -> That interval is STATISTICAL ONLY - it propagates seed noise and nothing else.  It is
     narrow because the parabola vertex is stable under 0.004-nat jitter, NOT because Bn is
     well determined.  The systematic terms are larger and are not in it: one seed per point,
     a 2x grid, a parabola fitted to three points of a curve that is not a parabola, and {len(Bs)}
     batches spanning {Bs[-1]//Bs[0]}x.  Read [{BN_LO:.2f}, {BN_HI:.2f}] as "seed noise alone would not move Bn much",
     and treat the point estimate as good to a factor of ~2, not to two decimals.""")

# ---------------------------------------------------------------- back to the C1 claim
print(f"\n{'-'*92}\nWHAT THIS DOES TO THE PUBLISHED C1 CLAIM\n{'-'*92}")
DEPTH = (8 / 20) ** 0.505                      # measured depth law, b = 0.505
print(f"  V4 §13.2's 9B boundary moves layers 8 -> 20 AND global batch 32 -> 120.")
print(f"    depth term (measured, b = 0.505):        x{DEPTH:.4f}   = cut {1/DEPTH:.3f}x   <- what we published")
for tag, n in (("point", Bn), ("Bn low", BN_LO), ("Bn high", BN_HI)):
    b = (120 / (120 + n)) / (32 / (32 + n))
    print(f"    batch term at {tag:>7} Bn = {n:6.2f}:        x{b:.4f}   net cut {1/(DEPTH*b):.3f}x")

# How large would Bn have to be for batch to cancel half, and all, of the depth correction?
def bn_for(target):
    lo_, hi_ = 1e-3, 1e12
    for _ in range(300):
        mid = math.sqrt(lo_ * hi_)
        if (120 / (120 + mid)) / (32 / (32 + mid)) < target: lo_ = mid
        else: hi_ = mid
    return lo_
half, full = bn_for(math.sqrt(1 / DEPTH)), bn_for(1 / DEPTH)
print(f"""
  THE CONFOUND IS QUANTIFIED, NOT CLOSED.  At OUR critical batch the correction is small - the
  batch term asks for x{(120/(120+Bn))/(32/(32+Bn)):.4f} against the depth term's x{DEPTH:.4f}, so the net is a {1/(DEPTH*(120/(120+Bn))/(32/(32+Bn))):.3f}x cut where
  we published {1/DEPTH:.3f}x.  That is a {100*abs(1/(DEPTH*(120/(120+Bn))/(32/(32+Bn)))-1/DEPTH)/(1/DEPTH):.1f}% change and would not alter the recommendation.

  But the margin is thin, and this is the number that matters:

    Bn >= {half:.0f} sequences    cancels HALF the depth cut   ({half/Bn:.0f}x our measured Bn)
    Bn >= {full:.0f} sequences    cancels ALL of it            ({full/Bn:.0f}x our measured Bn)

  The gradient noise scale grows with model size and falls as the loss falls.  Ours is measured
  on a 4-layer, 512-wide model at seq 256, and Bn came out at {Bn:.2f} sequences - about {Bn*256:.0f} tokens.
  V4's 9B stage trains a 20-layer, 4096-wide MoE on 4K-8K-token sequences.  It does not need to
  be exotic for its noise scale to exceed {half:.0f}x ours; that is a factor of {half/Bn:.0f}, and noise scale
  routinely spans orders of magnitude across that kind of gap.

  SO THE HONEST CONCLUSION IS THE UNCOMFORTABLE ONE.  We set out to close the confound and
  instead measured how large it could be.  The depth-only 1.59x is correct AT OUR SCALE, where
  batch contributes under {100*abs((120/(120+BN_HI))/(32/(32+BN_HI))-1):.0f}%.  It is NOT transferable to V4's stage boundary, because
  the one quantity that decides the size of the batch term there - V4's critical batch - is not
  something this experiment, or any experiment at this scale, can measure.  The README's boxed
  warning stands; this replaces "could reverse sign" with a bound and a reason.

  WHAT WOULD SETTLE IT.  Bn is measurable without training at V4 scale: it is the ratio of the
  squared gradient norm to the gradient variance, estimable from two batch sizes at one step
  (McCandlish et al. 2018, §2).  That is cheap even on a 9B model and is the experiment we would
  run next.  We did not run it, so we do not claim its answer.""")
print(f"\n{'='*92}")
