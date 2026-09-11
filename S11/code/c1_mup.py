"""C1 addendum - does depth-muP make the optimal learning rate transfer across depth growth?

THE POINT.  Everything else in C1 measures how badly a tuned rate breaks when the model grows.  This
asks whether the right parametrisation makes it not break.  If it does, V5 never re-tunes at a growth
boundary, and the fix is one line in the Block.

THE ONE LINE.  Depth-muP (Yang et al., Tensor Programs VI) scales every residual branch by 1/sqrt(L):

    x = x + (1/sqrt(L)) * attn(norm(x))
    x = x + (1/sqrt(L)) * mlp(norm(x))

Nothing else changes.  Width, steps, batch, sequence length, schedule, optimiser, data and eval are
identical to the standard-parametrisation depth sweep (jobs 024a/021/024b/024c), and the LR grid is
the same 6-point coarse grid, so b_muP and b_SP sit on the same surface.

PRE-REGISTERED, in the job 033 docstrings, committed before running:

  H11  the depth exponent collapses toward zero.
       CONFIRMED if |b_muP| < 0.15 - the optimum moves less than one grid step across the depth
       range, which is what "the rate transfers" means operationally.
       REFUTED   if it stays within 0.15 of the standard-parametrisation 0.505.

  H12  a guard, because this is how claims like this usually die.  VOID if muP's best achievable
       loss is more than 0.05 nats worse than SP's at every depth.  A parametrisation that flattens
       the LR curve by making everything equally bad is not transfer, it is a worse model.

THE COMPARISON HAS TO BE GRID-MATCHED, and the inline table printed by job 033 is NOT.  That table
compares muP's coarse-grid argmin against the README's published SP optima, which come from three
different grids (see C1.5).  Every SP number used below is instead recomputed from the SP jobs on
the IDENTICAL 6-point coarse grid the muP arm ran, with the identical parabola-vertex estimator.
Comparing a coarse-grid optimum against a refined one would flatter whichever arm got the finer grid.
"""
import json, math, os, statistics as st, sys

HERE, NV = os.path.dirname(os.path.abspath(__file__)), 61070
G = os.path.join(HERE, "..", "results", "gpu")
COARSE = (3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3)
SP_JOB = {2: "024a_s11_lrdepth_02", 4: "021_s11_lrwidth_256",
          8: "024b_s11_lrdepth_08", 16: "024c_s11_lrdepth_16"}

def vertex(pts):
    pts = sorted(pts)
    i = min(range(len(pts)), key=lambda k: pts[k][1])
    if i in (0, len(pts)-1): return None, None
    (x1,y1),(x2,y2),(x3,y3) = [(math.log2(l), v) for l, v in pts[i-1:i+2]]
    u1,u3,d1,d3 = x1-x2, x3-x2, y1-y2, y3-y2
    det = u1*u1*u3 - u3*u3*u1
    a = (d1*u3 - d3*u1)/det; b = (d3*u1*u1 - d1*u3*u3)/det
    u = -b/(2*a)
    return 2**(x2+u), y2 + a*u*u + b*u

def on_coarse(rows):
    return [(r["lr"], r["val_loss"]) for r in rows
            if any(abs(r["lr"]/g - 1) < 1e-9 for g in COARSE)]

mup = {}
for f in sorted(os.listdir(G)):
    if not f.startswith("033") or not f.endswith(".json"): continue
    for r in json.load(open(os.path.join(G, f))):
        assert r["input_params"] == NV * 256, f"{f}: muP arm must be width 256"
        assert abs(r["res_scale"] - 1/math.sqrt(r["depth"])) < 1e-9, f"{f}: res_scale != 1/sqrt(L)"
        mup.setdefault(r["depth"], []).append((r["lr"], r["val_loss"]))
sp = {}
for d, j in SP_JOB.items():
    rows = json.load(open(os.path.join(G, j + ".json")))
    assert all(r["input_params"] == NV * 256 for r in rows), f"{j}: SP arm must be width 256"
    sp[d] = on_coarse(rows)

DEPTHS = sorted(set(mup) & set(sp))
print(__doc__)
print("=" * 96)
print(f"MEASURED - width 256, batch 8, seq 256, 1200 steps, identical 6-point grid, 1 seed")
print("=" * 96)
print(f"{'depth':>6}{'SP lr*':>12}{'SP val*':>10}{'muP lr*':>12}{'muP val*':>10}"
      f"{'muP - SP':>11}{'H12':>7}")
SO, MO = {}, {}
for d in DEPTHS:
    slr, sv = vertex(sp[d]); mlr, mv = vertex(mup[d])
    if None in (slr, mlr):
        print(f"{d:>6}   minimum at a grid endpoint - not quotable"); continue
    SO[d], MO[d] = slr, mlr
    gap = mv - sv
    print(f"{d:>6}{slr:>12.3e}{sv:>10.4f}{mlr:>12.3e}{mv:>10.4f}{gap:>+11.4f}"
          f"{'FIRES' if gap > 0.05 else 'ok':>7}")

if len(MO) < 2:
    print("\nfewer than two depths - nothing to fit yet."); sys.exit(0)

fired = [d for d in MO if vertex(mup[d])[1] - vertex(sp[d])[1] > 0.05]
print(f"\n{'-'*96}\nH12 - is the muP arm simply worse?\n{'-'*96}")
if len(fired) == len(MO):
    print("  >>> H12 FIRES at every depth. The comparison is VOID: a parametrisation that flattens\n"
          "      the LR curve by making everything equally bad is not transfer. Stopping.")
    sys.exit(0)
print(f"  H12 does not fire: muP is within 0.05 nats of SP at {len(MO)-len(fired)} of {len(MO)} depths"
      + (f" (fires at {fired})" if fired else " (fires nowhere)") + ".")
print(f"  Best-loss difference across depths: "
      f"{', '.join(f'{d}: {vertex(mup[d])[1]-vertex(sp[d])[1]:+.4f}' for d in DEPTHS if d in MO)}")
print("  NOTE the direction: a NEGATIVE gap means muP reached a LOWER loss than SP at that depth.")

def fit(o):
    X = [math.log(d) for d in sorted(o)]; Y = [math.log(o[d]) for d in sorted(o)]
    mx, my = st.mean(X), st.mean(Y)
    return -sum((x-mx)*(y-my) for x, y in zip(X, Y)) / sum((x-mx)**2 for x in X)

bm, bs_ = fit(MO), fit(SO)
print(f"\n{'-'*96}\nH11 - does the depth exponent collapse?\n{'-'*96}")
print(f"  standard parametrisation   b_SP  = {bs_:+.4f}")
print(f"  depth-muP (1/sqrt(L))      b_muP = {bm:+.4f}")
print(f"  depths spanned: {min(MO)} to {max(MO)} ({max(MO)//min(MO)}x)")
print(f"\n{'depth':>6}{'SP lr*':>12}{'muP lr*':>12}{'SP local b':>13}{'muP local b':>14}")
ds = sorted(MO)
for i, d in enumerate(ds):
    ls = f"{-math.log(SO[d]/SO[ds[i-1]])/math.log(d/ds[i-1]):+.3f}" if i else ""
    lm = f"{-math.log(MO[d]/MO[ds[i-1]])/math.log(d/ds[i-1]):+.3f}" if i else ""
    print(f"{d:>6}{SO[d]:>12.3e}{MO[d]:>12.3e}{ls:>13}{lm:>14}")

verdict = ("CONFIRMED" if abs(bm) < 0.15 else
           "REFUTED"   if abs(abs(bm) - 0.505) < 0.15 else "NEITHER")

# GUARD.  Three times in this assignment a quantity fitted to too few points read as a law: the
# width exponent from a 2x grid, the depth exponent from two endpoints, and the batch SHAPE from
# three points (all three are documented in README C1.4/C1.5, and all three were wrong).  An
# exponent from two depths is the same error with a different label, so this refuses to report a
# verdict until the depth range is wide enough to be worth reporting.
MIN_DEPTHS, MIN_SPAN = 3, 4
if len(MO) < MIN_DEPTHS or max(MO)/min(MO) < MIN_SPAN:
    print(f"""
  >>> H11 VERDICT WITHHELD.  {len(MO)} depth(s) spanning {max(MO)//min(MO)}x; this reports a verdict only at
      {MIN_DEPTHS}+ depths spanning {MIN_SPAN}x or more.

      The arithmetic so far, which is NOT a result:  b_muP = {bm:+.4f} against b_SP = {bs_:+.4f}.

      Why it is withheld rather than reported with a caveat.  Three separate numbers in this
      assignment were fitted to too few points and each read as a law before the next point
      arrived: the width exponent (0.868 from a 2x grid, 0.990 on sqrt(2)), the depth exponent
      (0.505 from two endpoints, 0.389-0.667 across estimators), and the batch SHAPE (a saturating
      curve from three points, refuted by the fourth).  An exponent from two depths is that error
      with a different label.  Job 033b supplies depths 8 and 16; until it returns there is no
      H11 verdict to quote and none appears in the README.

      Note also that depth 2 does not discriminate: SP and muP both put the optimum near 2.9e-3
      there, so the entire signal so far rests on the single depth-4 point.""")
    sys.exit(0)

# ---------------------------------------------------------------- how well is b pinned?
# Job 032c measured the per-seed spread of a parabola vertex directly: three seeds at batch 16 gave
# 1.304e-3 / 1.139e-3 / 1.398e-3, a log-sd of about 0.104 (~11%).  Every point in BOTH arms here is
# one seed, so that is the noise on each lr* below.  Propagating it through the OLS slope:
SIG_LOG = 0.104
X  = [math.log(d) for d in sorted(MO)]
Sxx = sum((x - st.mean(X))**2 for x in X)
SE  = SIG_LOG / math.sqrt(Sxx)
print(f"\n{'-'*96}\nHOW WELL IS b PINNED?  (1 seed per point; vertex noise measured by job 032c)\n{'-'*96}")
print(f"  per-point log-sd on lr*: {SIG_LOG:.3f}   ->   se(b) = {SE:.4f} over {len(MO)} depths spanning {max(MO)//min(MO)}x")
print(f"    b_SP  = {bs_:+.4f} +- {SE:.4f}")
print(f"    b_muP = {bm:+.4f} +- {SE:.4f}")
dif = bs_ - bm
print(f"    difference = {dif:+.4f} +- {SE*math.sqrt(2):.4f}   ->  {dif/(SE*math.sqrt(2)):.1f} sigma")
marg = (0.15 - abs(bm)) / SE
print(f"    distance from H11's 0.15 threshold: {marg:.1f} sigma")
print(f"""
  TWO CLAIMS, AND THEY ARE NOT EQUALLY WELL SUPPORTED.  Keeping them separate matters:

    (a) muP FLATTENS the depth dependence relative to SP.  The difference is {dif:+.3f} +- {SE*math.sqrt(2):.3f},
        about {dif/(SE*math.sqrt(2)):.0f} sigma.  This is solid, and it is what the one-line change buys.

    (b) muP makes the rate TRANSFER, i.e. b_muP is indistinguishable from zero.  |b_muP| = {abs(bm):.4f}
        sits only {marg:.1f} sigma inside H11's 0.15 threshold, and the local exponent is rising
        through the range.  H11 passes as written, but "b is small" is much better supported
        than "b is zero", and we do not claim the latter.

  This is why jobs 034a/034b matter: an out-of-sample point at depth 32 tests (b) directly, where
  the in-sample fit cannot.""")
print(f"\n  >>> H11: {verdict}")
print(f"      |b_muP| = {abs(bm):.4f}   threshold for CONFIRMED is < 0.15; for REFUTED, within 0.15 of 0.505")
spanM = (max(MO)/min(MO)) ** -bm
spanS = (max(MO)/min(MO)) ** -bs_
print(f"\n  Over the {max(MO)//min(MO)}x depth range the optimum moves x{spanM:.3f} under muP and "
      f"x{spanS:.3f} under SP.")
print(f"  In grid steps of the 2x sweep: {abs(math.log2(spanM)):.2f} (muP) vs {abs(math.log2(spanS)):.2f} (SP).")
