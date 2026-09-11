"""How much of "lr* proportional to 1/(width*sqrt(depth))" is the data, and how much is the estimator?

The assignment asks for the LR at width 4,096 "and how confident you are in it".  The README answers
with exponents a = 0.990 and b = 0.505 and a growth-boundary factor of 1.59x.  Both exponents are
computed in verify.py as TWO-POINT ENDPOINT estimates:

    a = log(OW[256] / OW[1024]) / log(4)
    b = log(OD[2]  / OD[16])   / log(8)

That throws away the interior points entirely and puts the whole exponent on the two noisiest
places in the sweep - the ends.  It is the same fragility that made criterion H4 print SUPPORTED off
a 0.09-nat endpoint difference, and it is worth checking rather than assuming.

A second, independent problem: the four depth optima the README quotes do not come from the same
grid.  Depths 2 and 8 use the coarse 2x grid; depth 4 uses a sqrt(2) refinement (job 025) and depth
16 a sqrt(2) point (job 028).  A per-depth optimum estimated from a finer grid is not comparable to
one estimated from a coarser grid, and the tied-geometric-mean estimator is explicitly
grid-sensitive: more grid points inside the seed spread means more terms in the mean.

So: recompute both exponents under every combination of {estimator} x {grid}, and report the spread
as the confidence interval it actually is.  Nothing here is new data.  Every number below comes from
JSON already archived in results/gpu/.
"""
import json, math, os, statistics as st, sys

HERE   = os.path.dirname(os.path.abspath(__file__))
G      = os.path.join(HERE, "..", "results", "gpu")
SPREAD = 0.0073            # pre-registered seed spread, S7 job 018
COARSE = (3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3)

DEPTH_OF_FILE = {"021_s11_lrwidth_256": 4, "022_s11_lrwidth_512": 4,
                 "023a_s11_lrwidth_1024_lo": 4, "023b_s11_lrwidth_1024_hi": 4,
                 "025_s11_fine_and_replicate": 4, "028_s11_fine_depth": None,
                 "024a_s11_lrdepth_02": 2, "024b_s11_lrdepth_08": 8, "024c_s11_lrdepth_16": 16}
ROWS = []
for f in sorted(os.listdir(G)):
    if not f.endswith(".json"): continue
    stem = f[:-5]
    if stem.startswith(("027", "029", "030", "031", "032")): continue  # scar, held-out, batch
    for r in json.load(open(os.path.join(G, f))):
        r = dict(r)
        if r.get("depth") is None: r["depth"] = DEPTH_OF_FILE.get(stem)
        if r.get("width") is None: r["width"] = 256
        if r["depth"] is None: continue
        r["_job"] = stem[:4]
        ROWS.append(r)

def curve(width, depth, grid=None):
    d = {}
    for r in ROWS:
        if r["width"] != width or r["depth"] != depth: continue
        if grid is not None and not any(abs(r["lr"]/g - 1) < 1e-9 for g in grid): continue
        d.setdefault(r["lr"], []).append(r["val_loss"])
    return {k: sum(v)/len(v) for k, v in sorted(d.items())}

# ------------------------------------------------------------------ three estimators of an optimum
def est_argmin(c):
    return min(c.items(), key=lambda x: x[1])[0]

def est_tied(c):
    """README's estimator: geometric mean of every LR tied with the argmin inside the seed spread."""
    g = sorted(c.items(), key=lambda x: x[1])
    tied = [lr for lr, v in g if v - g[0][1] <= SPREAD]
    return math.exp(st.mean(math.log(x) for x in tied))

def est_vertex(c):
    """Parabola through the argmin and its two neighbours, in log2(lr)."""
    pts = sorted(c.items())
    i = min(range(len(pts)), key=lambda k: pts[k][1])
    if i in (0, len(pts)-1): return None
    (x1,y1),(x2,y2),(x3,y3) = [(math.log2(l), v) for l, v in pts[i-1:i+2]]
    u1,u3,d1,d3 = x1-x2, x3-x2, y1-y2, y3-y2
    det = u1*u1*u3 - u3*u3*u1
    a_  = (d1*u3 - d3*u1)/det
    b_  = (d3*u1*u1 - d1*u3*u3)/det
    return 2 ** (x2 - b_/(2*a_))

EST = {"grid argmin": est_argmin, "tied geo-mean (README)": est_tied, "parabola vertex": est_vertex}

# ------------------------------------------------------------------ two fits of an exponent
def fit_endpoints(xs, ys):
    return -math.log(ys[-1]/ys[0]) / math.log(xs[-1]/xs[0])

def fit_ols(xs, ys):
    X = [math.log(x) for x in xs]; Y = [math.log(y) for y in ys]
    mx, my = st.mean(X), st.mean(Y)
    return -sum((x-mx)*(y-my) for x, y in zip(X, Y)) / sum((x-mx)**2 for x in X)

FIT = {"2-point endpoints (README)": fit_endpoints, "OLS over all points": fit_ols}

print(__doc__)
def table(axis, vals, fixed, label):
    print("=" * 94)
    print(f"{axis.upper()} EXPONENT   ({label})")
    print("=" * 94)
    out = {}
    for gname, grid in (("identical coarse 2x grid only", COARSE), ("every point measured", None)):
        cur = {}
        for v in vals:
            c = curve(**{axis: v, **fixed}, grid=grid)
            cur[v] = c
        print(f"\n  grid: {gname}")
        print(f"    {'':>22}" + "".join(f"{v:>12}" for v in vals) + f"{'':>4}" +
              "".join(f"{f:>30}" for f in FIT))
        for ename, ef in EST.items():
            o = {v: ef(cur[v]) for v in vals}
            if any(o[v] is None for v in vals):
                print(f"    {ename:>22}" + "".join(f"{'n/a':>12}" for v in vals)); continue
            cells = "".join(f"{o[v]:>12.3e}" for v in vals)
            fits  = "".join(f"{ff([*vals], [o[v] for v in vals]):>30.4f}" for ff in FIT.values())
            print(f"    {ename:>22}{cells}    {fits}")
            for fname, ff in FIT.items():
                out[(gname, ename, fname)] = ff([*vals], [o[v] for v in vals])
    return out

DEPTHS = [2, 4, 8, 16]
WIDTHS = [256, 512, 1024]
bs = table("depth", DEPTHS, {"width": 256}, "width 256, batch 8; optimum estimator x exponent fit")
as_ = table("width", WIDTHS, {"depth": 4}, "depth 4, batch 8; optimum estimator x exponent fit")

def spread(d, name, published):
    v = sorted(d.values())
    u = sorted(set(round(x, 6) for x in v))
    rank = sum(1 for x in u if x < published)
    print(f"\n  {name}: {len(u)} distinct estimates ({len(v)} cells; estimators coincide where a grid "
          f"is coarse enough\n      that ties and argmin agree, and OLS over 3 log-spaced points "
          f"equals the endpoint fit)\n      range {u[0]:.4f} to {u[-1]:.4f}, median {st.median(u):.4f}. "
          f"README publishes {published:.3f} - rank {rank+1} of {len(u)}.")
    return u

print("\n" + "=" * 94 + "\nSPREAD ACROSS ESTIMATORS\n" + "=" * 94)
bv = spread(bs, "depth exponent b", 0.505)
av = spread(as_, "width exponent a", 0.990)

print(f"""
WHAT MOVES AND WHAT DOES NOT.

  The WIDTH exponent is robust.  {min(av):.3f} to {max(av):.3f} across every way of computing it, median {st.median(av):.3f}.
  All of them round to 1, so "lr* proportional to 1/width" survives the estimator choice, and the
  answer the assignment actually asks for - the rate at width 4,096 - does not move materially.

  The DEPTH exponent is not robust.  {min(bv):.3f} to {max(bv):.3f}, a {100*(max(bv)/min(bv)-1):.0f}% span.  Two things drive it,
  and they are separable:
    - ENDPOINT vs OLS.  The 2-point fit uses only depths 2 and 16 and discards 4 and 8.
    - GRID.  The README's four depth optima come from three different grids - depths 2 and 8 from
      the coarse 2x grid, depth 4 from a sqrt(2) refinement, depth 16 from a single sqrt(2) point.
      The tied-geometric-mean estimator is grid-sensitive by construction: more grid points inside
      the seed spread means more terms in the mean.  On ONE grid for all four depths the parabola
      vertex gives {bs[('identical coarse 2x grid only','parabola vertex','OLS over all points')]:.3f}; on the mixed grid the same estimator gives
      {bs[('every point measured','parabola vertex','OLS over all points')]:.3f}.

  WHERE THE PUBLISHED VALUE SITS.  b = 0.505 is at the HIGH end - 8th of the {len(bv)} distinct estimates -
  though it is the median if all 12 estimator-by-grid cells are weighted equally (0.5039).  Those two
  summaries disagree because the coincidences are not random: argmin and tied-geo-mean collapse
  together exactly on the coarse grid, so de-duplicating removes coarse-grid cells preferentially.
  We report both rather than choosing the flattering one.  Either way the point stands: 0.505 is
  quoted to three significant figures while the estimator choice alone moves it by {max(bv)-min(bv):.2f}, and it is
  not a conservative pick.  a = 0.990 is likewise the largest of its {len(av)} (median {st.median(av):.3f}).

WHAT IT DOES TO THE HEADLINE.""")
print(f"    {'estimate of b':>34}{'growth 8 -> 20 factor':>24}")
for nm, val in (("published (2-pt, mixed grid)", 0.505), ("median of the estimator family", st.median(bv)),
                ("smallest (parabola, mixed grid)", min(bv)), ("largest (argmin, one coarse grid)", max(bv))):
    print(f"    {nm:>34}{(20/8)**val:>23.3f}x")
print(f"""
  So the defensible headline is "about {(20/8)**st.median(bv):.1f}x, and between {(20/8)**min(bv):.2f}x and {(20/8)**max(bv):.2f}x depending on how the
  exponent is estimated" - not the bare point value {(20/8)**0.505:.2f}x.  The qualitative claim is untouched:
  every one of the {len(bv)} estimates has the rate coming DOWN on depth growth, by roughly half a power,
  and the practical recommendation is the same at either end of the range.  What is not supported is
  the third significant figure.

  DIRECTION.  This weakens our own published claim, and it was found by re-running our own analysis
  with a different estimator rather than by collecting new data.  The lesson generalises past this
  assignment: a scaling exponent read off two endpoints of a 2x grid is a choice, not a measurement,
  and the assignment's own epigraph - "a well tuned method measured against a badly tuned one" -
  applies to the ANALYSIS as much as to the training run.
""")
