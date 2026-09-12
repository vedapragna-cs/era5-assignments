"""How much of every depth-law claim survives the seed noise we actually measured?

WHY THIS EXISTS.  Job 032c measured something we had assumed away: seed noise is NOT constant along
an LR sweep.  At batch 16, three seeds gave a spread of 0.0050 nats at 6e-4, 0.0089 at 1.2e-3 and
0.1131 at 2.4e-3 - 23x larger just PAST the optimum, which is exactly the point a parabola vertex
leans on hardest.  Every optimum in this submission is a parabola through three points, one of which
sits on that noisy side, and almost every one is a single seed.

So: propagate the MEASURED noise through every vertex and every exponent, and report which claims
survive.  Nothing here is new data.

NOISE MODEL, grounded in job 032c rather than assumed:
    sigma = 0.005 nats at or below the argmin      (measured 0.0050 and 0.0089)
    sigma = 0.110 nats above the argmin            (measured 0.1131)
This is deliberately the pessimistic reading of three seeds at one config; we do not have the
replicates to do better, and saying so is part of the result.
"""
import json, math, os, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
G    = os.path.join(HERE, "..", "results", "gpu")
COARSE = (3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3)
NDRAW = 4000
# TWO NOISE MODELS, because we cannot honestly claim to know which is right.
#   "pre-registered"  the spread we assumed all along: 0.0073 nats everywhere (S7 job 018).
#   "measured"        job 032c's finding: ~0.005 at or below the argmin, ~0.110 above it.
# The measured model is grounded in three seeds at ONE config (batch 16, width 512, depth 4) and
# applying its 0.110 to every above-argmin point in every sweep is an extrapolation. The truth is
# probably between the two. Reporting both is the point: it shows which conclusions depend on
# the noise model and which do not.
MODELS = {"pre-registered (0.0073 flat)": (0.0073, 0.0073),
          "measured 032c (0.005 / 0.110)": (0.005, 0.110)}
SIG_LO, SIG_HI = MODELS["measured 032c (0.005 / 0.110)"]
SP_JOB = {2: "024a_s11_lrdepth_02", 4: "021_s11_lrwidth_256", 8: "024b_s11_lrdepth_08",
          16: "024c_s11_lrdepth_16", 32: "034b_s11_sp_heldout_d32"}
MU_FILE = ("033a_s11_depthmup_02_04", "033b_s11_depthmup_08_16", "034a_s11_mup_heldout_d32")

def load(f): return json.load(open(os.path.join(G, f + ".json")))
def vertex(c):
    pts = sorted(c.items())
    i = min(range(len(pts)), key=lambda k: pts[k][1])
    if i in (0, len(pts)-1): return None
    (x1,y1),(x2,y2),(x3,y3) = [(math.log2(l), v) for l, v in pts[i-1:i+2]]
    u1,u3,d1,d3 = x1-x2, x3-x2, y1-y2, y3-y2
    det = u1*u1*u3 - u3*u3*u1
    a = (d1*u3 - d3*u1)/det; b = (d3*u1*u1 - d1*u3*u3)/det
    if a <= 0: return None                      # not a minimum under this perturbation
    return 2 ** (x2 - b/(2*a))

SP = {d: {r["lr"]: r["val_loss"] for r in load(j)
          if any(abs(r["lr"]/g-1) < 1e-9 for g in COARSE)} for d, j in SP_JOB.items()}
MU = {}
for f in MU_FILE:
    for r in load(f): MU.setdefault(r["depth"], {})[r["lr"]] = r["val_loss"]

def draw(c, rng):
    am = min(c, key=c.get)
    return {k: v + rng.gauss(0, SIG_HI if k > am else SIG_LO) for k, v in c.items()}
def ols(o, ds):
    X = [math.log(d) for d in ds]; Y = [math.log(o[d]) for d in ds]
    mx, my = st.mean(X), st.mean(Y)
    return -sum((x-mx)*(y-my) for x, y in zip(X, Y))/sum((x-mx)**2 for x in X)
def pct(v, p): return sorted(v)[max(0, min(len(v)-1, int(p*len(v))))]

print(__doc__)
print("=" * 98)
print("PER-VERTEX: how well is each optimum pinned by its own three points?")
print("=" * 98)
print(f"{'depth':>6}{'arm':>5}{'lr*':>12}{'5th pct':>12}{'95th pct':>12}{'  ->  in grid steps':>22}")
rng = random.Random(0)
for arm, D in (("SP", SP), ("muP", MU)):
    for d in sorted(D):
        base = vertex(D[d])
        v = [x for _ in range(NDRAW) if (x := vertex(draw(D[d], rng))) is not None]
        lo, hi = pct(v, .05), pct(v, .95)
        print(f"{d:>6}{arm:>5}{base:>12.4e}{lo:>12.4e}{hi:>12.4e}"
              f"   −{abs(math.log2(lo/base)):.2f} / +{abs(math.log2(hi/base)):.2f}")

print("\n" + "=" * 98)
print("WHICH CLAIMS SURVIVE")
print("=" * 98)
def survive(label):
    global draws
    rng = random.Random(1)
    draws = []
    for _ in range(NDRAW):
        sp = {d: vertex(draw(SP[d], rng)) for d in SP}
        mu = {d: vertex(draw(MU[d], rng)) for d in MU}
        if any(x is None for x in list(sp.values()) + list(mu.values())): continue
        draws.append((sp, mu))
    print(f"\n{'-'*98}\n NOISE MODEL: {label}   ({len(draws)}/{NDRAW} draws kept)\n{'-'*98}")
    F16, F832 = [2, 4, 8, 16], [8, 16, 32]
    rows = [
      ("b_SP  over 2-16 (in-sample)",       lambda sp, mu: ols(sp, F16),                 None, ">"),
      ("b_muP over 2-16 (in-sample)",       lambda sp, mu: ols(mu, F16),                 None, ">"),
      ("H11: muP flattens vs SP (diff > 0)",lambda sp, mu: ols(sp,F16)-ols(mu,F16),      0.0,  ">"),
      ("H11 as written: |b_muP| < 0.15",    lambda sp, mu: abs(ols(mu, F16)),            0.15, "<"),
      ("held out: muP above SP at depth 32",lambda sp, mu: math.log2(mu[32]/sp[32]),     0.0,  ">"),
      ("H14: muP local exp 16->32 > 0.15",  lambda sp, mu: -math.log(mu[32]/mu[16])/math.log(2), 0.15, ">"),
      ("SP also decelerates (local < 0.15)",lambda sp, mu: -math.log(sp[32]/sp[16])/math.log(2), 0.15, "<"),
      ("V4 range 8-32: muP flattens vs SP", lambda sp, mu: ols(sp,F832)-ols(mu,F832),    0.0,  ">"),
    ]
    out = {}
    for name, f, th, dr in rows:
        v = [f(sp, mu) for sp, mu in draws]
        lo, hi = pct(v, .05), pct(v, .95)
        line = f"  {name:<40} {st.median(v):+.3f}  90% CI [{lo:+.3f}, {hi:+.3f}]"
        if th is not None:
            fr = sum(1 for x in v if (x > th if dr == ">" else x < th))/len(v)
            line += f"   P = {fr:.0%}"
            out[name] = fr
        print(line)
    return out

both = {}
for label, (SIG_LO_, SIG_HI_) in MODELS.items():
    SIG_LO, SIG_HI = SIG_LO_, SIG_HI_
    globals()["SIG_LO"], globals()["SIG_HI"] = SIG_LO_, SIG_HI_
    both[label] = survive(label)

print(f"\n{'='*98}\nVERDICT - probability each claim holds, under each noise model\n{'='*98}")
names = list(next(iter(both.values())).keys())
w = max(len(n) for n in names)
print(f"  {'claim':<{w}}" + "".join(f"{l.split(' (')[0]:>22}" for l in both))
for n in names:
    cells = "".join(f"{both[l][n]:>21.0%} " for l in both)
    mark = "  SURVIVES" if all(both[l][n] >= 0.90 for l in both) else \
           "  fails under measured noise" if both[list(both)[0]][n] >= 0.90 else "  NOT ESTABLISHED"
    print(f"  {n:<{w}}{cells}{mark}")
print("""
  A claim only counts if it survives BOTH models, because we do not know which is right.
  What that leaves, and what it costs us, is written up in README C1.6.

  WHAT WOULD FIX THIS, and it is cheap: replicate seeds at the points ABOVE each argmin. Those
  are the points a parabola vertex leans on hardest and the only ones whose noise we have not
  measured outside a single config. Three seeds at the two high-LR points of each depth would
  collapse most of these intervals. We did not run it.""")
