"""Recompute every load-bearing number in README.md from the raw results and check that the
README actually says it. Exits non-zero on any disagreement.

This exists because three numbers were published and then corrected during this work -- a width
exponent of 0.868 that was a grid artifact, a "freezing raises incidence" claim from n=4, and a
beta2 crossing quoted as ~190 steps that is 76. All three were caught by recomputing rather than
by rereading. This turns that into a check.

Each entry recomputes a value from source and asserts the rendered string occurs in README.md, so
it fails both when the computation is wrong and when the prose has drifted from the data.

    python3 code/verify.py            # from S11/
"""
import json, math, glob, os, re, statistics as st, sys

R_RAW = open("README.md").read()
# The README writes exponents as 2.37e-3; format() renders 2.37e-03. Normalise BOTH sides so a
# cosmetic difference is not reported as a data disagreement -- and so a real one still is.
_norm = lambda t: re.sub(r"e([+-])0(\d)", r"e\1\2", t)
R = _norm(R_RAW)
G = "results/gpu"
FAIL, N = [], 0

def check(name, rendered, note=""):
    """Assert `rendered` (recomputed from source) appears literally in the README."""
    global N
    N += 1
    rendered = _norm(rendered)
    ok = rendered in R
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} {rendered}{'  ' + note if note else ''}")
    if not ok: FAIL.append((name, rendered))

DEPTH_OF_FILE = {"021_s11_lrwidth_256": 4, "022_s11_lrwidth_512": 4,
                 "023a_s11_lrwidth_1024_lo": 4, "023b_s11_lrwidth_1024_hi": 4,
                 "024a_s11_lrdepth_02": 2, "024b_s11_lrdepth_08": 8, "024c_s11_lrdepth_16": 16}
NV = 61070   # vocabulary. input_params = NV * width, and it is computed at RUNTIME.

ROWS = []
for f in sorted(glob.glob(f"{G}/*_s11_*.json")):
    stem = os.path.basename(f)[:-5]
    for r in json.load(open(f)):
        r = dict(r); r.setdefault("depth", DEPTH_OF_FILE.get(stem))
        assert r["depth"] is not None, stem
        # Width is taken from input_params, NEVER from the recorded "width" field.  Job 032 records
        # width=256 for width-512 runs (cloned from the width-512 sweep; see
        # results/gpu/032_CORRECTION.md), and on the first run after 032 landed this verifier failed
        # six Part-5/C1 checks because the mislabelled rows were pooled into curve(width=256).
        # Deriving the label from a runtime-computed quantity closes the whole class of bug.
        assert r["input_params"] % NV == 0, f"{stem}: input_params not a multiple of the vocab"
        w = r["input_params"] // NV
        if r.get("width") not in (None, w):
            r["width_as_recorded"] = r["width"]
        r["width"] = w
        r.setdefault("batch", 8)        # every job before 032 ran at the fixed batch of 8
        r.setdefault("res_scale", 1.0)  # every job before 033 ran standard parametrisation
        r["_job"] = stem
        ROWS.append(r)
MISLABELLED = [r for r in ROWS if "width_as_recorded" in r]
assert MISLABELLED, "expected job 032's known width mislabel to be present and corrected"
print(f"  note: {len(MISLABELLED)} row(s) had width relabelled from input_params "
      f"({sorted({(r['width_as_recorded'], r['width']) for r in MISLABELLED})}) "
      f"-- see results/gpu/032_CORRECTION.md")

def curve(**sel):
    """Rows matching sel.  Defaults isolate each ARM from the others: batch 8 (the fixed batch
    every job before 032 used) and res_scale 1.0 (standard parametrisation, every job before 033).
    Both defaults exist because the leak actually happened and this verifier caught it:
      - job 032's batch sweep, mislabelled width 256, pooled into the width-256 curve (140 -> 134);
      - job 033's muP arm, genuinely width 256 depth 2/4/8/16 batch 8, pooled into the SP depth
        curves and collapsed the depth-exponent family from 0.389-0.667 to 0.163-0.376.
    A new arm is not more data for an old curve."""
    sel.setdefault("batch", 8)
    sel.setdefault("res_scale", 1.0)   # muP rows (job 033+) are a different ARM, not more data
    d = {}
    for r in ROWS:
        if all(r.get(k) == v for k, v in sel.items()): d.setdefault(r["lr"], []).append(r["val_loss"])
    return {k: sum(v)/len(v) for k, v in sorted(d.items())}

SPREAD = 0.0073
def optimum(c):
    """Geometric mean of every LR tied with the argmin inside the seed spread."""
    g = sorted(c.items(), key=lambda x: x[1])
    tied = [lr for lr, v in g if v - g[0][1] <= SPREAD]
    return math.exp(st.mean(math.log(x) for x in tied)), g

# ============================================================ PART 1
print("\nPART 1 - Adam by hand")
p1 = json.load(open("results/p1_adam.json"))
check("worst hand-vs-torch disagreement", f"{p1['worst']:.3e}".replace("e+00", "e+00"))
check("final weight, hand", f"{p1['hand'][-1]['w']:.12f}")
check("final weight, torch", f"{p1['torch'][-1]['w']:.12f}")
# every cell of the Part 1 table -- it was hand-transcribed once and three rows were wrong
for row in p1["hand"]:
    for col, fmt in (("m", ".8f"), ("v", ".10f"), ("mhat", ".8f"),
                     ("vhat", ".10f"), ("step", ".4e"), ("w", ".8f")):
        check(f"t={row['t']} {col}", format(row[col], fmt))

# ============================================================ PART 1a / 1b
print("\nPART 1a/1b - AdamW and epsilon placement")
pb = json.load(open("results/p1b_adamw_eps.json"))
a8 = pb["a8"]
for i in range(5):
    check(f"Adam(L2) t={i+1}", f"{a8['adam_l2'][i]:.12f}")
    check(f"AdamW   t={i+1}", f"{a8['adamw'][i]:.12f}")
check("Adam(L2) vs AdamW gap", f"{a8['gap']:.3e}")
check("gap as % of movement", f"{a8['pct_of_movement']:.1f}")
assert a8["err_l2"] == 0.0 and a8["err_w"] == 0.0, "hand implementation disagrees with torch"
for r in pb["a9"]:
    check(f"eps={r['eps']:.0e} textbook-vs-pytorch", f"{r['d_pytorch']:.1e}")
    check(f"eps={r['eps']:.0e} textbook-vs-eps_early", f"{r['d_eps_early']:.1e}")
    check(f"eps={r['eps']:.0e} rel err at t=1", f"{r['rel_t1']:.2%}")
assert max(r["d_pytorch"] for r in pb["a9"]) < 1e-15, "pytorch form is NOT algebraically identical"
assert pb["a9"][-1]["d_eps_early"] > pb["a9"][0]["d_eps_early"], "eps_early error should grow with eps"

# ============================================================ PART 2
print("\nPART 2 - bias correction")
B1 = 0.9
def ratio(t, b2): return (1 - B1**t) / math.sqrt(1 - b2**t)
pk = max(range(1, 200001), key=lambda t: ratio(t, 0.999))
check("peak ratio (beta2=0.999)", f"{ratio(pk, 0.999):.2f}")
check("peak step", f"step {pk}")
check("ratio at step 20", f"{ratio(20, 0.999):.2f}")
for b2, thr in ((0.95, 1.01), (0.99, 1.01), (0.999, 1.10), (0.999, 1.01), (0.9999, 1.01)):
    last = max((t for t in range(1, 400001) if ratio(t, b2) >= thr), default=0)
    v = f"{last+1:,}" if last else "never"
    check(f"beta2={b2} settles <{thr}", v)
check("beta2=0.9 never exceeds", f"{max(ratio(t,0.9) for t in range(1,200001)):.2f}")
# the 20-step empirical run must agree with the closed form
emp = p1["off20"][19] / p1["on20"][19]
assert abs(emp/ratio(20, 0.999) - 1) < 1e-6, f"empirical {emp} != closed form {ratio(20,0.999)}"
print(f"  PASS  {'empirical 20-step run == closed form':<52} {emp:.4f}")
N += 1

# ============================================================ PART 3
print("\nPART 3 - update-to-weight ratio")
p3 = json.load(open("results/p3p4.json"))
for k, v in p3["part3"]["summary"].items():
    lab = k.replace(".weight", "").replace("blocks.", "L")
    check(f"{lab} peak", f"{v['peak']:.2e}")
    check(f"{lab} peak step", f"| **{v['peak_step']}** |" if lab != "head" else f"| **{v['peak_step']}** |")
    check(f"{lab} peak/final", f"{v['peak']/v['final']:.1f}")
fu = json.load(open("results/p3p4_followups.json"))
for k in fu["warmup"]["on"]:
    lab = k.replace(".weight", "").replace("blocks.", "L")
    on, off = fu["warmup"]["on"][k], fu["warmup"]["off"][k]
    if lab in ("emb", "L0.qkv", "L0.mlp.0", "head"):
        check(f"warmup-off/{lab} ratio", f"{max(off)/max(on):.2f}")
check("warmup val@300", f"{fu['warmup']['val_on']['300']:.4f}")
check("no-warmup val@300", f"{fu['warmup']['val_off']['300']:.4f}")
check("warmup cost (nats)", f"{fu['warmup']['val_on']['300']-fu['warmup']['val_off']['300']:.3f}")
check("step-1 grad norm (both arms equal)", f"{fu['warmup']['gn_on'][0]:.3f}")
assert abs(fu["warmup"]["gn_on"][0] - fu["warmup"]["gn_off"][0]) < 1e-9, "step-1 grad norms differ"
d = fu["emb_dilution"]
check("embedding dilution, median", f"{d['median']:.1f}")
check("embedding rows touched, %", f"{100*d['touched_max']/d['vocab']:.2f}")
check("dilution at step 1", f"{d['touched'][0]/d['all'][0]:.1f}")

# ============================================================ PART 4
print("\nPART 4 - cosine vs WSD")
p4 = p3["part4"]
for arm in ("cosine", "wsd"):
    for i, r in enumerate(p4[arm]):
        if i == 0: check(f"{arm} seed0 @200", f"{r['stop']:.4f}")
    check(f"{arm} mean @200", f"{st.mean(r['stop'] for r in p4[arm]):.4f}")
    check(f"{arm} mean @300", f"{st.mean(r['final'] for r in p4[arm]):.4f}")
d200 = [w["stop"]-c["stop"] for c, w in zip(p4["cosine"], p4["wsd"])]
d300 = [w["final"]-c["final"] for c, w in zip(p4["cosine"], p4["wsd"])]
check("delta @200", f"+{st.mean(d200):.4f}")
check("paired seed spread @200", f"{max(d200)-min(d200):.4f}")
check("delta @300", f"{st.mean(d300):.4f}".replace("-", "−"))
mm = fu["mean_mult"]
check("cosine mean multiplier", f"{mm['cosine']:.4f}")
check("wsd mean multiplier", f"{mm['wsd']:.4f}")
check("multiplier ratio", f"{mm['ratio']:.3f}")
arms = fu["arms"]
matched = st.mean(a["val"]["300"] for a in arms["wsd@matched_lr"]) - \
          st.mean(a["val"]["300"] for a in arms["cosine"])
check("LR-matched delta @300", f"{matched:.4f}".replace("-", "−"))
check("share of effect that was avg LR", f"{100*(1-abs(matched)/abs(st.mean(d300))):.1f}")

# ============================================================ PART 5 + C1
print("\nPART 5 / C1 - the learning-rate law")
WID = {w: curve(width=w, depth=4, model_seed=0) for w in (256, 512, 1024)}
DEP = {d: curve(width=256, depth=d, model_seed=0) for d in (2, 4, 8, 16)}
OW, OD = {}, {}
for w, c in WID.items():
    OW[w], g = optimum(c)
    check(f"width {w} optimum", f"{OW[w]:.2e}")
    check(f"width {w} argmin loss", f"{g[0][1]:.4f}")
    assert 0 < sorted(c).index(g[0][0]) < len(c)-1, f"width {w} minimum at a grid endpoint"
for dep, c in DEP.items():
    OD[dep], g = optimum(c)
    check(f"depth {dep} optimum", f"{OD[dep]:.2e}")
a = math.log(OW[256]/OW[1024]) / math.log(4)
b = math.log(OD[2]/OD[16]) / math.log(8)
check("width exponent a", f"{a:.3f}")
check("depth exponent b", f"{b:.3f}")
check("value at width 4096", f"{OW[1024]*4**-a:.2e}")
K = OW[256] / (256**-a * 4**-b)
law = lambda w, dd: K * w**-a * dd**-b
check("V4 2B/5B  (4096 x 8)", f"{law(4096, 8):.2e}")
check("V4 9B/120B (4096 x 20)", f"{law(4096, 20):.2e}")
check("growth boundary 8->20 factor", f"{(20/8)**b:.2f}")
for (w, dd), pred_s in (((512, 8), "8.39e-4"), ((512, 16), "5.92e-4")):
    got = curve(width=w, depth=dd, model_seed=0)
    argmin = min(got, key=got.get)
    check(f"held-out {w}x{dd} measured", f"{argmin:.2e}")
    steps = abs(math.log(argmin/float(pred_s.replace("e-4", "e-04"))) / math.log(math.sqrt(2)))
    check(f"held-out {w}x{dd} grid-step error", f"{steps:.2f}")

# ============================================================ C2
print("\nC2 - the scar")
c2 = json.load(open(f"{G}/031_c2_scar_incidence.json"))
SP = 30.0
cell = lambda L, arm, cond: c2[f"L{L}/{arm}/{cond}"]["gn"]["ratio_seeds"]
cnt  = lambda L, arm, cond: sum(1 for x in cell(L, arm, cond) if x > SP)
CLEAN = (2, 8, 16)
for L in CLEAN:
    check(f"depth {L} frozen+shift incidence", f"| **{cnt(L,'frozen','shift')}/10** |")
check("max clean excursion (depth 16)", f"{max(cell(16,'frozen','shift')):.1f}")
check("median excursion depth 16", f"{st.median(cell(16,'frozen','shift')):.1f}")
ctl = sum(cnt(L, a, "noshift") for L in CLEAN for a in ("frozen", "trainable"))
check("control spikes at clean depths",
      f"zero spikes in {len(CLEAN)*2*10} control runs" if ctl == 0 else f"{ctl} CONTROL SPIKES")
assert ctl == 0, "a control spiked at a depth we kept -- C2 must be withdrawn"
check("depth 32 control spike value", f"{max(cell(32,'frozen','noshift')):.1f}")
fz = sum(cnt(L, "frozen", "shift") for L in CLEAN)
tr = sum(cnt(L, "trainable", "shift") for L in CLEAN)
check("frozen incidence over clean depths", f"frozen + shift      {fz}/30")
check("trainable incidence over clean depths", f"trainable + shift      {tr}/30")
for L in CLEAN:
    t = c2[f"L{L}/frozen/shift"]
    check(f"depth {L} top/bottom ratio", f"{t['top']['ratio']/t['bot']['ratio']:.2f}")
sp = sorted(s for L in (8, 16) for s, r in
            zip(c2[f"L{L}/frozen/shift"]["gn"]["peak_step_seeds"],
                c2[f"L{L}/frozen/shift"]["gn"]["ratio_seeds"]) if r > SP)
check("spike steps (shift at 200)", ", ".join(str(x) for x in sp))

# ============================================================ C1.4 - the batch term
# Recomputed here INDEPENDENTLY of code/c1_batch_law.py.  The verifier's job is to disagree with
# the analysis when the analysis is wrong, which it cannot do by importing it.
print("\nC1.4 - the batch term")

def _vertex(c):
    """Parabola through the argmin and its two neighbours, in log2(lr). Returns (lr*, val*)."""
    pts = sorted(c.items())
    i = min(range(len(pts)), key=lambda k: pts[k][1])
    assert 0 < i < len(pts)-1, "batch curve minimum at a grid endpoint"
    (x1,y1),(x2,y2),(x3,y3) = [(math.log2(l), v) for l, v in pts[i-1:i+2]]
    u1,u3,d1,d3 = x1-x2, x3-x2, y1-y2, y3-y2
    det = u1*u1*u3 - u3*u3*u1
    A_ = (d1*u3 - d3*u1)/det; B_ = (d3*u1*u1 - d1*u3*u3)/det
    u = -B_/(2*A_)
    return 2**(x2+u), y2 + A_*u*u + B_*u

# GRID RULE, and it changes the answer: each batch is estimated on the LR grid ITS OWN sweep ran.
# For batch 8 that is job 022's coarse 2x grid.  Job 025 later added two sqrt(2) points at
# width 512 depth 4 (7e-4 and 1.4e-3) for the WIDTH law; folding those in here would give batch 8
# a finer grid than every other batch and move its optimum 1.247e-3 -> 1.258e-3.  A batch series
# where one batch is measured more finely than the rest is not a batch series.
COARSE = (3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3)
BOPT = {}
for bs in (4, 8, 16, 32):
    c = curve(width=512, depth=4, batch=bs)
    if bs == 8: c = {k: v for k, v in c.items() if any(abs(k/g - 1) < 1e-9 for g in COARSE)}
    if len(c) < 3: continue
    BOPT[bs] = _vertex(c)[0]
    check(f"batch {bs} optimum", f"{BOPT[bs]:.3e}")
assert BOPT.get(8), "the batch-8 anchor (job 022, width 512) is missing"
Bk = sorted(BOPT)
for a_, b_ in zip(Bk, Bk[1:]):
    check(f"local exponent {a_}->{b_}",
          f"**+{math.log(BOPT[b_]/BOPT[a_])/math.log(b_/a_):.3f}**")
check(f"batch {Bk[0]}->{Bk[-1]} ratio", f"moves ×{BOPT[Bk[-1]]/BOPT[Bk[0]]:.2f}")
check("batch exponent over full range",
      f"**+{math.log(BOPT[Bk[-1]]/BOPT[Bk[0]])/math.log(Bk[-1]/Bk[0]):.3f}**, a fitted exponent")
# SHAPE.  The README used to fit lr* = lr_inf*B/(B+Bn) and quote Bn = 2.40.  A saturating curve
# requires the local exponent to decay monotonically; batch 32 gave +0.467 after +0.064, so the
# form is refuted and Bn plus everything downstream of it is withdrawn.  The assert below is what
# KEEPS it withdrawn: if a future batch point restored monotonicity it fires and forces §C1.4 to be
# rewritten rather than silently left standing.
LOC = [math.log(BOPT[y]/BOPT[x])/math.log(y/x) for x, y in zip(Bk, Bk[1:])]
MONO = all(u >= v - 1e-12 for u, v in zip(LOC, LOC[1:]))
check("local exponent sequence", f"**+{LOC[0]:.3f}, +{LOC[1]:.3f}, +{LOC[2]:.3f}**")
best = None
for k in range(1, 20001):
    n_ = k*0.01
    li = math.exp(st.mean(math.log(BOPT[b_]*(b_+n_)/b_) for b_ in Bk))
    q = sum((math.log(BOPT[b_]) - math.log(li*b_/(b_+n_)))**2 for b_ in Bk)
    if best is None or q < best[0]: best = (q, n_, li)
_q, BN, LINF = best
check("refuted-fit rms", f"rms **{100*(math.exp(math.sqrt(_q/len(Bk)))-1):.1f}%** in lr")
check("refuted-fit residuals",
      "(" + ", ".join(f"{100*(BOPT[b_]/(LINF*b_/(b_+BN))-1):+.1f}%".replace("-", "\u2212")
                      for b_ in Bk) + ")")
assert not MONO, (
    "the local batch exponents are monotone again -- the saturating form may be back in play and "
    "README C1.4's withdrawal of Bn must be revisited rather than left standing")
check("sqrt rule over the range", f"×{math.sqrt(Bk[-1]/Bk[0]):.2f}")
check("linear rule over the range", f"linear rule ×{Bk[-1]/Bk[0]:.1f}")
check("how much less sensitive than sqrt",
      f"**{math.sqrt(Bk[-1]/Bk[0])/(BOPT[Bk[-1]]/BOPT[Bk[0]]):.2f}× less sensitive**")
dterm = (20/8)**-b
check("depth term 8->20", f"**×{dterm:.3f}**")

assert BOPT[Bk[-1]]/BOPT[Bk[0]] < math.sqrt(Bk[-1]/Bk[0]), \
    "the batch move is not smaller than the sqrt rule -- C1.4's central claim fails"
assert all(math.log(BOPT[y]/BOPT[x])/math.log(y/x) > 0 for x, y in zip(Bk, Bk[1:])), \
    "a local batch exponent went negative -- the saturating form is the wrong family"

# ============================================================ C1.5 - estimator sensitivity
print("\nC1.5 - how much of the exponent is the estimator")
COARSE = (3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3)
def _sub(c, grid): return {k: v for k, v in c.items() if grid is None or
                           any(abs(k/g - 1) < 1e-9 for g in grid)}
def _argmin(c): return min(c.items(), key=lambda x: x[1])[0]
def _vx(c):
    try: return _vertex(c)[0]
    except AssertionError: return None
ESTS = (("argmin", _argmin), ("tied", lambda c: optimum(c)[0]), ("vertex", _vx))
def _family(axis, vals, fixed):
    out = []
    for grid in (COARSE, None):
        for _, ef in ESTS:
            o = [ef(_sub(curve(**{axis: v, **fixed}), grid)) for v in vals]
            if any(x is None for x in o): continue
            X = [math.log(v) for v in vals]; Y = [math.log(x) for x in o]
            mx, my = st.mean(X), st.mean(Y)
            out.append(-math.log(o[-1]/o[0])/math.log(vals[-1]/vals[0]))                  # endpoints
            out.append(-sum((x-mx)*(y-my) for x, y in zip(X, Y))/sum((x-mx)**2 for x in X))  # OLS
    return sorted(set(round(x, 6) for x in out))
FB = _family("depth", [2, 4, 8, 16], {"width": 256})
FA = _family("width", [256, 512, 1024], {"depth": 4})
check("depth exponent family size",  f"| depth `b` | {len(FB)} |")
check("depth exponent range lo",  f"**{min(FB):.3f}")
check("depth exponent range hi",  f"{max(FB):.3f}**")
check("depth exponent median",    f"{st.median(FB):.3f}")
check("width exponent family size", f"| width `a` | {len(FA)} |")
check("width exponent range",     f"{min(FA):.3f} – {max(FA):.3f}")
check("width exponent median",    f"{st.median(FA):.3f}")
check("published b rank",         f"{sum(1 for x in FB if x < 0.505)+1} of {len(FB)}")
check("published a rank",         f"{sum(1 for x in FA if x < 0.990)+1} of {len(FA)}")
check("depth span pct",           f"{100*(max(FB)/min(FB)-1):.0f}%")
check("growth factor smallest",   f"**{(20/8)**min(FB):.3f}×**")
check("growth factor largest",    f"**{(20/8)**max(FB):.3f}×**")
check("growth factor median",     f"{(20/8)**st.median(FB):.3f}×")
check("growth factor published",  f"{(20/8)**0.505:.3f}×")
assert min(FB) > 0, "an estimate of b was non-positive -- the direction of the claim would fail"
assert 0.9 < st.median(FA) < 1.15, "the width exponent no longer rounds to 1"

# ============================================================ C1.6 - depth-muP
# Recomputed independently of code/c1_mup.py, and grid-matched: both arms on the same 6-point
# coarse grid with the same parabola-vertex estimator.
print("\nC1.6 - depth-muP transfer")
MUP_D = sorted({r["depth"] for r in ROWS if abs(r["res_scale"] - 1.0) > 1e-12})
assert MUP_D, "no muP rows found"
# SAMPLING IS MATCHED TO THE muP ARM, DELIBERATELY.  The muP arm is one seed per (depth, lr).  The
# SP side has replicate seeds at depths 2 and 4 (jobs 025/028) and one seed at 8 and 16, so pooling
# them would compare a 3-seed mean at the shallow end against 1-seed points at the deep end, and
# against a 1-seed muP arm throughout.  Each depth therefore uses its designated single SP job, the
# same rule code/c1_mup.py applies and the same rule the batch series uses.  The alternative is not
# hidden: seed-averaging instead moves b_SP from +0.4397 to +0.4453, which changes nothing.
SP_JOB = {2: "024a_s11_lrdepth_02", 4: "021_s11_lrwidth_256",
          8: "024b_s11_lrdepth_08", 16: "024c_s11_lrdepth_16"}
SPO, MPO, SPV, MPV = {}, {}, {}, {}
for d in MUP_D:
    cs = {r["lr"]: r["val_loss"] for r in ROWS if r["_job"] == SP_JOB[d]
          and any(abs(r["lr"]/g - 1) < 1e-9 for g in COARSE)}
    cm = curve(width=256, depth=d, res_scale=1/math.sqrt(d))
    assert cm, f"depth {d}: muP rows must carry res_scale == 1/sqrt(depth)"
    SPO[d], SPV[d] = _vertex(cs)
    MPO[d], MPV[d] = _vertex(cm)
    check(f"depth {d} SP lr*",  f"| {SPO[d]:.3e} | {SPV[d]:.4f} |")
    check(f"depth {d} muP lr*", f"| {MPO[d]:.3e} | {MPV[d]:.4f} |")
    check(f"depth {d} muP-SP gap", f"{MPV[d]-SPV[d]:+.4f}".replace("-", "−"))
def _ols(o):
    X = [math.log(d) for d in sorted(o)]; Y = [math.log(o[d]) for d in sorted(o)]
    mx, my = st.mean(X), st.mean(Y)
    return -sum((x-mx)*(y-my) for x, y in zip(X, Y))/sum((x-mx)**2 for x in X)
BSP, BMU = _ols(SPO), _ols(MPO)
check("b_SP matched-grid",  f"b_SP = +{BSP:.4f}")
check("b_muP",              f"b_muP = +{BMU:.4f}")
D0, D1 = min(MUP_D), max(MUP_D)
check("muP span over depth range", f"×{(D1/D0)**-BMU:.3f} under muP")
check("SP span over depth range",  f"×{(D1/D0)**-BSP:.3f} under SP")
check("muP span in grid steps", f"**{abs(math.log2((D1/D0)**-BMU)):.2f} of a grid step")
check("SP span in grid steps",  f"which is {abs(math.log2((D1/D0)**-BSP)):.2f} grid")
LOCM = [-math.log(MPO[y]/MPO[x])/math.log(y/x) for x, y in zip(MUP_D, MUP_D[1:])]
check("muP local exponents rise", ", ".join(f"+{x:.3f}" for x in LOCM))
# H11 / H12, applied as pre-registered
assert abs(BMU) < 0.15, f"H11 REFUTED: |b_muP| = {abs(BMU):.4f} -- README C1.6 claims CONFIRMED"
assert not all(MPV[d] - SPV[d] > 0.05 for d in MUP_D), \
    "H12 FIRES at every depth -- the muP comparison is VOID and C1.6 must be withdrawn"
assert LOCM == sorted(LOCM), \
    "the muP local exponents no longer rise -- C1.6's degradation caveat must be restated"
assert LOCM[-1] > 0.15, \
    "the deepest muP local exponent no longer exceeds H11's own threshold -- restate the caveat"
# the held-out predictions, recomputed from the same fits that generated them
AM = math.exp(st.mean(math.log(MPO[d]) for d in MUP_D) + BMU*st.mean(math.log(d) for d in MUP_D))
AS = math.exp(st.mean(math.log(SPO[d]) for d in MUP_D) + BSP*st.mean(math.log(d) for d in MUP_D))
check("held-out muP prediction", f"**{AM*32**-BMU:.4e}**")
check("held-out SP prediction",  f"**{AS*32**-BSP:.4e}**")
check("prediction separation",   f"×{(AM*32**-BMU)/(AS*32**-BSP):.2f}")

print(f"\n{'='*74}\n{N-len(FAIL)}/{N} checks pass")
if FAIL:
    print("\nFAILURES -- the README disagrees with the data:")
    for n_, v in FAIL: print(f"  {n_}: recomputed {v}, not found in README.md")
    sys.exit(1)
print("every load-bearing number in README.md was recomputed from results/ and matches.")
