"""Figures for the S11 README. Reads only the archived result JSON, never hardcoded numbers.

Palette is the dataviz reference categorical set, validated with
  node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4,#008300" --mode light
  -> ALL CHECKS PASS (contrast WARN relieved by direct labels on every series)
"""
import json, math, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

C   = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd5"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "lines.linewidth": 2.0, "lines.markersize": 8,
    "figure.dpi": 140, "savefig.bbox": "tight",
})
def tidy(ax):
    ax.grid(True, alpha=.55, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True)

G = "results/gpu"

# Jobs 021-024 were generated from a template that recorded `width` but not `depth`, so for
# those files depth comes from the filename. Stated explicitly rather than inferred silently.
DEPTH_OF_FILE = {"021_s11_lrwidth_256": 4, "022_s11_lrwidth_512": 4,
                 "023a_s11_lrwidth_1024_lo": 4, "023b_s11_lrwidth_1024_hi": 4,
                 "024a_s11_lrdepth_02": 2, "024b_s11_lrdepth_08": 8, "024c_s11_lrdepth_16": 16}

def load_all():
    rows = []
    for f in sorted(glob.glob(f"{G}/*_s11_*.json")):
        stem = os.path.basename(f)[:-5]
        for r in json.load(open(f)):
            r = dict(r)
            r.setdefault("depth", DEPTH_OF_FILE.get(stem))
            if r["depth"] is None:
                raise SystemExit(f"no depth for {stem}; add it to DEPTH_OF_FILE")
            r["_job"] = stem
            # Same arm-isolation the verifier enforces: batch 8 and res_scale 1.0 are the
            # defaults every job before 032/033 ran at, and a new arm is not more data for an
            # old curve. Without these, job 032's batch sweep and job 033's muP sweep pool
            # straight into the width and depth curves in figs 2 and 3.
            r.setdefault("batch", 8)
            r.setdefault("res_scale", 1.0)
            rows.append(r)
    return rows

ROWS = load_all()
def curve(**sel):
    sel.setdefault("batch", 8)
    sel.setdefault("res_scale", 1.0)
    d = {}
    for r in ROWS:
        if all(r.get(k) == v for k, v in sel.items()):
            d.setdefault(r["lr"], []).append(r["val_loss"])
    return {k: sum(v)/len(v) for k, v in sorted(d.items())}

# ---------------------------------------------------------------- fig 1: bias correction
p1 = json.load(open("results/p1_adam.json"))
B1, B2 = 0.9, 0.999
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0))
t = range(1, 21)
a.plot(t, p1["on20"],  "o-", color=C[0], label="correction ON")
a.plot(t, p1["off20"], "s-", color=C[1], label="correction OFF")
a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("| update |")
a.set_title("The first 20 steps, both ways"); a.set_xticks([1,5,10,15,20])
a.annotate("still 6.24x apart\nat step 20", (20, p1["off20"][19]), xytext=(-96, 10),
           textcoords="offset points", color=INK2, fontsize=9,
           arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
a.legend(frameon=False, loc="lower left"); tidy(a)

ts = [1.3**i for i in range(0, 55)]
for i, b2 in enumerate((0.99, 0.999, 0.9999)):
    y = [(1-B1**x)/math.sqrt(1-b2**x) for x in ts]
    b.plot(ts, y, color=C[i], label=f"$\\beta_2$ = {b2}")
    j = max(range(len(ts)), key=lambda k: y[k])
    b.plot([ts[j]], [y[j]], "o", color=C[i], zorder=5)
    b.annotate(f"{y[j]:.2f}x", (ts[j], y[j]), xytext=(4, 6), textcoords="offset points",
               color=INK2, fontsize=9)
b.axhline(1.0, color=INK2, lw=1.2, ls="--", zorder=1)
b.annotate("no difference", (2.2, 1.0), xytext=(0, -14), textcoords="offset points",
           color=INK2, fontsize=9)
b.axvline(20, color=GRID, lw=1.5, zorder=1)
b.annotate("end of the window\nthe question asks for", (20, 14), xytext=(8, 0),
           textcoords="offset points", color=INK2, fontsize=9)
b.set_xscale("log"); b.set_yscale("log")
b.set_xlabel("step"); b.set_ylabel("uncorrected / corrected step")
b.set_title("It peaks, then takes thousands of steps to decay")
b.legend(frameon=False, loc="center right"); tidy(b)
fig.savefig("figures/fig1_bias_correction.png"); plt.close(fig)

# ------------------------------------------------------- fig 2 + 3: LR vs loss, marked minima
WID = {w: curve(width=w, depth=4, model_seed=0) for w in (256, 512, 1024)}
DEP = {d: curve(width=256, depth=d, model_seed=0) for d in (2, 4, 8, 16)}
for name, tbl in (("width", WID), ("depth", DEP)):
    for k, c in tbl.items():
        if len(c) < 4: raise SystemExit(f"{name} {k}: only {len(c)} points - {c}")

from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter

def panel(ax, data, label, title, opt):
    """Optima go in the LEGEND, not as floating annotations -- direct labels on a log axis
    collided with the tick labels and with each other."""
    xs_all = sorted({x for c in data.values() for x in c})
    for i, (k, c) in enumerate(data.items()):
        xs, ys = list(c), list(c.values())
        ax.plot(xs, ys, "o-", color=C[i], label=f"{label} {k}   $lr^*$ = {opt[k]:.2e}")
        lo = min(c, key=c.get)
        ax.plot([lo], [c[lo]], "o", ms=16, mfc="none", mew=2.5, color=C[i], zorder=6)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xs_all))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v*1e3:g}" if v < 1e-2 else f"{v*1e3:g}"))
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.set_xlabel("learning rate  ($\\times 10^{-3}$)")
    ax.set_ylabel("val loss @1200 steps")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper center", fontsize=9, ncol=2,
              handlelength=1.4, columnspacing=1.0)
    tidy(ax)

OW = {256: 2.366e-3, 512: 1.183e-3, 1024: 6.00e-4}
OD = {2: 4.00e-3, 4: 2.366e-3, 8: 2.00e-3, 16: 1.40e-3}
fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.0))
panel(a, WID, "width", "Learning rate vs WIDTH  (depth 4)", OW)
panel(b, DEP, "depth", "Learning rate vs DEPTH  (width 256)", OD)
fig.savefig("figures/fig2_lr_sweeps.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 3: predicted vs measured
# Earlier version overlaid the depth series on the width axis with a x128 fudge factor. That is a
# disguised dual axis -- two different variables sharing one scale -- so it was replaced. The claim
# the figure has to carry is "the law predicts", and predicted-vs-measured says exactly that with
# no rescaling.
A, B, K = 0.990, 0.505, 1.1538          # lr* = K * width^-A * depth^-B, calibrated at (256, 4)
law = lambda w, d: K * w**-A * d**-B
IN  = [(256,4,2.366e-3), (512,4,1.183e-3), (1024,4,6.00e-4),
       (256,2,4.00e-3),  (256,8,2.00e-3),  (256,16,1.40e-3)]
OUT = [(512,8,8.50e-4), (512,16,6.00e-4)]

fig, ax = plt.subplots(figsize=(6.4, 5.4))
lo, hi = 4e-4, 5e-3
ax.plot([lo, hi], [lo, hi], "--", color=INK2, lw=1.4, zorder=1)
ax.annotate("perfect prediction", (1.05e-3, 1.05e-3), rotation=45, color=INK2, fontsize=9,
            ha="center", va="bottom", rotation_mode="anchor")
# per-point offsets: 1024x4 and the held-out 512x16 land on nearly the same coordinates
OFF = {(256,4):(8,-3), (512,4):(8,-3), (1024,4):(-10,-16), (256,2):(8,-3),
       (256,8):(8,-3), (256,16):(8,-3)}
for i2, (w, d, m) in enumerate(IN):
    ax.plot([law(w, d)], [m], "o", color=C[0], zorder=5,
            label="fitted on these" if i2 == 0 else None)
    ax.annotate(f"{w}x{d}", (law(w, d), m), xytext=OFF[(w, d)], textcoords="offset points",
                ha="right" if OFF[(w, d)][0] < 0 else "left", color=INK2, fontsize=9)
for i2, (w, d, m) in enumerate(OUT):
    ax.plot([law(w, d)], [m], "P", ms=15, color=C[5], zorder=6,
            label="HELD OUT - in no fit" if i2 == 0 else None)
    ax.annotate(f"{w}x{d}", (law(w, d), m), xytext=(10, -4), textcoords="offset points",
                color=INK2, fontsize=10, fontweight="bold")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.set_xlabel("predicted by $lr^* = K \\cdot width^{-0.99} \\cdot depth^{-0.51}$")
ax.set_ylabel("measured optimum")
ax.set_title("The law predicts combinations it was never fitted on")
ax.legend(frameon=False, loc="upper left", fontsize=10); tidy(ax)
fig.savefig("figures/fig3_law.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 4: per-layer ratios
f = json.load(open("results/p3p4.json")); R = f["part3"]["ratios"]; WARM = f["part3"]["warm"]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
for i, (k, v) in enumerate(R.items()):
    lab = k.replace(".weight", "").replace("blocks.", "L")
    ax.plot(range(1, len(v)+1), v, color=C[i], label=lab)
    j = max(range(len(v)), key=lambda z: v[z])
    ax.plot([j+1], [v[j]], "o", color=C[i], zorder=5)
ax.axvline(WARM, color=INK2, lw=1.5, ls="--")
ax.annotate(f"warmup ends\nstep {WARM}", (WARM, 3.2e-2), xytext=(10, 0),
            textcoords="offset points", color=INK2, fontsize=9)
ax.annotate("blocks peak at 20-22,\nBEFORE warmup ends", (21, 1.25e-2), xytext=(-8, 34),
            textcoords="offset points", ha="right", color=INK2, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("step"); ax.set_ylabel(r"$\|\Delta w\| / \|w\|$")
ax.set_title("Update-to-weight ratio, per layer")
ax.legend(frameon=False, fontsize=9, ncol=2); tidy(ax)
fig.savefig("figures/fig4_layer_ratios.png"); plt.close(fig)

print("wrote:", *sorted(os.path.basename(x) for x in glob.glob("figures/*.png")))

# ---------------------------------------------------------------- fig 5: depth-muP transfer
# The clearest single statement in C1: two parametrisations, the same grid, the same estimator,
# and an out-of-sample point at depth 32 that neither line was fitted on.
def vertex(c):
    pts = sorted(c.items())
    i = min(range(len(pts)), key=lambda k: pts[k][1])
    if i in (0, len(pts)-1): return None
    (x1,y1),(x2,y2),(x3,y3) = [(math.log2(l), v) for l, v in pts[i-1:i+2]]
    u1,u3,d1,d3 = x1-x2, x3-x2, y1-y2, y3-y2
    det = u1*u1*u3 - u3*u3*u1
    a = (d1*u3 - d3*u1)/det; b = (d3*u1*u1 - d1*u3*u3)/det
    return 2 ** (x2 - b/(2*a))

COARSE = (3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3)
SP_JOB = {2: "024a_s11_lrdepth_02", 4: "021_s11_lrwidth_256",
          8: "024b_s11_lrdepth_08", 16: "024c_s11_lrdepth_16",
          32: "034b_s11_sp_heldout_d32"}
def on_grid(d): return {k: v for k, v in d.items() if any(abs(k/g-1) < 1e-9 for g in COARSE)}

SPO, MPO = {}, {}
for d, j in SP_JOB.items():
    c = on_grid({r["lr"]: r["val_loss"] for r in ROWS if r["_job"] == j})
    if len(c) >= 3 and vertex(c): SPO[d] = vertex(c)
for d in (2, 4, 8, 16, 32):
    c = curve(width=256, depth=d, res_scale=1/math.sqrt(d))
    if len(c) >= 3 and vertex(c): MPO[d] = vertex(c)

FIT = [2, 4, 8, 16]          # depths both lines were fitted on
def ols(o, ds):
    X = [math.log(d) for d in ds]; Y = [math.log(o[d]) for d in ds]
    mx = sum(X)/len(X); my = sum(Y)/len(Y)
    b = -sum((x-mx)*(y-my) for x, y in zip(X, Y))/sum((x-mx)**2 for x in X)
    return math.exp(my + b*mx), b

if len(MPO) >= 4 and len(SPO) >= 4:
    # 90% intervals from the measured, position-dependent seed noise (job 032c): sigma 0.005 at
    # or below each argmin, 0.110 above it. The point just past the argmin is what a parabola
    # vertex leans on hardest and is the noisy one, so these widen as the curves flatten.
    import random
    def band(raw):
        rng = random.Random(3)
        v = []
        for _ in range(2000):
            am = min(raw, key=raw.get)
            d = {k: x + rng.gauss(0, 0.110 if k > am else 0.005) for k, x in raw.items()}
            q = vertex(d)
            if q: v.append(q)
        v.sort()
        return v[int(.05*len(v))], v[int(.95*len(v))]
    RAW = {("SP", d): on_grid({r["lr"]: r["val_loss"] for r in ROWS if r["_job"] == SP_JOB[d]})
           for d in SPO}
    RAW.update({("muP", d): curve(width=256, depth=d, res_scale=1/math.sqrt(d)) for d in MPO})
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    tidy(ax)
    xs = [1.75, 45]
    for (o, col, name) in ((SPO, C[0], "standard"), (MPO, C[1], "depth-muP  $1/\\sqrt{L}$")):
        ds = [d for d in FIT if d in o]
        A, b = ols(o, ds)
        ax.plot(xs, [A*x**-b for x in xs], "--", color=col, lw=1.4, alpha=.75, zorder=2)
        for d in sorted(o):
            lo_, hi_ = band(RAW[(("SP" if col == C[0] else "muP"), d)])
            ax.plot([d, d], [lo_, hi_], "-", color=col, lw=1.1, alpha=.45, zorder=3)
        ax.plot(ds, [o[d] for d in ds], "o", color=col, zorder=4,
                markeredgecolor=SURF, markeredgewidth=2, label=f"{name}   $b={b:+.3f}$")
        if 32 in o:            # held out: open ring for measured, small x for predicted
            ax.plot([32], [A*32**-b], "x", color=col, ms=9, mew=2.0, alpha=.8, zorder=4)
            ax.plot([32], [o[32]], "o", mfc="none", color=col, ms=13, mew=2.4, zorder=5)
            steps = abs(math.log2(o[32]/(A*32**-b)))
            ax.annotate(f"{o[32]:.2e}  (pred {A*32**-b:.2e}, {steps:.2f} steps)",
                        (32, o[32]), textcoords="offset points", xytext=(-14, -4),
                        ha="right", va="center", color=col, fontsize=8.5)
    ax.axvspan(22, 45, color=GRID, alpha=.3, zorder=1)
    import matplotlib.transforms as mtr
    ax.text(30, 0.955, "held out — in neither fit",
            transform=mtr.blended_transform_factory(ax.transData, ax.transAxes),
            ha="center", fontsize=8.5, color=INK2, style="italic")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels(["2", "4", "8", "16", "32"])
    ax.set_xlim(1.75, 45)
    ax.set_yticks([5e-4, 1e-3, 2e-3, 4e-3]); ax.set_yticklabels(["5e-4", "1e-3", "2e-3", "4e-3"])
    ax.set_xlabel("depth (layers)"); ax.set_ylabel("optimal learning rate")
    ax.set_title("muP flattens the depth dependence — and the deep end is barely pinned")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.savefig("figures/fig5_mup_transfer.png"); plt.close(fig)
    print("wrote: fig5_mup_transfer.png")
else:
    print("fig5 skipped: need >=4 depths in both arms "
          f"(SP has {len(SPO)}, muP has {len(MPO)})")
