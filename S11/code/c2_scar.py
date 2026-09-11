"""C2 - the V4 scar, instrumented.

S7 §9 prescribes a monitoring rule but never measures it:
  "the gradient norms of the layers IMMEDIATELY ABOVE the embedding are monitored as a
   leading indicator rather than the GLOBAL NORM, WHICH AVERAGES THE SIGNAL AWAY."
The V4 cookbook §7.5 grades the underlying incident as the thinnest-evidenced failure in
its section: "no clean saved trace isolates the instability."

This builds the trace. A sharp en -> indic mixture shift at step 200, embedding frozen vs
trainable, and a NO-SHIFT control for each so the spike is attributable to the shift rather
than to the step count.

PRE-REGISTERED before running:
  H1  With the embedding FROZEN, the shift produces a larger post-shift excursion in the
      layer immediately above the embedding than with it trainable.
  H2  The per-layer update-to-weight ratio above the embedding detects the shift EARLIER
      and at a HIGHER z-score than the global grad norm.
  H2 REFUTED if the global norm's z-score is equal or larger, or fires first. That would
      mean S7 §9's rule is wrong as written, which is the more useful outcome.
  H3  The no-shift controls show no excursion, in either arm.
Detection = first step after the shift exceeding baseline mean + 3 sd, where the baseline
is the 50 steps immediately before the shift.
"""
import json, math, os, time, statistics as st
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
exec(open("p3p4_ratio_schedules.py").read().split('print("="*78)')[0].split("# ---- the two")[0])

Q = os.path.expanduser("~/shared/gpu-queue")
LB = np.load(f"{Q}/data/lang_labels.npz")
tb, BLK = LB["train_block"], int(LB["blk"])
tr = np.load(f"{Q}/data/train.npy").astype(np.int64)
EN = np.where(tb == 0)[0]; IND = np.where((tb == 1) | (tb == 2))[0]
print(f"blocks: en {len(EN)}  indic {len(IND)}  (block-label accuracy on val: "
      f"{100*float(LB['val_block_acc']):.2f}%)")

STEPS, SHIFT, WARM_ = 400, 200, 30
LR_ = 2e-3

def stream(seed, shift=True):
    """Phase 1 = English only. Phase 2 = Indic only. Sequences never cross a block boundary."""
    g = np.random.default_rng(seed); s = 0
    while True:
        pool = EN if (not shift or s < SHIFT) else IND
        blocks = g.choice(pool, BATCH)
        off = g.integers(0, BLK - SEQ - 1, BATCH)
        x = np.stack([tr[b*BLK+o : b*BLK+o+SEQ] for b, o in zip(blocks, off)])
        y = np.stack([tr[b*BLK+o+1 : b*BLK+o+SEQ+1] for b, o in zip(blocks, off)])
        s += 1
        yield torch.from_numpy(x), torch.from_numpy(y)

ABOVE = "blocks.0.qkv.weight"          # the layer immediately above the embedding
TRACK = [ABOVE, "blocks.0.mlp.0.weight", "blocks.1.qkv.weight", "head.weight"]

def run(seed, frozen, shift):
    torch.manual_seed(seed)
    model = GPT()
    if frozen:
        model.emb.weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR_, betas=(0.9,0.95), weight_decay=0.1)
    named = dict(model.named_parameters()); it = stream(seed, shift)
    gn, R = [], {k: [] for k in TRACK}; losses = []
    for s in range(STEPS):
        m = min(1.0, (s+1)/WARM_)
        for g in opt.param_groups: g["lr"] = LR_*m
        x, y = next(it)
        loss = F.cross_entropy(model(x).reshape(-1,NV), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        gn.append(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1e9).item())  # measure, don't clip
        pre = {k: named[k].detach().clone() for k in TRACK}
        opt.step()
        for k in TRACK:
            w = named[k].detach(); R[k].append(((w-pre[k]).norm()/w.norm()).item())
        losses.append(loss.item())
    return dict(seed=seed, frozen=frozen, shift=shift, gn=gn, R=R, loss=losses)

def detect(sig, base=(SHIFT-50, SHIFT)):
    b = np.array(sig[base[0]:base[1]]); mu, sd = b.mean(), b.std() + 1e-12
    z = (np.array(sig) - mu) / sd
    post = z[SHIFT:]
    fire = np.where(post > 3.0)[0]
    return dict(first=int(fire[0]) if len(fire) else None,
                peak_z=float(post.max()), peak_at=int(post.argmax()),
                baseline=float(mu), peak_val=float(np.array(sig)[SHIFT:].max()))

SEEDS = (0, 1, 2)
out = {}
for frozen in (False, True):
    for shift in (True, False):
        tag = f"{'frozen' if frozen else 'trainable'}/{'shift' if shift else 'noshift'}"
        out[tag] = [run(sd, frozen, shift) for sd in SEEDS]
        print(f"  ran {tag}", flush=True)

print("\n" + "="*94)
print("H3 - no-shift controls (a spike here would mean the excursion is not the shift)")
print("="*94)
print(f"  {'arm':<22}{'signal':<26}{'peak z after step 200':>24}")
for tag in ("trainable/noshift", "frozen/noshift"):
    for name, key in (("global grad norm", "gn"), (f"ratio {ABOVE}", "R")):
        zs = [detect(r["gn"] if key=="gn" else r["R"][ABOVE])["peak_z"] for r in out[tag]]
        print(f"  {tag:<22}{name:<26}{st.mean(zs):>24.2f}")

print("\n" + "="*94)
print("H1/H2 - the shift, global grad norm vs the layer immediately above the embedding")
print("="*94)
rows = {}
for tag in ("trainable/shift", "frozen/shift"):
    print(f"\n  --- {tag} ---")
    print(f"  {'signal':<30}{'detect@':>9}{'peak z':>9}{'peak@':>8}{'baseline':>12}{'peak':>12}{'x base':>9}")
    for name, get in [("global grad norm", lambda r: r["gn"])] + \
                     [(f"ratio {k}", (lambda k: (lambda r: r["R"][k]))(k)) for k in TRACK]:
        ds = [detect(get(r)) for r in out[tag]]
        f = [d["first"] for d in ds if d["first"] is not None]
        rows[(tag, name)] = dict(
            detect=st.mean(f) if len(f) == len(ds) else None,
            peak_z=st.mean(d["peak_z"] for d in ds),
            peak_at=st.mean(d["peak_at"] for d in ds),
            base=st.mean(d["baseline"] for d in ds),
            peak=st.mean(d["peak_val"] for d in ds))
        r_ = rows[(tag, name)]
        print(f"  {name:<30}{str(r_['detect']):>9}{r_['peak_z']:>9.2f}{r_['peak_at']:>8.0f}"
              f"{r_['base']:>12.3e}{r_['peak']:>12.3e}{r_['peak']/r_['base']:>9.2f}")

g = rows[("frozen/shift"), "global grad norm"]; a = rows[("frozen/shift"), f"ratio {ABOVE}"]
print(f"\n  H2, frozen arm:  above-embedding peak z = {a['peak_z']:.2f} vs global {g['peak_z']:.2f}"
      f"  ({a['peak_z']/g['peak_z']:.2f}x)")
print(f"                   detection step: above {a['detect']}  vs global {g['detect']}")
print(f"  H2 {'SUPPORTED' if a['peak_z'] > g['peak_z'] else 'REFUTED -- the global norm is the better indicator'}")
gt = rows[("trainable/shift"), f"ratio {ABOVE}"]
print(f"\n  H1:  above-embedding peak/baseline  frozen {a['peak']/a['base']:.2f}x "
      f"vs trainable {gt['peak']/gt['base']:.2f}x  -> "
      f"{'SUPPORTED' if a['peak']/a['base'] > gt['peak']/gt['base'] else 'REFUTED'}")

json.dump(dict(config=dict(steps=STEPS, shift=SHIFT, lr=LR_, seeds=list(SEEDS),
                           block_acc=float(LB["val_block_acc"]), above=ABOVE),
               rows={f"{k[0]}|{k[1]}": v for k, v in rows.items()},
               traces={t: dict(gn=[r["gn"] for r in v],
                               above=[r["R"][ABOVE] for r in v],
                               loss=[r["loss"] for r in v]) for t, v in out.items()}),
          open("../results/c2_scar.json","w"), indent=1)
print("\nwrote ../results/c2_scar.json")
