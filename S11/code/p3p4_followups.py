"""S11 follow-ups forced by the first p3/p4 results.

F1  The step-200 comparison the assignment specifies is NOT average-LR matched.
    cosine's mean multiplier over the post-warmup run is 0.55; WSD's is 0.90 -- 1.64x.
    So "WSD ends 0.33 nats better" may be an average-LR effect, not a schedule-shape one.
    Controls: WSD with peak scaled DOWN to match cosine's mean, and cosine scaled UP to
    match WSD's mean. This is the assignment's own epigraph applied to its own experiment.

F2  "Which model would you keep" cannot be answered from the two arms it asks for.
    Cosine-for-300 stopped at 200 is MID-DECAY and committed; WSD-for-300 stopped at 200 is
    at full LR and UNANNEALED. Neither is what you would actually ship at 200. The missing
    arm is WSD ANNEALED TO 200. V4 hit exactly this: "forces a choice between recovering at
    the low annealed learning rate ... and raising the rate again, which discards the
    annealing already paid for."

F3  Negative control for part 3: warmup disabled.

F4  The embedding's update-to-weight ratio is DILUTED by the untouched rows. Measured over
    the whole table vs over only the rows the batch touched. This is the mechanism behind
    S7 §9's rule that the global norm "averages the signal away", measured inside one layer.
"""
import json, math, os, time, statistics as st
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
exec(open("p3p4_ratio_schedules.py").read().split('print("="*78)')[0].split("# ---- the two")[0])

WSD_DECAY = 0.2
def cosine(s, tot=STEPS):
    if s < WARM: return (s+1)/WARM
    return 0.1 + 0.45*(1+math.cos(math.pi*(s-WARM)/(tot-WARM)))
def wsd(s, tot=STEPS, decay=WSD_DECAY):
    if s < WARM: return (s+1)/WARM
    d0 = int(tot*(1-decay))
    return 1.0 if s < d0 else max(0.1, 1.0 - 0.9*(s-d0)/(tot-d0))
def wsd_to(stop):                       # anneal so the DECAY LEG ENDS AT `stop`
    d0 = int(stop*(1-WSD_DECAY))
    def f(s):
        if s < WARM: return (s+1)/WARM
        if s < d0:   return 1.0
        if s < stop: return max(0.1, 1.0 - 0.9*(s-d0)/(stop-d0))
        return 0.1
    return f
def nowarm(s, tot=STEPS):
    return 0.1 + 0.45*(1+math.cos(math.pi*min(1.0, s/tot)))

def mean_mult(sched, n=STEPS):          # average LR multiplier, warmup excluded
    return st.mean(sched(s) for s in range(WARM, n))

TRACK = ["emb.weight","blocks.0.qkv.weight","blocks.0.mlp.0.weight",
         "blocks.1.qkv.weight","blocks.1.mlp.0.weight","head.weight"]

def run(sched, tag, seed=0, lr=LR, ratios=False, emb_split=False, stops=(200, 300)):
    torch.manual_seed(seed)
    model = GPT(); opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                           betas=(0.9,0.95), weight_decay=0.1)
    named = dict(model.named_parameters()); it = batches(seed)
    R = {k: [] for k in TRACK}; E = {"all": [], "touched": []}; out = {}
    gn = []
    for s in range(STEPS):
        for g in opt.param_groups: g["lr"] = lr*sched(s)
        x, y = next(it)
        loss = F.cross_entropy(model(x).reshape(-1,NV), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        gn.append(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        pre = {k: named[k].detach().clone() for k in TRACK} if (ratios or emb_split) else None
        touched = torch.unique(x) if emb_split else None
        opt.step()
        if ratios:
            for k in TRACK:
                w = named[k].detach(); R[k].append(((w-pre[k]).norm()/w.norm()).item())
        if emb_split:
            w = named["emb.weight"].detach(); d = w - pre["emb.weight"]
            E["all"].append((d.norm()/w.norm()).item())
            E["touched"].append((d[touched].norm()/w[touched].norm()).item())
        if s+1 in stops: out[s+1] = val_loss(model)
    return dict(tag=tag, seed=seed, lr=lr, val=out, ratios=R, emb=E,
                grad_norm=gn, mean_mult=mean_mult(sched))

print("="*78); print("F1/F2 - the step-200 comparison, with the controls it needs"); print("="*78)
mc, mw = mean_mult(cosine), mean_mult(wsd)
print(f"  mean LR multiplier after warmup:  cosine {mc:.4f}   wsd {mw:.4f}   ratio {mw/mc:.3f}x")
print(f"  -> the assignment's two arms are NOT average-LR matched.\n")

ARMS = [
    ("cosine",           cosine,        LR),
    ("wsd",              wsd,           LR),
    ("wsd@matched_lr",   wsd,           LR*mc/mw),   # same mean LR as cosine
    ("cosine@matched_lr",cosine,        LR*mw/mc),   # same mean LR as wsd
    ("wsd_annealed_200", wsd_to(200),   LR),         # the arm the question actually needs
]
SEEDS = (0,1,2)
res = {}
for tag, sch, lr in ARMS:
    res[tag] = [run(sch, tag, seed=sd, lr=lr) for sd in SEEDS]
    v200 = [r["val"][200] for r in res[tag]]; v300 = [r["val"][300] for r in res[tag]]
    print(f"  {tag:<20} lr={lr:.2e} meanmult={res[tag][0]['mean_mult']:.3f}  "
          f"val@200 {st.mean(v200):.4f}  val@300 {st.mean(v300):.4f}", flush=True)

base = st.mean(r["val"][200] for r in res["cosine"])
print(f"\n  {'arm':<20}{'val@200':>10}{'vs cosine':>11}{'spread':>9}{'signs agree':>13}")
for tag,_,_ in ARMS:
    v = [r["val"][200] for r in res[tag]]
    d = [a-b for a,b in zip(v, [r["val"][200] for r in res["cosine"]])]
    print(f"  {tag:<20}{st.mean(v):>10.4f}{st.mean(d):>+11.4f}{max(v)-min(v):>9.4f}"
          f"{str(len(set(x>0 for x in d))==1 if tag!='cosine' else '-'):>13}")

print("\n"+"="*78); print("F3 - warmup negative control"); print("="*78)
w_on  = run(cosine, "warmup",    seed=0, ratios=True)
w_off = run(nowarm, "no_warmup", seed=0, ratios=True)
print(f"  {'layer':>24}{'peak ON':>12}{'@step':>7}{'peak OFF':>12}{'@step':>7}{'OFF/ON':>9}")
for k in TRACK:
    a, b = w_on["ratios"][k], w_off["ratios"][k]
    ia, ib = int(np.argmax(a)), int(np.argmax(b))
    print(f"  {k:>24}{a[ia]:>12.3e}{ia+1:>7}{b[ib]:>12.3e}{ib+1:>7}{b[ib]/a[ia]:>9.2f}")
print(f"\n  step-1 grad norm (pre-clip):  warmup {w_on['grad_norm'][0]:.3f}   "
      f"no-warmup {w_off['grad_norm'][0]:.3f}")
print(f"  clipped steps (>1.0):         warmup {sum(g>1.0 for g in w_on['grad_norm'])}/{STEPS}   "
      f"no-warmup {sum(g>1.0 for g in w_off['grad_norm'])}/{STEPS}")
print(f"  val@300:                      warmup {w_on['val'][300]:.4f}   no-warmup {w_off['val'][300]:.4f}")

print("\n"+"="*78)
print("F4 - the embedding ratio is diluted by untouched rows"); print("="*78)
e = run(cosine, "emb_split", seed=0, emb_split=True)
print(f"  vocab {NV}, tokens touched per step <= {BATCH*SEQ} ({100*BATCH*SEQ/NV:.2f}% of rows)")
print(f"\n{'step':>6}{'ratio over ALL rows':>22}{'ratio over TOUCHED rows':>26}{'dilution':>11}")
for s in (1,2,5,10,20,30,50,100,200,300):
    a, b = e["emb"]["all"][s-1], e["emb"]["touched"][s-1]
    print(f"{s:>6}{a:>22.3e}{b:>26.3e}{b/a:>11.1f}x")
dil = [b/a for a,b in zip(e["emb"]["all"], e["emb"]["touched"])]
print(f"\n  median dilution factor: {st.median(dil):.1f}x")

json.dump(dict(mean_mult=dict(cosine=mc, wsd=mw, ratio=mw/mc),
               arms={t: [dict(seed=r["seed"], lr=r["lr"], val=r["val"],
                              mean_mult=r["mean_mult"]) for r in res[t]] for t,_,_ in ARMS},
               warmup=dict(on={k: w_on["ratios"][k] for k in TRACK},
                           off={k: w_off["ratios"][k] for k in TRACK},
                           gn_on=w_on["grad_norm"][:50], gn_off=w_off["grad_norm"][:50],
                           val_on=w_on["val"], val_off=w_off["val"]),
               emb_dilution=dict(all=e["emb"]["all"], touched=e["emb"]["touched"],
                                 median=st.median(dil), vocab=NV, touched_max=BATCH*SEQ)),
          open("../results/p3p4_followups.json","w"), indent=1)
print("\nwrote ../results/p3p4_followups.json")
