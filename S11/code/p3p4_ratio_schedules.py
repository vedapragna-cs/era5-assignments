"""S11 Part 3 + Part 4 - update-to-weight ratios, and cosine vs WSD stopped early.

Part 3: log ||dw||/||w|| per layer per step; find where warmup stops changing it.
Part 4: same model, same data, same seed, two schedules, both stopped at step 200.

CPU. Uses the S7 corpus so the tokens are real, not synthetic.
"""
import json, math, os, time, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

Q = os.path.expanduser("~/shared/gpu-queue")
M = json.load(open(f"{Q}/data/meta.json")); NV = M["vocab"]
train = torch.from_numpy(np.load(f"{Q}/data/train.npy").astype(np.int64))
val   = torch.from_numpy(np.load(f"{Q}/data/val.npy").astype(np.int64))

D, L, H, SEQ, BATCH = 128, 2, 2, 128, 8
STEPS, WARM, LR, STOP = 300, 30, 2e-3, 200
torch.set_num_threads(os.cpu_count() or 4)

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.proj = nn.Linear(d, 3*d), nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        B, T, d = x.shape; y = s.n1(x)
        q, k, v = s.qkv(y).view(B, T, 3, s.h, d//s.h).permute(2,0,3,1,4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(a.transpose(1,2).reshape(B, T, d))
        return x + s.mlp(s.n2(x))

class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb, s.pos = nn.Embedding(NV, D), nn.Embedding(SEQ, D)
        nn.init.normal_(s.emb.weight, std=0.02)
        s.blocks = nn.ModuleList([Block(D, H) for _ in range(L)])
        s.nf, s.head = nn.LayerNorm(D), nn.Linear(D, NV, bias=False)
    def forward(s, x):
        h = s.emb(x) + s.pos(torch.arange(x.shape[1]))[None]
        for b in s.blocks: h = b(h)
        return s.head(s.nf(h))

def batches(seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        i = torch.randint(len(train)-SEQ-1, (BATCH,), generator=g)
        yield (torch.stack([train[j:j+SEQ] for j in i]),
               torch.stack([train[j+1:j+SEQ+1] for j in i]))

@torch.no_grad()
def val_loss(model, n=40):
    model.eval(); tot = cnt = 0.0
    for k in range(n):
        x = val[k*SEQ:(k+1)*SEQ][None]; y = val[k*SEQ+1:(k+1)*SEQ+1]
        l = F.cross_entropy(model(x)[0], y, reduction="sum")
        tot += l.item(); cnt += y.numel()
    model.train(); return tot/cnt

# ---- the two schedules, as multipliers on LR ------------------------------------
def cosine(s):
    """linear warmup, then cosine to 10% over the FULL 300 steps."""
    if s < WARM: return (s+1)/WARM
    p = (s-WARM)/(STEPS-WARM)
    return 0.1 + 0.9*0.5*(1+math.cos(math.pi*p))

WSD_DECAY = 0.2          # last 20% is the decay leg
def wsd(s):
    """warmup, then CONSTANT, then a short linear decay to 10% at the very end."""
    if s < WARM: return (s+1)/WARM
    d0 = int(STEPS*(1-WSD_DECAY))
    if s < d0: return 1.0
    return 1.0 - 0.9*(s-d0)/(STEPS-d0)

TRACK = ["emb.weight", "blocks.0.qkv.weight", "blocks.0.mlp.0.weight",
         "blocks.1.qkv.weight", "blocks.1.mlp.0.weight", "head.weight"]

def train_run(sched, tag, track_ratio=False, seed=0):
    torch.manual_seed(seed)
    model = GPT(); opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                           betas=(0.9,0.95), weight_decay=0.1)
    named = dict(model.named_parameters())
    it, t0 = batches(seed), time.time()
    ratios = {k: [] for k in TRACK}; curve = []; stop_loss = None
    for s in range(STEPS):
        mult = sched(s)
        for g in opt.param_groups: g["lr"] = LR*mult
        x, y = next(it)
        loss = F.cross_entropy(model(x).reshape(-1,NV), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        before = {k: named[k].detach().clone() for k in TRACK} if track_ratio else None
        opt.step()
        if track_ratio:
            for k in TRACK:
                w = named[k].detach()
                ratios[k].append(((w-before[k]).norm()/w.norm()).item())
        if (s+1) % 20 == 0 or s == 0:
            curve.append((s+1, round(loss.item(), 4), round(LR*mult, 6)))
        if s+1 == STOP: stop_loss = val_loss(model)
        if (s+1) % 50 == 0:
            print(f"  [{tag}] step {s+1:4d} lr {LR*mult:.2e} train {loss.item():.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    return dict(tag=tag, seed=seed, ratios=ratios, curve=curve,
                stop_loss=stop_loss, final_loss=val_loss(model),
                minutes=round((time.time()-t0)/60,1))

print("="*78); print("PART 3 - update-to-weight ratio per layer"); print("="*78)
r3 = train_run(cosine, "cosine+ratio", track_ratio=True)

print(f"\n{'step':>5}" + "".join(f"{k.replace('.weight','').replace('blocks.','L'):>16}" for k in TRACK))
for s in [1,2,5,10,15,20,25,29,30,31,35,40,50,60,80,100,150,200,300]:
    if s <= STEPS:
        print(f"{s:>5}" + "".join(f"{r3['ratios'][k][s-1]:>16.3e}" for k in TRACK))

print(f"\n  warmup ends at step {WARM} (schedule multiplier reaches 1.0 there)")
print(f"\n  {'layer':>26}{'argmax step':>13}{'peak ratio':>13}{'ratio@300':>12}{'peak/final':>12}")
summary = {}
for k in TRACK:
    a = r3["ratios"][k]; pk = int(np.argmax(a))+1
    summary[k] = dict(peak_step=pk, peak=a[pk-1], final=a[-1])
    print(f"  {k:>26}{pk:>13}{a[pk-1]:>13.3e}{a[-1]:>12.3e}{a[pk-1]/a[-1]:>12.2f}")

print("\n"+"="*78); print("PART 4 - cosine vs WSD, both stopped at step 200"); print("="*78)
print("  Three seeds per schedule, paired on seed. A single-seed schedule comparison is exactly")
print("  the unreplicated claim this assignment's epigraph warns about, and it is cheap to avoid.\n")
SEEDS = (0, 1, 2)
p4 = {"cosine": [], "wsd": []}
for sd in SEEDS:
    p4["cosine"].append(train_run(cosine, f"cosine s{sd}", seed=sd))
    p4["wsd"].append(train_run(wsd,    f"wsd s{sd}",    seed=sd))

print(f"\n  {'seed':>5}{'cos@200':>11}{'wsd@200':>11}{'d@200':>9}{'cos@300':>11}{'wsd@300':>11}{'d@300':>9}")
d200, d300 = [], []
for i, sd in enumerate(SEEDS):
    c, w = p4["cosine"][i], p4["wsd"][i]
    a, b = w["stop_loss"]-c["stop_loss"], w["final_loss"]-c["final_loss"]
    d200.append(a); d300.append(b)
    print(f"  {sd:>5}{c['stop_loss']:>11.4f}{w['stop_loss']:>11.4f}{a:>+9.4f}"
          f"{c['final_loss']:>11.4f}{w['final_loss']:>11.4f}{b:>+9.4f}")
import statistics as st
print(f"\n  lr at step 200:  cosine {LR*cosine(STOP-1):.2e}   wsd {LR*wsd(STOP-1):.2e}")
print(f"  wsd - cosine @200: mean {st.mean(d200):+.4f}  spread {max(d200)-min(d200):.4f}  "
      f"signs agree: {len(set(x>0 for x in d200))==1}")
print(f"  wsd - cosine @300: mean {st.mean(d300):+.4f}  spread {max(d300)-min(d300):.4f}  "
      f"signs agree: {len(set(x>0 for x in d300))==1}")
json.dump(dict(config=dict(D=D,L=L,H=H,SEQ=SEQ,BATCH=BATCH,STEPS=STEPS,WARM=WARM,LR=LR,STOP=STOP,
                           vocab=NV, wsd_decay=WSD_DECAY),
               part3=dict(ratios=r3["ratios"], summary=summary, warm=WARM),
               part4=dict(seeds=list(SEEDS),
                          lr_at_stop=dict(cosine=LR*cosine(STOP-1), wsd=LR*wsd(STOP-1)),
                          cosine=[dict(seed=r["seed"], stop=r["stop_loss"], final=r["final_loss"],
                                       curve=r["curve"]) for r in p4["cosine"]],
                          wsd=[dict(seed=r["seed"], stop=r["stop_loss"], final=r["final_loss"],
                                    curve=r["curve"]) for r in p4["wsd"]],
                          delta200=d200, delta300=d300)),
          open("../results/p3p4.json","w"), indent=1)
print("\nwrote ../results/p3p4.json")
