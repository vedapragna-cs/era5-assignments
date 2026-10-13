JOB = "018_e21f_lrsweep"
"""E21f - IS THE BASELINE TUNED?  A crippled baseline makes every comparison worthless, and
nothing in S7 has ever swept a hyperparameter -- every arm used lr=3e-4 because that is what the
first script used.  This sweeps the learning rate on the PUBLISHED byte codec, the baseline every
other claim is measured against, at the original 3,000-step scale.

If 3e-4 is not at or near the optimum, then the byte baseline was handicapped, every "arm X beats
byte" statement in S7 is inflated by the handicap, and the affected comparisons must be re-run at
the tuned setting.  Committed in advance: if any swept LR beats 3e-4 by more than the byte seed
spread (0.0073 nats), we report the baseline as untuned and re-run the headline pairs.

Also sweeps the dense control, because a tuned codec against an untuned dense table would flatter
the codec in the opposite direction.

"""
"""E21 - HARDENING PASS.  Everything in S7 to date rests on 2-3 seeds at a single scale
(3,000 steps, ~0.9 epochs).  Three things a reviewer kills first, and what this fixes:

 (1) SCALE.  arXiv:2605.09751 finds randomly-recoded fixed input codes reach PARITY with
     structured ones at 17B tokens - 375x our budget.  That is the best-calibrated prior
     that every effect we report is a learning-speed artifact of an undertrained regime.
     Fix: run 9,000 steps (3x) and evaluate at [1000, 2000, 3000, 4500, 6000, 9000], so one
     run yields a six-point trajectory.  If delta_perm shrinks monotonically toward zero,
     the parity result explains us and we say so.  The 3,000-step point also REPLICATES
     the existing E7/E14/E19 numbers on an independent set of seeds.

 (2) A REAL METHODOLOGICAL FLAW IN E14/E18/E19.  The permuted arms sampled permutation
     draws at a SINGLE model seed, so their init variance was never sampled, and a 3-draw
     mean was compared against a 2-seed mean as if the two spreads were commensurable.
     Fix: every shuf run varies model seed AND permutation draw together.

 (3) STATISTICS.  "18.3x the seed spread" is not a test.  Fix: save per-token validation
     losses so arms can be compared by a PAIRED bootstrap over the 200K val tokens, which
     controls the val-split sampling that n=1 val split never did, and report seed-level
     means with intervals rather than spreads.

Nothing else changes: same data, same model, same optimiser, same eval protocol as
007_e7 / 010_e14 / 012_e19, so every number here is directly comparable to those.
"""
import json, math, time, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

M = json.load(open("data/meta.json"))
NV, P, NB = M["vocab"], M["P"], M["NB"]
train = torch.from_numpy(np.load("data/train.npy").astype(np.int64))
val   = torch.from_numpy(np.load("data/val.npy").astype(np.int64))
vlang = torch.from_numpy(np.load("data/val_lang.npy").astype(np.int64))
C  = np.load("data/codec.npz")
blk, off = C["blk_idx"], C["off_idx"]
bv = C["byte_idx"].astype(np.int64)
nb = C["nbytes"].astype(np.float32)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

D_MODEL, N_LAYER, N_HEAD, SEQ, BATCH = 256, 4, 4, 256, 8
STEPS, WARM, LR = 3000, 150, 3e-4
CKPT = [3000]
print(f"device={dev} vocab={NV} steps={STEPS} ckpts={CKPT}", flush=True)
freq = np.bincount(train.numpy().ravel(), minlength=NV).astype(np.int64)
AR = np.arange(P)[None, :]

def build(factors):
    cols, base = [], 0
    for mat, card in factors:
        cols.append(np.where(mat >= 0, base + mat*P + AR, -1)); base += card*P
    idx = np.concatenate(cols, 1)
    return np.where(idx >= 0, idx, base).astype(np.int64), base

def byte_idx():   return build([(C["byte_idx"], 256)])
def char4_idx():  return build([(blk, NB), (np.where(off>=0, off>>5,-1),4),
                                (np.where(off>=0,(off>>2)&7,-1),8), (np.where(off>=0, off&3,-1),4)])
NBIG = 8192
def bigrampos_idx():
    D = 256*P + NBIG
    bc = np.where(bv >= 0, bv*P + AR, -1)
    gc = np.full((bv.shape[0], P-1), -1, np.int64)
    for i in range(P-1):
        a, b = bv[:, i], bv[:, i+1]; m = (a >= 0) & (b >= 0)
        gc[m, i] = 256*P + ((a[m]*257 + b[m]*1013904223 + i*2654435761) & 0x7FFFFFFF) % NBIG
    idx = np.concatenate([bc, gc], 1)
    return np.where(idx >= 0, idx, D).astype(np.int64), D

def rand4_idx():
    idx, D = char4_idx(); a = idx.copy(); live = a != D
    a[live] = np.random.default_rng(12345).integers(0, D, size=live.sum())
    return a, D

def shuf(idx, D, seed, cols=None, ns=400):
    """Permute WHICH TOKEN receives which code, within (freq quantile x slot count) strata.
    cols=None permutes the whole code; cols=slice permutes only those columns."""
    g = np.random.default_rng(seed); new = idx.copy()
    sub = idx if cols is None else idx[:, cols]
    slots = (sub != D).sum(1)
    lf = np.where(freq > 0, np.log10(freq + 1), -1.0)
    qs = np.quantile(lf[freq > 0], np.linspace(0, 1, ns + 1))
    key = np.digitize(lf, qs[1:-1]) * 1000 + slots
    for k in np.unique(key):
        ix = np.where(key == k)[0]
        if len(ix) > 1:
            if cols is None: new[ix] = idx[g.permutation(ix)]
            else:            new[ix, cols] = idx[g.permutation(ix)][:, cols]
    return new

def audit(a, b, D, name):
    la = len(np.unique(a[a != D])); lb = len(np.unique(b[b != D]))
    ka = np.bincount(a[a != D].ravel(), minlength=D+1); kb = np.bincount(b[b != D].ravel(), minlength=D+1)
    qs = [10,25,50,75,90,99]
    rat = [round(float(np.percentile(kb[kb>0],q)/max(np.percentile(ka[ka>0],q),1e-9)),3) for q in qs]
    print(f"  [audit {name}] live rows {la}->{lb} | touches {int(ka.sum())}->{int(kb.sum())} | load q{qs}={rat}", flush=True)
    return dict(live_rows=[la,lb], touches=[int(ka.sum()),int(kb.sum())], load_ratio=rat)

class KronEmb(nn.Module):
    def __init__(s, idx, D, lens, d):
        super().__init__()
        s.register_buffer("idx", torch.from_numpy(idx))
        s.register_buffer("inv", (1.0/torch.from_numpy(lens).clamp(min=1).sqrt()).unsqueeze(1))
        s.D = D
        s.W = nn.Parameter(torch.randn(D+1, d)*0.02)
        with torch.no_grad(): s.W[D].zero_()
    def forward(s, ids):
        u, inv = torch.unique(ids, return_inverse=True)
        e = s.W[s.idx[u]].sum(1) * s.inv[u]
        e = (e - e.mean(1, keepdim=True)) / (e.std(1, keepdim=True) + 1e-5)
        return e[inv].view(*ids.shape, -1)

class DenseEmb(nn.Module):
    def __init__(s, V, d, zn=True):
        super().__init__(); s.e = nn.Embedding(V, d); s.zn = zn
        nn.init.normal_(s.e.weight, std=0.02); s.D = None
    def forward(s, ids):
        e = s.e(ids)
        if s.zn: e = (e - e.mean(-1, keepdim=True)) / (e.std(-1, keepdim=True) + 1e-5)
        return e

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
    def __init__(s, emb, d, L, h):
        super().__init__()
        s.emb, s.pos = emb, nn.Embedding(SEQ, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        s.nf, s.head = nn.LayerNorm(d), nn.Linear(d, NV, bias=False)
    def forward(s, x):
        h = s.emb(x) + s.pos(torch.arange(x.shape[1], device=x.device))[None]
        for b in s.blocks: h = b(h)
        return s.head(s.nf(h))

def batches(data, seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        i = torch.randint(len(data)-SEQ-1, (BATCH,), generator=g)
        yield (torch.stack([data[j:j+SEQ] for j in i]).to(dev),
               torch.stack([data[j+1:j+SEQ+1] for j in i]).to(dev))

nb_t = torch.from_numpy(nb); freq_t = torch.from_numpy(freq)
BUCKETS = [(0,0),(1,9),(10,99),(100,999),(1000,10**9)]

@torch.no_grad()
def evaluate(model, keep_tokens=False):
    """Returns aggregate metrics AND, optionally, the per-token loss vector so that arms
    can be compared by a paired bootstrap over val tokens rather than over 2-3 seeds."""
    model.eval(); tot, cnt, byt = torch.zeros(3), torch.zeros(3), torch.zeros(3)
    bs = {f"{a}-{b}": [0.0, 0] for a,b in BUCKETS}; per_tok = []
    for k in range((len(val)-1)//SEQ):
        x = val[k*SEQ:(k+1)*SEQ][None].to(dev); y = val[k*SEQ+1:(k+1)*SEQ+1]
        l = F.cross_entropy(model(x)[0], y.to(dev), reduction="none").float().cpu()
        if keep_tokens: per_tok.append(l.numpy().astype(np.float32))
        lg, bt, fq = vlang[k*SEQ+1:(k+1)*SEQ+1], nb_t[y], freq_t[y]
        for g in range(3):
            m = lg == g
            if m.any(): tot[g]+=l[m].sum(); cnt[g]+=m.sum(); byt[g]+=bt[m].sum()
        for a,b in BUCKETS:
            m = (fq >= a) & (fq <= b)
            if m.any():
                key=f"{a}-{b}"; bs[key][0]+=l[m].sum().item(); bs[key][1]+=int(m.sum())
    model.train()
    per = {M["langs"][g]: {"loss": round((tot[g]/cnt[g]).item(),4),
                           "bpb": round((tot[g]/byt[g]/math.log(2)).item(),4)} for g in range(3)}
    buck = {k: {"nll": round(v[0]/max(v[1],1),4), "n": v[1]} for k,v in bs.items()}
    return ((tot.sum()/cnt.sum()).item(), per, round((tot.sum()/byt.sum()/math.log(2)).item(),4),
            buck, (np.concatenate(per_tok) if keep_tokens else None))

def run(tag, make_emb, model_seed, extra=None, lr=None):
    lr = LR if lr is None else lr
    torch.manual_seed(model_seed)
    emb = make_emb()
    model = GPT(emb, D_MODEL, N_LAYER, N_HEAD).to(dev)
    n_in = sum(p.numel() for p in model.emb.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.1)
    sch = lambda s: min(1.0,(s+1)/WARM)*(0.1+0.45*(1+math.cos(math.pi*min(1.0,s/STEPS))))
    it, t0 = batches(train, model_seed), time.time()
    traj = {}
    for s in range(STEPS):
        for g in opt.param_groups: g["lr"] = lr*sch(s)
        x, y = next(it)
        loss = F.cross_entropy(model(x).reshape(-1,NV), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if getattr(model.emb,"D",None) is not None:
            with torch.no_grad(): model.emb.W[model.emb.D].zero_()
        st = s+1
        if st in CKPT:
            last = st == CKPT[-1]
            vl, per, bpb, buck, pt = evaluate(model, keep_tokens=last)
            traj[st] = dict(val_loss=round(vl,4), val_bpb=bpb, per_lang=per, buckets=buck)
            print(f"  {tag} s{model_seed} step {st:5d} val {vl:.4f} bpb {bpb:.4f} {time.time()-t0:.0f}s", flush=True)
            if last and pt is not None:
                os.makedirs("results/pertok", exist_ok=True)
                np.save(f"results/pertok/{JOB}_{tag}_s{model_seed}.npy", pt.astype(np.float16))
    r = dict(arm=tag, model_seed=model_seed, input_params=n_in, steps=STEPS,
             trajectory=traj, val_loss=traj[CKPT[-1]]["val_loss"],
             minutes=round((time.time()-t0)/60,1))
    if extra: r.update(extra)
    print(json.dumps({k:v for k,v in r.items() if k!="trajectory"}), flush=True)
    return r

LENS = C["nbytes"].astype(np.float32)
results = []
def save(): json.dump(results, open(f"results/{JOB}.json","w"), indent=1)
idx, D = byte_idx()
for lr in (1e-4, 3e-4, 6e-4, 1e-3):
    results.append(run("byte", lambda: KronEmb(idx, D, LENS.copy(), D_MODEL), 0,
                       extra=dict(lr=lr), lr=lr)); save()
for lr in (3e-4, 1e-3):
    results.append(run("dense_zn", lambda: DenseEmb(NV, D_MODEL, zn=True), 0,
                       extra=dict(lr=lr), lr=lr)); save()

print(f"\n=== {JOB} summary ===")
print(f"{'arm':<12}{'lr':>9}{'val loss @3000':>16}")
for r in results:
    print(f"{r['arm']:<12}{r['lr']:>9.1e}{r['val_loss']:>16.4f}")
print("\nPublished byte @3000 with lr=3e-4: 6.1521 (seeds 0-1, spread 0.0073)")
print("Pre-registered: if any LR beats 3e-4 by >0.0073 nats, the baseline was untuned.")
