import numpy as np, json
C = np.load("codec_local.npz"); H = np.load("hash_local.npz")
tr = np.load("train_local.npy")
M = json.load(open("meta_local.json"))
NV, P, NB = M["vocab"], M["P"], M["NB"]
blk, off = C["blk_idx"], C["off_idx"]
AR = np.arange(P)[None, :]
def build(factors):
    cols, base = [], 0
    for mat, card in factors:
        cols.append(np.where(mat >= 0, base + mat*P + AR, -1)); base += card*P
    idx = np.concatenate(cols, 1)
    return np.where(idx >= 0, idx, base).astype(np.int64), base
SPECS = {
 "byte":  lambda: build([(C["byte_idx"], 256)]),
 "char2": lambda: build([(blk, NB), (off, 128)]),
 "char3": lambda: build([(blk, NB), (np.where(off>=0, off>>3,-1),16), (np.where(off>=0, off&7,-1),8)]),
 "char4": lambda: build([(blk, NB), (np.where(off>=0, off>>5,-1),4),
                         (np.where(off>=0,(off>>2)&7,-1),8), (np.where(off>=0, off&3,-1),4)]),
}
def rand4():
    idx, D = SPECS["char4"]()
    a = idx.copy(); live = a != D
    g = np.random.default_rng(12345)
    a[live] = g.integers(0, D, size=live.sum())
    return a, D
SPECS["rand4"] = rand4

freq = np.bincount(tr.ravel(), minlength=NV)
seen = np.where(freq >= 1)[0]
print(f"train tokens={tr.size}  vocab={NV}  types seen={len(seen)}\n")
rows=[]
for name in ["byte","char2","char3","char4","rand4"]:
    idx, D = SPECS[name]()
    sub = idx[seen]                      # rows touched by any token that appears in train
    liveslots = sub[sub != D]
    nlive = len(np.unique(liveslots))
    rows.append((name, D, (D+1)*256, nlive, nlive*256))
for name in ["hashbyte4","hashchar4"]:
    idx = H["hashbyte" if name=="hashbyte4" else "hashchar"].astype(np.int64)
    D = int(H["D"])
    sub = idx[seen]; liveslots = sub[sub != D]
    nlive = len(np.unique(liveslots))
    rows.append((name, D, (D+1)*256, nlive, nlive*256))
print(f"{'arm':<11}{'D':>7}{'alloc_params':>14}{'live_rows':>11}{'live%':>8}{'trained_params':>16}")
for n,D,ap,lr,tp in rows:
    print(f"{n:<11}{D:>7}{ap:>14,}{lr:>11,}{100*lr/D:>7.1f}%{tp:>16,}")
