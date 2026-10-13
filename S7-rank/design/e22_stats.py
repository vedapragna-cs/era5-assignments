"""E22 -- statistical hardening of the rank claim.  Three reviewer objections, answered:
 (1) rank was fitted and evaluated on the same 7 arms -> test it on the 5 added afterwards;
 (2) "rho = -1.0000" hides how much is actually explained -> fit it and report R^2 + residuals;
 (3) rand4 was treated as "structure destroyed" -> it also changed row-load uniformity, measured here.
Run:  python e22_stats.py        (self-contained; reads design/data/)
"""
import numpy as np, json, os

# arm, rank(S) [computed in rankall.py, independently of loss], val loss @3000 steps, structure-intact
ARMS = [("char4",349,6.3232,1),("char4_shuf",349,6.5577,0),("char3",444,6.2971,1),
        ("char2",1199,6.1768,1),("byte",1733,6.1521,1),("hashbyte4",1733,6.1426,1),
        ("byte_shuf",1733,6.2858,0),("hashchar4",2382,6.1220,1),("rand4",2560,6.0328,0),
        ("bigram_free",4459,6.0723,1),("bigram_pos",5668,6.0388,1),("bigram_pos_shuf",5668,6.0004,0)]
INSAMPLE = {"char4","char3","char2","byte","hashbyte4","hashchar4","rand4"}   # the 7 E16 was built on

def spearman(x, y):
    rk = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(rk(x), rk(y))[0, 1])

def main():
    name = np.array([a[0] for a in ARMS]); rk = np.array([a[1] for a in ARMS], float)
    ls = np.array([a[2] for a in ARMS]); st = np.array([a[3] for a in ARMS], float)
    tr = np.array([n in INSAMPLE for n in name])
    print(f"in-sample  (7 arms E16 was built on): Spearman = {spearman(rk[tr], ls[tr]):+.4f}")
    print(f"held out   (5 arms added afterwards): Spearman = {spearman(rk[~tr], ls[~tr]):+.4f}")
    print(f"  {', '.join(name[~tr])}")
    print(f"pooled (12)                         : Spearman = {spearman(rk, ls):+.4f}\n")

    X = np.c_[np.ones(len(rk)), np.log10(rk)]
    b, *_ = np.linalg.lstsq(X, ls, rcond=None); pred = X @ b; res = ls - pred
    r2 = 1 - (res**2).sum() / ((ls - ls.mean())**2).sum()
    print(f"loss ~ {b[0]:.4f} {b[1]:+.4f}*log10(rank)   R^2={r2:.4f}  resid SD={res.std(ddof=2):.4f}\n")
    print(f"{'arm':<17}{'rank':>7}{'loss':>9}{'pred':>9}{'resid':>9}  code")
    for i in np.argsort(res):
        print(f"{name[i]:<17}{int(rk[i]):>7}{ls[i]:>9.4f}{pred[i]:>9.4f}{res[i]:>+9.4f}"
              f"  {'intact' if st[i] else 'randomised/permuted'}")
    X2 = np.c_[X, st]; b2, *_ = np.linalg.lstsq(X2, ls, rcond=None); r22 = 1 - ((ls - X2@b2)**2).sum()/((ls-ls.mean())**2).sum()
    print(f"\n+structure indicator: R^2 {r2:.4f} -> {r22:.4f}, coefficient {b2[2]:+.4f} nats")
    print("NOTE: this indicator is mis-specified -- rand4 is non-structured but has the most")
    print("negative residual.  See load_uniformity(): rand4 also changed the load distribution.\n")

    print("exactly-rank-matched pairs (model-free; rank and load identical within pair):")
    for a, c in [("char4","char4_shuf"),("byte","byte_shuf"),("bigram_pos","bigram_pos_shuf")]:
        i, j = list(name).index(a), list(name).index(c)
        print(f"  {a:<16}{ls[i]:.4f}  vs  {c:<17}{ls[j]:.4f}   delta {ls[j]-ls[i]:+.4f}  (rank {int(rk[i])})")

def load_uniformity():
    """Frequency-weighted gradient touches per live row.  A permutation preserves this exactly;
    rand4 does not -- it is uniform by construction, which is the third confound."""
    D_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    C = np.load(os.path.join(D_, "codec.npz")); M = json.load(open(os.path.join(D_, "meta.json")))
    NV, P, NB = M["vocab"], M["P"], M["NB"]
    bv = C["byte_idx"].astype(np.int64); blk, off = C["blk_idx"], C["off_idx"]
    freq = np.load(os.path.join(D_, "train_freq.npy")).astype(float); AR = np.arange(P)[None, :]
    def build(fs):
        cols, base = [], 0
        for m, card in fs: cols.append(np.where(m >= 0, base + m*P + AR, -1)); base += card*P
        idx = np.concatenate(cols, 1); return np.where(idx >= 0, idx, base).astype(np.int64), base
    S = {"byte": build([(C["byte_idx"],256)]), "char2": build([(blk,NB),(off,128)]),
         "char4": build([(blk,NB),(np.where(off>=0,off>>5,-1),4),
                         (np.where(off>=0,(off>>2)&7,-1),8),(np.where(off>=0,off&3,-1),4)])}
    i4, D4 = S["char4"]; a = i4.copy(); lv = a != D4
    a[lv] = np.random.default_rng(12345).integers(0, D4, size=lv.sum()); S["rand4"] = (a, D4)
    NBIG = 8192; D = 256*P + NBIG
    bc = np.where(bv >= 0, bv*P + AR, -1); gc = np.full((NV, P-1), -1, np.int64)
    for i in range(P-1):
        x, y = bv[:, i], bv[:, i+1]; m = (x >= 0) & (y >= 0)
        gc[m, i] = 256*P + ((x[m]*257 + y[m]*1013904223 + i*2654435761) & 0x7FFFFFFF) % NBIG
    cat = np.concatenate([bc, gc], 1); S["bigram_pos"] = (np.where(cat >= 0, cat, D).astype(np.int64), D)
    print(f"\n{'arm':<13}{'live':>8}{'Gini':>9}{'eff rows':>11}{'eff/live':>10}")
    for n, (idx, Dd) in S.items():
        w = np.zeros(Dd+1)
        for c in range(idx.shape[1]): np.add.at(w, idx[:, c], freq)
        w = w[:Dd]; w = w[w > 0]; p = w / w.sum()
        gini = 1 - 2*np.sum(np.cumsum(np.sort(p))/len(p)) + 1/len(p)
        eff = float(np.exp(-(p*np.log(p)).sum()))
        print(f"{n:<13}{len(w):>8}{gini:>9.4f}{eff:>11.1f}{eff/len(w):>10.4f}")

if __name__ == "__main__":
    main()
    try: load_uniformity()
    except FileNotFoundError as e: print(f"\n(load_uniformity needs local npz files: {e})")
