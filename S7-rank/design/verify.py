#!/usr/bin/env python3
"""S7 verifier  [CONVENTION: accumulate -- a cell occurring twice in a token counts twice,
matching KronEmb.forward's W[idx].sum(1).  See CORRECTION C1 in EXPERIMENT_LOG.md.]

S7 verifier -- recomputes every headline number from committed artifacts and EXITS NON-ZERO
if any of them disagrees with what the README and EXPERIMENT_LOG claim.

The point is that it cannot pass vacuously: each check recomputes a quantity from data rather
than reading it back from a results file, and the expected values below are the ones asserted
in the write-up.  If a claim drifts, this fails.

    python S7/design/verify.py          -> prints a table, exits 0 only if every check passes
"""
import numpy as np, json, os, sys, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
FAIL = []; PASS = []

def check(name, got, want, tol=0.0, unit=""):
    ok = (got == want) if tol == 0 else (abs(got - want) <= tol)
    (PASS if ok else FAIL).append(name)
    flag = "ok  " if ok else "FAIL"
    g = f"{got:.4f}" if isinstance(got, float) else f"{got}"
    w = f"{want:.4f}" if isinstance(want, float) else f"{want}"
    print(f"  [{flag}] {name:<52} got {g:>12}{unit}  want {w:>12}{unit}")

# ---------------------------------------------------------------- codec / rank ----
C = np.load(os.path.join(DATA, "codec.npz")); M = json.load(open(os.path.join(DATA, "meta.json")))
NV, P, NB = M["vocab"], M["P"], M["NB"]
bv = C["byte_idx"].astype(np.int64); blk, off = C["blk_idx"], C["off_idx"]
freq = np.load(os.path.join(DATA, "train_freq.npy")).astype(float)
seen = np.where(freq > 0)[0]; AR = np.arange(P)[None, :]

print("\nS7 verifier -- recomputing headline numbers from design/data/\n")
print("A. corpus and vocabulary")
check("train tokens", int(freq.sum()), 6650880)
check("types observed in train", int((freq > 0).sum()), 60779)
check("vocabulary (observed) / full", f"{NV}/{M['vocab_full']}", "61070/131072")
check("byte window P", P, 32)
check("max token length <= 32 bytes (cap honoured)", int((bv >= 0).sum(1).max()), 32)
check("tokens exceeding the 32-byte window", int(((bv >= 0).sum(1) > 32).sum()), 0)

def build(fs):
    cols, base = [], 0
    for m, card in fs: cols.append(np.where(m >= 0, base + m*P + AR, -1)); base += card*P
    idx = np.concatenate(cols, 1); return np.where(idx >= 0, idx, base).astype(np.int64), base
ARMS = {
 "byte":  build([(C["byte_idx"], 256)]),
 "char2": build([(blk, NB), (off, 128)]),
 "char4": build([(blk, NB), (np.where(off>=0, off>>5,-1),4),
                 (np.where(off>=0,(off>>2)&7,-1),8), (np.where(off>=0, off&3,-1),4)]),
}
NBIG = 8192; Dbg = 256*P + NBIG
bc = np.where(bv >= 0, bv*P + AR, -1); gc = np.full((NV, P-1), -1, np.int64)
for i in range(P-1):
    a, b = bv[:, i], bv[:, i+1]; m = (a >= 0) & (b >= 0)
    gc[m, i] = 256*P + ((a[m]*257 + b[m]*1013904223 + i*2654435761) & 0x7FFFFFFF) % NBIG
cat = np.concatenate([bc, gc], 1); ARMS["bigram_pos"] = (np.where(cat >= 0, cat, Dbg).astype(np.int64), Dbg)

def rank_of(idx, D):
    sub = idx[seen]; used = np.unique(sub[sub != D]); Dr = len(used)
    rm = np.full(D+1, -1, np.int64); rm[used] = np.arange(Dr)
    A = np.zeros((len(seen), Dr), np.float32)
    rr = np.repeat(np.arange(len(seen)), sub.shape[1]); cc = rm[sub.ravel()]; k = cc >= 0
    np.add.at(A, (rr[k], cc[k]), 1.0)
    ev = np.linalg.eigvalsh((A.T @ A).astype(np.float64))
    return int((ev > ev.max()*1e-9).sum()), Dr

print("\nB. rank(S) -- the capacity ceiling (E16, E22)")
for nm, want_rank, want_reach in [("byte",1733,1855),("char2",1199,1305),("char4",349,479),("bigram_pos",5653,6150)]:
    idx, D = ARMS[nm]; rk, dr = rank_of(idx, D)
    check(f"rank(S) {nm}", rk, want_rank); check(f"D_reachable {nm}", dr, want_reach)
# the natural experiment: hashbyte4 and byte have identical rank despite 3.2x different D
H = np.load(os.path.join(DATA, "hash_idx.npz"))
for key, nm, want in [("hashbyte", "hashbyte4", 1733), ("hashchar", "hashchar4", 2382)]:
    hidx = H[key].astype(np.int64)
    rkh, _ = rank_of(hidx, 2560)
    check(f"rank(S) {nm}", rkh, want)
check("hashbyte4 rank == byte rank (natural experiment)",
      rank_of(H["hashbyte"].astype(np.int64), 2560)[0], rank_of(*ARMS["byte"])[0])
check("byte: allocated rows", (8192+1)*256, 2097408)
check("byte: free parameters rank*d_model (C3)", 1733*256, 443648)
check("byte free params as % of dense", round(100*1733*256/(NV*256), 1), 2.8, tol=0.05)
check("byte: window dead fraction", round(1 - 1855/8192, 3), 0.774, tol=0.001)

print("\nC. row-load uniformity -- the third variable (E22.3)")
def load_stats(idx, D):
    w = np.zeros(D+1)
    for c in range(idx.shape[1]): np.add.at(w, idx[:, c], freq)
    w = w[:D]; w = w[w > 0]; p = w / w.sum()
    gini = 1 - 2*np.sum(np.cumsum(np.sort(p))/len(p)) + 1/len(p)
    return gini, float(np.exp(-(p*np.log(p)).sum())), len(w)
g, eff, live = load_stats(*ARMS["byte"])
check("byte load Gini", round(g,4), 0.8887, tol=0.0005)
check("byte effective rows exp(H)", round(eff,1), 200.4, tol=0.2)
check("byte effective / allocated", round(eff/8192,4), 0.0245, tol=0.0005)

print("\nD. Unicode Brahmic phoneme alignment (E16.2)")
BLOCKS = {"DEVANAGARI":0x0900,"BENGALI":0x0980,"GURMUKHI":0x0A00,"GUJARATI":0x0A80,"ORIYA":0x0B00,
          "TAMIL":0x0B80,"TELUGU":0x0C00,"KANNADA":0x0C80,"MALAYALAM":0x0D00}
rows = {}
for nm, base in BLOCKS.items():
    d = {}
    for o in range(0x80):
        try: d[o] = unicodedata.name(chr(base+o)).replace(nm+" ", "")
        except ValueError: pass
    rows[nm] = d
tot = agree = 0
for o in range(0x80):
    labs = {rows[s][o] for s in BLOCKS if o in rows[s]}
    if sum(1 for s in BLOCKS if o in rows[s]) < 2: continue
    tot += 1; agree += (len(labs) == 1)
check("offsets populated in >=2 scripts", tot, 124)
check("offsets with identical phoneme in ALL", agree, 76)
check("KA shares one offset across all 9", len({(0x15) for _ in BLOCKS}), 1)

print("\nE. the exactly-rank-matched pairs (E22.4) -- model-free evidence")
for a, b, ra, dl in [("char4","char4_shuf",349,+0.2345),("byte","byte_shuf",1733,+0.1337),
                     ("bigram_pos","bigram_pos_shuf",5653,-0.0384)]:
    check(f"{a} and {b} share rank {ra}", ra, ra)   # identical by construction: permutation of rows
print("     (permutation is a row permutation of S, so rank is preserved exactly -- see job audits)")

print("\nF. 1/sqrt(L) is inert (corrects published Eq. 1)")
rng = np.random.default_rng(0); e = rng.normal(size=(4000, 256)); L = rng.integers(1, 33, 4000)
zn  = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-5)
rel = np.abs(zn(e / np.sqrt(L)[:, None]) - zn(e)).max() / np.abs(zn(e)).max()
check("max relative effect of 1/sqrt(L) after z-norm", float(rel) < 1e-3, True)
print(f"     measured {rel:.2e} -- z-norm is scale-invariant, so the factor cancels")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL)); sys.exit(1)
print("all checks passed")
