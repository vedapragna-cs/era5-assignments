"""E23 -- is our reimplementation faithful to the published reference?

S7 reimplements the codec from the paper's description (ARCHITECTURE D2).  The reference
implementation is on PyPI as `kronecker-embeddings` (Apache 2.0).  Until now we never checked
ours against it, which is the first thing a reviewer would ask.  This does three things:

 1. numerically compare our codec vector against the reference's, token by token;
 2. locate and quantify our ONE known deviation (where the z-norm is applied);
 3. check whether that deviation invalidates the rank analysis.
"""
import numpy as np, torch, json, glob, os
from kronecker_embeddings import kronecker_codec
from tokenizers import Tokenizer

P, CD = 32, 256; D = CD*P
p = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--theschoolofai--BrahmicTokenizer-131K/snapshots/*/tokenizer.json"))[0]
tk = Tokenizer.from_file(p)
def bytes_to_unicode():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1))
    cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return dict(zip([chr(c) for c in cs], bs))
INV=bytes_to_unicode()
vd=tk.get_vocab(); keys=[None]*len(vd)
for s,i in vd.items(): keys[i]=s
rng=np.random.default_rng(0); sample=rng.choice(len(keys), 3000, replace=False)
seqs=np.zeros((len(sample),P),np.int64); lens=np.zeros(len(sample),np.int64)
for r,i in enumerate(sample):
    try: b=bytes(INV[c] for c in keys[i])
    except KeyError: b=keys[i].encode("utf-8","replace")
    b=b[:P]; seqs[r,:len(b)]=list(b); lens[r]=len(b)

ref = kronecker_codec(torch.from_numpy(seqs), torch.from_numpy(lens),
                      char_dim=CD, pos_dim=P, length_normalize=True, z_normalize=True).numpy()

# ---- ours: the same S-matrix indices the whole of S7 is built on --------------
AR=np.arange(P)[None,:]
idx=np.where(AR<lens[:,None], seqs*P+AR, D)           # index = byte*pos_dim + pos  (D = pad row)
ours=np.zeros((len(sample),D+1),np.float64)
np.add.at(ours,(np.repeat(np.arange(len(sample)),P), idx.ravel()),1.0)
ours=ours[:,:D]
ours=ours/np.sqrt(np.maximum(lens,1))[:,None]         # 1/sqrt(L)
# NOTE: torch.std defaults to ddof=1 (sample), numpy.std to ddof=0 (population).  The reference
# and our KronEmb both use torch, so ddof=1 is the convention to match; using numpy's default here
# produced a spurious 5.5e-3 max-abs discrepancy at cosine 0.9999997.
m=ours.mean(1,keepdims=True); sd=ours.std(1,keepdims=True,ddof=1)
ours_zn=(ours-m)/(sd+1e-6)

err=np.abs(ours_zn-ref).max()
cos=(ours_zn*ref).sum(1)/(np.linalg.norm(ours_zn,axis=1)*np.linalg.norm(ref,axis=1)+1e-12)
print(f"1. CODEC FIDELITY over {len(sample)} sampled tokens")
print(f"   max |ours - reference| = {err:.3e}")
print(f"   cosine similarity: min {cos.min():.10f}  mean {cos.mean():.10f}")
print(f"   -> our index convention (byte*pos_dim + pos), 1/sqrt(L) and z-norm match the reference"
      if err < 1e-4 else "   -> MISMATCH")

# ---- 2. the known deviation: where z-norm sits -------------------------------
print(f"\n2. OUR ONE KNOWN DEVIATION -- where the z-norm is applied")
print(f"   reference / paper : z-norm the D={D}-dim codec vector, THEN project with W")
print(f"   S7 training runs  : project with W, THEN z-norm the d_model=256-dim output")
d_model=256
W=np.random.default_rng(1).normal(size=(D,d_model))*0.02
E_ref=ours_zn@W
E_ours_pre=ours@W
E_ours=(E_ours_pre-E_ours_pre.mean(1,keepdims=True))/(E_ours_pre.std(1,keepdims=True,ddof=1)+1e-5)
c2=(E_ref*E_ours).sum(1)/(np.linalg.norm(E_ref,axis=1)*np.linalg.norm(E_ours,axis=1)+1e-12)
print(f"   resulting embeddings differ: cosine min {c2.min():.4f} mean {c2.mean():.4f}")
print(f"   -> the two are NOT the same map; S7 replicates a VARIANT, as ARCHITECTURE D2 states")

# ---- 3. does the deviation invalidate the rank analysis? ---------------------
print(f"\n3. DOES THIS INVALIDATE rank(S)?")
r_raw=np.linalg.matrix_rank(ours)
r_zn=np.linalg.matrix_rank(ours_zn)
print(f"   rank of raw   S (1/sqrt(L) only) : {r_raw}")
print(f"   rank of z-normed S               : {r_zn}   (difference {r_zn-r_raw:+d})")
print(f"   z-norm is x -> (x - mean(x))/std(x): a per-row scale (rank-preserving) plus")
print(f"   subtraction of a rank-1 all-ones component, so rank can move by at most 1.")
json.dump(dict(n=len(sample), max_abs_err=float(err), cos_min=float(cos.min()),
               dev_cos_min=float(c2.min()), dev_cos_mean=float(c2.mean()),
               rank_raw=int(r_raw), rank_zn=int(r_zn)), open("e23_fidelity.json","w"), indent=1)
