"""E24b -- if not redundancy, what?  E19's arms have IDENTICAL rank (permutation preserves it),
so the difference must be in WHICH directions, not how many.  Candidate: the structured bigram
cells are derived from the same bytes as the byte cells, so the two blocks are CORRELATED and the
frequency-weighted Gram is worse conditioned.  Permuting decorrelates them.

Measure the eigenvalue spectrum of S^T diag(f) S for structured vs permuted, at identical rank.
"""
import numpy as np, json
C=np.load("codec_local.npz"); M=json.load(open("meta_local.json"))
NV,P=M["vocab"],M["P"]; bv=C["byte_idx"].astype(np.int64)
freq=np.load("train_freq.npy").astype(np.float64)
seen=np.where(freq>0)[0]; fs=freq[seen]; AR=np.arange(P)[None,:]; NBIG=8192
bc=np.where(bv>=0,bv*P+AR,-1)
gc=np.full((NV,P-1),-1,np.int64)
for i in range(P-1):
    a,b=bv[:,i],bv[:,i+1]; m=(a>=0)&(b>=0)
    gc[m,i]=256*P+((a[m]*257+b[m]*1013904223+i*2654435761)&0x7FFFFFFF)%NBIG

def shuf(g,seed,ns=400):
    rng=np.random.default_rng(seed); new=g.copy()
    n=(g>=0).sum(1); lf=np.where(freq>0,np.log10(freq+1),-1.0)
    qs=np.quantile(lf[freq>0],np.linspace(0,1,ns+1)); key=np.digitize(lf,qs[1:-1])*1000+n
    for k in np.unique(key):
        ix=np.where(key==k)[0]
        if len(ix)>1: new[ix]=g[rng.permutation(ix)]
    return new

def spec(cols):
    sub=cols[seen]; used=np.unique(sub[sub>=0]); Dr=len(used)
    rm=np.full(int(used.max())+2,-1,np.int64); rm[used]=np.arange(Dr)
    A=np.zeros((len(seen),Dr),np.float32)
    rr=np.repeat(np.arange(len(seen)),sub.shape[1]); cc=sub.ravel(); k=cc>=0
    np.add.at(A,(rr[k],rm[cc[k]]),1.0)
    Aw=A*np.sqrt(fs).astype(np.float32)[:,None]
    ev=np.linalg.eigvalsh((Aw.T@Aw).astype(np.float64))
    ev=ev[ev>ev.max()*1e-12]
    p=ev/ev.sum()
    return dict(Dr=Dr, rank=int((ev>ev.max()*1e-9).sum()),
                eff_rank=float(np.exp(-(p*np.log(p)).sum())),
                cond=float(ev.max()/ev[ev>ev.max()*1e-9].min()),
                top10_share=float(np.sort(p)[::-1][:10].sum()))
print(f"{'arm':<26}{'D_reach':>9}{'rank':>7}{'eff_rank':>11}{'top-10 share':>14}{'cond':>12}")
for nm,g in [("byte only", None),
             ("byte + bigram (structured)", gc),
             ("byte + bigram (permuted d1)", shuf(gc,1)),
             ("byte + bigram (permuted d2)", shuf(gc,2))]:
    cols = bc if g is None else np.concatenate([bc,g],1)
    r=spec(cols)
    print(f"{nm:<26}{r['Dr']:>9}{r['rank']:>7}{r['eff_rank']:>11.1f}{100*r['top10_share']:>13.1f}%{r['cond']:>12.3e}")
