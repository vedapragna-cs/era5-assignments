import numpy as np, json
C=np.load("codec_local.npz"); tr=np.load("train_local.npy"); M=json.load(open("meta_local.json"))
NV,P,NB=M["vocab"],M["P"],M["NB"]
byte=C["byte_idx"]; blk,off=C["blk_idx"],C["off_idx"]
freq=np.bincount(tr.ravel(),minlength=NV).astype(np.float64)
L=(byte>=0).sum(1).astype(np.float64)
seen=freq>0
lf=np.log10(freq[seen]+1); Ls=L[seen]
def rk(x): 
    o=np.argsort(x,kind="stable"); r=np.empty(len(x)); r[o]=np.arange(len(x)); return r
print(f"types seen={seen.sum()}  mean L={Ls.mean():.3f}  (L = capped byte length = slots/token)")
print(f"Pearson  corr(log10 freq, L) = {np.corrcoef(lf,Ls)[0,1]:+.4f}")
print(f"Spearman corr(log10 freq, L) = {np.corrcoef(rk(lf),rk(Ls))[0,1]:+.4f}")
print("\n  freq bucket        n     mean L    share of slot-uses")
edges=[1,10,100,1000,10**12]; names=["1-9","10-99","100-999",">=1000"]
tot=(freq*L).sum()
for i,nm in enumerate(names):
    m=seen&(freq>=edges[i])&(freq<edges[i+1])
    print(f"  {nm:>10}  {m.sum():7d}   {L[m].mean():6.2f}    {100*(freq[m]*L[m]).sum()/tot:5.1f}%")

AR=np.arange(P)[None,:]
def build(fs):
    cols,base=[],0
    for m,card in fs: cols.append(np.where(m>=0,base+m*P+AR,-1)); base+=card*P
    idx=np.concatenate(cols,1); return np.where(idx>=0,idx,base).astype(np.int64),base
SP={"byte":build([(byte,256)]),"char2":build([(blk,NB),(off,128)])}
sidx=np.where(seen)[0]; fs=freq[seen]
for name,(idx,D) in SP.items():
    sub=idx[sidx]
    used=np.unique(sub[sub!=D]); Dr=len(used)
    remap=np.full(D+1,-1,np.int64); remap[used]=np.arange(Dr)
    A=np.zeros((len(sidx),Dr),np.float32)
    rr=np.repeat(np.arange(len(sidx)),sub.shape[1]); cc=remap[sub.ravel()]
    k=cc>=0; np.add.at(A,(rr[k],cc[k]),1.0)
    G=A.T@A
    Aw=A*np.sqrt(fs).astype(np.float32)[:,None]; Gw=Aw.T@Aw
    ev=np.linalg.eigvalsh(G.astype(np.float64)); evw=np.linalg.eigvalsh(Gw.astype(np.float64))
    r0=int((ev>ev.max()*1e-9).sum()); r1=int((evw>evw.max()*1e-9).sum())
    p=evw[evw>evw.max()*1e-12]; p=p/p.sum(); eff=float(np.exp(-(p*np.log(p)).sum()))
    print(f"\n{name}: D_alloc={D}  D_reachable={Dr}  rank(S)={r0}  eff_rank(freq-wtd)={eff:.1f}")
    print(f"    {seen.sum()} live types / {r0} independent directions = {seen.sum()/r0:.1f} types per DOF")
    print(f"    -> {seen.sum()-r0} of {seen.sum()} token embeddings are forced linear combos of the rest")
