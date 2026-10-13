import numpy as np, json
C=np.load("codec_local.npz"); H=np.load("hash_local.npz")
tr=np.load("train_local.npy"); M=json.load(open("meta_local.json"))
NV,P,NB=M["vocab"],M["P"],M["NB"]
byte=C["byte_idx"]; blk,off=C["blk_idx"],C["off_idx"]; AR=np.arange(P)[None,:]
def build(fs):
    cols,base=[],0
    for m,card in fs: cols.append(np.where(m>=0,base+m*P+AR,-1)); base+=card*P
    idx=np.concatenate(cols,1); return np.where(idx>=0,idx,base).astype(np.int64),base
S={"byte":build([(byte,256)]),
   "char2":build([(blk,NB),(off,128)]),
   "char3":build([(blk,NB),(np.where(off>=0,off>>3,-1),16),(np.where(off>=0,off&7,-1),8)]),
   "char4":build([(blk,NB),(np.where(off>=0,off>>5,-1),4),(np.where(off>=0,(off>>2)&7,-1),8),(np.where(off>=0,off&3,-1),4)])}
i4,D4=S["char4"]; a=i4.copy(); lv=a!=D4
a[lv]=np.random.default_rng(12345).integers(0,D4,size=lv.sum()); S["rand4"]=(a,D4)
for nm,k in [("hashbyte4","hashbyte"),("hashchar4","hashchar")]: S[nm]=(H[k].astype(np.int64),2560)
freq=np.bincount(tr.ravel(),minlength=NV).astype(np.float64); seen=freq>0
sidx=np.where(seen)[0]; fs=freq[seen]
VAL={"byte":6.1521,"char2":6.1768,"char3":6.2971,"char4":6.3232,"rand4":6.0328,
     "hashbyte4":6.1426,"hashchar4":6.1220}
LIVE={"byte":1855,"char2":1305,"char3":547,"char4":479,"rand4":2560,"hashbyte4":2415,"hashchar4":2520}
out={}
print(f"{'arm':<11}{'D':>6}{'reach':>7}{'rank':>7}{'effrank':>9}{'val loss':>10}")
for name in ["byte","char2","char3","char4","rand4","hashbyte4","hashchar4"]:
    idx,D=S[name]; sub=idx[sidx]
    used=np.unique(sub[sub!=D]); Dr=len(used)
    rm=np.full(D+1,-1,np.int64); rm[used]=np.arange(Dr)
    A=np.zeros((len(sidx),Dr),np.float32)
    rr=np.repeat(np.arange(len(sidx)),sub.shape[1]); cc=rm[sub.ravel()]; k=cc>=0
    np.add.at(A,(rr[k],cc[k]),1.0)
    ev=np.linalg.eigvalsh((A.T@A).astype(np.float64))
    Aw=A*np.sqrt(fs).astype(np.float32)[:,None]
    evw=np.linalg.eigvalsh((Aw.T@Aw).astype(np.float64))
    r0=int((ev>ev.max()*1e-9).sum())
    p=evw[evw>evw.max()*1e-12]; p=p/p.sum(); eff=float(np.exp(-(p*np.log(p)).sum()))
    out[name]=(D,Dr,r0,eff)
    print(f"{name:<11}{D:>6}{Dr:>7}{r0:>7}{eff:>9.1f}{VAL[name]:>10.4f}")
def spear(x,y):
    r=lambda v:(lambda o:(lambda q:(q.__setitem__(slice(None),0),q)[1])(np.empty(len(v))))(None)
    def rk(v):
        o=np.argsort(v,kind="stable"); q=np.empty(len(v)); q[o]=np.arange(len(v)); return q
    return np.corrcoef(rk(np.array(x)),rk(np.array(y)))[0,1]
names=list(out)
v=[VAL[n] for n in names]
print()
for lbl,vals in [("live rows",[LIVE[n] for n in names]),("rank(S)",[out[n][2] for n in names]),
                 ("eff_rank(freq-wtd)",[out[n][3] for n in names]),
                 ("D_reachable",[out[n][1] for n in names]),("D_alloc",[out[n][0] for n in names])]:
    print(f"  Spearman({lbl:<20}, val loss) = {spear(vals,v):+.4f}")
json.dump(out,open("rankall.json","w"),indent=1)
