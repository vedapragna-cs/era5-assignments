"""E24 -- quantify REDUNDANCY, the variable E19's mechanism rests on.

E19 proposed: a conjunction family that is redundant given the byte family adds no information
but does add TIES, so permuting it helps.  That was inferred from three points.  Redundancy was
never measured.  It is measurable without training:

    R_F      = rank of family F on its own
    dR_F     = rank(byte + F) - rank(byte)      how much F adds ON TOP of the byte family
    redundancy = 1 - dR_F / R_F                  1.0 = F adds nothing new; 0.0 = fully independent

Families are ordered by how far the conjunction reaches, which should trade redundancy against
how much of the string it can see.
"""
import numpy as np, json, os
D_=os.path.join(os.path.dirname(os.path.abspath(__file__)))
C=np.load("codec_local.npz"); M=json.load(open("meta_local.json"))
NV,P=M["vocab"],M["P"]; bv=C["byte_idx"].astype(np.int64)
freq=np.load("train_freq.npy") if os.path.exists("train_freq.npy") else np.bincount(np.load("train_local.npy").ravel(),minlength=NV)
seen=np.where(freq>0)[0]; AR=np.arange(P)[None,:]
NBIG=8192
H=lambda a,b,extra: ((a*257+b*1013904223+extra)&0x7FFFFFFF)%NBIG

def bytefam():
    return np.where(bv>=0, bv*P+AR, -1)

def skipfam(gap, posbound):
    """conjunction of bytes (i, i+gap); position-bound or not"""
    g=np.full((NV,P-gap),-1,np.int64)
    for i in range(P-gap):
        a,b=bv[:,i],bv[:,i+gap]; m=(a>=0)&(b>=0)
        g[m,i]=H(a[m],b[m], i*2654435761 if posbound else 0)
    return g

def tokhash():
    """maximally independent: hash the whole byte string -> near token identity"""
    out=np.full((NV,1),-1,np.int64)
    for v in range(NV):
        b=bv[v]; b=b[b>=0]
        if len(b)==0: continue
        h=1469598103934665603
        for x in b: h=((h^int(x))*1099511628211)&0xFFFFFFFFFFFFFFFF
        out[v,0]=h%NBIG
    return out

def rank_of(cols, ncol):
    sub=cols[seen]
    used=np.unique(sub[sub>=0]); Dr=len(used)
    if Dr==0: return 0,0
    rm=np.full(int(used.max())+2,-1,np.int64); rm[used]=np.arange(Dr)
    A=np.zeros((len(seen),Dr),np.float32)
    rr=np.repeat(np.arange(len(seen)),sub.shape[1]); cc=sub.ravel()
    k=cc>=0
    np.add.at(A,(rr[k],rm[cc[k]]),1.0)
    ev=np.linalg.eigvalsh((A.T@A).astype(np.float64))
    return int((ev>ev.max()*1e-9).sum()), Dr

bf=bytefam(); r_byte,_=rank_of(bf,256*P)
print(f"byte family alone: rank {r_byte}\n")
print(f"{'family':<22}{'rank(F)':>9}{'rank(byte+F)':>14}{'dR':>7}{'redundancy':>12}")
rows=[]
FAMS=[("bigram gap1 pos-bound",skipfam(1,True)),("bigram gap1 pos-free",skipfam(1,False)),
      ("skip gap2 pos-bound",skipfam(2,True)),("skip gap2 pos-free",skipfam(2,False)),
      ("skip gap4 pos-bound",skipfam(4,True)),("skip gap8 pos-bound",skipfam(8,True)),
      ("whole-token hash",tokhash())]
for nm,g in FAMS:
    gg=np.where(g>=0, g+256*P, -1)
    rF,_=rank_of(gg,NBIG)
    both=np.concatenate([bf,gg],1)
    rB,_=rank_of(both,256*P+NBIG)
    dR=rB-r_byte
    red=1-dR/max(rF,1)
    rows.append(dict(family=nm,rank_F=rF,rank_byte_plus_F=rB,dR=int(dR),redundancy=float(red)))
    print(f"{nm:<22}{rF:>9}{rB:>14}{dR:>7}{red:>12.4f}")
json.dump(rows,open("e24_redundancy.json","w"),indent=2)
print("\nMEASURED delta_perm, for the two families we have trained:")
print("   bigram gap1 pos-bound : -0.0383 nats   (permuting it HELPS)")
print("   bigram gap1 pos-free  : +0.0008 nats   (permuting it is neutral)")
print("   byte family itself    : +0.1337 nats   (permuting it HURTS)")
