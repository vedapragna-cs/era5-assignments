"""E17d -- the collision cost, which unlike the unigram floor is a HARD lower bound.

If two tokens have identical rows in S then for EVERY context they receive identical
logits, so any model with this head must assign them equal probability.  The minimum
achievable loss therefore exceeds the true entropy by the within-group KL:

    cost = sum_G sum_{v in G} p_v * log( p_v / (p_G/|G|) )

exactly, with no optimisation and no model.  This is the number that answers the
assignment's claim of "a vocab of 1M as well without any issues".
"""
import numpy as np, sys, os, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import e17_rank as M
W=np.load("e17_wordtypes.npz",allow_pickle=True); words,wf=W["words"],W["freq"]

def collision_cost(S, freq):
    Sc=S.tocsr()
    keys=np.array([hash(Sc.indices[Sc.indptr[i]:Sc.indptr[i+1]].tobytes())
                   for i in range(Sc.shape[0])],dtype=np.int64)
    p=freq/freq.sum()
    order=np.argsort(keys,kind="stable"); ks=keys[order]
    bnd=np.flatnonzero(np.r_[True, ks[1:]!=ks[:-1], True])
    cost=0.0; ngrp=0; ntok=0
    for a,b in zip(bnd[:-1],bnd[1:]):
        if b-a<2: continue
        idx=order[a:b]; pv=p[idx]; pg=pv.sum()
        if pg<=0: continue
        ngrp+=1; ntok+=len(idx)
        nz=pv>0
        cost+=float((pv[nz]*np.log(pv[nz]/(pg/len(idx)))).sum())
    return cost, ngrp, ntok

res=[]
for V in [50257,131072,1000000]:
    toks=[str(w) for w in words[:V]]; fr=wf[:V].astype(np.float64)
    bpt=float((fr*np.array([len(t.encode('utf-8','replace')) for t in toks])).sum()/fr.sum())
    for cname,P in [("byte",32),("byte",64),("bigram",32),("phon",32)]:
        S=M.build_S(toks,cname,P=P); S,_=M.compact(S)
        c,ng,nt=collision_cost(S,fr)
        r=dict(V=V,codec=f"{cname}{P}",groups=ng,tokens_in_groups=nt,
               pct_vocab=100*nt/V,cost_nats=c,cost_bpb=c/(np.log(2)*bpt))
        res.append(r); print(json.dumps(r))
        json.dump(res,open("e17d_collision_cost.json","w"),indent=1)
print("DONE")
