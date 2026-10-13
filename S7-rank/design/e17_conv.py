"""Is the unigram floor converged? All surface codecs reported conv=False at maxiter=400.
An unconverged minimisation OVERSTATES the floor, so this bounds the error in the right
direction -- but the size of the error has to be measured, not assumed."""
import numpy as np, sys, os, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import e17_rank as M
W=np.load("e17_wordtypes.npz",allow_pickle=True); words,wf=W["words"],W["freq"]
for V,cod,P in [(50257,"byte",32),(131072,"byte",32)]:
    toks=[str(w) for w in words[:V]]; fr=wf[:V].astype(np.float64)
    S=M.build_S(toks,cod,P=P); S,Dr=M.compact(S)
    print(f"\nV={V} {cod}{P}  D_reach={Dr}")
    prev=None
    for it in [400,1200,4000,12000]:
        f,H,ok,nit=M.unigram_floor(S,fr,maxiter=it)
        d=f"  delta vs prev {f-prev:+.5f}" if prev is not None else ""
        print(f"  maxiter={it:>6}  nit={nit:>6}  floor={f:.5f} nats  conv={ok}{d}")
        prev=f
        if ok: print("   -> converged, stop"); break
