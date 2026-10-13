"""E17c -- corrected final sweep.
Fixes vs E17: real-tokenizer anchors now use ONE consistent surface representation
(tk.decode([i]) for every id, never a mix); collisions are split into
  irreducible (tokens sharing an identical string -- no surface codec can separate them)
  and codec-induced (the codec's own fault).
Adds: bigram codec (matched to byte64 at D=16384), factored-with-shuffled-ids."""
import numpy as np, sys, os, json, glob, gzip, time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import e17_rank as M
from tokenizers import Tokenizer

def samestring_dups(toks):
    seen={}
    for t in toks: seen[t]=seen.get(t,0)+1
    return sum(c-1 for c in seen.values() if c>1)

VOCABS=[]
W=np.load("e17_wordtypes.npz",allow_pickle=True); words,wf=W["words"],W["freq"]
for V in [50257,131072,1000000]:
    VOCABS.append((f"words{V//1000}K",[str(w) for w in words[:V]],wf[:V].astype(np.float64)))

def tokfreq(tk, langs, nlines):
    n=tk.get_vocab_size(); cnt=np.zeros(n,np.float64)
    for lg in langs:
        f=f"corpus/{lg}.txt.gz"
        if not os.path.exists(f): continue
        buf=[]
        with gzip.open(f,"rt",encoding="utf-8",errors="replace") as fh:
            for i,line in enumerate(fh):
                buf.append(line)
                if i>=nlines: break
        for e in tk.encode_batch(buf): np.add.at(cnt,np.asarray(e.ids),1.0)
    return cnt

p=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--theschoolofai--BrahmicTokenizer-131K/snapshots/*/tokenizer.json"))[0]
tk=Tokenizer.from_file(p)
bt=[tk.decode([i]) for i in range(tk.get_vocab_size())]          # ONE representation, always
bf=np.maximum(tokfreq(tk,["en","hi","bn","ta","te","kn","ml","mr","gu","pa","or","as"],20000),1e-9)
VOCABS.append(("brahmic131K",bt,bf))
gp=Tokenizer.from_pretrained("gpt2")
gt=[gp.decode([i]) for i in range(gp.get_vocab_size())]
gf=np.maximum(tokfreq(gp,["en"],100000),1e-9)
VOCABS.append(("gpt2_50257",gt,gf))

CODECS=[("byte",32,None),("byte",64,None),("bigram",32,None),("char",32,None),
        ("phon",32,None),("factored",None,2)]
rng=np.random.default_rng(7); res=[]
for vname,toks,freq in VOCABS:
    irr=samestring_dups(toks)
    bpt=float((freq*np.array([len(t.encode('utf-8','replace')) for t in toks])).sum()/freq.sum())
    print(f"\n=== {vname}  V={len(toks)}  irreducible(same-string)={irr}  bytes/token={bpt:.3f}")
    for cname,P,nf in CODECS:
        t0=time.time()
        S=M.build_S(toks,cname,P=P or 32,nfac=nf); Dall=S.shape[1]; S,Dr=M.compact(S)
        if Dr>26000: print(f"  {cname}: D_reach={Dr} skipped"); continue
        rk,_=M.rank_of(S); tot,ndist,worst=M.collisions(S)
        floor,H,ok,nit=M.unigram_floor(S,freq,maxiter=1200)
        row=dict(vocab=vname,V=len(toks),codec=f"{cname}{P if P else ''}{'x'+str(nf) if nf else ''}",
                 D_alloc=int(Dall),D_reach=int(Dr),rank=int(rk),
                 collide_total=int(tot),irreducible=int(irr),codec_induced=int(tot-irr),
                 floor_nats=floor,H_nats=float(H),bytes_per_tok=bpt,
                 floor_bpb=floor/(np.log(2)*bpt),H_bpb=H/(np.log(2)*bpt),secs=round(time.time()-t0,1))
        if cname=="factored":
            fs,_,_,_=M.unigram_floor(S,freq[rng.permutation(len(toks))],maxiter=1200)
            row["floor_nats_shuffled_ids"]=fs
        res.append(row); print("  "+json.dumps({k:v for k,v in row.items() if k not in('H_nats','secs')}))
        json.dump(res,open("e17c_results.json","w"),indent=1)
print("\nDONE")
