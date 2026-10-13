"""E17e -- real-tokenizer anchors, done on RAW BYTES.

Three successive representations were wrong and each was caught by a check:
  1. mixed (decoded for seen ids, byte-level string for unseen)  -> inconsistent
  2. tk.decode([i]) for all ids                                  -> LOSSY: 658 ids decode to
     '�' and 396 to ' �' because they are partial UTF-8 fragments.  Those are
     distinct byte strings that a byte codec separates perfectly; the collapse was my
     decoder's, not the codec's.
  3. raw bytes via the ByteLevel inverse alphabet                -> correct, used here.

A codec keyed on bytes must be measured on bytes."""
import numpy as np, scipy.sparse as sp, json, os, glob, gzip, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import e17_rank as M
from tokenizers import Tokenizer

def bytes_to_unicode():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1))
    cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return dict(zip([chr(c) for c in cs], bs))          # unicode char -> byte
INV=bytes_to_unicode()

def raw_bytes(key):
    try: return bytes(INV[c] for c in key)
    except KeyError: return key.encode("utf-8","replace")

def cells_byte_b(b, P):   return [int(v)*P+i for i,v in enumerate(b[:P])], 256*P
def cells_bigram_b(b, P, NBIG=8192):
    bb=b[:P]; out=[int(v)*P+i for i,v in enumerate(bb)]
    for i in range(len(bb)-1):
        out.append(256*P + ((int(bb[i])*257+int(bb[i+1])*1013904223)&0x7FFFFFFF)%NBIG)
    return out, 256*P+NBIG
def cells_char_b(b,P):
    s=b.decode("utf-8","replace"); return M.cells_char(s,P)
def cells_phon_b(b,P):
    s=b.decode("utf-8","replace"); return M.cells_phoneme(s,P)
CODECS={"byte32":(cells_byte_b,32),"byte64":(cells_byte_b,64),
        "bigram32":(cells_bigram_b,32),"char32":(cells_char_b,32),"phon32":(cells_phon_b,32)}

def tokfreq(tk,langs,n):
    c=np.zeros(tk.get_vocab_size())
    for lg in langs:
        f=f"corpus/{lg}.txt.gz"
        if not os.path.exists(f): continue
        buf=[]
        with gzip.open(f,"rt",encoding="utf-8",errors="replace") as fh:
            for i,l in enumerate(fh):
                buf.append(l)
                if i>=n: break
        for e in tk.encode_batch(buf): np.add.at(c,np.asarray(e.ids),1.0)
    return c

ANCH=[]
p=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--theschoolofai--BrahmicTokenizer-131K/snapshots/*/tokenizer.json"))[0]
tk=Tokenizer.from_file(p); vd=tk.get_vocab(); keys=[None]*len(vd)
for s,i in vd.items(): keys[i]=s
ANCH.append(("brahmic131K",[raw_bytes(k) for k in keys],
             tokfreq(tk,["en","hi","bn","ta","te","kn","ml","mr","gu","pa","or","as"],20000)))
gp=Tokenizer.from_pretrained("gpt2"); gv=gp.get_vocab(); gk=[None]*len(gv)
for s,i in gv.items(): gk[i]=s
ANCH.append(("gpt2_50257",[raw_bytes(k) for k in gk],tokfreq(gp,["en"],100000)))

res=[]
for name,toks,freq in ANCH:
    seen={}
    for t in toks: seen[t]=seen.get(t,0)+1
    irr=sum(c-1 for c in seen.values() if c>1)
    f=np.maximum(freq,1e-9)
    bpt=float((f*np.array([len(t) for t in toks])).sum()/f.sum())
    print(f"\n=== {name}  V={len(toks)}  same-BYTE-STRING dups={irr}  bytes/token={bpt:.3f}  "
          f"observed={int((freq>0).sum()):,}")
    for cn,(fn,P) in CODECS.items():
        rows,cols,D=[],[],None
        for i,t in enumerate(toks):
            c,D=fn(t,P); c=sorted(set(c)); rows.extend([i]*len(c)); cols.extend(c)
        S=sp.csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(toks),D))
        S,Dr=M.compact(S)
        if Dr>26000: print(f"  {cn}: D_reach={Dr} skipped"); continue
        rk,_=M.rank_of(S); tot,nd,worst=M.collisions(S)
        fl,H,ok,nit=M.unigram_floor(S,f,maxiter=1200)
        r=dict(vocab=name,V=len(toks),codec=cn,D_alloc=int(D),D_reach=int(Dr),rank=int(rk),
               collide_total=int(tot),irreducible=int(irr),codec_induced=int(tot-irr),
               floor_nats=fl,H_nats=float(H),bytes_per_tok=bpt,
               floor_bpb=fl/(np.log(2)*bpt),H_bpb=H/(np.log(2)*bpt))
        res.append(r); print("  "+json.dumps({k:v for k,v in r.items() if k!='H_nats'}))
        json.dump(res,open("e17e_anchors.json","w"),indent=1)
print("\nDONE")
