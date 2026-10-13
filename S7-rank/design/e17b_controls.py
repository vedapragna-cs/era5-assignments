"""E17b -- two controls the main sweep needs before any number can be quoted.

C1. SAME-STRING FLOOR.  Two tokens with an identical surface string cannot be told apart
    by ANY function of surface form.  Those collisions are a property of the tokenizer,
    not of the codec, and must be subtracted before a codec is blamed for them.

C2. INDEX-ORDER CONFOUND.  An index-factored head keys on the integer id.  If ids are
    sorted by frequency it gets the unigram distribution nearly free.  Shuffle the ids
    and re-measure -- if the floor moves, the factored advantage was the ordering."""
import numpy as np, sys, os, json, glob, gzip
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import e17_rank as M
from tokenizers import Tokenizer

out={}
# ---------------- C1 -------------------------------------------------------
def samestring(toks):
    seen={}
    for t in toks: seen[t]=seen.get(t,0)+1
    dup=sum(c-1 for c in seen.values() if c>1)
    empt=sum(1 for t in toks if t=="")
    return dup, empt, len(seen)

W=np.load("e17_wordtypes.npz",allow_pickle=True); words,wf=W["words"],W["freq"]
print("C1 -- tokens sharing an identical surface string (irreducible for any surface codec)")
for V in [50257,131072,1000000]:
    d,e,n=samestring([str(w) for w in words[:V]])
    print(f"  words{V//1000}K   V={V:>8}  same-string dups={d:>7}  empty={e:>5}  distinct={n:>8}")
    out[f"words{V//1000}K"]=d

p=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--theschoolofai--BrahmicTokenizer-131K/snapshots/*/tokenizer.json"))[0]
tk=Tokenizer.from_file(p); n=tk.get_vocab_size()
bt=[tk.decode([i]) for i in range(n)]
d,e,nd=samestring(bt)
print(f"  brahmic131K V={n:>8}  same-string dups={d:>7}  empty={e:>5}  distinct={nd:>8}")
out["brahmic131K"]=d
gp=Tokenizer.from_pretrained("gpt2"); n2=gp.get_vocab_size()
gt=[gp.decode([i]) for i in range(n2)]
d2,e2,nd2=samestring(gt)
print(f"  gpt2_50257  V={n2:>8}  same-string dups={d2:>7}  empty={e2:>5}  distinct={nd2:>8}")
out["gpt2_50257"]=d2
json.dump(out,open("e17b_samestring.json","w"),indent=1)

# ---------------- C2 -------------------------------------------------------
print("\nC2 -- index-factored head: frequency-sorted ids vs shuffled ids")
rng=np.random.default_rng(7)
for V in [50257,131072,1000000]:
    fr=wf[:V].astype(np.float64)
    for nf in [2,3]:
        toks=[str(w) for w in words[:V]]
        S=M.build_S(toks,"factored",nfac=nf); S,_=M.compact(S)
        f_sorted,H,_,_=M.unigram_floor(S,fr)
        perm=rng.permutation(V)                 # same code matrix, ids reassigned at random
        f_shuf,_,_,_=M.unigram_floor(S,fr[perm])
        print(f"  V={V:>8} factored x{nf}:  freq-sorted ids {f_sorted:.4f}   "
              f"shuffled ids {f_shuf:.4f}   penalty {f_shuf-f_sorted:+.4f} nats  (H={H:.4f})")
