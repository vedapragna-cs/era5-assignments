"""E17 stage 1 — build the three vocabularies the rank predictor needs.
Each is (list[str] tokens, np.ndarray unigram counts). Provenance printed for every one."""
import gzip, json, collections, os, sys, numpy as np

OUT = "e17_vocabs.npz"
V_TARGETS = [50257, 131072, 1000000]

# ---- 1. BrahmicTokenizer-131K : the artifact, exact -------------------------
from tokenizers import Tokenizer
import glob
p = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--theschoolofai--BrahmicTokenizer-131K/snapshots/*/tokenizer.json"))
assert p, "BrahmicTokenizer not in cache"
tok = Tokenizer.from_file(p[0])
vocab = tok.get_vocab()
brahmic = [None]*len(vocab)
for s,i in vocab.items(): brahmic[i]=s
print(f"BrahmicTokenizer-131K: {len(brahmic)} tokens   (source: {p[0].split('snapshots/')[1][:12]}...)")

# ---- 2. word-type vocabulary from the 12 IndicCorpV2 prefixes ---------------
LANGS = ["en","hi","bn","ta","te","kn","ml","mr","gu","pa","or","as"]
MAXL  = int(sys.argv[1]) if len(sys.argv)>1 else 400_000     # lines per language
cnt = collections.Counter()
for lg in LANGS:
    f=f"corpus/{lg}.txt.gz"
    if not os.path.exists(f): print(f"  MISSING {f}"); continue
    n=0
    with gzip.open(f,"rt",encoding="utf-8",errors="replace") as fh:
        for line in fh:
            cnt.update(line.split())
            n+=1
            if n>=MAXL: break
    print(f"  {lg}: {n} lines -> running distinct types {len(cnt):,}")
print(f"word-type pool: {len(cnt):,} distinct types, {sum(cnt.values()):,} tokens")

items = cnt.most_common()
np.savez_compressed("e17_wordtypes.npz",
                    words=np.array([w for w,_ in items],dtype=object),
                    freq=np.array([c for _,c in items],dtype=np.int64))
json.dump({"langs":LANGS,"max_lines_per_lang":MAXL,"distinct":len(cnt),
           "total_tokens":int(sum(cnt.values()))}, open("e17_wordtypes_meta.json","w"), indent=1)
print("saved e17_wordtypes.npz")
