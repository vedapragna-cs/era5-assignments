"""E17 driver. Primary experiment is a SCALING FAMILY: one vocabulary distribution
(real word types from 12 IndicCorpV2 prefixes, real frequencies) sampled at
V = 50,257 / 131,072 / 1,000,000 -- so the V-dependence is measured, not extrapolated.
Anchors: BrahmicTokenizer-131K (the artifact) and GPT-2-50,257 (the instructor's demo point)."""
import numpy as np, json, time, sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e17_rank as M

W = np.load("e17_wordtypes.npz", allow_pickle=True)
words, wfreq = W["words"], W["freq"]
print(f"word-type pool: {len(words):,} types, {wfreq.sum():,} tokens")

VOCABS = []
for V in [50257, 131072, 1000000]:
    VOCABS.append((f"words{V//1000}K", [str(w) for w in words[:V]], wfreq[:V].astype(np.float64)))

# anchor 1: the artifact, with frequencies measured by tokenising a 12-language slice
try:
    from tokenizers import Tokenizer
    p = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--theschoolofai--BrahmicTokenizer-131K/snapshots/*/tokenizer.json"))[0]
    tk = Tokenizer.from_file(p)
    vd = tk.get_vocab(); toks = [None]*len(vd)
    for s,i in vd.items(): toks[i] = s
    import gzip
    cnt = np.zeros(len(toks), np.float64)
    for lg in ["en","hi","bn","ta","te","kn","ml","mr","gu","pa","or","as"]:
        f = f"corpus/{lg}.txt.gz"
        if not os.path.exists(f): continue
        buf=[]
        with gzip.open(f,"rt",encoding="utf-8",errors="replace") as fh:
            for n,line in enumerate(fh):
                buf.append(line)
                if n>=20000: break
        for e in tk.encode_batch(buf):
            np.add.at(cnt, np.asarray(e.ids), 1.0)
    print(f"BrahmicTokenizer freq: {int(cnt.sum()):,} tokens, {(cnt>0).sum():,}/{len(toks)} types seen")
    # decode byte-level tokens back to their surface string
    dec = [tk.decode([i]) if cnt[i]>0 else toks[i] for i in range(len(toks))]
    VOCABS.append(("brahmic131K", [d if d else toks[i] for i,d in enumerate(dec)], np.maximum(cnt,1e-9)))
except Exception as e:
    print("BrahmicTokenizer anchor skipped:", e)

# anchor 2: GPT-2, the instructor's demo vocabulary
try:
    from tokenizers import Tokenizer
    gp = Tokenizer.from_pretrained("gpt2")
    gv = gp.get_vocab(); gt=[None]*len(gv)
    for s,i in gv.items(): gt[i]=s
    import gzip
    gc = np.zeros(len(gt), np.float64); buf=[]
    with gzip.open("corpus/en.txt.gz","rt",encoding="utf-8",errors="replace") as fh:
        for n,line in enumerate(fh):
            buf.append(line)
            if n>=100000: break
    for e in gp.encode_batch(buf): np.add.at(gc, np.asarray(e.ids), 1.0)
    dec=[gp.decode([i]) for i in range(len(gt))]
    print(f"GPT-2 anchor: {len(gt)} tokens, {int(gc.sum()):,} corpus tokens, {(gc>0).sum():,} types seen")
    VOCABS.append(("gpt2_50257",[d if d else gt[i] for i,d in enumerate(dec)], np.maximum(gc,1e-9)))
except Exception as e:
    print("GPT-2 anchor UNAVAILABLE:", repr(e)[:120])

CODECS = [("byte",32,None),("byte",64,None),("char",32,None),("phon",32,None),
          ("factored",None,2),("factored",None,3)]
res=[]
for vname, toks, freq in VOCABS:
    for cname,P,nf in CODECS:
        t0=time.time()
        S = M.build_S(toks, cname, P=P or 32, nfac=nf)
        Dall = S.shape[1]
        S, Dr = M.compact(S)
        if Dr > 26000:
            print(f"  {vname:<12}{cname}{P or nf}: D_reach={Dr} too large for dense rank, skipped"); continue
        rk,_ = M.rank_of(S)
        ncol, ndist, worst = M.collisions(S)
        floor,H,ok,nit = M.unigram_floor(S, freq)
        r = dict(vocab=vname, V=len(toks), codec=f"{cname}{P if P else ''}{'x'+str(nf) if nf else ''}",
                 D_alloc=int(Dall), D_reach=int(Dr), rank=int(rk),
                 collided=int(ncol), distinct=int(ndist), worst_bucket=int(worst),
                 floor_nats=floor, H_nats=float(H), conv=ok, nit=nit, secs=round(time.time()-t0,1))
        res.append(r); print("  "+json.dumps(r))
        json.dump(res, open("e17_results.json","w"), indent=1)
print("DONE")
