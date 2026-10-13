import gzip, unicodedata, json, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("theschoolofai/BrahmicTokenizer-131K")
LANGS=["en","hi","mr","bn","te","ta","ml","kn","gu","or","pa","as"]
NLINES=40000

def words(s):
    out=[]
    for w in s.split():
        while w and unicodedata.category(w[0])[0] in "PS": w=w[1:]
        while w and unicodedata.category(w[-1])[0] in "PS": w=w[:-1]
        if w: out.append(w)
    return out

rows=[]
for L in LANGS:
    nb=nt=nw=nc=0; mb_chars=0
    with gzip.open(f"corpus/{L}.txt.gz","rt",encoding="utf-8",errors="ignore") as f:
        for i,line in enumerate(f):
            if i>=NLINES: break
            line=line.strip()
            if not line: continue
            W=words(line)
            if not W: continue
            nw+=len(W)
            for w in W:
                b=w.encode(); nb+=len(b); nc+=len(w)
                mb_chars+=sum(1 for ch in w if len(ch.encode())>1)
                nt+=len(tok.encode(" "+w))
    rows.append(dict(lang=L, words=nw, bytes_per_word=nb/nw, tokens_per_word=nt/nw,
                     bytes_per_token=nb/nt, chars_per_word=nc/nw,
                     pct_multibyte_chars=100*mb_chars/nc))

en=[r for r in rows if r["lang"]=="en"][0]
print(f"{'lang':<5}{'words':>9}{'B/word':>8}{'tok/word':>9}{'B/token':>9}{'ch/word':>8}{'%mbyte':>8}"
      f"{'denseR':>8}{'byteR':>8}{'amplif':>8}")
for r in rows:
    dR=r["tokens_per_word"]/en["tokens_per_word"]
    bR=r["bytes_per_word"]/en["bytes_per_word"]
    r.update(dense_ratio=dR, byte_ratio=bR, amplification=bR/dR)
    print(f"{r['lang']:<5}{r['words']:>9,}{r['bytes_per_word']:>8.2f}{r['tokens_per_word']:>9.3f}"
          f"{r['bytes_per_token']:>9.2f}{r['chars_per_word']:>8.2f}{r['pct_multibyte_chars']:>8.1f}"
          f"{dR:>8.2f}{bR:>8.2f}{r['amplification']:>8.2f}")
json.dump(rows, open("bytecost_results.json","w"), indent=1)

print("\nP(token decodes exactly) with independent per-byte accuracy p:")
print(f"{'lang':<5}{'B/token':>9}" + "".join(f"{'p='+str(p):>10}" for p in (0.99,0.97,0.95,0.90)))
for r in rows:
    bt=r["bytes_per_token"]
    print(f"{r['lang']:<5}{bt:>9.2f}" + "".join(f"{p**bt:>10.3f}" for p in (0.99,0.97,0.95,0.90)))
