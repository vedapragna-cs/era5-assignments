import unicodedata, pathlib, math, json
import pandas as pd
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("theschoolofai/BrahmicTokenizer-131K")
LANGS=["eng_Latn","hin_Deva","mar_Deva","ben_Beng","tel_Telu","tam_Taml",
       "mal_Mlym","kan_Knda","guj_Gujr","ory_Orya","pan_Guru","asm_Beng"]

def words(s):
    out=[]
    for w in s.split():
        w=w.strip("".join(c for c in w if unicodedata.category(c)[0] in "PSZ" ))
        w2=w
        while w2 and unicodedata.category(w2[0])[0] in "PS": w2=w2[1:]
        while w2 and unicodedata.category(w2[-1])[0] in "PS": w2=w2[:-1]
        if w2: out.append(w2)
    return out

def floor_tokens(w, cap):
    """Minimum tokens for this word under a `cap`-byte codec window.
    First token also carries the leading space (1 byte); breaks must fall on character boundaries."""
    n=0; used=cap-1          # first chunk loses one byte to the leading space
    for ch in w:
        b=len(ch.encode())
        if b>used:
            n+=1; used=cap
        used-=b
    return n+1

rows=[]
for L in LANGS:
    df=pd.read_parquet(f"flores/{L}.parquet")
    col="text" if "text" in df.columns else df.columns[-1]
    sents=[str(x) for x in df[col].tolist()]
    W=[w for s in sents for w in words(s)]
    nb=[len(w.encode()) for w in W]; nc=[len(w) for w in W]
    N=len(W)
    over31=sum(1 for b in nb if b>31)
    fl32=sum(floor_tokens(w,32) for w in W)/N
    fl48=sum(floor_tokens(w,48) for w in W)/N
    fl64=sum(floor_tokens(w,64) for w in W)/N
    actual=sum(len(tok.encode(" "+w)) for w in W)/N
    rows.append(dict(lang=L, words=N,
        chars=sum(nc)/N, bytes=sum(nb)/N,
        pct_over_31B=100*over31/N,
        floor32=fl32, floor48=fl48, floor64=fl64,
        actual=actual, headroom=actual-fl32))

print(f"{'lang':<10}{'words':>7}{'ch/w':>7}{'B/w':>7}{'>31B%':>8}{'floor32':>9}{'floor48':>9}{'floor64':>9}{'actual':>8}{'gap':>7}")
for r in rows:
    print(f"{r['lang']:<10}{r['words']:>7}{r['chars']:>7.2f}{r['bytes']:>7.2f}{r['pct_over_31B']:>8.2f}"
          f"{r['floor32']:>9.4f}{r['floor48']:>9.4f}{r['floor64']:>9.4f}{r['actual']:>8.3f}{r['headroom']:>7.3f}")
json.dump(rows, open("wordlen_results.json","w"), indent=1)
