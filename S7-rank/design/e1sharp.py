import numpy as np, pickle, unicodedata
raw=pickle.load(open('full_raw.pkl','rb')); scripts=np.load('full_scripts.npy',allow_pickle=True)
SP='Ġ'  # GPT-2 byte-level marker for leading space
def content(i):
    try: s=raw[i].decode('utf-8')
    except UnicodeDecodeError: return None
    return s[1:] if s.startswith(' ') else s
print(f"{'script':<12}{'n':>7}{'bytes/char':>11}{'cap_chars':>10}{'max_chars':>10}{'at_cap?':>9}{'max_bytes':>10}")
rows=[]
for sc in ['LATIN','DEVANAGARI','BENGALI','MALAYALAM','GUJARATI','KANNADA','TELUGU','TAMIL','ORIYA','GURMUKHI']:
    m=np.where(scripts==sc)[0]
    best=0; bestb=0; bpc=[]
    for i in m:
        c=content(i)
        if c is None or len(c)==0: continue
        b=len(c.encode('utf-8'))
        bpc.append(b/len(c))
        if len(c)>best: best=len(c)
        bestb=max(bestb,len(raw[i]))
    modal=round(np.median(bpc))
    cap=(32-1)//modal          # 31 content bytes available when the token carries a leading space
    rows.append((sc,len(m),modal,cap,best,best>=cap,bestb))
    print(f"{sc:<12}{len(m):>7}{modal:>11}{cap:>10}{best:>10}{'YES' if best>=cap else 'no':>9}{bestb:>10}")
print()
lat=[r for r in rows if r[0]=='LATIN'][0]
bra=[r for r in rows if r[0]!='LATIN']
atcap=[r for r in bra if r[5]]
print(f"Latin: cap allows {lat[3]} chars, longest observed {lat[4]} -> cap NON-binding (vocabulary-bound), headroom {lat[3]-lat[4]}")
print(f"Brahmic scripts at their arithmetic ceiling: {len(atcap)}/{len(bra)} -> {[r[0] for r in atcap]}")
print(f"Below ceiling: {[(r[0],r[4],r[3],r[1]) for r in bra if not r[5]]}  (script, max, cap, n_tokens)")
print(f"\nAsymmetry, longest content token: Latin {lat[4]} vs Brahmic-at-cap {atcap[0][4]} = {lat[4]/atcap[0][4]:.2f}x")
