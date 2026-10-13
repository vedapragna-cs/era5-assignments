import glob, numpy as np, pyarrow.parquet as pq
FILES=sorted(glob.glob('flores/*.parquet'))
W={}
for f in FILES:
    lang=f.split('/')[-1].split('.')[0]
    t=pq.read_table(f); col='sentence' if 'sentence' in t.column_names else t.column_names[-1]
    words=[]
    for s in t.column(col).to_pylist():
        if s: words.extend(s.split())
    W[lang]=(np.array([len(w.encode('utf-8')) for w in words]),
             np.array([len(w) for w in words]))
LANGS=['eng_Latn','hin_Deva','mal_Mlym','tam_Taml','kan_Knda']
print("FERTILITY FLOOR vs POSITION BUDGET d_p   (floor = mean ceil(size/d_p); no leading-space adjustment)")
print(f"{'d_p':>5} {'D_byte':>8} {'D_char':>8} | " + " | ".join(f"{l.split('_')[0]:>17}" for l in LANGS))
print(f"{'':>5} {'':>8} {'':>8} | " + " | ".join(f"{'byte':>8}{'char':>9}" for l in LANGS))
for dp in [16,32,48,64,96,128,256]:
    row=f"{dp:>5} {256*dp:>8} {192*dp:>8} | "
    cells=[]
    for l in LANGS:
        nb,nc=W[l]
        fb=np.ceil(nb/dp).mean(); fc=np.ceil(nc/dp).mean()
        cells.append(f"{fb:>8.4f}{fc:>9.4f}")
    print(row+" | ".join(cells))
print()
print("MAX REPRESENTABLE TOKEN LENGTH IN CHARACTERS (bound, not observed):")
print(f"{'d_p':>5} | {'Latin(1B/ch)':>13} | {'Brahmic(3B/ch)':>15} | {'ratio':>6} || char-basis: all scripts")
for dp in [16,32,48,64,96,128,256]:
    print(f"{dp:>5} | {dp:>13} | {dp//3:>15} | {dp/(dp//3):>6.2f} || {dp}")
