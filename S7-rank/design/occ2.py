import numpy as np, pickle, unicodedata
raw = pickle.load(open('full_raw.pkl','rb')); V=len(raw); P=32
def script_of(b):
    try: s=b.decode('utf-8')
    except UnicodeDecodeError: return 'partial'
    letters=[c for c in s if c.isalpha()]
    if not letters: return 'nonalpha'
    names=set()
    for c in letters:
        try: n=unicodedata.name(c).split()[0]
        except ValueError: n='UNK'
        names.add(n)
    if len(names)==1: return names.pop()
    return 'mixed'
scripts=np.array([script_of(b) for b in raw])
u,c=np.unique(scripts,return_counts=True)
print("script counts:", sorted(zip(u.tolist(),c.tolist()), key=lambda x:-x[1])[:12])

# full-vocab (byte,pos) occupancy
grid=np.zeros((256,P),dtype=bool)
for b in raw:
    for p,bv in enumerate(b[:P]): grid[bv,p]=True
print(f"\nFULL VOCAB (131,072 tokens): live (byte,pos) cells = {grid.sum()} / {256*P} = {100*grid.sum()/(256*P):.1f}%")

# per-script occupancy: cells reachable using ONLY tokens of that script
for sc in ['LATIN','DEVANAGARI','MALAYALAM','BENGALI','TAMIL','TELUGU','ORIYA','GUJARATI','KANNADA','GURMUKHI']:
    m = scripts==sc
    if m.sum()==0: continue
    g=np.zeros((256,P),dtype=bool)
    mx=0
    for b in np.array(raw,dtype=object)[m]:
        mx=max(mx,len(b))
        for p,bv in enumerate(b[:P]): g[bv,p]=True
    # chars
    ch=[]
    for b in np.array(raw,dtype=object)[m]:
        try: ch.append(len(b.decode('utf-8')))
        except: pass
    print(f"  {sc:<12} n={m.sum():>6}  live cells={g.sum():>5} ({100*g.sum()/8192:>4.1f}%)  max bytes={mx:>2}  max chars={max(ch) if ch else 0:>2}")
np.save('full_grid.npy', grid); np.save('full_scripts.npy', scripts)
