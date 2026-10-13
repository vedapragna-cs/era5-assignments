import numpy as np, pickle, random
raw=pickle.load(open('full_raw.pkl','rb')); scripts=np.load('full_scripts.npy',allow_pickle=True)
rng=random.Random(0)
def lcs_off(a,b):
    la,lb=len(a),len(b); best=(0,0,0); prev=[0]*(lb+1)
    for i in range(1,la+1):
        cur=[0]*(lb+1); ai=a[i-1]
        for j in range(1,lb+1):
            if ai==b[j-1]:
                cur[j]=prev[j-1]+1
                if cur[j]>best[0]: best=(cur[j], i-cur[j], j-cur[j])
        prev=cur
    return best
def code(bs): return {(v,p) for p,v in enumerate(bs)}
def cos(A,B): return len(A&B)/(len(A)*len(B))**0.5 if A and B else 0.0
MINCH=3   # common substring of at least 3 CHARACTERS, script-fair
print(f"LCS threshold = {MINCH} CHARACTERS (byte threshold scaled by script's bytes/char)")
print(f"{'script':<12}{'B/ch':>5}{'minB':>6}{'hits':>7} | {'aligned':>8}{'cos':>7} | {'shifted':>8}{'cos':>7} | {'%shift':>7}{'overlap lost':>13}")
for sc in ['LATIN','DEVANAGARI','MALAYALAM','TAMIL','KANNADA','TELUGU','BENGALI']:
    idx=np.where(scripts==sc)[0]
    toks=[]
    for i in idx:
        b=raw[i]
        try: s=b.decode('utf-8')
        except: continue
        if 3<=len(s)<=12: toks.append(b)
    rng.shuffle(toks); toks=toks[:1200]
    bpc = 1 if sc=='LATIN' else 3
    minB = MINCH*bpc
    al=[]; sh=[]; n=0
    trials = 60000
    for _ in range(trials):
        a,b=rng.choice(toks),rng.choice(toks)
        if a==b: continue
        L,oa,ob=lcs_off(a,b)
        if L<minB: continue
        n+=1
        c=cos(code(a),code(b))
        (al if oa==ob else sh).append(c)
    ma=np.mean(al) if al else 0; ms=np.mean(sh) if sh else 0
    pct=100*len(sh)/max(n,1); loss=100*(1-ms/ma) if ma>0 else 0
    print(f"{sc:<12}{bpc:>5}{minB:>6}{n:>7} | {len(al):>8}{ma:>7.3f} | {len(sh):>8}{ms:>7.3f} | {pct:>6.1f}%{loss:>12.1f}%")
