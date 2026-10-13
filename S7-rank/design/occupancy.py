import numpy as np, json, unicodedata, sys
sys.path.insert(0,'.venv-tok/lib/python3.13/site-packages')
from tokenizers import Tokenizer
import glob
p = glob.glob('/home/agent/.cache/huggingface/hub/models--theschoolofai--BrahmicTokenizer-131K/snapshots/*/tokenizer.json')[0]
tok = Tokenizer.from_file(p)
V = tok.get_vocab_size()
vocab = tok.get_vocab()
inv = [None]*V
for s,i in vocab.items(): inv[i]=s
# decode each token id to its byte string
import base64
def tok_bytes(i):
    s = inv[i]
    if s is None: return None
    # tokenizers ByteLevel uses the GPT-2 byte->unicode map
    return s
# build gpt2 byte decoder
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("\xa1"), ord("\xac")+1))+list(range(ord("\xae"), ord("\xff")+1))
    cs = bs[:]; n=0
    for b in range(2**8):
        if b not in bs: bs.append(b); cs.append(2**8+n); n+=1
    return dict(zip([chr(c) for c in cs], bs))
DEC = bytes_to_unicode()
lens_b=np.zeros(V,dtype=np.int32); ok=np.zeros(V,dtype=bool)
raw=[None]*V
for i in range(V):
    s=inv[i]
    if s is None: continue
    try:
        b=bytes(DEC[ch] for ch in s)
    except KeyError:
        continue
    raw[i]=b; lens_b[i]=len(b); ok[i]=True
print(f"vocab {V}, decoded {ok.sum()}, max bytes {lens_b.max()}, >32: {(lens_b>32).sum()}")
np.save('full_lens.npy', lens_b)
import pickle
pickle.dump(raw, open('full_raw.pkl','wb'))
