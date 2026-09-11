"""C2 step 1 - recover per-token language from the codec's UTF-8 bytes, and VALIDATE it
against the known val_lang labels before trusting it anywhere.

train.npy has no language labels; val_lang.npy does. codec.npz carries every token's raw
UTF-8 bytes, and en/hi/ml occupy disjoint Unicode blocks, so language is recoverable by
script. The validation below is the whole point: a silently wrong label map would invent
a mixture shift that isn't there.
"""
import json, os
import numpy as np

Q = os.path.expanduser("~/shared/gpu-queue")
M = json.load(open(f"{Q}/data/meta.json")); NV = M["vocab"]; LANGS = M["langs"]
C = np.load(f"{Q}/data/codec.npz")
bv = C["byte_idx"].astype(np.int64)          # (V, 32), -1 = no byte
print("codec byte_idx:", bv.shape, "| langs:", LANGS)

# UTF-8: Devanagari U+0900-U+097F -> E0 A4/A5 xx ; Malayalam U+0D00-U+0D7F -> E0 B4/B5 xx
DEV, MAL = {0xA4, 0xA5}, {0xB4, 0xB5}
def classify(row):
    b = row[row >= 0]
    if len(b) == 0: return -1
    dev = mal = lat = 0
    i = 0
    while i < len(b):
        if b[i] == 0xE0 and i+1 < len(b):
            if b[i+1] in DEV: dev += 1; i += 3; continue
            if b[i+1] in MAL: mal += 1; i += 3; continue
        if 0x20 <= b[i] <= 0x7E: lat += 1
        i += 1
    # continuation-only fragments: fall back to any A4/A5 vs B4/B5 byte present
    if dev == mal == lat == 0:
        dev = int(np.isin(b, list(DEV)).sum()); mal = int(np.isin(b, list(MAL)).sum())
        if dev == mal == 0: return -1
    return int(np.argmax([lat, dev, mal]))    # 0=en 1=hi 2=ml, matching meta["langs"]

tok_lang = np.array([classify(bv[i]) for i in range(NV)], np.int8)
print("per-token labels:", {LANGS[i]: int((tok_lang==i).sum()) for i in range(3)},
      "| unresolved:", int((tok_lang==-1).sum()))

# ---- VALIDATION against the ground-truth val labels -----------------------------------
val   = np.load(f"{Q}/data/val.npy").astype(np.int64)
vlang = np.load(f"{Q}/data/val_lang.npy").astype(np.int64)
pred  = tok_lang[val]
ok    = pred >= 0
print(f"\ntoken-level agreement on val: {100*(pred[ok]==vlang[ok]).mean():.2f}% "
      f"over {ok.sum():,} resolvable of {len(val):,}")
print(f"{'':>6}" + "".join(f"{'pred '+l:>10}" for l in LANGS))
for g in range(3):
    m = (vlang == g) & ok
    print(f"{'true '+LANGS[g]:>6}" + "".join(f"{100*(pred[m]==p).mean():>9.1f}%" for p in range(3)))

# block-level: the corpus is built from 1024-token single-language blocks
BLK = 1024
def block_labels(arr):
    n = len(arr)//BLK
    lab = tok_lang[arr[:n*BLK]].reshape(n, BLK)
    out = np.full(n, -1, np.int8)
    for i in range(n):
        r = lab[i][lab[i] >= 0]
        if len(r): out[i] = np.bincount(r, minlength=3).argmax()
    return out
vb_pred = block_labels(val)
vb_true = vlang[:len(vb_pred)*BLK].reshape(-1, BLK)[:, 0]
acc = (vb_pred == vb_true).mean()
print(f"\nBLOCK-level agreement on val: {100*acc:.2f}% over {len(vb_pred)} blocks of {BLK}")

train = np.load(f"{Q}/data/train.npy").astype(np.int64)
tb = block_labels(train)
print(f"\ntrain: {len(tb)} blocks of {BLK} ({len(tb)*BLK:,} tokens)")
print("  block mixture:", {LANGS[i]: f"{100*(tb==i).mean():.1f}%" for i in range(3)},
      "| unresolved:", int((tb==-1).sum()))

assert acc > 0.95, f"block accuracy {acc:.3f} too low to build a mixture shift on"
np.savez(f"{Q}/data/lang_labels.npz", tok_lang=tok_lang, train_block=tb,
         blk=BLK, val_block_acc=acc)
json.dump(dict(token_agreement=float((pred[ok]==vlang[ok]).mean()),
               block_agreement=float(acc), blk=BLK,
               n_train_blocks=int(len(tb)), unresolved_tokens=int((tok_lang==-1).sum()),
               train_mixture={LANGS[i]: float((tb==i).mean()) for i in range(3)}),
          open("../results/c2_langmap.json","w"), indent=1)
print(f"\nwrote {Q}/data/lang_labels.npz and ../results/c2_langmap.json")
