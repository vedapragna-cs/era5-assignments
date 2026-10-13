"""E20b -- BEHAVIOURAL pre-tokenizer audit.  The static regex check (e20_pretok.py) is not
sufficient: ByteLevel's GPT-2 regex is compiled into the Rust implementation and never
appears in tokenizer.json, so a JSON scan reports \\p{M}=False for a tokenizer whose real
defect is invisible there.  Drive the pre-tokenizer and read what it actually emits.

METRIC: pieces per whitespace-word produced by the PRE-TOKENIZER ALONE, before BPE.
1.000 = the pre-tokenizer leaves words intact.  Anything above is a floor that BPE cannot
undo, because BPE never merges across a pre-tokenizer boundary.
"""
import json, os, glob, unicodedata
import pyarrow.parquet as pq
from tokenizers import Tokenizer

LANGS = {"en":"eng_Latn","hi":"hin_Deva","ta":"tam_Taml","ml":"mal_Mlym","kn":"kan_Knda","bn":"ben_Beng"}
WORDS = {}
for lg, fc in LANGS.items():
    t = pq.read_table(f"flores/{fc}.parquet")
    txt = t.column("text").to_pylist() if "text" in t.column_names else t.column(t.column_names[-1]).to_pylist()
    w = [x for line in txt for x in line.split()]
    WORDS[lg] = w
    marks = sum(1 for ch in "".join(w[:5000]) if unicodedata.category(ch).startswith("M"))
    tot   = len("".join(w[:5000]))
    print(f"{lg}: {len(w):,} words, Mark chars = {100*marks/max(tot,1):.1f}% of first 5000 words")

NAMES = ["gpt2","EleutherAI/gpt-neox-20b","bigscience/bloom","Qwen/Qwen2.5-0.5B",
         "tiiuae/falcon-7b","google/gemma-2-2b","mistralai/Mistral-7B-v0.1","sarvamai/sarvam-1",
         "facebook/nllb-200-distilled-600M","xlm-roberta-base","bert-base-multilingual-cased",
         "ai4bharat/IndicBERTv2-MLM-only"]
p = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--theschoolofai--BrahmicTokenizer-131K/snapshots/*/tokenizer.json"))[0]

def measure(tk, label, res):
    pt = tk.pre_tokenizer
    row = {"tokenizer": label}
    if pt is None:
        row["note"] = "no pre_tokenizer"; res.append(row); print(f"{label:<38} (none)"); return
    out = []
    for lg in LANGS:
        w = WORDS[lg][:8000]
        n = sum(len(pt.pre_tokenize_str(x)) for x in w)
        row[lg] = round(n/len(w), 4); out.append(f"{lg} {row[lg]:.3f}")
    row["indic_penalty"] = round(max(row[l] for l in ["hi","ta","ml","kn","bn"]) / row["en"], 3)
    res.append(row)
    print(f"{label:<38} " + "  ".join(out) + f"   worst-Indic/en = {row['indic_penalty']:.2f}x")

res = []
print(f"\n{'tokenizer':<38} pieces per whitespace-word, PRE-TOKENIZER ONLY")
measure(Tokenizer.from_file(p), "theschoolofai/BrahmicTokenizer-131K", res)
for nm in NAMES:
    try: measure(Tokenizer.from_pretrained(nm), nm, res)
    except Exception as e: print(f"{nm:<38} UNAVAILABLE"); res.append({"tokenizer":nm,"err":repr(e)[:70]})
json.dump(res, open("e20b_pretok_behavior.json","w"), indent=1)
print("\nsaved e20b_pretok_behavior.json")
