"""E20 -- do shipped tokenizers' pre-tokenizers handle Unicode Mark characters?

WHY THIS IS A REAL QUESTION.  BPE can never merge across a pre-tokenizer boundary.  So if
the pre-tokenizer regex splits on \\p{L}+ (GPT-2's), Indic vowel signs -- Unicode category
Mc/Mn, MARK, not Letter -- become their own pieces BEFORE BPE runs, and no amount of
vocabulary can repair it.  This is invisible in every end-to-end fertility comparison
because everyone measures the tokenizer as a whole.

We hit this ourselves: it invalidated E13 and cost Malayalam 5.82x in pre-tokenizer pieces
alone.  The question here is whether shipped tokenizers have the same defect.

MEASURE.  (1) read each tokenizer.json's pre_tokenizer regex and test it for \\p{M};
(2) measure pieces-per-word produced by the PRE-TOKENIZER ALONE on FLORES devtest.
"""
import json, glob, os, re, sys, unicodedata
from tokenizers import Tokenizer, pre_tokenizers

CANDIDATES = ["gpt2","bert-base-multilingual-cased","xlm-roberta-base","google/mt5-small",
              "facebook/nllb-200-distilled-600M","EleutherAI/gpt-neox-20b","bigscience/bloom",
              "Qwen/Qwen2.5-0.5B","tiiuae/falcon-7b","ai4bharat/IndicBERTv2-MLM-only",
              "google/gemma-2-2b","mistralai/Mistral-7B-v0.1","meta-llama/Llama-3.1-8B",
              "sarvamai/sarvam-1","CohereForAI/aya-23-8B"]

def regexes(pt, out=None):
    """walk a pre_tokenizer config and collect every regex pattern in it"""
    out = [] if out is None else out
    if isinstance(pt, dict):
        for k, v in pt.items():
            if k == "pattern" and isinstance(v, dict):
                for kk in ("Regex","String"):
                    if kk in v: out.append(v[kk])
            else: regexes(v, out)
    elif isinstance(pt, list):
        for v in pt: regexes(v, out)
    return out

def kinds(pt, out=None):
    out = [] if out is None else out
    if isinstance(pt, dict):
        if "type" in pt and isinstance(pt["type"], str): out.append(pt["type"])
        for v in pt.values(): kinds(v, out)
    elif isinstance(pt, list):
        for v in pt: kinds(v, out)
    return out

rows=[]
for name in CANDIDATES:
    try:
        tk = Tokenizer.from_pretrained(name)
    except Exception as e:
        rows.append(dict(name=name, ok=False, err=repr(e)[:90])); print(f"  {name:<38} UNAVAILABLE"); continue
    cfg = json.loads(tk.to_str())
    pt  = cfg.get("pre_tokenizer")
    pats = regexes(pt); ks = kinds(pt)
    has_M   = any("\\p{M}" in p or "\\p{Mn}" in p or "\\p{Mc}" in p for p in pats)
    has_Lp  = any("\\p{L}" in p for p in pats)
    rows.append(dict(name=name, ok=True, kinds=sorted(set(ks)), npat=len(pats),
                     has_mark=has_M, has_letter_class=has_Lp,
                     sample=(pats[0][:70] if pats else None)))
    print(f"  {name:<38} kinds={sorted(set(ks))!s:<52} \\p{{M}}={has_M} \\p{{L}}={has_Lp}")
json.dump(rows, open("e20_pretok_regex.json","w"), indent=1)
print("\nsaved e20_pretok_regex.json")
