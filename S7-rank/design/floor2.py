import glob, math, numpy as np, pyarrow.parquet as pq
from tokenizers import Tokenizer
tp = glob.glob('/home/agent/.cache/huggingface/hub/models--theschoolofai--BrahmicTokenizer-131K/snapshots/*/tokenizer.json')[0]
tok = Tokenizer.from_file(tp)
FILES = sorted(glob.glob('flores/*.parquet'))
print(f"{'lang':<10}{'words':>7}{'actual':>9}{'floorB32':>10}{'floorC32':>10}{'%multi_B':>10}{'%multi_C':>10}{'gain':>8}")
out={}
for f in FILES:
    lang = f.split('/')[-1].split('.')[0]
    t = pq.read_table(f)
    col = 'sentence' if 'sentence' in t.column_names else t.column_names[-1]
    sents = [s for s in t.column(col).to_pylist() if s]
    words=[]
    for s in sents: words.extend(s.split())
    nb = np.array([len(w.encode('utf-8')) for w in words])
    nc = np.array([len(w) for w in words])
    # floor: a word needs at least ceil(size / cap) tokens; cap is 32 minus 1 for the leading-space marker
    fb = np.ceil(nb/31).mean()
    fc = np.ceil(nc/31).mean()
    multiB = 100*(nb>31).mean(); multiC = 100*(nc>31).mean()
    ids_total = sum(len(tok.encode(s).ids) for s in sents)
    actual = ids_total/len(words)
    out[lang]=(len(words),actual,fb,fc,multiB,multiC)
    print(f"{lang:<10}{len(words):>7}{actual:>9.4f}{fb:>10.4f}{fc:>10.4f}{multiB:>9.2f}%{multiC:>9.2f}%{fb-fc:>8.4f}")
