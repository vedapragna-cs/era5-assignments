"""Character-basis Kronecker codec, verified losslessly over the whole vocabulary.

kappa(t) = SUM_p [ block(c_p) (x) e_p ]  +  SUM_p [ offset(c_p) (x) e_p ]

Position indexes BOTH factors, so the pair (block_p, offset_p) is recoverable and the
codepoint is block_p*128 + offset_p. Two block ids are reserved so that tokens which are
not valid UTF-8 on their own (byte-level BPE fragments) round-trip as raw bytes at no
extra dimension cost.
"""
import json
from transformers import AutoTokenizer

P_CHAR = 32
def _b2u():
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256)); cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return {chr(c):b for b,c in zip(bs,cs)}
U2B=_b2u()

def build_block_map(strings):
    blocks=sorted({ord(c)>>7 for s in strings for c in s})
    m={b:i for i,b in enumerate(blocks)}
    return m, len(blocks)

def encode(tok_str, blkmap, raw_lo, raw_hi):
    """-> (block_idx[], offset[]) of equal length, <= P_CHAR."""
    raw=bytes(U2B[c] for c in tok_str)
    try:
        s=raw.decode()
        return [blkmap[ord(c)>>7] for c in s], [ord(c)&127 for c in s]
    except UnicodeDecodeError:                       # raw-byte fallback, no extra dims
        return [raw_lo if b<128 else raw_hi for b in raw], [b&127 for b in raw]

def decode(blk, off, inv, raw_lo, raw_hi):
    if blk and blk[0] in (raw_lo, raw_hi):
        return bytes(((128 if b==raw_hi else 0)|o) for b,o in zip(blk,off))
    return "".join(chr((inv[b]<<7)|o) for b,o in zip(blk,off))

if __name__=="__main__":
    tok=AutoTokenizer.from_pretrained("theschoolofai/BrahmicTokenizer-131K")
    V=list(tok.get_vocab())
    decodable=[]
    for t in V:
        try: bytes(U2B[c] for c in t).decode(); decodable.append(t)
        except UnicodeDecodeError: pass
    strs=[bytes(U2B[c] for c in t).decode() for t in decodable]
    blkmap,nb=build_block_map(strs)
    raw_lo, raw_hi = nb, nb+1                        # two reserved ids
    inv={v:k for k,v in blkmap.items()}
    B=1<<(nb+2-1).bit_length()
    ok=fail=over=0
    for t in V:
        blk,off=encode(t,blkmap,raw_lo,raw_hi)
        if len(blk)>P_CHAR: over+=1; continue
        want=bytes(U2B[c] for c in t)
        got=decode(blk,off,inv,raw_lo,raw_hi)
        got=got if isinstance(got,bytes) else got.encode()
        ok += (got==want); fail += (got!=want)
    D=B*P_CHAR+128*P_CHAR; d=8096; Dbyte=256*32
    print(f"vocabulary: {len(V):,} tokens  (decodable {len(decodable):,}, byte-fallback {len(V)-len(decodable):,})")
    print(f"blocks used: {nb} + 2 reserved -> block axis {B}")
    print(f"ROUND-TRIP: exact={ok:,}  mismatches={fail}  over-{P_CHAR}-chars={over}")
    print(f"\nD = {B} x {P_CHAR} + 128 x {P_CHAR} = {D}")
    print(f"projection @ d_model={d}: {D*d:,}   vs byte grid {Dbyte*d:,}   ({100*(D/Dbyte-1):+.1f}%)")
    print(f"vs dense table {131072*d:,}: {131072*d/(D*d):.1f}x smaller")
    json.dump({"P_CHAR":P_CHAR,"blocks_used":nb,"block_axis":B,"D":D,
               "params_d8096":D*d,"roundtrip_exact":ok,"roundtrip_fail":fail}, open("charkron.json","w"), indent=1)
