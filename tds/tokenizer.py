"""A byte-level BPE tokenizer that is trained once, frozen, and hashed.

Byte-level rather than character-level for a specific reason: it is Indic-safe by construction. A
character-level vocabulary trained on a corpus that is mostly Latin will emit <unk> for Devanagari
conjuncts it has not seen; a byte-level one degrades to multiple tokens instead, which costs fertility
but never loses information. S2 fixed the tokenizer contract as "the same raw text must always produce
the same token ids"; losing bytes to <unk> breaks round-tripping and therefore breaks that contract.

The tokenizer hash covers vocabulary, merges and special tokens. It is written into every shard
manifest, because token ids are meaningless without the table that produced them - a shard and a
tokenizer are a matched pair, and mixing versions silently corrupts a run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .canon import chash

# Special tokens occupy the first ids so that they are stable across any retrain.
PAD, BOS, EOS = 0, 1, 2
SPECIALS = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS}


@dataclass(frozen=True)
class Tokenizer:
    vocab: dict[str, int]          # token string (as latin-1 escaped bytes) -> id
    merges: tuple[tuple[str, str], ...]
    version: str

    @property
    def hash(self) -> str:
        return chash({
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "specials": SPECIALS,
            "version": self.version,
        })

    @property
    def size(self) -> int:
        return len(self.vocab)

    # -- encoding -------------------------------------------------------------------------
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        symbols = [bytes([b]).decode("latin-1") for b in text.encode("utf-8")]
        rank = {pair: i for i, pair in enumerate(self.merges)}
        while len(symbols) > 1:
            best, best_rank, best_i = None, None, None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                r = rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank, best_i = pair, r, i
            if best is None:
                break
            symbols[best_i:best_i + 2] = [best[0] + best[1]]
        ids = [self.vocab[s] for s in symbols if s in self.vocab]
        if add_bos:
            ids = [BOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        out = []
        for i in ids:
            if i in (PAD, BOS, EOS):
                continue
            out.append(inv.get(i, ""))
        return "".join(out).encode("latin-1", errors="ignore").decode("utf-8", errors="replace")

    def to_record(self) -> dict:
        return {"version": self.version, "size": self.size, "n_merges": len(self.merges),
                "hash": self.hash, "specials": SPECIALS}


def train_tokenizer(texts: list[str], *, vocab_size: int = 512, version: str = "v5-tok-1") -> Tokenizer:
    """Deterministic byte-level BPE.

    Ties in pair frequency are broken by the pair itself (lexicographic), never by dict order, so the
    result does not depend on Python's hash seed or insertion order. Without that tie-break the
    tokenizer hash would drift between runs and every shard manifest would become unverifiable.
    """
    vocab: dict[str, int] = dict(SPECIALS)
    for b in range(256):
        vocab.setdefault(bytes([b]).decode("latin-1"), len(vocab))

    corpora = [[bytes([b]).decode("latin-1") for b in t.encode("utf-8")] for t in texts]
    merges: list[tuple[str, str]] = []

    while len(vocab) < vocab_size:
        counts: Counter = Counter()
        for seq in corpora:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += 1
        if not counts:
            break
        best_count = max(counts.values())
        if best_count < 2:
            break
        # deterministic tie-break: highest count, then lexicographically smallest pair
        pair = min((p for p, c in counts.items() if c == best_count))
        merged = pair[0] + pair[1]
        if merged not in vocab:
            vocab[merged] = len(vocab)
        merges.append(pair)
        for seq in corpora:
            i = 0
            while i < len(seq) - 1:
                if (seq[i], seq[i + 1]) == pair:
                    seq[i:i + 2] = [merged]
                else:
                    i += 1
    return Tokenizer(vocab=vocab, merges=tuple(merges), version=version)


def fertility(tok: Tokenizer, text: str) -> float:
    """Tokens per whitespace-delimited word. Reported per language in the run log.

    S5 sized the Indic lane in content rather than tokens because the same content costs 1.375
    tokens/word in Hindi and 3.025 in Malayalam. A data system that reports only token counts hides
    that, so the shard report carries fertility per language.
    """
    words = max(1, len(text.split()))
    return len(tok.encode(text)) / words
