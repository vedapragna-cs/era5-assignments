"""Tokenizer, byte tables, and batching.

The tokenizer is an ordinary byte-level BPE and is *not* part of the
contribution. It is here because the whole point of the method is that it keeps
standard tokenization and changes only what happens at the two ends of the
model. Every arm sees exactly the same token stream.

Vocabulary size
---------------
Set to 16,384 rather than 8,192 on purpose. The codec dimension is
``D = 256 * 32 = 8192``; choosing |V| = 8192 too would make the baseline head and
the byte head identical in size by coincidence and turn the parameter comparison
into a wash. The structural claim is that the byte head's size does not depend on
|V| at all, so the two have to differ for the measurement to say anything.

Byte tables
-----------
Each token id is resolved once to its byte labels ``(P,)`` and content length.
Those two tables are all the codec ever needs -- the dense vectors are never
built. This is the paper's `gpu_dynamic` variant: for |V| = 16,384 and P = 32 the
tables are ~0.5 MB, against 537 MB for the equivalent dense table in float32.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from .codec import DEFAULT as DEFAULT_CODEC
from .codec import CodecConfig, PAD_BYTE, target_bytes, token_bytes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")

DEFAULT_VOCAB = 16384


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------


def train_tokenizer(text: str, vocab_size: int = DEFAULT_VOCAB,
                    path: str = TOKENIZER_PATH, force: bool = False):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if os.path.exists(path) and not force:
        return Tokenizer.from_file(path)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator([text], trainer=trainer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok.save(path)
    return tok


def _bytes_to_unicode() -> Dict[int, str]:
    """The GPT-2 byte-level alphabet: byte value -> printable stand-in character."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_UNICODE_TO_BYTE = {c: b for b, c in _bytes_to_unicode().items()}


def token_raw_bytes(tok, token_id: int) -> bytes:
    """The literal bytes a token id stands for.

    Byte-level BPE stores tokens as printable stand-ins -- a leading space is
    U+0120, and raw bytes like 0x80 get their own characters -- so the mapping
    has to be inverted directly.

    Going through `decoder.decode` instead returns a *str*, and any token whose
    bytes are not valid UTF-8 on their own comes back as U+FFFD. Re-encoding that
    yields the three bytes of the replacement character rather than the one byte
    the token actually is. Measured: 256 single-byte tokens were corrupted this
    way, so the codec would have been trained on bytes the tokenizer never
    emitted.
    """
    piece = tok.id_to_token(token_id)
    out = bytearray()
    for ch in piece:
        if ch in _UNICODE_TO_BYTE:
            out.append(_UNICODE_TO_BYTE[ch])
        else:
            # Special tokens such as <pad> are literal text, not stand-ins.
            out.extend(ch.encode("utf-8"))
    return bytes(out)


def token_surface(tok, token_id: int) -> str:
    """Human-readable form. For display only -- never feed this to the codec."""
    return token_raw_bytes(tok, token_id).decode("utf-8", errors="replace")


def build_byte_tables(tok, cfg: CodecConfig = DEFAULT_CODEC
                      ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """``(V, P)`` byte labels and ``(V,)`` content lengths for every token id."""
    V = tok.get_vocab_size()
    labels = np.zeros((V, cfg.positions), dtype=np.int64)
    lens = np.zeros(V, dtype=np.int64)

    truncated = 0
    empty = 0
    has_null = 0
    for i in range(V):
        raw_full = token_raw_bytes(tok, i)
        if len(raw_full) > cfg.positions:
            truncated += 1
        raw = raw_full[:cfg.positions]
        if PAD_BYTE in raw:
            # A literal 0x00 in a token would be indistinguishable from padding.
            # Real UTF-8 text never contains one; counted so the assumption is
            # checked rather than asserted.
            has_null += 1
        if len(raw) == 0:
            empty += 1
        labels[i, :len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
        lens[i] = len(raw)

    stats = {
        "vocab_size": V,
        "tokens_truncated": truncated,
        "tokens_with_empty_surface": empty,
        "tokens_containing_null_byte": has_null,
        "mean_bytes_per_token": float(lens.mean()),
        "max_bytes_per_token": int(lens.max()),
        "table_bytes": int(labels.nbytes + lens.nbytes),
        "dense_table_bytes_would_be": int(V * cfg.dim * 4),
    }
    return labels, lens, stats


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


class TokenStream:
    """Encoded corpus split into train and validation, with batching."""

    def __init__(self, text: str, tok, cfg: CodecConfig = DEFAULT_CODEC,
                 val_fraction: float = 0.05, seq_len: int = 256):
        ids = np.array(tok.encode(text).ids, dtype=np.int64)
        n_val = int(len(ids) * val_fraction)
        # Contiguous tail rather than a random split: a shuffled split would put
        # neighbouring tokens of the same sentence on both sides and leak.
        self.train = ids[:-n_val]
        self.val = ids[-n_val:]
        self.seq_len = seq_len
        self.cfg = cfg
        self.labels, self.lens, self.table_stats = build_byte_tables(tok, cfg)
        self.vocab_size = tok.get_vocab_size()

        # PAD_BYTE doubles as the terminator, so a token whose bytes contain a
        # literal 0x00 would be indistinguishable from a shorter token. Such a
        # token *exists* in the vocabulary -- byte-level BPE seeds all 256 bytes --
        # but it must never actually occur. Checked rather than assumed.
        null_ids = [i for i in range(self.vocab_size)
                    if PAD_BYTE in self.labels[i, :self.lens[i]]]
        if null_ids:
            hit = np.isin(ids, null_ids).sum()
            if hit:
                raise ValueError(
                    "corpus contains {} occurrences of token ids {} whose bytes "
                    "include 0x00; PAD would be ambiguous".format(hit, null_ids))
        self.table_stats["null_byte_token_occurrences"] = 0

        # Filled in by `set_heldout`. Default: nothing is withheld, so the
        # ordinary comparison runs unchanged.
        self.heldout_ids = np.array([], dtype=np.int64)
        self.is_heldout = np.zeros(self.vocab_size, dtype=bool)

    def set_heldout(self, ids: np.ndarray) -> None:
        self.heldout_ids = np.asarray(ids, dtype=np.int64)
        self.is_heldout = np.zeros(self.vocab_size, dtype=bool)
        self.is_heldout[self.heldout_ids] = True

    def select_heldout(self, coverage: float = 0.01, min_count: int = 30,
                       max_count: int = 3000, seed: int = 0) -> np.ndarray:
        """Token ids to withhold from every arm's training loss.

        This is the experiment that actually tests vocabulary-freedom. A held-out
        token still appears in the *context* -- so every arm's shared Kronecker
        input pathway sees its bytes -- but it is never a prediction target during
        training. At evaluation it is scored on exactly those positions.

        That isolates the head and nothing else. The ``vocab`` head's output row
        for such a token receives gradient only as a negative example, pushed down
        by softmax normalisation, so it should be at or below chance. The byte
        heads have never been asked for the token either, but they have been asked
        for every one of its bytes in other words, and they compose. Whether that
        composition happens is the whole claim.

        Selection avoids both tails on purpose. Tokens seen fewer than
        ``min_count`` times would be badly estimated even by an arm that had
        trained on them, so failure there would prove nothing; tokens seen more
        than ``max_count`` times are common enough that withholding them would
        distort the language model itself rather than probe the head.
        """
        counts = np.bincount(self.train, minlength=self.vocab_size)
        eligible = np.where((counts >= min_count) & (counts <= max_count))[0]
        if len(eligible) == 0:
            return np.array([], dtype=np.int64)

        rng = np.random.default_rng(seed)
        order = rng.permutation(eligible)
        target = coverage * len(self.train)

        taken, total = [], 0
        for tid in order:
            if total >= target:
                break
            taken.append(int(tid))
            total += int(counts[tid])
        return np.sort(np.array(taken, dtype=np.int64))

    def heldout_report(self, ids: np.ndarray, tok=None) -> Dict:
        counts = np.bincount(self.train, minlength=self.vocab_size)
        val_counts = np.bincount(self.val, minlength=self.vocab_size)
        rep = {
            "n_tokens_withheld": int(len(ids)),
            "train_positions_masked": int(counts[ids].sum()),
            "train_coverage": round(float(counts[ids].sum()) / len(self.train), 6),
            "val_positions_scored": int(val_counts[ids].sum()),
            "val_coverage": round(float(val_counts[ids].sum()) / len(self.val), 6),
            "mean_train_count": round(float(counts[ids].mean()), 2) if len(ids) else 0.0,
            "mean_bytes": round(float(self.lens[ids].mean()), 2) if len(ids) else 0.0,
        }
        if tok is not None and len(ids):
            rep["examples"] = [token_surface(tok, int(i)) for i in ids[:40]]
        return rep

    def stats(self) -> Dict:
        return {
            "train_tokens": int(len(self.train)),
            "val_tokens": int(len(self.val)),
            "seq_len": self.seq_len,
            **self.table_stats,
        }

    def _batch_from(self, ids: np.ndarray, idx: np.ndarray, device: str):
        T = self.seq_len
        # windows of T+1: the first T are input, the last T are the target
        win = np.stack([ids[i:i + T + 1] for i in idx])
        src, tgt = win[:, :-1], win[:, 1:]

        labels = torch.from_numpy(self.labels[src]).to(device)      # (B, T, P)
        lens = torch.from_numpy(self.lens[src]).to(device)          # (B, T)
        tgt_labels = torch.from_numpy(self.labels[tgt]).to(device)  # (B, T, P)
        tgt_lens = torch.from_numpy(self.lens[tgt]).to(device)      # (B, T)
        tgt_ids = torch.from_numpy(tgt).to(device)                  # (B, T)
        # True where the *target* is a withheld token. Note this is keyed on the
        # target only: withheld tokens still appear in `src`, so every arm's
        # input pathway sees their bytes. The head is the only thing deprived.
        held = torch.from_numpy(self.is_heldout[tgt]).to(device)     # (B, T)
        return {"in_labels": labels, "in_lens": lens,
                "tgt_labels": tgt_labels, "tgt_lens": tgt_lens, "tgt_ids": tgt_ids,
                "tgt_heldout": held}

    def batches(self, split: str, batch_size: int, n_batches: int,
                seed: int = 0, device: str = "cpu") -> Iterator[Dict]:
        ids = self.train if split == "train" else self.val
        rng = np.random.default_rng(seed)
        hi = len(ids) - self.seq_len - 1
        for _ in range(n_batches):
            yield self._batch_from(ids, rng.integers(0, hi, size=batch_size), device)

    def sequential_batches(self, split: str, batch_size: int,
                           device: str = "cpu") -> Iterator[Dict]:
        """Non-overlapping sweep, for evaluation.

        Random sampling would score some tokens twice and others never, which is
        fine for a training signal and wrong for a held-out measurement.
        """
        ids = self.train if split == "train" else self.val
        T = self.seq_len
        starts = np.arange(0, len(ids) - T - 1, T)
        for i in range(0, len(starts), batch_size):
            chunk = starts[i:i + batch_size]
            if len(chunk):
                yield self._batch_from(ids, chunk, device)
