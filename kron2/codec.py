"""The invertible Kronecker codec.

The V1 codec maps a token's bytes to a fixed D-dimensional vector, where
``D = 256 * P`` and each byte contributes a single 1 at flat index
``byte_value * P + position``. That map is deterministic in the forward
direction, and the paper stops there: the output head stays a learned
``d_model x |V|`` matrix, so the model can *encode* any byte string but can only
ever *predict* one of the |V| tokens it was given.

This module supplies the missing direction. The observation the whole project
rests on is structural rather than numerical:

    a codec vector reshaped to (P, 256) is P independent one-hot rows

so recovering the token need not be a nearest-neighbour search over the
vocabulary in 8192-dimensional space -- it can be P independent 256-way choices.

Why the classification reading and not the lookup
-------------------------------------------------
Not because nearest neighbour is fragile. `analysis.py` measures that directly
and it is *not*: under equal signal energy it decodes more accurately than
per-position argmax at every noise level tested, and the codec vectors are not
pathologically crowded either (median best-to-second cosine gap 0.052). That is
the tempting argument, it is wrong, and it is refuted in the repository rather
than quietly dropped.

The reasons that do survive measurement:

* **Reachability.** Nearest neighbour is an argmin over a fixed table, so its
  range is exactly the |V| strings in that table. Measured on 15 misspellings,
  proper nouns, identifiers and multi-byte strings, it can emit 0 of them;
  per-position argmax can emit all 15. A decoder that cannot leave the vocabulary
  cannot deliver a vocabulary-free model, which is the entire point of P5.
* **Likelihood.** An MSE head defines no normalised distribution over tokens, so
  it cannot be scored in bits per byte and cannot be sampled from.
* **Credit assignment.** At initialisation a prediction is nearly equidistant
  from every candidate -- measured gap between best and second-best cosine,
  0.0025 over 4,000 candidates -- so a lookup returns one almost arbitrary answer
  and no indication of which part was wrong. P classifications give a dense
  gradient at every position from the first step. This is a claim about
  training, not about decoding accuracy.

Two layouts and the hazard between them
---------------------------------------
The flat index ``byte * P + position`` means a row-major reshape to (256, P)
yields ``[byte, position]``. The natural layout for logits is the transpose,
``[position, byte]``, because cross-entropy wants the class axis last.

``reshape(P, 256)`` does **not** produce that transpose. It interleaves the two
axes and silently yields garbage that still trains, just badly. Every conversion
here goes through :func:`to_grid` / :func:`to_flat` rather than a bare reshape,
and ``test_layout_hazard`` pins the distinction.

Scaling
-------
The ``1/sqrt(L)`` factor in V1 is an input-side convenience: it keeps the vector
that feeds the projection at roughly unit norm regardless of token length. It is
actively harmful on the output side, because it shrinks the signal while any
prediction noise stays fixed, so long tokens decode worse than short ones.
Measured on a 20-byte token under Gaussian noise at sigma=0.10: decode accuracy
is 0.000 with the factor applied and 1.000 without it.

So the two directions deliberately use different scales: :func:`encode` applies
``1/sqrt(L)`` for the model input, and decoding consumes *logits* whose scale the
model learns. We never regress onto the normalised codec vector.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# Byte alphabet. Fixed at the full 256 so every byte value gets its own basis
# direction; collapsing it would conflate unrelated bytes.
N_BYTES = 256

# Default byte window. V1's production setting; kept here so that this work
# isolates invertibility rather than also changing the window (that is a
# separate problem in the assignment and mixing them would confound both).
DEFAULT_POSITIONS = 32

# The pad class. Real UTF-8 never contains a null byte, so 0x00 is free to mean
# "no byte at this position" without stealing a class or widening the grid to
# 257. Keeping the grid at exactly 256 * P is what lets the output space stay
# the same shape as the input space, which is what makes tying possible.
PAD_BYTE = 0


class CodecConfig:
    """Geometry of the codec. Immutable once a run starts."""

    __slots__ = ("n_bytes", "positions", "normalize_input")

    def __init__(self, positions: int = DEFAULT_POSITIONS,
                 n_bytes: int = N_BYTES, normalize_input: bool = True) -> None:
        self.n_bytes = n_bytes
        self.positions = positions
        self.normalize_input = normalize_input

    @property
    def dim(self) -> int:
        """D = 256 * P -- the flat codec dimension."""
        return self.n_bytes * self.positions

    def to_dict(self) -> dict:
        return {
            "n_bytes": self.n_bytes,
            "positions": self.positions,
            "dim": self.dim,
            "normalize_input": self.normalize_input,
            "pad_byte": PAD_BYTE,
        }

    def __repr__(self) -> str:
        return "CodecConfig(positions={}, n_bytes={}, dim={})".format(
            self.positions, self.n_bytes, self.dim)


DEFAULT = CodecConfig()


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def to_grid(flat: np.ndarray, cfg: CodecConfig = DEFAULT) -> np.ndarray:
    """Flat ``(..., D)`` -> grid ``(..., P, 256)`` indexed ``[position, byte]``.

    Goes through (256, P) and then swaps the last two axes. A direct
    ``reshape(P, 256)`` would interleave byte and position and produce a
    plausible-looking array that decodes to nothing; see ``test_layout_hazard``.
    """
    lead = flat.shape[:-1]
    if flat.shape[-1] != cfg.dim:
        raise ValueError("expected last dim {}, got {}".format(cfg.dim, flat.shape[-1]))
    byte_pos = flat.reshape(lead + (cfg.n_bytes, cfg.positions))
    return np.swapaxes(byte_pos, -2, -1)


def to_flat(grid: np.ndarray, cfg: CodecConfig = DEFAULT) -> np.ndarray:
    """Grid ``(..., P, 256)`` -> flat ``(..., D)``. Exact inverse of :func:`to_grid`."""
    if grid.shape[-2:] != (cfg.positions, cfg.n_bytes):
        raise ValueError("expected trailing shape {}, got {}".format(
            (cfg.positions, cfg.n_bytes), grid.shape[-2:]))
    byte_pos = np.swapaxes(grid, -2, -1)
    return byte_pos.reshape(grid.shape[:-2] + (cfg.dim,))


def flat_index(byte_value: int, position: int, cfg: CodecConfig = DEFAULT) -> int:
    """The V1 index formula, kept in one place so nothing re-derives it."""
    return byte_value * cfg.positions + position


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


def token_bytes(text: str, cfg: CodecConfig = DEFAULT) -> bytes:
    """UTF-8 bytes of ``text``, truncated to the window on a codepoint boundary.

    Truncating mid-codepoint would emit a byte sequence that is not valid UTF-8,
    so a token could encode to something that never decodes back to text. We back
    off to the previous boundary instead, which loses a character rather than
    producing an invalid one.
    """
    raw = text.encode("utf-8")
    if len(raw) <= cfg.positions:
        return raw
    cut = cfg.positions
    while cut > 0 and (raw[cut] & 0xC0) == 0x80:   # inside a continuation byte
        cut -= 1
    return raw[:cut]


def encode_grid(text: str, cfg: CodecConfig = DEFAULT) -> np.ndarray:
    """``text`` -> ``(P, 256)`` grid for the **model input**. Pad rows are zero.

    Input and target grids differ, and conflating them is a real bug rather than
    a stylistic choice. On the input side V1's ``1/sqrt(L)`` assumes exactly L
    nonzero entries; giving pad positions a one-hot would put 32 nonzeros in
    every vector and the norm would scale with *padding* instead of content
    (measured: "cat" came out at norm 3.27 instead of 1.0). Pad therefore
    contributes nothing here.

    On the target side the opposite is required -- see :func:`target_grid`.
    """
    b = token_bytes(text, cfg)
    grid = np.zeros((cfg.positions, cfg.n_bytes), dtype=np.float32)
    for p, value in enumerate(b):
        grid[p, value] = 1.0
    return grid


def target_grid(text: str, cfg: CodecConfig = DEFAULT) -> np.ndarray:
    """``text`` -> ``(P, 256)`` one-hot grid for the **decoder target**.

    Here pad *is* an explicit class. An all-zero target row carries no gradient
    and would give the decoder nothing to predict at that position, so the model
    could never learn where a token ends -- and token length is precisely what
    the output side must produce. Every row is one-hot.
    """
    b = token_bytes(text, cfg)
    grid = np.zeros((cfg.positions, cfg.n_bytes), dtype=np.float32)
    grid[:, PAD_BYTE] = 1.0
    for p, value in enumerate(b):
        grid[p, PAD_BYTE] = 0.0
        grid[p, value] = 1.0
    return grid


def encode(text: str, cfg: CodecConfig = DEFAULT) -> np.ndarray:
    """``text`` -> flat ``(D,)`` codec vector for the *model input*.

    Applies ``1/sqrt(L)`` when ``cfg.normalize_input``, matching V1. Pad
    positions are excluded from the scale: they are structural filler, not
    content, and letting them count would make short tokens quieter than long
    ones for no reason.
    """
    b = token_bytes(text, cfg)
    grid = encode_grid(text, cfg)
    flat = to_flat(grid, cfg)
    if cfg.normalize_input:
        flat = flat / np.sqrt(max(1, len(b)))
    return flat.astype(np.float32)


def encode_batch(texts: Sequence[str], cfg: CodecConfig = DEFAULT) -> np.ndarray:
    return np.stack([encode(t, cfg) for t in texts])


def target_bytes(text: str, cfg: CodecConfig = DEFAULT) -> np.ndarray:
    """``text`` -> ``(P,)`` int64 of class labels: byte values, then PAD.

    This is what the decoder is trained against. Cross-entropy over these labels
    is the whole training signal on the output side -- there is no regression
    onto the codec vector anywhere in this project.
    """
    b = token_bytes(text, cfg)
    labels = np.full(cfg.positions, PAD_BYTE, dtype=np.int64)
    labels[:len(b)] = np.frombuffer(b, dtype=np.uint8).astype(np.int64)
    return labels


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


# NOTE ON DECODING A ZERO ROW. `encode_grid` leaves pad rows all-zero, and
# `argmax` of an all-zero row returns index 0, which is PAD_BYTE. So a clean
# input encoding still round-trips. That is by construction, not luck -- PAD_BYTE
# is 0 for exactly this reason -- and `test_zero_row_decodes_as_pad` pins it so
# the constant cannot drift.


def decode_labels(labels: Sequence[int], stop_at_pad: bool = True) -> bytes:
    """``(P,)`` class labels -> bytes.

    ``stop_at_pad`` truncates at the first pad. Independent per-position
    prediction can produce a pad in the middle followed by more bytes -- a hole.
    Stopping is the conservative reading; :func:`decode_report` counts how often
    holes occur so the cost of independence is measured rather than hidden.
    """
    out = bytearray()
    for value in labels:
        v = int(value)
        if v == PAD_BYTE:
            if stop_at_pad:
                break
            continue
        out.append(v)
    return bytes(out)


def decode_grid(grid: np.ndarray, cfg: CodecConfig = DEFAULT,
                stop_at_pad: bool = True) -> bytes:
    """``(P, 256)`` scores (logits or one-hot) -> bytes, by per-position argmax."""
    return decode_labels(np.asarray(grid).argmax(axis=-1), stop_at_pad=stop_at_pad)


def decode(flat: np.ndarray, cfg: CodecConfig = DEFAULT,
           stop_at_pad: bool = True) -> bytes:
    """Flat ``(D,)`` -> bytes."""
    return decode_grid(to_grid(np.asarray(flat), cfg), cfg, stop_at_pad=stop_at_pad)


def decode_text(flat: np.ndarray, cfg: CodecConfig = DEFAULT,
                errors: str = "replace") -> str:
    return decode(flat, cfg).decode("utf-8", errors=errors)


def decode_report(grid: np.ndarray, cfg: CodecConfig = DEFAULT) -> dict:
    """Decode one grid and describe *how* it decoded.

    Beyond the bytes, reports the per-position margin (top logit minus runner-up)
    and whether the prediction contains a hole. The margin is the quantity that
    decides whether a noisy prediction survives, so it is the natural thing to
    report alongside an accuracy number.
    """
    g = np.asarray(grid, dtype=np.float64)
    labels = g.argmax(axis=-1)
    part = np.partition(g, -2, axis=-1)
    margins = part[..., -1] - part[..., -2]

    pads = labels == PAD_BYTE
    first_pad = int(np.argmax(pads)) if pads.any() else cfg.positions
    hole = bool(pads.any() and (~pads[first_pad:]).any())

    return {
        "bytes": decode_labels(labels),
        "labels": labels.tolist(),
        "length": first_pad,
        "has_hole": hole,
        "margin_min": float(margins[:max(1, first_pad)].min()),
        "margin_mean": float(margins[:max(1, first_pad)].mean()),
    }


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


def round_trip(text: str, cfg: CodecConfig = DEFAULT) -> Tuple[bool, str]:
    """Encode then decode. Returns ``(exact, recovered_text)``.

    Exactness is judged against ``token_bytes`` rather than the original string,
    so a token longer than the window counts as a faithful round trip of what the
    codec actually represents. Truncation is a declared property of the window,
    not a decoding failure, and conflating the two would hide real bugs.
    """
    recovered = decode(encode(text, cfg), cfg)
    return recovered == token_bytes(text, cfg), recovered.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# torch mirrors of the layout conversions
# ---------------------------------------------------------------------------
#
# Kept beside the numpy versions rather than in the model, so that the layout
# rule lives in exactly one file. The same hazard applies: a bare
# `reshape(P, 256)` interleaves the axes. `test_torch_layout_matches_numpy`
# pins the two implementations against each other.


def to_grid_torch(flat, cfg: CodecConfig = DEFAULT):
    """Flat ``(..., D)`` -> ``(..., P, 256)`` indexed ``[position, byte]``."""
    lead = flat.shape[:-1]
    byte_pos = flat.reshape(lead + (cfg.n_bytes, cfg.positions))
    return byte_pos.transpose(-2, -1)


def to_flat_torch(grid, cfg: CodecConfig = DEFAULT):
    """``(..., P, 256)`` -> flat ``(..., D)``. Exact inverse of :func:`to_grid_torch`."""
    byte_pos = grid.transpose(-2, -1)
    return byte_pos.reshape(grid.shape[:-2] + (cfg.dim,))
