"""Invariants of the invertible codec.

These are the properties everything downstream assumes. The layout test in
particular guards a failure that produces no error and no crash -- just a model
that trains to a worse place for a reason nobody can see.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kron2 import codec as C

WORDS = ["apple", "a", "cat", "Kubernetes", "netwrok", "kronekticus",
         "internationalisation", "नमस्ते", "日本語", "café", "x" * 40,
         "def f(x):", "  leading", "trailing  ", "MiXeD", "42", "!@#$%"]


# ---------------------------------------------------------------------------
# the hazard
# ---------------------------------------------------------------------------


def test_layout_hazard_reshape_is_not_transpose():
    """`reshape(P, 256)` silently interleaves byte and position.

    This is the single most dangerous line that could be written in this project:
    it runs, it produces the right shape, it trains, and it is wrong. The correct
    conversion is reshape-to-(256,P)-then-swap, which is what `to_grid` does.
    """
    cfg = C.DEFAULT
    flat = C.encode("apple", cfg)

    good = C.to_grid(flat, cfg)                                    # (P, 256)
    naive = flat.reshape(cfg.positions, cfg.n_bytes)               # (P, 256), wrong

    assert good.shape == naive.shape, "the bug is invisible in the shape"
    assert not np.array_equal(good, naive), "reshape must not equal transpose"
    assert C.decode_grid(good, cfg) == b"apple"
    assert C.decode_grid(naive, cfg) != b"apple", "naive reshape must not decode"


def test_to_grid_and_to_flat_are_inverse():
    cfg = C.DEFAULT
    flat = C.encode("Kubernetes", cfg)
    assert np.allclose(C.to_flat(C.to_grid(flat, cfg), cfg), flat)


def test_flat_index_matches_grid_position():
    """The V1 index formula and the grid layout must agree, or encode and decode
    are describing different objects."""
    cfg = C.DEFAULT
    for byte_value, position in [(97, 0), (112, 3), (255, 31), (0, 0)]:
        flat = np.zeros(cfg.dim, dtype=np.float32)
        flat[C.flat_index(byte_value, position, cfg)] = 1.0
        grid = C.to_grid(flat, cfg)
        assert grid[position, byte_value] == 1.0
        assert grid.sum() == 1.0


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("word", WORDS)
def test_round_trip_is_exact(word):
    """Encode then decode must return exactly what the codec represents."""
    exact, recovered = C.round_trip(word)
    assert exact, "{!r} -> {!r}".format(word, recovered)


def test_round_trip_on_bytes_never_seen_in_any_vocabulary():
    """The whole point of the output side: strings no tokenizer would contain.

    If these do not round trip, the vocabulary-free claim is empty.
    """
    for junk in ["qwxzjv", "aaaaaaaa", "Zz9!~", "kronekticus", "netwrok", "ﬁ‰∂"]:
        exact, recovered = C.round_trip(junk)
        assert exact, "{!r} -> {!r}".format(junk, recovered)


def test_truncation_is_utf8_safe():
    """Cutting mid-codepoint would emit bytes that are not valid UTF-8."""
    cfg = C.DEFAULT
    text = "नमस्ते" * 5                       # 3 bytes per character
    b = C.token_bytes(text, cfg)
    assert len(b) <= cfg.positions
    b.decode("utf-8")                        # must not raise
    assert C.decode(C.encode(text, cfg), cfg) == b


def test_distinct_tokens_get_distinct_codes():
    cfg = C.DEFAULT
    seen = {}
    for w in WORDS:
        key = C.encode(w, cfg).tobytes()
        assert key not in seen, "collision: {!r} and {!r}".format(w, seen.get(key))
        seen[key] = w


# ---------------------------------------------------------------------------
# padding and labels
# ---------------------------------------------------------------------------


def test_input_grid_and_target_grid_differ_on_padding():
    """The two grids have different jobs and must not be conflated.

    Regression test: reusing the target grid as input made `1/sqrt(L)` normalise
    against 32 nonzeros instead of L, so "cat" encoded to norm 3.27 rather than
    1.0 -- padding, not content, set the scale.
    """
    cfg = C.DEFAULT
    inp = C.encode_grid("cat")
    tgt = C.target_grid("cat")

    assert inp.shape == tgt.shape == (cfg.positions, cfg.n_bytes)
    assert inp[0, ord("c")] == 1.0 and tgt[0, ord("c")] == 1.0
    assert inp[3].sum() == 0.0, "input pad rows carry nothing"
    assert tgt[3, C.PAD_BYTE] == 1.0, "target pad rows are a real class"
    assert np.allclose(tgt.sum(axis=-1), 1.0), "every target row is one-hot"
    assert inp.sum() == 3.0, "input has exactly L nonzeros"


def test_zero_row_decodes_as_pad():
    """PAD_BYTE is 0 so that argmax of an all-zero input row lands on pad."""
    assert C.PAD_BYTE == 0
    zeros = np.zeros((C.DEFAULT.positions, C.DEFAULT.n_bytes))
    assert C.decode_grid(zeros, C.DEFAULT) == b""


def test_target_labels_match_the_grid():
    cfg = C.DEFAULT
    labels = C.target_bytes("cat", cfg)
    assert labels.shape == (cfg.positions,)
    assert list(labels[:3]) == [ord("c"), ord("a"), ord("t")]
    assert set(labels[3:].tolist()) == {C.PAD_BYTE}
    assert C.decode_labels(labels) == b"cat"


def test_normalisation_is_input_side_only():
    """Encode carries 1/sqrt(L); the grid used for targets does not."""
    cfg = C.DEFAULT
    flat = C.encode("cat", cfg)
    assert np.isclose(np.linalg.norm(flat), 1.0, atol=1e-5)
    grid = C.target_grid("cat")
    assert grid[0, ord("c")] == 1.0, "targets must be unscaled"


# ---------------------------------------------------------------------------
# noise, which is what the model will actually hand the decoder
# ---------------------------------------------------------------------------


def test_decoding_survives_noise_that_breaks_nearest_neighbour():
    """Per-position argmax must tolerate noise on unscaled logits.

    Uses the unscaled grid, which is what the head emits. The normalised codec
    vector degrades much faster with length -- the reason decoding consumes
    logits rather than the encoded vector.
    """
    rng = np.random.default_rng(0)
    cfg = C.DEFAULT
    for word in ["apple", "internationalisation"]:
        target = C.token_bytes(word, cfg)
        ok = 0
        for _ in range(200):
            noisy = C.target_grid(word) + rng.normal(0, 0.10, (cfg.positions, cfg.n_bytes))
            ok += C.decode_grid(noisy, cfg) == target
        assert ok / 200 > 0.95, "{}: {}".format(word, ok / 200)


def test_decode_report_flags_holes():
    """A pad followed by more bytes is an inconsistency independence can produce."""
    cfg = C.DEFAULT
    grid = np.zeros((cfg.positions, cfg.n_bytes))
    grid[:, C.PAD_BYTE] = 1.0
    grid[0, C.PAD_BYTE], grid[0, ord("a")] = 0.0, 1.0
    grid[2, C.PAD_BYTE], grid[2, ord("b")] = 0.0, 1.0   # hole at position 1
    rep = C.decode_report(grid, cfg)
    assert rep["has_hole"] is True
    assert rep["bytes"] == b"a", "conservative decode stops at the first pad"


def test_margin_is_reported_and_positive_for_clean_codes():
    rep = C.decode_report(C.target_grid("apple"))
    assert rep["bytes"] == b"apple"
    assert rep["has_hole"] is False
    assert rep["margin_min"] == 1.0
    assert rep["length"] == 5


# ---------------------------------------------------------------------------
# the torch mirrors must agree with numpy exactly
# ---------------------------------------------------------------------------


def test_torch_layout_matches_numpy():
    """Two implementations of the same layout rule are two chances to get it
    wrong; this pins them together."""
    import torch
    cfg = C.DEFAULT
    flat = C.encode("Kubernetes", cfg)
    np_grid = C.to_grid(flat, cfg)
    t_grid = C.to_grid_torch(torch.from_numpy(flat), cfg).numpy()
    assert np.array_equal(np_grid, t_grid)
    back = C.to_flat_torch(torch.from_numpy(np_grid), cfg).numpy()
    assert np.allclose(back, flat)
    assert C.decode_grid(t_grid, cfg) == b"Kubernetes"


def test_sparse_gather_equals_dense_codec_matmul():
    """The input pathway avoids materialising the dense codec vector.

    That optimisation must be exact, not approximate: `embed` gathers P rows of
    the projection and scales, and it has to equal `encode(text) @ proj` to
    floating-point tolerance. If it drifts, every arm is quietly training on a
    different input than the one documented.
    """
    import torch
    from kron2.model import KronLM, ModelConfig

    cfg = ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, seq_len=8,
                      vocab_size=64, head="kron")
    torch.manual_seed(0)
    m = KronLM(cfg)

    words = ["apple", "a", "Kubernetes", "café"]
    labels = torch.tensor([[C.target_bytes(w) for w in words]])          # (1, T, P)
    lens = torch.tensor([[len(C.token_bytes(w)) for w in words]])        # (1, T)

    got = m.embed(labels, lens)                                          # (1, T, d_model)

    dense = torch.from_numpy(np.stack([C.encode(w) for w in words])).unsqueeze(0)
    want = dense @ m.proj_weight                                         # (1, T, d_model)

    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


# ---------------------------------------------------------------------------
# corpus integrity
# ---------------------------------------------------------------------------


def test_corpus_matches_its_manifest():
    """The recorded hash must describe what `load` returns, not what was written.

    Regression test. Gutenberg serves CRLF; a freshly downloaded book kept it
    while a cache-warm rebuild silently collapsed it in text mode. The manifest
    hashed the written bytes, `load` returned the text-mode reading, and the two
    differed by 40,634 bytes -- a mismatch that reports a corpus nothing trained
    on and would make the run unreproducible for anyone else.
    """
    from kron2 import corpus

    report = corpus.verify()
    assert report["match"], report


def test_corpus_contains_no_carriage_returns():
    """The normalisation that keeps the hash stable, checked directly."""
    from kron2 import corpus

    text = corpus.load()
    assert "\r" not in text
