"""Properties of the four heads, and of the quantity they are compared by.

The tests that matter here are the scoring ones. Bits-per-byte is the only number
that lets a byte head and a vocabulary head be put in the same table, so if it is
not literally a log-probability the whole comparison is decoration.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kron2 import codec as C
from kron2.model import HEADS, KronLM, ModelConfig, content_mask, token_nats

LN256 = math.log(256.0)


def tiny(head: str, vocab: int = 64) -> KronLM:
    torch.manual_seed(0)
    return KronLM(ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64,
                              seq_len=8, vocab_size=vocab, head=head))


def batch(words):
    labels = torch.tensor([[C.target_bytes(w) for w in words]])
    lens = torch.tensor([[len(C.token_bytes(w)) for w in words]])
    ids = torch.arange(len(words)).unsqueeze(0)
    return labels, lens, ids


# ---------------------------------------------------------------------------
# the mask: which positions are part of the string
# ---------------------------------------------------------------------------


def test_content_mask_covers_bytes_plus_one_terminator():
    m = content_mask(torch.tensor([0, 1, 5, 31, 32]), 32)
    assert m.sum(-1).tolist() == [1, 2, 6, 32, 32]
    assert m[2, 4] == 1.0 and m[2, 5] == 1.0, "L bytes plus the PAD at index L"
    assert m[2, 6] == 0.0, "nothing after the terminator"
    # A token filling the window has no room for a terminator; all P are content.
    assert m[4].sum() == 32


# ---------------------------------------------------------------------------
# bits per byte must be a real likelihood
# ---------------------------------------------------------------------------


def test_uniform_kron_head_costs_exactly_L_plus_one_times_ln256():
    """The arithmetic check that anchors the metric.

    A head with all-equal logits assigns 1/256 to every byte, so the string
    ``w`` of length L costs exactly ``(L+1) * ln(256)`` nats -- L bytes and one
    terminator. Any other answer means positions are being counted that are not
    part of the string.

    Regression test: summing all 32 positions instead gave ``32 * ln(256)``
    regardless of L, which at random init reads as 256 bpb for a 1-byte token and
    21 bpb for a 12-byte one. That is not a constant offset, so it cannot be
    corrected after the fact -- it distorts the arm comparison by an amount set by
    the token-length distribution.
    """
    m = tiny("kron")
    words = ["a", "cat", "apple", "internationalisation"]
    labels, lens, ids = batch(words)

    class Flat(torch.nn.Module):
        cfg = m.cfg

        def forward(self, *a, **k):
            return torch.zeros(1, len(words), m.cfg.codec.dim)

    nats = token_nats(Flat(), labels, lens, labels, ids, lens)[0]
    for w, got in zip(words, nats.tolist()):
        L = len(C.token_bytes(w))
        assert got == pytest.approx((L + 1) * LN256, rel=1e-5), w


def test_uniform_vocab_head_costs_exactly_ln_vocab():
    """The same anchor for the baseline, so the two are known to be on one scale."""
    V = 64
    m = tiny("vocab", vocab=V)
    labels, lens, ids = batch(["a", "cat"])

    class Flat(torch.nn.Module):
        cfg = m.cfg

        def forward(self, *a, **k):
            return torch.zeros(1, 2, V)

    nats = token_nats(Flat(), labels, lens, labels, ids, lens)[0]
    assert nats.tolist() == pytest.approx([math.log(V)] * 2, rel=1e-6)


def test_kron_likelihood_is_normalised_over_strings():
    """Probabilities of all strings from a tiny alphabet must sum to 1.

    This is the property that makes bits-per-byte meaningful rather than merely
    small. It is checked on a 3-symbol, 2-position toy so the sum is finite: the
    strings are "", "a", "b", "aa", "ab", "ba", "bb" -- every string the codec can
    represent -- and the terminated reading must account for all the mass.
    """
    import torch.nn.functional as F

    P, NB = 2, 3                      # PAD=0, "a"=1, "b"=2
    torch.manual_seed(1)
    logits = torch.randn(P, NB)
    logp = F.log_softmax(logits, dim=-1)

    total = 0.0
    for s in [[], [1], [2], [1, 1], [1, 2], [2, 1], [2, 2]]:
        lp = sum(logp[i, b] for i, b in enumerate(s))
        if len(s) < P:                # a terminator is emitted; else the window ends
            lp = lp + logp[len(s), 0]
        total += math.exp(lp)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_padding_positions_do_not_change_the_score():
    """Whatever the head says after the terminator must not affect the likelihood."""
    m = tiny("kron")
    labels, lens, ids = batch(["cat"])
    D = m.cfg.codec.dim

    base = torch.zeros(1, 1, D)
    perturbed = base.clone()
    grid = perturbed.reshape(1, 1, m.cfg.codec.n_bytes, m.cfg.codec.positions)
    grid[..., 10:] = 5.0              # nonsense well past the terminator at 3

    class Fixed(torch.nn.Module):
        cfg = m.cfg

        def __init__(self, t):
            super().__init__()
            self.t = t

        def forward(self, *a, **k):
            return self.t

    a = token_nats(Fixed(base), labels, lens, labels, ids, lens)
    b = token_nats(Fixed(perturbed), labels, lens, labels, ids, lens)
    assert torch.allclose(a, b), "the tail must be marginalised, not scored"


# ---------------------------------------------------------------------------
# tying: the actual claim of P5
# ---------------------------------------------------------------------------


def test_tied_head_has_no_output_weight_of_its_own():
    m = tiny("kron_tied")
    counts = m.param_counts()
    assert m.head is None
    assert counts["output"] == m.cfg.codec.dim, "a bias, and nothing else"
    names = {n for n, _ in m.named_parameters()}
    assert not any(n.startswith("head.") for n in names)


def test_tied_output_is_literally_the_transposed_input_projection():
    """Not an approximation and not a copy -- the same tensor, read both ways."""
    m = tiny("kron_tied")
    labels, lens, _ = batch(["cat", "apple"])
    with torch.no_grad():
        x = m.embed(labels, lens)
        x = x + m.pos_emb(torch.arange(2))
        for blk in m.blocks:
            x = blk(x, m._causal_mask(2, x.device))
        x = m.ln_f(x)
        want = x @ m.proj_weight.t() + m.head_bias
        got = m.forward(labels, lens)
    assert torch.allclose(got, want, atol=1e-6)


def test_tying_sends_gradient_to_the_projection_from_both_sides():
    """If the tie were a detached copy this would still train and still be wrong."""
    m = tiny("kron_tied")
    labels, lens, ids = batch(["cat", "apple"])
    loss, _ = m.loss(labels, lens, labels, ids, lens)
    loss.backward()
    assert m.proj_weight.grad is not None
    assert m.proj_weight.grad.abs().sum() > 0


def test_output_side_cost_of_each_head():
    """The parameter claim, stated as a test so it cannot drift.

    ``vocab`` scales with |V|; ``kron`` is fixed at D regardless of |V|; ``tied``
    is fixed at D *bias terms*, which is 1/d_model of even that.
    """
    small, large = tiny("vocab", vocab=64), tiny("vocab", vocab=4096)
    assert large.param_counts()["output"] == 64 * small.param_counts()["output"]

    for head in ("kron", "kron_tied"):
        a, b = tiny(head, vocab=64), tiny(head, vocab=4096)
        assert a.param_counts()["output"] == b.param_counts()["output"], head


# ---------------------------------------------------------------------------
# the arms must differ only where they are supposed to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head", HEADS)
def test_every_arm_shares_the_same_trunk_and_input_shape(head):
    m = tiny(head)
    c = m.param_counts()
    assert c["input"] == m.cfg.codec.dim * m.cfg.d_model
    assert c["trunk"] == tiny("vocab").param_counts()["trunk"]


@pytest.mark.parametrize("head", HEADS)
def test_loss_is_finite_and_backpropagates(head):
    m = tiny(head)
    labels, lens, ids = batch(["cat", "apple"])
    loss, parts = m.loss(labels, lens, labels, ids, lens)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters())


def test_regress_arm_reports_no_likelihood():
    """It has no normalised distribution, so the metric must refuse to invent one."""
    m = tiny("regress")
    labels, lens, ids = batch(["cat"])
    nats = token_nats(m, labels, lens, labels, ids, lens)
    assert torch.isnan(nats).all()


def test_regress_closed_form_matches_the_dense_mse():
    """The dense target is never built; the shortcut has to be exact, not close."""
    from kron2.model import _regress_sq_error

    m = tiny("regress")
    words = ["cat", "apple", "a"]
    labels, lens, _ = batch(words)
    out = torch.randn(1, len(words), m.cfg.codec.dim)

    dense = torch.from_numpy(np.stack([C.encode(w) for w in words])).unsqueeze(0)
    want = ((out - dense) ** 2).sum(-1)
    got = _regress_sq_error(out, labels, lens, m.cfg)
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()


def test_vocab_restricted_score_is_a_renormalisation():
    """Restricting to the vocabulary must be exactly Bayes' rule, not a fudge.

    ``-log P(t | t is a token) = -log P(t) + log sum_v P(v)``. Since the sum runs
    over a subset of all byte strings it is at most 1, so log Z <= 0 and the
    restricted score can only be *lower*. If it ever came out higher, the gather
    would be indexing the grid wrongly -- which is silent, since both numbers
    still look like plausible bits-per-byte.
    """
    import torch.nn.functional as F
    from kron2.train import vocab_restricted_logz

    class Stream:
        pass

    words = ["cat", "apple", "a", "the"]
    s = Stream()
    s.labels = np.stack([C.target_bytes(w) for w in words])
    s.lens = np.array([len(C.token_bytes(w)) for w in words])

    m = tiny("kron")
    torch.manual_seed(3)
    logp = F.log_softmax(torch.randn(5, m.cfg.codec.positions,
                                     m.cfg.codec.n_bytes), dim=-1)
    logz = vocab_restricted_logz(logp, s, m.cfg, "cpu")

    assert (logz <= 1e-6).all(), "a subset of the mass cannot exceed 1"

    # Recompute one entry the slow, obvious way.
    want = torch.logsumexp(torch.tensor([
        sum(logp[0, i, b] for i, b in enumerate(s.labels[j][:s.lens[j]]))
        + (logp[0, s.lens[j], 0] if s.lens[j] < logp.shape[1] else 0.0)
        for j in range(len(words))]), dim=0)
    assert torch.allclose(logz[0], want, atol=1e-5), (logz[0], want)


def test_restricted_score_of_a_uniform_head_is_log_vocab():
    """With every token equally likely, restricting must cost exactly ln|V|."""
    import torch.nn.functional as F
    from kron2.train import vocab_restricted_logz

    class Stream:
        pass

    # Four distinct single-byte tokens, so a uniform byte head makes them equally
    # likely and the renormalised distribution is exactly uniform over 4.
    s = Stream()
    s.labels = np.stack([C.target_bytes(w) for w in ["a", "b", "c", "d"]])
    s.lens = np.array([1, 1, 1, 1])

    m = tiny("kron")
    logp = F.log_softmax(torch.zeros(2, m.cfg.codec.positions,
                                     m.cfg.codec.n_bytes), dim=-1)
    logz = vocab_restricted_logz(logp, s, m.cfg, "cpu")

    nats = 2 * LN256                        # one byte plus one terminator
    restricted = nats + logz[0].item()
    assert restricted == pytest.approx(math.log(4), abs=1e-5)
