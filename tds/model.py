"""A tiny but real language model, so the learning ledger carries measured loss rather than fiction.

The assignment permits fake training ("just send to a loop and come back"). We train for real because
the learning ledger is the part of the system that feeds V6, and per-token perplexity invented by a
random number generator cannot support any of the claims that ledger is supposed to make.

Why there is a real attention layer
-----------------------------------
An earlier version of this file was a bigram MLP, justified on the grounds that "a transformer would
add compute without changing a single ledger property". That justification was wrong, and it was
wrong about the one thing this system claims loudest.

Without attention, a token can only ever see the token immediately before it. `attention_crossings`
and `blocked_cross_segment_pairs` then measure a property of a *mask array* - correct, but never
exercised, because no mechanism existed that could have leaked across a document boundary in the
first place. The packing contract was asserted, not enforced.

Now the mask is applied inside the arithmetic: attention scores at disallowed (query, key) pairs are
set to -inf before the softmax, so a document physically cannot attend outside itself. That makes
`test_attention_isolation_is_enforced_in_the_arithmetic` possible - perturb a neighbouring document
inside the same packed window and every logit of this document is bit-identical. A mask you can
prove is load-bearing is worth more than a mask you can only validate.

Why positions are a model input, not bookkeeping
------------------------------------------------
The same argument applies a second time, and an earlier version failed it. `position_ids` were
computed by the packer, validated by `validate_masks` and written to the ledger - but
`forward_backward` never took them as an argument and the architecture had no positional term. The
model was permutation-invariant over its context: scrambling the order of the tokens a position could
see left the loss bit-identical. "Position ids reset at every document boundary" was therefore a
claim about an array, provable only against a restatement of the rule that built it.

`P` is a learned positional embedding indexed by `position_ids`, added to the token embedding. The
ids now change the arithmetic, which makes the reset-at-boundary contract falsifiable the same way
the attention mask is: scramble the ids and the loss must move (`test_position_ids_are_load_bearing`,
and the `scrambled_position_ids` negative control). Because the ids restart per document, two copies
of the same document at different offsets in a pack get identical positional treatment - which is the
property packing is supposed to deliver and previously could not demonstrate.

Architecture: (token embedding + positional embedding) -> multi-head masked self-attention with
residual -> tanh hidden -> vocab logits, trained with plain SGD and hand-written gradients (no
autograd, so the backward pass is auditable line by line). Heads are real heads - the mask is
broadcast across all of them, so isolation holds per head rather than only in aggregate.

Determinism scope
-----------------
`float64`, fixed reduction order, no threading-dependent kernels, and every initialisation drawn from
a derived seed. Within one machine and one thread count this is bitwise reproducible, and the demo
asserts it. It is NOT a portable guarantee: BLAS reduction order varies across hardware. The ledger
does not depend on model determinism - only on data-stream determinism - which is why replay restores
a checkpoint rather than recomputing one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .canon import chash, derive_rng
from .tokenizer import PAD


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class TinyLM:
    """(token emb + positional emb) -> masked self-attention (+residual) -> tanh -> vocab logits."""
    vocab_size: int
    d_model: int
    d_hidden: int
    n_heads: int
    max_positions: int
    E: np.ndarray
    P: np.ndarray
    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    PARAMS = ("E", "P", "Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2")

    @classmethod
    def init(cls, vocab_size: int, *, d_model: int = 32, d_hidden: int = 64,
             n_heads: int = 4, max_positions: int = 512, seed: int = 0) -> "TinyLM":
        assert d_model % n_heads == 0, "d_model must divide evenly into heads"
        rng = derive_rng(seed, "model_init", vocab_size, d_model, d_hidden)
        s = 0.08
        def n(*shape):
            return (rng.standard_normal(shape) * s).astype(np.float64)
        return cls(
            vocab_size=vocab_size, d_model=d_model, d_hidden=d_hidden, n_heads=n_heads,
            max_positions=max_positions,
            E=n(vocab_size, d_model), P=n(max_positions, d_model),
            Wq=n(d_model, d_model), Wk=n(d_model, d_model),
            Wv=n(d_model, d_model), Wo=n(d_model, d_model),
            W1=n(d_model, d_hidden), b1=np.zeros(d_hidden, dtype=np.float64),
            W2=n(d_hidden, vocab_size), b2=np.zeros(vocab_size, dtype=np.float64),
        )

    # -------------------------------------------------------------------- positions
    @staticmethod
    def positions_from_segments(segment_ids: np.ndarray) -> np.ndarray:
        """Fallback for callers that hold no packed sequence (synthetic rigs, gradient probes).

        The packer is the authority: `trainer` and the OPUS/reference scorers pass
        `seq.position_ids` straight through. This exists so a hand-built (tokens, segments) pair in a
        test still has a defined positional input, and it is never used to *check* the packer.
        """
        seg = np.asarray(segment_ids)
        pos = np.zeros(len(seg), dtype=np.int64)
        run = 0
        for i in range(len(seg)):
            if i > 0 and seg[i] == seg[i - 1]:
                run += 1
            else:
                run = 0
            pos[i] = run
        return pos

    # -------------------------------------------------------------------- masking
    @staticmethod
    def attention_mask(segment_ids: np.ndarray) -> np.ndarray:
        """(L, L) boolean: True where query i may attend to key j.

        Causal AND same-document. This is the packing contract expressed as arithmetic rather than
        as an assertion - the identical condition `tds/packing.py` validates structurally.

        The diagonal is always permitted so that padding rows (segment -1) still have a defined
        softmax; they carry no loss and therefore no gradient, so this cannot leak anything.
        """
        L = len(segment_ids)
        idx = np.arange(L)
        causal = idx[:, None] >= idx[None, :]
        same = (segment_ids[:, None] == segment_ids[None, :]) & (segment_ids[:, None] >= 0)
        allow = causal & same
        allow[idx, idx] = True
        return allow

    # -------------------------------------------------------------------- forward / backward
    def forward_backward(self, tokens: np.ndarray, loss_mask: np.ndarray, segment_ids: np.ndarray,
                         position_ids: np.ndarray | None = None):
        """Next-token prediction over one packed sequence.

        Loss is applied only where `loss_mask[t+1]` is 1 AND positions t, t+1 are in the same
        segment. Attention is additionally restricted to the same segment, so a token can neither be
        trained to predict across a document boundary nor read across one.

        `position_ids` is the packer's within-document position, reset at every boundary. It indexes
        the learned positional embedding, so it is a real input to the arithmetic - scrambling it
        changes the loss. Omitting it derives the same array from `segment_ids` (see
        `positions_from_segments`) for callers that hold no packed sequence.
        Returns (per_position_loss, n_loss_positions, grads).
        """
        L = len(tokens)
        if L < 2:
            return np.zeros(0), 0, None

        pos = (self.positions_from_segments(segment_ids) if position_ids is None
               else np.asarray(position_ids, dtype=np.int64))
        if len(pos) != L:
            raise ValueError(f"position_ids has length {len(pos)}, expected {L}")
        if L and int(pos.max()) >= self.max_positions:
            raise ValueError(
                f"position id {int(pos.max())} exceeds max_positions={self.max_positions}; "
                "the positional embedding cannot represent this sequence")

        allow = self.attention_mask(segment_ids)             # (L, L)
        H_, dh_ = self.n_heads, self.d_model // self.n_heads
        X = self.E[tokens] + self.P[pos]                     # (L, d)  token + position

        def split(M):                                        # (L, d) -> (H, L, d_head)
            return M.reshape(L, H_, dh_).transpose(1, 0, 2)

        Q, K, V = split(X @ self.Wq), split(X @ self.Wk), split(X @ self.Wv)
        scale = 1.0 / np.sqrt(dh_)
        S = (Q @ K.transpose(0, 2, 1)) * scale               # (H, L, L)
        S = np.where(allow[None, :, :], S, -1e30)
        A = softmax(S)                                       # (H, L, L), row-wise over keys
        Ctx = A @ V                                          # (H, L, d_head)
        C = Ctx.transpose(1, 0, 2).reshape(L, self.d_model)  # (L, d)
        H = X + C @ self.Wo                                  # residual
        h_pre = H @ self.W1 + self.b1
        h = np.tanh(h_pre)
        logits = h @ self.W2 + self.b2                       # (L, V)
        probs = softmax(logits)

        # position t predicts token t+1
        tgt = tokens[1:]
        valid = (loss_mask[1:] == 1) & (segment_ids[1:] == segment_ids[:-1]) & (segment_ids[1:] >= 0)
        valid &= (tgt != PAD)

        n = int(valid.sum())
        per_pos = np.zeros(L - 1, dtype=np.float64)
        if n == 0:
            return per_pos, 0, None
        idx = np.arange(L - 1)
        p_true = np.clip(probs[idx, tgt], 1e-12, 1.0)
        per_pos = -np.log(p_true) * valid

        # ---- backward, averaged over loss-bearing positions only ----
        dlogits = np.zeros_like(logits)
        d = probs[:-1].copy()
        d[idx, tgt] -= 1.0
        d *= valid[:, None]
        dlogits[:-1] = d / n

        gW2 = h.T @ dlogits
        gb2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T
        dh_pre = dh * (1.0 - h ** 2)
        gW1 = H.T @ dh_pre
        gb1 = dh_pre.sum(axis=0)
        dH = dh_pre @ self.W1.T                              # (L, d)

        dX = dH.copy()                                       # residual path
        gWo = C.T @ dH
        dC = dH @ self.Wo.T                                  # (L, d)
        dCtx = dC.reshape(L, H_, dh_).transpose(1, 0, 2)     # (H, L, d_head)
        gA = dCtx @ V.transpose(0, 2, 1)                     # (H, L, L)
        dV = A.transpose(0, 2, 1) @ dCtx                     # (H, L, d_head)
        # softmax Jacobian, per head, row-wise over keys
        dS = A * (gA - (gA * A).sum(axis=-1, keepdims=True))
        dS = np.where(allow[None, :, :], dS, 0.0)            # masked pairs get no gradient
        dQ = (dS @ K) * scale                                # (H, L, d_head)
        dK = (dS.transpose(0, 2, 1) @ Q) * scale

        def merge(M):                                        # (H, L, d_head) -> (L, d)
            return M.transpose(1, 0, 2).reshape(L, self.d_model)

        dQf, dKf, dVf = merge(dQ), merge(dK), merge(dV)
        gWq = X.T @ dQf
        gWk = X.T @ dKf
        gWv = X.T @ dVf
        dX += dQf @ self.Wq.T + dKf @ self.Wk.T + dVf @ self.Wv.T
        gE = np.zeros_like(self.E)
        np.add.at(gE, tokens, dX)                            # embeddings stay trainable (S5 seam rule 1)
        # X = E[tokens] + P[pos], so both embeddings receive the identical upstream gradient,
        # scattered by their own index. A position that never occurs gets exactly zero.
        gP = np.zeros_like(self.P)
        np.add.at(gP, pos, dX)

        return per_pos, n, {"E": gE, "P": gP, "Wq": gWq, "Wk": gWk, "Wv": gWv, "Wo": gWo,
                            "W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}

    # -------------------------------------------------------------------- optimisation
    def apply(self, grads: dict[str, np.ndarray], lr: float) -> float:
        """SGD step. Returns the global gradient norm (recorded in the learning ledger)."""
        total = 0.0
        for k, g in grads.items():
            total += float((g ** 2).sum())
        gnorm = float(np.sqrt(total))
        for k, g in grads.items():
            setattr(self, k, getattr(self, k) - lr * g)
        return gnorm

    # -------------------------------------------------------------------- state
    def state(self) -> dict:
        return {k: getattr(self, k).tolist() for k in self.PARAMS}

    def load_state(self, st: dict) -> None:
        for k, v in st.items():
            setattr(self, k, np.array(v, dtype=np.float64))
        # A checkpoint carries the positional table it was trained with. Take the bound from the
        # loaded array rather than from this instance's constructor default, so a probe built with
        # `TinyLM.init(1, seed=0)` reconstructs the saved geometry instead of the default one.
        if "P" in st:
            self.max_positions = int(self.P.shape[0])
        if "E" in st:
            self.vocab_size = int(self.E.shape[0])

    @property
    def weight_hash(self) -> str:
        return chash({k: np.round(getattr(self, k), 10).tolist() for k in self.PARAMS})

    def proxy_direction(self, seed: int) -> dict[str, np.ndarray]:
        """The golden-proxy gradient direction OPUS scores candidates against.

        In a real run this is the gradient of a held-out high-quality proxy set. Here it is a fixed
        derived direction in parameter space, which is enough to make OPUS a real cosine-similarity
        decision over live gradients rather than a coin flip.
        """
        rng = derive_rng(seed, "opus_proxy", self.vocab_size)
        d = {k: rng.standard_normal(getattr(self, k).shape) for k in self.PARAMS}
        norm = np.sqrt(sum(float((v ** 2).sum()) for v in d.values()))
        return {k: v / norm for k, v in d.items()}


def cosine(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    dot = sum(float((a[k] * b[k]).sum()) for k in a)
    na = np.sqrt(sum(float((a[k] ** 2).sum()) for k in a))
    nb = np.sqrt(sum(float((b[k] ** 2).sum()) for k in b))
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))
