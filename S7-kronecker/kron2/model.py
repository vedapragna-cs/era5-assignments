"""A small transformer, and the four output heads under comparison.

Every arm shares an identical trunk and an identical Kronecker **input** pathway.
Only the head differs. That is deliberate: the question is whether the output
side can be made invertible, so anything else varying between arms would
confound the answer.

    bytes -> codec (fixed) -> W_proj -> [ transformer ] -> HEAD -> next token
                                                            ^^^^
                                                    the only thing that varies

The four heads
--------------
``vocab``      the standard baseline: ``d_model x |V|`` softmax. Scales with the
               vocabulary and can only ever emit a token it was given.

``kron``       ours: ``d_model x D``, reshaped to ``(P, 256)`` and trained as P
               independent 256-way classifications. Size is independent of |V|,
               and any byte string is reachable.

``kron_tied``  ours, completed: the output weight *is* the input projection
               transposed, so the output side has no parameters of its own. This
               is the configuration V1 calls architecturally impossible, and it
               only becomes possible because the output space is now the codec
               space rather than the vocabulary.

``regress``    the approach described in the session: predict the codec vector
               under MSE and decode by nearest neighbour over the vocabulary.
               Included because reproducing its failure is evidence, not
               decoration -- the measured cosine gap between the best and second
               best candidate at initialisation is 0.0009.

Comparability
-------------
Byte cross-entropy and token cross-entropy are different units, so the arms are
scored in **bits per byte**: the total negative log-probability the model assigns
to the correct next token, divided by that token's length in bytes. For ``vocab``
that is one softmax term; for ``kron`` it is the sum over the P positions, which
under the head's independence assumption is exactly the log-probability of that
byte string.

Note what this costs ``kron`` and why it is not hidden: its distribution ranges
over all 256^P byte strings, so it must spend probability mass ruling out
nonsense that ``vocab`` gets to ignore for free. That is the real price of
vocabulary-freedom, and it belongs in the number rather than in a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import DEFAULT as DEFAULT_CODEC
from .codec import CodecConfig, PAD_BYTE, to_grid_torch

HEADS = ("vocab", "kron", "kron_tied", "regress")


@dataclass
class ModelConfig:
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    seq_len: int = 256
    dropout: float = 0.0
    vocab_size: int = 8192
    head: str = "kron"
    codec: CodecConfig = field(default_factory=lambda: DEFAULT_CODEC)

    def to_dict(self) -> Dict:
        return {
            "d_model": self.d_model, "n_layers": self.n_layers,
            "n_heads": self.n_heads, "d_ff": self.d_ff, "seq_len": self.seq_len,
            "vocab_size": self.vocab_size, "head": self.head,
            "codec": self.codec.to_dict(),
        }


# ---------------------------------------------------------------------------
# trunk
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class KronLM(nn.Module):
    """Shared trunk with a swappable output head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.head not in HEADS:
            raise ValueError("unknown head {!r}".format(cfg.head))
        self.cfg = cfg
        D = cfg.codec.dim

        # ---- input: identical in every arm ----
        # `proj` is stored as (D, d_model) rather than nn.Linear's (d_model, D)
        # because the input pathway reads *rows* of it. The codec vector is
        # sparse -- exactly L nonzeros at indices `byte*P + p`, each 1/sqrt(L) --
        # so `codec @ proj` is a gather of L rows followed by a scaled sum.
        # Materialising the dense (B, T, 8192) tensor instead would cost ~134 MB
        # per batch at B=16, T=256 to represent at most 32 nonzeros per token.
        # This is the paper's `gpu_dynamic` variant, and it is exact rather than
        # an approximation.
        self.proj_weight = nn.Parameter(torch.empty(D, cfg.d_model))
        nn.init.normal_(self.proj_weight, std=0.02)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # ---- output: the variable ----
        if cfg.head == "vocab":
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        elif cfg.head in ("kron", "regress"):
            self.head = nn.Linear(cfg.d_model, D, bias=False)
        elif cfg.head == "kron_tied":
            # No weight of its own -- only a bias. The output map is
            # `proj_weight.T`, the same numbers the input pathway gathers from.
            # This is the standard tying identity, available here only because
            # the output space is now the codec space rather than the vocabulary.
            self.head = None
            self.head_bias = nn.Parameter(torch.zeros(D))

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---- parameter accounting -------------------------------------------

    def param_counts(self) -> Dict[str, int]:
        trunk = sum(p.numel() for b in self.blocks for p in b.parameters())
        trunk += sum(p.numel() for p in self.ln_f.parameters())
        trunk += self.pos_emb.weight.numel()
        inp = self.proj_weight.numel()
        if self.cfg.head == "kron_tied":
            out = self.head_bias.numel()
        else:
            out = self.head.weight.numel()
        return {"input": inp, "output": out, "trunk": trunk,
                "io_total": inp + out, "total": inp + out + trunk}

    # ---- forward ---------------------------------------------------------

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def embed(self, in_labels: torch.Tensor, in_lens: torch.Tensor) -> torch.Tensor:
        """``(B, T, P)`` byte labels -> ``(B, T, d_model)``.

        Equivalent to building the sparse codec vector and multiplying by the
        projection, without ever materialising it. Pad positions are masked out
        so they contribute nothing, matching `encode_grid`; the 1/sqrt(L) scale
        then normalises against content length, not window length.
        """
        cfg = self.cfg
        P = cfg.codec.positions
        B, T, _ = in_labels.shape
        pos = torch.arange(P, device=in_labels.device)
        flat_idx = in_labels * P + pos                       # byte*P + position
        rows = self.proj_weight[flat_idx]                    # (B, T, P, d_model)
        keep = (in_labels != PAD_BYTE).unsqueeze(-1).to(rows.dtype)
        scale = torch.rsqrt(in_lens.clamp(min=1).to(rows.dtype)).unsqueeze(-1)
        return (rows * keep).sum(dim=2) * scale

    def forward(self, in_labels: torch.Tensor, in_lens: torch.Tensor) -> torch.Tensor:
        """Returns raw head output; interpretation is the head's business."""
        T = in_labels.shape[1]
        x = self.embed(in_labels, in_lens)
        x = x + self.pos_emb(torch.arange(T, device=in_labels.device))
        mask = self._causal_mask(T, in_labels.device)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)

        if self.cfg.head == "kron_tied":
            # proj_weight is (D, d_model); d_model -> D is a plain matmul with
            # it, no transpose bookkeeping needed. This is the tying identity.
            # proj_weight is (D, d_model); its transpose maps d_model -> D.
            return x @ self.proj_weight.t() + self.head_bias
        return self.head(x)

    # ---- loss ------------------------------------------------------------

    def loss(self, in_labels: torch.Tensor, in_lens: torch.Tensor,
             tgt_labels: torch.Tensor, tgt_ids: torch.Tensor,
             tgt_lens: torch.Tensor, weights: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Training loss for whichever head is active.

        ``in_labels``  (B, T, P) int64  -- byte classes of the input token
        ``tgt_labels`` (B, T, P) int64  -- byte classes of the next token
        ``tgt_ids``    (B, T)    int64  -- vocabulary id of the next token
        ``tgt_lens``   (B, T)    int64  -- content byte length of the next token
        ``weights``    (B, T)    float  -- optional per-position weight, 0 to drop

        ``weights`` is how the held-out-vocabulary experiment is run: positions
        whose target is a withheld token get weight 0, so no arm is ever trained
        to predict them. It is applied identically to all four heads, which is the
        point -- the arms must differ only in the head, including here.
        """
        out = self.forward(in_labels, in_lens)
        cfg = self.cfg
        P, NB = cfg.codec.positions, cfg.codec.n_bytes

        def reduce(per_token: torch.Tensor) -> torch.Tensor:
            if weights is None:
                return per_token.mean()
            w = weights.to(per_token.dtype)
            return (per_token * w).sum() / w.sum().clamp(min=1.0)

        if cfg.head == "vocab":
            per = F.cross_entropy(out.reshape(-1, cfg.vocab_size),
                                  tgt_ids.reshape(-1),
                                  reduction="none").reshape(tgt_ids.shape)
            loss = reduce(per)
            return loss, {"nats_per_token": loss.detach()}

        if cfg.head == "regress":
            # The session's formulation: regress onto the codec vector, then
            # decode by nearest neighbour. Computed in closed form rather than by
            # materialising the dense target, which would cost 67 MB per batch to
            # represent at most 32 nonzeros per token:
            #     |out - t|^2 = |out|^2 - 2 <out, t> + |t|^2,  with |t|^2 = 1
            # and <out, t> a gather of L entries scaled by 1/sqrt(L).
            loss = reduce(_regress_sq_error(out, tgt_labels, tgt_lens, cfg))
            return loss, {"mse": loss.detach()}

        # kron / kron_tied: P independent 256-way classifications, but only over
        # the positions that are part of the string. See `content_mask`.
        grid = to_grid_torch(out, cfg.codec)                       # (B, T, P, 256)
        mask = content_mask(tgt_lens, P)                           # (B, T, P)
        if weights is not None:
            mask = mask * weights.to(mask.dtype).unsqueeze(-1)
        per_pos = F.cross_entropy(grid.reshape(-1, NB), tgt_labels.reshape(-1),
                                  reduction="none").reshape(tgt_labels.shape)
        loss = (per_pos * mask).sum() / mask.sum().clamp(min=1.0)
        return loss, {"nats_per_position": loss.detach()}


def content_mask(lens: torch.Tensor, positions: int) -> torch.Tensor:
    """``(..., P)`` float mask selecting the positions that define the string.

    The byte head is a *terminated* sequence model: it emits bytes until it emits
    PAD, and the decoder stops there. So a token of length L occupies positions
    ``0 .. L-1`` plus one terminating PAD at position ``L``; positions after that
    do not change the decoded string and are marginalised out -- their conditional
    probabilities sum to 1 and contribute nothing to the likelihood.

    Scoring them anyway would charge the head for re-predicting padding up to 31
    extra times. That is not a constant offset: at random init it costs a 1-byte
    token 256 bpb and a 12-byte token 21 bpb, so it distorts the comparison by an
    amount that depends on the token-length distribution.

    ``lens == P`` (a token that fills the whole window) has no room for a
    terminator, and ``pos <= lens`` correctly selects all P positions as content.
    """
    pos = torch.arange(positions, device=lens.device)
    return (pos <= lens.unsqueeze(-1)).to(torch.float32)


def _gather_codec_dot(out: torch.Tensor, labels: torch.Tensor, lens: torch.Tensor,
                      cfg) -> torch.Tensor:
    """<out, codec(token)> without building codec(token).

    The codec has exactly L nonzeros, at flat indices ``byte*P + p``, each equal
    to 1/sqrt(L). So the inner product is a gather of those L entries, summed and
    scaled.
    """
    P = cfg.codec.positions
    pos = torch.arange(P, device=out.device)
    idx = labels * P + pos                                    # (B, T, P)
    picked = out.gather(-1, idx)                              # (B, T, P)
    keep = (labels != PAD_BYTE).to(out.dtype)
    return (picked * keep).sum(-1) * torch.rsqrt(lens.clamp(min=1).to(out.dtype))


def _regress_sq_error(out: torch.Tensor, labels: torch.Tensor, lens: torch.Tensor,
                      cfg) -> torch.Tensor:
    """Per-token squared error against the (never materialised) codec target."""
    dot = _gather_codec_dot(out, labels, lens, cfg)
    return (out * out).sum(-1) - 2.0 * dot + 1.0


# ---------------------------------------------------------------------------
# scoring: everything reduced to bits per byte
# ---------------------------------------------------------------------------


@torch.no_grad()
def token_nats(model: KronLM, in_labels: torch.Tensor, in_lens: torch.Tensor,
               target_labels: torch.Tensor, target_ids: torch.Tensor,
               target_lens: torch.Tensor,
               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Negative log-probability the model assigns to the *correct next token*.

    Returns ``(B, T)`` in nats. This is the one quantity that means the same
    thing for a vocabulary head and a byte head, and it is what bits-per-byte is
    computed from.
    """
    cfg = model.cfg
    # `out` is accepted so evaluation does not run the trunk twice for the same
    # batch -- it already has the head output and only needs it scored.
    if out is None:
        out = model.forward(in_labels, in_lens)

    if cfg.head == "vocab":
        logp = F.log_softmax(out, dim=-1)
        return -logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    if cfg.head == "regress":
        # No normalised distribution over tokens exists here, so a likelihood is
        # not defined. Reported as NaN rather than substituted with something
        # that looks comparable but is not.
        return torch.full(target_ids.shape, float("nan"), device=out.device)

    grid = to_grid_torch(out, cfg.codec)                      # (B, T, P, 256)
    logp = F.log_softmax(grid, dim=-1)
    picked = logp.gather(-1, target_labels.unsqueeze(-1)).squeeze(-1)   # (B, T, P)
    # Only the L content bytes and the one terminating PAD are part of the
    # string; the rest marginalise to 1. Summing all P would not be a likelihood.
    mask = content_mask(target_lens, cfg.codec.positions)
    return -(picked * mask).sum(-1)
