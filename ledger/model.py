"""A small real transformer.

The assignment says the training loop may be faked -- "just send to a loop and come back".
It is not faked here, and the reason is specific: the artifact the session calls gold is the
**token-level perplexity ledger**, and a ledger of invented numbers demonstrates nothing.
Shard perplexity falling as the model ages, the 1.2 reuse cutoff, perplexity at EOS,
gradient alignment for the OPUS selector -- all of these are only meaningful if a real
model really produced them.

So the model is small enough to train in a demo and real enough that its losses mean
something: a decoder-only pre-norm transformer with learned position embeddings, GELU MLP,
weight tying, and an attention implementation that accepts an explicit (B, L, L) boolean
mask so the block-diagonal packing policy is actually enforced in the arithmetic rather
than merely recorded in the manifest.

Learned absolute position embeddings rather than RoPE, deliberately: the position policy is
one of the axes this system exists to measure, and an explicit position-id input makes
`reset_per_document` a one-line change instead of a rotation-cache rewrite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 1024
    dropout: float = 0.0        # zero: dropout is a second RNG stream to keep deterministic
    tie_embeddings: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size, "d_model": self.d_model,
            "n_layers": self.n_layers, "n_heads": self.n_heads, "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len, "dropout": self.dropout,
            "tie_embeddings": self.tie_embeddings,
        }


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            # attn_mask is (B, L, L), True where attention is permitted.
            scores = scores.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def n_params(self) -> int:
        # Tied weights are one tensor counted once.
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, L) int64
        position_ids: torch.Tensor,     # (B, L) int64
        attn_mask: Optional[torch.Tensor] = None,   # (B, L, L) bool
    ) -> torch.Tensor:
        x = self.tok_emb(input_ids) + self.pos_emb(position_ids)
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.head(self.ln_f(x))


def token_losses(
    logits: torch.Tensor,       # (B, L, V)
    input_ids: torch.Tensor,    # (B, L)
    loss_mask: torch.Tensor,    # (B, L) float or bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token next-token loss, and the mask of positions that count.

    Returns `(losses, valid)` both shaped (B, L-1). Keeping the per-token tensor rather than
    only its mean is the whole point: the mean is the training signal, the per-token values
    are the learning ledger.
    """
    pred = logits[:, :-1, :]
    target = input_ids[:, 1:]
    valid = loss_mask[:, 1:].to(torch.bool)

    flat = F.cross_entropy(
        pred.reshape(-1, pred.size(-1)),
        target.reshape(-1),
        reduction="none",
    ).view(target.shape)
    return flat, valid


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    n = valid.sum()
    if n == 0:
        return values.sum() * 0.0
    return (values * valid.to(values.dtype)).sum() / n
