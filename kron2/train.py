"""Training and evaluation for the four arms.

Everything reported here reduces to quantities that mean the same thing across a
vocabulary head and a byte head. The training losses do not: token cross-entropy
and byte cross-entropy are different units, and comparing them directly would be
meaningless. So the headline metric is **bits per byte**, computed from the
negative log-probability each arm assigns to the *correct next token*, divided by
that token's length in bytes.

The ``regress`` arm has no normalised distribution over tokens at all, so a
likelihood is undefined for it. It is reported as N/A rather than given a
substitute number that would look comparable and not be.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .codec import DEFAULT as DEFAULT_CODEC
from .codec import PAD_BYTE, decode_labels, to_grid_torch
from .model import HEADS, KronLM, ModelConfig, content_mask, token_nats

LN2 = math.log(2.0)


def pick_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# nearest-neighbour decoding, for the regress arm
# ---------------------------------------------------------------------------


def _chunk_for(n_rows: int, positions: int, budget: int = 32_000_000) -> int:
    """Vocabulary chunk size that keeps the gather temporary bounded.

    Both vocabulary sweeps build an ``(N, C, P)`` intermediate. At the real eval
    shape -- N = 16*256 predictions, P = 32 -- a chunk of 2048 candidates is 1.07
    GB in float32, which on MPS means thrashing or an allocation failure. Sizing
    the chunk from N instead keeps it near ``budget`` elements regardless of the
    batch shape.
    """
    return max(32, min(4096, budget // max(1, n_rows * positions)))


@torch.no_grad()
def nearest_token(pred: torch.Tensor, stream, cfg, device: str,
                  chunk: Optional[int] = None) -> torch.Tensor:
    """Nearest vocabulary token to each predicted codec vector.

    Never materialises the (V, D) codec table -- at V=16,384 and D=8,192 that is
    537 MB in float32, to hold at most 32 nonzeros per row. Instead the squared
    distance is expanded,

        |p - k|^2 = |p|^2 - 2<p, k> + |k|^2      with |k|^2 = 1

    and ``<p, k>`` is a gather of the candidate's L byte positions. ``|p|^2`` is
    constant across candidates and drops out of the argmin, so only the inner
    product is computed, in chunks over the vocabulary.
    """
    P = cfg.codec.positions
    labels = torch.from_numpy(stream.labels).to(device)        # (V, P)
    lens = torch.from_numpy(stream.lens).to(device)            # (V,)
    pos = torch.arange(P, device=device)
    chunk = chunk or _chunk_for(pred.shape[0], P)

    best_score = torch.full((pred.shape[0],), -float("inf"), device=device)
    best_id = torch.zeros(pred.shape[0], dtype=torch.long, device=device)

    for lo in range(0, labels.shape[0], chunk):
        hi = min(lo + chunk, labels.shape[0])
        idx = (labels[lo:hi] * P + pos)                        # (C, P)
        keep = (labels[lo:hi] != PAD_BYTE).to(pred.dtype)      # (C, P)
        scale = torch.rsqrt(lens[lo:hi].clamp(min=1).to(pred.dtype))

        # (N, C): inner product of every prediction with every candidate
        gathered = pred[:, idx.reshape(-1)].reshape(pred.shape[0], hi - lo, P)
        score = (gathered * keep).sum(-1) * scale

        top, arg = score.max(dim=-1)
        better = top > best_score
        best_score = torch.where(better, top, best_score)
        best_id = torch.where(better, arg + lo, best_id)

    return best_id


@torch.no_grad()
def vocab_restricted_logz(logp: torch.Tensor, stream, cfg, device: str,
                          chunk: Optional[int] = None) -> torch.Tensor:
    """``log sum_v P(v)`` over the real vocabulary, under the byte head.

    The byte head's distribution ranges over every byte string, not just the
    16,384 that are tokens. So its bits-per-byte carries a cost the vocabulary
    head never pays: mass spent ruling out strings that could never occur.

    Restricting to the vocabulary and renormalising separates the two causes of
    the gap. ``bpb`` is what the head actually costs; ``bpb_vocab_restricted`` is
    what it would cost if it were told, as the baseline effectively is, that the
    answer is one of |V| options. The difference between them is the price of
    vocabulary-freedom, stated as a number instead of a caveat.

    ``logp`` is ``(N, P, 256)`` log-softmax over bytes. Returns ``(N,)``.
    """
    P = cfg.codec.positions
    labels = torch.from_numpy(stream.labels).to(device)        # (V, P)
    lens = torch.from_numpy(stream.lens).to(device)            # (V,)
    N = logp.shape[0]
    flat = logp.reshape(N, -1)                                 # (N, P*256)
    chunk = chunk or _chunk_for(N, P)

    parts = []
    for lo in range(0, labels.shape[0], chunk):
        hi = min(lo + chunk, labels.shape[0])
        lab = labels[lo:hi]                                    # (C, P)
        # index into the (P, 256) grid flattened as p*256 + byte
        idx = (torch.arange(P, device=device) * cfg.codec.n_bytes + lab)
        m = content_mask(lens[lo:hi], P)                       # (C, P)
        picked = flat[:, idx.reshape(-1)].reshape(N, hi - lo, P)
        parts.append((picked * m).sum(-1))                     # (N, C)
    return torch.logsumexp(torch.cat(parts, dim=1), dim=1)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


class _Bucket:
    """Running totals for one slice of the validation set."""

    def __init__(self):
        self.nats = 0.0
        self.bytes = 0
        self.tokens = 0
        self.correct = 0
        self.right_length = 0
        self.nats_restricted = 0.0
        self.restricted_bytes = 0

    def report(self, head: str) -> Dict:
        out = {
            "tokens": self.tokens,
            "bytes": self.bytes,
            "exact_token_accuracy": round(self.correct / max(1, self.tokens), 6),
        }
        if head == "regress":
            out["bits_per_byte"] = None
        else:
            out["bits_per_byte"] = round(self.nats / max(1, self.bytes) / LN2, 6)
            out["nats_per_token"] = round(self.nats / max(1, self.tokens), 6)
        if head in ("kron", "kron_tied"):
            out["length_accuracy"] = round(self.right_length / max(1, self.tokens), 6)
            if self.restricted_bytes:
                # Divided by the bytes it was actually computed over, not by the
                # full sweep's -- the two cover different numbers of batches.
                out["bits_per_byte_vocab_restricted"] = round(
                    self.nats_restricted / self.restricted_bytes / LN2, 6)
                out["vocab_restricted_bytes_scored"] = self.restricted_bytes
        return out


@torch.no_grad()
def evaluate(model: KronLM, stream, device: str, max_batches: Optional[int] = 40,
             batch_size: int = 16, restricted: bool = False,
             restricted_batches: int = 24) -> Dict:
    """Validation metrics, swept sequentially so every token is scored once.

    Reported three ways: over everything, over the tokens each arm was trained to
    predict (``seen``), and over the withheld tokens no arm was ever trained to
    predict (``heldout``). The last slice is the vocabulary-freedom measurement,
    and it is empty unless `stream.set_heldout` was called.
    """
    model.eval()
    cfg = model.cfg
    P = cfg.codec.positions

    allb, seen, held = _Bucket(), _Bucket(), _Bucket()
    margin_sum, margin_n = 0.0, 0

    for i, b in enumerate(stream.sequential_batches("val", batch_size, device)):
        if max_batches is not None and i >= max_batches:
            break
        tgt_lens = b["tgt_lens"]
        is_held = b["tgt_heldout"]

        out = model.forward(b["in_labels"], b["in_lens"])

        if cfg.head == "vocab":
            ok = out.argmax(-1) == b["tgt_ids"]
            right_len = torch.zeros_like(ok)

        elif cfg.head in ("kron", "kron_tied"):
            grid = to_grid_torch(out, cfg.codec)              # (B, T, P, 256)
            pred = grid.argmax(-1)                            # (B, T, P)

            # The head emits bytes until PAD, so the decoded string is whatever
            # precedes the first PAD. Everything is judged on that string, not on
            # the raw 32 slots -- comparing all P labels would credit the model
            # for agreeing about padding it was never asked to predict.
            is_pad = pred == PAD_BYTE
            first_pad = torch.where(
                is_pad.any(-1), is_pad.to(torch.int64).argmax(-1),
                torch.full_like(tgt_lens, P))                 # (B, T)

            right_len = first_pad == tgt_lens
            content = torch.arange(P, device=device) < tgt_lens.unsqueeze(-1)
            bytes_ok = ((pred == b["tgt_labels"]) | ~content).all(-1)
            ok = right_len & bytes_ok        # same length *and* same bytes

            # Margin over the positions that are part of the string. The tail is
            # unsupervised by construction, so its margin means nothing.
            scored = torch.arange(P, device=device) <= tgt_lens.unsqueeze(-1)
            top2 = grid.topk(2, dim=-1).values
            margin_sum += float(((top2[..., 0] - top2[..., 1]) * scored).sum().item())
            margin_n += int(scored.sum().item())

        else:  # regress: decode by nearest neighbour over the vocabulary
            nn_ids = nearest_token(out.reshape(-1, cfg.codec.dim), stream, cfg, device)
            ok = (nn_ids.reshape(b["tgt_ids"].shape) == b["tgt_ids"])
            right_len = torch.zeros_like(ok)

        if cfg.head != "regress":
            nats = token_nats(model, b["in_labels"], b["in_lens"],
                              b["tgt_labels"], b["tgt_ids"], tgt_lens, out=out)
        else:
            nats = torch.zeros_like(tgt_lens, dtype=torch.float32)

        # The same score, renormalised over the real vocabulary. Expensive
        # (|V| candidates per position), so only on the final sweep.
        nats_r = torch.zeros_like(nats)
        do_restricted = (restricted and cfg.head in ("kron", "kron_tied")
                         and i < restricted_batches)
        if do_restricted:
            lp = F.log_softmax(to_grid_torch(out, cfg.codec), dim=-1)
            logz = vocab_restricted_logz(
                lp.reshape(-1, P, cfg.codec.n_bytes), stream, cfg, device)
            nats_r = nats + logz.reshape(nats.shape)

        for bucket, sel in ((allb, None), (seen, ~is_held), (held, is_held)):
            m = torch.ones_like(is_held) if sel is None else sel
            bucket.tokens += int(m.sum().item())
            bucket.bytes += int((tgt_lens * m).sum().item())
            bucket.nats += float((nats * m).sum().item())
            bucket.correct += int((ok & m).sum().item())
            bucket.right_length += int((right_len & m).sum().item())
            if do_restricted:
                bucket.nats_restricted += float((nats_r * m).sum().item())
                bucket.restricted_bytes += int((tgt_lens * m).sum().item())

    model.train()
    rep: Dict = dict(allb.report(cfg.head))
    rep["full_val_sweep"] = max_batches is None
    rep["seen"] = seen.report(cfg.head)
    rep["heldout"] = held.report(cfg.head)
    if cfg.head == "regress":
        rep["note"] = ("no normalised distribution over tokens exists for an MSE "
                       "regression head, so a likelihood is undefined")
    if margin_n:
        rep["decode_margin_mean"] = round(margin_sum / margin_n, 6)
    return rep


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_arm(head: str, stream, *, steps: int = 1500, batch_size: int = 8,
              lr: float = 3e-4, warmup: float = 0.05, seed: int = 0,
              device: Optional[str] = None, eval_every: int = 250,
              model_cfg: Optional[Dict] = None, log=print,
              checkpoint: Optional[str] = None) -> Dict:
    device = pick_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # seq_len and vocab_size are properties of the data, so they are taken from
    # the stream; an explicit model_cfg may still override anything else.
    kw = dict(model_cfg or {})
    kw.pop("seq_len", None)
    kw.pop("vocab_size", None)
    cfg = ModelConfig(head=head, vocab_size=stream.vocab_size,
                      seq_len=stream.seq_len, **kw)
    model = KronLM(cfg).to(device)
    params = model.param_counts()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    n_warm = max(1, int(steps * warmup))

    def lr_at(s):
        if s < n_warm:
            return lr * (s + 1) / n_warm
        prog = (s - n_warm) / max(1, steps - n_warm)
        return 0.1 * lr + 0.45 * lr * (1 + math.cos(math.pi * min(1.0, prog)))

    history: List[Dict] = []
    t0 = time.time()
    it = stream.batches("train", batch_size, steps, seed=seed, device=device)

    for step, b in enumerate(it):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        # Withheld targets get weight 0, identically for every arm. This is what
        # makes the held-out slice of the evaluation mean anything.
        w = (~b["tgt_heldout"]).to(torch.float32)
        loss, parts = model.loss(b["in_labels"], b["in_lens"], b["tgt_labels"],
                                 b["tgt_ids"], b["tgt_lens"], weights=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()

        if (step + 1) % eval_every == 0 or step == steps - 1:
            # The last evaluation sweeps the whole validation set. Intermediate
            # ones are capped for speed, which at 1% coverage leaves only ~800
            # held-out targets -- too few to quote a rate from.
            last = step == steps - 1
            ev = evaluate(model, stream, device,
                          max_batches=None if last else 40, restricted=last)
            row = {"step": step + 1, "train_loss": round(float(loss.item()), 6),
                   "grad_norm": round(gn, 4), "elapsed_s": round(time.time() - t0, 1),
                   **ev}
            history.append(row)

            def fmt(x):
                return "  n/a " if x is None else "{:6.3f}".format(x)
            log("  [{:>5}] {:<10} loss {:7.4f} | bpb all {} seen {} held {}"
                " | exact {:.4f} held {:.4f}".format(
                    step + 1, head, row["train_loss"],
                    fmt(row["bits_per_byte"]), fmt(ev["seen"]["bits_per_byte"]),
                    fmt(ev["heldout"]["bits_per_byte"]),
                    row["exact_token_accuracy"],
                    ev["heldout"]["exact_token_accuracy"]))

    if checkpoint:
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        torch.save({"cfg": cfg.to_dict(), "state_dict": model.state_dict()}, checkpoint)

    return {
        "head": head,
        "config": cfg.to_dict(),
        "checkpoint": checkpoint,
        "params": params,
        "device": device,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "wall_clock_s": round(time.time() - t0, 1),
        "history": history,
        "final": history[-1] if history else {},
        "model": model,
    }
