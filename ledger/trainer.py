"""The training loop, and the two ledgers it writes.

The loop itself is ordinary: AdamW, cosine decay with warmup, gradient accumulation over
(rank, accum) slots, gradient clipping. What matters here is what it *records*.

**The consumption ledger** is the outbound half: for every step, which shards were read,
which documents were in the batch, how many tokens each lane contributed, where every lane
cursor stood. It answers "what did the model see, and in what order" without re-running
anything.

**The learning ledger** is the inbound half, and it is the artifact the session says nobody
publishes. Training normally throws away everything except the scalar mean loss. Here the
per-token loss tensor is kept and attributed back through the packing plan to the document,
the shard, the lane, and the role that produced each token. That makes questions answerable
that a loss curve cannot answer:

* Which *shard* is the model finding hard? -> feeds the v6 re-placement advisory.
* Is the model learning where documents end? -> EOS perplexity, tracked separately.
* On agentic data, is loss falling on the tool calls or only on the prose around them?
  -> per-role loss. S5 measured 53% supervision yield on agentic traces; this says whether
  that 53% is the part that is actually improving.
* Which individual tokens are hardest? -> the high-loss tail, kept as exemplars.

Storage is bounded deliberately. Per-token detail is kept in full only for sampled steps
(`token_detail_every`); every other step contributes to the running aggregates. A real run
would write the per-token stream to columnar storage; keeping the whole thing in memory at
demo scale would be dishonest about what the design costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .determinism import content_hash
from .masks import ROLE_NAMES, build_attention_mask
from .model import ModelConfig, TinyLM, masked_mean, token_losses
from .stream import Batch, DataStream, MicroBatch, Sample

# High-loss exemplars retained per step, and overall. Bounded so the ledger stays a ledger
# rather than a copy of the corpus.
TOP_TOKENS_PER_STEP = 3
TOP_TOKENS_KEPT = 40


@dataclass
class TrainConfig:
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.06
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    token_detail_every: int = 10
    eps: float = 1e-8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lr": self.lr, "min_lr_ratio": self.min_lr_ratio,
            "warmup_ratio": self.warmup_ratio, "weight_decay": self.weight_decay,
            "betas": list(self.betas), "grad_clip": self.grad_clip,
            "token_detail_every": self.token_detail_every,
        }


# ---------------------------------------------------------------------------
# tensors
# ---------------------------------------------------------------------------


def batch_tensors(samples: Sequence[Sample], attention_policy: Optional[str] = None
                  ) -> Dict[str, Any]:
    """Turn samples into the tensors the model consumes, including the (B, L, L) mask.

    The attention mask is built per sample from its own declared policy, not from a global
    setting -- the whole point of per-lane packing policy is that a long-context sequence and
    a packed web sequence in the same batch are masked differently.
    """
    import torch

    arrs = {
        "input_ids": np.stack([s.seq.input_ids for s in samples]),
        "loss_mask": np.stack([s.seq.loss_mask for s in samples]),
        "segment_ids": np.stack([s.seq.segment_ids for s in samples]),
        "position_ids": np.stack([s.seq.position_ids for s in samples]),
        "role_codes": np.stack([s.seq.role_codes for s in samples]),
    }
    masks = np.stack([
        build_attention_mask(s.seq.segment_ids, attention_policy or s.seq.attention_policy)
        for s in samples
    ])
    return {
        "input_ids": torch.from_numpy(arrs["input_ids"]).to(torch.long),
        "position_ids": torch.from_numpy(arrs["position_ids"]).to(torch.long),
        "loss_mask": torch.from_numpy(arrs["loss_mask"]).to(torch.float32),
        "segment_ids": torch.from_numpy(arrs["segment_ids"]).to(torch.long),
        "role_codes": torch.from_numpy(arrs["role_codes"]).to(torch.long),
        "attn_mask": torch.from_numpy(masks),
    }


def tensors_from_sequences(sequences) -> Dict[str, Any]:
    """`batch_tensors` for bare MaterializedSequences (the OPUS golden set, which has no
    Sample wrapper because it never travels through the loader)."""
    class _Shim:
        __slots__ = ("seq",)

        def __init__(self, s):
            self.seq = s

    return batch_tensors([_Shim(s) for s in sequences])


def batch_content_hash(batch: Batch) -> str:
    """Hash the tokens and masks the model actually consumed, in micro-batch order."""
    parts: List[Any] = [batch.step, batch.seq_len]
    for mb in batch.micro_batches:
        for s in mb.samples:
            parts.append([
                mb.rank, mb.accum_index, s.lane, s.shard_id,
                int(s.seq.input_ids.sum()), int(s.seq.loss_mask.sum()),
                content_hash(s.seq.input_ids.tolist()),
            ])
    return content_hash(parts)


def batch_id(batch: Batch) -> str:
    """The stable identity of a batch: its step and the content it carried.

    Distinct from `batch_content_hash` only in that it is short and readable. Resume proves
    equality on this, on the spans, and on the full content hash -- three levels, because
    matching ids with different tokens is exactly the failure that would otherwise pass.
    """
    return "b{:06d}-{}".format(batch.step, batch_content_hash(batch)[:12])


def batch_token_spans(batch: Batch) -> List[Dict[str, Any]]:
    """Every token span in the batch, addressed in shard coordinates.

    A span is `(shard_id, doc_id, doc_start, doc_end)` -- an offset range inside an immutable
    shard -- placed at `(seq_start, seq_end)` inside a specific sequence of a specific
    micro-batch. Because shards are immutable and content-addressed, this tuple is a
    permanent address: it resolves to the same tokens on any machine, forever, which is what
    makes replay a lookup rather than a re-derivation.

    `token_hash` covers the actual token values in the span, so a replay that reproduces the
    coordinates but not the content is caught rather than credited.
    """
    spans: List[Dict[str, Any]] = []
    for mb in batch.micro_batches:
        for si, s in enumerate(mb.samples):
            for pl in s.seq.placements_view():
                lo, hi = pl["seq_start"], pl["seq_end"]
                spans.append({
                    "rank": mb.rank, "accum": mb.accum_index, "sample": si,
                    "shard_id": s.shard_id, "doc_id": pl["doc_id"],
                    "doc_start": pl["doc_start"], "doc_end": pl["doc_end"],
                    "seq_start": lo, "seq_end": hi,
                    "tokens": hi - lo,
                    "loss_bearing_tokens": int(s.seq.loss_mask[lo:hi].sum()),
                    "token_hash": content_hash(
                        s.seq.input_ids[lo:hi].tolist())[:16],
                })
    return spans


def spans_hash(spans: Sequence[Dict[str, Any]]) -> str:
    return content_hash([
        [s["rank"], s["accum"], s["sample"], s["shard_id"], s["doc_id"],
         s["doc_start"], s["doc_end"], s["seq_start"], s["seq_end"], s["token_hash"]]
        for s in spans
    ])


# ---------------------------------------------------------------------------
# the learning ledger
# ---------------------------------------------------------------------------


class _Running:
    """Sum and count, so a mean can be reported without keeping every value."""

    __slots__ = ("total", "n", "sq")

    def __init__(self) -> None:
        self.total = 0.0
        self.n = 0
        self.sq = 0.0

    def add(self, total: float, n: int, sq: float = 0.0) -> None:
        self.total += total
        self.n += n
        self.sq += sq

    def mean(self) -> Optional[float]:
        return self.total / self.n if self.n else None

    def std(self) -> Optional[float]:
        if self.n < 2 or self.sq == 0.0:
            return None
        var = max(0.0, self.sq / self.n - (self.total / self.n) ** 2)
        return math.sqrt(var)

    def row(self) -> Dict[str, Any]:
        m = self.mean()
        return {
            "tokens": self.n,
            "mean_loss": round(m, 6) if m is not None else None,
            "perplexity": round(math.exp(min(m, 20.0)), 4) if m is not None else None,
            "std": round(self.std(), 6) if self.std() is not None else None,
        }


class LearningLedger:
    """What came back. Per-token loss, attributed all the way back to its source."""

    def __init__(self, eos_id: int, pad_id: int) -> None:
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.by_shard: Dict[str, _Running] = {}
        self.by_lane: Dict[str, _Running] = {}
        self.by_band: Dict[str, _Running] = {}
        self.by_role: Dict[str, _Running] = {}
        self.by_doc: Dict[str, _Running] = {}
        self.by_lane_stage: Dict[Tuple[str, str], _Running] = {}
        self.eos: _Running = _Running()
        self.first_seen_shard: Dict[str, int] = {}
        self.shard_history: Dict[str, List[Tuple[int, float]]] = {}
        self.sample_rows: List[Dict[str, Any]] = []
        self.token_detail: List[Dict[str, Any]] = []
        self.high_loss: List[Dict[str, Any]] = []
        self.step_rows: List[Dict[str, Any]] = []
        self.unknown_roles: List[str] = []

    # ---- ingest --------------------------------------------------------

    def record(
        self,
        *,
        step: int,
        stage: str,
        samples: Sequence[Sample],
        losses: np.ndarray,     # (B, L-1)
        valid: np.ndarray,      # (B, L-1) bool
        input_ids: np.ndarray,  # (B, L)
        role_codes: np.ndarray, # (B, L)
        tokenizer,
        keep_token_detail: bool,
    ) -> Dict[str, Any]:
        """Attribute one batch's per-token losses back through the packing plan.

        The `[:, 1:]` shift is not cosmetic. Position i of `losses` is the loss of
        *predicting* token i+1, so it must be attributed to token i+1's document, role, and
        shard -- attributing it to token i would credit the prompt token for the response
        token's difficulty, which is exactly the confusion the loss mask exists to prevent.
        """
        step_total, step_n = 0.0, 0
        for b, s in enumerate(samples):
            v = valid[b]
            if not v.any():
                continue
            row_losses = losses[b][v]
            tot = float(row_losses.sum())
            sq = float((row_losses ** 2).sum())
            n = int(v.sum())
            step_total += tot
            step_n += n

            self.by_shard.setdefault(s.shard_id, _Running()).add(tot, n, sq)
            self.by_lane.setdefault(s.lane, _Running()).add(tot, n, sq)
            self.by_band.setdefault(s.difficulty_band, _Running()).add(tot, n, sq)
            self.by_lane_stage.setdefault((s.lane, stage), _Running()).add(tot, n, sq)
            self.first_seen_shard.setdefault(s.shard_id, step)
            self.shard_history.setdefault(s.shard_id, []).append((step, tot / n))

            # Per-document attribution: segment_ids index into the sample's doc list.
            seg = s.seq.segment_ids[1:]
            for j, doc_id in enumerate(s.seq.doc_ids):
                sel = v & (seg == j)
                k = int(sel.sum())
                if k:
                    dl = losses[b][sel]
                    self.by_doc.setdefault(doc_id, _Running()).add(
                        float(dl.sum()), k, float((dl ** 2).sum()))

            # Per-role attribution, and EOS as its own line.
            roles = role_codes[b][1:]
            for code in np.unique(roles[v]):
                sel = v & (roles == code)
                k = int(sel.sum())
                if not k:
                    continue
                rl = losses[b][sel]
                name = ROLE_NAMES[int(code)] if int(code) < len(ROLE_NAMES) else "?"
                self.by_role.setdefault(name, _Running()).add(
                    float(rl.sum()), k, float((rl ** 2).sum()))

            targets = input_ids[b][1:]
            eos_sel = v & (targets == self.eos_id)
            if eos_sel.any():
                el = losses[b][eos_sel]
                self.eos.add(float(el.sum()), int(eos_sel.sum()), float((el ** 2).sum()))

            mean = tot / n
            self.sample_rows.append({
                "step": step, "stage": stage, "lane": s.lane,
                "shard_id": s.shard_id, "band": s.difficulty_band,
                "doc_ids": s.seq.doc_ids, "supervised_tokens": n,
                "mean_loss": round(mean, 6),
                "perplexity": round(math.exp(min(mean, 20.0)), 4),
                "max_token_loss": round(float(row_losses.max()), 6),
            })

            # Hardest tokens, decoded. The exemplars are what make a high-loss shard
            # actionable rather than merely flagged.
            idx = np.flatnonzero(v)
            order = idx[np.argsort(-losses[b][idx])][:TOP_TOKENS_PER_STEP]
            for p in order:
                self.high_loss.append({
                    "step": step, "lane": s.lane, "shard_id": s.shard_id,
                    "position": int(p) + 1,
                    "token": tokenizer.decode([int(input_ids[b][p + 1])]),
                    "role": ROLE_NAMES[int(role_codes[b][p + 1])],
                    "loss": round(float(losses[b][p]), 6),
                })

            if keep_token_detail and b == 0:
                self.token_detail.append({
                    "step": step, "lane": s.lane, "shard_id": s.shard_id,
                    "doc_ids": s.seq.doc_ids,
                    "tokens": [
                        {
                            "pos": int(p) + 1,
                            "token": tokenizer.decode([int(input_ids[b][p + 1])]),
                            "role": ROLE_NAMES[int(role_codes[b][p + 1])],
                            "loss": round(float(losses[b][p]), 5),
                        }
                        for p in np.flatnonzero(v)[:48]
                    ],
                })

        self.high_loss.sort(key=lambda r: -r["loss"])
        del self.high_loss[TOP_TOKENS_KEPT:]

        mean = step_total / step_n if step_n else float("nan")
        return {"mean_loss": mean, "supervised_tokens": step_n}

    # ---- report --------------------------------------------------------

    def shard_table(self) -> List[Dict[str, Any]]:
        rows = []
        for sid, run in sorted(self.by_shard.items()):
            hist = self.shard_history.get(sid, [])
            row = run.row()
            row.update({
                "shard_id": sid,
                "first_seen_step": self.first_seen_shard.get(sid),
                "times_seen": len(hist),
                "first_loss": round(hist[0][1], 6) if hist else None,
                "last_loss": round(hist[-1][1], 6) if hist else None,
                "delta": round(hist[-1][1] - hist[0][1], 6) if len(hist) > 1 else None,
            })
            rows.append(row)
        return sorted(rows, key=lambda r: -(r["mean_loss"] or 0.0))

    def report(self) -> Dict[str, Any]:
        by_lane_stage = {}
        for (lane, stage), run in sorted(self.by_lane_stage.items()):
            by_lane_stage.setdefault(lane, {})[stage] = run.row()
        return {
            "what_this_is": (
                "Token-level and sample-level loss attributed back to shard, document, lane, "
                "band and role. The mean loss is the training signal; everything below it is "
                "the part that is normally discarded."),
            "samples_recorded": len(self.sample_rows),
            "documents_scored": len(self.by_doc),
            "by_lane": {k: v.row() for k, v in sorted(self.by_lane.items())},
            "by_band": {k: v.row() for k, v in sorted(self.by_band.items())},
            "by_band_caveat": (
                "Band perplexity is confounded with time, and badly. The curriculum gates "
                "B0 to the opening steps and B5 to the end, so a band's loss reflects how "
                "old the model was when it saw that band at least as much as how hard the "
                "band is. Expect it to run *backwards* -- lowest numbers on the hardest "
                "bands -- and do not read it as a difficulty measurement. The band figure "
                "is only interpretable within a stage, which is what by_lane_by_stage is "
                "for."),
            "by_role": {k: v.row() for k, v in sorted(self.by_role.items())},
            "by_lane_by_stage": by_lane_stage,
            "eos_prediction": {
                **self.eos.row(),
                "note": ("EOS is the document boundary. If its perplexity does not fall, the "
                         "model is not learning where documents end, and packed sequences "
                         "will bleed one document into the next at inference."),
            },
            "hardest_documents": [
                {"doc_id": d, **self.by_doc[d].row()}
                for d in sorted(self.by_doc, key=lambda x: -(self.by_doc[x].mean() or 0))[:15]
            ],
            "hardest_tokens": self.high_loss[:20],
            "shards": self.shard_table(),
            "token_detail_samples": len(self.token_detail),
        }


# ---------------------------------------------------------------------------
# the consumption ledger
# ---------------------------------------------------------------------------


class ConsumptionLedger:
    """What went out. One row per step, plus running per-lane and per-shard totals."""

    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []
        self.lane_tokens: Dict[str, int] = {}
        self.shard_tokens: Dict[str, int] = {}
        self.doc_exposures: Dict[str, int] = {}
        self.total_tokens = 0
        self.total_supervised = 0

    def record(self, batch: Batch, *, loss: float, lr: float, grad_norm: float,
               cursors: Dict[str, Any], opus_round: Optional[int] = None) -> Dict[str, Any]:
        """Record one step.

        `cursors` is expected to carry both `before` and `after` -- the loader state either
        side of this batch. `after` is the resume point; `before` is what replay needs in
        order to regenerate the batch without guessing. Recording only one of them makes the
        ledger resumable *or* replayable but not both.
        """
        lane_tokens: Dict[str, int] = {}
        shard_ids: Dict[str, int] = {}
        supervised = 0
        for s in batch.samples:
            lane_tokens[s.lane] = lane_tokens.get(s.lane, 0) + s.seq.seq_len
            shard_ids[s.shard_id] = shard_ids.get(s.shard_id, 0) + s.seq.seq_len
            supervised += s.seq.supervised_tokens
            for d in s.seq.doc_ids:
                self.doc_exposures[d] = self.doc_exposures.get(d, 0) + 1

        for k, v in lane_tokens.items():
            self.lane_tokens[k] = self.lane_tokens.get(k, 0) + v
        for k, v in shard_ids.items():
            self.shard_tokens[k] = self.shard_tokens.get(k, 0) + v
        self.total_tokens += batch.token_count()
        self.total_supervised += supervised

        row = {
            "step": batch.step,
            "stage": batch.stage,
            "seq_len": batch.seq_len,
            "in_seam": batch.mixture.in_seam,
            "seam": batch.mixture.seam,
            "tokens": batch.token_count(),
            "supervised_tokens": supervised,
            "cumulative_tokens": self.total_tokens,
            "lane_tokens": dict(sorted(lane_tokens.items())),
            "target_weights": {k: round(v, 6)
                               for k, v in sorted(batch.mixture.target_weights.items())},
            "band_gate": batch.mixture.band_gate,
            "shards_read": sorted(shard_ids),
            "documents": sorted({d for s in batch.samples for d in s.seq.doc_ids}),
            "micro_batches": len(batch.micro_batches),
            # The fingerprint replay checks itself against: the actual token content of the
            # batch, not a description of it. A replay that reproduces the shard list but
            # not the tokens has reproduced the bookkeeping and not the training.
            "batch_id": batch_id(batch),
            "batch_hash": batch_content_hash(batch),
            # Spans are the permanent address of what was consumed: an offset range inside
            # an immutable shard. The hash lets replay assert equality without the ledger
            # having to carry every span for every step.
            "spans_hash": spans_hash(batch_token_spans(batch)),
            "mean_loss": round(loss, 6),
            "perplexity": round(math.exp(min(loss, 20.0)), 4),
            "lr": round(lr, 8),
            "grad_norm": round(grad_norm, 6),
            "cursors": cursors,
        }
        if opus_round is not None:
            row["opus_round"] = opus_round
        self.steps.append(row)
        return row

    def report(self) -> Dict[str, Any]:
        return {
            "what_this_is": (
                "What was sent to the model, step by step: which shards, which documents, "
                "how many tokens per lane, and where every lane cursor stood."),
            "steps": len(self.steps),
            "total_tokens": self.total_tokens,
            "total_supervised_tokens": self.total_supervised,
            "supervision_yield": round(
                self.total_supervised / max(1, self.total_tokens), 6),
            "lane_tokens": dict(sorted(self.lane_tokens.items())),
            "lane_shares": {
                k: round(v / max(1, self.total_tokens), 6)
                for k, v in sorted(self.lane_tokens.items())
            },
            "shards_touched": len(self.shard_tokens),
            "documents_touched": len(self.doc_exposures),
            "most_repeated_documents": sorted(
                ({"doc_id": k, "exposures": v} for k, v in self.doc_exposures.items()),
                key=lambda r: -r["exposures"])[:10],
            "ledger_hash": content_hash([
                [r["step"], r["shards_read"], r["tokens"]] for r in self.steps]),
        }


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


class Trainer:
    def __init__(self, model: TinyLM, cfg: TrainConfig, total_steps: int,
                 tokenizer, *, eos_id: int, pad_id: int) -> None:
        import torch

        self.model = model
        self.cfg = cfg
        self.total_steps = max(1, total_steps)
        self.tokenizer = tokenizer
        self.warmup_steps = max(1, int(self.total_steps * cfg.warmup_ratio))

        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Weight decay on matrices only. Decaying LayerNorm gains and biases toward zero
            # is a well-known way to make a small model quietly worse.
            (no_decay if p.ndim < 2 else decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)

        self.consumption = ConsumptionLedger()
        self.learning = LearningLedger(eos_id=eos_id, pad_id=pad_id)
        self.step_count = 0
        self.preflight: Dict[str, Any] = {}

    # ---- preflight -----------------------------------------------------

    def assert_embeddings_trainable(self) -> Dict[str, Any]:
        """S5 seam rule 1: the embedding table must be trainable at every seam.

        The failure it guards against is silent. A frozen embedding table still trains --
        the loss still falls, the run still finishes -- but any token whose distribution
        changed at the seam is stuck with the representation it had before, and the damage
        only shows up in evaluation on the new mixture.
        """
        emb = self.model.tok_emb.weight
        pos = self.model.pos_emb.weight
        row = {
            "tok_emb_requires_grad": bool(emb.requires_grad),
            "pos_emb_requires_grad": bool(pos.requires_grad),
            "tied_head": self.model.head.weight is emb,
            "in_optimizer": any(
                any(p is emb for p in g["params"]) for g in self.optimizer.param_groups),
        }
        row["ok"] = row["tok_emb_requires_grad"] and row["in_optimizer"]
        if not row["ok"]:
            raise RuntimeError("S5 seam rule 1 violated: embeddings are not trainable")
        self.preflight["embeddings"] = row
        return row

    def check_initial_loss(self, batch: Batch, vocab_size: int) -> Dict[str, Any]:
        """Initial loss must be about ln(V).

        The session's sanity check: an untrained model is uniform over the vocabulary, so
        its cross-entropy is ln(V). A materially lower starting loss means something has
        leaked -- most often a loss mask that is scoring padding, which is trivially
        predictable and drags the mean down.
        """
        import torch

        expected = math.log(vocab_size)
        with torch.no_grad():
            t = batch_tensors(batch.samples)
            logits = self.model(t["input_ids"], t["position_ids"], t["attn_mask"])
            losses, valid = token_losses(logits, t["input_ids"], t["loss_mask"])
            observed = float(masked_mean(losses, valid))
        ratio = observed / expected
        row = {
            "vocab_size": vocab_size,
            "expected_ln_v": round(expected, 6),
            "observed": round(observed, 6),
            "ratio": round(ratio, 4),
            "ok": 0.85 <= ratio <= 1.15,
            "note": ("A ratio well below 1.0 usually means padding or masked context is "
                     "contributing loss; well above 1.0 means initialisation is too wide."),
        }
        self.preflight["initial_loss"] = row
        return row

    # ---- schedule ------------------------------------------------------

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.cfg.lr * (step + 1) / self.warmup_steps
        prog = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        prog = min(1.0, max(0.0, prog))
        floor = self.cfg.lr * self.cfg.min_lr_ratio
        return floor + 0.5 * (self.cfg.lr - floor) * (1 + math.cos(math.pi * prog))

    # ---- one step ------------------------------------------------------

    def train_step(self, batch: Batch, cursors: Dict[str, Any],
                   opus_round: Optional[int] = None,
                   accepted_only: Optional[Sequence[Sample]] = None) -> Dict[str, Any]:
        """One optimizer step: accumulate over every (rank, accum) micro-batch, then update.

        Micro-batches are summed with weight 1/n rather than averaged per micro-batch and
        then averaged again. The distinction matters when micro-batches have different
        supervised-token counts, which under mixed packing policies they routinely do.
        """
        import torch

        lr = self.lr_at(self.step_count)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        keep_detail = (self.step_count % max(1, self.cfg.token_detail_every)) == 0

        groups: List[List[Sample]] = []
        if accepted_only is not None:
            # OPUS steps train only on what survived selection; the rejected samples are
            # still recorded, in the opus trail, as having been considered and dropped.
            groups = [list(accepted_only)] if accepted_only else []
        else:
            groups = [mb.samples for mb in batch.micro_batches if mb.samples]

        if not groups:
            return {"step": batch.step, "skipped": "no_samples"}

        total_loss, total_tokens = 0.0, 0
        n_groups = len(groups)
        for samples in groups:
            t = batch_tensors(samples)
            logits = self.model(t["input_ids"], t["position_ids"], t["attn_mask"])
            losses, valid = token_losses(logits, t["input_ids"], t["loss_mask"])
            loss = masked_mean(losses, valid)
            (loss / n_groups).backward()

            det = self.learning.record(
                step=batch.step, stage=batch.stage, samples=samples,
                losses=losses.detach().numpy(), valid=valid.numpy(),
                input_ids=t["input_ids"].numpy(), role_codes=t["role_codes"].numpy(),
                tokenizer=self.tokenizer, keep_token_detail=keep_detail,
            )
            if det["supervised_tokens"]:
                total_loss += det["mean_loss"] * det["supervised_tokens"]
                total_tokens += det["supervised_tokens"]

        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip))
        self.optimizer.step()
        self.step_count += 1

        mean_loss = total_loss / total_tokens if total_tokens else float("nan")
        row = self.consumption.record(
            batch, loss=mean_loss, lr=lr, grad_norm=grad_norm,
            cursors=cursors, opus_round=opus_round)
        self.learning.step_rows.append({
            "step": batch.step, "stage": batch.stage,
            "mean_loss": round(mean_loss, 6),
            "perplexity": round(math.exp(min(mean_loss, 20.0)), 4),
            "grad_norm": round(grad_norm, 6), "lr": round(lr, 8),
        })
        return row

    # ---- state ---------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_count = state["step_count"]

    def loss_curve(self) -> List[Dict[str, Any]]:
        return self.learning.step_rows
