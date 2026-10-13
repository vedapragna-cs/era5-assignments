"""Reference model and excess loss.

The learning ledger records how hard each token was for the model *being trained*. That number alone
cannot answer the question V6 actually needs answered, which is not "was this hard?" but "was this
hard in a way more training can fix?"

RHO-1 (arXiv:2404.07965) measured the split directly: over a training run, 51% of tokens are already
learned, 26% are high-loss and genuinely being learned, **11% are high-loss and stay high-loss**
because they are aleatoric - noise, arbitrary identifiers, unpredictable content - and 12% get worse.
The 26% and the 11% are *indistinguishable in a snapshot* and have opposite implications: the first
says collect more data like this, the second says this data cannot be learned and the budget spent on
it is wasted.

Excess loss separates them. Train a reference model on a fixed high-quality set, freeze it, and
score every token with both models:

    excess_loss = loss_trained - loss_reference

- **high loss, high excess**  -> the reference finds it easy and we do not: learnable, and we are
  behind. This is RHO-1's selection signal - the tokens worth spending gradient on.
- **high loss, ~zero excess** -> the reference finds it just as hard: irreducible for this model
  class and data. Collecting more of it buys nothing.
- **low loss, negative excess** -> we are ahead of the reference; the lane is saturating.

Aggregated per lane rather than per token, the same quantity is DoReMi's domain-reweighting signal:
lanes with high mean excess loss are the ones where additional budget still moves the model.

Why the golden proxy set is the reference domain
------------------------------------------------
It is already the high-quality direction OPUS scores against, and it is firewalled out of training
entirely (`build.py` asserts no golden document reaches a trainable sample), so using it here cannot
leak. It is also the only set in this system that is *defined* as "what good looks like".

The caveat is RHO-1's own and it is real: a reference model narrows the signal toward the reference
domain, so a lane the reference has never seen scores as high-excess simply for being unfamiliar.
The honest fix at scale is a per-lane reference; at this corpus size there is not enough data to
train seven of them, so we train one, say so, and report `reference_domain` alongside every number
it produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .canon import chash
from .model import TinyLM
from .packing import PackedSequence


@dataclass
class ReferenceModel:
    """A frozen model trained on the reference domain, used only to score."""
    model: TinyLM
    domain: str
    train_steps: int
    train_sequences: int
    final_mean_loss: float
    weight_hash: str
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def train(cls, golden: list[PackedSequence], vocab_size: int, *,
              seed: int, steps: int = 40, lr: float = 5e-3,
              domain: str = "golden_proxy") -> "ReferenceModel":
        """Fit the reference on the reference domain, then freeze it.

        Deliberately the same architecture and the same optimiser as the trained model. The point of
        excess loss is to hold model class constant so the difference is attributable to DATA; a
        reference of a different size would confound capability with domain.
        """
        m = TinyLM.init(vocab_size, seed=seed + 977)   # different draw, same architecture
        last = 0.0
        for _ in range(steps):
            acc: dict[str, np.ndarray] | None = None
            tot, n_seq = 0.0, 0
            for seq in golden:
                toks = np.array(seq.tokens, dtype=np.int64)
                mask = np.array(seq.loss_mask, dtype=np.int64)
                segs = np.array(seq.segment_ids, dtype=np.int64)
                poss = np.array(seq.position_ids, dtype=np.int64)
                per_pos, n, g = m.forward_backward(toks, mask, segs, poss)
                if not n or g is None:
                    continue
                acc = g if acc is None else {k: acc[k] + g[k] for k in acc}
                tot += float(per_pos.sum()) / n
                n_seq += 1
            if acc is None or not n_seq:
                break
            m.apply({k: v / n_seq for k, v in acc.items()}, lr)
            last = tot / n_seq
        return cls(model=m, domain=domain, train_steps=steps, train_sequences=len(golden),
                   final_mean_loss=round(last, 6),
                   weight_hash=chash(m.state())[:16])

    def per_position_loss(self, seq: PackedSequence) -> np.ndarray:
        """Reference loss at every position of a packed sequence.

        Cached by sample id: the reference is frozen, so this is a pure function of the sequence and
        recomputing it once per exposure would be wasted work (and would make the cost of the
        measurement scale with repetition rather than with corpus size).
        """
        key = seq.sample_id
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        toks = np.array(seq.tokens, dtype=np.int64)
        mask = np.array(seq.loss_mask, dtype=np.int64)
        segs = np.array(seq.segment_ids, dtype=np.int64)
        poss = np.array(seq.position_ids, dtype=np.int64)
        per_pos, _, _ = self.model.forward_backward(toks, mask, segs, poss)
        self._cache[key] = per_pos
        return per_pos

    def to_record(self) -> dict:
        return {
            "reference_domain": self.domain,
            "architecture": "identical to the trained model - excess loss must isolate DATA, so "
                            "model class is held constant",
            "train_steps": self.train_steps,
            "train_sequences": self.train_sequences,
            "final_mean_loss_on_reference_domain": self.final_mean_loss,
            "weight_hash": self.weight_hash,
            "caveat": "a single reference narrows the signal toward its own domain (RHO-1 s5); a "
                      "lane the reference never saw scores high-excess for being unfamiliar rather "
                      "than for being learnable. A per-lane reference is the scale answer and is "
                      "not affordable on this corpus.",
        }


# Excess-loss bands. The thresholds are relative to the reference's own scale rather than absolute,
# because absolute loss depends on vocabulary size and training depth, neither of which is stable
# across the demo's two depths.
def classify(loss: float, reference_loss: float, *, high_loss: float) -> str:
    """RHO-1's four quadrants, expressed through excess loss."""
    excess = loss - reference_loss
    if loss >= high_loss:
        return "learnable_hard" if excess > 0.05 else "irreducible_hard"
    return "saturating" if excess < -0.05 else "already_learned"
