"""Materialising a packed sequence: token ids, loss mask, attention mask, position ids.

The session leaves the position policy explicitly open ("right now we've not discussed
that -- how are we going to be handling a position policy?") and argues that for
cross-document attention, EOS plus back-propagation is sufficient: the model learns not to
attend across a boundary because doing so raises the loss.

That argument is correct and it is also not the only option. Modern packed pretraining
usually blocks the attention outright, so the model *cannot* look across a document
boundary rather than *learning not to*. The two differ in what they cost and what they
risk:

* **EOS-only** -- cheap, and the boundary is a learned soft constraint. Early in training
  the model does attend across boundaries and is penalised for it, which is wasted signal,
  but it also means the mechanism is visible in the loss and can be measured.
* **Block-diagonal** -- the boundary is a hard structural constraint. No wasted signal, no
  possibility of leakage, but the model never learns what a boundary *is*, so it depends on
  the packing being correct forever.

Rather than assert a winner, both are implemented, assigned per lane, and recorded in every
ledger row -- so the choice becomes an A/B the fork machinery can actually run
(`run_demo.py --fork` does exactly this on the web lane).

Where leakage would be unsafe rather than merely wasteful -- agentic trajectories,
reasoning chains, packed source files -- block-diagonal is the default and is not a matter
of taste.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .packing import PackedSequence
from .plan import AGENTIC, CODE, INDIC, LONGCTX, REASONING, STEM, WEB
from .shards import Shard

# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------

ATTN_EOS_ONLY = "causal_eos_only"        # attention may cross a document boundary
ATTN_BLOCK_DIAGONAL = "causal_block_diagonal"  # attention cannot cross a boundary

POS_CONTINUOUS = "continuous"            # 0..L-1 across the whole packed sequence
POS_RESET_PER_DOCUMENT = "reset_per_document"

LANE_ATTENTION: Dict[str, str] = {
    # The lecture's position, kept where the cost of leakage is only wasted signal.
    WEB: ATTN_EOS_ONLY,
    INDIC: ATTN_EOS_ONLY,
    LONGCTX: ATTN_EOS_ONLY,   # one document per sequence anyway
    # Where a packed neighbour would be actively misleading.
    CODE: ATTN_BLOCK_DIAGONAL,
    STEM: ATTN_BLOCK_DIAGONAL,
    REASONING: ATTN_BLOCK_DIAGONAL,
    AGENTIC: ATTN_BLOCK_DIAGONAL,
}

LANE_POSITION: Dict[str, str] = {
    WEB: POS_CONTINUOUS,
    INDIC: POS_CONTINUOUS,
    LONGCTX: POS_CONTINUOUS,
    CODE: POS_RESET_PER_DOCUMENT,
    STEM: POS_RESET_PER_DOCUMENT,
    REASONING: POS_RESET_PER_DOCUMENT,
    AGENTIC: POS_RESET_PER_DOCUMENT,
}

PAD_SEGMENT = -1   # segment id reserved for padding


# ---------------------------------------------------------------------------
# materialised sequence
# ---------------------------------------------------------------------------


@dataclass
class MaterializedSequence:
    input_ids: np.ndarray      # int64  (seq_len,)
    loss_mask: np.ndarray      # uint8  (seq_len,) 1 = this token contributes loss
    segment_ids: np.ndarray    # int32  (seq_len,) document index, PAD_SEGMENT for padding
    position_ids: np.ndarray   # int32  (seq_len,)
    role_codes: np.ndarray     # int8   (seq_len,) index into ROLE_NAMES; 0 = padding
    attention_policy: str
    position_policy: str
    doc_ids: List[str]
    lane: str
    seq_len: int

    @property
    def real_tokens(self) -> int:
        return int((self.segment_ids != PAD_SEGMENT).sum())

    @property
    def supervised_tokens(self) -> int:
        return int(self.loss_mask.sum())

    def stats(self) -> Dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "real_tokens": self.real_tokens,
            "pad_tokens": self.seq_len - self.real_tokens,
            "supervised_tokens": self.supervised_tokens,
            "supervision_yield": round(self.supervised_tokens / max(1, self.real_tokens), 6),
            "documents": len(self.doc_ids),
            "attention_policy": self.attention_policy,
            "position_policy": self.position_policy,
        }


# ---------------------------------------------------------------------------
# roles, carried through to the learning ledger
# ---------------------------------------------------------------------------

# Roles are carried alongside the tokens so that loss can be attributed by role, not just by
# lane. S5's BENCHMARK-MAP measured agentic supervision yield at 356/668 = 53% precisely
# because assistant turns are green and tool observations are grey; without a per-token role
# array the learning ledger could report "agentic loss" but not "loss on the *tool calls*",
# which is the number that says whether the model is learning to act or learning to narrate.
ROLE_NAMES: List[str] = [
    "<pad>", "bos", "eos", "text", "system", "user", "prompt",
    "response", "assistant", "thinking", "tool_call", "tool_result",
]
ROLE_CODES: Dict[str, int] = {name: i for i, name in enumerate(ROLE_NAMES)}
PAD_ROLE = 0


def role_code(role: str) -> int:
    """Unknown roles map to `text` rather than raising: a role the ledger has not seen is a
    reporting gap, not a training fault, and failing the run over it would be the wrong
    trade. New roles show up in `unknown_roles` in the learning ledger instead."""
    return ROLE_CODES.get(role, ROLE_CODES["text"])


# ---------------------------------------------------------------------------
# materialisation
# ---------------------------------------------------------------------------


def materialize(
    seq: PackedSequence,
    shard: Shard,
    pad_id: int,
    *,
    attention_policy: str,
    position_policy: str,
    lane: str = "",
) -> MaterializedSequence:
    """Build the four parallel arrays for one packed sequence.

    The loss mask comes from the shard, where it was computed at tokenization time from the
    document's segment roles. It is never reconstructed here -- a loss mask derived twice
    by two different code paths is a loss mask that will eventually disagree with itself.
    """
    L = seq.seq_len
    input_ids = np.full(L, pad_id, dtype=np.int64)
    loss_mask = np.zeros(L, dtype=np.uint8)
    segment_ids = np.full(L, PAD_SEGMENT, dtype=np.int32)
    position_ids = np.zeros(L, dtype=np.int32)
    role_codes = np.full(L, PAD_ROLE, dtype=np.int8)

    by_id = {d.doc_id: d for d in shard.documents}

    for seg_index, pl in enumerate(seq.placements):
        doc = by_id[pl.doc_id]
        src_lo = doc.start + pl.doc_start
        src_hi = doc.start + pl.doc_end
        dst_lo, dst_hi = pl.seq_start, pl.seq_end

        input_ids[dst_lo:dst_hi] = shard.tokens[src_lo:src_hi]
        loss_mask[dst_lo:dst_hi] = shard.loss_mask[src_lo:src_hi]
        segment_ids[dst_lo:dst_hi] = seg_index

        # Expand the document's run-length-encoded role spans over just the slice that
        # landed in this window. Spans are relative to the document, placements are too.
        for role, r_lo, r_hi in doc.role_spans:
            lo, hi = max(r_lo, pl.doc_start), min(r_hi, pl.doc_end)
            if lo < hi:
                role_codes[dst_lo + (lo - pl.doc_start):dst_lo + (hi - pl.doc_start)] = \
                    role_code(role)

        if position_policy == POS_RESET_PER_DOCUMENT:
            position_ids[dst_lo:dst_hi] = np.arange(pl.doc_start, pl.doc_end, dtype=np.int32)
        elif position_policy == POS_CONTINUOUS:
            position_ids[dst_lo:dst_hi] = np.arange(dst_lo, dst_hi, dtype=np.int32)
        else:
            raise ValueError("unknown position policy {!r}".format(position_policy))

    # Padding never contributes loss. This is separate from the role-based mask: a padded
    # tail is not "masked context", it is not context at all.
    loss_mask[segment_ids == PAD_SEGMENT] = 0

    # Next-token prediction: the final real token of the sequence has no successor inside
    # this window, so it cannot be scored. Dropping it here rather than in the trainer
    # keeps the ledger's supervised-token count honest.
    real = np.flatnonzero(segment_ids != PAD_SEGMENT)
    if real.size:
        loss_mask[real[-1]] = 0

    return MaterializedSequence(
        input_ids=input_ids,
        loss_mask=loss_mask,
        segment_ids=segment_ids,
        position_ids=position_ids,
        role_codes=role_codes,
        attention_policy=attention_policy,
        position_policy=position_policy,
        doc_ids=seq.doc_ids,
        lane=lane,
        seq_len=L,
    )


# ---------------------------------------------------------------------------
# attention masks
# ---------------------------------------------------------------------------


def build_attention_mask(segment_ids: np.ndarray, policy: str) -> np.ndarray:
    """Boolean (L, L) mask, True where query i may attend to key j.

    Always causal. Under the block-diagonal policy, additionally restricted to the query's
    own document. Padding attends to nothing and is attended to by nothing.
    """
    L = segment_ids.shape[0]
    idx = np.arange(L)
    allowed = idx[None, :] <= idx[:, None]           # causal

    real = segment_ids != PAD_SEGMENT
    allowed &= real[None, :]
    allowed &= real[:, None]

    if policy == ATTN_BLOCK_DIAGONAL:
        allowed &= segment_ids[:, None] == segment_ids[None, :]
    elif policy != ATTN_EOS_ONLY:
        raise ValueError("unknown attention policy {!r}".format(policy))

    # A row with no permitted key produces NaN in softmax. Let a fully-masked position
    # attend to itself; its loss is masked out anyway.
    dead = ~allowed.any(axis=1)
    if dead.any():
        allowed[dead, np.flatnonzero(dead)] = True
    return allowed


def batch_attention_mask(segment_ids: np.ndarray, policy: str) -> np.ndarray:
    """(B, L, L) mask for a batch of segment id arrays."""
    return np.stack([build_attention_mask(s, policy) for s in segment_ids])


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def verify_masks(m: MaterializedSequence, shard: Shard, eos_id: int) -> List[str]:
    """Assertions that must hold for every materialised sequence.

    Run on a sample of sequences in the demo; a violation means the model is being taught
    something we did not intend.
    """
    problems: List[str] = []
    L = m.seq_len

    for name, arr in (("input_ids", m.input_ids), ("loss_mask", m.loss_mask),
                      ("segment_ids", m.segment_ids), ("position_ids", m.position_ids)):
        if arr.shape[0] != L:
            problems.append("{} has length {}, expected {}".format(name, arr.shape[0], L))

    pad_positions = m.segment_ids == PAD_SEGMENT
    if m.loss_mask[pad_positions].any():
        problems.append("a padding position is marked supervised")

    if m.position_policy == POS_RESET_PER_DOCUMENT:
        for seg in np.unique(m.segment_ids):
            if seg == PAD_SEGMENT:
                continue
            pos = m.position_ids[m.segment_ids == seg]
            if pos.size and pos[0] != 0:
                problems.append("segment {} does not restart position ids at 0".format(seg))
    elif m.position_policy == POS_CONTINUOUS:
        real = np.flatnonzero(~pad_positions)
        if real.size and not np.array_equal(m.position_ids[real], real.astype(np.int32)):
            problems.append("continuous position ids are not contiguous over real tokens")

    # Every document placed whole must end with EOS, or the boundary signal is missing.
    by_id = {d.doc_id: d for d in shard.documents}
    for seg_index, doc_id in enumerate(m.doc_ids):
        doc = by_id.get(doc_id)
        if doc is None:
            continue
        span = np.flatnonzero(m.segment_ids == seg_index)
        if span.size == 0:
            continue
        if span.size == doc.token_count and int(m.input_ids[span[-1]]) != eos_id:
            problems.append("document {} was placed whole but does not end with EOS".format(doc_id))

    if m.attention_policy == ATTN_BLOCK_DIAGONAL and len(m.doc_ids) > 1:
        mask = build_attention_mask(m.segment_ids, m.attention_policy)
        cross = (m.segment_ids[:, None] != m.segment_ids[None, :]) & mask
        real2 = (~pad_positions)[:, None] & (~pad_positions)[None, :]
        if (cross & real2).any():
            problems.append("block-diagonal mask permits cross-document attention")

    return problems
