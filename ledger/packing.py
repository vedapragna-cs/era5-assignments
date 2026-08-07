"""Packing policies: turning variable-length documents into fixed-shape sequences.

The GPU needs one shape. Natural text does not have one. Everything here is about the cost
of that mismatch, and the cost is real money -- a padding token consumes the same compute
as a real one and teaches the model nothing except that empty is easy to predict.

Six policies, because the correct one depends on the data type:

| policy                | splits documents | concatenates | used for                    |
|-----------------------|------------------|--------------|-----------------------------|
| pad_only              | crops overflow   | no           | nothing (baseline)          |
| concat_chop           | yes              | yes          | nothing (baseline)          |
| greedy                | no               | yes          | general web, Indic          |
| best_fit              | no               | yes          | code, STEM                  |
| structure_preserving  | no               | no           | agentic, reasoning, SFT     |
| long_context          | no               | no           | long-context stage          |

The first two exist to be measured against, not to be used. `compare_policies` runs all six
over the same documents and reports what each wastes -- the session's packing widget,
reproduced as numbers.

**Deferral instead of chopping.** A document marked `indivisible` that does not fit the
current window is *not* cropped; it is returned in `deferred_doc_ids` and left for a stage
whose window is long enough. This is the mechanism behind the long-context reserve: "the
moment you find documents that are more than 4k and 16k, you are lucky, because now you can
actually use it for that stage of training."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .determinism import content_hash, stable_shuffle
from .shards import Shard, ShardDocument

POLICY_PAD_ONLY = "pad_only"
POLICY_CONCAT_CHOP = "concat_chop"
POLICY_GREEDY = "greedy"
POLICY_BEST_FIT = "best_fit"
POLICY_STRUCTURE_PRESERVING = "structure_preserving"
POLICY_LONG_CONTEXT = "long_context"

ALL_POLICIES = (
    POLICY_PAD_ONLY, POLICY_CONCAT_CHOP, POLICY_GREEDY,
    POLICY_BEST_FIT, POLICY_STRUCTURE_PRESERVING, POLICY_LONG_CONTEXT,
)

# Lane -> policy. Justified per lane in README; the short version is that a policy which
# may concatenate unrelated content is unsafe wherever the sequence *is* the unit of
# meaning (a trajectory, a reasoning chain), and a policy which may split a document is
# unsafe wherever the document is a syntactic whole (a source file).
from .plan import AGENTIC, CODE, INDIC, LONGCTX, REASONING, STEM, WEB  # noqa: E402

LANE_POLICY: Dict[str, str] = {
    WEB: POLICY_GREEDY,
    INDIC: POLICY_GREEDY,
    CODE: POLICY_BEST_FIT,
    STEM: POLICY_BEST_FIT,
    REASONING: POLICY_STRUCTURE_PRESERVING,
    AGENTIC: POLICY_STRUCTURE_PRESERVING,
    LONGCTX: POLICY_LONG_CONTEXT,
}


# ---------------------------------------------------------------------------
# plan objects
# ---------------------------------------------------------------------------


@dataclass
class Placement:
    """One document (or a slice of one) placed inside one packed sequence."""

    doc_id: str
    shard_id: str
    doc_start: int      # offset within the document's token array
    doc_end: int
    seq_start: int      # offset within the packed sequence
    seq_end: int
    lane: str
    split: str
    difficulty_band: str
    truncated: bool = False   # the document was cut to fit

    @property
    def length(self) -> int:
        return self.seq_end - self.seq_start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id, "shard_id": self.shard_id,
            "doc_start": self.doc_start, "doc_end": self.doc_end,
            "seq_start": self.seq_start, "seq_end": self.seq_end,
            "lane": self.lane, "split": self.split,
            "difficulty_band": self.difficulty_band, "truncated": self.truncated,
        }


@dataclass
class PackedSequence:
    placements: List[Placement]
    seq_len: int

    @property
    def used(self) -> int:
        return sum(p.length for p in self.placements)

    @property
    def pad(self) -> int:
        return self.seq_len - self.used

    @property
    def doc_ids(self) -> List[str]:
        return [p.doc_id for p in self.placements]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "placements": [p.to_dict() for p in self.placements],
            "seq_len": self.seq_len, "used": self.used, "pad": self.pad,
        }


@dataclass
class PackPlan:
    """A deterministic packing of some documents into fixed-length sequences.

    Hashed, so the ledger can record *which* packing was used without storing the whole
    plan, and replay can assert the recomputed plan is the same one.
    """

    policy: str
    seq_len: int
    sequences: List[PackedSequence]
    deferred_doc_ids: List[str]
    dropped_doc_ids: List[str]
    stats: Dict[str, Any]
    plan_hash: str = ""

    def __post_init__(self) -> None:
        if not self.plan_hash:
            self.plan_hash = content_hash({
                "policy": self.policy,
                "seq_len": self.seq_len,
                "sequences": [s.to_dict() for s in self.sequences],
                "deferred": sorted(self.deferred_doc_ids),
                "dropped": sorted(self.dropped_doc_ids),
            })

    @property
    def n_sequences(self) -> int:
        return len(self.sequences)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def _stats(sequences: Sequence[PackedSequence], seq_len: int, source_tokens: int,
           deferred: Sequence[str], dropped: Sequence[str],
           truncated_tokens: int) -> Dict[str, Any]:
    capacity = len(sequences) * seq_len
    used = sum(s.used for s in sequences)
    pad = capacity - used
    return {
        "sequences": len(sequences),
        "capacity_tokens": capacity,
        "used_tokens": used,
        "pad_tokens": pad,
        "padding_fraction": round(pad / capacity, 6) if capacity else 0.0,
        "packing_efficiency": round(used / capacity, 6) if capacity else 0.0,
        "source_tokens": source_tokens,
        "truncated_tokens": truncated_tokens,
        "deferred_documents": len(deferred),
        "dropped_documents": len(dropped),
        "mean_documents_per_sequence": (
            round(sum(len(s.placements) for s in sequences) / len(sequences), 4)
            if sequences else 0.0
        ),
        "holes": sum(1 for s in sequences if 0 < s.pad),
    }


def _placement(doc: ShardDocument, shard_id: str, doc_start: int, doc_end: int,
               seq_start: int, truncated: bool = False) -> Placement:
    return Placement(
        doc_id=doc.doc_id, shard_id=shard_id,
        doc_start=doc_start, doc_end=doc_end,
        seq_start=seq_start, seq_end=seq_start + (doc_end - doc_start),
        lane=doc.lane, split=doc.split, difficulty_band=doc.difficulty_band,
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------


def _pack_pad_only(docs, shard_id, seq_len, **_):
    """One document per sequence. Overflow is cropped, the remainder is padded.

    The baseline. Maximum wasted compute, and the session's "college level is okay but at
    our level it's not, because every run is going to cost us money".
    """
    seqs, truncated = [], 0
    for d in docs:
        take = min(d.token_count, seq_len)
        truncated += d.token_count - take
        seqs.append(PackedSequence([_placement(d, shard_id, 0, take, 0, take < d.token_count)],
                                   seq_len))
    return seqs, [], [], truncated


def _pack_concat_chop(docs, shard_id, seq_len, **_):
    """Concatenate everything and cut every `seq_len` tokens.

    Zero padding, but documents are split at arbitrary points -- fine for plain
    pretraining prose, destructive for a source file or a trajectory.
    """
    seqs: List[PackedSequence] = []
    cur: List[Placement] = []
    cur_len = 0
    for d in docs:
        off = 0
        while off < d.token_count:
            room = seq_len - cur_len
            take = min(room, d.token_count - off)
            cur.append(_placement(d, shard_id, off, off + take, cur_len,
                                  truncated=(take < d.token_count)))
            cur_len += take
            off += take
            if cur_len == seq_len:
                seqs.append(PackedSequence(cur, seq_len))
                cur, cur_len = [], 0
    if cur:
        seqs.append(PackedSequence(cur, seq_len))
    return seqs, [], [], 0


def _pack_greedy(docs, shard_id, seq_len, **_):
    """First-fit in document order. Fast, preserves order, leaves holes.

    "Greedy is speed. I want to do it as fast as possible." Document order is preserved,
    so a document that does not fit the current sequence opens a new one and the tail of
    the old one becomes padding.

    A document *larger* than the whole window has to be dealt with either way. If it is
    divisible (prose) it is split across consecutive sequences, which is exactly what the
    session permits for pretraining. If it is indivisible (a source file, a trajectory) it
    is deferred to a stage with a wider window rather than damaged.
    """
    seqs, deferred = [], []
    cur: List[Placement] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            seqs.append(PackedSequence(cur, seq_len))
            cur, cur_len = [], 0

    for d in docs:
        if d.token_count > seq_len:
            if d.indivisible or d.atomic:
                deferred.append(d.doc_id)
                continue
            off = 0
            while off < d.token_count:
                if cur_len == seq_len:
                    flush()
                take = min(seq_len - cur_len, d.token_count - off)
                cur.append(_placement(d, shard_id, off, off + take, cur_len, truncated=True))
                cur_len += take
                off += take
            continue
        if cur_len + d.token_count > seq_len:
            flush()
        cur.append(_placement(d, shard_id, 0, d.token_count, cur_len))
        cur_len += d.token_count
    flush()
    return seqs, deferred, [], 0


def _pack_best_fit(docs, shard_id, seq_len, **_):
    """Best-fit-decreasing bin packing. Fewest sequences, tightest geometry.

    "Best fit packing is when you are changing the country -- you spend a lot of time
    thinking, is this the right way, to get the best thing in that 30 kg of payload."

    Sorted by (-token_count, doc_id): the length sort is the algorithm, and the doc_id
    tiebreak makes it independent of input order.
    """
    seqs, deferred = [], []
    order = sorted(docs, key=lambda d: (-d.token_count, d.doc_id))
    bins: List[Tuple[int, List[Placement]]] = []   # (remaining, placements)
    oversized: List[Any] = []

    for d in order:
        if d.token_count > seq_len:
            if d.indivisible or d.atomic:
                deferred.append(d.doc_id)
            else:
                oversized.append(d)
            continue
        # tightest bin that still fits
        best_i, best_rem = -1, seq_len + 1
        for i, (rem, _pl) in enumerate(bins):
            if d.token_count <= rem < best_rem:
                best_i, best_rem = i, rem
        if best_i < 0:
            bins.append((seq_len - d.token_count,
                         [_placement(d, shard_id, 0, d.token_count, 0)]))
        else:
            rem, pl = bins[best_i]
            start = seq_len - rem
            pl.append(_placement(d, shard_id, 0, d.token_count, start))
            bins[best_i] = (rem - d.token_count, pl)

    # Oversized divisible documents are split last, filling the loosest bins first so the
    # split lands in space that would otherwise have been padding.
    for d in oversized:
        off = 0
        while off < d.token_count:
            bins.sort(key=lambda b: (-b[0], b[1][0].doc_id if b[1] else ""))
            if not bins or bins[0][0] == 0:
                bins.insert(0, (seq_len, []))
            rem, pl = bins[0]
            take = min(rem, d.token_count - off)
            pl.append(_placement(d, shard_id, off, off + take, seq_len - rem, truncated=True))
            bins[0] = (rem - take, pl)
            off += take

    for _rem, pl in bins:
        if pl:
            seqs.append(PackedSequence(pl, seq_len))
    return seqs, deferred, [], 0


def _pack_structure_preserving(docs, shard_id, seq_len, **_):
    """One document per sequence, never concatenated, never split.

    For agentic trajectories and reasoning chains, where the sequence *is* the unit of
    meaning. "You will not be adding two traces together." A trace longer than the window
    is deferred, not truncated -- truncating a trajectory at token 256 teaches the model
    that tasks end halfway through.
    """
    seqs, deferred = [], []
    for d in docs:
        if d.token_count > seq_len:
            deferred.append(d.doc_id)
            continue
        seqs.append(PackedSequence([_placement(d, shard_id, 0, d.token_count, 0)], seq_len))
    return seqs, deferred, [], 0


def _pack_long_context(docs, shard_id, seq_len, min_fill: float = 0.5, **_):
    """Long-context stage packing.

    Admits only documents that genuinely need the wide window: a 200-token document in a
    1024-token long-context batch wastes 80% of an expensive step and teaches nothing
    about long-range retrieval. Documents below `min_fill` of the window are deferred back
    to a normal stage, and oversized ones are cropped at a document boundary.
    """
    seqs, deferred = [], []
    threshold = int(seq_len * min_fill)
    truncated = 0
    for d in sorted(docs, key=lambda x: (-x.token_count, x.doc_id)):
        if d.token_count < threshold:
            deferred.append(d.doc_id)
            continue
        take = min(d.token_count, seq_len)
        truncated += d.token_count - take
        seqs.append(PackedSequence(
            [_placement(d, shard_id, 0, take, 0, take < d.token_count)], seq_len))
    return seqs, deferred, [], truncated


_POLICY_FN: Dict[str, Callable] = {
    POLICY_PAD_ONLY: _pack_pad_only,
    POLICY_CONCAT_CHOP: _pack_concat_chop,
    POLICY_GREEDY: _pack_greedy,
    POLICY_BEST_FIT: _pack_best_fit,
    POLICY_STRUCTURE_PRESERVING: _pack_structure_preserving,
    POLICY_LONG_CONTEXT: _pack_long_context,
}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def pack(
    shard: Shard,
    policy: str,
    seq_len: int,
    *,
    doc_ids: Optional[Sequence[str]] = None,
    shuffle_key: Optional[Sequence[Any]] = None,
    min_fill: float = 0.5,
) -> PackPlan:
    """Pack (a subset of) a shard's documents under one policy.

    Deterministic: given the same shard, policy, seq_len and shuffle_key, the returned
    plan and its hash are identical on any machine.
    """
    if policy not in _POLICY_FN:
        raise ValueError("unknown packing policy {!r}".format(policy))

    selected = shard.documents
    if doc_ids is not None:
        wanted = set(doc_ids)
        selected = [d for d in shard.documents if d.doc_id in wanted]

    # Deterministic base order, then an optional keyed shuffle. Never rely on the order
    # documents happened to arrive in.
    selected = sorted(selected, key=lambda d: d.doc_id)
    if shuffle_key is not None:
        selected = stable_shuffle(selected, *shuffle_key)

    if not selected:
        return PackPlan(policy=policy, seq_len=seq_len, sequences=[], deferred_doc_ids=[],
                        dropped_doc_ids=[], stats=_stats([], seq_len, 0, [], [], 0))

    seqs, deferred, dropped, truncated = _POLICY_FN[policy](
        selected, shard.shard_id, seq_len, min_fill=min_fill)

    source_tokens = sum(d.token_count for d in selected)
    stats = _stats(seqs, seq_len, source_tokens, deferred, dropped, truncated)
    stats["policy"] = policy
    stats["shard_id"] = shard.shard_id
    return PackPlan(policy=policy, seq_len=seq_len, sequences=seqs,
                    deferred_doc_ids=sorted(deferred), dropped_doc_ids=sorted(dropped),
                    stats=stats)


def compare_policies(shard: Shard, seq_len: int,
                     policies: Sequence[str] = ALL_POLICIES) -> Dict[str, Any]:
    """Run every policy over the same documents and report what each costs.

    This is the session's packing widget as a table: sequences produced, padding wasted,
    documents split, documents deferred.
    """
    rows = {}
    for p in policies:
        plan = pack(shard, p, seq_len)
        st = dict(plan.stats)
        st["plan_hash"] = plan.plan_hash[:12]
        st["documents_truncated"] = sum(
            1 for s in plan.sequences for pl in s.placements if pl.truncated)
        rows[p] = st
    usable = {p: r for p, r in rows.items() if r["sequences"] > 0}
    return {
        "seq_len": seq_len,
        "shard_id": shard.shard_id,
        "documents": len(shard.documents),
        "source_tokens": int(shard.tokens.shape[0]),
        "policies": rows,
        "best_efficiency": (
            max(usable, key=lambda p: usable[p]["packing_efficiency"]) if usable else None
        ),
        "fewest_sequences": (
            min(usable, key=lambda p: usable[p]["sequences"]) if usable else None
        ),
    }
