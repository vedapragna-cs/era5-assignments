"""Packing: variable-length documents into fixed-length training windows.

Six policies, each genuinely distinct. The course simulator aliases three of them onto the same
best-fit search (measured: greedy and best-fit were identical in 10 of 12 swept cells) and its
"concat and chop" closes a bin before an overflowing document so nothing is ever truncated. Ours
truncates for real and first-fits for real, because the whole point of comparing policies is that
they behave differently.

Every packed sequence carries four parallel arrays of equal length:

    tokens        int   the token ids
    loss_mask     0/1   does this position contribute to the gradient
    segment_ids   int   which document this position belongs to (attention isolation)
    position_ids  int   position WITHIN the document, reset at every boundary

`segment_ids` is what makes concatenation safe. EOS tells the model a document ended, but EOS is a
learned signal; the segment mask is a structural one. Attention is computed as
`allowed[i][j] = (j <= i) and (segment_ids[j] == segment_ids[i])`, so a token physically cannot
attend across a document boundary. Both are used: EOS so the model learns the boundary convention,
segments so a packing decision can never leak context between unrelated samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .canon import chash
from .corpus import CONTEXT_ROLES, SUPERVISED_ROLES, VERIFIER_ROLES
from .shards import Shard, TokenSpan
from .tokenizer import EOS, PAD

POLICIES = ("pad_only", "concat_chop", "greedy", "best_fit", "structure_preserving", "long_context")


@dataclass
class PackedSequence:
    tokens: list[int]
    loss_mask: list[int]
    segment_ids: list[int]
    position_ids: list[int]
    members: list[str]                 # doc ids, in order
    span_ids: list[str]
    lane: str
    policy: str
    truncated: list[str] = field(default_factory=list)   # doc ids cut by this window

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def n_pad(self) -> int:
        return sum(1 for t in self.tokens if t == PAD)

    @property
    def n_loss_bearing(self) -> int:
        return sum(self.loss_mask)

    @property
    def utilization(self) -> float:
        return (self.length - self.n_pad) / self.length if self.length else 0.0

    @property
    def sample_id(self) -> str:
        return chash({"tokens": self.tokens, "loss_mask": self.loss_mask,
                      "segment_ids": self.segment_ids, "position_ids": self.position_ids})[:16]

    @property
    def loss_mask_hash(self) -> str:
        return chash(self.loss_mask)

    def to_record(self) -> dict:
        return {"sample_id": self.sample_id, "lane": self.lane, "policy": self.policy,
                "length": self.length, "n_pad": self.n_pad,
                "n_loss_bearing": self.n_loss_bearing,
                "utilization": round(self.utilization, 4),
                "members": self.members, "span_ids": self.span_ids,
                "truncated": self.truncated, "loss_mask_hash": self.loss_mask_hash}


def role_is_supervised(role: str) -> bool:
    """S5's agentic mask, applied uniformly.

    supervised: assistant_plan, tool_call, final_answer, text  (a FAILED tool_call is supervised -
                recovery is the taught behaviour)
    context   : user, tool_observation, repo_file, shell_output
    verifier  : verifier_outcome  (recorded, not trained on)
    boundary  : EOS is supervised in pretraining - the model must learn to emit it
    """
    if role in VERIFIER_ROLES:
        return False
    if role in CONTEXT_ROLES:
        return False
    if role == "boundary":
        return True
    return role in SUPERVISED_ROLES


@dataclass
class DocUnit:
    """One document's tokens plus its per-token loss mask, ready to place in a window."""
    doc_id: str
    span_id: str
    lane: str
    tokens: list[int]
    loss_mask: list[int]

    @property
    def length(self) -> int:
        return len(self.tokens)


# Trainer-side repeated-n-gram safeguard, the second of OLMo 2's two interventions
# (arXiv:2501.00656 s3.1): "an additional safeguard in the trainer that detects these sequences
# during data loading and masks them when computing the loss."
#
# Curation already excludes flagged documents (tds/build.py), so this is defence in depth - the same
# two-line structure the eval firewall uses, and for the same reason: a curation pass is a promise
# about a corpus, while this is a property of the exact tokens about to produce a gradient. It also
# covers the case curation cannot: a repeated run that only APPEARS once documents are packed
# together, or one introduced by a tokenizer edge case after the text-level screen ran.
REPEAT_TOKEN_RUN = 32
REPEAT_TOKEN_MAX_N = 13


def mask_repeated_ngrams(tokens: list[int], mask: list[int]) -> tuple[list[int], int]:
    """Zero the loss mask over any run of >=32 repeats of a 1..13-token span.

    Returns (new_mask, n_positions_masked). The tokens are left untouched: the model still SEES the
    degenerate span as context, it simply is not asked to predict it. That is OLMo 2's choice and it
    is the conservative one - deleting tokens would change every downstream offset and break the
    span ids the ledger uses to prove what was consumed.
    """
    from .clean import repeated_ngram_run
    run, n, start = repeated_ngram_run(tokens)
    if run < REPEAT_TOKEN_RUN or n > REPEAT_TOKEN_MAX_N:
        return mask, 0
    out = list(mask)
    masked = 0
    for i in range(start, min(start + run * n, len(out))):
        if out[i]:
            masked += 1
        out[i] = 0
    return out, masked


def units_from_shard(shard: Shard) -> list[DocUnit]:
    units: list[DocUnit] = []
    for span in shard.spans:
        toks = shard.tokens[span.start:span.end]
        mask = [0] * len(toks)
        for role, s, e in span.roles:
            sup = 1 if role_is_supervised(role) else 0
            for i in range(s - span.start, min(e - span.start, len(mask))):
                mask[i] = sup
        mask, _ = mask_repeated_ngrams(toks, mask)
        units.append(DocUnit(span.doc_id, span.span_id, span.lane, toks, mask))
    return units


# No module-level tally of safeguard firings: a mutable global would accumulate across the E1 arms
# and the two determinism runs, and a counter that depends on how many times the process re-built a
# corpus is not a measurement. In the demo the safeguard never fires anyway, because curation
# already removed the flagged document - so its evidence is a negative control that disables the
# curation filter and shows the mask still catches it (tds/controls.py), plus a direct unit test.
# "Never fired" and "never wired up" are separated by the control, not by a log line.


def _finish(tokens, masks, segs, members, span_ids, lane, policy, ctx, truncated) -> PackedSequence:
    """Pad to the window and build position ids that reset at every segment boundary."""
    pad_n = ctx - len(tokens)
    if pad_n < 0:
        raise AssertionError(f"packed sequence exceeds context: {len(tokens)} > {ctx}")
    tokens = tokens + [PAD] * pad_n
    masks = masks + [0] * pad_n          # padding is NEVER loss-bearing
    segs = segs + [-1] * pad_n           # padding belongs to no segment
    # A position may only be supervised if its predecessor is in the SAME document: predicting the
    # first token of document N+1 from the last token of document N is cross-document supervision,
    # and it is a different failure from cross-document attention. Measured before this: 14 of 5138
    # loss-bearing positions sat at a segment start, and nothing in the graded pipeline noticed -
    # the only thing preventing it was a single condition inside the model's forward pass.
    for _i in range(len(masks)):
        if masks[_i] and (_i == 0 or segs[_i] != segs[_i - 1] or segs[_i] < 0):
            masks[_i] = 0
    positions, last, counter = [], None, 0
    for s in segs:
        counter = 0 if s != last else counter + 1
        positions.append(counter)
        last = s
    return PackedSequence(tokens, masks, segs, positions, members, span_ids, lane, policy, truncated)


def pack(units: list[DocUnit], context_length: int, policy: str, lane: str) -> list[PackedSequence]:
    if policy not in POLICIES:
        raise ValueError(f"unknown packing policy {policy!r}")
    ctx = context_length
    out: list[PackedSequence] = []

    if policy == "pad_only":
        for u in units:
            t, m = u.tokens[:ctx], u.loss_mask[:ctx]
            trunc = [u.doc_id] if u.length > ctx else []
            out.append(_finish(list(t), list(m), [0] * len(t), [u.doc_id], [u.span_id],
                               lane, policy, ctx, trunc))
        return out

    if policy == "long_context":
        # One document owns the window. Documents longer than the window are split into consecutive
        # windows rather than discarded - long-context data is the scarcest thing we have.
        for u in units:
            for off in range(0, max(1, u.length), ctx):
                t, m = u.tokens[off:off + ctx], u.loss_mask[off:off + ctx]
                if not t:
                    break
                out.append(_finish(list(t), list(m), [0] * len(t), [u.doc_id], [u.span_id],
                                   lane, policy, ctx, []))
        return out

    if policy == "structure_preserving":
        # Never mix samples, and never CONTINUE one into a second window either. This is the real
        # difference from `long_context`, which an earlier version did not implement - the two were
        # byte-identical, so "six distinct policies" was really five.
        #
        # An agentic trajectory split mid-trace produces a second window whose supervised spans
        # (assistant plan, tool call, final answer) have no preceding user turn or tool observation.
        # The mask would still be structurally valid and the model would be trained to emit a tool
        # call it was never shown the reason for. So an over-long sample is TRUNCATED at the window
        # and the loss recorded, rather than silently continued.
        # Pack COMPLETE samples together, first-fit, and never split one. That is the difference
        # from `pad_only` (one sample per window regardless, so it wastes the remainder) and from
        # `best_fit`/`concat_chop` (which continue an over-long sample into the next window).
        bins: list[list[DocUnit]] = []
        used: list[int] = []
        trunc_ids: list[list[str]] = []
        for u in units:
            t, m = list(u.tokens[:ctx]), list(u.loss_mask[:ctx])
            cut = [u.span_id] if u.length > ctx else []
            unit = DocUnit(u.doc_id, u.span_id, u.lane, t, m)
            placed = False
            for i, cap in enumerate(used):
                if cap + len(t) <= ctx:
                    bins[i].append(unit)
                    used[i] += len(t)
                    trunc_ids[i].extend(cut)
                    placed = True
                    break
            if not placed:
                bins.append([unit])
                used.append(len(t))
                trunc_ids.append(list(cut))
        for b, tr in zip(bins, trunc_ids):
            tokens: list[int] = []
            masks: list[int] = []
            segs: list[int] = []
            members: list[str] = []
            span_ids: list[str] = []
            for seg, u in enumerate(b):
                tokens.extend(u.tokens)
                masks.extend(u.loss_mask)
                segs.extend([seg] * len(u.tokens))
                members.append(u.doc_id)
                span_ids.append(u.span_id)
            out.append(_finish(tokens, masks, segs, members, span_ids, lane, policy, ctx,
                               sorted(set(tr))))
        return out

    if policy == "concat_chop":
        # Join everything into one stream and cut fixed windows. Documents ARE split across window
        # boundaries; segment ids keep the halves from attending to their neighbours.
        stream_t: list[int] = []
        stream_m: list[int] = []
        stream_owner: list[str] = []
        stream_span: list[str] = []
        for u in units:
            stream_t.extend(u.tokens)
            stream_m.extend(u.loss_mask)
            stream_owner.extend([u.doc_id] * u.length)
            stream_span.extend([u.span_id] * u.length)
        for off in range(0, len(stream_t), ctx):
            t = stream_t[off:off + ctx]
            m = stream_m[off:off + ctx]
            owners = stream_owner[off:off + ctx]
            spans = stream_span[off:off + ctx]
            segs, members, span_ids, seg = [], [], [], -1
            last_owner = None
            for o in owners:
                if o != last_owner:
                    seg += 1
                    members.append(o)
                    last_owner = o
                segs.append(seg)
            for s in spans:
                if s not in span_ids:
                    span_ids.append(s)
            # a document is truncated if it continues past this window or started before it
            trunc = []
            if off > 0 and stream_owner[off - 1] == owners[0]:
                trunc.append(owners[0])
            if off + ctx < len(stream_owner) and stream_owner[off + ctx] == owners[-1]:
                trunc.append(owners[-1])
            out.append(_finish(list(t), list(m), segs, members, span_ids, lane, policy, ctx,
                               sorted(set(trunc))))
        return out

    # ---- bin-packing policies -------------------------------------------------------------
    if policy == "greedy":
        order = list(units)                                   # incoming order preserved
        first_fit = True
    else:                                                     # best_fit
        order = sorted(units, key=lambda u: -u.length)        # largest first
        first_fit = False

    bins: list[list[DocUnit]] = []
    used: list[int] = []
    for u in order:
        if u.length > ctx:
            # too long for any window: give it its own windows rather than dropping it
            for off in range(0, u.length, ctx):
                bins.append([DocUnit(u.doc_id, u.span_id, u.lane,
                                     u.tokens[off:off + ctx], u.loss_mask[off:off + ctx])])
                used.append(len(bins[-1][0].tokens))
            continue
        target = None
        if first_fit:
            for i, cap in enumerate(used):                    # FIRST bin with room
                if cap + u.length <= ctx:
                    target = i
                    break
        else:
            best_left = None                                  # TIGHTEST remaining gap
            for i, cap in enumerate(used):
                left = ctx - cap - u.length
                if left >= 0 and (best_left is None or left < best_left):
                    best_left, target = left, i
        if target is None:
            bins.append([u])
            used.append(u.length)
        else:
            bins[target].append(u)
            used[target] += u.length

    for b in bins:
        tokens: list[int] = []
        masks: list[int] = []
        segs: list[int] = []
        members: list[str] = []
        span_ids: list[str] = []
        for seg, u in enumerate(b):
            tokens.extend(u.tokens)
            masks.extend(u.loss_mask)
            segs.extend([seg] * u.length)
            members.append(u.doc_id)
            span_ids.append(u.span_id)
        assert len(tokens) <= ctx, "bin overfilled - packing invariant violated"
        out.append(_finish(tokens, masks, segs, members, span_ids, lane, policy, ctx, []))
    return out


# --------------------------------------------------------------------------------------------
# validators - these are what the negative controls try to break
# --------------------------------------------------------------------------------------------

def validate_masks(seq: PackedSequence) -> tuple[bool, list[str]]:
    """Structural invariants every packed sequence must satisfy."""
    errs: list[str] = []
    n = len(seq.tokens)
    if not (len(seq.loss_mask) == len(seq.segment_ids) == len(seq.position_ids) == n):
        errs.append("parallel arrays have different lengths")
        return False, errs
    for i, t in enumerate(seq.tokens):
        if t == PAD and seq.loss_mask[i] != 0:
            errs.append(f"pad token at {i} is loss-bearing")
        if t == PAD and seq.segment_ids[i] != -1:
            errs.append(f"pad token at {i} belongs to segment {seq.segment_ids[i]}")
    # position ids must restart at every segment change and increase by one within a segment
    last_seg, expect = None, 0
    for i, s in enumerate(seq.segment_ids):
        expect = 0 if s != last_seg else expect + 1
        if seq.position_ids[i] != expect:
            errs.append(f"position id at {i} is {seq.position_ids[i]}, expected {expect}")
            break
        last_seg = s
    return (not errs), errs


def cross_document_supervision(seq: PackedSequence) -> int:
    """Positions supervised whose predecessor belongs to a different document. Must be 0.

    Distinct from `attention_crossings`: that asks what a token may READ, this asks what it is
    trained to PREDICT. A packed window can be perfectly isolated for attention and still teach the
    model that document N+1 follows document N.
    """
    bad = 0
    for t in range(len(seq.tokens)):
        if not seq.loss_mask[t]:
            continue
        if t == 0 or seq.segment_ids[t] < 0 or seq.segment_ids[t] != seq.segment_ids[t - 1]:
            bad += 1
    return bad


def attention_crossings(seq: PackedSequence) -> int:
    """Cross-document pairs the model's real attention mask permits. Must be 0.

    Read the predicate carefully, because the previous version could not fail. It built the mask from
    `segment_ids[j] == segment_ids[i]` and then counted entries where `segment_ids[j] !=
    segment_ids[i]` - `(A == B) and (A != B)`, unsatisfiable for every input. The check reported
    success on a corpus, on random layouts, and on a build with segment isolation deliberately
    removed from the trainer.

    The fix is to derive the mask from the model (`TinyLM.attention_mask`, the same call the forward
    pass makes) and count against the segment ids independently. Now the count rises the moment the
    model stops isolating documents, which is what the negative control asserts.

    Padding rows are excluded: the model permits the diagonal on segment -1 so the softmax stays
    defined, those positions carry no loss and therefore no gradient, and counting a padding token
    attending to itself as a "cross-document crossing" would be noise, not a leak.
    """
    mask = build_attention_mask(seq)
    bad = 0
    for i in range(len(seq.tokens)):
        si = seq.segment_ids[i]
        for j in range(i + 1):
            sj = seq.segment_ids[j]
            if not mask[i][j]:
                continue
            if i == j:
                continue                      # self-attention is never a crossing
            if si < 0 or sj < 0:
                continue                      # padding: no loss, no gradient, not a document
            if si != sj:
                bad += 1
    return bad


def blocked_cross_segment_pairs(seq: PackedSequence) -> int:
    """Pairs a plain causal mask WOULD have allowed that segment isolation blocks.

    Reported alongside `attention_crossings` because a zero-crossing result is only meaningful if
    the mask was actually doing work. On a single-document window this is 0 and the isolation claim
    is vacuous; on a packed window it is large, and that difference is the evidence that packing
    without segment masks would have leaked context between unrelated documents.
    """
    blocked = 0
    for i in range(len(seq.tokens)):
        si = seq.segment_ids[i]
        if si < 0:
            continue
        for j in range(i + 1):
            sj = seq.segment_ids[j]
            if sj >= 0 and sj != si:
                blocked += 1
    return blocked


def build_attention_mask(seq: PackedSequence) -> list[list[int]]:
    """The mask the MODEL applies, re-expressed as a list so the evidence bundle can hash it.

    Delegates to `TinyLM.attention_mask` rather than restating the rule. An earlier version of this
    function computed the condition independently, which meant `attention_crossings` could validate a
    mask the trainer never used - and did: removing segment isolation from the model left this
    function, and therefore every graded verdict, reporting zero crossings.
    """
    import numpy as np
    from .model import TinyLM
    allow = TinyLM.attention_mask(np.array(seq.segment_ids))
    return [[int(v) for v in row] for row in allow]
