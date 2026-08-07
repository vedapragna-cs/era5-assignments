"""The evaluation firewall, deduplication, and the admission gate.

The session asks for the firewall to work from both sides: a shard declares whether it is
train or evaluation, *and* the loader refuses anything that is not train, "because who
knows maybe a mistake in copying or something may still happen". Both checks exist, and
they are deliberately implemented independently -- a shared helper would mean a single bug
disables both.

This module is the build-side half:

* **admission** -- a document that fails `Document.validate()` never reaches a shard.
  Restricted licence, un-scrubbed PII, no trainable segment: refused, with a reason.
* **deduplication** -- exact (content fingerprint) and near (token-shingle Jaccard).
  "What is the point of going through the same example which your teacher taught, then you
  saw with your friend, then in tuition, then in revision."
* **contamination** -- word-level n-gram overlap between train documents and the held-out
  evaluation set. Any train document sharing an n-gram with evaluation text is quarantined,
  not silently dropped, so the quarantine is auditable.

`stream.py` holds the load-side half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from .determinism import content_hash, sha256_bytes
from .documents import Document, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL

# 13-gram overlap is the convention used by GPT-3 / Llama-style contamination audits.
DEFAULT_NGRAM = 13
NEAR_DUP_THRESHOLD = 0.80
SHINGLE_SIZE = 5

# Word segmentation must be script-agnostic, and `\w+` is not.
#
# Python's `re` does not match Unicode combining marks (category Mn) with `\w`, so every
# Devanagari matra and Telugu vowel sign splits a word: "నమోదు చేసింది" becomes
# "నమ ద చ స ద". Thirteen-grams built from those fragments are meaningless and collide
# constantly -- the first version of this module quarantined 186 Indic documents as
# contaminated for that reason alone. On a plan that spends 17% of its budget on Indic,
# a contamination scan that cannot see Indic words is worse than none, because it reports
# a clean lane while doing nothing.
#
# Splitting on whitespace and trimming edge punctuation is correct for every script here.
_PUNCT = " \t\n\r\f\v.,;:!?()[]{}<>\"'`|/\\-–—_=+*&^%$#@~।॥"


def _words(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.split():
        w = raw.strip(_PUNCT).lower()
        if w:
            out.append(w)
    return out


def _ngrams(words: Sequence[str], n: int) -> Set[str]:
    if len(words) < n:
        # Short documents still need a fingerprint, or a two-line eval item is
        # uncheckable. Fall back to the whole string.
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _shingles(words: Sequence[str], k: int = SHINGLE_SIZE) -> FrozenSet[int]:
    grams = _ngrams(words, k)
    return frozenset(int(sha256_bytes(g.encode("utf-8"))[:8], 16) for g in grams)


def _jaccard(a: FrozenSet[int], b: FrozenSet[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / float(len(a) + len(b) - inter)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class FirewallResult:
    admitted: List[Document]
    refused: List[Tuple[str, List[str]]]          # (doc_id, reasons)
    exact_duplicates: List[Tuple[str, str]]       # (dropped, kept)
    near_duplicates: List[Tuple[str, str, float]] # (dropped, kept, jaccard)
    quarantined: List[Tuple[str, str]]            # (doc_id, matched eval doc)
    eval_documents: List[Document]
    report: Dict[str, Any]

    def admitted_by_split(self, split: str) -> List[Document]:
        return [d for d in self.admitted if d.split == split]


# ---------------------------------------------------------------------------
# passes
# ---------------------------------------------------------------------------


def admit(docs: Sequence[Document]) -> Tuple[List[Document], List[Tuple[str, List[str]]]]:
    """Refuse any document that is unsafe to train on, with reasons."""
    admitted, refused = [], []
    for doc in sorted(docs, key=lambda d: d.doc_id):
        problems = doc.validate()
        if problems:
            refused.append((doc.doc_id, problems))
        else:
            admitted.append(doc)
    return admitted, refused


def deduplicate(
    docs: Sequence[Document], threshold: float = NEAR_DUP_THRESHOLD
) -> Tuple[List[Document], List[Tuple[str, str]], List[Tuple[str, str, float]]]:
    """Exact then near duplicate removal, within each lane.

    Restricted to within-lane comparison: cross-lane near-duplicates are usually a real
    signal (the same algorithm described in prose and in code) rather than redundancy, and
    an all-pairs scan across the whole corpus is quadratic for no benefit here.

    The kept document of a duplicate pair is always the lexicographically smallest doc_id,
    so the choice does not depend on iteration order.
    """
    ordered = sorted(docs, key=lambda d: d.doc_id)

    seen: Dict[str, str] = {}
    exact: List[Tuple[str, str]] = []
    survivors: List[Document] = []
    for doc in ordered:
        fp = doc.content_fingerprint()
        if fp in seen:
            exact.append((doc.doc_id, seen[fp]))
        else:
            seen[fp] = doc.doc_id
            survivors.append(doc)

    near: List[Tuple[str, str, float]] = []
    kept: List[Document] = []
    by_lane: Dict[str, List[Tuple[Document, FrozenSet[int]]]] = {}
    for doc in survivors:
        sig = _shingles(_words(doc.text))
        bucket = by_lane.setdefault(doc.lane, [])
        hit = None
        for other, other_sig in bucket:
            j = _jaccard(sig, other_sig)
            if j >= threshold:
                hit = (other.doc_id, j)
                break
        if hit is None:
            bucket.append((doc, sig))
            kept.append(doc)
        else:
            near.append((doc.doc_id, hit[0], round(hit[1], 4)))

    return kept, exact, near


def contamination_scan(
    train_docs: Sequence[Document],
    eval_docs: Sequence[Document],
    n: int = DEFAULT_NGRAM,
) -> List[Tuple[str, str]]:
    """Find train documents sharing an n-gram with any evaluation document.

    Returns (train_doc_id, eval_doc_id) pairs. This is the check that has to catch a
    benchmark leaking into pretraining -- the session's "the model beats OpenAI in the
    first 200 samples" failure, which is not a triumph but a leaked exam paper.
    """
    index: Dict[str, str] = {}
    for ed in sorted(eval_docs, key=lambda d: d.doc_id):
        for gram in _ngrams(_words(ed.text), n):
            index.setdefault(gram, ed.doc_id)

    hits: List[Tuple[str, str]] = []
    for td in sorted(train_docs, key=lambda d: d.doc_id):
        matched: Optional[str] = None
        for gram in _ngrams(_words(td.text), n):
            if gram in index:
                matched = index[gram]
                break
        if matched:
            hits.append((td.doc_id, matched))
    return hits


# ---------------------------------------------------------------------------
# the whole gate
# ---------------------------------------------------------------------------


def run_firewall(docs: Sequence[Document], ngram: int = DEFAULT_NGRAM) -> FirewallResult:
    """Admission -> dedup -> contamination, in that order, with a full report.

    Order matters. Deduplicating before admission would waste work on documents that are
    about to be refused, and scanning for contamination before dedup would report the same
    leak twice under two doc_ids.
    """
    admitted, refused = admit(docs)

    deduped, exact, near = deduplicate(admitted)

    eval_docs = [d for d in deduped if d.split in (SPLIT_VAL, SPLIT_TEST)]
    train_docs = [d for d in deduped if d.split == SPLIT_TRAIN]

    hits = contamination_scan(train_docs, eval_docs, n=ngram)
    quarantined_ids = {t for t, _ in hits}

    clean_train = [d for d in train_docs if d.doc_id not in quarantined_ids]
    final = sorted(clean_train + eval_docs, key=lambda d: d.doc_id)

    report = {
        "ngram": ngram,
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "input_documents": len(docs),
        "refused_documents": len(refused),
        "refusal_reasons": {d: r for d, r in refused},
        "exact_duplicates_dropped": len(exact),
        "exact_duplicate_pairs": [{"dropped": a, "kept": b} for a, b in exact],
        "near_duplicates_dropped": len(near),
        "near_duplicate_pairs": [
            {"dropped": a, "kept": b, "jaccard": j} for a, b, j in near
        ],
        "eval_documents": len(eval_docs),
        "train_documents_before_quarantine": len(train_docs),
        "contaminated_quarantined": len(hits),
        "contamination_hits": [
            {"train_doc": t, "matched_eval_doc": e} for t, e in hits
        ],
        "admitted_documents": len(final),
        "firewall_fingerprint": content_hash([d.doc_id for d in final]),
    }

    return FirewallResult(
        admitted=final,
        refused=refused,
        exact_duplicates=exact,
        near_duplicates=near,
        quarantined=hits,
        eval_documents=eval_docs,
        report=report,
    )
