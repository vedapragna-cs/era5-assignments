"""Evaluation and validation firewall.

Test data is registered precisely so that it can be blocked. The registry knows every eval document's
content hash, benchmark id, canary strings and never-train flag; the gate is consulted before any
document is tokenized into a training shard, and again before any batch is served.

Six independent gates, checked together:

  1. never_train flag        - the registry says this is eval data
  2. content-hash identity   - byte-identical to a registered eval document
  3. canary match            - a planted marker string appears in the candidate
  4. shingle-Jaccard overlap - the candidate is globally similar to a registered eval document
  5. n-gram containment      - ANY 13-gram of a registered eval document appears in the candidate
  6. derived-from-benchmark  - the candidate is a paraphrase or worked explanation of eval content

Gates 5 and 6 are the ones that matter, and they fail differently, which is why both exist.

Gate 6 was the thin spot measured on the course firewall widget: disabling any *single* gate leaked
nothing, because the obvious eval dumps trip three or four gates each. Only disabling overlap AND
derived together leaked anything, and what leaked was a blog explaining GSM8K rationales - not a copy
of the benchmark, a paraphrase of it.

Gate 5 is the published decontamination standard - GPT-3 filtered on 13-gram overlap, Llama and the
open replications use 8-13 token windows, GPT-4 moved to a 50-character span - and it closes a hole
Jaccard cannot. Jaccard measures GLOBAL similarity, so a long article quoting one benchmark question
scores near zero against it: measured on our planted case web-0102, Jaccard is 0.0806 against an
overlap threshold of 0.12, no canary, not marked derived, and every gate except containment admits
it. Contamination is usually a small passage inside a large host, which is the case similarity
metrics are worst at.

Stated limitation: exact n-gram matching is defeated by paraphrase, translation and reformatting -
all documented in the contamination literature. Gate 5 is a floor, not a ceiling; gate 6 covers the
paraphrase case here only because our corpus labels it, which a real corpus will not. Semantic
retrieval is the named next layer and is not implemented.

Overlap is measured with shingle-Jaccard against registered eval text rather than with a flat
substring test, because a paraphrase shares n-grams without sharing spans.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .canon import chash
from .clean import _shingles, jaccard


def _ngrams_of(text: str, n: int) -> set[str]:
    """Word n-grams, lowercased. Matching is on normalised whitespace, not raw bytes."""
    w = text.lower().split()
    if len(w) < n:
        return set()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
from .corpus import Document

OVERLAP_THRESHOLD = 0.12

# Exact n-gram containment, the decontamination standard used by the large published runs: GPT-3
# filtered on 13-gram overlap, Llama and several open replications use an 8-13 token window, and
# GPT-4 moved to a 50-character span. We take 13.
#
# This gate exists because it catches what Jaccard structurally cannot. Jaccard is a measure of
# GLOBAL similarity between two documents: a 4,000-word article that quotes one 30-word benchmark
# question has a Jaccard score near zero against that benchmark and sails through the overlap gate,
# yet the model has now seen the test item verbatim. Containment asks the other question - does ANY
# benchmark n-gram appear anywhere in this document - and is the standard precisely because
# contamination is usually a small passage inside a large host.
#
# Known limitation, stated rather than papered over: exact n-gram matching is defeated by
# paraphrase, translation and reformatting, all documented in the contamination literature. It is a
# floor, not a ceiling; semantic retrieval is the named next layer and is not implemented here.
NGRAM_CONTAINMENT_N = 13


@dataclass(frozen=True)
class EvalRecord:
    doc_id: str
    benchmark_id: str
    split: str
    content_hash: str
    canary: str | None
    version_tag: str
    never_train: bool

    def to_record(self) -> dict:
        return {"doc_id": self.doc_id, "benchmark_id": self.benchmark_id, "split": self.split,
                "content_hash": self.content_hash, "canary": self.canary,
                "version_tag": self.version_tag, "never_train": self.never_train}


@dataclass
class FirewallDecision:
    doc_id: str
    allowed: bool
    reasons: list[str]
    max_overlap: float
    matched_benchmarks: list[str]

    def to_record(self) -> dict:
        return {"doc_id": self.doc_id, "allowed": self.allowed, "reasons": self.reasons,
                "max_overlap": round(self.max_overlap, 4),
                "matched_benchmarks": self.matched_benchmarks}


@dataclass
class Firewall:
    records: dict[str, EvalRecord] = field(default_factory=dict)
    _hashes: dict[str, str] = field(default_factory=dict)          # content hash -> doc_id
    _canaries: dict[str, str] = field(default_factory=dict)        # canary -> benchmark
    _shingles: list[tuple[str, str, set[str]]] = field(default_factory=list)
    _ngrams: dict[str, str] = field(default_factory=dict)          # benchmark n-gram -> benchmark
    access_log: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    # gates can be individually disabled ONLY by the negative-control suite
    gates: dict[str, bool] = field(default_factory=lambda: {
        "never_train": True, "content_hash": True, "canary": True, "derived": True,
        "overlap": True, "ngram_containment": True})

    @classmethod
    def from_corpus(cls, documents: list[Document]) -> "Firewall":
        fw = cls()
        for doc in documents:
            if doc.split in ("validation", "test"):
                rec = EvalRecord(
                    doc_id=doc.doc_id, benchmark_id=doc.benchmark_id or "unknown", split=doc.split,
                    content_hash=chash(doc.text), canary=doc.canary,
                    version_tag="2026-08", never_train=(doc.split == "test"),
                )
                fw.records[doc.doc_id] = rec
                fw._hashes[rec.content_hash] = doc.doc_id
                if doc.canary:
                    fw._canaries[doc.canary] = rec.benchmark_id
                fw._shingles.append((doc.doc_id, rec.benchmark_id, _shingles(doc.text)))
                for g in _ngrams_of(doc.text, NGRAM_CONTAINMENT_N):
                    fw._ngrams.setdefault(g, rec.benchmark_id)
        return fw

    def check(self, doc: Document, *, purpose: str = "train_admission") -> FirewallDecision:
        reasons: list[str] = []
        matched: list[str] = []
        max_ov = 0.0

        if self.gates["never_train"] and doc.split in ("validation", "test"):
            reasons.append(f"registry: split={doc.split} is never-train")
            if doc.benchmark_id:
                matched.append(doc.benchmark_id)

        if self.gates["content_hash"]:
            h = chash(doc.text)
            if h in self._hashes and self._hashes[h] != doc.doc_id:
                reasons.append(f"content hash identical to eval doc {self._hashes[h]}")

        if self.gates["canary"]:
            for canary, bench in self._canaries.items():
                if canary in doc.text:
                    reasons.append(f"canary match {canary} ({bench})")
                    matched.append(bench)

        if self.gates["overlap"]:
            sh = _shingles(doc.text)
            for other_id, bench, other_sh in self._shingles:
                if other_id == doc.doc_id:
                    continue
                ov = jaccard(sh, other_sh)
                if ov > max_ov:
                    max_ov = ov
                if ov >= OVERLAP_THRESHOLD:
                    reasons.append(f"benchmark overlap {ov:.0%} with {bench}")
                    matched.append(bench)

        if self.gates["ngram_containment"] and self._ngrams:
            hits = sorted({self._ngrams[g] for g in _ngrams_of(doc.text, NGRAM_CONTAINMENT_N)
                           if g in self._ngrams})
            if hits:
                reasons.append(f"{NGRAM_CONTAINMENT_N}-gram containment match: {', '.join(hits)}")
                matched.extend(hits)

        if self.gates["derived"] and doc.derived_from_benchmark:
            reasons.append("benchmark-derived content")

        decision = FirewallDecision(doc.doc_id, not reasons, reasons, max_ov, sorted(set(matched)))
        self.access_log.append({"doc_id": doc.doc_id, "purpose": purpose,
                                "allowed": decision.allowed, "reasons": reasons})
        if not decision.allowed:
            self.rejections.append(decision.to_record())
        return decision

    def overlap_status(self, documents: list[Document]) -> dict[str, str]:
        """doc_id -> 'clear' | 'blocked_or_unknown', for the shard manifest."""
        out = {}
        for doc in documents:
            out[doc.doc_id] = "clear" if self.check(doc).allowed else "blocked_or_unknown"
        return out

    def assert_no_eval_in_batch(self, doc_ids: list[str]) -> tuple[bool, list[str]]:
        """Second line of defence: checked on every served batch, not only at admission."""
        bad = [d for d in doc_ids if d in self.records]
        return (not bad), bad


    def to_record(self) -> dict:
        return {
            "registered": [r.to_record() for r in sorted(self.records.values(), key=lambda r: r.doc_id)],
            "n_registered": len(self.records),
            "n_access_events": len(self.access_log),
            "n_rejections": len(self.rejections),
            "rejections": self.rejections,
            "overlap_threshold": OVERLAP_THRESHOLD,
            "ngram_containment_n": NGRAM_CONTAINMENT_N,
            "benchmark_ngrams_indexed": len(self._ngrams),
            "gates": dict(self.gates),
        }

def independent_contamination_scan(served: list, registry: list,
                                   n: int = NGRAM_CONTAINMENT_N) -> dict[str, list[str]]:
    """Re-derive contamination from document TEXT, without consulting any Firewall state.

    This is deliberately a second, independent implementation, and it exists because the graded
    predicate was not satisfiable. `no_eval_document_in_any_training_sample` tested
    `doc_id in firewall.records` - but `shards.shards_from_documents` already drops every document
    whose split is not `train`, so a *registered* document cannot reach a batch whatever the
    firewall does. An audit stubbed `Firewall.check` and `assert_no_eval_in_batch` to return
    "allowed" and the check still printed PASS, while `web-0100` (a benchmark paraphrase) and
    `web-0102` (an article quoting a GSM8K item verbatim) were served in loss-bearing batches. Those
    are train-split documents contaminated by CONTENT, which is the failure the assignment actually
    names, and registry membership cannot see them.

    So this reads the served text and the registry text and compares them directly: exact-hash
    identity and n-gram containment (the GPT-3 standard, n=13). Sabotaging the admission gate does
    not silence it, because it shares no code with the admission gate.

    Returns {doc_id: [reasons]} for every served document that should never have been served.
    """
    ref_ngrams: dict[str, str] = {}
    ref_hashes: dict[str, str] = {}
    for d in registry:
        ref_hashes.setdefault(chash(d.text), d.doc_id)
        for g in _ngrams_of(d.text, n):
            ref_ngrams.setdefault(g, d.doc_id)

    hits: dict[str, list[str]] = {}
    for d in served:
        why: list[str] = []
        owner = ref_hashes.get(chash(d.text))
        if owner is not None and owner != d.doc_id:
            why.append(f"text identical to registered eval document {owner}")
        shared = sorted({ref_ngrams[g] for g in _ngrams_of(d.text, n) if g in ref_ngrams})
        if shared:
            why.append(f"{n}-gram containment with {', '.join(shared)}")
        if why:
            hits[d.doc_id] = why
    return hits
