"""Immutable tokenized shards, manifests, and the admission gate.

A shard is a sealed object: token bytes plus a manifest. Its identity is `content_hash`, taken over
the token bytes themselves. Modify a shard and it is a different shard with different lineage - the
manifest carries `parent_shard_ids` so a derived shard points at what it came from.

Admission gate
--------------
Seven HARD blockers: tokenizer_hash, cleaning_pipeline_hash, dedup_status, pii_screen_status,
eval_overlap_status, repeat_ngram_status, and a license tier that is not `unsafe`.

`repeat_ngram_status` is ours, after OLMo 2 (arXiv:2501.00656 s3.1): documents carrying long runs of
repeated n-grams are over-represented in the batches where loss spikes occur, and a shard that has
not been screened for them has not been checked for a documented failure trigger.

We deliberately make all four Session-4 fields hard. The course's own manifest widget makes only
contamination hard: measured there, a shard with `pii_screen_status` missing scores 87/100 and reads
"Admitted to registry", and the same is true with dedup missing. The lesson text quotes S4's contract
as "cleaned, deduplicated, PII-screened and contamination-scanned" and then makes half of it
advisory. An advisory PII gate is not a firewall, so we differ and say why.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .canon import bhash, chash
from .clean import CleanRecord
from .corpus import Document
from .tokenizer import EOS, Tokenizer, fertility

HARD_FIELDS = (
    "tokenizer_hash",
    "cleaning_pipeline_hash",
    "dedup_status",
    "pii_screen_status",
    "eval_overlap_status",
    # OLMo 2 arXiv:2501.00656 section 3.1 - a shard whose repeated-n-gram screen is missing has not
    # been checked for the documented spike trigger, and "not checked" is not "clean".
    "repeat_ngram_status",
)


@dataclass(frozen=True)
class TokenSpan:
    """Where one document lives inside a shard's flat token array."""
    doc_id: str
    start: int
    end: int
    lane: str
    roles: tuple[tuple[str, int, int], ...]   # (role, start, end) relative to the shard array

    @property
    def span_id(self) -> str:
        return f"{self.doc_id}:{self.start}-{self.end}"

    def to_record(self) -> dict:
        return {"doc_id": self.doc_id, "span_id": self.span_id, "start": self.start,
                "end": self.end, "lane": self.lane,
                "roles": [{"role": r, "start": s, "end": e} for r, s, e in self.roles]}


@dataclass
class Shard:
    shard_id: str
    lane: str
    tokens: list[int]
    spans: list[TokenSpan]
    manifest: dict = field(default_factory=dict)

    @property
    def token_bytes(self) -> bytes:
        return struct.pack(f"<{len(self.tokens)}I", *self.tokens)

    @property
    def content_hash(self) -> str:
        """Identity of the shard: SHA-256 over the token bytes.

        Not over the metadata. The course widget regenerates its content_hash with Math.random() on
        every render - measured: nudging a slider and returning to the identical setting produced a
        different hash - which cannot support the immutability it teaches.
        """
        return bhash(self.token_bytes)


def build_shards(
    documents: list[Document],
    clean_records: dict[str, CleanRecord],
    tok: Tokenizer,
    *,
    overlap_status: dict[str, str],
    split: str = "train",
    max_tokens_per_shard: int = 2048,
) -> list[Shard]:
    """Tokenize admitted-split documents into immutable per-lane shards.

    Documents are grouped by lane (the mixture scheduler selects by lane, so a shard that mixes lanes
    cannot be scheduled) and emitted in stable doc_id order.
    """
    by_lane: dict[str, list[Document]] = {}
    for doc in sorted(documents, key=lambda d: d.doc_id):
        if doc.split != split:
            continue
        by_lane.setdefault(doc.lane, []).append(doc)

    shards: list[Shard] = []
    for lane in sorted(by_lane):
        buf_tokens: list[int] = []
        buf_spans: list[TokenSpan] = []
        buf_docs: list[Document] = []
        index = 0

        def flush() -> None:
            nonlocal buf_tokens, buf_spans, buf_docs, index
            if not buf_tokens:
                return
            shard = Shard(
                shard_id=f"v5_{lane}_{split}_{len(shards):03d}",
                lane=lane, tokens=list(buf_tokens), spans=list(buf_spans),
            )
            shard.manifest = _manifest(shard, buf_docs, clean_records, tok, overlap_status, split)
            shards.append(shard)
            buf_tokens, buf_spans, buf_docs, index = [], [], [], 0

        for doc in by_lane[lane]:
            rec = clean_records[doc.doc_id]
            # Re-clean per segment so role boundaries survive; the concatenation equals rec.text
            # for single-segment documents.
            roles: list[tuple[str, int, int]] = []
            doc_tokens: list[int] = []
            for seg in doc.segments:
                from .clean import clean_text, redact_pii
                seg_text = clean_text(seg.text)
                if rec.pii_screen_status == "redacted":
                    seg_text = redact_pii(seg_text)
                seg_ids = tok.encode(seg_text)
                roles.append((seg.role, len(doc_tokens), len(doc_tokens) + len(seg_ids)))
                doc_tokens.extend(seg_ids)
            doc_tokens.append(EOS)
            roles.append(("boundary", len(doc_tokens) - 1, len(doc_tokens)))

            if buf_tokens and len(buf_tokens) + len(doc_tokens) > max_tokens_per_shard:
                flush()

            start = len(buf_tokens)
            buf_spans.append(TokenSpan(
                doc_id=doc.doc_id, start=start, end=start + len(doc_tokens), lane=lane,
                roles=tuple((r, start + s, start + e) for r, s, e in roles),
            ))
            buf_tokens.extend(doc_tokens)
            buf_docs.append(doc)
        flush()
    return shards


def _manifest(shard: Shard, docs: list[Document], clean_records: dict[str, CleanRecord],
              tok: Tokenizer, overlap_status: dict[str, str], split: str) -> dict:
    recs = [clean_records[d.doc_id] for d in docs]
    langs = sorted({d.language for d in docs})
    # A shard inherits the WORST status of its members: one unscreened document contaminates the
    # shard, because the shard is the unit the scheduler admits.
    dedup = "unique" if all(r.dedup_status == "unique" for r in recs) else "contains_duplicates"
    pii = "screened" if all(r.pii_screen_status in ("clean", "redacted") for r in recs) else "missing"
    rep = "screened" if all(r.repeat_ngram_status == "clean" for r in recs) else "contains_repeats"
    overlaps = [overlap_status.get(d.doc_id, "unknown") for d in docs]
    overlap = "clear" if all(o == "clear" for o in overlaps) else "blocked_or_unknown"
    licenses = {d.license_tier for d in docs}
    license_tier = "unsafe" if "unsafe" in licenses else ("review" if "review" in licenses else "safe")

    text_by_lang: dict[str, str] = {}
    for d in docs:
        text_by_lang.setdefault(d.language, "")
        text_by_lang[d.language] += d.text

    return {
        "shard_id": shard.shard_id,
        "split": split,
        "capability_lane": shard.lane,
        "source_ids": sorted({d.source_id for d in docs}),
        "document_ids": [d.doc_id for d in docs],
        "token_count": len(shard.tokens),
        "tokenizer_hash": tok.hash,
        "tokenizer_version": tok.version,
        "content_hash": shard.content_hash,
        "cleaning_pipeline_hash": recs[0].cleaning_pipeline_hash if recs else None,
        "dedup_status": dedup,
        "repeat_ngram_status": rep,
        "dedup_clusters": sorted({r.dedup_cluster_id for r in recs if r.dedup_cluster_id}),
        "pii_screen_status": pii,
        "pii_kinds_found": sorted({k for r in recs for k in r.pii_kinds}),
        "language_validated": all(r.language_validated for r in recs),
        "eval_overlap_status": overlap,
        "languages": langs,
        "scripts": sorted({d.script for d in docs}),
        "difficulty_bands": sorted({d.difficulty_band for d in docs}),
        "license_tier": license_tier,
        "provenance_tier": "audited" if license_tier == "safe" else "needs_review",
        "fertility_by_language": {lg: round(fertility(tok, t), 4) for lg, t in sorted(text_by_lang.items())},
        "token_spans": [s.to_record() for s in shard.spans],
        "parent_shard_ids": [],
    }


@dataclass
class AdmissionResult:
    shard_id: str
    admitted: bool
    blocking_reasons: list[str]
    manifest_hash: str

    def to_record(self) -> dict:
        return {"shard_id": self.shard_id, "admitted": self.admitted,
                "blocking_reasons": self.blocking_reasons, "manifest_hash": self.manifest_hash}


def admit(manifest: dict) -> AdmissionResult:
    """Hard gate. Any missing hard field or unsafe license blocks the shard from training."""
    reasons: list[str] = []
    for f in HARD_FIELDS:
        v = manifest.get(f)
        if v in (None, "", "missing", "unknown", "blocked_or_unknown"):
            reasons.append(f"missing_or_unsafe:{f}={v}")
    if manifest.get("license_tier") == "unsafe":
        reasons.append("license_tier=unsafe")
    if manifest.get("dedup_status") == "contains_duplicates":
        reasons.append("dedup_status=contains_duplicates")
    if manifest.get("repeat_ngram_status") == "contains_repeats":
        reasons.append("repeat_ngram_status=contains_repeats")
    if not manifest.get("language_validated", False):
        reasons.append("language_validated=False")
    if manifest.get("split") != "train":
        reasons.append(f"split={manifest.get('split')} is not trainable")
    return AdmissionResult(
        shard_id=manifest.get("shard_id", "<unknown>"),
        admitted=not reasons,
        blocking_reasons=reasons,
        manifest_hash=chash(manifest),
    )


def verify_shard_integrity(shard: Shard) -> tuple[bool, str]:
    """Recompute the content hash and compare against the manifest's recorded value."""
    recorded = shard.manifest.get("content_hash")
    actual = shard.content_hash
    if recorded != actual:
        return False, f"content_hash mismatch: manifest {recorded[:12]} vs actual {actual[:12]}"
    if shard.manifest.get("token_count") != len(shard.tokens):
        return False, "token_count mismatch"
    return True, "intact"
