"""Immutable tokenized shards and their manifests.

A shard here holds **tokenized documents**, not packed sequences. That is a deliberate
departure from the session's shorthand ("a shard is a thousand examples of training"), and
it is worth stating plainly because it is the load-bearing design decision in this repo.

Packing depends on the *stage*: the window widens from 256 to 1024 at the long-context
stage, and the packing policy differs by lane. If packing were baked into the shard, a
shard could only ever be replayed into the stage it was built for -- and the whole point of
the V6 advisory is to *re-place* shards into different stages on the next run. So:

    shard      = immutable tokenized documents + manifest   (content-addressed)
    pack plan  = deterministic f(shard, policy, seq_len, seed) -> sequences  (hashed)

Both are hashed, both are recorded in the ledger, and replay reads the recorded pack-plan
hash rather than trusting the packing function to be stable across versions.

**What makes a shard immutable in practice.** Nothing that varies between two runs of the
same code may enter the hashed payload -- no wall-clock, no absolute paths, no set
iteration order, no dict insertion order. `build_shard` is a pure function of
(documents, tokenizer, shard key). The demo proves this by building every shard twice and
comparing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .determinism import (
    canonical_bytes,
    content_hash,
    hash_source_tree,
    hash_token_array,
    sha256_bytes,
)
from .documents import Document, SPLIT_TRAIN, TRAINABLE_SPLITS
from .tokenization import FrozenTokenizer

SHARD_SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# per-document record inside a shard
# ---------------------------------------------------------------------------


@dataclass
class ShardDocument:
    """One tokenized document, located by offset into the shard's token array."""

    doc_id: str
    start: int
    end: int                       # exclusive; token_count = end - start
    lane: str
    split: str
    difficulty_band: str
    reasoning_band: Optional[str]
    language: str
    script: str
    atomic: bool
    indivisible: bool
    role_spans: List[Tuple[str, int, int]]   # (role, start, end) relative to `start`
    quality_score: float
    dataset: str
    license: str
    license_tier: str
    content_fingerprint: str

    @property
    def token_count(self) -> int:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "start": self.start,
            "end": self.end,
            "token_count": self.token_count,
            "lane": self.lane,
            "split": self.split,
            "difficulty_band": self.difficulty_band,
            "reasoning_band": self.reasoning_band,
            "language": self.language,
            "script": self.script,
            "atomic": self.atomic,
            "indivisible": self.indivisible,
            "role_spans": [[r, s, e] for r, s, e in self.role_spans],
            "quality_score": self.quality_score,
            "dataset": self.dataset,
            "license": self.license,
            "license_tier": self.license_tier,
            "content_fingerprint": self.content_fingerprint,
        }


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


@dataclass
class ShardManifest:
    """Everything that must be true about a shard before it may be trained on.

    The field list is the session's, made mandatory. `required_fields_present` is checked
    by the loader, and a shard missing any of them is refused -- "we are not going to be
    training on a shard that does not have any of these".
    """

    shard_id: str
    schema_version: int
    lane: str
    split: str
    difficulty_bands: List[str]
    reasoning_bands: List[str]

    token_count: int
    document_count: int
    document_ids: List[str]

    language_histogram: Dict[str, int]
    script_histogram: Dict[str, int]
    dataset_histogram: Dict[str, int]
    license_histogram: Dict[str, int]
    license_tiers: List[str]

    capability_lane: str
    benchmarks: List[str]

    tokenizer_id: str
    tokenizer_hash: str

    cleaning_pipeline: str
    cleaning_code_hash: str
    builder_code_hash: str

    dedup_status: str
    contamination_status: str
    eval_overlap_status: str
    pii_scrubbed: bool

    content_hash: str            # hash of the token buffer
    payload_hash: str            # hash of tokens + masks + document layout
    manifest_hash: str           # hash of this manifest, self-excluded

    parent_manifest: Optional[str]
    curriculum_stages: List[str]         # stages this shard is eligible for
    min_seq_len_required: int            # window needed to place EVERY indivisible doc whole
    min_document_tokens: int             # window needed to place ANY document at all
    max_document_tokens: int
    mean_document_tokens: float

    quality_mean: float
    quality_min: float

    REQUIRED = (
        "shard_id", "tokenizer_hash", "content_hash", "payload_hash", "cleaning_code_hash",
        "dedup_status", "contamination_status", "eval_overlap_status", "capability_lane",
        "split", "token_count", "document_count",
    )

    def to_dict(self, include_hash: bool = True) -> Dict[str, Any]:
        d = {
            "shard_id": self.shard_id,
            "schema_version": self.schema_version,
            "lane": self.lane,
            "split": self.split,
            "difficulty_bands": self.difficulty_bands,
            "reasoning_bands": self.reasoning_bands,
            "token_count": self.token_count,
            "document_count": self.document_count,
            "document_ids": self.document_ids,
            "language_histogram": self.language_histogram,
            "script_histogram": self.script_histogram,
            "dataset_histogram": self.dataset_histogram,
            "license_histogram": self.license_histogram,
            "license_tiers": self.license_tiers,
            "capability_lane": self.capability_lane,
            "benchmarks": self.benchmarks,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_hash": self.tokenizer_hash,
            "cleaning_pipeline": self.cleaning_pipeline,
            "cleaning_code_hash": self.cleaning_code_hash,
            "builder_code_hash": self.builder_code_hash,
            "dedup_status": self.dedup_status,
            "contamination_status": self.contamination_status,
            "eval_overlap_status": self.eval_overlap_status,
            "pii_scrubbed": self.pii_scrubbed,
            "content_hash": self.content_hash,
            "payload_hash": self.payload_hash,
            "parent_manifest": self.parent_manifest,
            "curriculum_stages": self.curriculum_stages,
            "min_seq_len_required": self.min_seq_len_required,
            "min_document_tokens": self.min_document_tokens,
            "max_document_tokens": self.max_document_tokens,
            "mean_document_tokens": self.mean_document_tokens,
            "quality_mean": self.quality_mean,
            "quality_min": self.quality_min,
        }
        if include_hash:
            d["manifest_hash"] = self.manifest_hash
        return d

    def missing_required_fields(self) -> List[str]:
        out = []
        for name in self.REQUIRED:
            val = getattr(self, name, None)
            if val is None or val == "" or val == "unknown":
                out.append(name)
        return out

    def is_trainable(self) -> Tuple[bool, List[str]]:
        """Can an optimizer step ever consume this shard? Refusal reasons if not."""
        reasons: List[str] = []
        missing = self.missing_required_fields()
        if missing:
            reasons.append("manifest missing required fields: {}".format(missing))
        if self.split not in TRAINABLE_SPLITS:
            reasons.append("split={!r} is not trainable".format(self.split))
        if self.eval_overlap_status != "clean":
            reasons.append("eval_overlap_status={!r}".format(self.eval_overlap_status))
        if self.contamination_status != "clean":
            reasons.append("contamination_status={!r}".format(self.contamination_status))
        if not self.pii_scrubbed:
            reasons.append("pii_scrubbed=False")
        if self.dedup_status == "not_run":
            reasons.append("deduplication has not been run")
        bad_tiers = [t for t in self.license_tiers if t not in ("A", "B")]
        if bad_tiers:
            reasons.append("non-trainable licence tiers present: {}".format(bad_tiers))
        return (not reasons), reasons


# ---------------------------------------------------------------------------
# shard
# ---------------------------------------------------------------------------


@dataclass
class Shard:
    manifest: ShardManifest
    tokens: np.ndarray            # int32, concatenated document token ids
    loss_mask: np.ndarray         # uint8, 1 = supervised
    documents: List[ShardDocument]

    @property
    def shard_id(self) -> str:
        return self.manifest.shard_id

    def doc_tokens(self, doc: ShardDocument) -> np.ndarray:
        return self.tokens[doc.start:doc.end]

    def doc_mask(self, doc: ShardDocument) -> np.ndarray:
        return self.loss_mask[doc.start:doc.end]

    def verify(self) -> List[str]:
        """Recompute every hash in the manifest and report disagreements."""
        problems: List[str] = []
        if hash_token_array(self.tokens) != self.manifest.content_hash:
            problems.append("token buffer does not match manifest content_hash")
        if _payload_hash(self.tokens, self.loss_mask, self.documents) != self.manifest.payload_hash:
            problems.append("payload does not match manifest payload_hash")
        if _manifest_hash(self.manifest) != self.manifest.manifest_hash:
            problems.append("manifest body does not match manifest_hash")
        if int(self.tokens.shape[0]) != self.manifest.token_count:
            problems.append("token_count disagrees with buffer length")
        if self.tokens.shape != self.loss_mask.shape:
            problems.append("loss mask length differs from token length")
        return problems


# ---------------------------------------------------------------------------
# hashing helpers
# ---------------------------------------------------------------------------


def _payload_hash(tokens: np.ndarray, mask: np.ndarray, docs: Sequence[ShardDocument]) -> str:
    h = sha256_bytes(
        np.asarray(tokens, dtype=np.int32).tobytes(order="C")
        + np.asarray(mask, dtype=np.uint8).tobytes(order="C")
    )
    layout = content_hash([d.to_dict() for d in docs])
    return content_hash([h, layout])


def _manifest_hash(m: ShardManifest) -> str:
    return content_hash(m.to_dict(include_hash=False))


def _builder_code_hash() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return hash_source_tree(
        [os.path.join(here, f) for f in ("shards.py", "tokenization.py", "documents.py")]
    )


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build_shard(
    docs: Sequence[Document],
    tokenizer: FrozenTokenizer,
    lane: str,
    split: str,
    shard_index: int,
    *,
    dedup_status: str = "unknown",
    contamination_status: str = "unknown",
    eval_overlap_status: str = "unknown",
    parent_manifest: Optional[str] = None,
    curriculum_stages: Optional[Sequence[str]] = None,
) -> Shard:
    """Build one immutable shard. Pure function of its inputs.

    Documents are sorted by `doc_id` before packing into the token buffer, so the layout
    cannot depend on the order they happened to arrive in.
    """
    from .plan import LANE_SPEC

    ordered = sorted(docs, key=lambda d: d.doc_id)
    if not ordered:
        raise ValueError("cannot build an empty shard")

    token_chunks: List[np.ndarray] = []
    mask_chunks: List[np.ndarray] = []
    records: List[ShardDocument] = []
    cursor = 0

    for doc in ordered:
        ids, mask, roles = tokenizer.encode_document(doc)

        # Compress the per-token role list into spans; storing one role per token would
        # triple the manifest for no additional information.
        spans: List[Tuple[str, int, int]] = []
        if roles:
            cur, s = roles[0], 0
            for i in range(1, len(roles)):
                if roles[i] != cur:
                    spans.append((cur, s, i))
                    cur, s = roles[i], i
            spans.append((cur, s, len(roles)))

        token_chunks.append(np.asarray(ids, dtype=np.int32))
        mask_chunks.append(np.asarray(mask, dtype=np.uint8))
        records.append(ShardDocument(
            doc_id=doc.doc_id,
            start=cursor,
            end=cursor + len(ids),
            lane=doc.lane,
            split=doc.split,
            difficulty_band=doc.difficulty_band,
            reasoning_band=doc.reasoning_band,
            language=doc.language,
            script=doc.script,
            atomic=doc.atomic,
            indivisible=doc.indivisible,
            role_spans=spans,
            quality_score=doc.quality.score,
            dataset=doc.provenance.dataset,
            license=doc.provenance.license,
            license_tier=doc.provenance.license_tier,
            content_fingerprint=doc.content_fingerprint(),
        ))
        cursor += len(ids)

    tokens = np.concatenate(token_chunks).astype(np.int32)
    loss_mask = np.concatenate(mask_chunks).astype(np.uint8)

    def _hist(key) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for d in ordered:
            out[key(d)] = out.get(key(d), 0) + 1
        return dict(sorted(out.items()))

    doc_lens = [r.token_count for r in records]
    indivisible_lens = [r.token_count for r in records if r.indivisible or r.atomic]
    qualities = [r.quality_score for r in records]

    content_h = hash_token_array(tokens)
    payload_h = _payload_hash(tokens, loss_mask, records)

    # The shard id is derived from content, not assigned. Two builds of the same documents
    # under the same tokenizer produce the same id, on any machine, on any day.
    shard_id = "shard-{}-{}-{:04d}-{}".format(
        lane, split, shard_index, content_hash([payload_h, tokenizer.tokenizer_hash])[:10]
    )

    manifest = ShardManifest(
        shard_id=shard_id,
        schema_version=SHARD_SCHEMA_VERSION,
        lane=lane,
        split=split,
        difficulty_bands=sorted({r.difficulty_band for r in records}),
        reasoning_bands=sorted({r.reasoning_band for r in records if r.reasoning_band}),
        token_count=int(tokens.shape[0]),
        document_count=len(records),
        document_ids=[r.doc_id for r in records],
        language_histogram=_hist(lambda d: d.language),
        script_histogram=_hist(lambda d: d.script),
        dataset_histogram=_hist(lambda d: d.provenance.dataset),
        license_histogram=_hist(lambda d: d.provenance.license),
        license_tiers=sorted({r.license_tier for r in records}),
        capability_lane=lane,
        benchmarks=list(LANE_SPEC[lane].benchmarks) if lane in LANE_SPEC else [],
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_hash=tokenizer.tokenizer_hash,
        cleaning_pipeline=ordered[0].quality.cleaning_pipeline,
        cleaning_code_hash=ordered[0].quality.cleaning_code_hash,
        builder_code_hash=_builder_code_hash(),
        dedup_status=dedup_status,
        contamination_status=contamination_status,
        eval_overlap_status=eval_overlap_status,
        pii_scrubbed=all(d.quality.pii_scrubbed for d in ordered),
        content_hash=content_h,
        payload_hash=payload_h,
        manifest_hash="",  # filled below
        parent_manifest=parent_manifest,
        curriculum_stages=sorted(curriculum_stages) if curriculum_stages else [],
        min_seq_len_required=int(max(indivisible_lens)) if indivisible_lens else 0,
        min_document_tokens=int(min(doc_lens)),
        max_document_tokens=int(max(doc_lens)),
        mean_document_tokens=round(float(np.mean(doc_lens)), 3),
        quality_mean=round(float(np.mean(qualities)), 4),
        quality_min=round(float(np.min(qualities)), 4),
    )
    manifest.manifest_hash = _manifest_hash(manifest)

    return Shard(manifest=manifest, tokens=tokens, loss_mask=loss_mask, documents=records)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def write_shard(shard: Shard, root: str) -> Dict[str, str]:
    """Write a shard as a binary token array plus a JSON manifest.

    Binary, because the session is explicit that shipping raw token text from object
    storage to the GPU spends the budget on egress. int32 tokens and uint8 masks are the
    compact honest form; a real system would compress further.
    """
    os.makedirs(root, exist_ok=True)
    base = os.path.join(root, shard.shard_id)

    _atomic_write(base + ".tokens.bin", shard.tokens.astype(np.int32).tobytes(order="C"))
    _atomic_write(base + ".mask.bin", shard.loss_mask.astype(np.uint8).tobytes(order="C"))
    _atomic_write(base + ".docs.json",
                  canonical_bytes([d.to_dict() for d in shard.documents]))
    _atomic_write(base + ".manifest.json", canonical_bytes(shard.manifest.to_dict()))

    return {
        "tokens": base + ".tokens.bin",
        "mask": base + ".mask.bin",
        "docs": base + ".docs.json",
        "manifest": base + ".manifest.json",
    }


def _atomic_write(path: str, data: bytes) -> None:
    """Write-then-rename, with fsync. A half-written shard must never be readable."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_shard(root: str, shard_id: str, tokenizer: Optional[FrozenTokenizer] = None) -> Shard:
    """Read a shard back and verify it against its own manifest before returning it."""
    import json

    base = os.path.join(root, shard_id)
    with open(base + ".manifest.json", "r", encoding="utf-8") as fh:
        mdict = json.load(fh)
    with open(base + ".docs.json", "r", encoding="utf-8") as fh:
        ddicts = json.load(fh)

    tokens = np.fromfile(base + ".tokens.bin", dtype=np.int32)
    mask = np.fromfile(base + ".mask.bin", dtype=np.uint8)

    stored_hash = mdict.pop("manifest_hash")
    manifest = ShardManifest(manifest_hash=stored_hash, **{
        k: v for k, v in mdict.items()
    })
    documents = [
        ShardDocument(
            doc_id=d["doc_id"], start=d["start"], end=d["end"], lane=d["lane"],
            split=d["split"], difficulty_band=d["difficulty_band"],
            reasoning_band=d["reasoning_band"], language=d["language"], script=d["script"],
            atomic=d["atomic"], indivisible=d["indivisible"],
            role_spans=[(r, s, e) for r, s, e in d["role_spans"]],
            quality_score=d["quality_score"], dataset=d["dataset"], license=d["license"],
            license_tier=d["license_tier"], content_fingerprint=d["content_fingerprint"],
        )
        for d in ddicts
    ]

    shard = Shard(manifest=manifest, tokens=tokens, loss_mask=mask, documents=documents)

    problems = shard.verify()
    if problems:
        raise RuntimeError("shard {} failed verification on read: {}".format(shard_id, problems))
    if tokenizer is not None:
        tokenizer.assert_matches(manifest.tokenizer_hash, "reading shard " + shard_id)
    return shard
