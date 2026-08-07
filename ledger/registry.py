"""Building the full shard set, and the registry that indexes it.

Shards are cut per (lane, split, difficulty band). Splitting on band as well as lane is what
makes the curriculum's band gate a cheap lookup at sampling time rather than a per-document
filter -- the mechanism S5 deferred to S6.

Stage eligibility is derived, not declared: a shard is eligible for a stage when its band is
in that stage's permitted range *and* its longest indivisible document fits the stage's
window. The second condition is the long-context reserve, computed rather than curated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .determinism import content_hash
from .documents import Document, SPLIT_TRAIN
from .firewall import FirewallResult
from .plan import LANES, STAGES, Stage
from .shards import Shard, ShardManifest, build_shard, read_shard, write_shard
from .tokenization import FrozenTokenizer

# Target shard size in tokens. A real shard is ~1B tokens ("a thousand steps of training");
# scaled to this demo it is a few thousand, chosen so that a stage consumes several shards
# and shard-level accounting is visible in the ledger rather than degenerate.
DEFAULT_SHARD_TOKENS = 6000


@dataclass
class ShardRegistry:
    """Every shard, with the indexes the sampler and the firewall need."""

    shards: Dict[str, Shard]
    order: List[str]                                  # deterministic shard ordering
    by_lane_split: Dict[Tuple[str, str], List[str]]
    by_lane_band: Dict[Tuple[str, str], List[str]]
    paths: Dict[str, Dict[str, str]]
    tokenizer_hash: str
    build_report: Dict[str, Any]

    # ---- lookup --------------------------------------------------------

    def get(self, shard_id: str) -> Shard:
        return self.shards[shard_id]

    def manifest(self, shard_id: str) -> ShardManifest:
        return self.shards[shard_id].manifest

    def trainable_ids(self) -> List[str]:
        out = []
        for sid in self.order:
            ok, _reasons = self.shards[sid].manifest.is_trainable()
            if ok:
                out.append(sid)
        return out

    def eligible(self, lane: str, stage: Stage, seq_len: int) -> List[str]:
        """Trainable shards of `lane` admissible in `stage` at `seq_len`.

        Two gates. The band gate is the curriculum. The window gate is the long-context
        reserve.

        The window gate tests `min_document_tokens`, not `min_seq_len_required`: a shard is
        admissible if *any* of its documents can be placed, and packing defers the rest
        individually. Testing the longest document instead would disqualify a whole shard
        because one file in it is oversized -- which in the first version silently starved
        the long-context lane to zero tokens for the entire run.
        """
        out = []
        for band in stage.band_range:
            for sid in self.by_lane_band.get((lane, band), []):
                m = self.shards[sid].manifest
                ok, _ = m.is_trainable()
                if not ok:
                    continue
                if m.min_document_tokens > seq_len:
                    continue
                out.append(sid)
        return sorted(set(out))

    def withheld_for_window(self, lane: str, seq_len: int) -> List[str]:
        """Shards this window cannot hold. The reserve, made explicit."""
        return sorted(
            sid for sid in self.by_lane_split.get((lane, SPLIT_TRAIN), [])
            if self.shards[sid].manifest.min_seq_len_required > seq_len
        )

    def lane_supply_tokens(self, lane: str) -> int:
        return sum(
            self.shards[sid].manifest.token_count
            for sid in self.by_lane_split.get((lane, SPLIT_TRAIN), [])
            if self.shards[sid].manifest.is_trainable()[0]
        )

    def summary(self) -> Dict[str, Any]:
        rows = []
        for sid in self.order:
            m = self.shards[sid].manifest
            ok, reasons = m.is_trainable()
            rows.append({
                "shard_id": sid, "lane": m.lane, "split": m.split,
                "bands": m.difficulty_bands, "tokens": m.token_count,
                "documents": m.document_count,
                "min_seq_len_required": m.min_seq_len_required,
                "trainable": ok, "refusal_reasons": reasons,
                "content_hash": m.content_hash[:12],
                "manifest_hash": m.manifest_hash[:12],
            })
        return {
            "shards": len(self.order),
            "tokenizer_hash": self.tokenizer_hash[:12],
            "total_tokens": sum(r["tokens"] for r in rows),
            "trainable_shards": sum(1 for r in rows if r["trainable"]),
            "registry_hash": content_hash([r["manifest_hash"] for r in rows]),
            "rows": rows,
        }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _chunk(docs: Sequence[Document], target_tokens: int,
           tokenizer: FrozenTokenizer) -> List[List[Document]]:
    """Split documents into groups of roughly `target_tokens`, in doc_id order."""
    groups: List[List[Document]] = []
    cur: List[Document] = []
    cur_tokens = 0
    for d in sorted(docs, key=lambda x: x.doc_id):
        n = len(tokenizer.encode_document(d)[0])
        if cur and cur_tokens + n > target_tokens:
            groups.append(cur)
            cur, cur_tokens = [], 0
        cur.append(d)
        cur_tokens += n
    if cur:
        groups.append(cur)
    return groups


def build_registry(
    firewall: FirewallResult,
    tokenizer: FrozenTokenizer,
    *,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
    root: Optional[str] = None,
    verify_immutability: bool = True,
) -> ShardRegistry:
    """Build every shard from the admitted documents, and index them.

    When `verify_immutability` is set, every shard is built twice from the same inputs and
    the two hashes compared. That is the actual test of the claim "same code, different
    day, same shard" -- a claim which is easy to make and easy to get wrong, because a
    single `set()` iteration or a `time.time()` anywhere in the path breaks it silently.
    """
    admitted = firewall.admitted

    groups: List[Tuple[str, str, str, List[Document]]] = []
    keys = sorted({(d.lane, d.split, d.difficulty_band) for d in admitted})
    for lane, split, band in keys:
        subset = [d for d in admitted
                  if d.lane == lane and d.split == split and d.difficulty_band == band]
        for chunk in _chunk(subset, shard_tokens, tokenizer):
            groups.append((lane, split, band, chunk))

    shards: Dict[str, Shard] = {}
    order: List[str] = []
    paths: Dict[str, Dict[str, str]] = {}
    immutability: List[Dict[str, Any]] = []

    counters: Dict[Tuple[str, str], int] = {}
    for lane, split, band, chunk in groups:
        idx = counters.get((lane, split), 0)
        counters[(lane, split)] = idx + 1

        # Eligible stages: bands intersect, and the window can hold the longest
        # indivisible document. Window is checked at read time (it depends on the stage),
        # so here we record only the band intersection.
        stages = [st.name for st in STAGES if band in st.band_range]

        shard = build_shard(
            chunk, tokenizer, lane=lane, split=split, shard_index=idx,
            dedup_status="exact+near_complete",
            contamination_status="clean",
            eval_overlap_status="clean",
            parent_manifest="era5-s6-registry-v1",
            curriculum_stages=stages,
        )

        if verify_immutability:
            twin = build_shard(
                chunk, tokenizer, lane=lane, split=split, shard_index=idx,
                dedup_status="exact+near_complete",
                contamination_status="clean",
                eval_overlap_status="clean",
                parent_manifest="era5-s6-registry-v1",
                curriculum_stages=stages,
            )
            immutability.append({
                "shard_id": shard.shard_id,
                "identical": (twin.shard_id == shard.shard_id
                              and twin.manifest.manifest_hash == shard.manifest.manifest_hash
                              and twin.manifest.payload_hash == shard.manifest.payload_hash),
            })

        shards[shard.shard_id] = shard
        order.append(shard.shard_id)
        if root:
            paths[shard.shard_id] = write_shard(shard, root)

    order.sort()

    by_lane_split: Dict[Tuple[str, str], List[str]] = {}
    by_lane_band: Dict[Tuple[str, str], List[str]] = {}
    for sid in order:
        m = shards[sid].manifest
        by_lane_split.setdefault((m.lane, m.split), []).append(sid)
        for band in m.difficulty_bands:
            by_lane_band.setdefault((m.lane, band), []).append(sid)

    report = {
        "shards_built": len(order),
        "shard_target_tokens": shard_tokens,
        "immutability_checked": verify_immutability,
        "immutability_all_identical": all(r["identical"] for r in immutability) if immutability else None,
        "immutability_rows": immutability,
        "written_to": root,
    }

    return ShardRegistry(
        shards=shards, order=order,
        by_lane_split=by_lane_split, by_lane_band=by_lane_band,
        paths=paths, tokenizer_hash=tokenizer.tokenizer_hash, build_report=report,
    )


def reload_and_verify(registry: ShardRegistry, root: str,
                      tokenizer: FrozenTokenizer) -> Dict[str, Any]:
    """Read every shard back from disk and confirm it still hashes to its manifest.

    Catches the failure mode where a shard is written correctly and read back wrongly --
    a dtype mismatch, an endianness assumption, a truncated file.
    """
    rows = []
    for sid in registry.order:
        try:
            back = read_shard(root, sid, tokenizer)
            rows.append({
                "shard_id": sid,
                "ok": True,
                "content_hash_matches": back.manifest.content_hash == registry.manifest(sid).content_hash,
                "tokens": int(back.tokens.shape[0]),
            })
        except Exception as exc:
            rows.append({"shard_id": sid, "ok": False, "error": str(exc)})
    return {
        "shards_checked": len(rows),
        "all_ok": all(r["ok"] and r.get("content_hash_matches") for r in rows),
        "rows": rows,
    }
