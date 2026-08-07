"""The data stream: schedule -> shards -> packed sequences -> micro-batches.

This is the loader, and it is where the second, independent half of the evaluation firewall
lives. `firewall.py` refuses unsafe documents at build time; this module refuses unsafe
*shards* at load time, re-deriving the verdict from the manifest rather than trusting a flag
set earlier. The session is explicit that one check is not enough.

It is also where the dataloader **state** lives -- the thing a checkpoint must capture for a
resume to be exact. "I need to ask the data loader, do here, give me the thousand step
onwards data." That state is three integers per (lane, stage, window): which epoch over the
lane's shards, which shard, which sequence within it. Nothing else, and in particular no
RNG object: every shuffle is keyed by (run_id, lane, stage, epoch) so it can be recomputed
from those integers alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .determinism import content_hash, stable_shuffle
from .documents import SPLIT_TRAIN, TRAINABLE_SPLITS
from .masks import (
    LANE_ATTENTION,
    LANE_POSITION,
    MaterializedSequence,
    materialize,
    verify_masks,
)
from .mixture import MixtureSchedule, RunShape, StepMixture
from .packing import LANE_POLICY, PackPlan, pack
from .plan import stage_by_name
from .registry import ShardRegistry
from .tokenization import FrozenTokenizer


class EvaluationFirewallError(RuntimeError):
    """Raised when something that is not training data reaches the training path."""


# ---------------------------------------------------------------------------
# a single training sample and its provenance
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    seq: MaterializedSequence
    lane: str
    shard_id: str
    split: str
    difficulty_band: str
    packing_policy: str
    pack_plan_hash: str
    sequence_index: int
    epoch: int

    def provenance(self) -> Dict[str, Any]:
        st = self.seq.stats()
        return {
            "lane": self.lane,
            "shard_id": self.shard_id,
            "split": self.split,
            "difficulty_band": self.difficulty_band,
            "packing_policy": self.packing_policy,
            "pack_plan_hash": self.pack_plan_hash[:12],
            "sequence_index": self.sequence_index,
            "lane_epoch": self.epoch,
            "doc_ids": self.seq.doc_ids,
            "attention_policy": self.seq.attention_policy,
            "position_policy": self.seq.position_policy,
            "real_tokens": st["real_tokens"],
            "pad_tokens": st["pad_tokens"],
            "supervised_tokens": st["supervised_tokens"],
            "supervision_yield": st["supervision_yield"],
        }


@dataclass
class MicroBatch:
    rank: int
    accum_index: int
    samples: List[Sample]

    def tensors(self) -> Dict[str, np.ndarray]:
        return {
            "input_ids": np.stack([s.seq.input_ids for s in self.samples]),
            "loss_mask": np.stack([s.seq.loss_mask for s in self.samples]),
            "segment_ids": np.stack([s.seq.segment_ids for s in self.samples]),
            "position_ids": np.stack([s.seq.position_ids for s in self.samples]),
            "role_codes": np.stack([s.seq.role_codes for s in self.samples]),
        }


@dataclass
class Batch:
    step: int
    stage: str
    seq_len: int
    micro_batches: List[MicroBatch]
    mixture: StepMixture

    @property
    def samples(self) -> List[Sample]:
        return [s for mb in self.micro_batches for s in mb.samples]

    def token_count(self) -> int:
        return sum(s.seq.seq_len for s in self.samples)

    def realized_lane_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in self.samples:
            out[s.lane] = out.get(s.lane, 0) + 1
        return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# per-lane cursor
# ---------------------------------------------------------------------------


@dataclass
class LaneCursor:
    """Where a lane has read up to. This is the resumable part of the loader."""

    epoch: int = 0
    shard_pos: int = 0
    seq_pos: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {"epoch": self.epoch, "shard_pos": self.shard_pos, "seq_pos": self.seq_pos}

    @staticmethod
    def from_dict(d: Dict[str, int]) -> "LaneCursor":
        return LaneCursor(epoch=d["epoch"], shard_pos=d["shard_pos"], seq_pos=d["seq_pos"])


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------


class DataStream:
    """Turns a compiled schedule into batches, and remembers exactly where it is."""

    def __init__(
        self,
        registry: ShardRegistry,
        schedule: MixtureSchedule,
        tokenizer: FrozenTokenizer,
        run_id: str,
        *,
        attention_override: Optional[Dict[str, str]] = None,
        verify_sample_masks: bool = True,
    ) -> None:
        self.registry = registry
        self.schedule = schedule
        self.tokenizer = tokenizer
        self.run_id = run_id
        self.shape = schedule.shape
        self.attention = dict(LANE_ATTENTION)
        if attention_override:
            self.attention.update(attention_override)
        self.position = dict(LANE_POSITION)
        self.verify_sample_masks = verify_sample_masks

        self.cursors: Dict[str, LaneCursor] = {}
        self._plan_cache: Dict[Tuple[str, str, int, int], PackPlan] = {}
        self.firewall_rejections: List[Dict[str, Any]] = []
        self.mask_violations: List[Dict[str, Any]] = []
        self.deferred: Dict[str, int] = {}

    # ---- state ---------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "cursors": {k: v.to_dict() for k, v in sorted(self.cursors.items())},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.cursors = {k: LaneCursor.from_dict(v) for k, v in state["cursors"].items()}

    # ---- firewall (load side) -----------------------------------------

    def _admit_shard(self, shard_id: str) -> bool:
        """Independent re-derivation of trainability from the manifest.

        Deliberately does not consult any decision made at build time. If the build-side
        firewall had a bug, this is the check that has to catch it.
        """
        m = self.registry.manifest(shard_id)

        reasons: List[str] = []
        if m.split not in TRAINABLE_SPLITS:
            reasons.append("split={!r}".format(m.split))
        if m.eval_overlap_status != "clean":
            reasons.append("eval_overlap={!r}".format(m.eval_overlap_status))
        if m.contamination_status != "clean":
            reasons.append("contamination={!r}".format(m.contamination_status))
        if not m.pii_scrubbed:
            reasons.append("pii_not_scrubbed")
        if m.tokenizer_hash != self.tokenizer.tokenizer_hash:
            reasons.append("tokenizer_hash_mismatch")
        missing = m.missing_required_fields()
        if missing:
            reasons.append("missing_manifest_fields={}".format(missing))

        if reasons:
            self.firewall_rejections.append({"shard_id": shard_id, "reasons": reasons})
            return False
        return True

    def assert_no_eval_leak(self) -> None:
        """Hard failure if an evaluation shard was ever admitted to the training path."""
        leaked = [r for r in self.firewall_rejections if any("split=" in x for x in r["reasons"])]
        if leaked:
            # These were caught, which is the correct outcome. The error case is the
            # opposite: a non-train shard that was *not* rejected.
            pass
        for key, cur in self.cursors.items():
            lane, stage, seq_len = _parse_key(key)
            for sid in self._eligible(lane, stage, seq_len):
                if self.registry.manifest(sid).split != SPLIT_TRAIN:
                    raise EvaluationFirewallError(
                        "shard {} with split={} reached the training stream".format(
                            sid, self.registry.manifest(sid).split))

    # ---- packing -------------------------------------------------------

    def _eligible(self, lane: str, stage_name: str, seq_len: int) -> List[str]:
        stage = stage_by_name(stage_name)
        ids = self.registry.eligible(lane, stage, seq_len)
        return [sid for sid in ids if self._admit_shard(sid)]

    def _plan_for(self, lane: str, shard_id: str, seq_len: int, epoch: int) -> PackPlan:
        key = (lane, shard_id, seq_len, epoch)
        cached = self._plan_cache.get(key)
        if cached is not None:
            return cached
        policy = LANE_POLICY[lane]
        plan = pack(
            self.registry.get(shard_id), policy, seq_len,
            # Shuffle key includes the epoch, so the second pass over a shard packs its
            # documents in a different order -- repetition without identical batches.
            shuffle_key=(self.run_id, "pack", lane, shard_id, seq_len, epoch),
        )
        if plan.deferred_doc_ids:
            self.deferred[lane] = self.deferred.get(lane, 0) + len(plan.deferred_doc_ids)
        self._plan_cache[key] = plan
        return plan

    # ---- drawing -------------------------------------------------------

    def _take(self, lane: str, stage_name: str, seq_len: int, n: int) -> List[Sample]:
        if n <= 0:
            return []
        key = _make_key(lane, stage_name, seq_len)
        cur = self.cursors.setdefault(key, LaneCursor())

        eligible = self._eligible(lane, stage_name, seq_len)
        if not eligible:
            return []

        out: List[Sample] = []
        guard = 0
        while len(out) < n:
            guard += 1
            if guard > 10000:
                break

            order = stable_shuffle(
                eligible, self.run_id, "shard_order", lane, stage_name, seq_len, cur.epoch)
            if cur.shard_pos >= len(order):
                cur.epoch += 1
                cur.shard_pos = 0
                cur.seq_pos = 0
                continue

            shard_id = order[cur.shard_pos]
            plan = self._plan_for(lane, shard_id, seq_len, cur.epoch)
            if cur.seq_pos >= len(plan.sequences):
                cur.shard_pos += 1
                cur.seq_pos = 0
                continue

            pseq = plan.sequences[cur.seq_pos]
            shard = self.registry.get(shard_id)
            mseq = materialize(
                pseq, shard, self.tokenizer.pad_id,
                attention_policy=self.attention[lane],
                position_policy=self.position[lane],
                lane=lane,
            )
            if self.verify_sample_masks and len(out) == 0:
                problems = verify_masks(mseq, shard, self.tokenizer.eos_id)
                if problems:
                    self.mask_violations.append(
                        {"shard_id": shard_id, "lane": lane, "problems": problems})

            out.append(Sample(
                seq=mseq, lane=lane, shard_id=shard_id,
                split=shard.manifest.split,
                difficulty_band=shard.manifest.difficulty_bands[0],
                packing_policy=plan.policy, pack_plan_hash=plan.plan_hash,
                sequence_index=cur.seq_pos, epoch=cur.epoch,
            ))
            cur.seq_pos += 1

        return out

    # ---- batches -------------------------------------------------------

    def batch(self, step: int) -> Batch:
        """Build the batch for one optimizer step.

        Sequences are drawn per lane, then interleaved by a deterministic keyed shuffle and
        dealt round-robin to (rank, accumulation slot). Round-robin rather than contiguous
        slicing matters: contiguous slicing would give rank 0 all the web sequences and
        rank 3 all the agentic ones, so each rank would see a different distribution and
        the gradients being summed would not be samples of the same mixture.
        """
        mix = self.schedule.step(step)
        seq_len = mix.seq_len

        drawn: List[Sample] = []
        for lane in sorted(mix.sequences_per_lane):
            want = mix.sequences_per_lane[lane]
            got = self._take(lane, mix.stage, seq_len, want)
            drawn.extend(got)

        # If a lane under-delivered (no eligible shard), backfill from the largest lane
        # that can supply, so the batch keeps its declared shape.
        deficit = self.shape.sequences_per_step - len(drawn)
        if deficit > 0:
            for lane in sorted(mix.sequences_per_lane,
                               key=lambda l: -mix.sequences_per_lane[l]):
                if deficit <= 0:
                    break
                extra = self._take(lane, mix.stage, seq_len, deficit)
                drawn.extend(extra)
                deficit -= len(extra)

        drawn = stable_shuffle(drawn, self.run_id, "interleave", step)

        slots = [(r, a) for a in range(self.shape.grad_accum)
                 for r in range(self.shape.world_size)]
        buckets: Dict[Tuple[int, int], List[Sample]] = {s: [] for s in slots}
        for i, sample in enumerate(drawn):
            buckets[slots[i % len(slots)]].append(sample)

        micro = [MicroBatch(rank=r, accum_index=a, samples=buckets[(r, a)])
                 for (r, a) in slots if buckets[(r, a)]]

        return Batch(step=step, stage=mix.stage, seq_len=seq_len,
                     micro_batches=micro, mixture=mix)

    # ---- reporting -----------------------------------------------------

    def report(self) -> Dict[str, Any]:
        return {
            "cursors": self.state_dict()["cursors"],
            "firewall_rejections": len(self.firewall_rejections),
            "firewall_rejection_detail": _dedupe_rejections(self.firewall_rejections),
            "mask_violations": self.mask_violations,
            "documents_deferred_by_lane": dict(sorted(self.deferred.items())),
            "pack_plans_cached": len(self._plan_cache),
            "attention_policy_by_lane": dict(sorted(self.attention.items())),
            "position_policy_by_lane": dict(sorted(self.position.items())),
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_key(lane: str, stage: str, seq_len: int) -> str:
    return "{}|{}|{}".format(lane, stage, seq_len)


def _parse_key(key: str) -> Tuple[str, str, int]:
    lane, stage, seq_len = key.split("|")
    return lane, stage, int(seq_len)


def _dedupe_rejections(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        seen.setdefault(r["shard_id"], r)
    return sorted(seen.values(), key=lambda r: r["shard_id"])
