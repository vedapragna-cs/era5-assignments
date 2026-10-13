"""The consumption ledger: an append-only, tamper-evident record of what was actually served.

Why a ledger rather than a seed
-------------------------------
A seed reproduces a stream only if every consumer draws from it in the same order. A distributed
loader cannot promise that: workers restart, ranks retry, worker counts change between runs, and
Python's own seeded generator is only stable within one live process. So the planned order is
computable but the *consumed* order is recorded, and replay reads the record.

Tamper evidence
---------------
"Append-only" is a convention unless something enforces it. Each entry commits to the hash of its
predecessor, so editing any historical record invalidates every entry after it and `verify()` names
the first broken link. This is the lakehouse transaction-log idea (Iceberg / Delta / LakeFS) applied
to a delivery stream rather than a table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .canon import HashChain, chash

LEDGER_VERSION = "tds-ledger-1"
DATALOADER_VERSION = "tds-loader-1"


@dataclass
class BatchRecord:
    """One consumed global batch. Carries every field the lesson enumerates."""
    run_id: str
    branch_id: str
    global_step: int
    ledger_offset: int
    checkpoint_id: str | None
    world_size: int
    rank_assignment: dict[int, list[str]]      # rank -> sample ids
    microbatch_ids: list[str]
    packed_sample_ids: list[str]
    shard_ids: list[str]
    token_span_ids: list[str]
    loss_mask_hash: str
    attention_policy: str
    position_policy: str
    mixture_lanes: dict[str, int]
    curriculum_stage: str
    tokenizer_version: str
    tokenizer_hash: str
    dataloader_version: str
    opus_decision_ids: list[str]
    batch_hash: str
    n_tokens: int
    n_loss_bearing: int

    def to_record(self) -> dict:
        return {
            "event": "batch_committed",
            "run_id": self.run_id, "branch_id": self.branch_id,
            "global_step": self.global_step, "ledger_offset": self.ledger_offset,
            "checkpoint_id": self.checkpoint_id, "world_size": self.world_size,
            "rank_assignment": {str(k): v for k, v in sorted(self.rank_assignment.items())},
            "microbatch_ids": self.microbatch_ids,
            "packed_sample_ids": self.packed_sample_ids, "shard_ids": self.shard_ids,
            "token_span_ids": self.token_span_ids, "loss_mask_hash": self.loss_mask_hash,
            "attention_policy": self.attention_policy, "position_policy": self.position_policy,
            "mixture_lanes": self.mixture_lanes, "curriculum_stage": self.curriculum_stage,
            "tokenizer_version": self.tokenizer_version, "tokenizer_hash": self.tokenizer_hash,
            "dataloader_version": self.dataloader_version,
            "opus_decision_ids": self.opus_decision_ids,
            "batch_hash": self.batch_hash,
            "n_tokens": self.n_tokens, "n_loss_bearing": self.n_loss_bearing,
        }


def batch_hash(sample_ids: list[str], span_ids: list[str], loss_mask_hash: str) -> str:
    """Identity of a served batch. Replay must reproduce this exactly."""
    return chash({"samples": sample_ids, "spans": span_ids, "loss_mask": loss_mask_hash})


@dataclass
class ConsumptionLedger:
    run_id: str
    chain: HashChain = field(default_factory=HashChain)

    @property
    def offset(self) -> int:
        return len(self.chain.entries)

    def append(self, payload: dict) -> dict:
        return self.chain.append(payload)

    def commit_batch(self, rec: BatchRecord) -> dict:
        return self.append(rec.to_record())

    def bind_checkpoint(self, checkpoint_id: str, step: int, branch: str, state: dict) -> dict:
        return self.append({
            "event": "checkpoint_bound", "run_id": self.run_id, "branch_id": branch,
            "global_step": step, "checkpoint_id": checkpoint_id,
            "ledger_offset_at_bind": self.offset, **state,
        })

    def record_crash(self, step: int, branch: str, checkpoint_id: str | None, reason: str) -> dict:
        return self.append({
            "event": "trainer_crashed", "run_id": self.run_id, "branch_id": branch,
            "global_step": step, "checkpoint_id": checkpoint_id, "reason": reason,
            "recovery_mode": "resume_from_last_committed_offset",
            "next_expected_offset": self.offset,
        })

    def record_resume(self, step: int, branch: str, checkpoint_id: str,
                      from_offset: int, world_size: int, prev_world_size: int) -> dict:
        return self.append({
            "event": "run_resumed", "run_id": self.run_id, "branch_id": branch,
            "global_step": step, "checkpoint_id": checkpoint_id,
            "resumed_from_ledger_offset": from_offset,
            "world_size": world_size, "previous_world_size": prev_world_size,
            "rank_partition_rule": "rank = global_sample_index % world_size, recomputed at load",
        })

    def record_fork(self, step: int, parent_branch: str, new_branch: str,
                    checkpoint_id: str, offset: int) -> dict:
        return self.append({
            "event": "branch_forked", "run_id": self.run_id, "branch_id": new_branch,
            "parent_branch_id": parent_branch, "global_step": step,
            "checkpoint_id": checkpoint_id, "divergence_ledger_offset": offset,
        })

    # ------------------------------------------------------------------ queries
    def superseded_offsets(self) -> set[int]:
        out: set[int] = set()
        for e in self.chain.entries:
            if e.get("event") == "batches_superseded":
                out.update(e.get("ledger_offsets", []))
        return out

    def mark_superseded(self, offsets: list[int], *, reason: str,
                        branch_id: str = "run-a") -> dict:
        """Append-only supersession.

        The ledger never deletes: an orphaned batch really was served to a model state that later
        vanished, and that fact is part of the run's history. Marking it superseded keeps the record
        while making the EFFECTIVE stream unambiguous.
        """
        # branch_id is carried because every other record is branch-scoped: a consumer filtering
        # on branch_id == "run-a" would otherwise see the duplicate pre-crash commits and NOT the
        # event that retires them, which is precisely the trap for the external audit the
        # assignment asks for.
        return self.append({"event": "batches_superseded", "run_id": self.run_id,
                            "branch_id": branch_id,
                            "ledger_offsets": sorted(offsets), "reason": reason})

    def batches(self, branch: str | None = None, *, include_superseded: bool = False) -> list[dict]:
        dead = set() if include_superseded else self.superseded_offsets()
        return [e for e in self.chain.entries
                if e.get("event") == "batch_committed"
                and e.get("ledger_offset") not in dead
                and (branch is None or e.get("branch_id") == branch)]

    def batch_at_step(self, step: int, branch: str | None = None) -> dict | None:
        for e in self.batches(branch):
            if e["global_step"] == step:
                return e
        return None

    def interval(self, step_from: int, step_to: int, branch: str | None = None) -> list[dict]:
        return [e for e in self.batches(branch) if step_from <= e["global_step"] < step_to]

    def shards_in_interval(self, step_from: int, step_to: int, branch: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.interval(step_from, step_to, branch):
            for s in e["shard_ids"]:
                counts[s] = counts.get(s, 0) + 1
        return dict(sorted(counts.items()))

    def lane_token_shares(self, branch: str | None = None) -> dict[str, float]:
        """Delivered lane shares, measured from what was consumed - not from the plan."""
        tally: dict[str, int] = {}
        for e in self.batches(branch):
            for lane, n in e["mixture_lanes"].items():
                tally[lane] = tally.get(lane, 0) + n
        total = sum(tally.values())
        return {k: (v / total if total else 0.0) for k, v in sorted(tally.items())}

    def verify(self) -> tuple[bool, int | None, str]:
        return self.chain.verify()

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.chain.to_jsonl(), encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path, run_id: str) -> "ConsumptionLedger":
        entries = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
        led = cls(run_id=run_id)
        led.chain = HashChain.from_entries(entries)
        return led
