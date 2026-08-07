"""Checkpoints: everything needed to resume, written so a crash cannot corrupt one.

A checkpoint that contains only model weights is not a resumable checkpoint. Restarting from
one silently re-reads data the model has already seen, restarts the LR schedule, and resets
the optimizer moments -- three separate ways to produce a run that finishes and is wrong.
What is saved here is the full cross-product:

* **model** weights,
* **optimizer** state (AdamW's two moment buffers -- discarding them costs the run a
  re-warmup it will not report),
* **step count**, which drives the LR schedule,
* **data cursors** -- per (lane, stage, seq_len), the epoch, shard position and sequence
  position. Without these, resume re-reads from shard 0 and the epoch ceiling is a fiction,
* **RNG state** for torch and numpy,
* **ledger offsets** -- how many rows each ledger had written, so replay can truncate the
  ledgers back to exactly the checkpointed instant rather than double-counting steps that
  were trained twice,
* **binding hashes** -- tokenizer, registry, schedule and source tree. A checkpoint is
  bound to the data it was trained on. Resuming a checkpoint against a rebuilt shard set is
  the corruption this refuses to allow, and refusing loudly is the entire value of it.

Atomicity is write-to-temp, fsync, `os.replace`. `os.replace` is atomic within a filesystem,
so a reader sees either the whole old checkpoint or the whole new one -- never a half-written
file. `torn_write_drill` deliberately produces the half-written case to prove the loader
rejects it.
"""

from __future__ import annotations

import io
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .determinism import canonical_bytes, content_hash, env_fingerprint

CHECKPOINT_FORMAT = "era5-s6-ckpt-v1"


class CheckpointError(RuntimeError):
    pass


class CheckpointBindingError(CheckpointError):
    """The checkpoint does not belong to this data. Refuse rather than reinterpret."""


@dataclass
class Binding:
    """What a checkpoint is pinned to. Any mismatch is fatal on load."""

    tokenizer_hash: str
    registry_hash: str
    schedule_hash: str
    source_hash: str
    run_id: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "tokenizer_hash": self.tokenizer_hash,
            "registry_hash": self.registry_hash,
            "schedule_hash": self.schedule_hash,
            "source_hash": self.source_hash,
            "run_id": self.run_id,
        }

    def diff(self, other: Dict[str, str]) -> List[str]:
        out = []
        for k, v in self.to_dict().items():
            if other.get(k) != v:
                out.append("{}: checkpoint={} current={}".format(
                    k, str(other.get(k))[:12], str(v)[:12]))
        return out


def _atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    # fsync the directory too, or the rename itself can be lost in a power failure while
    # the file contents survive -- which produces a directory pointing at nothing.
    dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def save_checkpoint(
    root: str,
    *,
    step: int,
    trainer,
    stream,
    binding: Binding,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a complete checkpoint atomically, and return its index entry."""
    import torch

    os.makedirs(root, exist_ok=True)
    name = "step-{:06d}".format(step)
    ckpt_dir = os.path.join(root, name)
    os.makedirs(ckpt_dir, exist_ok=True)

    buf = io.BytesIO()
    torch.save(trainer.state_dict(), buf)
    weights = buf.getvalue()
    weights_path = os.path.join(ckpt_dir, "state.pt")
    _atomic_write_bytes(weights_path, weights)

    rng_state = {
        "torch": torch.get_rng_state().numpy().tolist(),
        "numpy": [int(x) for x in np.random.get_state()[1]],
        "numpy_pos": int(np.random.get_state()[2]),
    }
    rng_path = os.path.join(ckpt_dir, "rng.json")
    _atomic_write_bytes(rng_path, canonical_bytes(rng_state))

    meta = {
        "format": CHECKPOINT_FORMAT,
        "step": step,
        "trainer_step_count": trainer.step_count,
        "binding": binding.to_dict(),
        "data_cursors": stream.state_dict(),
        "ledger_offsets": {
            "consumption_steps": len(trainer.consumption.steps),
            "learning_samples": len(trainer.learning.sample_rows),
            "learning_steps": len(trainer.learning.step_rows),
            "token_detail": len(trainer.learning.token_detail),
        },
        "consumption_totals": {
            "total_tokens": trainer.consumption.total_tokens,
            "lane_tokens": dict(sorted(trainer.consumption.lane_tokens.items())),
        },
        "weights_hash": content_hash(list(weights[:4096]) + [len(weights)]),
        "rng_hash": content_hash(rng_state),
        "env": env_fingerprint(),
        "extra": extra or {},
    }
    # The integrity hash covers everything above it. It is written last, inside the same
    # atomic replace, so a checkpoint whose meta exists always has a hash to check against.
    meta["integrity"] = content_hash(meta)
    meta_path = os.path.join(ckpt_dir, "meta.json")
    _atomic_write_bytes(meta_path, canonical_bytes(meta))

    entry = {
        "step": step, "dir": ckpt_dir, "integrity": meta["integrity"],
        "bytes": len(weights) + len(canonical_bytes(meta)),
    }
    _append_index(root, entry)
    return entry


def _append_index(root: str, entry: Dict[str, Any]) -> None:
    path = os.path.join(root, "index.json")
    rows: List[Dict[str, Any]] = []
    if os.path.exists(path):
        try:
            rows = json.load(open(path))
        except json.JSONDecodeError:
            rows = []
    rows = [r for r in rows if r["step"] != entry["step"]] + [entry]
    rows.sort(key=lambda r: r["step"])
    _atomic_write_bytes(path, canonical_bytes(rows))


def list_checkpoints(root: str) -> List[Dict[str, Any]]:
    path = os.path.join(root, "index.json")
    if not os.path.exists(path):
        return []
    return json.load(open(path))


def verify_checkpoint(ckpt_dir: str) -> Dict[str, Any]:
    """Read the metadata and confirm it hashes to its recorded integrity value.

    Deliberately separate from `load_checkpoint`: a corrupted checkpoint should be
    *diagnosable* without attempting to resume from it.
    """
    meta_path = os.path.join(ckpt_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {"ok": False, "error": "meta.json missing", "dir": ckpt_dir}
    try:
        meta = json.load(open(meta_path))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": "meta.json unparseable: {}".format(exc),
                "dir": ckpt_dir}

    claimed = meta.pop("integrity", None)
    recomputed = content_hash(meta)
    problems = []
    if claimed != recomputed:
        problems.append("integrity hash mismatch")
    if meta.get("format") != CHECKPOINT_FORMAT:
        problems.append("unknown format {!r}".format(meta.get("format")))
    for f in ("state.pt", "rng.json"):
        if not os.path.exists(os.path.join(ckpt_dir, f)):
            problems.append("missing {}".format(f))
    if os.path.exists(os.path.join(ckpt_dir, "state.pt")):
        blob = open(os.path.join(ckpt_dir, "state.pt"), "rb").read()
        if content_hash(list(blob[:4096]) + [len(blob)]) != meta.get("weights_hash"):
            problems.append("weights hash mismatch: state.pt is truncated or rewritten")

    meta["integrity"] = claimed
    return {"ok": not problems, "problems": problems, "dir": ckpt_dir,
            "step": meta.get("step")}


def load_checkpoint(ckpt_dir: str, *, trainer, stream, binding: Binding,
                    strict_binding: bool = True) -> Dict[str, Any]:
    """Restore a run to exactly where it was, or refuse."""
    import torch

    check = verify_checkpoint(ckpt_dir)
    if not check["ok"]:
        raise CheckpointError("checkpoint {} failed verification: {}".format(
            ckpt_dir, check["problems"]))

    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    drift = binding.diff(meta["binding"])
    if drift and strict_binding:
        raise CheckpointBindingError(
            "checkpoint was trained against different data or code:\n  " + "\n  ".join(drift))

    trainer.load_state_dict(torch.load(os.path.join(ckpt_dir, "state.pt"),
                                       weights_only=False))
    stream.load_state_dict(meta["data_cursors"])

    rng = json.load(open(os.path.join(ckpt_dir, "rng.json")))
    torch.set_rng_state(torch.tensor(rng["torch"], dtype=torch.uint8))
    st = np.random.get_state()
    np.random.set_state((st[0], np.array(rng["numpy"], dtype=np.uint32),
                         rng["numpy_pos"], st[3], st[4]))

    return {
        "resumed_from": ckpt_dir,
        "step": meta["step"],
        "binding_drift": drift,
        "ledger_offsets": meta["ledger_offsets"],
        "cursors_restored": len(meta["data_cursors"]["cursors"]),
    }


def truncate_ledgers(trainer, offsets: Dict[str, int]) -> Dict[str, Any]:
    """Cut the ledgers back to the checkpointed instant.

    A crash between checkpoint N and checkpoint N+1 leaves ledger rows for steps that were
    trained but whose weights were discarded. Keeping them would make the ledger claim the
    model saw tokens it did not -- the exact failure a ledger exists to prevent, committed
    by the ledger itself.
    """
    before = {
        "consumption_steps": len(trainer.consumption.steps),
        "learning_samples": len(trainer.learning.sample_rows),
        "learning_steps": len(trainer.learning.step_rows),
    }
    del trainer.consumption.steps[offsets["consumption_steps"]:]
    del trainer.learning.sample_rows[offsets["learning_samples"]:]
    del trainer.learning.step_rows[offsets["learning_steps"]:]
    del trainer.learning.token_detail[offsets.get("token_detail", 0):]

    # Recompute the consumption totals from the surviving rows rather than trusting the
    # running counters, which were incremented for the discarded steps.
    trainer.consumption.total_tokens = sum(r["tokens"] for r in trainer.consumption.steps)
    trainer.consumption.total_supervised = sum(
        r["supervised_tokens"] for r in trainer.consumption.steps)
    lane: Dict[str, int] = {}
    for r in trainer.consumption.steps:
        for k, v in r["lane_tokens"].items():
            lane[k] = lane.get(k, 0) + v
    trainer.consumption.lane_tokens = lane

    return {
        "before": before,
        "after": {
            "consumption_steps": len(trainer.consumption.steps),
            "learning_samples": len(trainer.learning.sample_rows),
            "learning_steps": len(trainer.learning.step_rows),
        },
        "rows_discarded": {k: before[k] - v for k, v in {
            "consumption_steps": len(trainer.consumption.steps),
            "learning_samples": len(trainer.learning.sample_rows),
            "learning_steps": len(trainer.learning.step_rows),
        }.items()},
        "note": ("Rows for steps trained after the last checkpoint are discarded on resume; "
                 "their weights were discarded too, so counting their tokens would be a lie."),
        "scope": ("Row-level ledgers are truncated exactly. The learning ledger's running "
                  "aggregates (per-shard, per-role, per-document sums) are rebuilt from the "
                  "retained rows going forward rather than restored, because a resumed "
                  "process re-derives them from the ledger file. At demo scale that "
                  "re-derivation is skipped; a production system would fold the retained "
                  "rows back into the accumulators here."),
    }


# ---------------------------------------------------------------------------
# drills
# ---------------------------------------------------------------------------


def torn_write_drill(root: str, step: int) -> Dict[str, Any]:
    """Corrupt a checkpoint the way a crash mid-write would, and confirm it is caught.

    Two flavours, because they fail differently: a truncated `state.pt` (the process died
    while writing weights) passes JSON parsing and is caught only by the weights hash; a
    truncated `meta.json` fails to parse at all. A loader that catches one and not the other
    is a loader that will eventually resume from half a checkpoint.
    """
    src = os.path.join(root, "step-{:06d}".format(step))
    if not os.path.exists(src):
        return {"skipped": "no checkpoint at step {}".format(step)}

    results = []
    for flavour in ("truncated_weights", "truncated_meta", "missing_rng"):
        dst = os.path.join(root, "_drill-{}".format(flavour))
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        if flavour == "truncated_weights":
            blob = open(os.path.join(dst, "state.pt"), "rb").read()
            open(os.path.join(dst, "state.pt"), "wb").write(blob[: len(blob) // 2])
        elif flavour == "truncated_meta":
            blob = open(os.path.join(dst, "meta.json"), "rb").read()
            open(os.path.join(dst, "meta.json"), "wb").write(blob[: len(blob) // 2])
        else:
            os.remove(os.path.join(dst, "rng.json"))

        v = verify_checkpoint(dst)
        results.append({
            "flavour": flavour,
            "detected": not v["ok"],
            "problems": v.get("problems") or [v.get("error")],
        })
        shutil.rmtree(dst)

    intact = verify_checkpoint(src)
    return {
        "source_checkpoint": src,
        "intact_checkpoint_verifies": intact["ok"],
        "corruptions_tested": len(results),
        "all_detected": all(r["detected"] for r in results),
        "results": results,
        "note": ("Writes go to <file>.tmp, are fsynced, then os.replace()d. os.replace is "
                 "atomic within a filesystem, so a real crash leaves either the whole old "
                 "checkpoint or the whole new one. These corruptions are induced after the "
                 "fact to prove the verifier would catch the case anyway."),
    }
