"""Crash, resume, replay, fork, audit -- the four things a ledger is for.

The distinction that matters here is between **resume** and **replay**.

*Resume* continues an interrupted run: restore weights, optimizer, cursors, RNG, truncate
the ledgers to the checkpointed instant, keep going. It is a recovery operation.

*Replay* answers "what exactly did step 431 train on?" months later, and it does so **from
the ledger, not by re-running the sampler**. The session is explicit about why: "if I want
to go back in history and run something, I will not run the code, because some
non-determinism can creep in. I am going to run the ledger." Here the ledger stores the
cursor state on both sides of every step, so replay seeks the loader to the recorded
`before` state and regenerates the batch, then checks the regenerated batch's content hash
against the one recorded at training time. If they differ, something in the path is not
deterministic and the mismatch is reported rather than smoothed over.

*Fork* is the counterfactual: branch from a checkpoint under a different mixture and compare.
Both branches share a ledger prefix, so the comparison is genuinely controlled -- the only
thing that differs after the fork point is the mixture.

*Audit* is the question-answering layer: which steps saw this document, was any evaluation
document ever consumed, what is the chain from a token back to its licence.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .checkpoint import (Binding, load_checkpoint, save_checkpoint,
                         truncate_ledgers, verify_checkpoint)
from .determinism import content_hash
from .documents import SPLIT_TRAIN
from .stream import DataStream
from .trainer import (Trainer, batch_content_hash, batch_id, batch_token_spans,
                      spans_hash)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def replay_steps(
    consumption_rows: Sequence[Dict[str, Any]],
    *,
    registry,
    schedule,
    tokenizer,
    run_id: str,
    steps: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Regenerate the recorded batches from the ledger and check they still hash the same.

    Note what is *not* done: the model is never loaded and no forward pass is run. Replay
    reconstructs what was fed in, which is the question the ledger is asked. Reproducing the
    loss would additionally require the weights at that step, which is what the checkpoints
    are for -- a different question with a different cost.
    """
    by_step = {r["step"]: r for r in consumption_rows}
    targets = list(steps) if steps is not None else sorted(by_step)

    results: List[Dict[str, Any]] = []
    for step in targets:
        row = by_step.get(step)
        if row is None:
            results.append({"step": step, "ok": False, "error": "not in ledger"})
            continue
        cursors = row.get("cursors", {})
        before = cursors.get("before")
        if before is None:
            results.append({"step": step, "ok": False,
                            "error": "ledger row has no 'before' cursor state"})
            continue

        stream = DataStream(registry, schedule, tokenizer, run_id=run_id,
                            verify_sample_masks=False)
        stream.load_state_dict({"run_id": run_id, "cursors": before})
        batch = stream.batch(step)

        recomputed = batch_content_hash(batch)
        spans = batch_token_spans(batch)
        recomputed_spans = spans_hash(spans)
        rid = batch_id(batch)
        shards = sorted({s.shard_id for s in batch.samples})
        docs = sorted({d for s in batch.samples for d in s.seq.doc_ids})

        checks = {
            "batch_id_matches": rid == row["batch_id"],
            "batch_hash_matches": recomputed == row["batch_hash"],
            "spans_hash_matches": recomputed_spans == row["spans_hash"],
            "shards_match": shards == row["shards_read"],
            "documents_match": docs == row["documents"],
            "tokens_match": batch.token_count() == row["tokens"],
            "cursors_after_match": stream.state_dict()["cursors"] == cursors.get("after"),
        }
        results.append({
            "step": step,
            "ok": all(checks.values()),
            "ledger_batch_id": row["batch_id"],
            "replayed_batch_id": rid,
            "ledger_batch_hash": row["batch_hash"][:16],
            "replayed_batch_hash": recomputed[:16],
            "ledger_spans_hash": row["spans_hash"][:16],
            "replayed_spans_hash": recomputed_spans[:16],
            "spans": len(spans),
            "span_sample": spans[:3],
            **checks,
        })

    ok = [r for r in results if r.get("ok")]
    return {
        "what_this_is": (
            "Batches regenerated from the ledger's recorded cursor state, then compared on "
            "batch id, token spans and content hash. The ledger is the source of truth; the "
            "code is only asked to agree with it."),
        "compared_on": [
            "batch_id -- step plus content fingerprint",
            "token spans -- (shard_id, doc_id, doc_start, doc_end) placed at "
            "(seq_start, seq_end), each with its own token hash",
            "batch content hash -- the actual token values",
        ],
        "steps_replayed": len(results),
        "steps_matching": len(ok),
        "all_match": len(ok) == len(results) and bool(results),
        "batch_ids_match": all(r.get("batch_id_matches") for r in results),
        "spans_match": all(r.get("spans_hash_matches") for r in results),
        "hashes_match": all(r.get("batch_hash_matches") for r in results),
        "results": results,
        "mismatches": [r for r in results if not r.get("ok")],
    }


def replay_step_detail(consumption_rows: Sequence[Dict[str, Any]], step: int,
                       *, registry, schedule, tokenizer, run_id: str,
                       max_docs: int = 6) -> Dict[str, Any]:
    """The human-readable answer to 'what did step N actually train on?'."""
    row = next((r for r in consumption_rows if r["step"] == step), None)
    if row is None:
        return {"error": "step {} not in ledger".format(step)}

    stream = DataStream(registry, schedule, tokenizer, run_id=run_id,
                        verify_sample_masks=False)
    stream.load_state_dict({"run_id": run_id, "cursors": row["cursors"]["before"]})
    batch = stream.batch(step)

    docs = []
    for s in batch.samples[:max_docs]:
        shard = registry.get(s.shard_id)
        by_id = {d.doc_id: d for d in shard.documents}
        for doc_id in s.seq.doc_ids[:2]:
            rec = by_id.get(doc_id)
            docs.append({
                "doc_id": doc_id,
                "lane": s.lane,
                "shard_id": s.shard_id,
                "band": s.difficulty_band,
                "packing_policy": s.packing_policy,
                "attention_policy": s.seq.attention_policy,
                "position_policy": s.seq.position_policy,
                "tokens_in_shard": (rec.end - rec.start) if rec else None,
                "text_preview": (
                    tokenizer.decode(
                        [int(t) for t in shard.tokens[rec.start:rec.start + 40]]
                    )[:160] if rec else None),
            })

    return {
        "step": step,
        "stage": row["stage"],
        "seq_len": row["seq_len"],
        "in_seam": row["in_seam"],
        "mean_loss": row["mean_loss"],
        "lr": row["lr"],
        "lane_tokens": row["lane_tokens"],
        "batch_hash_matches": batch_content_hash(batch) == row["batch_hash"],
        "documents": docs,
    }


# ---------------------------------------------------------------------------
# crash and resume
# ---------------------------------------------------------------------------


def crash_resume_drill(
    *,
    make_run,
    binding: Binding,
    ckpt_root: str,
    checkpoint_at: int,
    crash_at: int,
) -> Dict[str, Any]:
    """Train past a checkpoint, kill the process state, resume, and prove the two agree.

    `make_run()` must return a fresh `(model, trainer, stream)` triple built from identical
    inputs -- a fresh *process state*, not a fresh dataset. The drill is:

      1. Train to `checkpoint_at`, save. The ledger keeps being appended to on disk.
      2. Keep training to `crash_at`. Record the weights and the on-disk ledger.
      3. Throw the process away. Build a fresh run, load the checkpoint, **reload the ledger
         from disk**, truncate it back to the checkpoint's offset, and retrain those steps.
      4. Compare. Weights must match bitwise and the ledger must match row for row.

    Step 3's truncation is the part that is easy to omit and expensive to omit, and the
    reason it has to be modelled with a real file is that the discrepancy only exists on
    disk. The ledger was appended to for every step up to the crash; the *weights* only got
    as far as `checkpoint_at`. A resume that keeps all the rows claims the model consumed
    `crash_at - checkpoint_at` steps' worth of tokens that its weights have never seen --
    and it claims it in the one artifact whose whole purpose is to be believed.
    """
    import torch

    ledger_path = os.path.join(ckpt_root, "consumption.jsonl")
    curve_path = os.path.join(ckpt_root, "loss_curve.jsonl")
    os.makedirs(ckpt_root, exist_ok=True)

    def flush(trainer) -> None:
        """The append-only ledger file, rewritten from the in-memory rows."""
        with open(ledger_path, "w") as fh:
            for row in trainer.consumption.steps:
                fh.write(_json_line(row))
        with open(curve_path, "w") as fh:
            for row in trainer.learning.step_rows:
                fh.write(_json_line(row))

    model_a, trainer_a, stream_a = make_run()
    ck_entry = None
    for step in range(crash_at):
        before = stream_a.state_dict()["cursors"]
        batch = stream_a.batch(step)
        after = stream_a.state_dict()["cursors"]
        trainer_a.train_step(batch, cursors={"before": before, "after": after})
        flush(trainer_a)
        if step + 1 == checkpoint_at:
            ck_entry = save_checkpoint(ckpt_root, step=step + 1, trainer=trainer_a,
                                       stream=stream_a, binding=binding)

    if ck_entry is None:
        return {"error": "checkpoint_at must be <= crash_at"}

    ref_weights = {k: v.clone() for k, v in trainer_a.model.state_dict().items()}
    ref_rows = [dict(r) for r in trainer_a.consumption.steps]
    ref_curve = [dict(r) for r in trainer_a.learning.step_rows]
    ledger_on_disk_at_crash = _read_lines(ledger_path)

    # --- the crash: the process is gone, the files are not ---
    model_b, trainer_b, stream_b = make_run()
    restored = load_checkpoint(ck_entry["dir"], trainer=trainer_b, stream=stream_b,
                               binding=binding)
    # A resumed process reads the ledger back from disk -- including the rows written after
    # the checkpoint, which is exactly the state that has to be repaired.
    trainer_b.consumption.steps = [dict(r) for r in ledger_on_disk_at_crash]
    trainer_b.learning.step_rows = _read_lines(curve_path)
    truncation = truncate_ledgers(trainer_b, restored["ledger_offsets"])

    # --- the central claim: the NEXT batch is exactly the expected batch ---
    #
    # Not "a batch of the right shape", and not "the loss looks similar". The batch the
    # resumed loader produces first must be identical to the one the uninterrupted run
    # produced at that step, on three independent levels:
    #
    #   id     -- same step, same content fingerprint
    #   spans  -- same (shard, document, offset range) placed at the same window offsets
    #   hash   -- same actual token values
    #
    # Checking only the id would pass a loader that skipped a batch and relabelled it.
    # Checking only the hash would pass one that produced the right tokens from the wrong
    # documents. The failure mode this is really guarding -- a resumed cursor that is one
    # batch ahead or behind -- shows up as a mismatch on all three at once, which is why
    # all three are recorded rather than just the cheapest.
    expected_row = ref_rows[checkpoint_at]
    peek_stream = make_run()[2]
    peek_stream.load_state_dict({"run_id": stream_b.run_id,
                                 "cursors": restored_cursors(ck_entry["dir"])})
    next_batch = peek_stream.batch(checkpoint_at)
    next_spans = batch_token_spans(next_batch)
    next_batch_proof = {
        "step": checkpoint_at,
        "expected_batch_id": expected_row["batch_id"],
        "resumed_batch_id": batch_id(next_batch),
        "batch_id_matches": batch_id(next_batch) == expected_row["batch_id"],
        "expected_batch_hash": expected_row["batch_hash"],
        "resumed_batch_hash": batch_content_hash(next_batch),
        "batch_hash_matches": batch_content_hash(next_batch) == expected_row["batch_hash"],
        "expected_spans_hash": expected_row["spans_hash"],
        "resumed_spans_hash": spans_hash(next_spans),
        "spans_hash_matches": spans_hash(next_spans) == expected_row["spans_hash"],
        "expected_documents": expected_row["documents"],
        "resumed_documents": sorted({d for s in next_batch.samples for d in s.seq.doc_ids}),
        "expected_shards": expected_row["shards_read"],
        "resumed_shards": sorted({s.shard_id for s in next_batch.samples}),
        "expected_tokens": expected_row["tokens"],
        "resumed_tokens": next_batch.token_count(),
        "span_sample": next_spans[:4],
    }
    next_batch_proof["no_batch_skipped_or_repeated"] = all([
        next_batch_proof["batch_id_matches"],
        next_batch_proof["batch_hash_matches"],
        next_batch_proof["spans_hash_matches"],
        next_batch_proof["resumed_documents"] == next_batch_proof["expected_documents"],
        next_batch_proof["resumed_shards"] == next_batch_proof["expected_shards"],
        next_batch_proof["resumed_tokens"] == next_batch_proof["expected_tokens"],
    ])

    for step in range(checkpoint_at, crash_at):
        before = stream_b.state_dict()["cursors"]
        batch = stream_b.batch(step)
        after = stream_b.state_dict()["cursors"]
        trainer_b.train_step(batch, cursors={"before": before, "after": after})
        flush(trainer_b)

    weight_diffs = []
    for k, v in trainer_b.model.state_dict().items():
        if not torch.equal(v, ref_weights[k]):
            weight_diffs.append({
                "param": k,
                "max_abs_diff": float((v - ref_weights[k]).abs().max()),
            })

    new_rows = trainer_b.consumption.steps
    row_diffs = []
    for a, b in zip(ref_rows, new_rows):
        for field in ("step", "batch_hash", "shards_read", "documents", "tokens",
                      "mean_loss", "lr"):
            if a[field] != b[field]:
                row_diffs.append({"step": a["step"], "field": field,
                                  "original": a[field], "resumed": b[field]})

    return {
        "what_this_is": (
            "A checkpoint is saved, training continues, the run state is destroyed, and the "
            "run is rebuilt from the checkpoint and re-trained over the same steps. Bitwise "
            "weight equality is the claim; ledger row equality is the proof it was the same "
            "data in the same order."),
        "checkpoint_at": checkpoint_at,
        "crash_at": crash_at,
        "checkpoint": ck_entry,
        "checkpoint_verifies": verify_checkpoint(ck_entry["dir"])["ok"],
        "restored": restored,
        "next_batch_proof": next_batch_proof,
        "ledger_rows_on_disk_at_crash": len(ledger_on_disk_at_crash),
        "ledger_truncation": truncation,
        "orphan_rows_repaired": len(ledger_on_disk_at_crash) - checkpoint_at,
        "steps_replayed_after_resume": crash_at - checkpoint_at,
        "weights_bitwise_identical": not weight_diffs,
        "weight_differences": weight_diffs[:5],
        "ledger_rows_original": len(ref_rows),
        "ledger_rows_resumed": len(new_rows),
        "ledger_rows_identical": not row_diffs and len(ref_rows) == len(new_rows),
        "ledger_row_differences": row_diffs[:5],
        "consumption_totals_match": (
            trainer_b.consumption.total_tokens
            == sum(r["tokens"] for r in ref_rows)),
        "loss_curve_original_tail": ref_curve[-3:],
        "loss_curve_resumed_tail": trainer_b.learning.step_rows[-3:],
    }


def restored_cursors(ckpt_dir: str) -> Dict[str, Any]:
    """The loader cursors a checkpoint carries, read without touching the model.

    Used by the next-batch proof so that the peek stream is seeded from the *checkpoint*
    rather than from the already-resumed stream -- otherwise the proof would be checking
    the resumed loader against itself.
    """
    import json as _json
    with open(os.path.join(ckpt_dir, "meta.json")) as fh:
        return _json.load(fh)["data_cursors"]["cursors"]


def _json_line(row: Dict[str, Any]) -> str:
    import json as _json
    return _json.dumps(row, sort_keys=True) + "\n"


def _read_lines(path: str) -> List[Dict[str, Any]]:
    import json as _json
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [_json.loads(ln) for ln in fh if ln.strip()]


# ---------------------------------------------------------------------------
# fork
# ---------------------------------------------------------------------------


def fork_branch(
    *,
    make_branch,
    ckpt_dir: str,
    binding: Binding,
    fork_at: int,
    steps: int,
    branches: Dict[str, Optional[Dict[str, float]]],
) -> Dict[str, Any]:
    """Run two futures from one checkpoint and compare them.

    `branches` maps a branch name to a lane-weight override (None = the run's own mixture).
    `make_branch(override)` returns a fresh `(model, trainer, stream)` whose schedule was
    compiled under that override -- the caller owns how a mixture is altered, because that
    is a plan decision and not a replay mechanism.

    Note that the branch loads the checkpoint with `strict_binding=False`. The schedule hash
    genuinely differs on the reweighted branch, and that difference is the experiment. What
    must still match is the tokenizer and the shard registry, which are checked in the
    returned `binding_drift`: a fork that also changed the data would not be a controlled
    comparison at all.

    The value is not that forking is difficult -- it is that both branches provably share
    everything up to `fork_at`, so a difference after it is attributable to the mixture and
    to nothing else. That is a controlled experiment, which a pair of independent runs with
    different seeds is not.
    """
    out: Dict[str, Any] = {
        "what_this_is": (
            "Two continuations of the same checkpoint under different mixtures. The shared "
            "prefix makes the comparison controlled."),
        "fork_at": fork_at,
        "steps_per_branch": steps,
        "branches": {},
    }

    for name, override in branches.items():
        model, trainer, stream = make_branch(override)
        restored = load_checkpoint(ckpt_dir, trainer=trainer, stream=stream,
                                   binding=binding, strict_binding=False)

        for step in range(fork_at, fork_at + steps):
            before = stream.state_dict()["cursors"]
            batch = stream.batch(step)
            after = stream.state_dict()["cursors"]
            trainer.train_step(batch, cursors={"before": before, "after": after})

        lane = trainer.learning.by_lane
        out["branches"][name] = {
            "lane_weight_override": override,
            "binding_drift": restored["binding_drift"],
            "final_loss": trainer.learning.step_rows[-1]["mean_loss"]
            if trainer.learning.step_rows else None,
            "mean_loss": round(
                sum(r["mean_loss"] for r in trainer.learning.step_rows)
                / max(1, len(trainer.learning.step_rows)), 6),
            "lane_tokens": dict(sorted(trainer.consumption.lane_tokens.items())),
            "loss_by_lane": {k: v.row() for k, v in sorted(lane.items())},
            "ledger_hash": trainer.consumption.report()["ledger_hash"][:16],
        }

    names = list(out["branches"])
    if len(names) == 2:
        a, b = out["branches"][names[0]], out["branches"][names[1]]
        shared = set(a["loss_by_lane"]) & set(b["loss_by_lane"])
        out["comparison"] = {
            "branches": names,
            "mean_loss_delta": round(b["mean_loss"] - a["mean_loss"], 6),
            "ledgers_differ": a["ledger_hash"] != b["ledger_hash"],
            "per_lane_loss_delta": {
                lane: round((b["loss_by_lane"][lane]["mean_loss"] or 0)
                            - (a["loss_by_lane"][lane]["mean_loss"] or 0), 6)
                for lane in sorted(shared)
            },
            "per_lane_token_delta": {
                lane: b["lane_tokens"].get(lane, 0) - a["lane_tokens"].get(lane, 0)
                for lane in sorted(set(a["lane_tokens"]) | set(b["lane_tokens"]))
            },
        }
    return out


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def audit(trainer: Trainer, registry, *, firewall=None,
          sample_docs: int = 3) -> Dict[str, Any]:
    """Questions the ledger can answer that a training script cannot."""
    rows = trainer.consumption.steps

    # Which steps saw a given document, and how many times.
    doc_steps: Dict[str, List[int]] = {}
    for r in rows:
        for d in r["documents"]:
            doc_steps.setdefault(d, []).append(r["step"])

    # The firewall question, asked from the consumption side rather than the build side.
    # This is the check that would catch a leak the build-time firewall missed, because it
    # looks at what was *consumed*, not at what was *approved*.
    consumed_docs = set(doc_steps)
    non_train_consumed = []
    for sid in registry.order:
        m = registry.manifest(sid)
        if m.split == SPLIT_TRAIN:
            continue
        for d in registry.get(sid).documents:
            if d.doc_id in consumed_docs:
                non_train_consumed.append({"doc_id": d.doc_id, "split": m.split,
                                           "shard_id": sid, "steps": doc_steps[d.doc_id]})

    # Provenance chain for a few consumed documents: token -> document -> shard -> licence.
    chains = []
    for r in rows[:1]:
        for sid in r["shards_read"][:sample_docs]:
            m = registry.manifest(sid)
            shard = registry.get(sid)
            doc = shard.documents[0]
            chains.append({
                "question": "where did the tokens of {} come from?".format(doc.doc_id),
                "doc_id": doc.doc_id,
                "shard_id": sid,
                "shard_content_hash": m.content_hash[:12],
                "manifest_hash": m.manifest_hash[:12],
                "tokenizer_hash": m.tokenizer_hash[:12],
                "lane": m.lane,
                "split": m.split,
                "license": doc.license,
                "license_tier": doc.license_tier,
                "dedup_status": m.dedup_status,
                "contamination_status": m.contamination_status,
                "first_trained_at_step": min(doc_steps.get(doc.doc_id, [None]) or [None])
                if doc.doc_id in doc_steps else None,
                "exposures": len(doc_steps.get(doc.doc_id, [])),
            })

    repeats = sorted(((len(v), k) for k, v in doc_steps.items()), reverse=True)[:10]

    return {
        "what_this_is": (
            "Post-hoc questions answered from the ledgers alone: what was consumed, how "
            "often, from where, and under what licence."),
        "steps_audited": len(rows),
        "distinct_documents_consumed": len(consumed_docs),
        "evaluation_leak": {
            "documents_from_non_train_splits_consumed": len(non_train_consumed),
            "clean": not non_train_consumed,
            "detail": non_train_consumed[:5],
            "note": ("Asked from the consumption side. The build-time firewall approves; "
                     "this checks what was actually eaten."),
        },
        "most_repeated_documents": [
            {"doc_id": d, "exposures": n, "steps": doc_steps[d][:8]} for n, d in repeats],
        "provenance_chains": chains,
        "ledger_hash": trainer.consumption.report()["ledger_hash"],
        "firewall_report": firewall.report if firewall is not None else None,
    }
