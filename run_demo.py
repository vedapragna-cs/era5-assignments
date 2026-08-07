#!/usr/bin/env python3
"""One command that builds the ledger, trains on it, crashes, resumes, replays and audits.

    python run_demo.py

Runs offline. No API key, no network, no downloaded dataset. Every artifact lands in
`logs/` as JSON, and `EVIDENCE.md` is regenerated from those artifacts rather than written
by hand -- a claims document that is typed out separately from the run it describes is a
claims document that will eventually disagree with it.

Optional flags:
    --with-api        additionally annotate OPUS-accepted samples with a Claude quality
                      rating. Strictly additive; the run is identical without it.
    --fast            fewer training tokens, for a quick structural check.
    --tokens N        override the training budget.
    --no-train        build and verify the data path only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ledger import advisory as advisory_mod
from ledger import plan as plan_mod
from ledger.checkpoint import (Binding, list_checkpoints, save_checkpoint,
                               torn_write_drill, verify_checkpoint)
from ledger.corpus import build_corpus
from ledger.determinism import (canonical_json, configure_torch_determinism, content_hash,
                                env_fingerprint, git_commit, hash_source_tree)
from ledger.documents import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, documents_fingerprint
from ledger.firewall import run_firewall
from ledger.masks import build_attention_mask, verify_masks
from ledger.mixture import RunShape, compile_schedule
from ledger.model import ModelConfig, TinyLM
from ledger.opus import OpusSelector, build_golden_proxy, golden_set
from ledger.packing import LANE_POLICY, compare_policies
from ledger.plan import Scale
from ledger.registry import build_registry, reload_and_verify
from ledger.replay import audit, crash_resume_drill, fork_branch, replay_step_detail, replay_steps
from ledger.stream import DataStream
from ledger.tokenization import (TokenizerSpec, expected_initial_loss, fertility_report,
                                 save_tokenizer, train_tokenizer)
from ledger.trainer import (TrainConfig, Trainer, batch_tensors,
                            tensors_from_sequences)

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
ARTIFACTS = os.path.join(ROOT, "artifacts")
RUN_ID = "era5-s6-demo"
SEED = 20260801


class Runner:
    """Sequences the phases and writes one JSON artifact per phase."""

    def __init__(self, args) -> None:
        self.args = args
        self.t0 = time.time()
        self.phases: List[Dict[str, Any]] = []
        self.artifacts: Dict[str, str] = {}

    def emit(self, name: str, payload: Dict[str, Any], headline: str) -> Dict[str, Any]:
        path = os.path.join(LOGS, name)
        with open(path, "w") as fh:
            fh.write(canonical_json(payload))
            fh.write("\n")
        elapsed = time.time() - self.t0
        self.phases.append({
            "artifact": name, "headline": headline,
            "elapsed_s": round(elapsed, 2),
            "bytes": os.path.getsize(path),
        })
        self.artifacts[name] = path
        print("  [{:6.1f}s] {:<34} {}".format(elapsed, name, headline))
        return payload

    def banner(self, title: str) -> None:
        print("\n== {} {}".format(title, "=" * max(0, 60 - len(title))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-api", action="store_true",
                    help="annotate OPUS decisions with a Claude quality rating (optional)")
    ap.add_argument("--fast", action="store_true", help="smaller training budget")
    ap.add_argument("--tokens", type=int, default=None, help="training token budget")
    ap.add_argument("--no-train", action="store_true", help="data path only")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(ARTIFACTS, exist_ok=True)
    r = Runner(args)

    total_tokens = args.tokens or (120_000 if args.fast else 500_000)

    # -- 0. environment ----------------------------------------------------
    r.banner("environment")
    det = configure_torch_determinism(SEED)
    src_paths = [os.path.join(ROOT, "ledger", f)
                 for f in sorted(os.listdir(os.path.join(ROOT, "ledger")))
                 if f.endswith(".py")]
    source_hash = hash_source_tree(src_paths + [os.path.join(ROOT, "run_demo.py")])
    r.emit("00_environment.json", {
        "run_id": RUN_ID, "seed": SEED,
        "determinism": det,
        "environment": env_fingerprint(),
        "git_commit": git_commit(),
        "source_tree_hash": source_hash,
        "note": ("The source hash is part of every checkpoint's binding. A checkpoint whose "
                 "code has changed underneath it is refused rather than silently resumed."),
    }, "seed={} source={}".format(SEED, source_hash[:10]))

    # -- 1. the plan inherited from S5 -------------------------------------
    r.banner("plan (inherited from S5)")
    problems = plan_mod.verify_plan()
    # The demo budget in real tokens is TOTAL_TOKENS_B; `factor` divides it down to a run
    # that finishes while you watch it. Shares, seam widths and floors are scale-invariant.
    scale = Scale(factor=plan_mod.TOTAL_TOKENS_B * 1e9 / total_tokens)
    r.emit("01_plan.json", {
        **plan_mod.plan_summary(),
        "verification_problems": problems,
        "plan_verified": not problems,
        "demo_scale": {"factor": scale.factor, "demo_tokens": total_tokens},
        "integrated_lane_shares_main_run": plan_mod.integrated_lane_shares(),
        "planned_lane_epochs": plan_mod.planned_lane_epochs(),
        "mixture_rounding_residue": plan_mod.mixture_rounding_residue(),
    }, "{} stages, {} lanes, plan {}".format(
        len(plan_mod.STAGES), len(plan_mod.LANE_SPEC),
        "verified" if not problems else "FAILED: {}".format(problems)))

    # -- 2. corpus ---------------------------------------------------------
    r.banner("corpus")
    docs, corpus_meta = build_corpus()
    r.emit("02_corpus.json", {
        **corpus_meta,
        "by_split": {s: sum(1 for d in docs if d.split == s)
                     for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)},
        "by_lane": {ln: sum(1 for d in docs if d.lane == ln)
                    for ln in sorted({d.lane for d in docs})},
        "by_band": {b: sum(1 for d in docs if d.difficulty_band == b)
                    for b in sorted({d.difficulty_band for d in docs})},
        "documents_fingerprint": documents_fingerprint(docs),
    }, "{} documents, fingerprint {}".format(
        len(docs), corpus_meta["corpus_fingerprint"][:10]))

    # -- 3. tokenizer ------------------------------------------------------
    r.banner("tokenizer")
    tok = train_tokenizer(docs, TokenizerSpec(vocab_size=1024))
    tok_path = save_tokenizer(tok, os.path.join(ARTIFACTS, "tokenizer.json"))
    fert = fertility_report(tok, docs)
    r.emit("03_tokenizer.json", {
        "tokenizer_id": tok.tokenizer_id,
        "tokenizer_hash": tok.tokenizer_hash,
        "vocab_size": tok.vocab_size,
        "saved_to": os.path.relpath(tok_path, ROOT),
        "trained_on": "train split only",
        "why": ("A tokenizer fitted on held-out text has already seen it. The split is "
                "enforced here, not assumed."),
        "expected_initial_loss": round(expected_initial_loss(tok.vocab_size), 6),
        "fertility": fert,
    }, "vocab={} hash={} indic fertility={}".format(
        tok.vocab_size, tok.tokenizer_hash[:10],
        fert.get("_indic_aggregate", {}).get("fertility_tokens_per_word")))

    # -- 4. firewall -------------------------------------------------------
    r.banner("evaluation firewall")
    fw = run_firewall(docs)
    r.emit("04_firewall.json", {
        **fw.report,
        "admitted": len(fw.admitted),
        "refused": [{"doc_id": d, "reasons": rs} for d, rs in fw.refused],
        "exact_duplicates": [{"dropped": a, "kept": b} for a, b in fw.exact_duplicates],
        "near_duplicates": [{"dropped": a, "kept": b, "jaccard": round(j, 4)}
                            for a, b, j in fw.near_duplicates],
        "quarantined": [{"doc_id": d, "matched_eval_doc": e} for d, e in fw.quarantined],
        "eval_documents": len(fw.eval_documents),
    }, "{} admitted, {} refused, {} dup, {} quarantined".format(
        len(fw.admitted), len(fw.refused),
        len(fw.exact_duplicates) + len(fw.near_duplicates), len(fw.quarantined)))

    # -- 5. shards ---------------------------------------------------------
    r.banner("shards")
    shard_root = os.path.join(ARTIFACTS, "shards")
    if os.path.exists(shard_root):
        shutil.rmtree(shard_root)
    reg = build_registry(fw, tok, root=shard_root, verify_immutability=True)
    reload = reload_and_verify(reg, shard_root, tok)
    summary = reg.summary()
    r.emit("05_shards.json", {
        **summary,
        "build_report": reg.build_report,
        "reload_verification": reload,
        "trainable": len(reg.trainable_ids()),
        "lane_supply_tokens": {ln: reg.lane_supply_tokens(ln)
                               for ln in sorted(plan_mod.LANE_SPEC)},
    }, "{} shards, {} tokens, immutable={} reload_ok={}".format(
        summary["shards"], summary["total_tokens"],
        reg.build_report["immutability_all_identical"], reload["all_ok"]))

    # The run's shape: micro-batch, accumulation, world size, and the two windows. Fixed
    # here because the packing bake-off has to be measured at the windows the run uses.
    shape = RunShape(total_tokens=total_tokens)

    # -- 6. packing bake-off ----------------------------------------------
    r.banner("packing")
    bakeoff = {}
    for lane in sorted(plan_mod.LANE_SPEC):
        ids = reg.by_lane_split.get((lane, SPLIT_TRAIN), [])
        if not ids:
            continue
        # The largest shard in the lane, not the first: a one-document shard compares
        # every policy at zero and says nothing about any of them.
        sid = max(ids, key=lambda s: reg.manifest(s).token_count)
        # At the lane's *own* window. Comparing the long-context lane at 256 would measure
        # a window it never trains in and make its policy look like it drops everything.
        win = shape.long_seq_len if lane == "long_context" else shape.seq_len
        bakeoff[lane] = {
            "shard_id": sid,
            "seq_len": win,
            "documents": reg.manifest(sid).document_count,
            "tokens": reg.manifest(sid).token_count,
            "min_document_tokens": reg.manifest(sid).min_document_tokens,
            "max_document_tokens": reg.manifest(sid).min_seq_len_required,
            "lane_policy": LANE_POLICY[lane],
            **compare_policies(reg.get(sid), win),
        }
    r.emit("06_packing.json", {
        "what_this_is": ("Every policy run against the largest shard of each lane, at that "
                         "lane's own window, so the cost of the policy the lane actually "
                         "uses is a measurement rather than an assertion."),
        "windows": {"base": shape.seq_len, "long_context": shape.long_seq_len},
        "reading_note": (
            "A policy showing 0 sequences deferred every document rather than truncating "
            "any: structure_preserving and the fitting policies refuse to split an atomic "
            "document, so a shard whose documents all exceed the window yields nothing "
            "under them. That is the policy working -- concat_chop's higher efficiency on "
            "the same shard is bought by truncating documents mid-structure, which for an "
            "agentic trace means cutting a tool call in half."),
        "by_lane": bakeoff,
    }, "{} lanes compared across policies".format(len(bakeoff)))

    # -- 7. masks ----------------------------------------------------------
    r.banner("masks")
    sched = compile_schedule(reg, shape, scale)
    stream = DataStream(reg, sched, tok, run_id=RUN_ID)
    probe = stream.batch(0)
    mask_rows = []
    for s in probe.samples[:8]:
        shard = reg.get(s.shard_id)
        mask_rows.append({
            "lane": s.lane, "shard_id": s.shard_id,
            "attention_policy": s.seq.attention_policy,
            "position_policy": s.seq.position_policy,
            "problems": verify_masks(s.seq, shard, tok.eos_id),
            **s.seq.stats(),
        })
    stream = DataStream(reg, sched, tok, run_id=RUN_ID)  # rewound after the probe
    r.emit("07_masks.json", {
        "what_this_is": ("Loss mask, attention mask and position ids checked against the "
                         "packed sequence they describe. Block-diagonal masks are verified "
                         "to actually block cross-document attention, not merely to be "
                         "labelled as doing so."),
        "samples_checked": len(mask_rows),
        "all_clean": all(not m["problems"] for m in mask_rows),
        "rows": mask_rows,
    }, "{} samples, all clean={}".format(
        len(mask_rows), all(not m["problems"] for m in mask_rows)))

    # -- 8. curriculum -----------------------------------------------------
    r.banner("curriculum")
    r.emit("08_curriculum.json", {**sched.report, "schedule_hash": sched.schedule_hash},
           "{} steps, {} seam steps".format(
               len(sched.steps), sum(1 for s in sched.steps if s.in_seam)))

    if args.no_train:
        finish(r, None, None, None)
        return 0

    # -- 9. training -------------------------------------------------------
    r.banner("training")
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=128, n_layers=4, n_heads=4,
                      d_ff=512, max_seq_len=shape.long_seq_len)

    def make_run():
        configure_torch_determinism(SEED)
        m = TinyLM(cfg)
        t = Trainer(m, TrainConfig(), total_steps=len(sched.steps), tokenizer=tok,
                    eos_id=tok.eos_id, pad_id=tok.pad_id)
        st = DataStream(reg, sched, tok, run_id=RUN_ID)
        return m, t, st

    model, trainer, stream = make_run()
    emb = trainer.assert_embeddings_trainable()
    init = trainer.check_initial_loss(stream.batch(0), tok.vocab_size)
    stream = DataStream(reg, sched, tok, run_id=RUN_ID)

    selector = OpusSelector(model, accept_rate=0.25, use_api=args.with_api)
    # Held out once, re-measured every round. See `selection_drift` for why re-measuring
    # matters: a proxy computed once goes stale and its decay would be misread as the data
    # pool degrading.
    golden = golden_set(reg, tok, shape.seq_len)
    opus_every = max(1, len(sched.steps) // 4)
    ckpt_root = os.path.join(ARTIFACTS, "checkpoints")
    if os.path.exists(ckpt_root):
        shutil.rmtree(ckpt_root)
    binding = Binding(
        tokenizer_hash=tok.tokenizer_hash,
        registry_hash=summary["registry_hash"],
        schedule_hash=sched.schedule_hash,
        source_hash=source_hash,
        run_id=RUN_ID,
    )
    ckpt_every = max(1, len(sched.steps) // 3)

    for i, step_plan in enumerate(sched.steps):
        before = stream.state_dict()["cursors"]
        batch = stream.batch(i)
        after = stream.state_dict()["cursors"]

        opus_round = None
        if i > 0 and i % opus_every == 0 and batch.samples and golden:
            proxy = build_golden_proxy(model, [tensors_from_sequences(golden[:4])])
            rnd = selector.select(
                batch.samples, proxy, step=i, stage=batch.stage,
                stage_bands=step_plan.band_gate,
                model_age_tokens=trainer.consumption.total_tokens,
                to_batch=batch_tensors)
            opus_round = rnd.round_index

        trainer.train_step(batch, cursors={"before": before, "after": after},
                           opus_round=opus_round)

        if (i + 1) % ckpt_every == 0 or i + 1 == len(sched.steps):
            save_checkpoint(ckpt_root, step=i + 1, trainer=trainer, stream=stream,
                            binding=binding)

    curve = trainer.loss_curve()
    r.emit("09_consumption_ledger.json", {
        **trainer.consumption.report(),
        "preflight": {"embeddings": emb, "initial_loss": init},
        "steps_detail": trainer.consumption.steps,
    }, "{} steps, {} tokens, loss {:.3f} -> {:.3f}".format(
        len(curve), trainer.consumption.total_tokens,
        curve[0]["mean_loss"], curve[-1]["mean_loss"]))

    learning = trainer.learning.report()
    r.emit("10_learning_ledger.json", {
        **learning,
        "loss_curve": curve,
        "sample_rows": trainer.learning.sample_rows,
        "token_detail": trainer.learning.token_detail,
    }, "{} samples scored, {} documents, EOS ppl {}".format(
        learning["samples_recorded"], learning["documents_scored"],
        learning["eos_prediction"]["perplexity"]))

    # -- 10. OPUS ----------------------------------------------------------
    r.banner("opus selector")
    opus_report = selector.report()
    r.emit("11_opus.json", opus_report,
           "{} rounds, drift={}".format(len(opus_report["rounds"]),
                                        opus_report["drift"]["verdict"].split(":")[0]))
    with open(os.path.join(LOGS, "opus_trail.jsonl"), "w") as fh:
        for rnd in selector.rounds:
            for d in rnd.decisions:
                fh.write(canonical_json({"round": rnd.round_index, "step": rnd.step,
                                         **d.to_dict()}) + "\n")

    # -- 11. checkpoints ---------------------------------------------------
    r.banner("checkpoints")
    entries = list_checkpoints(ckpt_root)
    drill = torn_write_drill(ckpt_root, entries[0]["step"]) if entries else {}
    r.emit("12_checkpoints.json", {
        "what_this_is": ("Checkpoints carry weights, optimizer moments, step count, data "
                         "cursors, RNG state, ledger offsets and the binding hashes of the "
                         "data they were trained on."),
        "checkpoints": entries,
        "verification": [verify_checkpoint(e["dir"]) for e in entries],
        "torn_write_drill": drill,
    }, "{} checkpoints, corruption detected={}".format(
        len(entries), drill.get("all_detected")))

    # -- 12. crash and resume ---------------------------------------------
    r.banner("crash / resume")
    drill_root = os.path.join(ARTIFACTS, "drill-checkpoints")
    if os.path.exists(drill_root):
        shutil.rmtree(drill_root)
    n = len(sched.steps)
    cr = crash_resume_drill(make_run=make_run, binding=binding, ckpt_root=drill_root,
                            checkpoint_at=max(2, n // 3), crash_at=max(4, n // 3 + 4))
    r.emit("13_crash_resume.json", cr,
           "weights identical={} ledger identical={}".format(
               cr.get("weights_bitwise_identical"), cr.get("ledger_rows_identical")))

    # -- 13. replay --------------------------------------------------------
    r.banner("replay")
    rows = trainer.consumption.steps
    picks = sorted({0, len(rows) // 3, len(rows) // 2, len(rows) - 1})
    rep = replay_steps(rows, registry=reg, schedule=sched, tokenizer=tok, run_id=RUN_ID,
                       steps=picks)
    rep["worked_example"] = replay_step_detail(rows, picks[-1], registry=reg,
                                               schedule=sched, tokenizer=tok, run_id=RUN_ID)
    r.emit("14_replay.json", rep,
           "{}/{} steps reproduce bit-for-bit".format(rep["steps_matching"],
                                                      rep["steps_replayed"]))

    # -- 14. fork ----------------------------------------------------------
    r.banner("fork")

    def make_branch(override):
        """A branch is a fresh run whose *schedule* was compiled under the override."""
        configure_torch_determinism(SEED)
        branch_sched = sched if not override else compile_schedule(
            reg, shape, scale, lane_multipliers=override)
        m = TinyLM(cfg)
        t = Trainer(m, TrainConfig(), total_steps=len(branch_sched.steps), tokenizer=tok,
                    eos_id=tok.eos_id, pad_id=tok.pad_id)
        st = DataStream(reg, branch_sched, tok, run_id=RUN_ID)
        return m, t, st

    fork_at = entries[0]["step"] if entries else 2
    fk = fork_branch(make_branch=make_branch, ckpt_dir=entries[0]["dir"], binding=binding,
                     fork_at=fork_at, steps=min(4, n - fork_at),
                     branches={"as_planned": None, "indic_boost": {"indic": 2.5}})
    r.emit("15_fork.json", fk, "2 branches from step {}".format(fork_at))

    # -- 15. audit ---------------------------------------------------------
    r.banner("audit")
    au = audit(trainer, reg, firewall=fw)
    r.emit("16_audit.json", au,
           "eval leak clean={}".format(au["evaluation_leak"]["clean"]))

    # -- 16. v6 advisory ---------------------------------------------------
    r.banner("v6 advisory")
    adv = advisory_mod.build_advisory(trainer, reg, sched)
    r.emit("v6_shard_advisory.json", adv,
           "{} shards scored: {}".format(adv.get("shards_scored"), adv.get("actions")))

    finish(r, trainer, sched, adv)
    return 0


def finish(r: Runner, trainer, sched, adv) -> None:
    elapsed = time.time() - r.t0
    summary = {
        "run_id": RUN_ID,
        "wall_clock_s": round(elapsed, 2),
        "phases": r.phases,
        "artifacts": sorted(r.artifacts),
    }
    if trainer is not None:
        curve = trainer.loss_curve()
        summary["training"] = {
            "steps": len(curve),
            "tokens": trainer.consumption.total_tokens,
            "initial_loss": curve[0]["mean_loss"],
            "final_loss": curve[-1]["mean_loss"],
            "initial_perplexity": curve[0]["perplexity"],
            "final_perplexity": curve[-1]["perplexity"],
            "consumption_ledger_hash": trainer.consumption.report()["ledger_hash"],
        }
    with open(os.path.join(LOGS, "summary.json"), "w") as fh:
        fh.write(canonical_json(summary) + "\n")
    print("\n-- done in {:.1f}s; {} artifacts in logs/".format(elapsed, len(r.artifacts) + 1))

    try:
        from ledger.report import write_evidence
        path = write_evidence(LOGS, ROOT)
        print("-- EVIDENCE.md regenerated from logs/ ({})".format(
            os.path.relpath(path, ROOT)))
    except Exception as exc:
        print("-- EVIDENCE.md not regenerated: {}: {}".format(type(exc).__name__, exc))


if __name__ == "__main__":
    raise SystemExit(main())
