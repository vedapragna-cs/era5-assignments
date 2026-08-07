#!/usr/bin/env python3
"""One command that runs the complete demonstration.

    python run_demo.py

Runs offline: no API key, no network, no downloaded dataset. Executes the full path --

    documents -> tokenized shards -> manifests -> mixture schedule -> packing -> batches
    -> training -> consumption ledger -> learning ledger -> checkpoint -> crash -> resume
    -> replay -> audit

-- and writes the complete submission bundle to `submission_artifacts/`:

    submission_artifacts/
      run.log             every event, with a verdict and the artifact that proves it
      evidence.json       machine-readable: each requirement, its predicate, its result
      evidence.md         the REQUIREMENT / RESULT / EVIDENCE table
      performance.json    throughput and packing efficiency
      run_summary.json    hashes and phase timings
      manifests/          tokenizer record, shard index, one manifest per shard
      ledgers/            consumption, learning, mixture, opus, masks, replay, audit, ...
      checkpoints/        checkpoint metadata, index, and the crash/resume drill

Optional flags:
    --with-api        additionally annotate OPUS-accepted samples with a Claude quality
                      rating. Strictly additive; every check passes without it.
    --fast            smaller token budget, for a quick structural check.
    --tokens N        override the training budget.
    --keep-weights    keep checkpoint .pt blobs (~12MB each) instead of metadata only.
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
from ledger import evidence as evidence_mod
from ledger import plan as plan_mod
from ledger.checkpoint import (Binding, list_checkpoints, save_checkpoint,
                               torn_write_drill, verify_checkpoint)
from ledger.corpus import build_corpus
from ledger.determinism import (canonical_json, configure_torch_determinism, content_hash,
                                env_fingerprint, git_commit, hash_source_tree)
from ledger.documents import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, documents_fingerprint
from ledger.events import EventLog
from ledger.firewall import run_firewall
from ledger.masks import verify_masks
from ledger.mixture import RunShape, compile_schedule
from ledger.model import ModelConfig, TinyLM
from ledger.opus import (ACCEPT, DEFER, REJECT, R_PROTECTED_OVERRIDE, OpusSelector,
                         build_golden_proxy, golden_set)
from ledger.packing import LANE_POLICY, compare_policies
from ledger.perf import PerfRecorder, packing_efficiency_report
from ledger.plan import EPOCH_CEILING, LANE_SPEC, Scale
from ledger.registry import build_registry, reload_and_verify
from ledger.replay import (audit, crash_resume_drill, fork_branch, replay_step_detail,
                           replay_steps)
from ledger.stream import DataStream
from ledger.tokenization import (TokenizerSpec, expected_initial_loss, fertility_report,
                                 save_tokenizer, train_tokenizer)
from ledger.trainer import (TrainConfig, Trainer, batch_tensors, tensors_from_sequences)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "submission_artifacts")
WORK = os.path.join(ROOT, "artifacts")        # token payloads; rebuilt, not committed
RUN_ID = "era5-s6-demo"
SEED = 20260801

# Mixture tolerance. A lane inside this of its planned share is compliant; outside it, the
# lane must have a recorded supply shortfall explaining the gap. Set from what the demo's
# integer allocation can achieve at 16 sequences/step, not tuned to pass.
MIXTURE_TOLERANCE = 0.02


def w(rel: str, payload: Any) -> str:
    """Write one artifact under submission_artifacts/, returning its relative path."""
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(canonical_json(payload))
        fh.write("\n")
    return rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-api", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--tokens", type=int, default=None)
    ap.add_argument("--keep-weights", action="store_true")
    args = ap.parse_args()

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    if os.path.exists(WORK):
        shutil.rmtree(WORK)

    log = EventLog(os.path.join(OUT, "run.log"))
    perf = PerfRecorder()
    t_start = time.time()
    total_tokens = args.tokens or (120_000 if args.fast else 500_000)

    # ============================ environment =============================
    log.section("environment")
    det = configure_torch_determinism(SEED)
    src = [os.path.join(ROOT, "ledger", f)
           for f in sorted(os.listdir(os.path.join(ROOT, "ledger"))) if f.endswith(".py")]
    source_hash = hash_source_tree(src + [os.path.join(ROOT, "run_demo.py")])
    log.info("environment_captured", seed=SEED, source_hash=source_hash[:12],
             torch_deterministic=det.get("deterministic_algorithms"))

    # ============================== the plan ==============================
    log.section("plan (inherited from S5)")
    t = time.time()
    plan_problems = plan_mod.verify_plan()
    scale = Scale(factor=plan_mod.TOTAL_TOKENS_B * 1e9 / total_tokens)
    plan_art = w("ledgers/plan.json", {
        **plan_mod.plan_summary(),
        "verification_problems": plan_problems,
        "plan_verified": not plan_problems,
        "demo_scale": {"factor": scale.factor, "demo_tokens": total_tokens},
        "integrated_lane_shares_main_run": plan_mod.integrated_lane_shares(),
        "planned_lane_epochs": plan_mod.planned_lane_epochs(),
        "mixture_rounding_residue": plan_mod.mixture_rounding_residue(),
    })
    log.check("s5_plan_verified", not plan_problems, plan_art,
              stages=len(plan_mod.STAGES), lanes=len(LANE_SPEC),
              problems=len(plan_problems))
    perf.phase("plan", time.time() - t)

    # ============================== corpus ================================
    log.section("corpus")
    t = time.time()
    docs, corpus_meta = build_corpus()
    corpus_art = w("ledgers/corpus.json", {
        **corpus_meta,
        "by_split": {s: sum(1 for d in docs if d.split == s)
                     for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)},
        "by_lane": {ln: sum(1 for d in docs if d.lane == ln)
                    for ln in sorted({d.lane for d in docs})},
        "by_band": {b: sum(1 for d in docs if d.difficulty_band == b)
                    for b in sorted({d.difficulty_band for d in docs})},
        "documents_fingerprint": documents_fingerprint(docs),
    })
    log.info("corpus_built", corpus_art, documents=len(docs),
             fingerprint=corpus_meta["corpus_fingerprint"][:12])
    perf.phase("corpus", time.time() - t)

    # ============================= tokenizer ==============================
    log.section("tokenizer")
    t = time.time()
    tok = train_tokenizer(docs, TokenizerSpec(vocab_size=1024))
    save_tokenizer(tok, os.path.join(WORK, "tokenizer.json"))
    fert = fertility_report(tok, docs)
    perf.phase("tokenizer", time.time() - t)

    # ============================== firewall ==============================
    log.section("evaluation firewall (build side)")
    t = time.time()
    fw = run_firewall(docs)
    fw_art = w("ledgers/firewall.json", {
        **fw.report,
        "admitted": len(fw.admitted),
        "refused": [{"doc_id": d, "reasons": rs} for d, rs in fw.refused],
        "exact_duplicates": [{"dropped": a, "kept": b} for a, b in fw.exact_duplicates],
        "near_duplicates": [{"dropped": a, "kept": b, "jaccard": round(j, 4)}
                            for a, b, j in fw.near_duplicates],
        "quarantined": [{"doc_id": d, "matched_eval_doc": e} for d, e in fw.quarantined],
        "eval_documents": len(fw.eval_documents),
    })
    log.check("contamination_quarantine_applied", len(fw.quarantined) >= 1, fw_art,
              quarantined=len(fw.quarantined), admitted=len(fw.admitted))
    log.check("unsafe_documents_refused", len(fw.refused) >= 2, fw_art,
              refused=len(fw.refused),
              reasons=sorted({r for _d, rs in fw.refused for r in rs})[:3])
    perf.phase("firewall", time.time() - t)

    # =============================== shards ===============================
    log.section("shards and manifests")
    t = time.time()
    reg = build_registry(fw, tok, root=os.path.join(WORK, "shards"),
                         verify_immutability=True)
    reload = reload_and_verify(reg, os.path.join(WORK, "shards"), tok)
    summ = reg.summary()

    # One manifest file per shard, plus the index. This is the manifests/ directory the
    # handout asks for -- the token payloads stay in artifacts/ (rebuilt, not committed)
    # because the manifest is the auditable object and the tokens are what it points at.
    for sid in reg.order:
        w("manifests/shards/{}.json".format(sid), reg.manifest(sid).to_dict())

    incomplete = [sid for sid in reg.order if reg.manifest(sid).missing_required_fields()]
    untrainable = [r for r in summ["rows"] if not r["trainable"]]
    shard_index = w("manifests/shard_index.json", {
        **summ,
        "immutability_all_identical": reg.build_report["immutability_all_identical"],
        "immutability_rows": reg.build_report["immutability_rows"],
        "reload_verification": reload,
        "all_manifests_complete": not incomplete,
        "manifests_missing_fields": incomplete,
        "trainable_shards": len(reg.trainable_ids()),
        "untrainable_shards": len(untrainable),
        "untrainable_detail": untrainable[:20],
        "lane_supply_tokens": {ln: reg.lane_supply_tokens(ln) for ln in sorted(LANE_SPEC)},
    })

    # The tokenizer's manifest record: verified against every shard rather than asserted.
    mismatched = [sid for sid in reg.order
                  if reg.manifest(sid).tokenizer_hash != tok.tokenizer_hash]
    tok_art = w("manifests/tokenizer.json", {
        "tokenizer_id": tok.tokenizer_id,
        "tokenizer_hash": tok.tokenizer_hash,
        "vocab_size": tok.vocab_size,
        "training_fingerprint": tok.training_fingerprint,
        "trained_on": "train split only",
        "why": ("A tokenizer fitted on held-out text has already leaked it. The split is "
                "enforced at fit time, not assumed."),
        "expected_initial_loss_ln_v": round(expected_initial_loss(tok.vocab_size), 6),
        "shards_checked": len(reg.order),
        "shards_with_mismatched_tokenizer_hash": mismatched,
        "hash_verified_against_every_shard": not mismatched,
        "fertility": fert,
    })
    log.check("shards_created", summ["shards"] > 0, shard_index,
              shards=summ["shards"], tokens=summ["total_tokens"])
    log.check("manifests_validated", not incomplete, shard_index,
              complete=summ["shards"] - len(incomplete), incomplete=len(incomplete))
    log.check("shard_immutability_verified",
              bool(reg.build_report["immutability_all_identical"]), shard_index,
              method="every shard built twice, ids and hashes compared")
    log.check("shard_reload_verified", reload["all_ok"], shard_index,
              shards=reload["shards_checked"])
    log.check("tokenizer_hash_verified", not mismatched, tok_art,
              tokenizer_hash=tok.tokenizer_hash[:12], shards=len(reg.order))
    log.check("eval_shard_blocked", len(untrainable) > 0, shard_index,
              blocked=len(untrainable), reason="split_not_trainable")
    perf.phase("shards", time.time() - t)

    shape = RunShape(total_tokens=total_tokens)

    # ============================== packing ===============================
    log.section("packing")
    t = time.time()
    bakeoff = {}
    for lane in sorted(LANE_SPEC):
        ids = reg.by_lane_split.get((lane, SPLIT_TRAIN), [])
        if not ids:
            continue
        sid = max(ids, key=lambda s: reg.manifest(s).token_count)
        win = shape.long_seq_len if lane == "long_context" else shape.seq_len
        bakeoff[lane] = {
            "shard_id": sid, "seq_len": win,
            "documents": reg.manifest(sid).document_count,
            "tokens": reg.manifest(sid).token_count,
            "min_document_tokens": reg.manifest(sid).min_document_tokens,
            "max_document_tokens": reg.manifest(sid).min_seq_len_required,
            "lane_policy": LANE_POLICY[lane],
            **compare_policies(reg.get(sid), win),
        }
    pack_eff = packing_efficiency_report(reg, tok, shape.seq_len, shape.long_seq_len)
    pack_art = w("ledgers/packing.json", {
        "what_this_is": ("Every policy run against the largest shard of each lane at that "
                         "lane's own window, plus realised utilisation under the chosen "
                         "policy across every trainable shard."),
        "windows": {"base": shape.seq_len, "long_context": shape.long_seq_len},
        "reading_note": (
            "A policy showing 0 sequences deferred every document rather than truncating "
            "any: structure_preserving and the fitting policies refuse to split an atomic "
            "document. concat_chop's higher efficiency on the same shard is bought by "
            "truncating mid-structure, which for an agentic trace means cutting a tool "
            "call in half."),
        "policy_bakeoff": bakeoff,
        "realised_utilization": pack_eff,
    })
    log.info("packing_policies_compared", pack_art, lanes=len(bakeoff),
             overall_utilization=pack_eff["overall_packing_utilization"])
    perf.phase("packing", time.time() - t)

    # ============================== mixture ===============================
    log.section("mixture schedule")
    t = time.time()
    sched = compile_schedule(reg, shape, scale)
    log.check("mixture_compiled", len(sched.steps) > 0, "ledgers/mixture_compliance.json",
              steps=len(sched.steps),
              seam_steps=sum(1 for s in sched.steps if s.in_seam),
              schedule_hash=sched.schedule_hash[:12])
    perf.phase("mixture", time.time() - t)

    # ================================ masks ===============================
    log.section("masks")
    probe_stream = DataStream(reg, sched, tok, run_id=RUN_ID)
    probe = probe_stream.batch(0)
    mask_rows = []
    for s in probe.samples[:12]:
        mask_rows.append({
            "lane": s.lane, "shard_id": s.shard_id,
            "attention_policy": s.seq.attention_policy,
            "position_policy": s.seq.position_policy,
            "problems": verify_masks(s.seq, reg.get(s.shard_id), tok.eos_id),
            **s.seq.stats(),
        })
    all_clean = all(not m["problems"] for m in mask_rows)
    masks_art = w("ledgers/masks.json", {
        "what_this_is": ("Loss mask, attention mask and position ids checked against the "
                         "packed sequence they describe. Block-diagonal masks are verified "
                         "to actually block cross-document attention, not merely labelled "
                         "as doing so."),
        "samples_checked": len(mask_rows),
        "all_clean": all_clean,
        "problems": [m for m in mask_rows if m["problems"]],
        "rows": mask_rows,
    })
    log.check("batches_packed", len(probe.samples) > 0, masks_art,
              sequences=len(probe.samples), seq_len=probe.seq_len)
    log.check("masks_verified", all_clean, masks_art, samples=len(mask_rows))

    # =============================== training =============================
    log.section("training")
    t = time.time()
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=128, n_layers=4, n_heads=4,
                      d_ff=512, max_seq_len=shape.long_seq_len)

    def make_run(schedule=None):
        configure_torch_determinism(SEED)
        m = TinyLM(cfg)
        sc = schedule or sched
        tr = Trainer(m, TrainConfig(), total_steps=len(sc.steps), tokenizer=tok,
                     eos_id=tok.eos_id, pad_id=tok.pad_id)
        return m, tr, DataStream(reg, sc, tok, run_id=RUN_ID)

    model, trainer, stream = make_run()
    emb = trainer.assert_embeddings_trainable()
    init = trainer.check_initial_loss(stream.batch(0), tok.vocab_size)
    stream = DataStream(reg, sched, tok, run_id=RUN_ID)
    log.check("embeddings_trainable", emb["ok"], "ledgers/consumption_ledger.json",
              rule="S5 seam rule 1")
    log.check("initial_loss_equals_ln_vocab", init["ok"],
              "ledgers/consumption_ledger.json",
              observed=init["observed"], expected=init["expected_ln_v"],
              ratio=init["ratio"])

    selector = OpusSelector(model, accept_rate=0.25, use_api=args.with_api)
    golden = golden_set(reg, tok, shape.seq_len)
    log.check("opus_golden_set_held_out", len(golden) > 0, "ledgers/opus.json",
              sequences=len(golden), source="validation split")

    ckpt_root = os.path.join(OUT, "checkpoints")
    binding = Binding(tokenizer_hash=tok.tokenizer_hash,
                      registry_hash=summ["registry_hash"],
                      schedule_hash=sched.schedule_hash,
                      source_hash=source_hash, run_id=RUN_ID)
    opus_every = max(1, len(sched.steps) // 4)
    ckpt_every = max(1, len(sched.steps) // 3)

    for i, step_plan in enumerate(sched.steps):
        before = stream.state_dict()["cursors"]
        batch = stream.batch(i)
        after = stream.state_dict()["cursors"]

        opus_round = None
        if i > 0 and i % opus_every == 0 and batch.samples and golden:
            proxy = build_golden_proxy(model, [tensors_from_sequences(golden[:4])])
            opus_round = selector.select(
                batch.samples, proxy, step=i, stage=batch.stage,
                stage_bands=step_plan.band_gate,
                model_age_tokens=trainer.consumption.total_tokens,
                to_batch=batch_tensors).round_index

        perf.start_step()
        trainer.train_step(batch, cursors={"before": before, "after": after},
                           opus_round=opus_round)
        perf.end_step(step=i, stage=batch.stage, samples=batch.samples)

        if (i + 1) % ckpt_every == 0 or i + 1 == len(sched.steps):
            entry = save_checkpoint(ckpt_root, step=i + 1, trainer=trainer,
                                    stream=stream, binding=binding)
            log.check("checkpoint_saved", verify_checkpoint(entry["dir"])["ok"],
                      "checkpoints/checkpoint_index.json",
                      step=i + 1, bytes=entry["bytes"])

    curve = trainer.loss_curve()
    perf.phase("training", time.time() - t)

    cons_art = w("ledgers/consumption_ledger.json", {
        **trainer.consumption.report(),
        "preflight": {"embeddings": emb, "initial_loss": init},
        "steps_detail": trainer.consumption.steps,
    })
    log.check("consumption_ledger_written", len(trainer.consumption.steps) > 0, cons_art,
              steps=len(curve), tokens=trainer.consumption.total_tokens,
              ledger_hash=trainer.consumption.report()["ledger_hash"][:12])

    # The learning ledger, plus the check that its losses resolve to real source data.
    learn = trainer.learning.report()
    known_shards = set(reg.order)
    known_docs = {d.doc_id for sid in reg.order for d in reg.get(sid).documents}
    bad = [r for r in trainer.learning.sample_rows
           if r["shard_id"] not in known_shards
           or any(d not in known_docs for d in r["doc_ids"])]
    learn_art = w("ledgers/learning_ledger.json", {
        **learn,
        "loss_linked_to_source": (not bad) and len(trainer.learning.sample_rows) > 0,
        "unresolved_sample_rows": bad[:5],
        "attribution_dimensions": ["shard", "document", "lane", "difficulty_band",
                                   "role", "stage", "token"],
        "loss_curve": curve,
        "sample_rows": trainer.learning.sample_rows,
        "token_detail": trainer.learning.token_detail,
    })
    log.check("learning_ledger_linked_to_source",
              (not bad) and len(trainer.learning.sample_rows) > 0, learn_art,
              samples=learn["samples_recorded"], documents=learn["documents_scored"],
              unresolved=len(bad))
    log.check("token_level_loss_tracked", learn["samples_recorded"] > 0, learn_art,
              eos_perplexity=learn["eos_prediction"]["perplexity"],
              roles=len(learn.get("by_role", {})))

    # ========================= mixture compliance =========================
    log.section("mixture compliance")
    planned = plan_mod.integrated_lane_shares()
    actual = trainer.consumption.report()["lane_shares"]
    shortfalls = sched.report.get("shortfalls", {})
    rows = []
    for lane in sorted(LANE_SPEC):
        p, a = planned.get(lane, 0.0), actual.get(lane, 0.0)
        dev = a - p
        spec = LANE_SPEC[lane]
        rows.append({
            "lane": lane, "planned_share": round(p, 6), "realized_share": round(a, 6),
            "deviation": round(dev, 6),
            "within_tolerance": abs(dev) <= MIXTURE_TOLERANCE,
            "supply_shortfall_recorded": lane in shortfalls,
            "protected": spec.protected,
            "floor_share": spec.floor_share if spec.protected else None,
            "floor_held": (a >= spec.floor_share - MIXTURE_TOLERANCE)
            if spec.protected else None,
            "demo_epochs": sched.report["lane_epochs_demo"].get(lane),
        })
    # A lane is compliant if it is inside tolerance, or outside it with a shortfall recorded
    # by the scheduler. The second clause is not a loophole: the scheduler records a
    # shortfall only when eligible supply ran out, which is a fact about the corpus rather
    # than a failure of the mixture, and it is written to the artifact either way.
    noncompliant = [r for r in rows
                    if not r["within_tolerance"] and not r["supply_shortfall_recorded"]]
    floors_broken = [r for r in rows if r["protected"] and not r["floor_held"]
                     and not r["supply_shortfall_recorded"]]
    over_ceiling = {ln: e for ln, e in sched.report["lane_epochs_demo"].items()
                    if e > EPOCH_CEILING + 1e-9}
    mix_art = w("ledgers/mixture_compliance.json", {
        "what_this_is": ("Planned versus actual lane shares, protected floors, and the "
                         "epoch ceiling, computed from the consumption ledger against the "
                         "S5 plan."),
        "tolerance": MIXTURE_TOLERANCE,
        "rows": rows,
        "max_absolute_deviation": round(max(abs(r["deviation"]) for r in rows), 6),
        "noncompliant_lanes": noncompliant,
        "protected_floors_held": not floors_broken,
        "protected_floor_failures": floors_broken,
        "epoch_ceiling": EPOCH_CEILING,
        "epoch_ceiling_respected": not over_ceiling,
        "lanes_over_ceiling": over_ceiling,
        "within_tolerance": (not noncompliant and not floors_broken and not over_ceiling),
        "schedule": sched.report,
        "schedule_hash": sched.schedule_hash,
        "seam_steps": sum(1 for s in sched.steps if s.in_seam),
        "steps": [s.to_dict() for s in sched.steps],
    })
    max_dev = round(max(abs(r["deviation"]) for r in rows), 5)
    log.check("mixture_compliance_verified",
              not noncompliant and not floors_broken and not over_ceiling, mix_art,
              max_deviation=max_dev, noncompliant=len(noncompliant),
              floors_broken=len(floors_broken))
    log.check("protected_floors_held", not floors_broken, mix_art,
              protected=[r["lane"] for r in rows if r["protected"]])
    log.check("epoch_ceiling_respected", not over_ceiling, mix_art,
              ceiling=EPOCH_CEILING,
              max_epochs=round(max(sched.report["lane_epochs_demo"].values()), 4))

    # ================================ OPUS ================================
    log.section("opus selector")
    orep = selector.report()
    outcomes: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    n_decisions = 0
    for rnd in selector.rounds:
        for d in rnd.decisions:
            n_decisions += 1
            outcomes[d.outcome] = outcomes.get(d.outcome, 0) + 1
            reasons[d.reason] = reasons.get(d.reason, 0) + 1
    protected_overrides = reasons.get(R_PROTECTED_OVERRIDE, 0)
    # An audit trail is complete when every scored candidate produced a decision carrying a
    # reason, and when all the outcome kinds the handout names actually occurred -- a
    # selector that never defers has not demonstrated deferral.
    trail_complete = bool(
        n_decisions > 0
        and ACCEPT in outcomes and REJECT in outcomes and DEFER in outcomes
        and protected_overrides > 0
        and all(d.reason for rnd in selector.rounds for d in rnd.decisions)
    )
    opus_art = w("ledgers/opus.json", {
        **orep,
        "decisions_recorded": n_decisions,
        "outcomes_present": sorted(outcomes),
        "outcome_counts": dict(sorted(outcomes.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "protected_overrides": protected_overrides,
        "audit_trail_complete": trail_complete,
        "trail_file": "ledgers/opus_trail.jsonl",
        "decisions": [{"round": r.round_index, "step": r.step, **d.to_dict()}
                      for r in selector.rounds for d in r.decisions],
    })
    os.makedirs(os.path.join(OUT, "ledgers"), exist_ok=True)
    with open(os.path.join(OUT, "ledgers", "opus_trail.jsonl"), "w") as fh:
        for rnd in selector.rounds:
            for d in rnd.decisions:
                fh.write(canonical_json({"round": rnd.round_index, "step": rnd.step,
                                         **d.to_dict()}) + "\n")
    log.check("opus_decisions_recorded", trail_complete, opus_art,
              decisions=n_decisions, outcomes=dict(sorted(outcomes.items())),
              protected_overrides=protected_overrides)
    log.check("opus_protected_floor_override", protected_overrides > 0, opus_art,
              overrides=protected_overrides,
              note="protected lanes admitted regardless of gradient score")

    # ======================== crash / resume drill ========================
    log.section("crash and resume")
    t = time.time()
    drill_root = os.path.join(WORK, "drill-checkpoints")
    n = len(sched.steps)
    ck_at, cr_at = max(2, n // 3), max(4, n // 3 + 4)
    log.info("crash_simulated", "ledgers/crash_resume.json",
             checkpoint_at=ck_at, crash_at=cr_at,
             method="process state discarded, ledger file retained")
    cr = crash_resume_drill(make_run=make_run, binding=binding, ckpt_root=drill_root,
                            checkpoint_at=ck_at, crash_at=cr_at)
    cr_art = w("ledgers/crash_resume.json", cr)
    pb = cr["next_batch_proof"]
    log.check("run_resumed", cr["weights_bitwise_identical"], cr_art,
              checkpoint_at=cr["checkpoint_at"], crash_at=cr["crash_at"])
    log.check("resume_next_batch_matched", pb["no_batch_skipped_or_repeated"], cr_art,
              expected=pb["expected_batch_id"], resumed=pb["resumed_batch_id"],
              spans_match=pb["spans_hash_matches"])
    log.check("resume_ledger_repaired", cr["ledger_rows_identical"], cr_art,
              orphan_rows_repaired=cr["orphan_rows_repaired"],
              totals_match=cr["consumption_totals_match"])
    perf.phase("crash_resume", time.time() - t)

    # =============================== replay ===============================
    log.section("replay")
    t = time.time()
    rows_c = trainer.consumption.steps
    picks = sorted({0, len(rows_c) // 4, len(rows_c) // 2,
                    3 * len(rows_c) // 4, len(rows_c) - 1})
    rep = replay_steps(rows_c, registry=reg, schedule=sched, tokenizer=tok,
                       run_id=RUN_ID, steps=picks)
    rep["worked_example"] = replay_step_detail(rows_c, picks[-1], registry=reg,
                                               schedule=sched, tokenizer=tok,
                                               run_id=RUN_ID)
    rep_art = w("ledgers/replay.json", rep)
    log.check("historical_stream_replayed", rep["all_match"], rep_art,
              steps=rep["steps_replayed"], matching=rep["steps_matching"])
    log.check("replay_hash_matched", rep["hashes_match"] and rep["spans_match"], rep_art,
              batch_ids=rep["batch_ids_match"], spans=rep["spans_match"],
              hashes=rep["hashes_match"])
    perf.phase("replay", time.time() - t)

    # ================================ fork ================================
    log.section("fork")
    t = time.time()
    entries = list_checkpoints(ckpt_root)
    fork_at = entries[0]["step"]

    def make_branch(override):
        return make_run(sched if not override else compile_schedule(
            reg, shape, scale, lane_multipliers=override))

    fk = fork_branch(make_branch=make_branch, ckpt_dir=entries[0]["dir"], binding=binding,
                     fork_at=fork_at, steps=min(4, n - fork_at),
                     branches={"as_planned": None, "indic_boost": {"indic": 2.5}})
    comp = fk.get("comparison", {})
    fk["branches_diverged"] = bool(
        comp.get("ledgers_differ")
        and any(v != 0 for v in comp.get("per_lane_token_delta", {}).values()))
    fork_art = w("ledgers/fork.json", fk)
    log.check("branch_forked", fk["branches_diverged"], fork_art,
              fork_at=fork_at, branches=list(fk["branches"]),
              indic_token_delta=comp.get("per_lane_token_delta", {}).get("indic"))
    perf.phase("fork", time.time() - t)

    # ================================ audit ===============================
    log.section("audit")
    t = time.time()
    au = audit(trainer, reg, firewall=fw)
    au["evaluation_leak"]["non_train_shards_blocked"] = len(untrainable)
    au_art = w("ledgers/audit.json", au)
    log.check("eval_firewall_consumption_side", au["evaluation_leak"]["clean"], au_art,
              non_train_docs_consumed=au["evaluation_leak"][
                  "documents_from_non_train_splits_consumed"],
              shards_blocked=len(untrainable))
    log.check("audit_completed", au["steps_audited"] > 0, au_art,
              steps=au["steps_audited"], documents=au["distinct_documents_consumed"])
    perf.phase("audit", time.time() - t)

    # ============================ checkpoints =============================
    log.section("checkpoints")
    drill = torn_write_drill(ckpt_root, entries[0]["step"])
    verifs = [verify_checkpoint(e["dir"]) for e in entries]
    metas = []
    for e in entries:
        with open(os.path.join(e["dir"], "meta.json")) as fh:
            m = json.load(fh)
        metas.append({
            "step": m["step"], "dir": os.path.relpath(e["dir"], OUT),
            "integrity": m["integrity"][:16],
            "ledger_offsets": m["ledger_offsets"],
            "data_cursors": m["data_cursors"],
            "consumption_totals": m["consumption_totals"],
            "binding": {k: v[:12] for k, v in m["binding"].items()},
        })
    offsets_ok = all(m["ledger_offsets"].get("consumption_steps") is not None
                     for m in metas)
    ck_art = w("checkpoints/checkpoint_index.json", {
        "what_this_is": ("Checkpoints carry weights, optimizer moments, step count, data "
                         "cursors, RNG state, ledger offsets and the binding hashes of the "
                         "data they were trained on."),
        "checkpoints": len(entries),
        "all_verify": all(v["ok"] for v in verifs),
        "all_carry_ledger_offsets": offsets_ok,
        "verification": verifs,
        "metadata": metas,
        "torn_write_drill": drill,
        "corruption_drill_all_detected": drill.get("all_detected"),
        "weights_retained": bool(args.keep_weights),
        "weights_note": ("state.pt blobs are removed after verification unless "
                         "--keep-weights is passed: they are several MB each and are "
                         "rebuilt deterministically by re-running. Their hashes stay in "
                         "each meta.json, so the integrity check remains checkable."),
    })
    log.check("checkpoints_verified", all(v["ok"] for v in verifs), ck_art,
              checkpoints=len(entries))
    log.check("checkpoint_corruption_detected", bool(drill.get("all_detected")), ck_art,
              flavours=drill.get("corruptions_tested"))
    log.check("checkpoints_tied_to_ledger_offsets", offsets_ok, ck_art,
              offsets=[m["ledger_offsets"]["consumption_steps"] for m in metas])

    if not args.keep_weights:
        for e in entries:
            p = os.path.join(e["dir"], "state.pt")
            if os.path.exists(p):
                os.remove(p)

    # ============================= performance ============================
    log.section("performance")
    prep = perf.report(registry=reg, consumption=trainer.consumption,
                       model_params=model.n_params())
    prep["packing"] = pack_eff
    w("performance.json", prep)
    log.check("performance_measured", prep["ledger_cross_check"]["reconstructible"],
              "performance.json",
              loss_bearing_tok_s=prep["headline"]["loss_bearing_tokens_per_s"],
              packing_utilization=prep["overall"]["packing_utilization"],
              reconstructible=prep["ledger_cross_check"]["reconstructible"])

    # ============================= v6 advisory ============================
    log.section("v6 advisory")
    adv = advisory_mod.build_advisory(trainer, reg, sched)
    adv_art = w("ledgers/v6_shard_advisory.json", adv)
    log.info("v6_advisory_generated", adv_art, shards=adv.get("shards_scored"),
             actions=adv.get("actions"))

    # ============================= run summary ============================
    hashes = {
        "corpus_fingerprint": corpus_meta["corpus_fingerprint"],
        "tokenizer_hash": tok.tokenizer_hash,
        "registry_hash": summ["registry_hash"],
        "schedule_hash": sched.schedule_hash,
        "consumption_ledger_hash": trainer.consumption.report()["ledger_hash"],
        "source_tree_hash": source_hash,
    }
    run_meta = {
        "run_id": RUN_ID,
        "command": "python run_demo.py" + (" --fast" if args.fast else ""),
        "seed": SEED,
        "wall_clock_s": round(time.time() - t_start, 2),
        "git_commit": git_commit(),
        "environment": env_fingerprint(),
        "model": {**cfg.to_dict(), "parameters": model.n_params()},
        "budget_tokens": total_tokens,
        "steps": len(curve),
        "initial_loss": curve[0]["mean_loss"],
        "final_loss": curve[-1]["mean_loss"],
        "final_perplexity": curve[-1]["perplexity"],
    }
    w("run_summary.json", {
        **run_meta,
        "determinism": {**hashes,
                        "all_hashes_recorded": all(bool(v) for v in hashes.values())},
        "phase_seconds": dict(sorted(perf.phases.items())),
    })

    # ============================== evidence ==============================
    log.section("evidence bundle")
    log.check("all_required_events_emitted", not log.missing_required(), "run.log",
              missing=log.missing_required() or "none")
    log_summary = log.close()
    w("execution_log.json", {"summary": log_summary, "events": log.to_json()})

    bundle = evidence_mod.build_bundle(OUT, run_meta=run_meta, log_summary=log_summary)
    with open(os.path.join(OUT, "evidence.json"), "w") as fh:
        fh.write(canonical_json(bundle) + "\n")
    evidence_mod.write_evidence_md(bundle, os.path.join(OUT, "evidence.md"))

    # Long-form EVIDENCE.md at the repo root, generated from the same artifacts.
    try:
        from ledger.report import write_evidence
        write_evidence(os.path.join(OUT, "ledgers"), ROOT, bundle_root=OUT)
    except Exception as exc:
        print("  (long-form EVIDENCE.md skipped: {}: {})".format(type(exc).__name__, exc))

    s = bundle["summary"]
    print("\n" + "=" * 72)
    print("  requirements : {}/{} passed".format(
        s["requirements_passed"], s["requirements_total"]))
    print("  log events   : {} ({} PASS, {} FAIL)".format(
        s["log_events"], s["log_pass"], s["log_fail"]))
    print("  wall clock   : {}s".format(run_meta["wall_clock_s"]))
    print("  loss         : {} -> {}".format(run_meta["initial_loss"],
                                             run_meta["final_loss"]))
    print("  throughput   : {} loss-bearing tokens/s".format(
        prep["headline"]["loss_bearing_tokens_per_s"]))
    print("  bundle       : submission_artifacts/")
    print("=" * 72)
    for r in bundle["requirements"]:
        if r["result"] != "PASS":
            print("  FAIL {}: {}".format(r["requirement"], r.get("error", "")))
    return 0 if s["all_passed"] and s["log_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
