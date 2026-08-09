#!/usr/bin/env python3
"""Independent verifier: recompute every reported number from the artifacts alone.

    python verify_evidence.py [submission_artifacts]

This is the grader's step 2 made executable. It deliberately does NOT import the pipeline that
produced the run - it reads `submission_artifacts/` as data and recomputes the claims from it, so a
number that cannot be reconstructed is reported as unverifiable rather than accepted.

Exit code 0 only if every check reconstructs.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


def sha256_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


class Check:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))
        print(f"  [{'OK ' if ok else 'BAD'}] {name}" + (f" — {detail}" if detail else ""))

    @property
    def failed(self) -> list[str]:
        return [n for n, ok, _ in self.rows if not ok]


def main(root: Path) -> int:
    print(f"Verifying {root}\n")
    c = Check()

    ev = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    perf = json.loads((root / "performance.json").read_text(encoding="utf-8"))
    ledger = load_jsonl(root / "ledgers" / "consumption.jsonl")
    opus = load_jsonl(root / "ledgers" / "opus.jsonl")
    manifests = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted((root / "manifests" / "shards").glob("*.json"))]
    tok = json.loads((root / "manifests" / "tokenizer.json").read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- ledger hash chain
    prev = "0" * 64
    broken = None
    for i, e in enumerate(ledger):
        payload = {k: v for k, v in e.items() if k not in ("prev_hash", "entry_hash")}
        if e["prev_hash"] != prev or sha256_json({"payload": payload, "prev_hash": e["prev_hash"]}) != e["entry_hash"]:
            broken = i
            break
        prev = e["entry_hash"]
    c("ledger_hash_chain_recomputes", broken is None,
      f"{len(ledger)} entries" if broken is None else f"first break at entry {broken}")

    # ---------------------------------------------------------------- shard content hashes
    # Recomputed from the shipped token bytes, not read from the manifest. This is the immutability
    # claim's only independently checkable form: previously the bundle carried the hash but not the
    # bytes, so `content_hash` could only be confirmed by re-running the build.
    tok_dir = root / "manifests" / "shard_tokens"
    bad_ch = []
    for m in manifests:
        f = tok_dir / f"{m['shard_id']}.bin"
        if not f.exists():
            bad_ch.append(f"{m['shard_id']}: token bytes missing")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != m["content_hash"]:
            bad_ch.append(f"{m['shard_id']}: sha256 mismatch")
    c("content_hash_recomputes_from_the_shipped_token_bytes", not bad_ch,
      f"{len(manifests)} shards, {sum((tok_dir / (m['shard_id'] + '.bin')).stat().st_size for m in manifests if (tok_dir / (m['shard_id'] + '.bin')).exists())} bytes"
      if not bad_ch else "; ".join(bad_ch))

    # ---------------------------------------------------------------- tokenizer consistency
    hashes = {m["tokenizer_hash"] for m in manifests}
    c("all_manifests_share_one_tokenizer_hash", len(hashes) == 1 and tok["hash"] in hashes,
      f"{len(manifests)} manifests, {len(hashes)} distinct hash(es)")

    # ---------------------------------------------------------------- batch hashes
    superseded: set[int] = set()
    for e in ledger:
        if e.get("event") == "batches_superseded":
            superseded.update(e.get("ledger_offsets", []))
    batches = [e for e in ledger
               if e.get("event") == "batch_committed" and e["ledger_offset"] not in superseded]
    # Mixture compliance is a per-branch claim: run-b is an intentional fork with a different data
    # policy, so folding it into the parent's shares would compare two different experiments.
    main_branch = [e for e in batches if e.get("branch_id") == "run-a"]
    bad = [e["global_step"] for e in batches
           if sha256_json({"samples": e["packed_sample_ids"], "spans": e["token_span_ids"],
                           "loss_mask": e["loss_mask_hash"]}) != e["batch_hash"]]
    c("every_batch_hash_recomputes_from_its_own_fields", not bad,
      f"{len(batches)} batches" if not bad else f"mismatched at steps {bad[:5]}")

    # ---------------------------------------------------------------- mixture compliance
    plan = json.loads((root / "manifests" / "plan.json").read_text(encoding="utf-8"))
    tally: dict[str, int] = {}
    for e in main_branch:
        for lane, n in e["mixture_lanes"].items():
            tally[lane] = tally.get(lane, 0) + n
    total = sum(tally.values())
    delivered = {k: v / total for k, v in tally.items()} if total else {}
    floors = plan["protected_floors"]
    breached = {ln: round(delivered.get(ln, 0.0), 4) for ln, f in floors.items()
                if delivered.get(ln, 0.0) < f - 1e-9}
    c("protected_floors_hold_in_the_delivered_stream", not breached,
      f"delivered={{{', '.join(f'{k}:{v:.3f}' for k, v in sorted(delivered.items()))}}}"
      if not breached else f"breached {breached}")

    claimed = ev_req(ev, "mixture_compliance")
    if claimed:
        cd = claimed["compared_values"]["delivered_shares"]
        agree = all(abs(cd.get(k, 0) - delivered.get(k, 0)) < 1e-3 for k in cd)
        c("evidence_delivered_shares_match_the_ledger", agree,
          "recomputed independently from mixture_lanes")

    # ---------------------------------------------------------------- throughput
    steps = perf.get("steps_detail", [])
    raw = sum(s["n_tokens"] for s in steps)
    lb = sum(s["n_loss_bearing"] for s in steps)
    pad = sum(s["n_pad"] for s in steps)
    wall = sum(s["seconds"] for s in steps)
    c("raw_tokens_reconstructs", raw == perf["raw_tokens"], f"{raw} vs {perf['raw_tokens']}")
    c("loss_bearing_tokens_reconstructs", lb == perf["loss_bearing_tokens"],
      f"{lb} vs {perf['loss_bearing_tokens']}")
    util = (raw - pad) / raw if raw else 0
    c("packing_utilization_reconstructs", abs(util - perf["packing_utilization"]) < 1e-6,
      f"{util:.6f} vs {perf['packing_utilization']}")
    tps = lb / wall if wall else 0
    c("useful_tokens_per_sec_reconstructs",
      abs(tps - perf["useful_loss_bearing_tokens_per_sec"]) < max(1.0, tps * 1e-3),
      f"{tps:.2f} vs {perf['useful_loss_bearing_tokens_per_sec']}")

    # ---------------------------------------------------------------- OPUS
    # The rejection rate is over the samples OPUS *judged*. Anneal-reserve deferrals share the
    # decision log but carry a null score because they were never scored - a deferral says the stage
    # is wrong, not that the sample lost. Counting them here would inflate the denominator and make
    # the rate incomparable to the plan's target. Independently recomputed from opus_score, not
    # copied from the summary's own bookkeeping.
    # Scope to the same branch the summary declares, but read the branch from each record rather
    # than trusting the summary's own label. Replay, forks and ablations re-judge the same steps.
    judged = [d for d in opus if d["opus_score"] is not None
              and d.get("branch_id", "run-a") == "run-a"]
    kept = sum(1 for d in judged if d["decision"] in ("accepted", "protected"))
    rate = 1 - kept / len(judged) if judged else 0
    osum = json.loads((root / "ledgers" / "opus_summary.json").read_text(encoding="utf-8"))
    c("opus_summary_declares_its_scope", osum.get("scope") == "branch run-a",
      f"scope={osum.get('scope')!r} (the rate is meaningless without it)")
    # tolerance is half a unit in the stored precision: the summary rounds to 4 dp, so an exact
    # equality test would fail on a correct value. (It did.)
    c("opus_rejection_rate_reconstructs", abs(rate - osum["observed_rejection_rate"]) <= 5e-5,
      f"recomputed {rate:.6f} vs stored {osum['observed_rejection_rate']} (stored at 4 dp)")
    served = {s for e in batches for s in e["packed_sample_ids"]}
    rejected_only = {d["sample_id"] for d in opus if d["decision"] == "rejected"} - {
        d["sample_id"] for d in opus if d["decision"] in ("accepted", "protected")}
    leaked = served & rejected_only
    c("no_rejected_only_sample_was_ever_served", not leaked, f"{len(leaked)} leaked")

    # ---------------------------------------------------------------- per-role attribution
    # Recomputed from the shipped token records, not read from the evidence bundle's own table.
    # Two independent things are checked and they fail differently: the per-role perplexities must
    # reproduce (arithmetic), and no context role may carry loss (alignment). The second is the one
    # that catches a role array shifted against its loss mask - the table would still sum correctly.
    SUPERVISED = {"assistant_plan", "tool_call", "final_answer", "text", "boundary"}
    tokrecs = [r for r in load_jsonl(root / "ledgers" / "learning_tokens.jsonl")
               if r.get("branch_id", "run-a") == "run-a" and not r.get("superseded")]
    by_role: dict[str, list[float]] = {}
    for r in tokrecs:
        by_role.setdefault(r["segment_role"], []).append(r["cross_entropy_loss"])
    _lt = next(r for r in ev["requirements"] if r["key"] == "learning_trace")
    stored_roles = _lt["compared_values"]["per_role_geometric_mean_ppl"]
    recomputed = {k: round(math.exp(min(sum(v) / len(v), 30.0)), 4)
                  for k, v in by_role.items()}
    # Tolerance derived from the storage tier rather than picked to make the check pass. The
    # `quantized` tier writes each loss rounded to 2 dp, so a geometric mean rebuilt from the file
    # can sit up to exp(0.005)-1 = 0.5% away from the one computed on unrounded losses; the `full`
    # tier writes 6 dp and the reconstruction is exact to well within 1e-4. Stating which tier
    # produced the file is what makes the looser bound a fact about the artifact rather than a
    # concession. A misaligned role array moves these perplexities by whole multiples (boundary
    # sits near 38, text near 191), so neither bound is loose enough to hide one.
    vol = json.loads((root / "reports" / "ledger_volume.json").read_text(encoding="utf-8"))
    quantized = vol.get("tier_used") == "quantized"
    tol = 0.005 if quantized else 1e-4          # relative
    c("per_role_perplexity_recomputes_from_token_records",
      set(recomputed) == set(stored_roles)
      and all(abs(recomputed[k] - stored_roles[k]) <= stored_roles[k] * tol + 1e-4
              for k in stored_roles),
      f"{len(recomputed)} roles from the '{vol.get('tier_used')}' tier "
      f"(tolerance {tol:.1%}), recomputed={recomputed}")
    off_role = sorted({r for r in by_role if r not in SUPERVISED})
    c("no_context_role_carries_training_loss", not off_role,
      f"roles carrying loss that are not supervised: {off_role or 'none'} "
      f"(a context role here means role_codes drifted against loss_mask)")

    # ---------------------------------------------------------------- firewall
    reg = json.loads((root / "manifests" / "eval_registry.json").read_text(encoding="utf-8"))
    never_train = {r["doc_id"] for r in reg["registered"]}
    docs_in_spans = {sp.split(":")[0] for e in batches for sp in e["token_span_ids"]}
    c("no_registered_eval_document_appears_in_any_served_batch",
      not (never_train & docs_in_spans),
      f"{len(never_train)} registered, {len(docs_in_spans)} distinct documents served")

    # ---------------------------------------------------------------- admission
    adm = json.loads((root / "manifests" / "admission.json").read_text(encoding="utf-8"))
    # The FULL rule, mirroring tds.shards.admit. This encoded five hard fields and none of the
    # value-based blockers, so it would have reported a legitimately blocked shard as an
    # inconsistency in the implementation. It never diverged here only because all 7 shards are
    # admitted - an independent verifier that agrees by luck is not independent.
    hard = ("tokenizer_hash", "cleaning_pipeline_hash", "dedup_status", "pii_screen_status",
            "eval_overlap_status", "repeat_ngram_status")
    wrong = []
    for m in manifests:
        should_block = (
            any(m.get(f) in (None, "", "missing", "unknown", "blocked_or_unknown") for f in hard)
            or m.get("license_tier") == "unsafe"
            or m.get("dedup_status") == "contains_duplicates"
            or m.get("repeat_ngram_status") == "contains_repeats"
            or not m.get("language_validated", False)
            or m.get("split") != "train")
        rec = next((a for a in adm if a["shard_id"] == m["shard_id"]), None)
        if rec and rec["admitted"] == should_block:
            wrong.append(m["shard_id"])
    c("admission_decisions_recompute_from_manifest_fields", not wrong,
      f"{len(manifests)} manifests, 6 hard fields + 4 value blockers"
      if not wrong else f"inconsistent: {wrong}")

    # ---------------------------------------------------------------- eval contamination, rescanned
    # The registry-membership check above cannot fail by construction (non-train splits never reach
    # a shard). This recomputes 13-gram containment from the stored text instead, so the firewall
    # claim is verifiable from the bundle by an implementation the pipeline never runs.
    scan = json.loads((root / "manifests" / "eval_content_scan.json").read_text(encoding="utf-8"))

    def ngrams(text: str, n: int = 13) -> set[str]:
        w = text.lower().split()
        return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)} if len(w) >= n else set()

    ref: dict[str, str] = {}
    for did, text in sorted(scan["registry_text"].items()):
        for g in ngrams(text):
            ref.setdefault(g, did)
    recomputed = sorted(d for d, text in scan["served_text"].items() if ngrams(text) & set(ref))
    c("no_served_document_contains_a_13gram_of_eval_data", not recomputed,
      f"{len(scan['served_text'])} served vs {len(scan['registry_text'])} registered"
      if not recomputed else f"contaminated: {recomputed}")
    c("content_scan_agrees_with_the_recorded_verdict",
      recomputed == sorted(scan["contaminated"]),
      "recomputed independently from the stored text, not read from `contaminated`")

    # ---------------------------------------------------------------- evidence self-consistency
    n_fail = sum(1 for r in ev["requirements"] if r["result"] == "FAIL")
    c("evidence_summary_matches_its_own_rows",
      ev["summary"]["n_failed"] == n_fail and
      ev["summary"]["all_passed"] == (n_fail == 0), f"{n_fail} FAIL rows")
    c("no_requirement_lacks_compared_values",
      all(r.get("compared_values") for r in ev["requirements"]),
      "every PASS carries the values it compared")

    print()
    if c.failed:
        print(f"UNVERIFIED: {len(c.failed)} check(s) did not reconstruct: {c.failed}")
        return 1
    print(f"All {len(c.rows)} checks reconstructed from artifacts alone.")
    return 0


def ev_req(ev: dict, key: str) -> dict | None:
    return next((r for r in ev["requirements"] if r["key"] == key), None)


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "submission_artifacts"
    sys.exit(main(root))
