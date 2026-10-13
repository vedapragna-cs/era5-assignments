#!/usr/bin/env python3
"""Query the ledgers. Answers the audit questions the session states, from artifacts alone.

    python audit.py --step 7                 what trained this optimizer step
    python audit.py --interval 0 6           which shards influenced this token range
    python audit.py --shard v5_indic_train_002    every exposure of one shard, and its verdict
    python audit.py --branches               divergence points and what each branch changed
    python audit.py --opus --lane indic      selection decisions for one lane
    python audit.py --chain                  verify the hash chain

The session's version of these questions:

    "Which shards influenced the model between 5.4B and 5.6B tokens?"
    "Which OPUS-selected batches appeared before a loss spike?"

Reads `submission_artifacts/` as data. It does not import the pipeline, so it answers the same way a
grader or a future run would - which is the point of an audit tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent / "submission_artifacts"


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


class Ledgers:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries = jsonl(root / "ledgers" / "consumption.jsonl")
        self.opus = jsonl(root / "ledgers" / "opus.jsonl")
        self.shard_exposures = jsonl(root / "ledgers" / "learning_shards.jsonl")
        self.superseded = {o for e in self.entries if e.get("event") == "batches_superseded"
                           for o in e.get("ledger_offsets", [])}

    def batches(self, branch: str | None = None) -> list[dict]:
        return [e for e in self.entries
                if e.get("event") == "batch_committed"
                and e["ledger_offset"] not in self.superseded
                and (branch is None or e["branch_id"] == branch)]

    def events(self, kind: str) -> list[dict]:
        return [e for e in self.entries if e.get("event") == kind]


def show_step(L: Ledgers, step: int, branch: str | None) -> int:
    hits = [e for e in L.batches(branch) if e["global_step"] == step]
    if not hits:
        print(f"no committed batch at step {step}"
              f"{' on branch ' + branch if branch else ''}")
        return 1
    for e in hits:
        print(f"\nstep {e['global_step']}  branch={e['branch_id']}  "
              f"ledger_offset={e['ledger_offset']}  checkpoint={e['checkpoint_id']}")
        print(f"  stage           {e['curriculum_stage']}")
        print(f"  lanes           {e['mixture_lanes']}")
        print(f"  batch_hash      {e['batch_hash']}")
        print(f"  loss_mask_hash  {e['loss_mask_hash']}")
        print(f"  attention       {e['attention_policy']}")
        print(f"  positions       {e['position_policy']}")
        print(f"  tokenizer       {e['tokenizer_version']} {e['tokenizer_hash'][:16]}")
        print(f"  dataloader      {e['dataloader_version']}")
        print(f"  world_size      {e['world_size']}")
        print(f"  tokens          {e['n_tokens']} ({e['n_loss_bearing']} loss-bearing)")
        print(f"  shards          {', '.join(e['shard_ids'])}")
        print(f"  samples         {', '.join(e['packed_sample_ids'])}")
        print(f"  token spans     {', '.join(e['token_span_ids'][:6])}"
              + (" ..." if len(e["token_span_ids"]) > 6 else ""))
        print("  rank assignment:")
        for r, ids in sorted(e["rank_assignment"].items(), key=lambda kv: int(kv[0])):
            print(f"    rank {r}: {', '.join(ids)}")
        dec = [d for d in L.opus if d["opus_decision_id"] in e["opus_decision_ids"]]
        if dec:
            print(f"  OPUS decisions ({len(dec)}):")
            for d in dec:
                # An anneal-reserve deferral is recorded but never scored, so score/rank/cut are
                # null by design - see tds/opus.py. Rendering them as 0.0 would claim a measurement
                # that was not taken.
                sc = d["opus_score"]
                score = f"score={sc:+.6f}" if sc is not None else "score=  (not scored)"
                rank = (f"rank {d['rank_in_candidate_pool']}/{d['candidate_pool_size']} "
                        f"(cut {d['top_k_cut_rank']})"
                        if d["rank_in_candidate_pool"] is not None
                        else f"pool {d['candidate_pool_size']}")
                print(f"    {d['opus_decision_id']}  {d['capability_lane']:<9} "
                      f"{d['decision']:<9} {score} {rank}  {d['rejection_reason'] or ''}")
    return 0


def show_interval(L: Ledgers, lo: int, hi: int, branch: str | None) -> int:
    sel = [e for e in L.batches(branch) if lo <= e["global_step"] < hi]
    if not sel:
        print(f"no batches in [{lo}, {hi})")
        return 1
    shards: Counter = Counter()
    lanes: Counter = Counter()
    docs: Counter = Counter()
    for e in sel:
        shards.update(e["shard_ids"])
        lanes.update(e["mixture_lanes"])
        docs.update(sp.split(":")[0] for sp in e["token_span_ids"])
    tok = sum(e["n_tokens"] for e in sel)
    lb = sum(e["n_loss_bearing"] for e in sel)
    print(f"\nsteps [{lo}, {hi})  branch={branch or 'all'}  batches={len(sel)}")
    print(f"  tokens {tok:,} ({lb:,} loss-bearing, {lb/tok:.1%})")
    print(f"  stages {sorted({e['curriculum_stage'] for e in sel})}")
    print("\n  shards that influenced this interval:")
    for s, n in shards.most_common():
        print(f"    {s:<28} {n:>3} batches")
    print("\n  lane sample counts:")
    total = sum(lanes.values())
    for ln, n in sorted(lanes.items()):
        print(f"    {ln:<10} {n:>4}  {n/total:6.2%}")
    print(f"\n  distinct documents exposed: {len(docs)}")
    return 0


def show_shard(L: Ledgers, shard_id: str) -> int:
    exps = [e for e in L.shard_exposures if e["shard_id"] == shard_id]
    batches = [e for e in L.batches() if shard_id in e["shard_ids"]]
    if not exps and not batches:
        print(f"unknown shard {shard_id}")
        return 1
    man = ROOT / "manifests" / "shards" / f"{shard_id}.json"
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        print(f"\n{shard_id}")
        print(f"  lane            {m['capability_lane']}")
        print(f"  tokens          {m['token_count']}")
        print(f"  content_hash    {m['content_hash']}")
        print(f"  tokenizer_hash  {m['tokenizer_hash'][:16]}")
        print(f"  sources         {', '.join(m['source_ids'])}")
        print(f"  languages       {m['languages']}  scripts {m['scripts']}")
        print(f"  bands           {m['difficulty_bands']}")
        print(f"  license         {m['license_tier']} ({m['provenance_tier']})")
        print(f"  dedup/pii/eval  {m['dedup_status']} / {m['pii_screen_status']} / "
              f"{m['eval_overlap_status']}")
        print(f"  fertility       {m['fertility_by_language']}")
    print(f"\n  served in {len(batches)} batches at steps "
          f"{[e['global_step'] for e in batches][:12]}")
    if exps:
        print("\n  learning exposures:")
        print(f"    {'step':>5} {'phase':<7} {'mean_loss':>10} {'ppl':>9} {'hot%':>6} "
              f"{'|g|':>9} {'delta':>9}  usefulness")
        for e in exps:
            print(f"    {e['global_step']:>5} {e['model_phase']:<7} "
                  f"{e['average_token_loss']:>10.4f} {e['geometric_mean_ppl']:>9.2f} "
                  f"{e['high_perplexity_token_pct']:>6.1f} {e['gradient_norm']:>9.4f} "
                  f"{e['loss_delta_before_after']:>+9.4f}  {e['usefulness']}")
    v6 = ROOT / "reports" / "v6_recommendations.json"
    if v6.exists():
        cards = json.loads(v6.read_text(encoding="utf-8"))["shard_cards"]
        card = next((c for c in cards if c["shard_id"] == shard_id), None)
        if card:
            r = card["recommendation"]
            print(f"\n  V6 recommendation: {r['action']}")
            print(f"    because: {r['because']}")
            print(f"    inverts with phase: {card['usefulness_inverts_with_phase']}"
                  f"  best_phase={card['best_phase']}  max_pass={card['max_pass_number']}")
    return 0


def show_branches(L: Ledgers) -> int:
    forks = L.events("branch_forked")
    changes = {e["branch_id"]: e.get("changes") for e in L.events("branch_policy_changed")}
    if not forks:
        print("no forks recorded")
        return 0
    for f in forks:
        print(f"\n{f['parent_branch_id']} -> {f['branch_id']}")
        print(f"  at global_step        {f['global_step']}")
        print(f"  checkpoint            {f['checkpoint_id']}")
        print(f"  divergence offset     {f['divergence_ledger_offset']}")
        print(f"  declared change       {changes.get(f['branch_id'])}")
        n = len(L.batches(f["branch_id"]))
        print(f"  batches on branch     {n}")
    for e in L.events("run_resumed"):
        print(f"\nresume: step {e['global_step']} from {e['checkpoint_id']} "
              f"offset {e['resumed_from_ledger_offset']}  "
              f"world_size {e['previous_world_size']} -> {e['world_size']}")
        print(f"  partition rule: {e['rank_partition_rule']}")
    for e in L.events("batches_superseded"):
        print(f"\nsuperseded offsets {e['ledger_offsets']} ({e['reason']}) "
              f"- retained, not deleted")
    return 0


def show_opus(L: Ledgers, lane: str | None, decision: str | None) -> int:
    sel = [d for d in L.opus
           if (lane is None or d["capability_lane"] == lane)
           and (decision is None or d["decision"] == decision)]
    if not sel:
        print("no matching OPUS decisions")
        return 1
    by = Counter(d["decision"] for d in sel)
    reasons = Counter(d["rejection_reason"] for d in sel if d["rejection_reason"])
    print(f"\n{len(sel)} decisions"
          f"{' lane=' + lane if lane else ''}{' decision=' + decision if decision else ''}")
    print(f"  by decision  {dict(by)}")
    print(f"  reasons      {dict(reasons)}")
    # Deferrals carry a null score by design (never scored, so no number to report). They are
    # excluded from the range rather than coerced to zero, which would invent a measurement.
    scores = [d["opus_score"] for d in sel if d["opus_score"] is not None]
    n_unscored = len(sel) - len(scores)
    if scores:
        print(f"  score range  {min(scores):+.6f} .. {max(scores):+.6f} "
              f"(mean {sum(scores)/len(scores):+.6f}, over {len(scores)} scored decisions)")
    if n_unscored:
        print(f"  unscored     {n_unscored} anneal-reserve deferrals - recorded, never scored")
    print(f"\n  {'id':<14} {'step':>5} {'lane':<10} {'decision':<10} {'score':>10} reason")
    for d in sel[:40]:
        sc = d["opus_score"]
        print(f"  {d['opus_decision_id']:<14} {d['global_step']:>5} "
              f"{d['capability_lane']:<10} {d['decision']:<10} "
              f"{f'{sc:>+10.6f}' if sc is not None else '         -'} "
              f"{d['rejection_reason'] or ''}")
    if len(sel) > 40:
        print(f"  ... {len(sel) - 40} more")
    return 0


def verify_chain(L: Ledgers) -> int:
    import hashlib
    prev = "0" * 64
    for i, e in enumerate(L.entries):
        payload = {k: v for k, v in e.items() if k not in ("prev_hash", "entry_hash")}
        want = hashlib.sha256(json.dumps(
            {"payload": payload, "prev_hash": e["prev_hash"]},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        if e["prev_hash"] != prev or e["entry_hash"] != want:
            print(f"CHAIN BROKEN at entry {i} (event={e.get('event')})")
            return 1
        prev = e["entry_hash"]
    print(f"chain intact across {len(L.entries)} entries; head {prev[:16]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=None)
    ap.add_argument("--step", type=int)
    ap.add_argument("--interval", type=int, nargs=2, metavar=("FROM", "TO"))
    ap.add_argument("--shard")
    ap.add_argument("--branches", action="store_true")
    ap.add_argument("--opus", action="store_true")
    ap.add_argument("--lane")
    ap.add_argument("--decision", choices=("accepted", "rejected", "deferred", "protected"))
    ap.add_argument("--branch")
    ap.add_argument("--chain", action="store_true")
    a = ap.parse_args()

    global ROOT
    if a.artifacts is not None:
        ROOT = a.artifacts
    if not (ROOT / "ledgers" / "consumption.jsonl").exists():
        print(f"no ledgers under {ROOT} - run run_demo.py first")
        return 2
    L = Ledgers(ROOT)

    if a.step is not None:
        return show_step(L, a.step, a.branch)
    if a.interval:
        return show_interval(L, a.interval[0], a.interval[1], a.branch)
    if a.shard:
        return show_shard(L, a.shard)
    if a.branches:
        return show_branches(L)
    if a.opus:
        return show_opus(L, a.lane, a.decision)
    if a.chain:
        return verify_chain(L)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
