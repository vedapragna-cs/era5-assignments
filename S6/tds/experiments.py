"""Pre-registered experiments. Hypotheses and refutation criteria live in S6/EXPERIMENT_LOG.md.

Each returns a record that goes into the evidence bundle verbatim, including results that came out
against the hypothesis. An experiment that only reports confirmations is not an experiment.
"""

from __future__ import annotations

from .build import build
from .canon import chash
from .harness import make_rig
from .mixture import MixturePlan


def e1_mixed_vs_pure_lane_shards(plan: MixturePlan, *, seed: int, steps: int,
                                 global_batch: int, sequence_length: int) -> dict:
    """E1 — the instructor's open question, stated in the session and left undecided:

        "let us say that 10% is indic. So the question is that do we make a shard in which we have
         10% and 90% and when we send to OPUS we say we don't care we still going to take it, or we
         keep the indic separately. That is one question that I've not thought about."

    Arm P: one lane per shard; the floor injects whole Indic batches.
    Arm M: some Indic documents relabelled into the web lane, so they ride inside web shards and can
           only reach the model when a batch the scheduler believes is web is accepted.
    """
    arms = {}
    for arm, mixed in (("P_pure", False), ("M_mixed", True)):
        built = build(plan, sequence_length=sequence_length, seed=seed, mixed_lane_shards=mixed)
        rig = make_rig(plan, built, run_id=f"e1-{arm}", seed=seed, global_batch=global_batch,
                       sequence_length=sequence_length, world_size=4)
        tr = rig.trainer
        for _ in range(steps):
            tr.train_step()

        # Delivered Indic share measured two ways. The distinction IS the finding: in arm M the
        # scheduler's lane tag no longer tells you what the model actually consumed.
        by_tag = tr.ledger.lane_token_shares().get("indic", 0.0)
        indic_docs = {d.doc_id for d in built.documents
                      if d.split == "train" and d.language in ("hi", "mr", "te")}
        total = tagged = 0
        for e in tr.ledger.batches(tr.branch_id):
            for sid in e["packed_sample_ids"]:
                total += 1
                if set(built.sample_meta[sid]["doc_ids"]) & indic_docs:
                    tagged += 1
        by_content = tagged / total if total else 0.0

        per_stage: dict[str, list[int]] = {}
        for e in tr.ledger.batches(tr.branch_id):
            per_stage.setdefault(e["curriculum_stage"], [0, 0])
            per_stage[e["curriculum_stage"]][1] += sum(e["mixture_lanes"].values())
            per_stage[e["curriculum_stage"]][0] += e["mixture_lanes"].get("indic", 0)
        stage_shares = {k: round(v[0] / v[1], 4) if v[1] else 0.0 for k, v in sorted(per_stage.items())}
        vals = list(stage_shares.values())
        spread = round(max(vals) - min(vals), 4) if vals else 0.0

        arms[arm] = {
            "indic_share_by_scheduler_lane_tag": round(by_tag, 4),
            "indic_share_by_actual_document_content": round(by_content, 4),
            "auditable_from_ledger": abs(by_tag - by_content) < 1e-9,
            "per_stage_indic_share": stage_shares,
            "across_stage_spread": spread,
            "opus": tr.opus.summary()["by_decision"],
        }

    floor = plan.protected_floors["indic"]
    p, m = arms["P_pure"], arms["M_mixed"]
    refuted = (abs(m["indic_share_by_actual_document_content"] - floor) <= 0.01
               and m["across_stage_spread"] <= p["across_stage_spread"])
    return {
        "experiment": "E1",
        "question": "mixed-lane vs pure-lane shards for delivering a protected floor",
        "floor": floor,
        "arms": arms,
        "refutation_criterion": "refuted if arm M delivers the floor within +/-1pp AND its "
                                "per-stage spread is no worse than arm P",
        "hypothesis_refuted": refuted,
        "finding": (
            "In arm P the scheduler's lane tag and the actual document content agree exactly, so the "
            "delivered Indic share is auditable straight from the consumption ledger. In arm M they "
            "diverge: the ledger reports the CARRIER lane, so the protected lane's true delivery is "
            "invisible to every downstream audit that reads mixture_lanes. That is the decisive "
            "difference - not the share itself but whether the share can be measured at all. A floor "
            "you cannot audit is not a floor, and S5's own measurement of the course's always-on "
            "lane (stated 8%, delivered ~15%) is exactly what an unauditable floor looks like."
        ),
        "decision": "pure-lane shards; protected lanes are injected post-selection as whole batches",
    }


def e6_indic_difficulty_ordering(plan: MixturePlan, *, seed: int, steps: int,
                                 global_batch: int, sequence_length: int) -> dict:
    """E6 — also stated as open by the instructor: order the Indic lane by difficulty, or not?

        "maybe there are some sentences that are easier to train on compared to other ... should we
         first train on it or should we focus on the simple news article that are there right?"
    """
    arms = {}
    for arm, ordered in (("A_difficulty_ordered", True), ("B_flat_shuffle", False)):
        built = build(plan, sequence_length=sequence_length, seed=seed,
                      indic_difficulty_ordered=ordered)
        rig = make_rig(plan, built, run_id=f"e6-{arm}", seed=seed, global_batch=global_batch,
                       sequence_length=sequence_length, world_size=4)
        tr = rig.trainer
        for _ in range(steps):
            tr.train_step()
        agg = tr.learning.aggregate("lane").get("indic", {})
        bands = [built.sample_meta[sid]["band"]
                 for e in tr.ledger.batches(tr.branch_id) for sid in e["packed_sample_ids"]
                 if built.sample_meta[sid]["lane"] == "indic"]
        arms[arm] = {
            "indic_mean_loss": agg.get("mean_loss"),
            "indic_geometric_mean_ppl": agg.get("geometric_mean_ppl"),
            "indic_tokens_traced": agg.get("n_tokens"),
            "band_order_served": bands[:12],
        }
    a = arms["A_difficulty_ordered"]["indic_mean_loss"]
    b = arms["B_flat_shuffle"]["indic_mean_loss"]
    delta = None if (a is None or b is None) else round(b - a, 6)
    return {
        "experiment": "E6",
        "question": "does ordering the protected Indic lane by difficulty beat a flat shuffle",
        "arms": arms,
        "loss_advantage_of_ordering": delta,
        "verdict": "UNDER-POWERED AT DEMO SCALE",
        "finding": (
            "Reported as inconclusive rather than as support. This run trains a 32-dim model for a "
            "few dozen steps over a corpus of tens of documents; any loss difference at that scale "
            "is dominated by initialisation and draw order, not by curriculum. The mechanism is "
            "implemented and the band order served is recorded, so the experiment is ready to run "
            "at a scale where it could actually decide - but claiming a curriculum effect from "
            "this measurement would be exactly the kind of unbacked number S5 exists to avoid."
        ),
    }


def determinism_check(plan: MixturePlan, *, seed: int, steps: int, global_batch: int,
                      sequence_length: int) -> dict:
    """Two independent full runs must produce identical artifacts.

    The assignment's phrasing is "I should get exactly what you got". This measures it: same seed,
    fresh objects, compare content hashes of the whole consumption ledger and the tokenizer.
    """
    out = []
    for i in range(2):
        built = build(plan, sequence_length=sequence_length, seed=seed)
        rig = make_rig(plan, built, run_id=f"det-{i}", seed=seed, global_batch=global_batch,
                       sequence_length=sequence_length, world_size=4)
        tr = rig.trainer
        for _ in range(steps):
            tr.train_step()
        batches = [{k: v for k, v in e.items() if k not in ("run_id", "prev_hash", "entry_hash")}
                   for e in tr.ledger.batches(tr.branch_id)]
        out.append({
            "tokenizer_hash": built.tokenizer.hash,
            "shard_content_hashes": [s.content_hash for s in built.shards],
            "batch_stream_hash": chash(batches),
            "model_weight_hash": tr.model.weight_hash,
            "final_loss": round(tr.step_log[-1]["mean_loss"], 12) if tr.step_log else None,
        })
    same = {k: out[0][k] == out[1][k] for k in out[0]}
    return {
        "runs": out,
        "identical": same,
        "all_identical": all(same.values()),
        "scope": "data-stream determinism is claimed exactly. Model weight equality holds within "
                 "one machine and thread count (float64, fixed reduction order) and is NOT claimed "
                 "portable across hardware - BLAS reduction order varies. The ledger does not "
                 "depend on model determinism.",
    }
