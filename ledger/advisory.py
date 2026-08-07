"""The v6 shard advisory: what the learning ledger tells you to do differently next time.

This is the payoff for keeping per-shard loss. A loss curve tells you the run is working.
Per-shard loss tells you *which shards* are working, and that is a decision, not a
diagnostic -- it is the input to how v6 places its data.

The rule is the session's shard-perplexity reuse cutoff. For each shard, compare the model's
final perplexity on it against the run-wide median:

    ratio = shard_perplexity / median_shard_perplexity

* **ratio >= 1.2** -- the model still finds this shard materially harder than the corpus.
  Either it is genuinely more informative (place it earlier, or give it another epoch) or it
  is broken (encoding damage, wrong language tag, truncated documents). Which of the two it
  is cannot be decided from the number alone, so the advisory says so rather than guessing.
* **ratio <= 1/1.2 (0.833)** -- the model finds this shard easy. Re-reading it in v6 spends
  budget on tokens that are already predicted. Reduce its exposure.
* **in between** -- keep the placement.

Two independent signals refine this:

* **delta** (first exposure loss minus last) -- a shard seen repeatedly whose loss has not
  moved is not teaching anything, whatever its absolute level. That is a quality flag, not
  a placement flag.
* **exposures against the 4-epoch ceiling** (arXiv:2305.16264, inherited via S5) -- a shard
  already at the ceiling cannot be given more epochs no matter how informative it looks, so
  its recommendation has to be "duplicate the *source*", not "repeat the shard".

Nothing here is applied automatically. The output is a recommendation with the evidence
attached, because "this shard is hard" and "this shard is bad" produce opposite actions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .plan import EPOCH_CEILING, LANE_SPEC

REUSE_CUTOFF = 1.2          # the session's ratio; symmetric, so 1/1.2 on the easy side
STALL_DELTA = 0.02          # loss movement below this over >=2 exposures is "not learning"


def _ppl(loss: Optional[float]) -> Optional[float]:
    return round(math.exp(min(loss, 20.0)), 4) if loss is not None else None


def build_advisory(trainer, registry, schedule=None) -> Dict[str, Any]:
    """Turn the learning ledger's per-shard table into v6 placement recommendations."""
    rows = trainer.learning.shard_table()
    consumed = trainer.consumption.shard_tokens

    losses = [r["mean_loss"] for r in rows if r["mean_loss"] is not None]
    if not losses:
        return {"status": "no_shard_losses_recorded"}
    ordered = sorted(losses)
    median = ordered[len(ordered) // 2]
    median_ppl = math.exp(min(median, 20.0))

    recommendations: List[Dict[str, Any]] = []
    for r in rows:
        sid = r["shard_id"]
        m = registry.manifest(sid)
        ppl = math.exp(min(r["mean_loss"], 20.0))
        ratio = ppl / median_ppl
        tokens_seen = consumed.get(sid, 0)
        epochs = tokens_seen / max(1, m.token_count)
        at_ceiling = epochs >= EPOCH_CEILING - 1e-9

        delta = r.get("delta")
        stalled = (r["times_seen"] >= 2 and delta is not None and abs(delta) < STALL_DELTA)

        if ratio >= REUSE_CUTOFF:
            if stalled:
                action = "investigate_quality"
                rationale = (
                    "Perplexity is {:.2f}x the run median and has not moved across {} "
                    "exposures. A shard that stays hard while being re-read is usually "
                    "damaged (encoding, truncation, mislabelled language), not "
                    "informative.".format(ratio, r["times_seen"]))
            elif at_ceiling:
                action = "acquire_more_like_this"
                rationale = (
                    "Perplexity is {:.2f}x the run median and still falling, but the shard "
                    "is at the {}-epoch ceiling. More epochs are not available; the correct "
                    "v6 move is to source more data of this kind, not to re-read this "
                    "shard.".format(ratio, EPOCH_CEILING))
            else:
                action = "promote_earlier_or_extra_epoch"
                rationale = (
                    "Perplexity is {:.2f}x the run median with {:.2f} epochs used of {}. "
                    "Headroom exists and the model is still learning from it."
                    .format(ratio, epochs, EPOCH_CEILING))
        elif ratio <= 1.0 / REUSE_CUTOFF:
            action = "reduce_exposure"
            rationale = (
                "Perplexity is {:.2f}x the run median -- the model already predicts this "
                "shard well. Re-reading it in v6 buys little; the budget is better spent "
                "on lanes above the cutoff.".format(ratio))
        else:
            action = "keep_placement"
            rationale = ("Perplexity is {:.2f}x the run median, inside the "
                         "{:.1f}x reuse band.".format(ratio, REUSE_CUTOFF))

        recommendations.append({
            "shard_id": sid,
            "lane": m.lane,
            "bands": m.difficulty_bands,
            "shard_tokens": m.token_count,
            "tokens_consumed": tokens_seen,
            "epochs_used": round(epochs, 4),
            "at_epoch_ceiling": at_ceiling,
            "mean_loss": r["mean_loss"],
            "perplexity": round(ppl, 4),
            "ratio_to_median": round(ratio, 4),
            "first_loss": r["first_loss"],
            "last_loss": r["last_loss"],
            "delta": delta,
            "stalled": stalled,
            "times_seen": r["times_seen"],
            "first_seen_step": r["first_seen_step"],
            "action": action,
            "rationale": rationale,
        })

    # Supply that was never read at all. Not a placement decision -- a budgeting one.
    never_read = []
    for sid in registry.trainable_ids():
        if sid not in consumed:
            m = registry.manifest(sid)
            never_read.append({
                "shard_id": sid, "lane": m.lane, "bands": m.difficulty_bands,
                "tokens": m.token_count,
            })

    by_action: Dict[str, int] = {}
    for r in recommendations:
        by_action[r["action"]] = by_action.get(r["action"], 0) + 1

    lane_roll: Dict[str, Dict[str, Any]] = {}
    for r in recommendations:
        d = lane_roll.setdefault(r["lane"], {"shards": 0, "loss_sum": 0.0, "tokens": 0,
                                             "actions": {}})
        d["shards"] += 1
        d["loss_sum"] += r["mean_loss"]
        d["tokens"] += r["tokens_consumed"]
        d["actions"][r["action"]] = d["actions"].get(r["action"], 0) + 1
    for lane, d in lane_roll.items():
        d["mean_loss"] = round(d["loss_sum"] / d["shards"], 6)
        d["perplexity"] = _ppl(d["mean_loss"])
        d["protected"] = LANE_SPEC[lane].protected if lane in LANE_SPEC else False
        del d["loss_sum"]

    return {
        "what_this_is": (
            "Per-shard placement recommendations for the next run, derived from the "
            "learning ledger. The cutoff is the session's shard-perplexity reuse rule."),
        "reuse_cutoff": REUSE_CUTOFF,
        "epoch_ceiling": EPOCH_CEILING,
        "median_shard_loss": round(median, 6),
        "median_shard_perplexity": round(median_ppl, 4),
        "shards_scored": len(recommendations),
        "actions": dict(sorted(by_action.items())),
        "by_lane": dict(sorted(lane_roll.items())),
        "promote": [r for r in recommendations
                    if r["action"] == "promote_earlier_or_extra_epoch"][:15],
        "investigate": [r for r in recommendations if r["action"] == "investigate_quality"],
        "acquire_more": [r for r in recommendations
                         if r["action"] == "acquire_more_like_this"][:15],
        "reduce": [r for r in recommendations if r["action"] == "reduce_exposure"][:15],
        "never_read": {
            "count": len(never_read),
            "tokens": sum(r["tokens"] for r in never_read),
            "note": ("Trainable supply the schedule never reached. Not waste -- it is the "
                     "headroom that let lanes stay above their floors -- but it is budget "
                     "that v6 can re-plan against."),
            "sample": never_read[:10],
        },
        "all_recommendations": recommendations,
        "caveat": (
            "A high ratio means 'the model finds this hard', which is ambiguous between "
            "informative and broken. The advisory separates the two using loss movement "
            "across exposures, but a stalled-and-hard shard should be read by a human "
            "before it is dropped."),
    }
