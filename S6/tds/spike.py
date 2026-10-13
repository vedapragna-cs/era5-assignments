"""Loss-spike detection and attribution.

The hole this closes
--------------------
The learning ledger classifies a shard `harmful` when the loss rises across the step that consumed
it. A loss spike does exactly that - and a spike is not, in general, the fault of the data in the
batch. PaLM (arXiv:2204.02311 §5) ran the decisive ablation: they restarted from a checkpoint ~100
steps before a spike and replayed **the same batches**, and the spike did not recur. Whatever caused
it was the interaction of that batch with that parameter state, not a property of the batch.

So a system that writes `harmful` onto a shard because it coincided with a spike will ship a
permanent, wrong label into the next cohort's data decisions. That is a data-poisoning bug with a
long fuse: the shard is fine, the label is not, and nothing downstream can tell.

What this module does instead
-----------------------------
1. A spike is a **run event**, recorded against the run and the step. It is never written onto a
   shard, a lane or a document.
2. Any shard whose classification window overlaps a spike is flagged `attribution_uncertain` and its
   verdict is withheld rather than guessed.
3. The uncertainty is *resolvable*, by PaLM's own procedure, and we implement it: fork from an
   earlier checkpoint, replay the identical recorded batches, and see whether the spike reproduces.
   - reproduces  -> the batch is implicated; the data is a legitimate suspect.
   - does not    -> the data is exonerated and the flag is cleared to `interaction`.

   This is only possible because replay is exact. The ablation is the payoff for the whole
   deterministic-stream design: without byte-identical re-serving there is no controlled experiment
   here, only a guess.

What this module does NOT claim
-------------------------------
PaLM observed spikes **only in the largest model**, and Wortsman et al. (arXiv:2309.14322, footnote
1) scope their small-scale reproduction to slow-divergence instabilities and explicitly not to loss
spikes. A 32-dimension model on a 50-document corpus will not spike on its own. So the detector and
the ablation are exercised against injected series in the negative controls, and the run reports
honestly when no spike occurred. The mechanism is demonstrated; the phenomenon is not manufactured.

Related, and separate: OLMo 2 (arXiv:2501.00656 §3.1) found that documents with long repeated n-gram
runs are over-represented in spiking batches, which is a *prior* on which data is a suspect - not a
verdict on any individual spike. That filter lives in `clean.py`; this module is what happens when a
spike occurs anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# A spike is a jump relative to the run's own recent variability, not an absolute threshold: loss
# scale depends on vocabulary size and training depth, so a fixed number would be meaningless across
# the demo's two depths.
SPIKE_Z = 4.0
SPIKE_WINDOW = 8          # steps of history used to estimate the local mean and spread
MIN_HISTORY = 4


@dataclass
class SpikeEvent:
    global_step: int
    branch_id: str
    metric: str                     # mean_loss | grad_norm
    value: float
    baseline_mean: float
    baseline_std: float
    z: float
    # Attribution starts UNRESOLVED. It is never inferred from coincidence.
    attribution: str = "unresolved"
    ablation: dict | None = None
    batches_skipped: int = 0
    # Set when the spike coincides with a curriculum stage transition. A seam changes the lane
    # COMPOSITION of the batch, so the ablation will faithfully reproduce it from any parameter
    # state - "reproducible" then means "the mixture changed here", not "these documents are bad".
    # Distinguishing the two is the difference between a seam-smoothing fix and deleting good data.
    stage_boundary: str | None = None

    def to_record(self) -> dict:
        return {
            "event": "loss_spike",
            "global_step": self.global_step,
            "branch_id": self.branch_id,
            "metric": self.metric,
            "value": round(self.value, 6),
            "baseline_mean": round(self.baseline_mean, 6),
            "baseline_std": round(self.baseline_std, 6),
            "z_score": round(self.z, 3),
            "attribution": self.attribution,
            "stage_boundary": self.stage_boundary,
            "batches_skipped": self.batches_skipped,
            "ablation": self.ablation,
            "note": "a spike is a property of the RUN at this step, never of the shards in the "
                    "batch. See tds/spike.py.",
        }


def detect(step_log: list[dict], *, metric: str = "mean_loss", z: float = SPIKE_Z,
           branch: str | None = None) -> list[SpikeEvent]:
    """Flag steps where `metric` jumps far above its own recent baseline.

    Uses a trailing window so the baseline cannot be contaminated by the spike it is meant to
    detect - a two-sided window would blunt exactly the event being looked for.
    """
    rows = [r for r in step_log if branch is None or r.get("branch") == branch]
    out: list[SpikeEvent] = []
    for i, row in enumerate(rows):
        if i < MIN_HISTORY:
            continue
        hist = [rows[j][metric] for j in range(max(0, i - SPIKE_WINDOW), i)]
        mean = sum(hist) / len(hist)
        var = sum((h - mean) ** 2 for h in hist) / len(hist)
        std = var ** 0.5
        if std <= 1e-12:
            continue
        score = (row[metric] - mean) / std
        if score >= z:
            out.append(SpikeEvent(global_step=row["step"], branch_id=row.get("branch", "run-a"),
                                  metric=metric, value=row[metric], baseline_mean=mean,
                                  baseline_std=std, z=score))
    return out


def spike_windows(events: list[SpikeEvent], *, before: int = 1, after: int = 1) -> set[int]:
    """Steps whose shard attribution is unsafe, given these spikes."""
    win: set[int] = set()
    for e in events:
        for s in range(e.global_step - before, e.global_step + after + 1):
            win.add(s)
    return win


def ablate(trainer, event: SpikeEvent, *, lookback: int = 6) -> dict:
    """PaLM's ablation: replay the spiking batches from an earlier parameter state.

    Forks a throwaway branch from a checkpoint before the spike and re-serves the identical recorded
    batches. If the spike does not recur, the batch content is exonerated and the cause is the
    interaction with the parameter state that no longer exists.

    Returns the verdict and every number behind it, so the conclusion is checkable rather than
    asserted.
    """
    target = event.global_step
    prior = [cid for cid, st in sorted(trainer.checkpoints.items())
             if st["global_step"] <= max(0, target - lookback)]
    if not prior:
        return {"ran": False,
                "reason": f"no checkpoint at least {lookback} steps before step {target}; "
                          f"PaLM's procedure needs one to restart from",
                "verdict": "unresolvable_here"}

    cid = prior[-1]
    recorded = [e for e in trainer.ledger.batches(event.branch_id)
                if e["global_step"] == target]
    if not recorded:
        return {"ran": False, "reason": f"no ledger record for step {target}",
                "verdict": "unresolvable_here"}

    # An ablation is an AUDIT operation and must leave the audited run untouched - the same rule
    # `replay_interval` states and implements. This restored only branch/step/index/cursors, so it
    # returned with the live model sitting at the ablation branch's weights (measured: weight hash
    # 01ce91522da54690 -> dbdcd3dfc52d1178) and the planner's cumulative delivery rewound to the
    # pre-spike checkpoint (total_delivered 82 -> 34). The next consumer of `planner.delivered` is
    # the rank-compute-skew measurement, so a shipped number was planned against a delivery history
    # that never happened.
    saved_branch, saved_step, saved_index = trainer.branch_id, trainer.step, trainer.global_index
    saved_cursors = trainer.planner.peek_cursors()
    saved_model = trainer.model.state()
    saved_delivered = dict(trainer.planner.delivered)
    saved_total = trainer.planner.total_delivered
    saved_proxy = (trainer.opus._proxy_cache, trainer.opus._proxy_step)
    try:
        trainer.restore(cid, branch=f"ablate-{target}")
        samples = trainer._samples_from_record(recorded[0])
        r = trainer.train_step(replay_samples=samples)
        # Compare the SAME metric that spiked. Comparing mean_loss against a grad_norm event would
        # answer a different question than the one asked.
        replayed = float(getattr(r, event.metric))
        # Half the original z, stated rather than left in the code: the ablation restarts from a
        # DIFFERENT parameter state, so demanding the identical excursion would make "reproduced"
        # nearly unreachable and bias every verdict toward exonerating the data. Half the original
        # excursion is the threshold for "the spike substantially recurred". It is a judgement call
        # on a single replayed batch, and the record says so.
        replay_z = ((replayed - event.baseline_mean) / event.baseline_std
                    if event.baseline_std > 0 else 0.0)
        reproduced = bool(event.baseline_std > 0 and replay_z >= event.z * 0.5)
    finally:
        trainer.branch_id, trainer.step = saved_branch, saved_step
        trainer.global_index = saved_index
        trainer.planner.set_cursors(saved_cursors)
        trainer.model.load_state(saved_model)
        trainer.planner.delivered = saved_delivered
        trainer.planner.total_delivered = saved_total
        trainer.opus._proxy_cache, trainer.opus._proxy_step = saved_proxy

    return {
        "ran": True,
        "procedure": "PaLM arXiv:2204.02311 s5: restart from an earlier checkpoint and replay the "
                     "identical batches",
        "criterion": "reproduced := replay_z >= 0.5 * original_z, over ONE replayed batch scored "
                     "against the baseline mean/std estimated from the original run's trailing "
                     "window. Half, not equal, because the restart is from a different parameter "
                     "state; single-batch, so this is evidence and not proof.",
        "replay_z": round(replay_z, 3),
        "original_z": round(event.z, 3),
        "state_restored_after": ["model_weights", "planner.delivered", "planner.total_delivered",
                                 "opus_proxy_cache", "branch_id", "step", "global_index",
                                 "pool_cursors"],
        "restarted_from": cid,
        "replayed_step": target,
        "metric": event.metric,
        "original_value": round(event.value, 6),
        "replayed_value": round(replayed, 6),
        "baseline_mean": round(event.baseline_mean, 6),
        "reproduced": reproduced,
        "verdict": ("seam_composition" if reproduced and event.stage_boundary
                    else "data_implicated" if reproduced else "interaction_not_data"),
        "consequence": (
            f"reproducible AND at the {event.stage_boundary} stage transition: the batch's lane "
            f"COMPOSITION changed here, so this is a curriculum seam, not a verdict on any "
            f"document. The fix is seam blending, not removing data."
            if reproduced and event.stage_boundary else
            "reproducible away from any stage boundary: the batch content is a legitimate suspect "
            "and its shards may be investigated"
            if reproduced else
            "not reproducible from an earlier parameter state: the shards in this batch are "
            "EXONERATED and must not be labelled harmful for coinciding with this spike"),
    }
