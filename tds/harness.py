"""Builds a fresh, fully wired Trainer. Used by run_demo, the tests and the E1/E2/E6 experiments.

Having one constructor matters: if the demo and the tests assembled the system differently, the
tests would not be exercising the thing that produced the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .build import BuiltCorpus, build
from .exactly_once import DeliveryTrace
from .learning import LearningLedger
from .ledger import ConsumptionLedger
from .mixture import MixturePlan, StreamPlanner
from .model import TinyLM
from .opus import Opus
from .trainer import TrainConfig, Trainer


@dataclass
class Rig:
    built: BuiltCorpus
    trainer: Trainer


def make_rig(plan: MixturePlan, built: BuiltCorpus, *, run_id: str, seed: int,
             global_batch: int, sequence_length: int, world_size: int,
             rejection_rate: float | None = None) -> Rig:
    _rr = float(plan.opus["rejection_rate_target"])
    planner = StreamPlanner(plan=plan, pools=built.pools, seed=seed,
                            global_batch=global_batch, sequence_length=sequence_length,
                            oversample=1.0 / max(1e-6, 1.0 - _rr))
    model = TinyLM.init(built.tokenizer.size, seed=seed)
    opus = Opus(rejection_rate=(plan.opus["rejection_rate_target"]
                                if rejection_rate is None else rejection_rate),
                proxy_version=plan.opus["proxy_version"], seed=seed,
                protected_lanes=set(plan.protected_floors))
    cfg = TrainConfig(run_id=run_id, seed=seed, global_batch=global_batch,
                      sequence_length=sequence_length, world_size=world_size)
    tr = Trainer(plan=plan, planner=planner, samples=built.samples,
                 sample_meta=built.sample_meta, tokenizer=built.tokenizer, model=model,
                 opus=opus, firewall=built.firewall, ledger=ConsumptionLedger(run_id=run_id),
                 learning=LearningLedger(tier="full"), cfg=cfg, golden=built.golden)
    return Rig(built=built, trainer=tr)


def trace_of(tr: Trainer, *, label: str) -> DeliveryTrace:
    """The ordered global sample sequence this trainer actually served, from its own ledger."""
    # Keyed by insertion order, NOT by global_step. Keying by step silently collapses a step that
    # was delivered twice into one entry - so the very fault the exactly-once test exists to detect
    # would be erased before the comparison ran.
    batches = tr.ledger.batches(tr.branch_id)
    return DeliveryTrace(label=label, world_size=tr.cfg.world_size,
                         per_step=[list(e["packed_sample_ids"]) for e in batches],
                         step_numbers=[e["global_step"] for e in batches])


def run_uninterrupted(plan: MixturePlan, built: BuiltCorpus, *, steps: int, seed: int,
                      global_batch: int, sequence_length: int, world_size: int,
                      checkpoint_every: int = 6) -> Trainer:
    rig = make_rig(plan, built, run_id="trace-a", seed=seed, global_batch=global_batch,
                   sequence_length=sequence_length, world_size=world_size)
    tr = rig.trainer
    for _ in range(steps):
        if tr.step % checkpoint_every == 0:
            tr.save_checkpoint()
        tr.train_step()
    return tr


def run_with_crash(plan: MixturePlan, built: BuiltCorpus, *, steps: int, crash_at: int, seed: int,
                   global_batch: int, sequence_length: int, world_size_before: int,
                   world_size_after: int, checkpoint_every: int = 6) -> Trainer:
    """Same plan, but the run dies at `crash_at` and comes back on a different machine shape."""
    rig = make_rig(plan, built, run_id="trace-b", seed=seed, global_batch=global_batch,
                   sequence_length=sequence_length, world_size=world_size_before)
    tr = rig.trainer
    for _ in range(crash_at):
        if tr.step % checkpoint_every == 0:
            tr.save_checkpoint()
        tr.train_step()
    tr.crash("simulated node failure", destroy_state=False)
    tr.resume(tr.last_checkpoint_id, world_size=world_size_after)
    while tr.step < steps:
        if tr.step % checkpoint_every == 0 and f"ckpt_{tr.step:05d}" not in tr.checkpoints:
            tr.save_checkpoint()
        tr.train_step()
    return tr
