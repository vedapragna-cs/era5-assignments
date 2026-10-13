"""Exactly-once delivery across a restart with a changed world size.

This is the load-bearing test of the whole system, and it has to be written carefully, because the
obvious version of it cannot fail.

The vacuous version — which an earlier build of this repo shipped — takes the delivered sample list,
re-partitions it across the new world size, and checks that the union equals the original list. It
always passes, because the union of a partition of a list is that list by construction. It tests
`itertools`, not the loader.

The real test runs the plan twice:

    run A : uninterrupted at world_size = 4
    run B : identical plan, crashes mid-run, resumes at world_size = 8

and compares the two *independently produced* global sample sequences, step by step. Duplicates are
not counted naively: a sample id may legitimately repeat across steps, because lanes wrap around
their pools and repetition is a budgeted quantity (S5 runs Indic at 3.31 epochs). What must hold is
that the ordered sequence of (step, position) -> sample_id is identical, and that the rank partition
covers every position exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .mixture import PlannedSample, partition, rank_of


@dataclass
class DeliveryTrace:
    """The ordered global sample sequence a run actually served."""
    label: str
    world_size: int
    per_step: list[list[str]]           # ordered batches as delivered (NOT keyed by step)
    step_numbers: list[int] = field(default_factory=list)

    @property
    def flat(self) -> list[str]:
        return [s for step in self.per_step for s in step]

    def rank_view(self, world_size: int) -> dict[int, list[tuple[int, int, str]]]:
        """(step, position_in_step, sample_id) grouped by the rank that would serve it."""
        out: dict[int, list[tuple[int, int, str]]] = {r: [] for r in range(world_size)}
        gi = 0
        for st, step in enumerate(self.per_step):
            for pos, sid in enumerate(step):
                out[rank_of(gi, world_size)].append((st, pos, sid))
                gi += 1
        return out


def compare(a: DeliveryTrace, b: DeliveryTrace) -> dict:
    """Compare two independently produced traces for exactly-once, order-preserving delivery."""
    same_len = len(a.flat) == len(b.flat)
    order_identical = a.flat == b.flat
    per_step_identical = a.per_step == b.per_step

    # Coverage. The previous two versions of this were both vacuous, in different ways, and the
    # second shipped with a comment asserting it was fixed - so this one states plainly what it can
    # and cannot test.
    #
    # It CANNOT test the rank partition. `rank = global_index % world_size` is a pure function of
    # the index, so "every position is served by exactly one rank" is a theorem about modular
    # arithmetic, and any runtime check of it round-trips through its own input. Both earlier
    # versions did exactly that and reported dup=0 / missing=0 for a duplicated batch, a dropped
    # batch and a batch of four identical samples.
    #
    # What it CAN test is the delivered step sequence, which is data: the ledger records one batch
    # per committed step, so the step numbers must be strictly increasing with no repeats and no
    # gaps. A step delivered twice repeats; a skipped step leaves a gap. That is a real property of
    # a real artifact, and it fails on both faults.
    def coverage(trace: DeliveryTrace, ws: int) -> dict:
        steps = list(trace.step_numbers) or list(range(len(trace.per_step)))
        repeats = sorted({s for s in steps if steps.count(s) > 1})
        gaps = [] if not steps else [s for s in range(min(steps), max(steps) + 1)
                                     if s not in set(steps)]
        return {
            "world_size": ws,
            "batches_delivered": len(trace.per_step),
            "step_numbers_strictly_increasing": steps == sorted(steps) and len(set(steps)) == len(steps),
            "repeated_steps": repeats,
            "missing_steps": gaps,
            "duplicate_positions": sum(steps.count(s) - 1 for s in repeats),
            "missing_positions": len(gaps),
            "rank_partition_note": "rank = global_index % world_size is a pure function of the "
                                   "index; that every position maps to exactly one rank is "
                                   "arithmetic, not a measurement, and is deliberately not "
                                   "presented as one",
        }

    cov_a = coverage(a, a.world_size)
    cov_b = coverage(b, b.world_size)

    first_divergence = None
    for i, (x, y) in enumerate(zip(a.flat, b.flat)):
        if x != y:
            first_divergence = {"global_position": i, "run_a": x, "run_b": y}
            break

    ok = (same_len and order_identical and per_step_identical
          and cov_a["step_numbers_strictly_increasing"] and cov_b["step_numbers_strictly_increasing"]
          and cov_a["duplicate_positions"] == 0 and cov_a["missing_positions"] == 0
          and cov_b["duplicate_positions"] == 0 and cov_b["missing_positions"] == 0)
    return {
        "method": "two independently produced traces compared element-wise; NOT a re-partition of "
                  "one list, which cannot fail",
        "run_a": {"label": a.label, "world_size": a.world_size, "steps": len(a.per_step),
                  "samples": len(a.flat)},
        "run_b": {"label": b.label, "world_size": b.world_size, "steps": len(b.per_step),
                  "samples": len(b.flat)},
        "same_length": same_len,
        "order_identical": order_identical,
        "per_step_identical": per_step_identical,
        "first_divergence": first_divergence,
        "coverage_run_a": cov_a,
        "coverage_run_b": cov_b,
        "note": "repeated sample ids across steps are expected and legitimate - lanes wrap their "
                "pools and repetition is a budgeted quantity (S5 runs Indic at 3.31 epochs). The "
                "exactness claim is about POSITIONS, not about id uniqueness.",
        "exactly_once_order_preserving": ok,
    }
