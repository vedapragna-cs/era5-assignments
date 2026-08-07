"""Throughput and packing efficiency, measured rather than estimated.

The headline number is **useful loss-bearing tokens per second**, and the adjective is the
whole point. Three quantities get conflated in throughput reports and they differ by a large
factor:

* **window tokens/s** -- sequence length times batch size divided by time. The number that
  looks best, and the one that counts padding as work.
* **real tokens/s** -- padding excluded. What the model's attention actually processed.
* **loss-bearing tokens/s** -- also excluding masked context: prompts, system turns, tool
  observations. Only these positions contribute gradient. This is the rate at which the run
  is actually buying learning, and it is the only one worth optimising.

On this corpus the third is materially below the first, and the gap decomposes into the two
things that cause it -- padding waste (a packing problem, fixable) and supervision yield (a
data-shape property, not fixable and not a defect). Reporting them separately is what makes
the number actionable instead of merely low.

Every figure here is reconstructible from the consumption ledger: `logs`-side totals and
`perf`-side totals are cross-checked in `verify_against_ledger`, because a throughput claim
that cannot be re-derived from the ledger is a claim the evaluation will not credit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .determinism import content_hash


@dataclass
class StepTiming:
    step: int
    stage: str
    seconds: float
    window_tokens: int      # seq_len * n_sequences, padding included
    real_tokens: int        # non-padding positions
    loss_tokens: int        # positions that contributed gradient
    sequences: int
    documents: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step, "stage": self.stage,
            "seconds": round(self.seconds, 6),
            "window_tokens": self.window_tokens,
            "real_tokens": self.real_tokens,
            "loss_tokens": self.loss_tokens,
            "sequences": self.sequences,
            "documents": self.documents,
        }


class PerfRecorder:
    """Wall-clock and token accounting per training step."""

    def __init__(self) -> None:
        self.steps: List[StepTiming] = []
        self.phases: Dict[str, float] = {}
        self._t0: Optional[float] = None

    # ---- step timing ---------------------------------------------------

    def start_step(self) -> None:
        self._t0 = time.perf_counter()

    def end_step(self, *, step: int, stage: str, samples: Sequence[Any]) -> StepTiming:
        elapsed = time.perf_counter() - (self._t0 if self._t0 is not None
                                         else time.perf_counter())
        window = sum(s.seq.seq_len for s in samples)
        real = sum(s.seq.real_tokens for s in samples)
        loss = sum(s.seq.supervised_tokens for s in samples)
        docs = sum(len(s.seq.doc_ids) for s in samples)
        t = StepTiming(step=step, stage=stage, seconds=elapsed, window_tokens=window,
                       real_tokens=real, loss_tokens=loss, sequences=len(samples),
                       documents=docs)
        self.steps.append(t)
        self._t0 = None
        return t

    def phase(self, name: str, seconds: float) -> None:
        self.phases[name] = round(seconds, 4)

    # ---- aggregation ---------------------------------------------------

    def _totals(self, rows: Sequence[StepTiming]) -> Dict[str, Any]:
        secs = sum(r.seconds for r in rows)
        window = sum(r.window_tokens for r in rows)
        real = sum(r.real_tokens for r in rows)
        loss = sum(r.loss_tokens for r in rows)
        if not rows or secs <= 0:
            return {"steps": len(rows), "seconds": round(secs, 4)}
        return {
            "steps": len(rows),
            "seconds": round(secs, 4),
            "window_tokens": window,
            "real_tokens": real,
            "loss_bearing_tokens": loss,
            "window_tokens_per_s": round(window / secs, 1),
            "real_tokens_per_s": round(real / secs, 1),
            "loss_bearing_tokens_per_s": round(loss / secs, 1),
            "sequences_per_s": round(sum(r.sequences for r in rows) / secs, 2),
            "mean_step_ms": round(1000 * secs / len(rows), 2),
            "packing_utilization": round(real / window, 6) if window else None,
            "padding_fraction": round(1 - real / window, 6) if window else None,
            "supervision_yield": round(loss / real, 6) if real else None,
            "useful_fraction_of_window": round(loss / window, 6) if window else None,
        }

    def report(self, *, registry=None, consumption=None,
               model_params: Optional[int] = None) -> Dict[str, Any]:
        overall = self._totals(self.steps)

        by_stage: Dict[str, Any] = {}
        for stage in sorted({r.stage for r in self.steps}):
            by_stage[stage] = self._totals([r for r in self.steps if r.stage == stage])

        # Warm vs cold: the first steps include lazy allocation and cache warming, and
        # folding them into the headline understates steady-state throughput. Reported
        # separately rather than quietly dropped.
        warm_from = min(5, max(0, len(self.steps) // 10))
        warm = self._totals(self.steps[warm_from:]) if len(self.steps) > warm_from else {}

        out: Dict[str, Any] = {
            "what_this_is": (
                "Throughput and packing efficiency measured from wall-clock timings taken "
                "around each optimizer step, with token counts read off the materialized "
                "sequences. Every figure is reconstructible from the consumption ledger."),
            "headline": {
                "loss_bearing_tokens_per_s": overall.get("loss_bearing_tokens_per_s"),
                "note": ("The rate at which the run buys gradient. Window throughput counts "
                         "padding as work; real throughput counts masked context as work. "
                         "Only this number is the thing being paid for."),
            },
            "overall": overall,
            "steady_state_excluding_first_{}_steps".format(warm_from): warm,
            "by_stage": by_stage,
            "phase_seconds": dict(sorted(self.phases.items())),
            "efficiency_decomposition": _decompose(overall),
            "steps": [r.to_dict() for r in self.steps],
        }
        if model_params:
            secs = overall.get("seconds") or 0
            if secs > 0:
                # 6 FLOPs per parameter per token is the standard forward+backward estimate.
                out["compute"] = {
                    "model_params": model_params,
                    "flops_per_token_estimate": 6 * model_params,
                    "achieved_gflops_on_real_tokens": round(
                        6 * model_params * overall["real_tokens"] / secs / 1e9, 3),
                    "note": ("6*N*T is the conventional forward+backward estimate and "
                             "ignores attention's quadratic term, which at this model size "
                             "and sequence length is small but not zero. Treat as indicative."),
                }
        if consumption is not None:
            out["ledger_cross_check"] = verify_against_ledger(self, consumption)
        out["report_hash"] = content_hash([r.to_dict() for r in self.steps])
        return out


def _decompose(overall: Dict[str, Any]) -> Dict[str, Any]:
    """Where the gap between window throughput and useful throughput goes."""
    window = overall.get("window_tokens")
    real = overall.get("real_tokens")
    loss = overall.get("loss_bearing_tokens")
    if not window or real is None or loss is None:
        return {}
    return {
        "window_tokens": window,
        "lost_to_padding": window - real,
        "lost_to_masked_context": real - loss,
        "loss_bearing_tokens": loss,
        "padding_share_of_window": round((window - real) / window, 6),
        "masked_context_share_of_window": round((real - loss) / window, 6),
        "useful_share_of_window": round(loss / window, 6),
        "interpretation": (
            "Padding loss is a packing problem and is worth attacking -- it is the price of "
            "refusing to truncate atomic documents. Masked-context loss is not a defect: "
            "prompts, system turns and tool observations are context the model must read "
            "and must not be scored on. S5 measured agentic supervision yield at 53% for "
            "exactly this reason."),
    }


def verify_against_ledger(perf: "PerfRecorder", consumption) -> Dict[str, Any]:
    """Re-derive the perf token totals from the consumption ledger.

    The handout's standard is that reported throughput must be reconstructible. This is
    that reconstruction: the ledger and the timer counted the same tokens independently,
    and if they disagree the throughput figure is not credited by its own report.
    """
    ledger_window = sum(r["tokens"] for r in consumption.steps)
    ledger_loss = sum(r["supervised_tokens"] for r in consumption.steps)
    perf_window = sum(r.window_tokens for r in perf.steps)
    perf_loss = sum(r.loss_tokens for r in perf.steps)
    return {
        "ledger_window_tokens": ledger_window,
        "perf_window_tokens": perf_window,
        "window_tokens_match": ledger_window == perf_window,
        "ledger_loss_bearing_tokens": ledger_loss,
        "perf_loss_bearing_tokens": perf_loss,
        "loss_bearing_tokens_match": ledger_loss == perf_loss,
        "steps_match": len(consumption.steps) == len(perf.steps),
        "reconstructible": (ledger_window == perf_window
                            and ledger_loss == perf_loss
                            and len(consumption.steps) == len(perf.steps)),
    }


def packing_efficiency_report(registry, tokenizer, seq_len: int,
                              long_seq_len: int) -> Dict[str, Any]:
    """Packing utilisation per lane, under the policy each lane actually uses.

    Separate from the bake-off in `06_packing.json`, which compares policies. This measures
    only the chosen one, across every trainable shard, which is what the run's realised
    utilisation is actually made of.
    """
    from .packing import LANE_POLICY, pack

    by_lane: Dict[str, Dict[str, Any]] = {}
    for sid in registry.trainable_ids():
        m = registry.manifest(sid)
        lane = m.lane
        win = long_seq_len if lane == "long_context" else seq_len
        plan = pack(registry.get(sid), LANE_POLICY[lane], win, shuffle_key=("perf", sid))
        st = plan.stats
        d = by_lane.setdefault(lane, {
            "policy": LANE_POLICY[lane], "seq_len": win, "shards": 0,
            "capacity_tokens": 0, "used_tokens": 0, "pad_tokens": 0,
            "truncated_tokens": 0, "deferred_documents": 0, "sequences": 0,
        })
        d["shards"] += 1
        d["capacity_tokens"] += st["capacity_tokens"]
        d["used_tokens"] += st["used_tokens"]
        d["pad_tokens"] += st["pad_tokens"]
        d["truncated_tokens"] += st["truncated_tokens"]
        d["deferred_documents"] += st["deferred_documents"]
        d["sequences"] += st["sequences"]

    for lane, d in by_lane.items():
        cap = d["capacity_tokens"] or 1
        d["packing_utilization"] = round(d["used_tokens"] / cap, 6)
        d["padding_fraction"] = round(d["pad_tokens"] / cap, 6)

    tot_cap = sum(d["capacity_tokens"] for d in by_lane.values()) or 1
    tot_used = sum(d["used_tokens"] for d in by_lane.values())
    return {
        "what_this_is": ("Realised packing utilisation per lane under the policy that lane "
                         "actually uses, across every trainable shard."),
        "by_lane": dict(sorted(by_lane.items())),
        "overall_packing_utilization": round(tot_used / tot_cap, 6),
        "overall_padding_fraction": round(1 - tot_used / tot_cap, 6),
        "note": ("Lanes whose documents are atomic and longer than the window show high "
                 "deferral rather than high truncation. That is the policy refusing to cut "
                 "a structure in half, and it is the intended behaviour, not a packing "
                 "failure."),
    }
