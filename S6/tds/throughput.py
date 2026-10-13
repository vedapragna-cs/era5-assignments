"""Throughput, measured from real wall-clock.

The metric that matters is not tokens/sec. It is **useful loss-bearing tokens/sec at the target
mixture**: a loader can look fast while shovelling padding, context-only tokens, and candidates OPUS
is about to discard.

Scope, stated so the number is not read as something it is not: this is a CPU model in float64 over a
small corpus. The absolute tokens/sec is a property of this demo and predicts nothing about 120B on
accelerators. What IS scale-independent, and therefore what we report as the finding, are the ratios:
packing utilization, the useful/raw token fraction, the accepted/prepared fraction, and loader time as
a share of step time.

We deliberately do not model "GPU idle". Measured across 648 configurations of the course throughput
widget, exactly 1 produced any loader wait (peak 2.6% idle) because its storage model is ~100x
under-scaled. Reproducing an unreachable metric would be worse than omitting it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThroughputReport:
    steps: int
    wall_seconds: float
    loader_seconds: float
    raw_tokens: int
    loss_bearing_tokens: int
    pad_tokens: int
    prepared_samples: int
    accepted_samples: int

    def to_record(self) -> dict:
        w = max(self.wall_seconds, 1e-9)
        packed = self.raw_tokens - self.pad_tokens
        return {
            "scope": "measured wall-clock, CPU float64 toy model, small corpus. Ratios are the "
                     "finding; absolute tokens/sec is not transferable to accelerator scale.",
            "steps": self.steps,
            "wall_seconds": round(self.wall_seconds, 6),
            "candidate_preparation_seconds": round(self.loader_seconds, 6),
            "candidate_preparation_share_of_step_time": round(self.loader_seconds / w, 6),
            "candidate_preparation_note": "this is planning PLUS OPUS scoring, and OPUS dominates "
                                          "it - scoring every candidate is a forward/backward per "
                                          "sample. Calling it 'loader time' overstated I/O by "
                                          "roughly two orders of magnitude; the selector is the "
                                          "cost, not the loader.",
            "wall_seconds_note": "wall-clock on a contended CPU box. useful_loss_bearing_tokens_"
                                 "per_sec moved 4.3% between two runs on the same machine purely "
                                 "from load, so treat it as an order of magnitude, not a figure.",
            "raw_tokens": self.raw_tokens,
            "pad_tokens": self.pad_tokens,
            "packed_tokens": packed,
            "loss_bearing_tokens": self.loss_bearing_tokens,
            "packing_utilization": round(packed / self.raw_tokens, 6) if self.raw_tokens else 0.0,
            "useful_fraction_of_raw": round(self.loss_bearing_tokens / self.raw_tokens, 6)
                                      if self.raw_tokens else 0.0,
            "raw_tokens_per_sec": round(self.raw_tokens / w, 2),
            # Three rates, because conflating them overstates throughput by a large factor:
            # window = padding counted as work; raw = padding excluded; useful = also excluding
            # masked context, i.e. the only positions that actually buy gradient.
            # raw_tokens is already the padded window length (sum of len(seq.tokens)), so window
            # tokens ARE raw tokens. An earlier version added pad_tokens on top and published a
            # quantity that does not exist, inflated by ~10.7%.
            "window_tokens": self.raw_tokens,
            "window_tokens_per_sec": round(self.raw_tokens / w, 2),
            "content_tokens": self.raw_tokens - self.pad_tokens,
            "content_tokens_per_sec": round((self.raw_tokens - self.pad_tokens) / w, 2),
            "useful_loss_bearing_tokens_per_sec": round(self.loss_bearing_tokens / w, 2),
            "prepared_samples": self.prepared_samples,
            "accepted_samples": self.accepted_samples,
            "opus_accept_fraction": round(self.accepted_samples / self.prepared_samples, 6)
                                    if self.prepared_samples else 0.0,
            "reconstruction": {
                "packing_utilization": "packed_tokens / raw_tokens",
                "useful_loss_bearing_tokens_per_sec": "loss_bearing_tokens / wall_seconds",
                "note": "every input is summed from the per-step log in performance.json.steps_detail (`steps` is a count, not the array)",
                "opus_scope": "prepared/accepted are scoped to branch run-a and to the steps in steps_detail - the same population as the throughput numbers",
            },
        }


def summarize(step_log: list[dict], *, prepared: int, accepted: int) -> ThroughputReport:
    return ThroughputReport(
        steps=len(step_log),
        wall_seconds=sum(s["seconds"] for s in step_log),
        loader_seconds=sum(s["loader_seconds"] for s in step_log),
        raw_tokens=sum(s["n_tokens"] for s in step_log),
        loss_bearing_tokens=sum(s["n_loss_bearing"] for s in step_log),
        pad_tokens=sum(s["n_pad"] for s in step_log),
        prepared_samples=prepared,
        accepted_samples=accepted,
    )


def packing_table(by_policy: dict[str, list]) -> list[dict]:
    """E3: what each policy actually costs, computed from real packed sequences."""
    rows = []
    for policy, seqs in sorted(by_policy.items()):
        if not seqs:
            continue
        total = sum(s.length for s in seqs)
        pad = sum(s.n_pad for s in seqs)
        loss = sum(s.n_loss_bearing for s in seqs)
        trunc = sum(len(s.truncated) for s in seqs)
        # A document appearing in more than one window was SPLIT (all its tokens survive, its
        # context does not); a document in `truncated` had content dropped. Reporting only the
        # second made the headline "concat_chop 74 vs best_fit 0" compare different events: both
        # policies split over-length documents, and only concat_chop labelled it.
        appearances: dict[str, int] = {}
        for sq in seqs:
            for d in set(sq.members):
                appearances[d] = appearances.get(d, 0) + 1
        split_docs = sum(1 for v in appearances.values() if v > 1)
        rows.append({
            "policy": policy,
            "n_sequences": len(seqs),
            "capacity_tokens": total,
            "pad_tokens": pad,
            "utilization": round((total - pad) / total, 4) if total else 0.0,
            "loss_bearing_tokens": loss,
            "loss_bearing_fraction": round(loss / total, 4) if total else 0.0,
            "documents_truncated": trunc,
            "documents_split_across_windows": split_docs,
            "documents_seen": len(appearances),
        })
    return rows

def rank_compute_skew(samples_by_rank: dict, seq_of, *, world_size: int) -> dict:
    """Per-rank attention cost implied by PACKING COMPOSITION, not by token count.

    MegaScale-Data (arXiv:2504.09844) makes the point precisely: because attention is quadratic in
    segment length, a sequence built from a 30-token and a 70-token document costs ~16% more than one
    built from two 50-token documents - **at identical token counts**. Data-parallel ranks that each
    receive the same number of tokens can therefore still finish at different times, and the slowest
    rank sets the step time. The straggler is invisible to any monitor that counts tokens.

    Our loader balances samples per rank (`global_index % world_size`), which balances token count
    and says nothing about composition. So this measures the exposure rather than fixing it:
    attention cost is modelled as sum(len^2) over segments, which is the shape of the real cost even
    though the constant is meaningless on a CPU toy.
    """
    cost_by_rank: dict[int, float] = {}
    tokens_by_rank: dict[int, int] = {}
    for rank, planned in sorted(samples_by_rank.items()):
        total_cost, total_tok = 0.0, 0
        for ps in planned:
            seq = seq_of(ps.sample_id)
            lens: dict[int, int] = {}
            for sid in seq.segment_ids:
                if sid >= 0:
                    lens[sid] = lens.get(sid, 0) + 1
            total_cost += sum(v * v for v in lens.values())
            total_tok += len(seq.tokens)
        cost_by_rank[rank] = total_cost
        tokens_by_rank[rank] = total_tok

    costs = list(cost_by_rank.values())
    if not costs or max(costs) == 0:
        return {"measured": False, "reason": "no packed samples to measure"}
    mean = sum(costs) / len(costs)
    toks = list(tokens_by_rank.values())
    tok_mean = sum(toks) / len(toks)
    return {
        "measured": True,
        "world_size": world_size,
        "model": "attention cost proxied by sum(segment_length^2); constant is meaningless on a CPU "
                 "toy, the RATIO between ranks is the measurement",
        "attention_cost_by_rank": {str(k): round(v, 1) for k, v in cost_by_rank.items()},
        "tokens_by_rank": {str(k): v for k, v in tokens_by_rank.items()},
        "token_imbalance_pct": round(100 * (max(toks) - min(toks)) / tok_mean, 2) if tok_mean else 0,
        "compute_imbalance_pct": round(100 * (max(costs) - min(costs)) / mean, 2),
        "straggler_rank": max(cost_by_rank, key=lambda k: cost_by_rank[k]),
        "step_time_is_set_by": "the slowest rank, so compute_imbalance_pct is wasted GPU time per "
                               "step, not an average",
        "interpretation": "if compute imbalance exceeds token imbalance, ranks are unequal because "
                          "of how documents were PACKED, not how many tokens they were given - a "
                          "straggler no token-counting monitor can see",
        "not_fixed_here": "balancing composition across ranks means the rank partition stops being "
                          "`global_index % world_size`, which is exactly the property that makes "
                          "resumption across a changed world size exact. Fixing this needs a "
                          "composition-aware partition that is still a pure function of the global "
                          "index; it is a real design problem, not an oversight.",
    }

def by_stage(step_log: list[dict], *, branch: str = "run-a") -> list[dict]:
    """Throughput per curriculum stage.

    A run total hides the thing a mixture plan most needs to know: stages differ in sequence length,
    lane composition and packing policy, so they do not cost the same per useful token. Reporting one
    number for a five-stage curriculum averages away exactly the variation the plan is made of.
    """
    rows: dict[str, dict] = {}
    for r in step_log:
        if branch and r.get("branch") != branch:
            continue
        st = rows.setdefault(r.get("stage", "?"), {
            "stage": r.get("stage", "?"), "steps": 0, "seconds": 0.0,
            "raw_tokens": 0, "loss_bearing_tokens": 0, "pad_tokens": 0})
        st["steps"] += 1
        st["seconds"] += r["seconds"]
        st["raw_tokens"] += r["n_tokens"]
        st["loss_bearing_tokens"] += r["n_loss_bearing"]
        st["pad_tokens"] += r["n_pad"]
    out = []
    for st in rows.values():
        secs = st["seconds"] or 1e-9
        raw = st["raw_tokens"] or 1
        out.append({
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()},
            "packing_utilization": round((raw - st["pad_tokens"]) / raw, 6),
            "useful_loss_bearing_tokens_per_sec": round(st["loss_bearing_tokens"] / secs, 2),
            "raw_tokens_per_sec": round(st["raw_tokens"] / secs, 2),
        })
    return out
