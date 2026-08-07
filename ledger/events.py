"""The execution log: every event the run emits, with a verdict attached.

`run.log` is not a transcript of print statements. Each line is a structured event with a
name, a verdict, and the artifact that substantiates it, so a reader can go from a `[PASS]`
line to the file that proves it without reading any code.

    [   2.11s] [PASS] eval_shard_blocked            blocked=48 reason=split_not_trainable
                                                    -> manifests/shard_index.json

The verdict is computed from the value being logged, never passed in as a literal. A helper
that lets a caller write `emit_pass("thing_worked")` unconditionally is a helper for
producing evidence that isn't evidence -- so `check()` takes the observed value and the
condition, and derives the verdict itself.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

PASS = "PASS"
FAIL = "FAIL"
INFO = "INFO"

# The events the handout requires the log to contain, in the order the run produces them.
# `EventLog.missing_required()` reports any that never fired -- a required event that is
# silently absent is worse than one that fails, because nothing draws attention to it.
REQUIRED_EVENTS = (
    "shards_created",
    "manifests_validated",
    "eval_shard_blocked",
    "mixture_compiled",
    "batches_packed",
    "opus_decisions_recorded",
    "checkpoint_saved",
    "crash_simulated",
    "run_resumed",
    "historical_stream_replayed",
    "branch_forked",
    "audit_completed",
    "performance_measured",
)


@dataclass
class Event:
    seq: int
    elapsed_s: float
    level: str
    name: str
    detail: Dict[str, Any]
    evidence: Optional[str] = None

    def line(self) -> str:
        bits = " ".join("{}={}".format(k, _short(v)) for k, v in self.detail.items())
        head = "[{:8.2f}s] [{}] {:<34}".format(self.elapsed_s, self.level, self.name)
        tail = " -> {}".format(self.evidence) if self.evidence else ""
        return "{} {}{}".format(head, bits, tail).rstrip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq, "elapsed_s": round(self.elapsed_s, 3),
            "level": self.level, "event": self.name,
            "detail": self.detail, "evidence": self.evidence,
        }


def _short(v: Any) -> str:
    if isinstance(v, float):
        return "{:.4f}".format(v)
    if isinstance(v, (list, tuple)):
        return "[{}]".format(",".join(str(x) for x in list(v)[:4]))
    if isinstance(v, dict):
        return "{" + ",".join("{}:{}".format(k, _short(x))
                              for k, x in list(v.items())[:4]) + "}"
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


class EventLog:
    def __init__(self, path: str, echo: bool = True) -> None:
        self.path = path
        self.echo = echo
        self.t0 = time.time()
        self.events: List[Event] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "w")
        self._banner()

    def _banner(self) -> None:
        for line in ("=" * 78,
                     "ERA V5 Session 6 -- the data ledger",
                     "run.log: every event, its verdict, and the artifact that proves it",
                     "=" * 78, ""):
            self._fh.write(line + "\n")
        self._fh.flush()

    # ---- emitting ------------------------------------------------------

    def _add(self, level: str, name: str, detail: Dict[str, Any],
             evidence: Optional[str]) -> Event:
        ev = Event(seq=len(self.events), elapsed_s=time.time() - self.t0,
                   level=level, name=name, detail=detail or {}, evidence=evidence)
        self.events.append(ev)
        self._fh.write(ev.line() + "\n")
        self._fh.flush()
        if self.echo:
            print("  " + ev.line())
        return ev

    def info(self, name: str, evidence: Optional[str] = None, **detail: Any) -> Event:
        return self._add(INFO, name, detail, evidence)

    def check(self, name: str, condition: bool, evidence: Optional[str] = None,
              **detail: Any) -> Event:
        """Record a check. The verdict comes from `condition`, evaluated by the caller
        against a real observed value -- never from a literal `True`."""
        return self._add(PASS if condition else FAIL, name, detail, evidence)

    def section(self, title: str) -> None:
        self._fh.write("\n--- {} {}\n".format(title, "-" * max(0, 60 - len(title))))
        self._fh.flush()
        if self.echo:
            print("\n--- {}".format(title))

    # ---- summary -------------------------------------------------------

    def counts(self) -> Dict[str, int]:
        out = {PASS: 0, FAIL: 0, INFO: 0}
        for e in self.events:
            out[e.level] = out.get(e.level, 0) + 1
        return out

    def failures(self) -> List[Event]:
        return [e for e in self.events if e.level == FAIL]

    def names(self) -> List[str]:
        return [e.name for e in self.events]

    def missing_required(self) -> List[str]:
        seen = set(self.names())
        return [r for r in REQUIRED_EVENTS if r not in seen]

    def find(self, name: str) -> Optional[Event]:
        for e in self.events:
            if e.name == name:
                return e
        return None

    def close(self) -> Dict[str, Any]:
        counts = self.counts()
        missing = self.missing_required()
        summary = {
            "events": len(self.events),
            "pass": counts[PASS],
            "fail": counts[FAIL],
            "info": counts[INFO],
            "required_events_present": len(REQUIRED_EVENTS) - len(missing),
            "required_events_total": len(REQUIRED_EVENTS),
            "required_events_missing": missing,
            "wall_clock_s": round(time.time() - self.t0, 2),
            "all_passed": counts[FAIL] == 0 and not missing,
        }
        self._fh.write("\n" + "=" * 78 + "\n")
        self._fh.write("SUMMARY  {} events  {} PASS  {} FAIL  {} INFO\n".format(
            summary["events"], summary["pass"], summary["fail"], summary["info"]))
        self._fh.write("required events present: {}/{}{}\n".format(
            summary["required_events_present"], summary["required_events_total"],
            "" if not missing else "  MISSING: {}".format(", ".join(missing))))
        for e in self.failures():
            self._fh.write("  FAIL  {}  {}\n".format(e.name, _short(e.detail)))
        self._fh.write("RESULT: {}\n".format("ALL CHECKS PASSED" if summary["all_passed"]
                                             else "FAILURES PRESENT"))
        self._fh.write("=" * 78 + "\n")
        self._fh.flush()
        self._fh.close()
        return summary

    def to_json(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events]
