"""Canonical serialization, content hashing, hash chains, and derived randomness.

Everything reproducible in this system rests on three primitives defined here:

1. `chash` - SHA-256 over canonical JSON (sorted keys, no incidental whitespace). Two objects are
   "the same" iff their canonical hashes match. This is what makes shard immutability, batch
   identity and replay verification checkable rather than asserted.

2. `HashChain` - append-only log where each entry commits to the previous entry's hash. Editing any
   historical record breaks every link after it, and `verify` reports the first broken link.

3. `derive_rng` / `derive_int` - all randomness in the data path is *derived* from
   (run_seed, global_step, purpose), never drawn from a stateful global generator.

Why (3) matters, and it is the core of the assignment: a seed only reproduces a stream if every
consumer draws from it in the same order. A distributed loader cannot guarantee that - workers
restart, ranks retry, worker counts change between runs. Deriving each decision independently means
the stream is a pure function of (seed, step), so it survives restarts, machine changes and worker
recounts. This is why the ledger stores what was served rather than trusting a seed to regenerate it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np

CANON_SEPARATORS = (",", ":")


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic byte representation of a JSON-able object."""
    return json.dumps(
        obj, sort_keys=True, separators=CANON_SEPARATORS, ensure_ascii=False, default=_default
    ).encode("utf-8")


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON-serializable: {type(o)!r}")


def chash(obj: Any) -> str:
    """Content hash of any JSON-able object."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def bhash(data: bytes) -> str:
    """Content hash of raw bytes (used for shard token arrays)."""
    return hashlib.sha256(data).hexdigest()


def short(h: str, n: int = 12) -> str:
    return h[:n]


# --------------------------------------------------------------------------------------
# derived randomness
# --------------------------------------------------------------------------------------

def derive_seed(run_seed: int, *parts: Any) -> int:
    """A 64-bit seed that is a pure function of the run seed and the named coordinates."""
    digest = hashlib.sha256(canonical_bytes([run_seed, *parts])).digest()
    return int.from_bytes(digest[:8], "big")


def derive_rng(run_seed: int, *parts: Any) -> np.random.Generator:
    """An independent generator for one specific decision point."""
    return np.random.default_rng(derive_seed(run_seed, *parts))


def derive_int(run_seed: int, *parts: Any, bound: int) -> int:
    """A deterministic integer in [0, bound) for one specific decision point."""
    if bound <= 0:
        raise ValueError("bound must be positive")
    return derive_seed(run_seed, *parts) % bound


# --------------------------------------------------------------------------------------
# append-only hash chain
# --------------------------------------------------------------------------------------

GENESIS = "0" * 64


class HashChain:
    """Append-only log in which each entry commits to its predecessor.

    Entry i stores `prev_hash` = hash of entry i-1 and `entry_hash` = chash(payload | prev_hash).
    Any modification of a historical payload changes its entry_hash, which no longer matches the
    `prev_hash` recorded by its successor. `verify` returns the index of the first break.
    """

    def __init__(self) -> None:
        self.entries: list[dict] = []

    @property
    def head(self) -> str:
        return self.entries[-1]["entry_hash"] if self.entries else GENESIS

    def append(self, payload: dict) -> dict:
        prev = self.head
        entry = dict(payload)
        entry["prev_hash"] = prev
        entry["entry_hash"] = chash({"payload": payload, "prev_hash": prev})
        self.entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, int | None, str]:
        """Return (ok, first_bad_index, message)."""
        prev = GENESIS
        for i, entry in enumerate(self.entries):
            payload = {k: v for k, v in entry.items() if k not in ("prev_hash", "entry_hash")}
            if entry["prev_hash"] != prev:
                return False, i, f"entry {i}: prev_hash {short(entry['prev_hash'])} != expected {short(prev)}"
            expected = chash({"payload": payload, "prev_hash": entry["prev_hash"]})
            if entry["entry_hash"] != expected:
                return False, i, f"entry {i}: payload was modified (entry_hash mismatch)"
            prev = entry["entry_hash"]
        return True, None, f"chain intact across {len(self.entries)} entries"

    def to_jsonl(self) -> str:
        return "".join(
            json.dumps(e, sort_keys=True, separators=CANON_SEPARATORS, ensure_ascii=False) + "\n"
            for e in self.entries
        )

    @classmethod
    def from_entries(cls, entries: Iterable[dict]) -> "HashChain":
        chain = cls()
        chain.entries = [dict(e) for e in entries]
        return chain
