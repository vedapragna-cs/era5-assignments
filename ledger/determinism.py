"""Determinism primitives.

The session made one point sharply: a Python seed is only stable *while that kernel is
alive*. Restart the process and a "seeded" run can diverge. Every source of randomness in
this system is therefore **counter-based**: a draw is a pure function of an explicit key
tuple, never of accumulated interpreter state. No global `random`, no bare `np.random`,
no wall-clock, no `hash()`.

Two rules the rest of the codebase relies on:

1.  Anything that is hashed goes through `canonical_json`. Sorted keys, no float
    ambiguity, UTF-8, no timestamps.
2.  Anything random takes a `key` describing *where in the run it is*, so it can be
    recomputed from scratch on another machine on another day.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# canonical serialisation
# ---------------------------------------------------------------------------

_FLOAT_DECIMALS = 12


def _canonicalize(obj: Any) -> Any:
    """Recursively coerce an object into a form that serialises identically everywhere.

    Floats are the subtle case: repr() of a float is platform-stable in CPython, but a
    value that arrives as 0.1+0.2 and one that arrives as 0.30000000000000004 should not
    produce different shard hashes. We round to a fixed number of decimals, which is far
    below any precision this system actually depends on.
    """
    if isinstance(obj, dict):
        return {str(k): _canonicalize(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, set):
        return [_canonicalize(v) for v in sorted(obj, key=str)]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        rounded = round(obj, _FLOAT_DECIMALS)
        # normalise -0.0 -> 0.0 so sign of zero cannot change a hash
        return rounded + 0.0
    if isinstance(obj, (np.ndarray,)):
        return [_canonicalize(v) for v in obj.tolist()]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON. Same object -> same string, on any machine, on any day."""
    return json.dumps(
        _canonicalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(obj: Any) -> str:
    """Content hash of any JSON-able object. This is the identity of an artifact."""
    return sha256_bytes(canonical_bytes(obj))


def short_hash(obj: Any, n: int = 12) -> str:
    return content_hash(obj)[:n]


def hash_token_array(tokens: Sequence[int]) -> str:
    """Hash a token array via its raw little-endian int32 buffer.

    Faster than JSON for the large arrays inside a shard, and independent of how the
    array happens to be typed in memory.
    """
    arr = np.asarray(tokens, dtype=np.int32)
    return sha256_bytes(arr.tobytes(order="C"))


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_source_tree(paths: Iterable[str]) -> str:
    """Hash of the code that produced an artifact.

    The manifest records this so a shard can be tied to the exact cleaning/packing code
    that built it -- the transcript's "what is the hash of that particular code file".
    """
    entries: List[Tuple[str, str]] = []
    for p in sorted(paths):
        if os.path.isfile(p):
            entries.append((os.path.basename(p), hash_file(p)))
    return content_hash(entries)


def git_commit(default: str = "unversioned") -> str:
    """Best-effort git commit of the working tree. Never raises, never blocks a run."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.decode().strip()
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# counter-based randomness
# ---------------------------------------------------------------------------


def derive_seed(*key: Any) -> int:
    """Derive a 64-bit seed from an explicit key tuple.

    `derive_seed("pack", run_id, stage, shard_id, 7)` returns the same integer in every
    process, on every machine, forever. That is the whole point.
    """
    digest = hashlib.blake2b(canonical_bytes(list(key)), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def rng(*key: Any) -> np.random.Generator:
    """A fresh numpy Generator bound to an explicit key. Stateless across calls."""
    return np.random.Generator(np.random.PCG64(derive_seed(*key)))


def stable_shuffle(items: Sequence[Any], *key: Any) -> List[Any]:
    """Shuffle that depends only on the key, not on call order or interpreter state."""
    idx = np.arange(len(items))
    rng(*key).shuffle(idx)
    return [items[int(i)] for i in idx]


def stable_uniform(*key: Any) -> float:
    """A single reproducible float in [0, 1) for a given key."""
    return float(derive_seed(*key) % (1 << 53)) / float(1 << 53)


# ---------------------------------------------------------------------------
# torch determinism
# ---------------------------------------------------------------------------


def configure_torch_determinism(seed: int) -> Dict[str, Any]:
    """Pin every torch/CUDA knob that can make two identical runs disagree.

    Returns the settings actually applied, so the ledger can record them. If a future run
    diverges, the first question is "were these the same", and the answer is in the log.
    """
    import torch

    applied: Dict[str, Any] = {}

    torch.manual_seed(seed)
    applied["torch_manual_seed"] = seed

    # Single-threaded BLAS: multi-threaded reduction order is not deterministic, and a
    # different summation order changes the low bits of the loss, which is enough to make
    # a replay assertion fail.
    torch.set_num_threads(int(os.environ.get("LEDGER_TORCH_THREADS", "1")))
    applied["torch_num_threads"] = torch.get_num_threads()

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        applied["deterministic_algorithms"] = True
    except Exception as exc:  # pragma: no cover - platform dependent
        applied["deterministic_algorithms"] = "unavailable: {}".format(exc)

    if torch.cuda.is_available():  # pragma: no cover - no GPU in the demo
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        applied["cudnn_deterministic"] = True

    applied["torch_version"] = torch.__version__
    return applied


def env_fingerprint() -> Dict[str, Any]:
    """Everything about the environment that could plausibly change a number.

    Recorded once per run. When a replay diverges this is the diff you read first.
    """
    import platform
    import sys

    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "git_commit": git_commit(),
    }


__all__ = [
    "canonical_json",
    "canonical_bytes",
    "content_hash",
    "short_hash",
    "sha256_bytes",
    "hash_token_array",
    "hash_file",
    "hash_source_tree",
    "git_commit",
    "derive_seed",
    "rng",
    "stable_shuffle",
    "stable_uniform",
    "configure_torch_determinism",
    "env_fingerprint",
]
