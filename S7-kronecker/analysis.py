"""Why the inverse is a classifier and not a nearest-neighbour lookup.

The session frames the reverse direction as: predict the Kronecker vector, then
find the token whose vector is closest. This script tests that reading against
the alternative -- P independent 256-way classifications -- on the codec alone,
so nothing here depends on how the training run happened to go.

The result is not the one the design argument wanted, and it is reported as
measured rather than as hoped:

1. **Concentration.** How separated the vocabulary's codec vectors are. Crowded,
   but not catastrophically so: mean cosine to the nearest other token is 0.76,
   and the median best-to-second gap is 0.05.

2. **Noise sensitivity.** Nearest neighbour turns out to be *robust* to additive
   noise -- better than per-position argmax under an equal-energy budget. So the
   case against it is not fragility, and this measurement is kept precisely
   because it refutes the easy version of the argument. See the docstring on
   `measure_robustness` for why equal-energy is the wrong yardstick anyway.

3. **Reachability.** This is what actually settles it. Nearest neighbour is an
   argmin over a fixed table, so its range is exactly the |V| strings in that
   table -- it cannot emit anything else, however good the model gets. Measured
   on 15 real misspellings, proper nouns, identifiers, and multi-byte strings:
   nearest neighbour reaches 0, per-position argmax reaches 15. A decoder that
   cannot leave the vocabulary cannot deliver a vocabulary-free model.

   The second structural point needs no experiment: an MSE head has no normalised
   distribution, so it has no likelihood, cannot be scored in bits per byte, and
   cannot be sampled from.

4. **Scaling.** Output-side parameters against vocabulary size -- the "1M vocab"
   claim reduced to arithmetic.

    python analysis.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kron2 import codec as C
from kron2 import corpus, data

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")


def codec_matrix(labels: np.ndarray, lens: np.ndarray, ids: np.ndarray,
                 cfg=C.DEFAULT) -> np.ndarray:
    """Dense codec vectors for a *subset* of ids. Only ever called on a sample."""
    out = np.zeros((len(ids), cfg.dim), dtype=np.float32)
    for row, tid in enumerate(ids):
        L = int(lens[tid])
        if L == 0:
            continue
        scale = 1.0 / math.sqrt(L)
        for p in range(L):
            out[row, int(labels[tid, p]) * cfg.positions + p] = scale
    return out


# ---------------------------------------------------------------------------
# 1. concentration
# ---------------------------------------------------------------------------


def measure_concentration(labels, lens, vocab_size, n_probe=400, n_cand=8000,
                          seed=0) -> Dict:
    """How separated are the codec vectors?

    Reported as the gap between the best and second-best cosine similarity, which
    is the margin a nearest-neighbour decoder has to work with. Tokens sharing a
    prefix land close together by construction -- the codec is a sum over byte
    positions, so "start" and "starts" agree on five of six terms.

    The measurement is included to check whether the space is *so* crowded that
    nearest neighbour is hopeless. It is not: the median gap is around 0.05. That
    is a real answer and it argues against the fragility story, not for it.
    """
    rng = np.random.default_rng(seed)
    cand_ids = rng.choice(vocab_size, size=min(n_cand, vocab_size), replace=False)
    K = codec_matrix(labels, lens, cand_ids)                    # (C, D), unit norm

    probe_rows = rng.choice(len(cand_ids), size=n_probe, replace=False)
    sims = K[probe_rows] @ K.T                                  # (n_probe, C)
    sims[np.arange(n_probe), probe_rows] = -np.inf              # exclude self

    best = sims.max(axis=1)
    part = np.partition(sims, -2, axis=1)
    second = part[:, -2]

    return {
        "candidates": int(len(cand_ids)),
        "probes": int(n_probe),
        "mean_cosine_to_nearest_other": round(float(best.mean()), 6),
        "max_cosine_to_nearest_other": round(float(best.max()), 6),
        "mean_gap_best_to_second": round(float((best - second).mean()), 6),
        "median_gap_best_to_second": round(float(np.median(best - second)), 6),
        "fraction_with_gap_below_0.01": round(float(((best - second) < 0.01).mean()), 4),
    }


# ---------------------------------------------------------------------------
# 2. noise sensitivity -- the measurement that refutes the easy argument
# ---------------------------------------------------------------------------


def measure_robustness(labels, lens, vocab_size, sigmas: List[float],
                       n_probe=300, trials=3, seed=1) -> List[Dict]:
    """Additive-noise sensitivity of the two decoders, at equal signal energy.

    Reported because the result is *not* what the design argument would like, and
    stating it is cheaper than being caught by it.

    Each decoder gets the representation its own head emits: nearest neighbour
    sees the codec vector (``1/sqrt(L)`` at L coordinates, zero elsewhere, unit
    norm), per-position argmax sees the logit grid (PAD is a real class, so pad
    rows are one-hot at PAD). Handing both the *same* vector would look fairer and
    would not be -- on the codec vector every pad row is exactly zero, so argmax
    over 256 noisy zeros returns an arbitrary byte and the string never
    terminates, scoring 0.003 at sigma = 0.01. That measures the wrong object.

    Both signals are then scaled to unit L2 norm and given the same sigma.

    **This does not show nearest neighbour is the better decoder.** The equal-
    energy constraint is arbitrary and it favours the codec vector: it packs its
    energy into L coordinates (1/sqrt(5) = 0.447 for a 5-byte token) while the
    grid spreads the same energy over all 32 rows (1/sqrt(32) = 0.177), so the
    grid faces a 2.5x smaller per-coordinate margin by construction. A real
    classifier is under no such constraint -- cross-entropy drives the correct
    logit as high as it likes, and logit scale is free. An MSE head's target norm
    is pinned at 1 by the codec and cannot be scaled away.

    So the honest reading is narrow: nearest neighbour is not fragile against
    additive noise, and the case against it rests on `measure_reachability` and on
    having no likelihood at all -- not on noise.
    """
    rng = np.random.default_rng(seed)
    all_ids = np.arange(vocab_size)
    K = codec_matrix(labels, lens, all_ids)                     # (V, D) unit norm

    rows = rng.choice(vocab_size, size=n_probe, replace=False)
    rows = np.array([r for r in rows if lens[r] > 0])
    truth = [bytes(labels[r, :lens[r]].astype(np.uint8)) for r in rows]

    # The classifier's target, normalised to the same energy as the codec vector.
    P, NB = C.DEFAULT.positions, C.DEFAULT.n_bytes
    grids = np.zeros((len(rows), P, NB), dtype=np.float32)
    for i, r in enumerate(rows):
        grids[i, :, C.PAD_BYTE] = 1.0
        for p in range(int(lens[r])):
            grids[i, p, C.PAD_BYTE] = 0.0
            grids[i, p, int(labels[r, p])] = 1.0
    grids /= np.linalg.norm(grids.reshape(len(rows), -1), axis=1)[:, None, None]

    clean_codec = K[rows]
    out = []
    for sigma in sigmas:
        nn_ok = argmax_ok = total = 0
        for _ in range(trials):
            noisy_codec = clean_codec + rng.normal(0, sigma, clean_codec.shape)
            nn = (noisy_codec.astype(np.float32) @ K.T).argmax(axis=1)
            nn_ok += int((nn == rows).sum())

            noisy_grid = grids + rng.normal(0, sigma, grids.shape)
            for i in range(len(rows)):
                argmax_ok += C.decode_grid(noisy_grid[i]) == truth[i]
            total += len(rows)

        out.append({
            "sigma": sigma,
            "nearest_neighbour_exact": round(nn_ok / total, 4),
            "per_position_argmax_exact": round(argmax_ok / total, 4),
        })
    return out


# ---------------------------------------------------------------------------
# 3. reachability: what each decoder can emit at all
# ---------------------------------------------------------------------------


def measure_reachability(labels, lens, vocab_size, tok, cfg=C.DEFAULT) -> Dict:
    """Which strings each decoder is *capable* of producing.

    This is the argument that actually settles the decoder choice, and it is not
    statistical -- it is structural. Nearest neighbour returns ``argmin`` over a
    fixed table, so its range is exactly the |V| tokens in that table. It cannot
    emit a string that is not already a token, no matter how good the model is.
    That forfeits the entire point of P5.

    Per-position argmax returns whatever P independent byte choices produce, so
    its range is every byte string the codec can represent. The strings below are
    real cases a fixed vocabulary handles badly: misspellings, rare proper nouns,
    coined identifiers, and multi-byte scripts. For each, the question asked is
    the blunt one -- is this reachable at all?
    """
    vocab_strings = set()
    for i in range(vocab_size):
        vocab_strings.add(bytes(labels[i, :lens[i]].astype(np.uint8)))

    probes = [
        "netwrok", "kronekticus", "recieve", "teh",          # misspellings
        "Kubernetes", "Anthropic", "Vedapragna",             # proper nouns
        "get_event_loop", "__init__", "AsyncIOMotorClient",  # identifiers
        "café", "日本語", "नमस्ते",                              # multi-byte
        "supercalifragilistic", "qwxzjv",                    # coinages
    ]

    rows = []
    for s in probes:
        b = C.token_bytes(s, cfg)
        exact, recovered = C.round_trip(s, cfg)
        rows.append({
            "string": s,
            "bytes": len(b),
            "is_a_single_vocabulary_token": b in vocab_strings,
            "codec_round_trips": bool(exact),
            "reachable_by_nearest_neighbour": b in vocab_strings,
            "reachable_by_per_position_argmax": bool(exact),
        })

    n_nn = sum(r["reachable_by_nearest_neighbour"] for r in rows)
    n_am = sum(r["reachable_by_per_position_argmax"] for r in rows)
    return {
        "probes": rows,
        "reachable_by_nearest_neighbour": n_nn,
        "reachable_by_per_position_argmax": n_am,
        "n_probes": len(rows),
        "nearest_neighbour_range": int(vocab_size),
        "per_position_argmax_range_note":
            "every byte string of length <= {} that is valid UTF-8".format(cfg.positions),
    }


# ---------------------------------------------------------------------------
# 4. scaling
# ---------------------------------------------------------------------------


def measure_scaling(d_model=256, cfg=C.DEFAULT) -> Dict:
    """Output-side parameters against |V|, which is the whole point of P5."""
    rows = []
    for V in [16_384, 50_257, 128_000, 256_000, 1_000_000]:
        rows.append({
            "vocab_size": V,
            "vocab_head": V * d_model,
            "kron_head": cfg.dim * d_model,
            "kron_tied_head": cfg.dim,          # a bias vector, nothing more
            "dense_codec_table_bytes": V * cfg.dim * 4,
            "sparse_byte_table_bytes": V * cfg.positions * 8 + V * 8,
        })
    return {"d_model": d_model, "codec_dim": cfg.dim, "rows": rows}


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    text = corpus.load()
    tok = data.train_tokenizer(text)
    labels, lens, table_stats = data.build_byte_tables(tok)
    V = tok.get_vocab_size()

    print("measuring concentration ...")
    conc = measure_concentration(labels, lens, V)
    print(json.dumps(conc, indent=2))

    print("\nmeasuring decoder robustness ...")
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    rob = measure_robustness(labels, lens, V, sigmas)
    print("{:>7} {:>22} {:>24}".format("sigma", "nearest neighbour", "per-position argmax"))
    for r in rob:
        print("{:>7.2f} {:>22.4f} {:>24.4f}".format(
            r["sigma"], r["nearest_neighbour_exact"], r["per_position_argmax_exact"]))

    print("\nmeasuring reachability ...")
    reach = measure_reachability(labels, lens, V, tok)
    print("{:<24} {:>6} {:>10} {:>16}".format(
        "string", "bytes", "in vocab", "argmax-reachable"))
    for r in reach["probes"]:
        print("{:<24} {:>6} {:>10} {:>16}".format(
            repr(r["string"]), r["bytes"],
            "yes" if r["is_a_single_vocabulary_token"] else "no",
            "yes" if r["reachable_by_per_position_argmax"] else "no"))
    print("  nearest neighbour can emit {}/{}; per-position argmax {}/{}".format(
        reach["reachable_by_nearest_neighbour"], reach["n_probes"],
        reach["reachable_by_per_position_argmax"], reach["n_probes"]))

    print("\nparameter scaling ...")
    scal = measure_scaling()
    print("{:>10} {:>14} {:>14} {:>14}".format(
        "|V|", "vocab head", "kron head", "kron tied"))
    for r in scal["rows"]:
        print("{:>10,} {:>14,} {:>14,} {:>14,}".format(
            r["vocab_size"], r["vocab_head"], r["kron_head"], r["kron_tied_head"]))

    payload = {
        "codec": C.DEFAULT.to_dict(),
        "vocab_size": V,
        "byte_tables": table_stats,
        "concentration": conc,
        "robustness": rob,
        "reachability": reach,
        "scaling": scal,
    }
    with open(os.path.join(RESULTS, "analysis.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\n-> results/analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
