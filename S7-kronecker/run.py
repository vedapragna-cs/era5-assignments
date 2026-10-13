"""Run the full comparison and write the evidence.

Four arms, N seeds, identical data and identical trunk. Everything the README
claims is computed here and written to `results/evidence.json`; nothing is
reported by hand.

    python run.py                 # the real run
    python run.py --smoke         # ~2 minutes, to check the pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kron2 import corpus, data, train
from kron2.model import HEADS

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

MODEL_CFG = dict(d_model=256, n_layers=4, n_heads=4, d_ff=1024)


class Tee:
    """Write to stdout and the log at once, so run.log is the real transcript."""

    def __init__(self, path: str):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def mean_std(xs: List[float]):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return round(xs[0], 6), 0.0
    return round(statistics.mean(xs), 6), round(statistics.stdev(xs), 6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--heldout-coverage", type=float, default=0.01)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to verify the pipeline, not to draw conclusions")
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.seeds, args.seq_len = 40, [0], 128

    os.makedirs(RESULTS, exist_ok=True)
    log = Tee(os.path.join(RESULTS, "run.log"))
    t_start = time.time()

    log("=" * 78)
    log("Kronecker Embeddings V2 -- P5, invertible Kronecker")
    log("=" * 78)

    # ---- data ------------------------------------------------------------
    manifest_path = os.path.join(ROOT, "data", "corpus_manifest.json")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    text = corpus.load()

    # The manifest hash must describe the string `load` actually returns, not the
    # bytes that happened to be written. These drifted apart once already, via
    # CRLF: text-mode reading collapsed 40,634 line endings, so the recorded hash
    # described a corpus no stage of the pipeline ever saw.
    integrity = corpus.verify()
    if not integrity["match"]:
        raise SystemExit("corpus does not match its manifest: {}".format(
            json.dumps(integrity, indent=2)))

    tok = data.train_tokenizer(text)
    stream = data.TokenStream(text, tok, seq_len=args.seq_len)

    held = stream.select_heldout(coverage=args.heldout_coverage, seed=0)
    stream.set_heldout(held)
    held_report = stream.heldout_report(held, tok)

    log("\ncorpus   {:,} bytes  sha256 {}  multibyte {:.2%}".format(
        manifest["bytes"], manifest["sha256"][:16], manifest["multibyte_fraction"]))
    log("tokens   {:,} train / {:,} val   |V| = {:,}".format(
        len(stream.train), len(stream.val), stream.vocab_size))
    log("byte table {:,} bytes, against {:,} for the dense equivalent".format(
        stream.table_stats["table_bytes"],
        stream.table_stats["dense_table_bytes_would_be"]))
    log("withheld {} token ids, {:.2%} of training positions, "
        "{:,} val positions scored".format(
            held_report["n_tokens_withheld"], held_report["train_coverage"],
            held_report["val_positions_scored"]))
    log("  e.g. " + " ".join(repr(e) for e in held_report.get("examples", [])[:14]))

    device = train.pick_device(args.device)
    log("\ndevice {}   steps {}   batch {}   seeds {}   {:.2f} epochs".format(
        device, args.steps, args.batch_size, args.seeds,
        args.steps * args.batch_size * args.seq_len / len(stream.train)))
    log("model  " + json.dumps(MODEL_CFG))

    # ---- arms ------------------------------------------------------------
    runs: List[Dict] = []
    for head in HEADS:
        log("\n" + "-" * 78)
        log(head)
        log("-" * 78)
        for seed in args.seeds:
            # One checkpoint per arm, from the first seed, so `demo.py` can show
            # what the head actually emits. Gitignored: 21-38 MB each.
            ckpt = (os.path.join(RESULTS, "ckpt_{}.pt".format(head))
                    if seed == args.seeds[0] and not args.smoke else None)
            # recorded relative, so the artefact does not carry a local home dir
            r = train.train_arm(head, stream, steps=args.steps,
                                batch_size=args.batch_size, lr=args.lr,
                                seed=seed, device=device,
                                eval_every=max(1, args.steps // 4),
                                model_cfg=MODEL_CFG, log=log, checkpoint=ckpt)
            r.pop("model", None)          # keep the record JSON-serialisable
            if r.get("checkpoint"):
                r["checkpoint"] = os.path.relpath(r["checkpoint"], ROOT)
            runs.append(r)

    # ---- aggregate -------------------------------------------------------
    summary = {}
    for head in HEADS:
        rs = [r for r in runs if r["head"] == head]
        finals = [r["final"] for r in rs]
        bpb_m, bpb_s = mean_std([f["bits_per_byte"] for f in finals])
        seen_m, _ = mean_std([f["seen"]["bits_per_byte"] for f in finals])
        hb_m, hb_s = mean_std([f["heldout"]["bits_per_byte"] for f in finals])
        acc_m, acc_s = mean_std([f["exact_token_accuracy"] for f in finals])
        hacc_m, hacc_s = mean_std([f["heldout"]["exact_token_accuracy"] for f in finals])
        summary[head] = {
            "seeds": len(rs),
            "params": rs[0]["params"],
            "bits_per_byte": {"mean": bpb_m, "std": bpb_s},
            "bits_per_byte_seen": {"mean": seen_m},
            "bits_per_byte_heldout": {"mean": hb_m, "std": hb_s},
            "exact_token_accuracy": {"mean": acc_m, "std": acc_s},
            "exact_token_accuracy_heldout": {"mean": hacc_m, "std": hacc_s},
            "wall_clock_s": round(statistics.mean([r["wall_clock_s"] for r in rs]), 1),
        }

    log("\n" + "=" * 78)
    log("SUMMARY   (mean +/- sd over {} seeds)".format(len(args.seeds)))
    log("=" * 78)
    log("{:<11} {:>11} {:>16} {:>16} {:>12}".format(
        "head", "out params", "bpb (all)", "bpb (held-out)", "exact held"))
    for head in HEADS:
        s = summary[head]
        def f(d, k="mean"):
            v = d.get(k)
            return "n/a" if v is None else "{:.4f}".format(v)
        log("{:<11} {:>11,} {:>16} {:>16} {:>12}".format(
            head, s["params"]["output"],
            "{} +/- {}".format(f(s["bits_per_byte"]), f(s["bits_per_byte"], "std")),
            "{} +/- {}".format(f(s["bits_per_byte_heldout"]),
                               f(s["bits_per_byte_heldout"], "std")),
            f(s["exact_token_accuracy_heldout"])))

    evidence = {
        "generated_by": "run.py",
        "argv": sys.argv[1:],
        "smoke": args.smoke,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": device,
        },
        "corpus": {k: v for k, v in manifest.items() if k != "sources"},
        "corpus_integrity": integrity,
        "corpus_sources": manifest["sources"],
        "tokenizer_and_tables": stream.stats(),
        "heldout": held_report,
        "config": {"steps": args.steps, "batch_size": args.batch_size,
                   "seq_len": args.seq_len, "lr": args.lr,
                   "seeds": args.seeds, "model": MODEL_CFG},
        "summary": summary,
        "runs": runs,
        "wall_clock_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(RESULTS, "evidence.json"), "w") as fh:
        json.dump(evidence, fh, indent=2)

    log("\ntotal {:.1f} min -> results/evidence.json, results/run.log".format(
        (time.time() - t_start) / 60))
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
