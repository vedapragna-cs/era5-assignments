"""Show what the heads actually emit, on tokens they were never trained to emit.

The tables in `run.py` say the byte head loses less on withheld tokens. This
shows the same fact one prediction at a time, which is harder to misread: the
context, the correct next token, and what each head produced for it.

Reads the checkpoints `run.py` writes. Run `run.py` first.

    python demo.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kron2 import codec as C
from kron2 import corpus, data, train
from kron2.model import KronLM, ModelConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")


def load_arm(head: str, device: str):
    path = os.path.join(RESULTS, "ckpt_{}.pt".format(head))
    if not os.path.exists(path):
        return None
    blob = torch.load(path, map_location=device, weights_only=False)
    c = blob["cfg"]
    cfg = ModelConfig(d_model=c["d_model"], n_layers=c["n_layers"],
                      n_heads=c["n_heads"], d_ff=c["d_ff"], seq_len=c["seq_len"],
                      vocab_size=c["vocab_size"], head=c["head"])
    m = KronLM(cfg).to(device)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m


def _edit(a: bytes, b: bytes) -> int:
    """Levenshtein distance. Exact-match rates are near zero on withheld tokens,
    so 'how close was it' carries the signal that a 0/1 tally cannot."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def show(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


@torch.no_grad()
def main() -> int:
    device = train.pick_device()
    text = corpus.load()
    tok = data.train_tokenizer(text)
    stream = data.TokenStream(text, tok, seq_len=256)
    stream.set_heldout(stream.select_heldout(coverage=0.01, seed=0))

    models = {h: load_arm(h, device) for h in ("vocab", "kron", "kron_tied")}
    if not any(models.values()):
        print("no checkpoints in results/; run `python run.py` first")
        return 1
    have = [h for h, m in models.items() if m is not None]
    print("loaded: {}\n".format(", ".join(have)))

    # Collect validation positions whose target is a withheld token.
    picks: List[Dict] = []
    for b in stream.sequential_batches("val", 8, device):
        held = b["tgt_heldout"]
        if not bool(held.any()):
            continue

        out = {h: models[h].forward(b["in_labels"], b["in_lens"])
               for h in have}

        idx = held.nonzero(as_tuple=False)
        for bi, ti in idx.tolist():
            if ti < 8:                       # need some context to print
                continue
            tgt_id = int(b["tgt_ids"][bi, ti])
            tgt_bytes = bytes(b["tgt_labels"][bi, ti, :b["tgt_lens"][bi, ti]]
                              .cpu().numpy().astype(np.uint8))
            ctx_ids = b["tgt_ids"][bi, max(0, ti - 8):ti].cpu().tolist()
            ctx = "".join(data.token_surface(tok, i) for i in ctx_ids)

            row = {"context": ctx, "target": show(tgt_bytes),
                   "target_bytes": len(tgt_bytes), "preds": {}}
            r_bytes = {}

            for h in have:
                cfg = models[h].cfg
                o = out[h][bi, ti]
                if h == "vocab":
                    pid = int(o.argmax(-1))
                    r_bytes[h] = data.token_raw_bytes(tok, pid)
                    row["preds"][h] = {
                        "emitted": data.token_surface(tok, pid),
                        "correct": pid == tgt_id,
                    }
                else:
                    grid = C.to_grid_torch(o, cfg.codec)          # (P, 256)
                    pred = grid.argmax(-1).cpu().numpy()
                    decoded = C.decode_labels(pred)
                    prob = F.softmax(grid, dim=-1)
                    per_byte = [round(float(prob[p, int(pred[p])]), 3)
                                for p in range(min(len(decoded) + 1, 12))]
                    r_bytes[h] = decoded
                    row["preds"][h] = {
                        "emitted": show(decoded),
                        "correct": decoded == tgt_bytes,
                        "byte_confidences": per_byte,
                    }
            row["edit"] = {h: _edit(r_bytes[h], tgt_bytes) for h in have}
            row["first_byte_ok"] = {
                h: bool(r_bytes[h][:1] == tgt_bytes[:1]) for h in have}
            picks.append(row)
        if False:
            break

    print("Every validation position whose target was withheld from all training "
          "losses:  {} positions.\n".format(len(picks)))
    w = max(len(h) for h in have)

    hits = [r for r in picks if r["preds"].get("kron_tied", {}).get("correct")]
    if hits:
        print("--- withheld tokens `kron_tied` spelled exactly right ---\n")
        for r in hits[:12]:
            print("context ...{}".format(r["context"][-46:].replace("\n", "\\n")))
            print("  {:<{w}} {!r}".format("TARGET", r["target"], w=w))
            for h in have:
                p = r["preds"][h]
                print("  {:<{w}} {!r:<22} {}".format(
                    h, p["emitted"], "correct" if p["correct"] else "", w=w))
            print()

    print("--- a sample of the rest ---\n")
    for r in picks[:6]:
        print("context ...{}".format(r["context"][-46:].replace("\n", "\\n")))
        print("  {:<{w}} {!r}".format("TARGET", r["target"], w=w))
        for h in have:
            p = r["preds"][h]
            print("  {:<{w}} {!r:<22} edit {}".format(
                h, p["emitted"], r["edit"][h], w=w))
        print()

    n = len(picks)
    tally = {}
    for h in have:
        corr = sum(1 for r in picks if r["preds"][h]["correct"])
        tally[h] = {
            "correct": corr, "total": n,
            "exact_rate": round(corr / max(1, n), 5),
            "mean_edit_distance": round(sum(r["edit"][h] for r in picks) / max(1, n), 3),
            "first_byte_accuracy": round(
                sum(r["first_byte_ok"][h] for r in picks) / max(1, n), 4),
        }

    print("over all {} withheld-target positions:\n".format(n))
    print("  {:<{w}}  {:>7}  {:>11}  {:>10}".format(
        "arm", "exact", "mean edit", "1st byte", w=w))
    for h in have:
        t = tally[h]
        print("  {:<{w}}  {:>7}  {:>11.3f}  {:>10.4f}".format(
            h, "{}/{}".format(t["correct"], t["total"]),
            t["mean_edit_distance"], t["first_byte_accuracy"], w=w))
    print("\nMean edit distance is to the correct byte string; '1st byte' is how "
          "often\nthe first byte was right. Both are defined for every arm because "
          "every\nprediction is ultimately a byte string.")

    # The tally is computed over every position; only a sample is serialised.
    # Keeping all 3,595 makes the artefact 2.8 MB of near-duplicate rows.
    keep = hits + [r for r in picks if not r["preds"]["kron_tied"]["correct"]][:150]
    with open(os.path.join(RESULTS, "demo.json"), "w") as fh:
        json.dump({"tally": tally,
                   "positions_scored": len(picks),
                   "positions_serialised": len(keep),
                   "exact_hits": hits,
                   "sample": keep,
                   "device": device}, fh, indent=2)
    print("\n-> results/demo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
