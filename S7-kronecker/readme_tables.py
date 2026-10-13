"""Fill the README's results tables from the artefacts.

The README's prose is written by hand; its numbers are not. This replaces the
region between the `<!-- RESULTS -->` / `<!-- HELDOUT -->` markers and their
`<!-- /... -->` counterparts, so re-running it after a new run updates the README
instead of leaving a stale table behind.

    python readme_tables.py
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
HEADS = ["vocab", "kron", "kron_tied", "regress"]


def fmt(x, d=3):
    return "—" if x is None else "{:.{d}f}".format(x, d=d)


def results_block(ev) -> str:
    S, C = ev["summary"], ev["config"]
    base = S["vocab"]["bits_per_byte"]["mean"]
    L = []
    L.append("{} seeds per arm, {:,} steps at batch {}, sequence length {} — "
             "{:.2f} epochs over {:,} training tokens. Mean ± sd across seeds.".format(
                 len(C["seeds"]), C["steps"], C["batch_size"], C["seq_len"],
                 C["steps"] * C["batch_size"] * C["seq_len"]
                 / ev["tokenizer_and_tables"]["train_tokens"],
                 ev["tokenizer_and_tables"]["train_tokens"]))
    L.append("")
    L.append("| arm | output params | bits/byte | vs. baseline | exact token acc. |")
    L.append("|---|---:|---:|---:|---:|")
    for h in HEADS:
        s = S[h]
        b = s["bits_per_byte"]["mean"]
        cell = ("*undefined*" if b is None
                else "**{} ± {}**".format(fmt(b), fmt(s["bits_per_byte"]["std"])))
        delta = "—" if b is None else "{}{}".format("+" if b > base else "", fmt(b - base))
        L.append("| `{}` | {:,} | {} | {} | {} |".format(
            h, s["params"]["output"], cell, delta,
            fmt(s["exact_token_accuracy"]["mean"], 4)))
    L.append("")

    kt, kr = S["kron_tied"], S["kron"]
    if kt["bits_per_byte"]["mean"] is not None:
        gap = kt["bits_per_byte"]["mean"] - kr["bits_per_byte"]["mean"]
        spread = max(kt["bits_per_byte"]["std"], kr["bits_per_byte"]["std"])
        saved = kr["params"]["output"] - kt["params"]["output"]
        L.append("**Tying costs {} bits/byte**, against a seed-to-seed spread of {}. "
                 "It removes {:,} output parameters — the entire {:,}-weight output "
                 "matrix, leaving a {:,}-element bias vector.".format(
                     fmt(abs(gap)), fmt(spread), saved,
                     kr["params"]["output"], kt["params"]["output"]))
        L.append("")

    # Decompose the gap to the baseline. This is the number that matters most
    # and it does not favour the method, so it is stated first and plainly.
    base = S["vocab"]["bits_per_byte"]["mean"]
    for h in ("kron", "kron_tied"):
        rows = [x["final"].get("bits_per_byte_vocab_restricted")
                for x in ev["runs"] if x["head"] == h]
        rows = [x for x in rows if x is not None]
        if not rows:
            continue
        restricted = sum(rows) / len(rows)
        free = S[h]["bits_per_byte"]["mean"]
        tax = free - restricted
        model_gap = restricted - base
        L.append("**Where `{}`'s {} bit/byte gap to the baseline comes from.** "
                 "Renormalising its distribution over just the {:,} real tokens gives "
                 "{} bits/byte. So {} of the gap is the price of ranging over every "
                 "byte string rather than a fixed list — the capability being bought — "
                 "and the remaining **{}** is not. That larger part is the head "
                 "predicting all 32 byte positions independently from one hidden state: "
                 "knowing a token starts with `th` tells it nothing about position 2.".format(
                     h, fmt(free - base), ev["tokenizer_and_tables"]["vocab_size"],
                     fmt(restricted), fmt(tax), fmt(model_gap)))
        L.append("")
        break

    return "\n".join(L).rstrip()


def heldout_block(ev) -> str:
    S, H = ev["summary"], ev["heldout"]
    L = []
    L.append("| arm | bits/byte, seen | bits/byte, held out | penalty |")
    L.append("|---|---:|---:|---:|")
    for h in ["vocab", "kron", "kron_tied"]:
        s = S[h]
        seen = s["bits_per_byte_seen"]["mean"]
        held = s["bits_per_byte_heldout"]["mean"]
        pen = None if (seen is None or held is None) else held - seen
        L.append("| `{}` | {} | {} | {} |".format(
            h, fmt(seen), fmt(held), "—" if pen is None else "+" + fmt(pen)))
    L.append("")
    L.append("{} token ids withheld, covering {:,} training positions ({:.2%}) and "
             "scored on {:,} validation positions. Examples: {}.".format(
                 H["n_tokens_withheld"], H["train_positions_masked"],
                 H["train_coverage"], H["val_positions_scored"],
                 ", ".join("`{}`".format(e) for e in H.get("examples", [])[:10])))
    return "\n".join(L)


def demo_block(demo) -> str:
    t = demo["tally"]
    L = ["| arm | exact | mean edit distance | first byte correct |",
         "|---|---:|---:|---:|"]
    for h in ("vocab", "kron", "kron_tied"):
        if h not in t:
            continue
        L.append("| `{}` | {}/{} | {} | {:.1%} |".format(
            h, t[h]["correct"], t[h]["total"],
            fmt(t[h]["mean_edit_distance"]), t[h]["first_byte_accuracy"]))
    L.append("")
    v, k = t.get("vocab"), t.get("kron_tied")
    if v and k:
        L.append("Exact reproduction of a never-trained token is out of reach for every "
                 "arm at this scale — {} and {} successes in {:,} are not a capability "
                 "claim, and they are reported as the small numbers they are. The signal "
                 "is in the partial credit: the byte head gets the **first byte** right "
                 "{:.1%} of the time against the baseline's {:.1%}, and lands closer by "
                 "edit distance. It composes part of the spelling; it just rarely "
                 "finishes it.".format(
                     t["kron"]["correct"], k["correct"], k["total"],
                     k["first_byte_accuracy"], v["first_byte_accuracy"]))
        L.append("")
        L.append("Note these two measures are defined for *every* arm, including the "
                 "baseline: whatever token `vocab` selects is ultimately a byte string "
                 "too, so it can be scored the same way. Nothing here is a metric the "
                 "byte head gets to grade itself on.")
    return "\n".join(L)


def splice(text: str, marker: str, body: str) -> str:
    open_tag, close_tag = "<!-- {} -->".format(marker), "<!-- /{} -->".format(marker)
    block = "{}\n\n{}\n\n{}".format(open_tag, body, close_tag)
    pattern = re.compile(re.escape(open_tag) + r".*?" + re.escape(close_tag), re.S)
    if pattern.search(text):
        return pattern.sub(lambda _: block, text)
    if open_tag in text:
        return text.replace(open_tag, block)
    raise SystemExit("marker {} not found in README".format(open_tag))


def main() -> int:
    path = os.path.join(ROOT, "results", "evidence.json")
    if not os.path.exists(path):
        print("results/evidence.json missing; run `python run.py` first")
        return 1
    with open(path) as fh:
        ev = json.load(fh)
    if ev.get("smoke"):
        print("refusing to write README tables from a --smoke run")
        return 1

    readme = os.path.join(ROOT, "README.md")
    with open(readme, encoding="utf-8") as fh:
        text = fh.read()
    text = splice(text, "RESULTS", results_block(ev))
    text = splice(text, "HELDOUT", heldout_block(ev))
    demo_path = os.path.join(ROOT, "results", "demo.json")
    if os.path.exists(demo_path):
        with open(demo_path) as fh:
            text = splice(text, "DEMO", demo_block(json.load(fh)))
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("README results tables updated from results/evidence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
