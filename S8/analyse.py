#!/usr/bin/env python3
"""Recompute every number quoted on the findings page, straight from mechanisms.json.

Run this to check Q2's arithmetic. It prints the lag table, the idea-to-use gaps and the
per-family era spread. Nothing here is hand-entered.
"""
import json, datetime, collections, os

HERE = os.path.dirname(os.path.abspath(__file__))
C = {x["id"]: x for x in json.load(open(os.path.join(HERE, "data", "mechanisms.json")))}
d = datetime.date.fromisoformat

def days(a, b):
    return (d(b) - d(a)).days

print("=== Q2-1  publish -> ship lag, in publication order ===")
rows = [(x["date_v1"], x["id"], x["shipped"]["date"], x["shipped"]["what"])
        for x in C.values() if x.get("shipped")]
for v1, mid, sh, what in sorted(rows):
    print(f"  {mid:<16} v1 {v1}  shipped {sh}  lag {days(v1, sh):>5} d   {what[:44]}")

print("\n=== Q2-2  idea -> use gaps ===")
for a, b, label in [("mqa", "gqa", "MQA -> GQA"),
                    ("sliding-window", "sliding-window", "Longformer -> Mistral (shipped)"),
                    ("sliding-window", "attention-sinks", "sliding window -> sinks explains it"),
                    ("delta-rule", "deltanet-parallel", "delta rule -> parallel DeltaNet")]:
    end = C[b]["shipped"]["date"] if a == b else C[b]["date_v1"]
    print(f"  {label:<38} {days(C[a]['date_v1'], end):>5} d")

print("\n=== Q2-7 / Q2-5 / Q2-6  adjacencies ===")
for a, b, label in [("rope", "alibi", "RoPE -> ALiBi (rivals)"),
                    ("ntk-aware", "position-interpolation", "NTK-aware -> PI"),
                    ("position-interpolation", "yarn", "PI -> YaRN"),
                    ("learned-abs", "standard", "learned positions -> Transformer")]:
    print(f"  {label:<38} {days(C[a]['date_v1'], C[b]['date_v1']):>5} d")

print("\n=== cards per year ===")
yr = collections.Counter(x["date_v1"][:4] for x in C.values())
for y in sorted(yr):
    print(f"  {y}  {'#' * yr[y]} {yr[y]}")

print("\n=== family across eras (Q2-4: note the sparsity hole) ===")
fam = collections.defaultdict(list)
for x in sorted(C.values(), key=lambda z: z["date_v1"]):
    fam[x["family"]].append(x["date_v1"][:4])
for f in sorted(fam):
    print(f"  {f:<11} {' '.join(fam[f])}")

print(f"\ncards: {len(C)}   with a shipped date: {len(rows)}   "
      f"required-list items covered: {len({x['list_item'] for x in C.values() if x['list_item']})}/17")
