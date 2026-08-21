#!/usr/bin/env python3
"""Fail the build if the rendered pages disagree with the data or with each other.

Three rounds of review found stale counts, contradictory numbers and a card that was
validated but never rendered. Those are all mechanically detectable, so they are checked
here rather than by re-reading. Exits non-zero on any finding.
"""
import json, os, re, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda n: os.path.join(HERE, n)
cards = json.load(open(P("data/mechanisms.json")))
findings = json.load(open(P("data/findings.json")))
corrections = json.load(open(P("data/corrections.json")))
pages = {os.path.basename(f): open(f, encoding="utf-8").read()
         for f in glob.glob(P("*.html"))}
docs = {os.path.basename(f): open(f, encoding="utf-8").read()
        for f in glob.glob(P("*.md"))}
text = {k: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ",
        re.sub(r"<(script|style)[^>]*>.*?</\1>", "", v, flags=re.S)))
        for k, v in pages.items()}

N = len(cards)
NADD = sum(1 for c in cards if c["added"])
NTL = sum(1 for c in cards if c.get("taught_not_listed"))
NNP = sum(1 for c in cards if not c.get("arxiv"))
errs = []

# 1. every validated card must actually render
ids = set(re.findall(r'<article class="card[^"]*"[^>]*id="([^"]+)"', pages["index.html"]))
missing = [c["id"] for c in cards if c["id"] not in ids]
if missing:
    errs.append(f"cards validated but not rendered: {missing}")

# 2. every in-page anchor must resolve
for name, html in pages.items():
    for href in set(re.findall(r'href="#([^"]+)"', html)):
        if f'id="{href}"' not in html:
            errs.append(f"{name}: dead anchor #{href}")
    for href in set(re.findall(r'href="([a-zA-Z0-9_.-]+\.(?:html|md))"', html)):
        if not os.path.exists(P(href)):
            errs.append(f"{name}: link to missing file {href}")

# 3. counts stated in prose must match the data
def stated(pat, blob):
    return {int(m) for m in re.findall(pat, blob)}
# EXPERIMENT_LOG records historical state and must not be rewritten; skip its counts.
for name, t in [(k, v) for k, v in text.items()] + [(k, v) for k, v in docs.items()
                                                    if k != "EXPERIMENT_LOG.md"]:
    for m in re.finditer(r"(\d+)\s+mechanisms", t):
        if int(m.group(1)) not in (N, 17):
            errs.append(f"{name}: '{m.group(1)} mechanisms' but data has {N}")
    for m in re.finditer(r"(\d+)\s+cards", t):
        # "across 21 cards" counts the subset covering the 17 required items -- legitimate
        if t[max(0, m.start() - 8):m.start()].rstrip().endswith("across"):
            continue
        if int(m.group(1)) != N:
            errs.append(f"{name}: '{m.group(1)} cards' but data has {N}")
    for m in re.finditer(r"plus \*{0,2}(\d+)\*{0,2} we added", t):
        if int(m.group(1)) != NADD:
            errs.append(f"{name}: 'plus {m.group(1)} we added' but data has {NADD}")

# 4. no raw markdown leaking into rendered text
for name, t in text.items():
    for m in re.findall(r"\*\*[^*]{2,40}\*\*", t):
        errs.append(f"{name}: unrendered markdown bold {m!r}")
    if "&lt;" in t:
        errs.append(f"{name}: escaped markup visible in body text")

# 5. dates used in antecedents must match the referenced card's own date
by_date = {}
for c in cards:
    by_date.setdefault(c["name"].lower(), c["date_v1"])
for c in cards:
    for a in c.get("antecedents", []):
        real = by_date.get(a["name"].lower())
        if real and real != a["date"]:
            errs.append(f"{c['id']}: antecedent '{a['name']}' dated {a['date']}, card says {real}")

# 6. every shipped row must carry the fields the finding relies on
for c in cards:
    sh = c.get("shipped")
    if sh and "adoption_class" not in sh:
        errs.append(f"{c['id']}: shipped row has no adoption_class (silently dropped from Q2-1)")

# 7. no date in the future relative to the newest card
newest = max(c["date_v1"] for c in cards)
for c in cards:
    if c.get("shipped") and c["shipped"]["date"] > newest:
        errs.append(f"{c['id']}: shipped date {c['shipped']['date']} is after the newest v1 {newest}")

print(f"data: {N} cards, {NADD} added, {NTL} taught-not-listed, {NNP} without a paper, "
      f"{len(findings['findings'])} findings, {len(corrections)} corrections")
if errs:
    print(f"CONSISTENCY FAILED -- {len(errs)} issue(s):")
    for e in sorted(set(errs)):
        print("  x", e)
    sys.exit(1)
print("consistency passed")
