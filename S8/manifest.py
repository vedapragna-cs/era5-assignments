#!/usr/bin/env python3
"""Write MANIFEST.md: every shipped file with its SHA-256, so the artifact list is visible
in the extracted text of the repo rather than only on disk."""
import hashlib, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = {"MANIFEST.md", "__pycache__", ".DS_Store"}
GROUPS = [("Pages (committed, no build step needed)", lambda p: p.endswith(".html")),
          ("Data — single source of truth", lambda p: p.startswith("data/")),
          ("Scripts — the checks", lambda p: p.endswith(".py")),
          ("Documents", lambda p: p.endswith(".md"))]

files = []
for root, dirs, names in os.walk(HERE):
    dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")]
    for n in sorted(names):
        if n in SKIP or n.startswith("."):
            continue
        full = os.path.join(root, n)
        rel = os.path.relpath(full, HERE)
        h = hashlib.sha256(open(full, "rb").read()).hexdigest()
        files.append((rel, h, os.path.getsize(full)))

out = ["# Manifest\n",
       f"Every file in this submission with its SHA-256. Regenerate with `python3 manifest.py` "
       f"and verify any single file with `sha256sum <path>`.\n",
       f"{len(files)} files, {sum(f[2] for f in files):,} bytes total. "
       f"Generated {datetime.date.today().isoformat()}.\n",
       "\n> **One file is deliberately volatile.** `data/date-verification.json` carries a\n"
       "> `checked_at` timestamp and is rewritten every time `verify_dates.py` runs, so its hash\n"
       "> below is the value as committed. Running the check will change it; that is the check\n"
       "> working, not the artifact drifting. Every other hash here is stable.\n"]
claimed = set()
for title, pred in GROUPS:
    rows = [f for f in files if pred(f[0]) and f[0] not in claimed]
    if not rows:
        continue
    claimed |= {f[0] for f in rows}
    out.append(f"\n## {title}\n")
    out.append("| file | bytes | sha256 |")
    out.append("|---|---:|---|")
    for rel, h, sz in rows:
        out.append(f"| `{rel}` | {sz:,} | `{h}` |")
rest = [f for f in files if f[0] not in claimed]
if rest:
    out.append("\n## Other\n")
    out.append("| file | bytes | sha256 |")
    out.append("|---|---:|---|")
    for rel, h, sz in rest:
        out.append(f"| `{rel}` | {sz:,} | `{h}` |")
open(os.path.join(HERE, "MANIFEST.md"), "w").write("\n".join(out) + "\n")
print(f"MANIFEST.md: {len(files)} files")
