#!/usr/bin/env python3
"""Re-fetch every arXiv v1 date in mechanisms.json from the arXiv API and diff it.

This exists so the chronology is checkable rather than asserted. It makes exactly one
network call (a single batched id_list query) and exits non-zero if any stored date
disagrees with the API, or if any card that claims `status: verified` cannot be checked.

    python3 verify_dates.py            # check, write data/date-verification.json
    python3 verify_dates.py --offline  # re-check against the last saved API response
"""
import json, sys, re, os, urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "mechanisms.json")
RAW  = os.path.join(HERE, "data", "arxiv-response.xml")
OUT  = os.path.join(HERE, "data", "date-verification.json")
API  = "https://export.arxiv.org/api/query?id_list={}&max_results=200"
NS   = {"a": "http://www.w3.org/2005/Atom"}

def fetch(ids, offline):
    if offline:
        return open(RAW, "rb").read()
    req = urllib.request.Request(API.format(",".join(ids)),
                                 headers={"User-Agent": "ERA-V5-S8-date-check"})
    body = urllib.request.urlopen(req, timeout=60).read()
    open(RAW, "wb").write(body)
    return body

def main():
    offline = "--offline" in sys.argv
    cards = json.load(open(DATA))
    ids = sorted({c["arxiv"] for c in cards if c.get("arxiv")})
    root = ET.fromstring(fetch(ids, offline))

    api = {}
    for e in root.findall("a:entry", NS):
        m = re.search(r"abs/([\d.]+)v?\d*", e.find("a:id", NS).text)
        if m:
            api[m.group(1)] = e.find("a:published", NS).text[:10]

    rows, bad = [], 0
    for c in cards:
        aid, stored, status = c.get("arxiv"), c["date_v1"], c["status"]
        if not aid:
            rows.append({"id": c["id"], "arxiv": None, "stored": stored,
                         "api": None, "verdict": "NO-PAPER", "status": status})
            if status == "verified":
                print(f"FAIL {c['id']}: claims verified but has no arXiv id"); bad += 1
            continue
        got = api.get(aid)
        if got is None:
            verdict = "NOT-RETURNED"; bad += 1
        elif got == stored:
            verdict = "MATCH"
        else:
            verdict = "MISMATCH"; bad += 1
        rows.append({"id": c["id"], "arxiv": aid, "stored": stored,
                     "api": got, "verdict": verdict, "status": status})

    w = max(len(r["id"]) for r in rows)
    print(f"{'card'.ljust(w)}  {'arxiv':<11} {'stored':<11} {'api v1':<11} verdict")
    print("-" * (w + 48))
    for r in sorted(rows, key=lambda r: r["stored"]):
        print(f"{r['id'].ljust(w)}  {str(r['arxiv'] or '-'):<11} {r['stored']:<11} "
              f"{str(r['api'] or '-'):<11} {r['verdict']}")

    n_match = sum(1 for r in rows if r["verdict"] == "MATCH")
    n_nopaper = sum(1 for r in rows if r["verdict"] == "NO-PAPER")
    summary = {
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "https://export.arxiv.org/api/query (published = v1 submission)",
        "cards": len(rows), "arxiv_dates_matched": n_match,
        "no_paper_by_design": n_nopaper, "failures": bad, "rows": rows,
    }
    json.dump(summary, open(OUT, "w"), indent=1)
    print(f"\n{n_match} arXiv dates re-verified, {n_nopaper} card(s) have no paper by design, "
          f"{bad} failure(s).")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
