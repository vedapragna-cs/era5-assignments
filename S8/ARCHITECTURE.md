# S8 architecture

## Contract

`data/*.json` is the **single source of truth**. Every HTML page is generated from it and nothing is
hand-edited in the output. Editing `index.html` directly is a mistake — `build.py` overwrites it.

```
data/mechanisms.json   34 cards, the timeline
data/findings.json     Question 2: seven findings + the uncovered-mechanism table
data/corrections.json  six entries: corrections, one limitation, one negative control
data/arxiv-response.xml    raw API response, committed so the check is reproducible offline
data/date-verification.json  last diff of stored dates against the API
        |
        v
build.py       validates, then writes index.html findings.html corrections.html
verify_dates.py    re-fetches every arXiv v1 date and diffs it
analyse.py     recomputes every number quoted on the findings page
manifest.py    writes MANIFEST.md with SHA-256 per file
```

## Invariants

1. **Every card states a cost.** `gives_up` may not be empty. `build.py` exits non-zero and writes
   no HTML if it is. This is the assignment's own rule made mechanical.
2. **Every card has a primary source URL and a status** of `verified` (matches the arXiv API) or
   `bounded` (no paper exists; the range and the reason are stated on the page).
3. **`verified` implies an arXiv id.** A card cannot claim verification without something to check.
4. **All 17 required list items are covered.** `build.py` fails if any is missing.
5. **Chronological order is the only default.** Family is a filter, never the top-level structure;
   the filter does not reorder. Sorting is by `date_v1`, then `same_day_order`, then `id`.
6. **Pages are readable with JavaScript disabled.** All content is in the HTML. JS only reveals the
   family filter, which is hidden by default (`#ctl{display:none}`) so it never appears inert.

## Decisions

- **arXiv v1 as sort key, shipped date as a second marker** (decided with the user before building).
  v1 is the day the idea became public and is mechanically checkable; publication venue dates are
  not comparable across a field that mostly does not use venues. Carrying both is what produced the
  D5 lag result — a single-date timeline could not have shown it.
- **Two cards may share a date.** Standard attention and sinusoidal encoding are the same paper.
  `same_day_order` puts the mechanism before its positional patch, because the assignment requires
  standard attention to be understood first.
- **Standard attention is explained above the timeline, not only inside it.** Learned absolute
  positions genuinely predate the Transformer by 35 days, so a purely chronological page would open
  on a positional scheme. The "Start here" section resolves the conflict between the chronology
  requirement and the "do not assume the reader already understands" requirement without distorting
  either.
- **Q2 is a page, not a caption.** The submission form has one link field and no text area, so an
  answer that is not reachable from a submitted link cannot be graded.
- **No build step at deploy time.** The HTML is committed. Any static host serves the directory
  as-is, and it also opens from the file system.
