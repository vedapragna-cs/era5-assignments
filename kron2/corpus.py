"""Corpus construction.

The claim under test is about *spelling*: whether a model can predict a token it
has never been given, byte by byte. That claim cannot be evaluated on synthetic
text, because synthetic text has no real morphology, no rare proper nouns, and no
genuine misspellings -- exactly the phenomena the method is supposed to handle.
So the corpus is real, and deliberately mixed:

* **Prose** (public-domain books). Inflection, compounding, proper nouns, and the
  long tail of ordinary English.
* **Code** (Python standard library). Identifiers like ``__init__`` and
  ``get_event_loop`` are byte-structured in a way prose never is, and they are
  the case where a fixed vocabulary fragments worst.
* **Multi-byte UTF-8**. Accented Latin and non-Latin scripts, where one character
  costs two to four bytes. The byte window is spent much faster here, and the V1
  paper flags this as its weakest case.

The result is written once to a single file and committed, so the repository
clones and runs with no network. The builder is retained for provenance: it
records which sources were used and their hashes, so the corpus can be rebuilt
and checked rather than taken on trust.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import sys
import urllib.request
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")
CORPUS_GZ_PATH = os.path.join(DATA_DIR, "corpus.txt.gz")
MANIFEST_PATH = os.path.join(DATA_DIR, "corpus_manifest.json")

# Public-domain texts. Chosen for script and register variety rather than
# literary merit. A third of the list is non-English on purpose: accented Latin
# and Cyrillic cost two bytes per character, and the V1 paper flags multi-byte
# UTF-8 as the case where a fixed byte window is spent fastest.
GUTENBERG = [
    # English prose, spread across register: novel, gothic, children's,
    # dialogue-heavy detective fiction, whaling encyclopaedia, epic verse.
    (1342, "Pride and Prejudice", "en"),
    (84,   "Frankenstein", "en"),
    (11,   "Alice's Adventures in Wonderland", "en"),
    (1661, "The Adventures of Sherlock Holmes", "en"),
    (2701, "Moby Dick", "en"),
    (1400, "Great Expectations", "en"),
    (768,  "Wuthering Heights", "en"),
    (98,   "A Tale of Two Cities", "en"),
    (2600, "War and Peace", "en"),
    (100,  "The Complete Works of Shakespeare", "en"),
    (74,   "The Adventures of Tom Sawyer", "en"),
    (345,  "Dracula", "en"),
    (1232, "The Prince", "en"),
    (5200, "Metamorphosis", "en"),
    # Non-English: heavy multi-byte UTF-8, and morphology English does not have.
    (2000, "Don Quijote", "es"),
    (2650, "Du cote de chez Swann", "fr"),
    (17489, "Les Miserables (fr)", "fr"),
    (13951, "Divina Commedia", "it"),
    (21000, "Effi Briest", "de"),
    (28054, "The Brothers Karamazov", "en"),
]

CACHE_DIR = os.path.join(DATA_DIR, "_cache")


def _normalise_newlines(text: str) -> str:
    """Collapse CRLF and lone CR to LF.

    Gutenberg serves CRLF. A freshly downloaded book is decoded straight from
    bytes and keeps them; a book re-read from the on-disk cache goes through text
    mode, which silently converts them. So the same build produced different
    bytes depending on whether the cache was warm -- and worse, the manifest hash
    was taken over the written file while `load` returns the text-mode reading of
    it, so the recorded hash described a string the loader never returns.
    Measured: 40,634 CRLF sequences, a 40,634-byte discrepancy.

    Normalising once, here, makes the download path and the cache path agree and
    makes the hash describe what `load` actually returns.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the licence header and footer.

    They are identical across every book, so leaving them in would hand the model
    several kilobytes of exactly repeated text and inflate every arm's score
    equally but meaninglessly.
    """
    start = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.S)
    end = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.S)
    if start:
        text = text[start.end():]
    if end:
        # `end` was located on the original string; re-find on the trimmed one.
        end2 = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.S)
        if end2:
            text = text[:end2.start()]
    return text.strip()


def fetch_gutenberg(book_id: int, timeout: int = 60) -> Optional[str]:
    """Download one book, cached on disk. Returns None if unavailable."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = os.path.join(CACHE_DIR, "pg{}.txt".format(book_id))
    if os.path.exists(cached):
        with open(cached, encoding="utf-8", errors="replace") as fh:
            return fh.read()

    for url in ("https://www.gutenberg.org/cache/epub/{}/pg{}.txt".format(book_id, book_id),
                "https://www.gutenberg.org/files/{}/{}-0.txt".format(book_id, book_id)):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                raw = _normalise_newlines(resp.read().decode("utf-8", errors="replace"))
            with open(cached, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(raw)
            return raw
        except Exception:
            continue
    return None


def collect_code(max_bytes: int, seed: int = 0) -> Tuple[str, int]:
    """Sample Python source from the local standard library.

    Uses whatever interpreter is running, so this is reproducible for a given
    environment but not across Python versions. The manifest records the version
    and the resulting hash, so a mismatch is visible rather than silent.
    """
    import sysconfig

    lib = sysconfig.get_paths()["stdlib"]
    paths: List[str] = []
    for dirpath, dirnames, filenames in os.walk(lib):
        dirnames[:] = [d for d in dirnames
                       if d not in ("test", "tests", "site-packages", "__pycache__")]
        for fn in filenames:
            if fn.endswith(".py"):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    random.Random(seed).shuffle(paths)

    chunks: List[str] = []
    total = 0
    used = 0
    for p in paths:
        if total >= max_bytes:
            break
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
        except OSError:
            continue
        if len(src) < 500:
            continue
        chunks.append(src)
        total += len(src.encode("utf-8"))
        used += 1
    return "\n\n".join(chunks), used


def build(prose_bytes: int = 6_000_000,
          code_bytes: int = 3_000_000,
          seed: int = 0,
          force: bool = False) -> Dict:
    """Assemble the corpus and write it, with a manifest describing the sources."""
    import json

    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CORPUS_PATH) and not force:
        with open(MANIFEST_PATH) as fh:
            return json.load(fh)

    sources = []
    prose_parts: List[str] = []

    # Two separate budgets rather than one shared pool. With a single budget the
    # English titles are simply listed first and exhaust it, and the non-English
    # books -- the entire reason the multi-byte case is testable -- never get
    # fetched. Measured: that dropped the multi-byte fraction from 3.08% to 1.88%.
    non_en_bytes = int(prose_bytes * 0.35)
    en_bytes = prose_bytes - non_en_bytes
    budgets = {"en": en_bytes, "other": non_en_bytes}
    spent = {"en": 0, "other": 0}

    for book_id, title, lang in GUTENBERG:
        bucket = "en" if lang == "en" else "other"
        if spent[bucket] >= budgets[bucket]:
            sources.append({"kind": "prose", "id": book_id, "title": title,
                            "lang": lang, "status": "skipped: budget spent"})
            continue
        raw = fetch_gutenberg(book_id)
        if raw is None:
            sources.append({"kind": "prose", "id": book_id, "title": title,
                            "lang": lang, "status": "unavailable"})
            continue
        text = _strip_gutenberg_boilerplate(raw)
        prose_parts.append(text)
        spent[bucket] += len(text.encode("utf-8"))
        sources.append({"kind": "prose", "id": book_id, "title": title, "lang": lang,
                        "status": "ok", "bytes": len(text.encode("utf-8"))})

    code_text, n_files = collect_code(code_bytes, seed=seed)
    sources.append({"kind": "code", "source": "python stdlib",
                    "python": sys.version.split()[0], "files": n_files,
                    "bytes": len(code_text.encode("utf-8"))})

    corpus = _normalise_newlines("\n\n".join(prose_parts) + "\n\n" + code_text)
    with open(CORPUS_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(corpus)

    raw_bytes = corpus.encode("utf-8")
    manifest = {
        "path": os.path.relpath(CORPUS_PATH, ROOT),
        "bytes": len(raw_bytes),
        "chars": len(corpus),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "multibyte_fraction": round(1 - sum(1 for b in raw_bytes if b < 0x80) / len(raw_bytes), 6),
        "prose_bytes_by_bucket": spent,
        "sources": sources,
        "seed": seed,
    }
    with open(MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def load() -> str:
    """Read the committed corpus.

    Prefers the gzipped copy, which is what the repository ships: 8.7 MB instead
    of 26 MB for byte-identical content. The plain file is used if present so a
    local rebuild does not have to be compressed first.
    """
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, encoding="utf-8") as fh:
            return fh.read()
    if os.path.exists(CORPUS_GZ_PATH):
        import gzip
        with gzip.open(CORPUS_GZ_PATH, "rt", encoding="utf-8") as fh:
            return fh.read()
    raise FileNotFoundError(
        "corpus missing; run `python -m kron2.corpus` once to build it")


def verify() -> Dict:
    """Check the corpus on disk against the hash the manifest recorded."""
    import json

    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)
    raw = load().encode("utf-8")
    got = hashlib.sha256(raw).hexdigest()
    return {
        "bytes_expected": manifest["bytes"], "bytes_got": len(raw),
        "sha256_expected": manifest["sha256"], "sha256_got": got,
        "match": got == manifest["sha256"] and len(raw) == manifest["bytes"],
    }


if __name__ == "__main__":
    import json
    m = build(force="--force" in sys.argv)
    print(json.dumps({k: v for k, v in m.items() if k != "sources"}, indent=2))
    for s in m["sources"]:
        print("  ", s)
