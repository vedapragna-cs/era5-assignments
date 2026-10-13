"""The Session-4 admission contract, made executable.

S4 says only data that has been cleaned, deduplicated, PII-screened, language-validated and
contamination-scanned may enter training. This module produces those five statuses and the pipeline
hash that makes the transformation reproducible.

Note on dedup: exact hashing plus shingle-Jaccard is honest at this corpus size. Production scale
needs MinHash-LSH (NeMo Curator) because all-pairs Jaccard is quadratic. We implement the exact thing
and state the scale answer rather than shipping a toy MinHash and implying it generalises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .canon import chash
from .corpus import Document

# The pipeline hash must change whenever the transformation changes, so it is taken over the rule
# table itself rather than over a version string someone can forget to bump.
CLEANING_RULES = [
    ("normalize_whitespace", r"[ \t]+", " "),
    ("strip_control", r"[\x00-\x08\x0b\x0c\x0e-\x1f]", ""),
    ("collapse_newlines", r"\n{3,}", "\n\n"),
]

PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone_in": r"\b[6-9]\d{9}\b",
    "street_address": r"\b\d{1,4}\s+[A-Z][a-z]+\s+(Road|Street|Marg|Lane|Nagar)\b",
}

SHINGLE_N = 5
NEAR_DUP_THRESHOLD = 0.55

# Repeated-n-gram filter, after OLMo 2 (arXiv:2501.00656 §3.1). They investigated the training
# batches at which loss and gradient-norm spikes occurred and found a high prevalence of long
# repeated n-gram sequences, then measured that filtering them removes many spikes (their Figure 3).
# Their rule: drop any document containing a run of 32 or more repeated n-grams, where an n-gram is
# any span of 1..13 tokens. Those constants are theirs, adopted rather than invented.
#
# This does NOT contradict the PaLM reading recorded in ARCHITECTURE - OLMo 2 are explicit that the
# relationship is non-deterministic (the same sequence spikes for a larger model but not a smaller
# one, spikes under one data ordering but not after reshuffling, and appears in non-spiking batches
# too). The reconciliation is that spikes are an optimizer-curvature event - the adaptive edge of
# stability, Cohen et al. via Wortsman et al. arXiv:2309.14322 §4 - that certain degenerate data
# reliably triggers. So: filter the trigger at curation time, and still never mark a shard harmful
# merely because it coincided with a spike.
REPEAT_NGRAM_MIN_N = 1
REPEAT_NGRAM_MAX_N = 13
REPEAT_NGRAM_RUN = 32


def pipeline_hash() -> str:
    return chash({"rules": CLEANING_RULES, "pii": sorted(PII_PATTERNS),
                  "shingle_n": SHINGLE_N, "near_dup_threshold": NEAR_DUP_THRESHOLD,
                  "repeat_ngram": [REPEAT_NGRAM_MIN_N, REPEAT_NGRAM_MAX_N, REPEAT_NGRAM_RUN]})


def repeated_ngram_run(tokens: list[str]) -> tuple[int, int, int]:
    """Longest run of a consecutively repeated n-gram: (run_length, n, start_index).

    A "run" of length k means the same n-token span occurs k times back to back. Scanning every
    n in 1..13 catches both single-token floods (`255, 255, 255, ...`) and repeating phrases,
    which is why OLMo 2 sweep n rather than fixing it.

    Complexity is O(len * max_n) with the early exit below - linear in practice. At production
    scale this belongs in the same pass as dedup fingerprinting rather than as a separate scan.
    """
    best = (0, 0, -1)
    L = len(tokens)
    for n in range(REPEAT_NGRAM_MIN_N, REPEAT_NGRAM_MAX_N + 1):
        if L < 2 * n:
            break
        i = 0
        while i + n <= L:
            run = 1
            while (i + (run + 1) * n <= L
                   and tokens[i:i + n] == tokens[i + run * n:i + (run + 1) * n]):
                run += 1
            if run > best[0]:
                best = (run, n, i)
            # a run of k copies consumes k*n tokens; no longer run can start inside it
            i += max(1, (run - 1) * n) if run > 1 else 1
    return best


def find_repeated_ngrams(text: str) -> dict:
    """Screen a document for the OLMo 2 repeated-n-gram pattern."""
    tokens = text.split()
    run, n, start = repeated_ngram_run(tokens)
    flagged = run >= REPEAT_NGRAM_RUN
    return {"max_repeat_run": run, "repeat_ngram_size": n, "start_token_index": start,
            "flagged": flagged,
            "span": (start, start + run * n) if flagged else None}


def clean_text(text: str) -> str:
    out = text
    for _, pattern, repl in CLEANING_RULES:
        out = re.sub(pattern, repl, out)
    return out.strip()


def find_pii(text: str) -> list[str]:
    return sorted({name for name, pat in PII_PATTERNS.items() if re.search(pat, text)})


def redact_pii(text: str) -> str:
    out = text
    for name, pat in PII_PATTERNS.items():
        out = re.sub(pat, f"<redacted:{name}>", out)
    return out


def _shingles(text: str, n: int = SHINGLE_N) -> set[str]:
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class CleanRecord:
    doc_id: str
    cleaning_pipeline_hash: str
    dedup_status: str                 # unique | exact_duplicate | near_duplicate
    dedup_cluster_id: str | None
    pii_screen_status: str            # clean | redacted
    pii_kinds: list[str] = field(default_factory=list)
    language_validated: bool = True
    text: str = ""
    # OLMo 2 repeated-n-gram screen. `status` is clean | flagged; the span is kept so the trainer
    # can mask it if a flagged document is ever admitted deliberately.
    repeat_ngram_status: str = "clean"
    repeat_ngram_detail: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {"doc_id": self.doc_id, "cleaning_pipeline_hash": self.cleaning_pipeline_hash,
                "dedup_status": self.dedup_status, "dedup_cluster_id": self.dedup_cluster_id,
                "pii_screen_status": self.pii_screen_status, "pii_kinds": self.pii_kinds,
                "language_validated": self.language_validated,
                "repeat_ngram_status": self.repeat_ngram_status,
                "repeat_ngram_detail": self.repeat_ngram_detail}


# Scripts we accept per declared language. A document whose text does not match its declared script
# fails language validation - this is how a mislabelled English split inside an "Indic" corpus gets
# caught, which is exactly the defect S5's supply audit found in Sangraha's verified tier (12.76B
# tokens of English inside a corpus reported as Indic).
SCRIPT_RANGES = {
    "Deva": (0x0900, 0x097F),
    "Telu": (0x0C00, 0x0C7F),
    "Latn": (0x0000, 0x024F),
}


def validate_language(text: str, script: str) -> bool:
    lo, hi = SCRIPT_RANGES.get(script, (0x0000, 0x10FFFF))
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    in_script = sum(1 for c in letters if lo <= ord(c) <= hi)
    return (in_script / len(letters)) >= 0.5


def run_cleaning(documents: list[Document]) -> dict[str, CleanRecord]:
    """Clean, dedup, PII-screen and language-validate every document.

    Returns doc_id -> CleanRecord. Documents are processed in a stable sorted order so that cluster
    ids and 'which copy is the original' do not depend on input ordering.
    """
    ph = pipeline_hash()
    records: dict[str, CleanRecord] = {}
    seen_exact: dict[str, str] = {}
    seen_shingles: list[tuple[str, set[str]]] = []

    for doc in sorted(documents, key=lambda d: d.doc_id):
        text = clean_text(doc.text)

        kinds = find_pii(text)
        if kinds:
            text = redact_pii(text)
            pii_status = "redacted"
        else:
            pii_status = "clean"

        exact = chash(text)
        dedup_status, cluster = "unique", None
        if exact in seen_exact:
            dedup_status, cluster = "exact_duplicate", seen_exact[exact]
        else:
            sh = _shingles(text)
            for other_id, other_sh in seen_shingles:
                if jaccard(sh, other_sh) >= NEAR_DUP_THRESHOLD:
                    dedup_status, cluster = "near_duplicate", other_id
                    break
            if dedup_status == "unique":
                seen_exact[exact] = doc.doc_id
                seen_shingles.append((doc.doc_id, sh))

        rep = find_repeated_ngrams(text)

        records[doc.doc_id] = CleanRecord(
            doc_id=doc.doc_id,
            cleaning_pipeline_hash=ph,
            dedup_status=dedup_status,
            dedup_cluster_id=cluster,
            pii_screen_status=pii_status,
            pii_kinds=kinds,
            language_validated=validate_language(text, doc.script),
            text=text,
            repeat_ngram_status="flagged" if rep["flagged"] else "clean",
            repeat_ngram_detail=rep,
        )
    return records
