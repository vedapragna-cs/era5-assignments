"""The document: what enters the pipeline, and everything we must know about it.

A raw string is not a training document. The session's list of what a clean document must
carry before it can become a training sequence -- provenance, quality metadata, license,
cleaning lineage, dedup and contamination status -- is encoded here as required fields,
not as optional annotations. A document missing any of them cannot be sharded; see
`Document.validate`.

The `segments` field is what makes structure-preserving packing and loss masking possible.
A pretraining document is one segment, fully trainable. An SFT document is prompt (not
trainable) + response (trainable). An agentic trace is a chain of assistant turns
(trainable, *including the failed tool call* -- recovery is the behaviour being taught)
and tool observations (not trainable, because applying loss to tool output teaches the
model to invent tool results instead of calling the tool).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .determinism import content_hash

# ---------------------------------------------------------------------------
# splits and the evaluation firewall
# ---------------------------------------------------------------------------

SPLIT_TRAIN = "train"
SPLIT_VAL = "validation"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# Only this split may ever reach an optimizer step. Enforced twice, independently:
# once when a shard is built (firewall.py) and once when the loader hands a batch to the
# trainer (stream.py). The session is explicit that one check is not enough, "because who
# knows maybe a mistake in copying or something may still happen".
TRAINABLE_SPLITS = (SPLIT_TRAIN,)

# ---------------------------------------------------------------------------
# segment roles
# ---------------------------------------------------------------------------

ROLE_TEXT = "text"              # plain pretraining prose
ROLE_PROMPT = "prompt"          # SFT question / instruction        -> masked
ROLE_RESPONSE = "response"      # SFT answer                        -> supervised
ROLE_SYSTEM = "system"          # policy / system prompt            -> masked
ROLE_USER = "user"              # user turn in a trace              -> masked
ROLE_ASSISTANT = "assistant"    # model turn, incl. failed calls    -> supervised
ROLE_TOOL_CALL = "tool_call"    # the call the model emits          -> supervised
ROLE_TOOL_RESULT = "tool_result"  # what came back from the tool    -> masked
ROLE_THINKING = "thinking"      # reasoning trace                   -> supervised

# The session's colour code: green = supervised, grey = masked context.
SUPERVISED_ROLES = frozenset(
    {ROLE_TEXT, ROLE_RESPONSE, ROLE_ASSISTANT, ROLE_TOOL_CALL, ROLE_THINKING}
)
MASKED_ROLES = frozenset({ROLE_PROMPT, ROLE_SYSTEM, ROLE_USER, ROLE_TOOL_RESULT})
ALL_ROLES = SUPERVISED_ROLES | MASKED_ROLES


@dataclass
class Segment:
    """A contiguous span of a document with a single role."""

    role: str
    text: str

    @property
    def trainable(self) -> bool:
        return self.role in SUPERVISED_ROLES

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "text": self.text}


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


@dataclass
class Provenance:
    """Where this came from, and under what terms we may train on it.

    `collector` answers the session's question "which group amongst you calculated that
    particular document" -- in a real run this is the team or job that ingested it.
    """

    dataset: str
    source_uri: str
    license: str
    license_tier: str        # A = permissive+verified, B = permissive, C = restricted
    collector: str
    retrieved_epoch: int     # integer epoch, NOT wall-clock -- must not float between runs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_uri": self.source_uri,
            "license": self.license,
            "license_tier": self.license_tier,
            "collector": self.collector,
            "retrieved_epoch": self.retrieved_epoch,
        }


TRAINABLE_LICENSE_TIERS = ("A", "B")


@dataclass
class Quality:
    """Quality metadata. `score` is the cleaning pipeline's own estimate in [0, 1]."""

    score: float
    dedup_status: str          # unique | near_dup_kept | exact_dup_dropped
    pii_scrubbed: bool
    cleaning_pipeline: str
    cleaning_code_hash: str
    language_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "dedup_status": self.dedup_status,
            "pii_scrubbed": self.pii_scrubbed,
            "cleaning_pipeline": self.cleaning_pipeline,
            "cleaning_code_hash": self.cleaning_code_hash,
            "language_confidence": self.language_confidence,
        }


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------


@dataclass
class Document:
    doc_id: str
    lane: str
    segments: List[Segment]
    language: str
    script: str
    difficulty_band: str                  # B0..B5, source-tagged at build time
    split: str
    provenance: Provenance
    quality: Quality
    reasoning_band: Optional[str] = None  # L0..L3, reasoning lane only
    # Set when the document must occupy a sequence alone (agentic traces, SFT
    # conversations). Drives structure-preserving packing.
    atomic: bool = False
    # Set when the document must never be split across sequences (source files). If it
    # does not fit the current window it is deferred to a stage with a longer one rather
    # than chopped -- "you have a 16,000 long code file, don't cut it in between".
    indivisible: bool = False
    tags: Tuple[str, ...] = ()

    # ---- derived -------------------------------------------------------

    @property
    def text(self) -> str:
        return "".join(seg.text for seg in self.segments)

    @property
    def char_len(self) -> int:
        return len(self.text)

    def content_fingerprint(self) -> str:
        """Hash of the content only -- not the metadata.

        Used for exact-duplicate detection: two documents with different provenance but
        identical text are the same training signal seen twice.
        """
        return content_hash([[s.role, s.text] for s in self.segments])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "lane": self.lane,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "script": self.script,
            "difficulty_band": self.difficulty_band,
            "reasoning_band": self.reasoning_band,
            "split": self.split,
            "provenance": self.provenance.to_dict(),
            "quality": self.quality.to_dict(),
            "atomic": self.atomic,
            "indivisible": self.indivisible,
            "tags": list(self.tags),
        }

    # ---- validation ----------------------------------------------------

    def validate(self) -> List[str]:
        """Everything that would make this document unsafe to train on."""
        from .plan import DIFFICULTY_BANDS, LANES, REASONING_LENGTH_BANDS

        problems: List[str] = []
        if self.lane not in LANES:
            problems.append("unknown lane {!r}".format(self.lane))
        if self.split not in SPLITS:
            problems.append("unknown split {!r}".format(self.split))
        if self.difficulty_band not in DIFFICULTY_BANDS:
            problems.append("unknown difficulty band {!r}".format(self.difficulty_band))
        if self.reasoning_band is not None and self.reasoning_band not in REASONING_LENGTH_BANDS:
            problems.append("unknown reasoning band {!r}".format(self.reasoning_band))
        if not self.segments:
            problems.append("document has no segments")
        for seg in self.segments:
            if seg.role not in ALL_ROLES:
                problems.append("unknown segment role {!r}".format(seg.role))
            if not seg.text:
                problems.append("empty segment with role {!r}".format(seg.role))
        if not any(seg.trainable for seg in self.segments):
            problems.append("no trainable segment -- document would contribute zero loss")
        if not self.quality.pii_scrubbed:
            problems.append("PII not scrubbed")
        if self.provenance.license_tier not in TRAINABLE_LICENSE_TIERS:
            problems.append(
                "license tier {!r} is not trainable".format(self.provenance.license_tier)
            )
        if self.quality.dedup_status == "exact_dup_dropped":
            problems.append("document was marked as a dropped duplicate")
        return problems


def documents_fingerprint(docs: Sequence[Document]) -> str:
    """Order-independent fingerprint of a document set."""
    return content_hash(sorted(d.content_fingerprint() for d in docs))
