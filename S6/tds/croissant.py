"""A Croissant-compatible provenance view of the shard manifests.

Our manifests already carry the facts a data auditor needs - source, licence, tokenizer identity,
cleaning pipeline hash, dedup and PII status, contamination verdict, parent lineage. They carry them
under *our* names, which means every consumer needs our documentation to read them.

Croissant (MLCommons) is the standard for exactly this, and 1.1 (February 2026) added
machine-actionable provenance built on PROV-O plus structured usage policies. Emitting a Croissant
view costs one function and makes the same facts portable to any tool that speaks the standard.

The timing is not incidental: the EU AI Act's provenance obligations for high-risk systems begin
enforcement in **August 2026**. For a 3T-token training run whose corpus mixes licences and
languages, "we recorded the lineage" and "we can hand a regulator a machine-readable lineage" are
different states of readiness.

Scope, stated honestly
----------------------
This is a **compatible view, not a validated artifact**. MLCommons ships a Python validator and a
visual editor; neither runs here (no network, and validation is not something to claim from a
docstring). What this function guarantees is that the fields are mapped and the lineage edges are
present. Conformance is the validator's verdict, and running it is the named next step.

The mapping is also deliberately lossy in one direction: `content_hash` is SHA-256 over the shard's
**token bytes**, not over a file on disk, because in this system the tokenized shard is the artifact
and the text is upstream of it. That is recorded in the description rather than silently presented
as a file checksum.
"""

from __future__ import annotations

CONTEXT = {
    "@vocab": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "prov": "http://www.w3.org/ns/prov#",
    "sc": "https://schema.org/",
}


def _distribution(manifest: dict) -> dict:
    """One shard as a Croissant FileObject with its provenance edges."""
    return {
        "@type": "cr:FileObject",
        "@id": manifest["shard_id"],
        "name": manifest["shard_id"],
        "description": (
            f"Immutable tokenized shard for the {manifest['capability_lane']} lane, "
            f"{manifest['token_count']} tokens, split={manifest['split']}."),
        "encodingFormat": "application/x-tokenized-shard",
        "sha256": manifest["content_hash"],
        "cr:sha256Scope": "shard token bytes (little-endian uint32), not a file on disk - the "
                          "tokenized shard is the artifact in this system",
        "sc:license": manifest["license_tier"],
        "cr:containedIn": sorted(manifest.get("source_ids", [])),
        # PROV-O lineage: what produced this, and what it came from.
        "prov:wasGeneratedBy": {
            "@type": "prov:Activity",
            "prov:used": [
                {"@type": "prov:Entity", "@id": "tokenizer",
                 "cr:version": manifest["tokenizer_version"],
                 "sha256": manifest["tokenizer_hash"]},
                {"@type": "prov:Entity", "@id": "cleaning_pipeline",
                 "sha256": manifest["cleaning_pipeline_hash"]},
            ],
            "cr:qualityGates": {
                "dedup_status": manifest["dedup_status"],
                "pii_screen_status": manifest["pii_screen_status"],
                "eval_overlap_status": manifest["eval_overlap_status"],
                "repeat_ngram_status": manifest.get("repeat_ngram_status"),
                "language_validated": manifest["language_validated"],
            },
        },
        "prov:wasDerivedFrom": list(manifest.get("parent_shard_ids") or []),
        "cr:documentIds": list(manifest.get("document_ids", [])),
    }


def build(shard_manifests: list[dict], *, plan_id: str, run_id: str,
          tokenizer: dict, description: str = "") -> dict:
    """Emit the Croissant-compatible metadata document for a set of admitted shards."""
    licences = sorted({m["license_tier"] for m in shard_manifests})
    languages = sorted({l for m in shard_manifests for l in m.get("languages", [])})
    lanes = sorted({m["capability_lane"] for m in shard_manifests})
    return {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "cr:conformanceClaim": "UNVALIDATED - fields are mapped and lineage edges are present, but "
                               "this has not been run through the MLCommons validator (no network "
                               "in this environment). Running it is the next step, not a claim "
                               "made here.",
        "name": f"{plan_id}-training-corpus",
        "description": description or (
            f"Tokenized training corpus executed by run {run_id} against mixture plan {plan_id}. "
            f"{len(shard_manifests)} immutable shards across {len(lanes)} capability lanes."),
        "sc:license": licences,
        "cr:capabilityLanes": lanes,
        "inLanguage": languages,
        "cr:tokenizer": tokenizer,
        "distribution": [_distribution(m) for m in
                         sorted(shard_manifests, key=lambda m: m["shard_id"])],
        "recordSet": [{
            "@type": "cr:RecordSet",
            "@id": "token_spans",
            "description": "Each shard records the byte range of every source document inside its "
                           "flat token array, which is what makes per-document consumption "
                           "auditable after the fact.",
            "field": [
                {"@type": "cr:Field", "@id": "doc_id", "dataType": "sc:Text"},
                {"@type": "cr:Field", "@id": "start", "dataType": "sc:Integer"},
                {"@type": "cr:Field", "@id": "end", "dataType": "sc:Integer"},
                {"@type": "cr:Field", "@id": "lane", "dataType": "sc:Text"},
            ],
        }],
        "cr:regulatoryNote": "EU AI Act provenance obligations for high-risk systems enter "
                             "enforcement August 2026; this view exists so the lineage already "
                             "recorded here is machine-readable by a third party rather than only "
                             "by this repository's own tooling.",
    }
