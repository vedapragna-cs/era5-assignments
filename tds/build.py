"""Pipeline assembly: documents -> cleaned -> firewalled -> shards -> admitted -> packed -> pools.

Shared by run_demo.py, the tests and the negative controls so that every one of them exercises the
same code path. If the demo and the tests built the corpus differently, the tests would not be
testing the thing that produced the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .canon import derive_rng
from .clean import CleanRecord, run_cleaning
from .corpus import Document, load_corpus
from .firewall import Firewall
from .mixture import LanePool, MixturePlan
from .packing import DocUnit, PackedSequence, pack, units_from_shard
from .shards import AdmissionResult, Shard, admit, build_shards
from .tokenizer import Tokenizer, train_tokenizer

# S5 reserves scarce material for the anneal: tier-A verified Indic, long agentic trajectories, long
# reasoning traces. Reserved documents are pooled but undrawable until the anneal stage.
RESERVED_DOC_IDS = {"indic-0004", "agent-0002", "reason-0003"}


@dataclass
class BuiltCorpus:
    documents: list[Document]
    clean_records: dict[str, CleanRecord]
    firewall: Firewall
    tokenizer: Tokenizer
    shards: list[Shard]
    admission: list[AdmissionResult]
    admitted_shards: list[Shard]
    samples: dict[str, PackedSequence]
    sample_meta: dict[str, dict]
    pools: dict[str, LanePool]
    packing_by_policy: dict[str, list[PackedSequence]] = field(default_factory=dict)
    blocked_docs: list[dict] = field(default_factory=list)
    excluded_docs: list[dict] = field(default_factory=list)
    golden: list[PackedSequence] = field(default_factory=list)
    doc_tokens: dict[str, int] = field(default_factory=dict)

    @property
    def reserved_sample_ids(self) -> set[str]:
        return {sid for sid, m in self.sample_meta.items()
                if any(d in RESERVED_DOC_IDS for d in m["doc_ids"])}


def build(plan: MixturePlan, *, sequence_length: int, seed: int,
          vocab_size: int = 512, mixed_lane_shards: bool = False,
          indic_difficulty_ordered: bool = False) -> BuiltCorpus:
    docs = load_corpus()

    # 1. cleaning / dedup / PII / language validation
    clean_records = run_cleaning(docs)

    # 2. firewall: register eval + validation, then screen every candidate
    fw = Firewall.from_corpus(docs)
    overlap = fw.overlap_status([d for d in docs if d.split == "train"])
    blocked = [{"doc_id": d, "status": s} for d, s in sorted(overlap.items()) if s != "clear"]

    # 3. tokenizer: trained ONCE on admitted training text only, then frozen.
    #    Training it on eval text would leak benchmark vocabulary into the model's tokenizer, which is
    #    a contamination channel that survives every downstream filter.
    train_text = [clean_records[d.doc_id].text for d in docs
                  if d.split == "train" and overlap.get(d.doc_id) == "clear"]
    tok = train_tokenizer(train_text, vocab_size=vocab_size)

    # 4. DOCUMENT-level admission, before anything is sealed into a shard.
    #
    #    This ordering is load-bearing and an earlier build got it wrong. A shard manifest reports
    #    the worst status among its members, so a single near-duplicate or one unsafe-license
    #    document blocks the entire shard. Measured on that build: every web shard was blocked and
    #    the web lane - 34% of S5's budget, the largest lane in the plan - silently vanished from
    #    the run. Delivered Indic then read 35% against a 17% plan, which looked like a floor bug
    #    and was actually a lane that no longer existed.
    #
    #    Real pipelines gate per document and only then seal, so a shard is a population that is
    #    already clean. Exclusions are recorded rather than dropped, because "why is this document
    #    not in training" is exactly the question the audit trail exists to answer.
    trainable_all = [d for d in docs if d.split == "train"]
    if mixed_lane_shards:
        trainable_all = _mix_indic_into_web(trainable_all)
    trainable, excluded = [], []
    for d in trainable_all:
        rec = clean_records[d.doc_id]
        reasons = []
        if d.license_tier == "unsafe":
            reasons.append("license_tier=unsafe")
        if rec.dedup_status != "unique":
            reasons.append(f"dedup={rec.dedup_status}:{rec.dedup_cluster_id}")
        if overlap.get(d.doc_id) != "clear":
            reasons.append("eval_overlap")
        if not rec.language_validated:
            reasons.append(f"language_validation_failed(declared={d.script})")
        # OLMo 2 arXiv:2501.00656 section 3.1: documents carrying long repeated n-gram runs are
        # over-represented in the batches where loss and gradient-norm spikes occur, and filtering
        # them measurably reduces spike frequency. Excluded at the document level for the same
        # reason dedup is - one degenerate document must not condemn the shard around it (F5).
        if rec.repeat_ngram_status != "clean":
            d_ = rec.repeat_ngram_detail
            reasons.append(f"repeated_ngram_run={d_.get('max_repeat_run')}"
                           f"x{d_.get('repeat_ngram_size')}-gram")
        if reasons:
            excluded.append({"doc_id": d.doc_id, "lane": d.lane, "reasons": reasons})
        else:
            trainable.append(d)
    shards = build_shards(trainable, clean_records, tok, overlap_status=overlap, split="train")

    # Stage eligibility, DERIVED onto the manifest rather than left implicit in the scheduler.
    # A curriculum whose eligibility exists only as a runtime branch cannot be audited without
    # running it; declaring it on the shard lets a reader check which stages may consume a shard
    # straight from `manifests/shards/`.
    #
    # Two conditions, and the second is the length reserve stated structurally: a shard is eligible
    # for a stage when one of its difficulty bands is permitted there AND its longest indivisible
    # document fits that stage's window.
    #
    # This is DECLARATIVE only - the scheduler still routes ineligible candidates through OPUS,
    # because a protected floor must be able to override the stage gate (S5: floors are
    # non-negotiable and outrank the curriculum). A design that simply never draws ineligible
    # shards cannot express that rescue, and this run records 70 of them.
    widest = max(s.sequence_length for s in plan.stages)
    for sh in shards:
        doc_lens = [sp.end - sp.start for sp in sh.spans]
        longest = max(doc_lens, default=0)
        elig = {}
        for st in plan.stages:
            band_ok = bool(set(sh.manifest.get("difficulty_bands", [])) & set(st.difficulty_bands))
            window = max(1, sequence_length * st.sequence_length // widest)
            fitting = sum(1 for n in doc_lens if n <= window)
            elig[st.id] = {
                "band_permitted": band_ok,
                "stage_window_tokens": window,
                "documents_fitting_window": fitting,
                "documents_total": len(doc_lens),
                "longest_document_tokens": longest,
                "eligible": band_ok,
            }
        sh.manifest["stage_eligibility"] = elig
        sh.manifest["eligible_stages"] = [k for k, v in elig.items() if v["eligible"]]
        # The window condition is reported PER DOCUMENT and deliberately does not gate the shard.
        # Letting one over-long document mark a whole shard ineligible is precisely the failure F5
        # recorded, where a single near-duplicate blocked every web shard and deleted 34% of the
        # budget. Length is a per-document reserve (see LanePool.draw), not a shard property.

    # 5. admission gate
    admission = [admit(s.manifest) for s in shards]
    ok_ids = {a.shard_id for a in admission if a.admitted}
    admitted = [s for s in shards if s.shard_id in ok_ids]

    # 6. packing, policy per lane
    samples: dict[str, PackedSequence] = {}
    meta: dict[str, dict] = {}
    by_lane: dict[str, list[PackedSequence]] = {}
    shard_of: dict[str, str] = {}
    doc_by_id = {d.doc_id: d for d in docs}

    for sh in admitted:
        policy = plan.packing_policy.get(sh.lane, "best_fit")
        seqs = pack(units_from_shard(sh), sequence_length, policy, sh.lane)
        for s in seqs:
            samples[s.sample_id] = s
            first = doc_by_id.get(s.members[0]) if s.members else None
            meta[s.sample_id] = {
                "shard_id": sh.shard_id, "lane": sh.lane,
                "doc_ids": list(dict.fromkeys(s.members)),
                "band": first.difficulty_band if first else "B2",
                "language": first.language if first else "en",
                "script": first.script if first else "Latn",
                "policy": policy,
            }
            shard_of[s.sample_id] = sh.shard_id
            by_lane.setdefault(sh.lane, []).append(s)

    # 7. lane pools with two reserves held back
    #    (a) the anneal reserve  - protects QUALITY (S5's Tier-A Indic, long agentic, long reasoning)
    #    (b) the length reserve  - protects LENGTH. A document that needs a wider window than the
    #        current stage provides is undrawable until a stage that can hold it. The session states
    #        this rule ("you need to preserve that shard for future") and the plan never carries it.
    reserved = {sid for sid, m in meta.items() if any(d in RESERVED_DOC_IDS for d in m["doc_ids"])}
    difficulty_of = {sid: m["band"] for sid, m in meta.items()}

    doc_tokens: dict[str, int] = {}
    for sh in admitted:
        for span in sh.spans:
            doc_tokens[span.doc_id] = span.end - span.start
    min_seq_len = {sid: max((doc_tokens.get(d, 0) for d in m["doc_ids"]), default=0)
                   for sid, m in meta.items()}
    pools = {}
    for lane, seqs in sorted(by_lane.items()):
        pools[lane] = LanePool.build(
            lane, seqs, shard_of, seed=seed, reserved_ids=reserved,
            difficulty_of=difficulty_of, min_seq_len=min_seq_len,
            difficulty_ordered=(indic_difficulty_ordered and lane == "indic"),
        )

    # 8. E3 comparison: every policy over the same units, for the packing table
    all_units: list[DocUnit] = [u for sh in admitted for u in units_from_shard(sh)]
    by_policy = {p: pack(list(all_units), sequence_length, p, "all")
                 for p in ("pad_only", "concat_chop", "greedy", "best_fit",
                           "structure_preserving", "long_context")}

    # 9. golden proxy set for OPUS. Packed but never pooled, so it can define the selection
    #    direction without ever being trainable. Asserted below rather than assumed.
    proxy_docs = [d for d in docs if d.split == "proxy"]
    golden: list[PackedSequence] = []
    if proxy_docs:
        gshards = build_shards(proxy_docs, clean_records, tok,
                               overlap_status={d.doc_id: "clear" for d in proxy_docs},
                               split="proxy")
        for gs in gshards:
            golden.extend(pack(units_from_shard(gs), sequence_length, "structure_preserving", gs.lane))
    proxy_ids = {d.doc_id for d in proxy_docs}
    leaked = [sid for sid, m in meta.items() if set(m["doc_ids"]) & proxy_ids]
    assert not leaked, f"golden proxy documents leaked into trainable samples: {leaked}"

    return BuiltCorpus(documents=docs, clean_records=clean_records, firewall=fw, tokenizer=tok,
                       shards=shards, admission=admission, admitted_shards=admitted,
                       samples=samples, sample_meta=meta, pools=pools,
                       packing_by_policy=by_policy, blocked_docs=blocked, golden=golden,
                       excluded_docs=excluded, doc_tokens=doc_tokens)


def _mix_indic_into_web(docs: list[Document]) -> list[Document]:
    """E1 Arm M: relabel a floor-ratio slice of Indic documents into the web lane.

    This is the arrangement the instructor described as undecided - "do we make a shard in which we
    have 10% [indic] and 90% [other]". Relabelling the lane is what makes the shard mixed: the
    scheduler now sees those tokens as web, and the only thing that can force them through is a
    protected-floor override on a batch it believes is web.
    """
    out = []
    for d in docs:
        if d.lane == "indic" and d.doc_id in ("indic-0003", "indic-0005"):
            out.append(Document(
                doc_id=d.doc_id, lane="web", source_id=d.source_id, license_tier=d.license_tier,
                language=d.language, script=d.script, difficulty_band=d.difficulty_band,
                split=d.split, segments=d.segments, benchmark_id=d.benchmark_id, canary=d.canary,
                derived_from_benchmark=d.derived_from_benchmark,
                notes=d.notes + " [E1 arm M: mixed into web shard]"))
        else:
            out.append(d)
    return out
