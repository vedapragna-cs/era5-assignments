"""Automated tests for the invariants the assignment grades.

Run: python -m pytest tests/ -q      (or: python tests/test_invariants.py)

Design note: these tests import the SAME assembly the demo uses (`tds.build` / `tds.harness`), so a
green suite is evidence about the code that produced `submission_artifacts/`, not about a parallel
fixture. Falsification lives in `tds/controls.py` and is asserted here too - a suite that can only
pass is not evidence that its checks work.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tds import controls, exactly_once, harness                       # noqa: E402
from tds.build import build                                            # noqa: E402
from tds.canon import HashChain, chash                                 # noqa: E402
from tds.firewall import Firewall                                      # noqa: E402
from tds.mixture import MixturePlan, rank_of                           # noqa: E402
from tds.packing import (attention_crossings, blocked_cross_segment_pairs,  # noqa: E402
                         pack, units_from_shard, validate_masks)
from tds.shards import admit, verify_shard_integrity                   # noqa: E402
from tds.tokenizer import PAD, train_tokenizer                         # noqa: E402

PLAN = Path(__file__).resolve().parents[1] / "plan" / "v5_plan.json"
SEQ, GB, SEED, SCALE = 128, 8, 20260806, 100_000_000


class Skip(Exception):
    """A test that could not run. Reported as SKIP and excluded from the passed count.

    A test that `return`s when its precondition is missing is indistinguishable from one that
    checked something, and the runner printed PASS for it. That turned "the advertised CLI works"
    into a no-op assertion on any checkout where the demo had not been run yet.
    """

_plan = MixturePlan.load(PLAN, scale_divisor=SCALE)
_built = build(_plan, sequence_length=SEQ, seed=SEED)


# ------------------------------------------------------------------ tokenizer
def test_tokenizer_is_deterministic():
    texts = [_built.clean_records[d.doc_id].text for d in _built.documents if d.split == "train"]
    a = train_tokenizer(texts, vocab_size=256)
    b = train_tokenizer(texts, vocab_size=256)
    assert a.hash == b.hash, "same inputs must produce the same tokenizer"
    assert a.merges == b.merges


def test_tokenizer_round_trips_indic():
    tok = _built.tokenizer
    for text in ("भारत में शिक्षा", "தமிழ்", "def f(x): return x"):
        assert tok.decode(tok.encode(text)) == text, f"round trip failed for {text!r}"


def test_every_manifest_carries_the_frozen_tokenizer_hash():
    assert _built.shards
    assert {s.manifest["tokenizer_hash"] for s in _built.shards} == {_built.tokenizer.hash}


# ------------------------------------------------------------------ shards
def test_shard_content_hash_is_over_token_bytes():
    for s in _built.shards:
        ok, msg = verify_shard_integrity(s)
        assert ok, msg
        assert s.manifest["content_hash"] == s.content_hash


def test_admission_blocks_every_hard_field():
    m = dict(_built.admitted_shards[0].manifest)
    assert admit(m).admitted
    for f in ("tokenizer_hash", "cleaning_pipeline_hash", "dedup_status",
              "pii_screen_status", "eval_overlap_status", "repeat_ngram_status"):
        assert not admit(dict(m, **{f: "missing"})).admitted, f"{f} must be a hard blocker"
    assert not admit(dict(m, license_tier="unsafe")).admitted


def test_pii_was_actually_redacted():
    rec = _built.clean_records["web-0006"]
    assert rec.pii_screen_status == "redacted"
    assert set(rec.pii_kinds) >= {"email", "phone_in"}
    assert "aditya.singh@example.com" not in rec.text
    assert "9876543210" not in rec.text


def test_near_duplicate_was_detected_and_excluded():
    assert _built.clean_records["web-0007"].dedup_status == "near_duplicate"
    excluded = {e["doc_id"] for e in _built.excluded_docs}
    assert "web-0007" in excluded


# ------------------------------------------------------------------ firewall
def test_no_eval_or_proxy_document_reaches_a_training_sample():
    forbidden = {d.doc_id for d in _built.documents if d.split in ("validation", "test", "proxy")}
    for sid, meta in _built.sample_meta.items():
        assert not (set(meta["doc_ids"]) & forbidden), f"{sid} contains held-out data"


def test_firewall_blocks_a_smuggled_test_document():
    fw = Firewall.from_corpus(_built.documents)
    test_doc = next(d for d in _built.documents if d.split == "test")
    assert not fw.check(test_doc).allowed


def test_derived_benchmark_content_is_caught_without_the_overlap_gate():
    fw = Firewall.from_corpus(_built.documents)
    fw.gates["overlap"] = False
    derived = next(d for d in _built.documents if d.derived_from_benchmark)
    assert not fw.check(derived).allowed, "the derived gate must be independent"


def test_ngram_containment_is_the_only_gate_that_catches_a_quoted_benchmark_item():
    """Decontamination standard: exact n-gram containment, after GPT-3's 13-gram filter.

    Jaccard measures global similarity, so a long host article quoting one benchmark item scores
    near zero against that benchmark. Containment asks whether ANY benchmark n-gram appears at all,
    which is the shape contamination usually takes. This test fails if the gate ever becomes
    redundant with the others - a redundant gate is not defence in depth, it is noise.
    """
    from tds.corpus import ALL_DOCUMENTS
    target = next(d for d in ALL_DOCUMENTS if d.doc_id == "web-0102")
    full = Firewall.from_corpus(ALL_DOCUMENTS)
    dec = full.check(target)
    assert not dec.allowed
    from tds.firewall import OVERLAP_THRESHOLD
    assert dec.max_overlap < OVERLAP_THRESHOLD, "the case is void if Jaccard alone would flag it"
    solo = []
    for g in sorted(full.gates):
        fw = Firewall.from_corpus(ALL_DOCUMENTS)
        for k in fw.gates:
            fw.gates[k] = (k == g)
        if not fw.check(target).allowed:
            solo.append(g)
    assert solo == ["ngram_containment"], f"expected exactly one catching gate, got {solo}"


# ------------------------------------------------------------------ packing / masks
def test_masks_are_structurally_valid_everywhere():
    for sid, seq in _built.samples.items():
        ok, errs = validate_masks(seq)
        assert ok, f"{sid}: {errs[:3]}"


def test_padding_is_never_loss_bearing():
    for seq in _built.samples.values():
        for t, m in zip(seq.tokens, seq.loss_mask):
            assert not (t == PAD and m == 1)


def test_attention_never_crosses_a_document_boundary():
    for sid, seq in _built.samples.items():
        assert attention_crossings(seq) == 0, f"{sid} leaks context across documents"


def test_attention_isolation_is_enforced_in_the_arithmetic():
    """The strongest form of the packing claim: rewrite a neighbour, get identical logits.

    Validating the mask array proves the array is right. This proves the MODEL cannot read across a
    document boundary, by perturbing the earlier document in a packed window and requiring every
    per-token loss of the later document to be bit-identical.

    The direction matters. Attention is causal, so a later document can never influence an earlier
    one whatever the mask says - testing that direction would pass even with segment masking removed
    and would prove nothing. The leak, if it existed, runs earlier -> later.

    The paired control is the point: with the segment ids merged into one document, the same
    perturbation DOES move the later document's losses. So the isolation is attributable to the
    segment mask rather than to the architecture happening not to look.
    """
    import numpy as np
    from tds.model import TinyLM
    m = TinyLM.init(64, seed=7)
    seg = np.array([0] * 6 + [1] * 6)
    lm = np.ones(12, dtype=np.int64)
    a = np.array([5, 6, 7, 8, 9, 10, 20, 21, 22, 23, 24, 25])
    b = a.copy()
    b[:6] = [41, 42, 43, 44, 45, 46]                 # rewrite the FIRST document

    pa, _, _ = m.forward_backward(a, lm, seg)
    pb, _, _ = m.forward_backward(b, lm, seg)
    assert np.array_equal(pa[6:], pb[6:]), "document 2 changed when document 1 was rewritten"

    merged = np.zeros(12, dtype=np.int64)            # same tokens, one document
    qa, _, _ = m.forward_backward(a, lm, merged)
    qb, _, _ = m.forward_backward(b, lm, merged)
    assert not np.array_equal(qa[6:], qb[6:]), (
        "control failed: without the segment mask the model still did not read across the "
        "boundary, so the test above proves nothing")


def test_attention_mask_is_causal_and_block_diagonal():
    import numpy as np
    from tds.model import TinyLM
    seg = np.array([0, 0, 0, 1, 1, -1])
    allow = TinyLM.attention_mask(seg)
    assert not allow[0, 1], "attention must be causal"
    assert allow[2, 0] and allow[2, 1], "same-document history must be visible"
    assert not allow[3, 2] and not allow[4, 0], "cross-document attention must be blocked"
    assert allow[5, 5], "padding rows keep a defined softmax via the diagonal"
    assert not allow[5, 0], "padding must not attend into a document"


def test_attention_is_genuinely_multi_head():
    """Heads must be distinct and the mask must hold in every one of them.

    A "multi-head" implementation whose heads all compute the same thing is a single head with extra
    parameters, and one that applies the mask only to the aggregate would leak inside a head while
    looking clean on the way out.
    """
    import numpy as np
    from tds.model import TinyLM, softmax
    m = TinyLM.init(64, d_model=16, d_hidden=24, n_heads=4, seed=11)
    assert m.n_heads == 4 and m.d_model % m.n_heads == 0
    seg = np.array([0, 0, 0, 1, 1, 1])
    allow = TinyLM.attention_mask(seg)
    toks = np.array([5, 6, 7, 20, 21, 22])
    L, H, dh = len(toks), m.n_heads, m.d_model // m.n_heads
    X = m.E[toks]
    split = lambda M: M.reshape(L, H, dh).transpose(1, 0, 2)
    Q, K = split(X @ m.Wq), split(X @ m.Wk)
    S = np.where(allow[None, :, :], (Q @ K.transpose(0, 2, 1)) / np.sqrt(dh), -1e30)
    A = softmax(S)
    # every head respects the mask
    for h in range(H):
        assert np.allclose(A[h][~allow], 0.0, atol=1e-12), f"head {h} attends across a boundary"
    # and the heads are not copies of one another
    assert not np.allclose(A[0], A[1]), "heads are identical - this is single-head in disguise"


def test_hand_written_gradients_match_finite_differences():
    """Directional derivative over ALL parameters at once, not per-entry sampling.

    The previous version took the three largest-magnitude entries per tensor and skipped anything
    below 1e-6. At initialisation `Wq` and `Wk` gradients are ~4e-08 — so every entry of the
    softmax-Jacobian and masking path was skipped, and an audit showed a 3.7x-wrong `dQ` plus a
    fully deleted `dK` passing the whole suite.

    A random-direction directional derivative fixes that: `d/de f(theta + e*v)` compares against
    `<grad, v>` over every tensor simultaneously, so no component can hide behind its own smallness.
    """
    import numpy as np
    from tds.model import TinyLM
    rng = np.random.default_rng(7)
    toks = np.array([3, 4, 5, 6, 7, 8, 9, 10])
    lm = np.ones(8, dtype=np.int64)
    seg = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    for trial in range(3):
        # max_positions is deliberately small: at the 512 default the positional table would be
        # 98% unused rows with zero gradient, and the directional derivative would be dominated by
        # components that cannot be wrong. At 8 every row of P is exercised.
        m = TinyLM.init(32, d_model=8, d_hidden=12, n_heads=2, max_positions=8, seed=3 + trial)
        per, n, g = m.forward_backward(toks, lm, seg)
        assert n and g
        v = {k: rng.standard_normal(getattr(m, k).shape) for k in m.PARAMS}
        analytic = sum(float((g[k] * v[k]).sum()) for k in m.PARAMS)

        def loss_at(eps):
            saved = {k: getattr(m, k).copy() for k in m.PARAMS}
            for k in m.PARAMS:
                setattr(m, k, saved[k] + eps * v[k])
            p_, n_, _ = m.forward_backward(toks, lm, seg)
            for k in m.PARAMS:
                setattr(m, k, saved[k])
            return float(p_.sum()) / n_

        eps = 1e-6
        numeric = (loss_at(eps) - loss_at(-eps)) / (2 * eps)
        rel = abs(numeric - analytic) / max(1e-12, abs(numeric))
        assert rel < 1e-5, f"trial {trial}: analytic {analytic:.8f} vs numeric {numeric:.8f}"


def test_position_ids_are_load_bearing():
    """The packer's position ids must change the arithmetic, or "ids reset at every boundary" is
    a claim about an array nothing reads.

    This exists because that was the case: `forward_backward` had no position argument and the
    architecture had no positional term, so the model was permutation-invariant over its context and
    correct ids were behaviourally indistinguishable from arbitrary ones. Three properties:

      1. scrambling the ids moves the loss (they are an input at all);
      2. the SAME document at two different offsets in a pack gets identical positional treatment
         (the ids are within-document, which is the property packing is supposed to deliver);
      3. an id beyond the table is refused rather than silently wrapping.
    """
    import numpy as np
    from tds.model import TinyLM
    m = TinyLM.init(64, d_model=16, d_hidden=24, n_heads=4, max_positions=32, seed=13)
    toks = np.array([5, 6, 7, 8, 9, 10, 11, 12])
    lm = np.ones(8, dtype=np.int64)
    seg = np.zeros(8, dtype=np.int64)

    a, _, ga = m.forward_backward(toks, lm, seg, np.arange(8))
    b, _, _ = m.forward_backward(toks, lm, seg, np.array([3, 1, 0, 2, 6, 4, 7, 5]))
    assert not np.array_equal(a, b), "scrambling position ids left the loss unchanged"
    assert np.abs(ga["P"]).sum() > 0, "positional embedding received no gradient"
    # a position the sequence never uses must receive exactly zero gradient
    assert np.allclose(ga["P"][8:], 0.0), "unused positions received gradient"

    # (2) the same three-token document packed at offset 0 and at offset 3 sees the same positions,
    # so its within-document positional treatment is offset-independent.
    doc = [5, 6, 7]
    left = np.array(doc + [40, 41, 42]); right = np.array([40, 41, 42] + doc)
    segs2 = np.array([0, 0, 0, 1, 1, 1])
    pos2 = np.array([0, 1, 2, 0, 1, 2])
    pl, _, _ = m.forward_backward(left, np.ones(6, dtype=np.int64), segs2, pos2)
    pr, _, _ = m.forward_backward(right, np.ones(6, dtype=np.int64), segs2, pos2)
    assert np.allclose(pl[:2], pr[3:5]), \
        "the same document at a different pack offset got different positional treatment"

    # (3) out-of-range ids are refused, not wrapped
    try:
        m.forward_backward(toks, lm, seg, np.arange(8) + 100)
    except ValueError:
        pass
    else:
        raise AssertionError("a position id beyond max_positions was silently accepted")


def test_segment_masking_is_load_bearing():
    """Zero crossings only means something if a causal-only mask would have allowed some."""
    multi = [s for s in _built.samples.values()
             if len({x for x in s.segment_ids if x >= 0}) > 1]
    assert multi, "no multi-document window packed - the isolation claim would be vacuous"
    assert sum(blocked_cross_segment_pairs(s) for s in multi) > 0


def test_agentic_tool_observations_are_context_not_supervision():
    from tds.corpus import CONTEXT_ROLES
    sh = next(s for s in _built.admitted_shards if s.lane == "agentic")
    for span in sh.spans:
        for role, start, end in span.roles:
            if role in CONTEXT_ROLES:
                for u in units_from_shard(sh):
                    if u.doc_id != span.doc_id:
                        continue
                    lo, hi = start - span.start, end - span.start
                    assert not any(u.loss_mask[lo:hi]), f"{role} must not be loss-bearing"


def test_concat_chop_actually_truncates():
    """The course widget's version never splits a document; ours must."""
    units = [u for sh in _built.admitted_shards for u in units_from_shard(sh)]
    seqs = pack(units, SEQ, "concat_chop", "all")
    assert sum(len(s.truncated) for s in seqs) > 0


def test_greedy_and_best_fit_are_not_the_same_algorithm():
    units = [u for sh in _built.admitted_shards for u in units_from_shard(sh)]
    g = pack(list(units), SEQ, "greedy", "all")
    b = pack(list(units), SEQ, "best_fit", "all")
    assert [s.members for s in g] != [s.members for s in b]


def test_all_six_policies_are_genuinely_distinct():
    """No two policies may produce the same packing.

    `long_context` and `structure_preserving` were once byte-identical implementations, so the
    "six genuinely distinct policies" claim was really five - and the course widget this submission
    criticises for aliasing three policies together was being matched, not improved on.
    """
    from tds.packing import POLICIES
    units = [u for sh in _built.admitted_shards for u in units_from_shard(sh)]
    shapes = {}
    for pol in POLICIES:
        seqs = pack(list(units), SEQ, pol, "all")
        shapes[pol] = (len(seqs), tuple(tuple(s.members) for s in seqs),
                       sum(len(s.truncated) for s in seqs))
    for a in POLICIES:
        for b in POLICIES:
            if a < b:
                assert shapes[a] != shapes[b], f"{a} and {b} produce identical packings"


def test_truncation_and_splitting_are_reported_separately():
    """A split keeps every token; a truncation drops content. Conflating them misleads."""
    from tds import throughput
    rows = {r["policy"]: r for r in throughput.packing_table(_built.packing_by_policy)}
    assert rows["best_fit"]["documents_truncated"] == 0
    assert rows["best_fit"]["documents_split_across_windows"] > 0, (
        "best_fit does split over-length documents; reporting 0 truncations without saying so "
        "makes the comparison against concat_chop apples-to-oranges")
    assert rows["concat_chop"]["documents_truncated"] > 0


def test_packing_never_reports_impossible_utilization():
    for policy in ("pad_only", "concat_chop", "greedy", "best_fit",
                   "structure_preserving", "long_context"):
        units = [u for sh in _built.admitted_shards for u in units_from_shard(sh)]
        for s in pack(units, SEQ, policy, "all"):
            assert 0 <= s.n_pad <= s.length
            assert 0.0 <= s.utilization <= 1.0


# ------------------------------------------------------------------ mixture / floors
def test_protected_floors_survive_normalisation():
    for st in _plan.stages:
        mix = _plan.effective_mixture(st.token_start)
        assert abs(sum(mix.values()) - 1.0) < 1e-9
        for lane, floor in _plan.protected_floors.items():
            assert mix[lane] >= floor - 1e-9, f"{lane} below floor in {st.id}"


def test_stage_demand_uses_the_stage_window_not_the_run_total():
    f = _plan.feasibility(0.0)
    for row in f["lanes"]:
        assert abs(sum(s["demand_tokens"] for s in row["per_stage"]) - row["demand_tokens"]) <= 2


def test_rejection_rate_inflates_demand():
    a = {r["lane"]: r["demand_tokens"] for r in _plan.feasibility(0.0)["lanes"]}
    b = {r["lane"]: r["demand_tokens"] for r in _plan.feasibility(0.5)["lanes"]}
    for lane in a:
        assert b[lane] > a[lane], "rejection must multiply required supply"


# ------------------------------------------------------------------ ledger
def test_hash_chain_detects_tampering():
    c = HashChain()
    for i in range(6):
        c.append({"event": "batch_committed", "global_step": i})
    assert c.verify()[0]
    c.entries[3]["global_step"] = 99
    ok, idx, _ = c.verify()
    assert not ok and idx == 3


def test_rank_partition_is_a_pure_function():
    for ws in (1, 2, 4, 8, 16):
        assert [rank_of(i, ws) for i in range(64)] == [i % ws for i in range(64)]


# ------------------------------------------------------------------ recovery (the graded ones)
def _traces(steps=12, crash_at=7):
    kw = dict(steps=steps, seed=SEED, global_batch=GB, sequence_length=SEQ)
    a = harness.run_uninterrupted(_plan, _built, world_size=4, **kw)
    b = harness.run_with_crash(_plan, _built, crash_at=crash_at,
                               world_size_before=4, world_size_after=8, **kw)
    return a, b


def test_resume_is_exactly_once_and_order_preserving_across_world_size():
    a, b = _traces()
    r = exactly_once.compare(harness.trace_of(a, label="a"), harness.trace_of(b, label="b"))
    assert r["order_identical"], f"diverged at {r['first_divergence']}"
    assert r["coverage_run_b"]["duplicate_positions"] == 0
    assert r["coverage_run_b"]["missing_positions"] == 0
    assert r["exactly_once_order_preserving"]


def test_replay_reproduces_batch_hashes_and_spans():
    a, _ = _traces()
    rep = a.replay_interval(2, 6)
    assert rep["n_batches"] > 0
    assert rep["all_match"]
    for row in rep["batches"]:
        assert row["batch_hash_match"] and row["span_ids_match"] and row["sample_ids_match"]


def test_checkpoint_binds_data_position_not_just_weights():
    a, _ = _traces()
    st = a.checkpoints[sorted(a.checkpoints)[0]]
    for k in ("model_state", "optimizer_state", "scheduler_state", "rng_policy",
              "dataloader_state", "ledger_offset"):
        assert k in st, f"checkpoint missing {k}"
    assert "pool_cursors" in st["dataloader_state"]
    assert "global_sample_index" in st["dataloader_state"]


def test_fork_with_no_change_reproduces_the_parent_stream():
    """Makes a real fork's divergence attributable to its declared change."""
    a, _ = _traces(steps=8, crash_at=5)
    ck = sorted(a.checkpoints)[0]
    parent = {e["global_step"]: e["batch_hash"] for e in a.ledger.batches("run-a")}
    a.fork(ck, "run-control")
    a.train_step()
    ctl = a.ledger.batches("run-control")
    assert ctl and all(e["batch_hash"] == parent[e["global_step"]] for e in ctl)


def test_ledger_never_deletes_superseded_batches():
    _, b = _traces()
    with_super = b.ledger.batches("run-a", include_superseded=True)
    without = b.ledger.batches("run-a")
    assert len(with_super) >= len(without)
    assert any(e.get("event") == "batches_superseded" for e in b.ledger.chain.entries)


# ------------------------------------------------------------------ repeated n-grams (OLMo 2)
def test_repeat_ngram_screen_catches_a_unique_degenerate_document():
    """The screen must catch what dedup structurally cannot.

    OLMo 2 (arXiv:2501.00656 s3.1) filter documents with a run of 32+ repeated n-grams because
    those documents are over-represented in spiking batches. The planted case is UNIQUE across the
    corpus, so every similarity-based check passes it - which is precisely why it needs its own
    screen rather than a tighter dedup threshold.
    """
    from tds.clean import find_repeated_ngrams, REPEAT_NGRAM_RUN
    flagged = [r for r in _built.clean_records.values() if r.repeat_ngram_status != "clean"]
    assert flagged, "planted repeated-n-gram document was not flagged"
    r = flagged[0]
    assert r.dedup_status == "unique", "the control is void if dedup would have caught it anyway"
    assert r.repeat_ngram_detail["max_repeat_run"] >= REPEAT_NGRAM_RUN
    # and it must not fire on ordinary prose
    assert not find_repeated_ngrams(
        "the quick brown fox jumps over the lazy dog and then goes back to sleep")["flagged"]


def test_flagged_documents_never_reach_a_shard():
    from tds.clean import repeated_ngram_run, REPEAT_NGRAM_RUN
    flagged = {r.doc_id for r in _built.clean_records.values()
               if r.repeat_ngram_status != "clean"}
    in_shards = {sp.doc_id for sh in _built.shards for sp in sh.spans}
    assert not (flagged & in_shards), f"degenerate documents reached shards: {flagged & in_shards}"


def test_trainer_side_repeat_mask_is_defence_in_depth_not_decoration():
    """Masks the span, keeps the tokens, and leaves clean sequences untouched."""
    from tds.packing import mask_repeated_ngrams
    toks = [7] * 40 + [1, 2, 3]
    masked, n = mask_repeated_ngrams(toks, [1] * len(toks))
    assert n == 40 and sum(masked) == 3, (n, sum(masked))
    assert len(masked) == len(toks), "masking must not change sequence length or token offsets"
    clean_toks = list(range(40))
    _, n_clean = mask_repeated_ngrams(clean_toks, [1] * 40)
    assert n_clean == 0, "safeguard fired on a non-degenerate sequence"


def test_repeat_screen_is_in_the_pipeline_hash():
    """Changing the filter must change the cleaning pipeline hash, or reruns are not comparable."""
    import tds.clean as C
    before = C.pipeline_hash()
    orig = C.REPEAT_NGRAM_RUN
    try:
        C.REPEAT_NGRAM_RUN = orig + 1
        assert C.pipeline_hash() != before, "pipeline hash ignores the repeat-ngram parameters"
    finally:
        C.REPEAT_NGRAM_RUN = orig
    assert C.pipeline_hash() == before


# ------------------------------------------------------------------ anneal reserve
def _reserved_pool():
    """A lane pool that actually holds anneal-reserved samples."""
    for lane, pool in sorted(_built.pools.items()):
        if pool.reserved_ids:
            return lane, pool
    raise AssertionError("no lane pool carries anneal-reserved samples")


def test_reserve_withholds_and_reports_what_it_withheld():
    """Outside anneal a reserved sample must be skipped AND named.

    Skipping silently is the defect: a document held back with no record is indistinguishable from
    a document that was never in the corpus. The `withheld` list is what turns the skip into an
    auditable OPUS deferral.
    """
    _, pool = _reserved_pool()
    seen, cursors = set(), []
    for cur in range(len(pool.order)):
        w: list[str] = []
        s, _, _ = pool.draw(cur, allow_reserved=False, withheld=w)
        assert s.sample_id not in pool.reserved_ids, "a reserved sample was served outside anneal"
        seen.update(w)
        cursors.append(w)
    assert seen, "reserve never reported a withheld sample, so no deferral could be recorded"
    assert seen <= pool.reserved_ids, f"reported non-reserved ids as withheld: {seen - pool.reserved_ids}"


def test_reserve_releases_every_reserved_sample_in_anneal():
    """The release property, for ALL reserved samples.

    The demo's scaled run contains a single anneal step, so it can only ever *draw* a couple of the
    reserved documents before the plan ends. That is a property of S5's anneal being 3% of the run,
    not of the mechanism - so the mechanism is asserted here over every reserved sample rather than
    inferred from which ones the run happened to reach.
    """
    # reserved_ids is a run-wide set handed to every pool, so a pool's copy names samples that live
    # in other lanes. The property is per-owning-pool: each reserved sample must be reachable in the
    # lane that actually holds it.
    unreachable, checked = set(), 0
    for pool in _built.pools.values():
        own = {s.sample_id for s in pool.samples}
        mine = pool.reserved_ids & own
        if not mine:
            continue
        drawable = set()
        for cur in range(len(pool.order)):
            w: list[str] = []
            s, _, _ = pool.draw(cur, allow_reserved=True, withheld=w)
            drawable.add(s.sample_id)
            assert not w, "nothing may be withheld for the reserve once anneal has opened"
        unreachable |= mine - drawable
        checked += len(mine)
    assert checked, "no pool owns any reserved sample"
    assert not unreachable, f"reserved samples still unreachable in anneal: {unreachable}"


def test_deferrals_are_not_counted_as_judged_candidates():
    """A deferral must not move the rejection rate.

    `deferred` says the STAGE is wrong; `rejected` says the sample lost to its peers. Folding
    deferrals into the denominator would silently redefine the rejection rate and make it
    incomparable to the plan's target, so unscored deferrals carry a null score and are excluded.
    """
    from tds.opus import Opus, OpusDecision
    o = Opus(rejection_rate=0.18, proxy_version="test", seed=1, protected_lanes=set())
    mk = lambda dec, score: OpusDecision(
        "d", "c", "s", "sh", "web", "S1_seed", 0, score, 0, 1, 1, dec, "r", 0, "test", 0, False)
    o.decisions = [mk("accepted", 0.5), mk("rejected", -0.5)]
    base = o.summary()
    o.decisions.append(mk("deferred", None))
    after = o.summary()
    assert after["observed_rejection_rate"] == base["observed_rejection_rate"]
    assert after["n_candidates"] == base["n_candidates"] == 2
    assert after["n_reserve_deferrals"] == 1


# ------------------------------------------------------------------ reference model / excess loss
def test_excess_loss_separates_learnable_from_irreducible():
    """The whole point of a reference model: two tokens with the SAME loss, different verdicts."""
    from tds.reference import classify
    # same absolute loss, different reference loss -> opposite conclusions
    assert classify(8.0, 6.0, high_loss=7.0) == "learnable_hard"
    assert classify(8.0, 8.0, high_loss=7.0) == "irreducible_hard"
    assert classify(2.0, 2.0, high_loss=7.0) == "already_learned"
    assert classify(2.0, 5.0, high_loss=7.0) == "saturating"


def test_reference_model_is_frozen_and_scores_deterministically():
    from tds.reference import ReferenceModel
    r = ReferenceModel.train(_built.golden, _built.tokenizer.size, seed=SEED, steps=5)
    before = r.weight_hash
    seq = next(iter(_built.samples.values()))
    a = r.per_position_loss(seq).tolist()
    b = r.per_position_loss(seq).tolist()
    assert a == b, "scoring must be a pure function of a frozen model"
    assert r.weight_hash == before, "scoring must not update the reference"


# ------------------------------------------------------------------ spike attribution
def test_a_spike_never_becomes_a_shard_verdict():
    """The data-poisoning bug this guards against: `harmful` written from mere coincidence."""
    from tds import spike
    from tds.learning import LearningLedger
    series = [{"step": i, "mean_loss": 6.0, "branch": "run-a"} for i in range(20)]
    series[14]["mean_loss"] = 12.0
    for i in range(20):
        series[i]["mean_loss"] += 0.001 * (i % 3)          # non-zero baseline variance
    hits = spike.detect(series, branch="run-a")
    assert len(hits) == 1 and hits[0].global_step == 14

    led = LearningLedger(tier="full")
    led.shard_exposures = [{"shard_id": "s1", "global_step": 14, "usefulness": "harmful",
                            "attribution_uncertain": False}]
    led.apply_spike_attribution(spike.spike_windows(hits), {14: "interaction_not_data"})
    rec = led.shard_exposures[0]
    assert rec["usefulness"] == "unattributed" and rec["attribution_uncertain"]
    assert rec["usefulness_withheld"] == "harmful"


def test_spike_detector_does_not_fire_on_a_flat_series():
    from tds import spike
    flat = [{"step": i, "mean_loss": 6.0 + 0.001 * (i % 3), "branch": "run-a"} for i in range(20)]
    assert spike.detect(flat, branch="run-a") == []


# ------------------------------------------------------------------ dedup sensitivity
def test_dedup_break_even_matches_its_own_formula():
    """d_max = 1 - demand/(supply*ceiling), recomputed independently of the implementation."""
    sens = _plan.dedup_sensitivity()
    feas = {r["lane"]: r for r in _plan.feasibility()["lanes"]}
    for row in sens["lanes"]:
        f = feas[row["lane"]]
        expected = 1 - f["demand_tokens"] / (f["supply_tokens"] * f["ceiling_applied"])
        assert abs(row["max_tolerable_dedup_fraction"] - expected) < 1e-3, row["lane"]
    # a lane already over its ceiling must report negative headroom, not zero
    over = [r for r in sens["lanes"] if r["status"] == "already_breached"]
    assert over and all(r["max_tolerable_dedup_fraction"] < 0 for r in over)


def test_ceiling_is_expressed_per_lane():
    feas = _plan.feasibility()
    for row in feas["lanes"]:
        assert "repetition_factor_r" in row and "ceiling_applied" in row
        assert row["within_ceiling"] == (row["repetition_factor_r"] <= row["ceiling_applied"])


# ------------------------------------------------------------------ croissant provenance
def test_croissant_view_carries_lineage_for_every_admitted_shard():
    from tds import croissant
    doc = croissant.build([sh.manifest for sh in _built.admitted_shards],
                          plan_id=_plan.plan_id, run_id="test",
                          tokenizer=_built.tokenizer.to_record())
    assert len(doc["distribution"]) == len(_built.admitted_shards)
    for d in doc["distribution"]:
        assert d["sha256"], "every shard must carry its content hash"
        used = d["prov:wasGeneratedBy"]["prov:used"]
        assert {u["@id"] for u in used} == {"tokenizer", "cleaning_pipeline"}
        assert d["prov:wasGeneratedBy"]["cr:qualityGates"]["repeat_ngram_status"] is not None
    # conformance must be claimed as unvalidated, not asserted
    assert "UNVALIDATED" in doc["cr:conformanceClaim"]


# ------------------------------------------------------------------ advertised CLI surface
def test_every_documented_audit_command_runs():
    """The README advertises six audit queries; a documented command that crashes is a broken claim.

    This exists because two of them did crash: reserve deferrals carry a null OPUS score by design,
    and `--step` and `--opus` formatted it as a float. Nothing caught it, because the tests exercised
    the library and the demo exercised the pipeline - neither ran the CLI a grader would type.
    """
    import subprocess
    root = Path(__file__).resolve().parents[1]
    if not (root / "submission_artifacts" / "ledgers" / "consumption.jsonl").exists():
        raise Skip("submission_artifacts/ledgers/consumption.jsonl absent - run run_demo.py first")
    for args in (["--step", "7"], ["--interval", "0", "6"], ["--branches"],
                 ["--opus"], ["--chain"]):
        r = subprocess.run([sys.executable, str(root / "audit.py"), *args],
                           capture_output=True, text=True, cwd=root)
        assert r.returncode == 0, f"audit.py {' '.join(args)} exited {r.returncode}: {r.stderr[-400:]}"


# ------------------------------------------------------------------ falsifiability
def test_spike_ablation_leaves_the_audited_run_untouched():
    """An ablation is a read-only audit. It restored only branch/step/index/cursors.

    Measured before the fix: the live model came back holding the ablation branch's weights
    (01ce91522da54690 -> dbdcd3dfc52d1178) and the planner's cumulative delivery was rewound to the
    pre-spike checkpoint (total_delivered 82 -> 34). The next consumer of `planner.delivered` is the
    rank-compute-skew measurement, so a shipped number was planned against a delivery history that
    never happened.
    """
    from tds import spike
    rig = harness.make_rig(_plan, _built, run_id="ablate-test", seed=SEED,
                           global_batch=GB, sequence_length=SEQ, world_size=4)
    tr = rig.trainer
    for _ in range(14):
        tr.train_step()
        if tr.step % 6 == 0:
            tr.save_checkpoint()

    before = (tr.model.weight_hash, dict(tr.planner.delivered), tr.planner.total_delivered,
              tr.branch_id, tr.step, tr.global_index, tr.planner.peek_cursors())
    ev = spike.SpikeEvent(global_step=tr.step - 1, branch_id=tr.branch_id, metric="grad_norm",
                          value=9.9, baseline_mean=1.0, baseline_std=0.5, z=4.5)
    out = spike.ablate(tr, ev, lookback=6)
    after = (tr.model.weight_hash, dict(tr.planner.delivered), tr.planner.total_delivered,
             tr.branch_id, tr.step, tr.global_index, tr.planner.peek_cursors())
    names = ("weight_hash", "delivered", "total_delivered", "branch_id", "step", "global_index",
             "cursors")
    drift = [n for n, a, b in zip(names, before, after) if a != b]
    assert not drift, f"ablation mutated the audited run: {drift}"
    if out.get("ran"):
        assert "criterion" in out, "the ablation verdict does not state the rule that produced it"


def test_learning_aggregates_count_one_live_branch_only():
    """Aggregates must exclude forks, control forks, replay re-derivations and superseded batches.

    `effective_tokens` was written for exactly this and documented why, but `aggregate` read the
    running fold instead - which is accumulated on every train_step of every branch. The shipped
    evidence.json then reported `tokens_traced = 4457` and, in the same record, per-lane counts
    summing to 23252, and the lane ORDERING changed when recomputed on run-a alone.
    """
    from tds.learning import LearningLedger, TokenObservation

    def obs(branch, lane, loss, step=0):
        return TokenObservation(
            step=step, sample_id=f"s{step}", shard_id=f"sh_{lane}", doc_id="d", lane=lane,
            language="en", script="Latn", position_in_pack=0, position_in_doc=0,
            position_bucket=0, is_boundary=False, is_special=False, token_id=1, decoded="x",
            loss=loss, opus_decision_id=None, model_age_tokens=0, stage="S1_seed",
            pass_number=1, role="text", branch_id=branch)

    ll = LearningLedger()
    ll.observe_tokens([obs("run-a", "web", 1.0), obs("run-a", "web", 1.0)])
    ll.observe_tokens([obs("run-a", "code", 5.0, step=9)])         # will be superseded
    ll.observe_tokens([obs("run-b", "web", 99.0), obs("run-control", "code", 99.0)])
    ll.mark_superseded("run-a", from_step=9)

    agg = ll.aggregate("lane")
    assert set(agg) == {"web"}, f"a non-live branch or superseded lane leaked in: {sorted(agg)}"
    assert agg["web"]["n_tokens"] == 2, agg["web"]["n_tokens"]
    assert agg["web"]["mean_loss"] == 1.0, "a fork's loss contaminated the mean"
    assert sum(v["n_tokens"] for v in agg.values()) == len(ll.effective_tokens("run-a")), \
        "aggregate and effective_tokens disagree about the population"
    # and the other branch is still reachable when explicitly asked for
    assert ll.aggregate("lane", branch="run-b")["web"]["n_tokens"] == 1


def test_readme_figures_still_match_the_artifacts():
    """Every load-bearing number the README quotes must appear in the current artifacts.

    This submission's whole thesis is that its numbers are reproducible from `submission_artifacts/`.
    A stale README figure is therefore the most expensive error it can contain, and it had six:
    the E1 table (27.03/23.81/52.38 vs 15.79/14.49/49.28 at the time), every value in the excess-loss
    lane table plus its ordering, agentic delivery 2.65% vs 2.58%, rank skew 71.2% vs 52.72%,
    "29 negative controls" vs 30, and the test count. The excess-loss row was the worst: the README
    built an argument on "indic +0.0001 (matched)" while the artifact said +0.0563 behind_reference.

    Each entry below is (artifact-derived string, what it describes). The test asserts the string is
    present in README.md, so a figure that moves fails here instead of shipping.

    Stated limitation: this guards figures from the DEFAULT run, because that is what
    `submission_artifacts/` holds. The README's `--deep` figures (lane perplexity spread, per-lane
    exposure shares) are not covered - guarding them would mean a ~10 minute test. They are labelled
    `--deep` in the README and were last reconciled by hand against a full deep run; that is weaker
    than the rest and is said so rather than implied.
    """
    import json
    root = Path(__file__).resolve().parents[1]
    art = root / "submission_artifacts"
    if not (art / "evidence.json").exists():
        raise Skip("submission_artifacts absent - run run_demo.py first")
    readme = (root / "README.md").read_text(encoding="utf-8")

    perf = json.loads((art / "performance.json").read_text())
    e1 = json.loads((art / "reports" / "experiments.json").read_text())["E1"]["arms"]
    xl = json.loads((art / "reports" / "v6_recommendations.json").read_text())["excess_loss"]["by_lane"]
    nc = json.loads((art / "reports" / "negative_controls.json").read_text())
    ev = json.loads((art / "evidence.json").read_text())
    mc = next(r for r in ev["requirements"] if r["key"] == "mixture_compliance")["compared_values"]

    want: list[tuple[str, str]] = [
        (f"{e1['P_pure']['indic_share_by_scheduler_lane_tag'] * 100:.2f}%", "E1 arm P lane tag"),
        (f"{e1['M_mixed']['indic_share_by_scheduler_lane_tag'] * 100:.2f}%", "E1 arm M lane tag"),
        (f"{e1['M_mixed']['indic_share_by_actual_document_content'] * 100:.2f}%", "E1 arm M content"),
        (f"{perf['rank_compute_skew']['compute_imbalance_pct']:.1f}%", "rank compute skew"),
        (f"{mc['delivered_shares']['agentic'] * 100:.2f}%", "delivered agentic share"),
        (f"{nc['n_controls']} negative controls", "negative control count"),
    ]
    # The excess-loss table, values AND order. The sign comes from the format, not from a literal
    # `+`: that literal was written when every lane was above the reference, and it produced the
    # impossible expectation "indic +-0.4278" the moment AdamW pushed six of the seven below it.
    for lane, rec in sorted(xl.items(), key=lambda kv: -kv[1]["mean_excess_loss"]):
        want.append((f"{lane} {rec['mean_excess_loss']:+.4f}", f"excess loss {lane}"))

    missing = [f"{d} -> expected {s!r}" for s, d in want if s not in readme]
    assert not missing, "README figures no longer match the artifacts:\n  " + "\n  ".join(missing)


def test_role_codes_survive_every_packing_policy():
    """Role must reach the packed window intact under all six policies, not just the simple ones.

    The policies slice tokens and loss masks in six different ways - truncate, split across
    windows, first-fit bin, best-fit bin, concatenate-and-chop, one-document-per-window. Each is a
    separate opportunity to cut two of the three parallel arrays and leave the third at its old
    offset, which would produce a by-role table that sums correctly and attributes wrongly.

    Checked structurally rather than by eyeballing one policy's output: for every policy, every
    loss-bearing position carries a supervised role, every pad position carries the pad role, and
    the arrays are the same length.
    """
    from tds import packing as pk

    shard = next(s for s in _built.admitted_shards if s.spans)
    units = pk.units_from_shard(shard)
    seen_roles = set()
    for policy in pk.POLICIES:
        seqs = pk.pack(units, 128, policy, shard.lane)
        assert seqs, f"{policy} produced no sequences"
        for sq in seqs:
            ok, errs = pk.validate_masks(sq)
            assert ok, f"{policy}: {errs[:2]}"
            assert len(sq.role_codes) == len(sq.tokens) == len(sq.loss_mask)
            for i, t in enumerate(sq.tokens):
                if t == PAD:
                    assert sq.role_codes[i] == pk.PAD_ROLE, \
                        f"{policy}: pad at {i} carries role {sq.role_codes[i]}"
            seen_roles.update(sq.role_codes)
    # Non-vacuity: if every position were PAD_ROLE the loop above would pass while attributing
    # nothing. At least one real role must have survived packing.
    assert seen_roles - {pk.PAD_ROLE}, "no real role survived any packing policy"


def test_optimizer_state_is_real_state_and_the_schedule_is_reproducible():
    """AdamW's moments must change the update, and `lr_at` must reproduce the stored schedule.

    Before this, `optimizer_state` in a checkpoint was `{"type": "sgd", "lr": 0.35}` and
    `scheduler_state` was `{"policy": "constant"}` - both restatements of the config. No mutation
    could invalidate either, so "resume restores the optimizer and scheduler" was a sentence with
    no check under it.

    Two properties, and the second is the one that makes the first meaningful:
      1. two identical gradients applied from a fresh optimizer vs a warmed one give different
         updates (the moments carry history);
      2. dropping the moments leaves the WEIGHTS bit-identical, so a weight-hash comparison at the
         restore instant cannot detect the loss.
    """
    import numpy as np
    from tds.model import TinyLM

    # The gradients must VARY across steps. With a constant gradient Adam's normalised step
    # m_hat/sqrt(v_hat) converges to sign(g) whatever the history, and bias correction makes a
    # fresh optimizer's very first step equal to it exactly - so a constant-gradient probe would
    # report "the moments are not load-bearing" and be wrong about why. That is a property of
    # Adam, not of this implementation, and it is why the drill uses a varying sequence.
    rng = np.random.default_rng(7)
    proto = TinyLM.init(32, d_model=8, d_hidden=8, n_heads=2, seed=1)

    def grads(scale):
        return {k: (rng.standard_normal(getattr(proto, k).shape) * scale)
                for k in TinyLM.PARAMS}

    warmed = TinyLM.init(32, d_model=8, d_hidden=8, n_heads=2, seed=1)
    for i in range(5):
        warmed.apply(grads(0.01 * (i + 1)), 1e-3)
    warm_weights = {k: getattr(warmed, k).copy() for k in TinyLM.PARAMS}
    g = grads(0.02)

    # Same weights, same gradient, different moment history -> different update.
    a = TinyLM.init(32, d_model=8, d_hidden=8, n_heads=2, seed=1)
    b = TinyLM.init(32, d_model=8, d_hidden=8, n_heads=2, seed=1)
    for k, v in warm_weights.items():
        setattr(a, k, v.copy())
        setattr(b, k, v.copy())
    a.load_optimizer_state(warmed.optimizer_state())
    b.reset_optimizer_state()
    assert a.weight_hash == b.weight_hash, "the two probes must start from identical weights"
    a.apply({k: v.copy() for k, v in g.items()}, 1e-3)
    b.apply({k: v.copy() for k, v in g.items()}, 1e-3)
    assert a.weight_hash != b.weight_hash, \
        "dropping AdamW's moments left the next update unchanged - they are not load-bearing"

    # The schedule is a pure function of (step, config), so the stored `lr_at_step` in every
    # checkpoint must be exactly reproducible - which is why scheduler_state can be config+step.
    rig = harness.make_rig(_plan, _built, run_id="test-sched", seed=20260806,
                           global_batch=8, sequence_length=128, world_size=4)
    tr = rig.trainer
    ck = tr.save_checkpoint()
    sch = ck["scheduler_state"]
    assert sch["policy"] == "linear_warmup_cosine_decay", sch["policy"]
    assert tr.lr_at(sch["step"]) == sch["lr_at_step"], "lr_at does not reproduce the stored value"
    # warmup rises, and the tail never falls below the declared floor
    warm = max(1, int(sch["total_steps"] * sch["warmup_ratio"]))
    assert tr.lr_at(0) <= tr.lr_at(warm), "warmup does not increase the learning rate"
    floor = sch["base_lr"] * sch["min_lr_ratio"]
    assert tr.lr_at(sch["total_steps"]) >= floor - 1e-12, "cosine decay fell below its floor"


def test_every_negative_control_fires():
    results = controls.run_all(_built, _plan)
    failed = [r.name for r in results if not r.caught]
    assert not failed, f"controls that did not catch their mutation: {failed}"
    assert len(results) >= 15


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    bad = 0
    skipped = []
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Skip as s:
            skipped.append(name)
            print(f"  SKIP  {name} ({s})")
        except Exception:
            bad += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    ran = len(fns) - len(skipped)
    print(f"\n{ran - bad}/{ran} passed"
          + (f", {len(skipped)} skipped: {', '.join(skipped)}" if skipped else ""))
    sys.exit(1 if bad else 0)
