"""Unit tests for the invariants the demo relies on.

Deliberately not a re-run of `run_demo.py` in assert form. These test the properties that
would break silently -- determinism across processes, mask semantics, the epoch ceiling,
the carry that keeps a small lane alive -- because those are the ones where a wrong result
still produces a complete-looking run.

    .venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ledger.corpus import build_corpus
from ledger.determinism import (canonical_json, content_hash, derive_seed, rng,
                                stable_shuffle)
from ledger.documents import MASKED_ROLES, SPLIT_TRAIN, SUPERVISED_ROLES
from ledger.firewall import run_firewall
from ledger.masks import (ATTN_BLOCK_DIAGONAL, ATTN_EOS_ONLY, PAD_SEGMENT,
                          POS_RESET_PER_DOCUMENT, build_attention_mask, verify_masks)
from ledger.mixture import RunShape, _allocate, compile_schedule
from ledger.packing import LANE_POLICY, compare_policies, pack
from ledger.plan import EPOCH_CEILING, LANE_SPEC, Scale, verify_plan
from ledger.registry import build_registry
from ledger.tokenization import TokenizerSpec, expected_initial_loss, train_tokenizer


# ---------------------------------------------------------------------------
# fixtures -- built once, they are the expensive part
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def corpus():
    return build_corpus()


@pytest.fixture(scope="session")
def tokenizer(corpus):
    return train_tokenizer(corpus[0], TokenizerSpec(vocab_size=1024))


@pytest.fixture(scope="session")
def firewall(corpus):
    return run_firewall(corpus[0])


@pytest.fixture(scope="session")
def registry(firewall, tokenizer):
    return build_registry(firewall, tokenizer, root=None, verify_immutability=False)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_seed_derivation_is_stable_across_processes():
    """The property Python's own seeding does not have.

    `random.seed(42)` is stable inside one interpreter. This has to survive a fresh
    process, which is what a replay eighteen months later actually is.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import sys; sys.path.insert(0, {!r});"
            "from ledger.determinism import derive_seed;"
            "print(derive_seed('run', 'lane', 7))".format(here))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert int(out.stdout.strip()) == derive_seed("run", "lane", 7)


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert content_hash({"x": [1, 2]}) == content_hash({"x": [1, 2]})


def test_canonical_json_normalizes_negative_zero():
    """-0.0 and 0.0 compare equal but serialize differently, which would make two
    identical shards hash differently on different platforms."""
    assert canonical_json({"v": -0.0}) == canonical_json({"v": 0.0})


def test_stable_shuffle_is_a_permutation_and_reproducible():
    items = list(range(50))
    a = stable_shuffle(items, "key", 1)
    b = stable_shuffle(items, "key", 1)
    c = stable_shuffle(items, "key", 2)
    assert a == b
    assert a != c
    assert sorted(a) == items


def test_rng_streams_are_independent_per_key():
    x = rng("a").integers(0, 1 << 30, size=20)
    y = rng("b").integers(0, 1 << 30, size=20)
    assert not np.array_equal(x, y)
    assert np.array_equal(x, rng("a").integers(0, 1 << 30, size=20))


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def test_s5_plan_reproduces_without_problems():
    assert verify_plan() == []


def test_protected_lanes_declare_a_floor():
    for name, spec in LANE_SPEC.items():
        if spec.protected:
            assert spec.floor_share > 0, "{} is protected with no floor".format(name)


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------


def test_tokenizer_is_fitted_on_train_split_only(corpus, tokenizer):
    """A tokenizer that has seen held-out text has already leaked it."""
    docs, _ = corpus
    train_ids = sorted(d.doc_id for d in docs if d.split == SPLIT_TRAIN)
    assert tokenizer.training_fingerprint == content_hash(train_ids)
    # And the held-out documents are genuinely absent from what it was fitted on.
    assert len(train_ids) < len(docs)


def test_expected_initial_loss_is_ln_vocab(tokenizer):
    assert expected_initial_loss(tokenizer.vocab_size) == pytest.approx(
        math.log(tokenizer.vocab_size))


def test_eos_is_supervised_and_bos_is_not_emitted(corpus, tokenizer):
    docs, _ = corpus
    doc = next(d for d in docs if d.split == SPLIT_TRAIN)
    ids, mask, roles = tokenizer.encode_document(doc)
    assert roles[-1] == "eos"
    assert mask[-1] == 1, "EOS must carry loss -- it is the boundary the model must predict"
    assert "bos" not in roles, "BOS is deliberately not emitted"


def test_role_masking_matches_the_declared_policy(corpus, tokenizer):
    docs, _ = corpus
    for doc in docs[:40]:
        _ids, mask, roles = tokenizer.encode_document(doc)
        for m, r in zip(mask, roles):
            if r in MASKED_ROLES:
                assert m == 0, "{} carried loss".format(r)
            elif r in SUPERVISED_ROLES:
                assert m == 1, "{} did not carry loss".format(r)


# ---------------------------------------------------------------------------
# firewall
# ---------------------------------------------------------------------------


def test_firewall_catches_the_planted_pathologies(corpus, firewall):
    _docs, meta = corpus
    planted = meta["planted"]
    refused_ids = {d for d, _ in firewall.refused}
    dropped = ({a for a, _ in firewall.exact_duplicates}
               | {a for a, _, _ in firewall.near_duplicates})
    quarantined = {d for d, _ in firewall.quarantined}

    assert planted["planted_pii_doc"] in refused_ids
    assert planted["planted_restricted_licence_doc"] in refused_ids
    a, b = planted["planted_exact_duplicate_pair"]
    assert (a in dropped) != (b in dropped), "exactly one of the pair should be dropped"
    assert planted["planted_contaminated_doc"] in quarantined


def test_firewall_does_not_quarantine_the_corpus(corpus, firewall):
    """The regression that motivated the script-agnostic word splitter.

    Python's `re` does not match Unicode combining marks with `\\w`, so every Devanagari
    matra split a word and 1,286 of 1,626 documents were quarantined as contaminated. A
    firewall that rejects most of the corpus is indistinguishable from a broken one.
    """
    docs, _ = corpus
    assert len(firewall.quarantined) <= 3, firewall.quarantined[:5]
    assert len(firewall.admitted) > 0.98 * len(docs)


def test_boilerplate_survives_deduplication(corpus, firewall):
    """Near-dedup must not collapse legitimately similar documents (licence headers,
    boilerplate). Over-aggressive dedup silently deletes a lane."""
    _docs, meta = corpus
    expected = meta["planted"]["boilerplate_docs"]
    survivors = sum(1 for d in firewall.admitted if d.doc_id.startswith("boiler-"))
    assert survivors == expected, \
        "{} of {} boilerplate documents were deduplicated away".format(
            expected - survivors, expected)


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------


def test_shards_are_immutable_across_rebuilds(firewall, tokenizer):
    a = build_registry(firewall, tokenizer, root=None, verify_immutability=False)
    b = build_registry(firewall, tokenizer, root=None, verify_immutability=False)
    assert a.order == b.order
    assert a.summary()["registry_hash"] == b.summary()["registry_hash"]


def test_shard_id_embeds_its_content_hash(registry):
    for sid in registry.order[:20]:
        m = registry.manifest(sid)
        expected = content_hash([m.payload_hash, m.tokenizer_hash])[:10]
        assert sid.endswith(expected), sid
        # And the id carries the lane and split it claims, so a misfiled shard is visible
        # without opening the manifest.
        assert sid.startswith("shard-{}-{}-".format(m.lane, m.split))


def test_non_train_shards_are_never_trainable(registry):
    for sid in registry.order:
        m = registry.manifest(sid)
        if m.split != SPLIT_TRAIN:
            ok, reasons = m.is_trainable()
            assert not ok
            assert any("split" in r for r in reasons)


def test_shards_are_single_band(registry):
    """The band-gating mechanism depends on this: a mixed-band shard would be admitted
    into both stages and reintroduce filter-at-draw-time cost."""
    for sid in registry.order:
        assert len(registry.manifest(sid).difficulty_bands) == 1


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------


def test_packing_never_exceeds_the_window(registry, tokenizer):
    sid = max(registry.trainable_ids(), key=lambda s: registry.manifest(s).token_count)
    shard = registry.get(sid)
    for policy in ("greedy", "best_fit", "pad_only", "concat_chop",
                   "structure_preserving", "long_context"):
        plan = pack(shard, policy, 256, shuffle_key=("test", policy))
        for seq in plan.sequences:
            assert seq.seq_len == 256
            for pl in seq.placements:
                assert 0 <= pl.seq_start < pl.seq_end <= 256


def test_packing_is_deterministic(registry):
    sid = registry.trainable_ids()[0]
    shard = registry.get(sid)
    a = pack(shard, "greedy", 256, shuffle_key=("k", 1))
    b = pack(shard, "greedy", 256, shuffle_key=("k", 1))
    assert a.plan_hash == b.plan_hash


def test_structure_preserving_never_truncates(registry):
    """The property that justifies its cost: a reasoning chain cut in half teaches half
    a chain, so this policy defers rather than truncates."""
    for sid in registry.trainable_ids()[:25]:
        plan = pack(registry.get(sid), "structure_preserving", 256,
                    shuffle_key=("test", sid))
        assert plan.stats["truncated_tokens"] == 0, sid
        assert plan.stats["dropped_documents"] == 0, sid


def test_every_lane_has_a_declared_policy():
    for lane in LANE_SPEC:
        assert lane in LANE_POLICY


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------


def test_block_diagonal_mask_actually_blocks():
    """Verifying the label is not verifying the mask."""
    seg = np.array([0, 0, 0, 1, 1, PAD_SEGMENT], dtype=np.int32)
    m = build_attention_mask(seg, ATTN_BLOCK_DIAGONAL)
    assert not m[3, 2], "document 1 attended into document 0"
    assert m[4, 3], "document 1 could not attend to its own earlier token"
    assert not m[1, 2], "attention was not causal"
    assert not m[:5, 5].any(), "a real token attended to padding"
    assert not m[5, :5].any(), "padding attended to a real token"
    # The padding row keeps a single self-attending entry on purpose: a fully-masked row
    # produces NaN in softmax. Its loss is masked out, so it contributes nothing.
    assert m[5, 5]


def test_eos_only_mask_permits_crossing():
    seg = np.array([0, 0, 1, 1], dtype=np.int32)
    assert build_attention_mask(seg, ATTN_EOS_ONLY)[2, 1], \
        "eos-only policy should allow cross-document attention"


def test_padding_never_carries_loss_and_final_token_is_dropped(registry, tokenizer):
    from ledger.masks import LANE_ATTENTION, LANE_POSITION, materialize
    sid = registry.trainable_ids()[0]
    shard = registry.get(sid)
    lane = shard.manifest.lane
    plan = pack(shard, LANE_POLICY[lane], 256, shuffle_key=("test", sid))
    for seq in plan.sequences[:5]:
        m = materialize(seq, shard, tokenizer.pad_id,
                        attention_policy=LANE_ATTENTION[lane],
                        position_policy=LANE_POSITION[lane], lane=lane)
        assert m.loss_mask[m.segment_ids == PAD_SEGMENT].sum() == 0
        real = np.flatnonzero(m.segment_ids != PAD_SEGMENT)
        assert m.loss_mask[real[-1]] == 0, "final token has no successor to predict"
        assert verify_masks(m, shard, tokenizer.eos_id) == []


def test_reset_positions_restart_at_each_document(registry, tokenizer):
    from ledger.masks import materialize
    checked = 0
    for sid in registry.trainable_ids():
        shard = registry.get(sid)
        if shard.manifest.document_count < 2:
            continue
        plan = pack(shard, "best_fit", 256, shuffle_key=("test", sid))
        for seq in plan.sequences:
            if len(seq.placements) < 2:
                continue
            m = materialize(seq, shard, tokenizer.pad_id,
                            attention_policy=ATTN_BLOCK_DIAGONAL,
                            position_policy=POS_RESET_PER_DOCUMENT,
                            lane=shard.manifest.lane)
            for pl in seq.placements:
                # Each document restarts at its own offset, not at its position in the
                # window. Under continuous positions the second document would inherit
                # the first one's length as its starting index.
                assert m.position_ids[pl.seq_start] == pl.doc_start
                assert m.position_ids[pl.seq_end - 1] == pl.doc_end - 1
            checked += 1
            if checked >= 5:
                return
    assert checked, "no multi-document sequence found to verify position reset against"


# ---------------------------------------------------------------------------
# allocation and the ceiling
# ---------------------------------------------------------------------------


def test_carry_keeps_a_sub_integer_lane_alive():
    """The bug this is the regression test for: largest-remainder rounding floors a
    2%-of-16-sequences lane to zero every step, so a *protected* lane realized 0.2%
    against a 2.0% floor. A floor that integer rounding can delete is decorative."""
    weights = {"big": 0.98, "tiny": 0.02}
    carry = {}
    total = 0
    for _ in range(100):
        alloc = _allocate(weights, 16, carry)
        total += alloc["tiny"]
    assert total >= 25, "tiny lane got {} of an expected ~32 sequences".format(total)


def test_allocation_conserves_sequences():
    carry = {}
    for _ in range(20):
        alloc = _allocate({"a": 0.5, "b": 0.3, "c": 0.2}, 16, carry)
        assert sum(alloc.values()) == 16


def test_epoch_ceiling_is_never_breached(registry):
    shape = RunShape(total_tokens=120_000)
    sched = compile_schedule(registry, shape, Scale(factor=3.0e12 / 120_000))
    for lane, epochs in sched.report["lane_epochs_demo"].items():
        assert epochs <= EPOCH_CEILING + 1e-6, \
            "{} used {} epochs against a ceiling of {}".format(lane, epochs, EPOCH_CEILING)


def test_schedule_is_deterministic(registry):
    shape = RunShape(total_tokens=120_000)
    scale = Scale(factor=3.0e12 / 120_000)
    a = compile_schedule(registry, shape, scale)
    b = compile_schedule(registry, shape, scale)
    assert a.schedule_hash == b.schedule_hash


def test_band_gate_admits_only_shards_in_the_stage_range(registry):
    from ledger.plan import STAGES
    for stage in STAGES:
        for lane in LANE_SPEC:
            for sid in registry.eligible(lane, stage, 256):
                band = registry.manifest(sid).difficulty_bands[0]
                assert band in stage.band_range, \
                    "{} ({}) reachable in {}".format(sid, band, stage.name)


# ---------------------------------------------------------------------------
# batch identity, spans and the evidence bundle
# ---------------------------------------------------------------------------


def test_batch_spans_address_immutable_shard_coordinates(registry, tokenizer):
    """A span must be a permanent address: shard + document + offset range.

    This is what makes replay a lookup instead of a re-derivation, so the coordinates have
    to resolve back to exactly the tokens that were consumed.
    """
    from ledger.mixture import RunShape, compile_schedule
    from ledger.stream import DataStream
    from ledger.trainer import batch_token_spans

    shape = RunShape(total_tokens=60_000)
    sched = compile_schedule(registry, shape, Scale(factor=3.0e12 / 60_000))
    stream = DataStream(registry, sched, tokenizer, run_id="span-test")
    batch = stream.batch(0)
    spans = batch_token_spans(batch)
    assert spans

    for sp in spans:
        shard = registry.get(sp["shard_id"])
        rec = next(d for d in shard.documents if d.doc_id == sp["doc_id"])
        # Resolve the span independently, straight out of the shard.
        lo = rec.start + sp["doc_start"]
        hi = rec.start + sp["doc_end"]
        assert hi - lo == sp["tokens"]
        assert content_hash(shard.tokens[lo:hi].tolist())[:16] == sp["token_hash"], sp


def test_batch_id_changes_when_content_changes(registry, tokenizer):
    """Two different steps must not collide on batch id."""
    from ledger.mixture import RunShape, compile_schedule
    from ledger.stream import DataStream
    from ledger.trainer import batch_id

    shape = RunShape(total_tokens=60_000)
    sched = compile_schedule(registry, shape, Scale(factor=3.0e12 / 60_000))
    stream = DataStream(registry, sched, tokenizer, run_id="id-test")
    ids = [batch_id(stream.batch(i)) for i in range(4)]
    assert len(set(ids)) == len(ids), ids


def test_event_log_verdicts_are_computed_not_asserted(tmp_path):
    """`check()` must derive PASS/FAIL from the condition it is given.

    The whole evidence bundle rests on this: if a caller could stamp PASS unconditionally,
    the log would be decoration rather than evidence.
    """
    from ledger.events import FAIL, PASS, EventLog

    log = EventLog(str(tmp_path / "run.log"), echo=False)
    log.check("truthy", 1 == 1)
    log.check("falsy", 1 == 2)
    assert log.events[0].level == PASS
    assert log.events[1].level == FAIL
    summary = log.close()
    assert summary["pass"] == 1 and summary["fail"] == 1
    assert summary["all_passed"] is False


def test_event_log_reports_missing_required_events(tmp_path):
    from ledger.events import REQUIRED_EVENTS, EventLog

    log = EventLog(str(tmp_path / "run.log"), echo=False)
    log.check(REQUIRED_EVENTS[0], True)
    missing = log.missing_required()
    assert REQUIRED_EVENTS[0] not in missing
    assert len(missing) == len(REQUIRED_EVENTS) - 1


def test_evidence_predicates_fail_on_missing_artifacts(tmp_path):
    """An empty bundle must fail every requirement, not pass by default.

    Guards the failure mode that would matter most: an evidence generator that reports PASS
    because it could not find the file that would have said otherwise.
    """
    from ledger.evidence import evaluate

    result = evaluate(str(tmp_path))
    assert result["total"] > 0
    assert result["passed"] == 0
    assert all(r["result"] == "FAIL" for r in result["requirements"])
    assert all("missing" in r.get("error", "") for r in result["requirements"])


def test_evidence_predicate_fails_on_false_value(tmp_path):
    """A predicate reading an explicit False must produce FAIL, not an error."""
    import json as _json

    from ledger.evidence import evaluate

    d = tmp_path / "ledgers"
    d.mkdir()
    (d / "replay.json").write_text(_json.dumps({"all_match": False}))
    rows = {r["key"]: r for r in evaluate(str(tmp_path))["requirements"]}
    assert rows["replay"]["result"] == "FAIL"
    assert rows["replay"]["observed_value"] is False
    assert "error" not in rows["replay"]


def test_perf_totals_reconstruct_from_the_ledger():
    """Throughput must be re-derivable from the consumption ledger, or it is not credited."""
    from ledger.perf import PerfRecorder, verify_against_ledger

    class _Seq:
        seq_len = 256
        real_tokens = 200
        supervised_tokens = 120
        doc_ids = ["d1"]

    class _S:
        seq = _Seq()

    class _Cons:
        steps = [{"tokens": 512, "supervised_tokens": 240}]

    perf = PerfRecorder()
    perf.start_step()
    perf.end_step(step=0, stage="S1", samples=[_S(), _S()])
    check = verify_against_ledger(perf, _Cons())
    assert check["reconstructible"], check
    rep = perf.report(consumption=_Cons())
    # Loss-bearing throughput must be strictly below window throughput whenever padding or
    # masked context exists -- the distinction the report is for.
    assert rep["overall"]["loss_bearing_tokens_per_s"] < rep["overall"]["window_tokens_per_s"]
    assert rep["efficiency_decomposition"]["lost_to_padding"] == 112
    assert rep["efficiency_decomposition"]["lost_to_masked_context"] == 160
