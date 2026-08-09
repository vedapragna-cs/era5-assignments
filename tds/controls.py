"""Negative controls: deliberately break each invariant and require the verifier to catch it.

A green test suite is consistent with the checks being vacuous. This is not a hypothetical concern -
the S5 widget teardown found exactly that bug class, where a state check compared against the wrong
CSS class name, was silently false-always, and inverted an entire swept result table before anyone
noticed.

So every invariant this system claims gets a paired control that breaks it. A control PASSES when the
verifier REJECTS the broken input. A control that does not trigger a rejection is reported as a
failure of the control, not as a success of the system.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .canon import HashChain, chash
from .clean import run_cleaning
from .corpus import Document, Segment, load_corpus
from .firewall import Firewall
from .mixture import MixturePlan
from .packing import (PackedSequence, attention_crossings, blocked_cross_segment_pairs,
                      validate_masks)
from .shards import admit, verify_shard_integrity
from .tokenizer import PAD, train_tokenizer


@dataclass
class ControlResult:
    name: str
    invariant: str
    mutation: str
    caught: bool
    detail: str

    def to_record(self) -> dict:
        return {"control": self.name, "invariant": self.invariant, "mutation": self.mutation,
                "caught_by_verifier": self.caught, "detail": self.detail,
                "result": "PASS" if self.caught else "FAIL"}


def run_all(built, plan: MixturePlan) -> list[ControlResult]:
    out: list[ControlResult] = []
    from . import harness as _harness
    import copy as _copy

    # ---------------------------------------------------------------- tokenizer frozen
    tok = built.tokenizer
    mutated = replace(tok, merges=tok.merges[:-1] if tok.merges else ())
    out.append(ControlResult(
        "tokenizer_hash_detects_merge_change", "tokenizer is frozen and hashed",
        "drop the last merge rule and re-hash",
        mutated.hash != tok.hash,
        f"original {tok.hash[:12]} vs mutated {mutated.hash[:12]}"))

    # ------------------------------------------------- exactly-once coverage must be able to FAIL
    # Two previous versions of coverage() were vacuous, and the second shipped a comment claiming it
    # was fixed. This control exists so that claim is never again made in prose alone.
    from .exactly_once import DeliveryTrace as _DT, compare as _cmp
    _mk = lambda ps, st: _DT("t", 4, ps, st)
    _good = _mk([[f"s{i*8+j}" for j in range(8)] for i in range(4)], [0, 1, 2, 3])
    _dup = _mk(_good.per_step + [_good.per_step[-1]], [0, 1, 2, 3, 3])
    _skip = _mk([_good.per_step[0], _good.per_step[1], _good.per_step[3]], [0, 1, 3])
    _cg = _cmp(_good, _good)["coverage_run_b"]
    _cd = _cmp(_good, _dup)["coverage_run_b"]
    _cs = _cmp(_good, _skip)["coverage_run_b"]
    out.append(ControlResult(
        "exactly_once_coverage_detects_duplicate_and_missing_batches",
        "the delivered step sequence has no repeats and no gaps",
        "deliver one batch twice, and separately drop one batch, then recompute coverage",
        (_cg["duplicate_positions"] == 0 and _cg["missing_positions"] == 0
         and _cd["duplicate_positions"] > 0 and _cs["missing_positions"] > 0),
        f"clean dup={_cg['duplicate_positions']}/missing={_cg['missing_positions']}; "
        f"duplicated batch dup={_cd['duplicate_positions']}; "
        f"dropped batch missing={_cs['missing_positions']}"))

    # ------------------------------------------------- replay must be able to FAIL
    # The previous replay built its sample list by iterating the very record it then compared
    # against, so `sample_ids_match` was a list compared to itself. An audit showed it reporting
    # all_match=True after every pool cursor was scrambled. Replay now re-derives from a checkpoint,
    # which means perturbing the loader state must break it - and that is what this asserts.
    _rig2 = _harness.make_rig(plan, built, run_id="control-replay", seed=20260806,
                              global_batch=8, sequence_length=128, world_size=4)
    _t2 = _rig2.trainer
    _t2.save_checkpoint()
    for _ in range(6):
        _t2.train_step()
    _clean = _t2.replay_interval(2, 6)
    # Perturb the POOL ORDER, not the cursors: replay restores cursors from the checkpoint (as it
    # must), so cursor state is correctly irrelevant to it. Pool order is not checkpointed, so
    # rotating it changes what the planner draws - if replay were still re-reading the ledger this
    # would have no effect, which is precisely what the old implementation did.
    _orders = {ln: list(pl.order) for ln, pl in _t2.planner.pools.items()}
    for _ln, _pl in _t2.planner.pools.items():           # every lane, so the interval must move
        _o = _orders[_ln]
        _pl.order = _o[1:] + _o[:1]
    _perturbed = _t2.replay_interval(2, 6)
    for _ln, _pl in _t2.planner.pools.items():
        _pl.order = _orders[_ln]
    _restored = _t2.replay_interval(2, 6)
    out.append(ControlResult(
        "replay_fails_when_the_stream_is_perturbed",
        "replay re-derives the stream rather than re-reading the record it checks",
        "rotate every lane's pool order by one and replay the same interval",
        bool(_clean["all_match"]) and not _perturbed["all_match"] and bool(_restored["all_match"]),
        f"clean all_match={_clean['all_match']}; rotated pool order all_match="
        f"{_perturbed['all_match']}; restored all_match={_restored['all_match']}"))

    # ------------------------------------------------- attention isolation, falsified properly
    # The previous version of attention_crossings() could not fail: it built the mask from
    # (seg[j] == seg[i]) and counted entries where (seg[j] != seg[i]). An independent audit removed
    # segment isolation from the trainer entirely and the whole graded pipeline still reported
    # 15/15 PASS. So the control now MUTATES THE MODEL'S OWN MASK and requires the count to rise.
    import numpy as _np
    from . import model as _model
    from .packing import attention_crossings as _ac, PackedSequence as _PS

    _probe = _PS(tokens=[1] * 6, loss_mask=[1] * 6, segment_ids=[0, 0, 0, 1, 1, 1],
                 position_ids=[0, 1, 2, 0, 1, 2], members=[], span_ids=[], lane="x", policy="p")
    _before = _ac(_probe)
    # Capture the DESCRIPTOR, not the function. `TinyLM.attention_mask` on a staticmethod returns
    # the plain function, and assigning that back makes it an instance method - every later call
    # then passes `self` and dies with "takes 1 positional argument but 2 were given". The control
    # would have left the model broken for everything that ran after it.
    _orig_mask = _model.TinyLM.__dict__["attention_mask"]
    try:
        _model.TinyLM.attention_mask = staticmethod(
            lambda seg: _np.arange(len(seg))[:, None] >= _np.arange(len(seg))[None, :])
        _after = _ac(_probe)
    finally:
        setattr(_model.TinyLM, "attention_mask", _orig_mask)
    _restored = _ac(_probe)
    out.append(ControlResult(
        "attention_crossings_detects_removed_segment_isolation",
        "cross-document attention is counted against the mask the MODEL applies",
        "replace TinyLM.attention_mask with a causal-only mask and recount",
        _before == 0 and _after > 0 and _restored == 0,
        f"crossings with isolation={_before}, with isolation removed={_after}, "
        f"after restore={_restored}"))

    # ------------------------------------------------- checkpoint integrity (torn-write drill)
    # A checkpoint that loads but is subtly wrong is worse than one that fails to load: the model
    # resumes happily from the wrong position in the data stream and nothing downstream notices.
    # So every field a resume depends on gets its own corruption flavour, and each must be caught.
    from .trainer import Trainer as _Tr

    _rig = _harness.make_rig(plan, built, run_id="control-ckpt", seed=20260806,
                             global_batch=8, sequence_length=128, world_size=4)
    _tr = _rig.trainer
    _ck = _tr.save_checkpoint()
    _base_ok, _ = _Tr.verify_checkpoint(_ck)

    _flavours = {
        "weight_bit_flip": lambda c: c["model_state"]["E"][0].__setitem__(
            0, c["model_state"]["E"][0][0] + 1e-3),
        "torn_write_truncated_weights": lambda c: c["model_state"].__setitem__(
            "E", c["model_state"]["E"][:1]),
        "loader_cursor_moved": lambda c: c["dataloader_state"].update(
            {"global_sample_index": c["dataloader_state"]["global_sample_index"] + 1}),
        "ledger_offset_moved": lambda c: c.update({"ledger_offset": c["ledger_offset"] + 1}),
        "step_counter_moved": lambda c: c.update({"global_step": c["global_step"] + 1}),
        "tokenizer_identity_swapped": lambda c: c.update({"tokenizer_hash": "0" * 64}),
    }
    _caught = {}
    for _name, _break in _flavours.items():
        _bad = _copy.deepcopy(_ck)
        _break(_bad)
        _ok, _why = _Tr.verify_checkpoint(_bad)
        _caught[_name] = (not _ok)
    out.append(ControlResult(
        "checkpoint_integrity_detects_every_corruption_flavour",
        "a checkpoint that fails integrity never reaches the model",
        f"induce {len(_flavours)} corruption flavours: {', '.join(_flavours)}",
        _base_ok and all(_caught.values()),
        f"clean checkpoint verifies={_base_ok}; caught " + ", ".join(
            f"{k}={v}" for k, v in _caught.items())))

    # And the consequence, not just the detection: restore() must REFUSE, not warn.
    _bad = _copy.deepcopy(_ck)
    _bad["dataloader_state"]["global_sample_index"] += 1
    _tr.checkpoints["ckpt_corrupt"] = _bad
    try:
        _tr.restore("ckpt_corrupt")
        _refused = False
    except AssertionError:
        _refused = True
    finally:
        _tr.checkpoints.pop("ckpt_corrupt", None)
    out.append(ControlResult(
        "restore_refuses_a_corrupt_checkpoint",
        "integrity failure blocks the restore rather than being logged",
        "move the loader cursor by one sample and call restore()",
        _refused,
        "restore() raised AssertionError" if _refused else
        "restore() ACCEPTED a checkpoint whose loader cursor was moved - the run would have "
        "silently resumed from the wrong data position"))

    # ------------------------------------------------- loss-spike detection and attribution
    from . import spike as _spike

    # 1. The detector must fire on an injected spike and stay silent on a flat series. A detector
    #    that never fires and a detector that always fires are equally useless.
    flat = [{"step": i, "mean_loss": 6.0 + 0.001 * (i % 3), "branch": "run-a"} for i in range(20)]
    spiked = [dict(r) for r in flat]
    spiked[14]["mean_loss"] = 9.0
    n_flat = len(_spike.detect(flat, branch="run-a"))
    hits = _spike.detect(spiked, branch="run-a")
    out.append(ControlResult(
        "spike_detector_fires_only_on_a_real_jump",
        "loss spikes are detected relative to the run's own trailing baseline",
        "inject one 9.0 loss into an otherwise flat 6.0 series, and separately check the flat series",
        n_flat == 0 and len(hits) == 1 and hits[0].global_step == 14,
        f"flat series -> {n_flat} detections; injected series -> "
        f"{[(h.global_step, round(h.z, 1)) for h in hits]}"))

    # 2. THE bug this guards against: a shard must never be labelled `harmful` merely because it was
    #    consumed at a spiking step. PaLM's ablation showed a spike is usually the interaction of a
    #    batch with a parameter state, so the label would be permanently wrong and undetectable
    #    downstream.
    from .learning import LearningLedger as _LL
    probe = _LL(tier="full")
    probe.shard_exposures = [
        {"shard_id": "sh-spiking", "global_step": 14, "usefulness": "harmful",
         "attribution_uncertain": False},
        {"shard_id": "sh-quiet", "global_step": 3, "usefulness": "harmful",
         "attribution_uncertain": False},
    ]
    res = probe.apply_spike_attribution(_spike.spike_windows(hits), {14: "interaction_not_data"})
    spiking = next(r for r in probe.shard_exposures if r["shard_id"] == "sh-spiking")
    quiet = next(r for r in probe.shard_exposures if r["shard_id"] == "sh-quiet")
    out.append(ControlResult(
        "spike_never_becomes_a_shard_verdict",
        "a shard consumed during a spike has its verdict withheld, not guessed",
        "classify two shards `harmful`, one inside a spike window and one outside, then apply "
        "spike attribution",
        spiking["usefulness"] == "unattributed" and spiking["attribution_uncertain"]
        and quiet["usefulness"] == "harmful" and not quiet["attribution_uncertain"],
        f"in-window shard -> {spiking['usefulness']} (was {spiking.get('usefulness_withheld')}); "
        f"out-of-window shard -> {quiet['usefulness']}; withheld={res['verdicts_withheld']}"))

    # 3. The withholding must be REVERSIBLE, or it is just censorship: if the ablation reproduces
    #    the spike away from a seam, the data is a legitimate suspect and the verdict stands.
    probe2 = _LL(tier="full")
    probe2.shard_exposures = [
        {"shard_id": "sh-implicated", "global_step": 14, "usefulness": "harmful",
         "attribution_uncertain": False}]
    res2 = probe2.apply_spike_attribution(_spike.spike_windows(hits), {14: "data_implicated"})
    upheld = probe2.shard_exposures[0]
    out.append(ControlResult(
        "ablation_can_restore_a_withheld_verdict",
        "withholding is resolvable by PaLM's ablation, not permanent",
        "re-run attribution with the ablation verdict `data_implicated`",
        upheld["usefulness"] == "harmful" and not upheld["attribution_uncertain"]
        and res2["verdicts_upheld_after_ablation"] == ["sh-implicated"],
        f"verdict={upheld['usefulness']}, upheld={res2['verdicts_upheld_after_ablation']}"))

    # ------------------------------------------------- n-gram containment decontamination
    # GPT-3 filtered on 13-gram overlap; Llama and open replications use 8-13 token windows. The
    # control is not "does the firewall block it" - it is "is this gate the ONLY thing that blocks
    # it", because a gate that is redundant with an existing one buys nothing.
    from .firewall import Firewall as _FW, NGRAM_CONTAINMENT_N
    from .corpus import ALL_DOCUMENTS as _DOCS

    _target = next((d for d in _DOCS if d.doc_id == "web-0102"), None)
    if _target is not None:
        _full = _FW.from_corpus(_DOCS)
        _blocked_by_all = not _full.check(_target).allowed
        _solo = {}
        for _g in sorted(_full.gates):
            _fw = _FW.from_corpus(_DOCS)
            for _k in _fw.gates:
                _fw.gates[_k] = (_k == _g)
            _solo[_g] = not _fw.check(_target).allowed
        _only = [g for g, caught in _solo.items() if caught]
        out.append(ControlResult(
            "ngram_containment_catches_what_jaccard_cannot",
            f"a benchmark passage quoted inside a long host is caught by "
            f"{NGRAM_CONTAINMENT_N}-gram containment",
            "plant a long ordinary article containing one verbatim benchmark item, with no canary, "
            "not marked derived, and global Jaccard below the overlap threshold",
            _blocked_by_all and _only == ["ngram_containment"],
            f"jaccard={_full.check(_target).max_overlap:.4f} < overlap_threshold; "
            f"gates that block it in isolation: {_only}"))

    # ------------------------------------------------- repeated n-grams: curation + trainer mask
    # OLMo 2 arXiv:2501.00656 s3.1. Two independent interventions, so two controls: the curation
    # filter must catch the planted document, and the trainer-side loss mask must still catch it if
    # curation is disabled. A safeguard that has never been shown to fire is not a safeguard.
    from .clean import find_repeated_ngrams, REPEAT_NGRAM_RUN
    from .packing import mask_repeated_ngrams

    planted = [r for r in built.clean_records.values() if r.repeat_ngram_status != "clean"]
    out.append(ControlResult(
        "repeat_ngram_screen_catches_planted_degenerate_document",
        "documents with long repeated n-gram runs are excluded at curation",
        "plant a unique document containing a 60x repeated token run",
        len(planted) == 1 and planted[0].doc_id == "web-0008",
        f"flagged {[r.doc_id for r in planted]} "
        f"(run={planted[0].repeat_ngram_detail['max_repeat_run']} >= {REPEAT_NGRAM_RUN})"
        if planted else "nothing flagged"))

    # The planted document is UNIQUE, so this proves the screen is not just dedup wearing a hat.
    out.append(ControlResult(
        "repeat_ngram_screen_is_not_subsumed_by_dedup",
        "the repeated-n-gram screen catches what similarity-based dedup cannot",
        "check the dedup verdict on the same planted document",
        bool(planted) and planted[0].dedup_status == "unique",
        f"dedup_status={planted[0].dedup_status} while repeat screen flagged it"
        if planted else "no planted document"))

    # Trainer-side safeguard, exercised directly: if curation were bypassed, the loss mask must
    # still refuse to train on the degenerate span.
    degenerate = [7] * 40 + [1, 2, 3]
    before = [1] * len(degenerate)
    after, n_masked = mask_repeated_ngrams(degenerate, before)
    clean_seq = list(range(40))
    _, n_clean = mask_repeated_ngrams(clean_seq, [1] * len(clean_seq))
    out.append(ControlResult(
        "trainer_side_repeat_mask_fires_when_curation_is_bypassed",
        "the trainer masks repeated-n-gram spans even if they reach the loader",
        "feed a 40x repeated token run straight to the packer's mask step",
        n_masked == 40 and sum(after) == 3 and n_clean == 0,
        f"masked {n_masked} of {len(degenerate)} positions, {sum(after)} left loss-bearing; "
        f"a non-degenerate sequence of the same length masks {n_clean}"))

    # ---------------------------------------------------------------- tokenizer determinism
    texts = [built.clean_records[d.doc_id].text for d in built.documents if d.split == "train"][:12]
    a = train_tokenizer(texts, vocab_size=256)
    b = train_tokenizer(list(reversed(texts))[::-1], vocab_size=256)
    out.append(ControlResult(
        "tokenizer_retrains_identically", "same inputs produce the same tokenizer",
        "retrain from the same texts and compare hashes",
        a.hash == b.hash,
        f"{a.hash[:12]} vs {b.hash[:12]} (this control PASSES when they MATCH)"))

    # ---------------------------------------------------------------- shard immutability
    if built.admitted_shards:
        sh = built.admitted_shards[0]
        ok_before, _ = verify_shard_integrity(sh)
        import copy
        broken = copy.deepcopy(sh)
        broken.tokens[0] = (broken.tokens[0] + 1) % 256
        ok_after, msg = verify_shard_integrity(broken)
        out.append(ControlResult(
            "shard_content_hash_detects_byte_flip", "shard identity is its token bytes",
            "flip one token id and re-verify against the manifest",
            ok_before and not ok_after, msg))

    # ---------------------------------------------------------------- admission gate
    if built.admitted_shards:
        m = dict(built.admitted_shards[0].manifest)
        assert admit(m).admitted, "baseline manifest should be admitted"
        for field_name in ("pii_screen_status", "dedup_status", "cleaning_pipeline_hash",
                           "tokenizer_hash", "eval_overlap_status"):
            broken_m = dict(m)
            broken_m[field_name] = "missing"
            r = admit(broken_m)
            out.append(ControlResult(
                f"admission_blocks_missing_{field_name}",
                "all four S4 fields are hard blockers",
                f"set {field_name}=missing",
                not r.admitted, f"reasons={r.blocking_reasons}"))
        broken_m = dict(m, license_tier="unsafe")
        r = admit(broken_m)
        out.append(ControlResult(
            "admission_blocks_unsafe_license", "unsafe license blocks training",
            "set license_tier=unsafe", not r.admitted, f"reasons={r.blocking_reasons}"))

    # ---------------------------------------------------------------- eval firewall
    fw = Firewall.from_corpus(load_corpus())
    test_doc = next(d for d in load_corpus() if d.split == "test")
    smuggled = Document(
        doc_id="smuggled-001", lane="web", source_id="attacker", license_tier="safe",
        language=test_doc.language, script=test_doc.script, difficulty_band="B2", split="train",
        segments=test_doc.segments)
    d = fw.check(smuggled)
    out.append(ControlResult(
        "firewall_blocks_smuggled_eval_doc", "eval data cannot enter training",
        "relabel a registered test document as split=train and resubmit",
        not d.allowed, f"reasons={d.reasons}"))

    ok, bad = fw.assert_no_eval_in_batch([test_doc.doc_id, "web-0001"])
    out.append(ControlResult(
        "firewall_blocks_eval_in_served_batch", "per-batch check is a second line of defence",
        "put a registered test doc id into a served batch",
        (not ok) and bad == [test_doc.doc_id], f"flagged={bad}"))

    # single-gate leak test: derived content must still be caught when overlap is disabled
    fw2 = Firewall.from_corpus(load_corpus())
    fw2.gates["overlap"] = False
    derived = next(d for d in load_corpus() if d.derived_from_benchmark)
    dd = fw2.check(derived)
    out.append(ControlResult(
        "firewall_derived_gate_independent", "derived content is caught without the overlap gate",
        "disable the overlap gate and submit a benchmark paraphrase",
        not dd.allowed, f"reasons={dd.reasons}"))

    # and the converse: with the content gates off it must leak, proving they are load-bearing.
    # ngram_containment is disabled here too - not because it is weak, but because it is strong: it
    # independently catches any document that reuses benchmark wording verbatim, so leaving it on
    # would mask whether `overlap` and `derived` do anything at all.
    fw3 = Firewall.from_corpus(load_corpus())
    fw3.gates["overlap"] = False
    fw3.gates["derived"] = False
    fw3.gates["ngram_containment"] = False
    dd3 = fw3.check(derived)
    out.append(ControlResult(
        "firewall_leaks_when_content_gates_off", "the content gates are genuinely load-bearing",
        "disable overlap, derived AND ngram_containment, then submit the paraphrase",
        dd3.allowed,
        "control PASSES when the doc LEAKS - proving none of the three is redundant "
        f"(allowed={dd3.allowed})"))

    # ---------------------------------------------------------------- mask correctness
    if built.samples:
        seq = next(iter(built.samples.values()))
        ok0, errs0 = validate_masks(seq)
        broken_seq = PackedSequence(
            tokens=list(seq.tokens), loss_mask=list(seq.loss_mask),
            segment_ids=list(seq.segment_ids), position_ids=list(seq.position_ids),
            members=list(seq.members), span_ids=list(seq.span_ids), lane=seq.lane,
            policy=seq.policy)
        # Pick a sample that actually CONTAINS padding. The previous version took the first sample
        # in the dict, which has none, so the whole control sat inside `if pad_idx is not None` and
        # silently never ran - and `summarize()` could not tell a skipped control from a passing
        # one. An audit found it absent from negative_controls.json while the bundle still reported
        # "all_invariants_are_falsifiable": true.
        padded = next((sq for sq in built.samples.values() if PAD in sq.tokens), None)
        if padded is None:
            out.append(ControlResult(
                "mask_validator_detects_loss_bearing_pad", "padding is never loss-bearing",
                "set loss_mask=1 on a pad position",
                False, "NO SAMPLE IN THIS CORPUS CONTAINS PADDING - the control could not run, "
                       "which is reported as a failure rather than silently skipped"))
        else:
            broken_seq = PackedSequence(
                tokens=list(padded.tokens), loss_mask=list(padded.loss_mask),
                segment_ids=list(padded.segment_ids), position_ids=list(padded.position_ids),
                members=list(padded.members), span_ids=list(padded.span_ids), lane=padded.lane,
                policy=padded.policy)
            ok_before, _ = validate_masks(broken_seq)
            pad_idx = next(i for i, t in enumerate(broken_seq.tokens) if t == PAD)
            broken_seq.loss_mask[pad_idx] = 1
            okb, errsb = validate_masks(broken_seq)
            out.append(ControlResult(
                "mask_validator_detects_loss_bearing_pad", "padding is never loss-bearing",
                f"set loss_mask=1 on pad position {pad_idx} of a sample that has padding",
                ok_before and not okb, f"clean={ok_before}, after mutation={okb}, "
                                       f"errors={errsb[:2]}"))

        # attention isolation. Two things must both hold, or the claim is empty:
        #   (a) the real mask allows zero cross-document pairs, and
        #   (b) the mask is actually doing work - i.e. a plain causal mask WOULD have allowed many.
        # (b) is the part that makes (a) meaningful. On a single-document window (a) is trivially
        # true and proves nothing.
        multi = next((s for s in built.samples.values()
                      if len(set(x for x in s.segment_ids if x >= 0)) > 1), None)
        if multi is not None:
            allowed = attention_crossings(multi)
            blocked = blocked_cross_segment_pairs(multi)
            n_segs = len(set(x for x in multi.segment_ids if x >= 0))
            out.append(ControlResult(
                "attention_isolation_is_real_and_non_vacuous",
                "no token attends across a document boundary, and the mask is load-bearing",
                "count pairs the real mask allows vs pairs a plain causal mask would have allowed",
                allowed == 0 and blocked > 0,
                f"{n_segs} documents in the window; real mask allows {allowed} cross-document "
                f"pairs; segment isolation blocks {blocked} pairs a causal-only mask would have "
                f"permitted"))

    # ---------------------------------------------------------------- protected floor
    # Two properties, and the second is what an earlier version of effective_mixture got wrong:
    # it raised protected lanes to their floor and then renormalised the WHOLE vector, which scaled
    # them straight back under it. A floor that is applied and then silently undone is worse than
    # no floor, because the plan says it is protected.
    floors = plan.protected_floors
    stage0 = plan.stages[0]
    mix0 = plan.effective_mixture(stage0.token_start)
    holds = all(mix0.get(ln, 0.0) >= f - 1e-9 for ln, f in floors.items())
    sums_to_one = abs(sum(mix0.values()) - 1.0) < 1e-9
    raised = {ln: (round(stage0.mixture.get(ln, 0.0), 4), round(mix0.get(ln, 0.0), 4))
              for ln in floors}
    out.append(ControlResult(
        "floors_survive_normalisation", "protected floors hold after the mixture is normalised",
        "read the effective mixture at stage 0 and compare each protected lane against its floor",
        holds and sums_to_one,
        f"raw->effective per protected lane {raised}; floors={floors}; "
        f"mixture sums to {round(sum(mix0.values()), 9)}"))

    # and the converse: a lane with NO floor must be free to fall below any given level, or the
    # "floor" is indistinguishable from a global minimum applied to everything.
    unprotected = [ln for ln in plan.lanes if ln not in floors]
    below = [ln for ln in unprotected if mix0.get(ln, 0.0) < min(floors.values())]
    out.append(ControlResult(
        "unprotected_lanes_are_not_floored", "the floor is selective, not a global minimum",
        "check that at least one unprotected lane sits below the smallest protected floor",
        bool(below),
        f"unprotected lanes below {min(floors.values())}: "
        f"{ {ln: round(mix0.get(ln, 0.0), 4) for ln in below} }"))

    # ---------------------------------------------------------------- ledger tamper evidence
    chain = HashChain()
    for i in range(5):
        chain.append({"event": "batch_committed", "global_step": i})
    ok_chain, _, msg_ok = chain.verify()
    chain.entries[2]["global_step"] = 999
    bad_chain, idx, msg_bad = chain.verify()
    out.append(ControlResult(
        "ledger_chain_detects_edit", "the ledger is tamper-evident",
        "edit global_step on historical entry #2",
        ok_chain and not bad_chain and idx == 2, f"{msg_ok} -> {msg_bad}"))

    return out


# Every control this module is expected to produce. A control that silently fails to run is
# indistinguishable from a passing one unless the expected set is declared somewhere - which is how
# `mask_validator_detects_loss_bearing_pad` went missing while the bundle reported
# "all_invariants_are_falsifiable": true.
EXPECTED_CONTROLS = {
    "tokenizer_hash_detects_merge_change",
    "tokenizer_retrains_identically",
    "shard_content_hash_detects_byte_flip",
    "mask_validator_detects_loss_bearing_pad",
    "attention_crossings_detects_removed_segment_isolation",
    "replay_fails_when_the_stream_is_perturbed",
    "exactly_once_coverage_detects_duplicate_and_missing_batches",
    "checkpoint_integrity_detects_every_corruption_flavour",
    "restore_refuses_a_corrupt_checkpoint",
    "spike_detector_fires_only_on_a_real_jump",
    "spike_never_becomes_a_shard_verdict",
    "ablation_can_restore_a_withheld_verdict",
    "ngram_containment_catches_what_jaccard_cannot",
    "repeat_ngram_screen_catches_planted_degenerate_document",
    "repeat_ngram_screen_is_not_subsumed_by_dedup",
    "trainer_side_repeat_mask_fires_when_curation_is_bypassed",
}


def summarize(results: list[ControlResult]) -> dict:
    failed = [r for r in results if not r.caught]
    produced = {r.name for r in results}
    missing = sorted(EXPECTED_CONTROLS - produced)
    return {
        "n_controls": len(results),
        "expected_controls_declared": len(EXPECTED_CONTROLS),
        "expected_controls_missing": missing,
        "every_declared_control_ran": not missing,
        "n_passed": len(results) - len(failed),
        "n_failed": len(failed),
        "all_invariants_are_falsifiable": (not failed) and not missing,
        "meaning": "each control breaks one invariant and requires the verifier to reject it. A "
                   "control that does not trigger a rejection means the corresponding check is "
                   "vacuous, and is reported as a FAILURE.",
        "controls": [r.to_record() for r in results],
    }
