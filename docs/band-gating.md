# Difficulty-band gating: the mechanism S5 deferred

S5's `CURRICULUM.md` closes with an open item:

> the mechanism for enforcing a difficulty band at sampling time (classifier? source
> tagging?) is not yet specified and belongs in the S6 data-stream work.

This is that specification, and the answer is **neither** — not a classifier at sampling
time, and not a tag consulted at sampling time. The band is resolved into the **shard
layout**, so that enforcing it at sampling time costs one dictionary lookup.

## Why not a classifier at sampling time

A difficulty classifier in the sampling path has to run on every candidate document, on
every step, on every rank. At 16 sequences per step it is cheap; at a real global batch it
is a second model in the hot loop. Worse, it is *non-deterministic across versions*: retrain
or quantize the classifier and the same step draws different documents, which silently
breaks replay for every run that came before. A ledger whose reconstruction depends on a
model that can be retrained is not a ledger.

## Why not a tag filtered at sampling time

Tagging documents and filtering at draw time is deterministic, but it makes the cost of a
band gate proportional to the documents rejected. In a stage that admits B0–B2, every draw
scans past the B3–B5 documents in the shard. The rejected fraction grows as the curriculum
narrows, which is exactly backwards: the late stages, which have the least headroom, pay
the most.

## What is done instead: the band is part of the shard key

Shards are cut per **(lane, split, difficulty_band)** — `registry.py`:

```python
keys = sorted({(d.lane, d.split, d.difficulty_band) for d in admitted})
```

A band is therefore never mixed inside a shard, and `ShardRegistry.by_lane_band` is a direct
index from `(lane, band)` to shard ids. Enforcing a stage's band range is:

```python
def eligible(self, lane, stage, seq_len):
    for band in stage.band_range:              # the curriculum
        for sid in self.by_lane_band.get((lane, band), []):
            ...
            if m.min_document_tokens > seq_len:  # the long-context reserve
                continue
            out.append(sid)
```

Three properties follow, and each of them is the reason for the choice:

1. **Zero rejection cost.** A stage never looks at a shard outside its bands. The gate is
   a lookup, not a filter, so the late stages get cheaper rather than more expensive.
2. **The gate is auditable from the artifacts alone.** `difficulty_bands` is a manifest
   field. Given only `logs/05_shards.json` and `logs/08_curriculum.json`, a reader can
   verify that no B4 shard was reachable in S1 without running any code.
3. **Replay is unaffected by re-classification.** If the band assignment is later revised,
   the revision produces *different shards with different content hashes*. The old shards
   still exist, still hash the same, and every historical ledger row still resolves. A
   re-tagged document under the sampling-time approach would silently change what a past
   step is reconstructed to have seen.

## Where the band comes from

At corpus construction, from `_LANE_BANDS` — a per-lane distribution over B0–B5. In this
demo it is generated, because the corpus is generated.

In a real pipeline the band would be assigned by whatever the ingest stage already knows:
document length, source tier, reading-level heuristics, code complexity, an offline
classifier run *once* at ingest. The design point is not which of these is used — it is
**where**. The assignment happens once, at ingest, and is then frozen into the shard layout
along with everything else in the manifest. It never runs in the training loop.

That relocation is the whole answer. The band gate is a build-time decision that the
sampling path merely reads, exactly like dedup status, contamination status and the
tokenizer hash. All four are properties a shard either has or does not, established before
training starts, and all four are refused at the same gate.

## The one cost

Cutting on band multiplies the shard count. Here: 122 shards where cutting on
(lane, split) alone would give ~21. At demo scale that is free; at real scale it means
shards are smaller than the ~1B-token target, or that a band with little data produces a
runt shard.

The mitigation is the same one the registry already uses for lanes: `_chunk` fills to a
target token count and a band with less data than the target simply produces one small
shard rather than being merged into a neighbouring band. Merging would be the wrong
trade — a mixed-band shard reintroduces exactly the filter-at-draw-time cost this design
exists to avoid, and it does so invisibly, since the manifest would then list two bands and
the gate would admit the shard for both.

## Verification

`logs/08_curriculum.json` records `band_gate` per step, and `logs/09_consumption_ledger.json`
records the band gate alongside the shards actually read. The OPUS selector independently
rejects any candidate whose band falls outside the current stage with reason code
`stage_mismatch`, *before* scoring it — a curriculum breach is not redeemable by a good
gradient, so the check runs ahead of the one that could excuse it.
