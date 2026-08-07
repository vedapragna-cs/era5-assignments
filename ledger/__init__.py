"""The data ledger: the accounting system between raw documents and the training loop.

Two halves, and the second is the one that does not usually exist:

* the **consumption ledger** records what was sent to the model -- shards, documents, lane
  tokens, and the loader cursor either side of every step;
* the **learning ledger** records what came back -- the per-token loss tensor, attributed
  back through the packing plan to the document, shard, lane, band and role that produced
  each token.

Module order below is the order data flows through them. Nothing here imports torch at
module scope except `model`, `trainer` and the parts of `opus`/`checkpoint` that need it, so
the data path can be built and audited without it.
"""

__all__ = [
    "determinism",
    "plan",
    "documents",
    "corpus",
    "tokenization",
    "firewall",
    "shards",
    "registry",
    "packing",
    "masks",
    "mixture",
    "stream",
    "model",
    "trainer",
    "opus",
    "checkpoint",
    "replay",
    "advisory",
    "report",
]

__version__ = "1.0.0"
