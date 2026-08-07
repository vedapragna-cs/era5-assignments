"""The frozen tokenizer.

Two properties matter and both are enforced here rather than assumed.

**Frozen.** The tokenizer is trained once, hashed, and every downstream artifact records
that hash. A shard tokenized under a different vocabulary is a different shard, and the
manifest makes that impossible to miss. `FrozenTokenizer.assert_matches` is called before
any shard is read back.

**Trained on the train split only.** The session's evaluation firewall is usually applied
to training batches, but a tokenizer fitted on held-out text is already a leak: the
vocabulary itself encodes which strings appeared in the evaluation set. So the firewall is
applied one step earlier than the assignment strictly asks for.

Special tokens follow the session's conclusion on BOS. Both BOS and EOS existed
historically; the session's position is that carrying both wastes a position per document
and "only EOS is covered". BOS is therefore *defined* in the vocabulary but *disabled* by
default, behind `TokenizerSpec.use_bos`, so the choice is a recorded configuration rather
than an assumption baked into the data.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .determinism import content_hash, sha256_bytes
from .documents import Document, SPLIT_TRAIN

PAD = "<pad>"
EOS = "<eos>"
BOS = "<bos>"
UNK = "<unk>"

# Structural markers. These must exist in the vocabulary *before* the corpus is built,
# because the corpus has to contain them for the model to ever see them -- the session's
# "data set is built together a bit of tokenizer is required, then we build a data set and
# then we finish the tokenizer".
ROLE_TOKENS: Dict[str, str] = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "prompt": "<|prompt|>",
    "response": "<|response|>",
    "tool_call": "<|tool_call|>",
    "tool_result": "<|tool_result|>",
    "thinking": "<|think|>",
    "text": "<|text|>",
}

SPECIAL_TOKENS: List[str] = [PAD, UNK, BOS, EOS] + [ROLE_TOKENS[k] for k in sorted(ROLE_TOKENS)]

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class TokenizerSpec:
    vocab_size: int = 8192
    min_frequency: int = 2
    use_bos: bool = False   # session: carrying both BOS and EOS wastes a position
    version: str = "era5-s6-bpe-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "use_bos": self.use_bos,
            "version": self.version,
            "special_tokens": SPECIAL_TOKENS,
        }


@dataclass
class FrozenTokenizer:
    """A trained tokenizer plus the hash that pins it."""

    spec: TokenizerSpec
    tokenizer: Any                    # tokenizers.Tokenizer
    tokenizer_hash: str
    vocab_size: int
    pad_id: int
    eos_id: int
    bos_id: int
    unk_id: int
    role_ids: Dict[str, int]
    training_fingerprint: str         # fingerprint of the documents it was fitted on

    # ---- identity ------------------------------------------------------

    @property
    def tokenizer_id(self) -> str:
        return "{}-{}".format(self.spec.version, self.tokenizer_hash[:12])

    def assert_matches(self, expected_hash: str, context: str = "") -> None:
        if expected_hash != self.tokenizer_hash:
            raise RuntimeError(
                "tokenizer mismatch{}: artifact was built with {} but the loaded "
                "tokenizer is {}. A shard tokenized under a different vocabulary is a "
                "different shard.".format(
                    " (" + context + ")" if context else "",
                    expected_hash[:12], self.tokenizer_hash[:12]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_hash": self.tokenizer_hash,
            "vocab_size": self.vocab_size,
            "spec": self.spec.to_dict(),
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
            "bos_id": self.bos_id,
            "training_fingerprint": self.training_fingerprint,
        }

    # ---- encoding ------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def decode_one(self, token_id: int) -> str:
        """Human-readable form of a single token, for the learning ledger."""
        tok = self.tokenizer.id_to_token(int(token_id))
        if tok is None:
            return "<id:{}>".format(int(token_id))
        # ByteLevel BPE marks a leading space with U+0120.
        return tok.replace("Ġ", " ").replace("Ċ", "\\n")

    def encode_document(self, doc: Document) -> Tuple[List[int], List[int], List[str]]:
        """Encode a document into (token_ids, loss_mask, per-token role).

        The loss mask is produced here, at encode time, from the document's segment
        structure -- not patched on later from a regex. Role markers themselves are
        supervised when they open a supervised segment (the model must learn to *emit*
        `<|assistant|>`) and masked when they open a masked one.
        """
        ids: List[int] = []
        mask: List[int] = []
        roles: List[str] = []

        if self.spec.use_bos:
            ids.append(self.bos_id)
            mask.append(0)
            roles.append("bos")

        for seg in doc.segments:
            marker = self.role_ids[seg.role]
            trainable = 1 if seg.trainable else 0
            ids.append(marker)
            mask.append(trainable)
            roles.append(seg.role)

            body = self.encode(seg.text)
            ids.extend(body)
            mask.extend([trainable] * len(body))
            roles.extend([seg.role] * len(body))

        # EOS closes the document. It is supervised: the model must learn *when to stop*,
        # and a masked EOS teaches it nothing about turn boundaries.
        ids.append(self.eos_id)
        mask.append(1)
        roles.append("eos")
        return ids, mask, roles


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_tokenizer(docs: Sequence[Document], spec: TokenizerSpec = TokenizerSpec()
                    ) -> FrozenTokenizer:
    """Fit a byte-level BPE on the *train split only*, then freeze and hash it."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    train_docs = sorted(
        [d for d in docs if d.split == SPLIT_TRAIN], key=lambda d: d.doc_id
    )
    if not train_docs:
        raise ValueError("no train-split documents to fit a tokenizer on")

    corpus_lines = [seg.text for d in train_docs for seg in d.segments]
    training_fingerprint = content_hash([d.doc_id for d in train_docs])

    tok = Tokenizer(models.BPE(unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=spec.vocab_size,
        min_frequency=spec.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(corpus_lines, trainer=trainer, length=len(corpus_lines))

    # Hash the serialized tokenizer, not the object. This is what pins a shard.
    serialized = tok.to_str()
    tokenizer_hash = sha256_bytes(serialized.encode("utf-8"))

    role_ids = {}
    for role, marker in ROLE_TOKENS.items():
        tid = tok.token_to_id(marker)
        if tid is None:
            raise RuntimeError("role marker {!r} missing from vocabulary".format(marker))
        role_ids[role] = tid

    return FrozenTokenizer(
        spec=spec,
        tokenizer=tok,
        tokenizer_hash=tokenizer_hash,
        vocab_size=tok.get_vocab_size(),
        pad_id=tok.token_to_id(PAD),
        eos_id=tok.token_to_id(EOS),
        bos_id=tok.token_to_id(BOS),
        unk_id=tok.token_to_id(UNK),
        role_ids=role_ids,
        training_fingerprint=training_fingerprint,
    )


def save_tokenizer(ft: FrozenTokenizer, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ft.tokenizer.save(path)
    return ft.tokenizer_hash


# ---------------------------------------------------------------------------
# fertility
# ---------------------------------------------------------------------------


def fertility_report(ft: FrozenTokenizer, docs: Sequence[Document]) -> Dict[str, Any]:
    """Tokens per whitespace word, per language and script.

    S5's target for the Indic lane is a fertility near 2.0. The session's warning is
    concrete: "if we have 13 tokens per word for Telugu, a thousand-word Telugu document
    becomes a 13,000 token document" -- so it has to be chopped, and the chopping is a
    tokenizer failure wearing a packing costume.
    """
    per_lang: Dict[str, Dict[str, float]] = {}
    for doc in docs:
        key = "{}/{}".format(doc.language, doc.script)
        bucket = per_lang.setdefault(key, {"words": 0, "tokens": 0, "docs": 0, "chars": 0})
        text = doc.text
        bucket["words"] += len(_WORD_RE.findall(text))
        bucket["tokens"] += len(ft.encode(text))
        bucket["chars"] += len(text)
        bucket["docs"] += 1

    out: Dict[str, Any] = {}
    for key, b in sorted(per_lang.items()):
        words = max(1, b["words"])
        out[key] = {
            "documents": b["docs"],
            "words": b["words"],
            "tokens": b["tokens"],
            "fertility_tokens_per_word": round(b["tokens"] / words, 4),
            "chars_per_token": round(b["chars"] / max(1, b["tokens"]), 4),
        }

    indic_scripts = {"Deva", "Telu", "Taml", "Beng", "Knda", "Orya", "Gujr", "Guru", "Mlym"}
    indic = {k: v for k, v in out.items() if k.split("/")[1] in indic_scripts}
    if indic:
        tot_tok = sum(v["tokens"] for v in indic.values())
        tot_wrd = sum(v["words"] for v in indic.values())
        out["_indic_aggregate"] = {
            "fertility_tokens_per_word": round(tot_tok / max(1, tot_wrd), 4),
            "s5_target": 2.0,
            "note": ("Synthetic-corpus figure. A real 8k-vocab BPE fitted mostly on Latin "
                     "text would be far worse on Indic scripts; the S5 target assumes a "
                     "vocabulary sized and balanced for the lane."),
        }
    return out


def expected_initial_loss(vocab_size: int) -> float:
    """Natural log of the vocabulary size.

    The session's check: before the model learns anything, every token is equally likely,
    so the loss is ln(V) -- 11.78 for a 131,072 vocabulary. A run that starts materially
    below this has seen the answers; a run that starts above it is broken. Asserted in
    trainer.py at step 0.
    """
    import math

    return math.log(vocab_size)
