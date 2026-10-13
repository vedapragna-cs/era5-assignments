"""The demo corpus: seven lanes matching S5, plus validation and test splits.

Small and authored rather than downloaded, for three reasons: the assignment asks for a corpus the
grader does not have to download, a fixed corpus makes every hash in the evidence bundle stable, and
planted defects (PII, duplicates, benchmark contamination) let the firewall and cleaning gates be
demonstrated *catching* something rather than passing vacuously.

Document model
--------------
A document is a list of role-tagged segments. Uniform segment roles mean the packer builds loss masks
the same way for every lane; only the role->supervised mapping differs. This matters for the agentic
lane, where S5 specifies the mask exactly:

    supervised : assistant_plan, tool_call, final_answer   (including a FAILED call - recovery is
                                                            the taught behaviour)
    context    : user, tool_observation, repo_file, shell_output
    verifier   : verifier_outcome

and measured a supervision yield of 356/668 = 53% on the reference trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Split = Literal["train", "validation", "test", "proxy"]

SUPERVISED_ROLES = {"assistant_plan", "tool_call", "final_answer", "text"}
CONTEXT_ROLES = {"user", "tool_observation", "repo_file", "shell_output"}
VERIFIER_ROLES = {"verifier_outcome"}

# Role has to survive packing, not just decide the loss mask and vanish.
#
# The mask answers "was this position trained on". It cannot answer "what KIND of text was it",
# because supervised and context roles both collapse into it - `tool_call` and `text` are
# indistinguishable 1s, `user` and `tool_observation` indistinguishable 0s. So a per-lane perplexity
# is the finest attribution the ledger could make, and S5's open question is not about lanes: it
# asks whether agentic data yields ~53% supervised tokens and whether the supervised part is
# actually harder. That is a per-ROLE question.
#
# `role_codes` is therefore carried alongside `loss_mask` through every packing policy and lands on
# each token observation. Codes rather than strings because it is one int8-sized value per position
# over every packed window; index 0 is padding so an unset entry is never silently a real role.
ROLE_NAMES = ("<pad>", "text", "user", "assistant_plan", "tool_call", "tool_observation",
              "final_answer", "verifier_outcome", "repo_file", "shell_output", "boundary")
PAD_ROLE = 0
_ROLE_CODE = {name: i for i, name in enumerate(ROLE_NAMES)}


def role_code(role: str) -> int:
    """Strict: an unknown role raises rather than falling into an 'other' bucket.

    A silent catch-all would let a corpus author add a role, get zero supervision for it, and see
    nothing in any report. The loud failure is the cheaper one.
    """
    try:
        return _ROLE_CODE[role]
    except KeyError:
        raise ValueError(
            f"unknown segment role {role!r}; add it to ROLE_NAMES and classify it as "
            "supervised/context/verifier") from None


def role_name(code: int) -> str:
    return ROLE_NAMES[code]


@dataclass(frozen=True)
class Segment:
    role: str
    text: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    lane: str
    source_id: str
    license_tier: str          # safe | review | unsafe
    language: str
    script: str
    difficulty_band: str
    split: Split
    segments: tuple[Segment, ...]
    benchmark_id: str | None = None     # set for validation/test shards
    canary: str | None = None           # canary string planted in eval data
    derived_from_benchmark: bool = False
    notes: str = ""

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)

    def to_record(self) -> dict:
        return {
            "doc_id": self.doc_id, "lane": self.lane, "source_id": self.source_id,
            "license_tier": self.license_tier, "language": self.language, "script": self.script,
            "difficulty_band": self.difficulty_band, "split": self.split,
            "benchmark_id": self.benchmark_id, "canary": self.canary,
            "derived_from_benchmark": self.derived_from_benchmark,
            "segments": [{"role": s.role, "text": s.text} for s in self.segments],
        }


def _d(doc_id, lane, source_id, text, *, license_tier="safe", language="en", script="Latn",
       band="B2", split="train", **kw) -> Document:
    return Document(doc_id, lane, source_id, license_tier, language, script, band, split,
                    (Segment("text", text),), **kw)


# ----------------------------------------------------------------------------------------------
# web  (concat_chop: documents are genuinely split across window boundaries)
# ----------------------------------------------------------------------------------------------
WEB = [
    _d("web-0001", "web", "dclm_baseline",
       "The monsoon report changed district irrigation policy because farmers needed earlier warnings. "
       "Officials published revised sowing windows for three crops and asked cooperatives to circulate "
       "them in local languages. Adoption was uneven: districts with existing extension officers moved "
       "within a season, while others waited for the following year. "),
    _d("web-0002", "web", "dclm_baseline",
       "Municipal water tariffs were restructured into four slabs. The lowest slab was left unchanged to "
       "protect low-consumption households, while the highest was raised to fund pipeline replacement. "
       "The utility published a monthly dashboard of leakage rates, which had not previously been "
       "available to the public. "),
    _d("web-0003", "web", "fineweb_edu",
       "A short history of the printing press in South Asia begins not with movable type but with "
       "lithography, which suited scripts with many conjuncts far better than metal sorts did. "
       "Presses in Lucknow and Kanpur produced Urdu and Persian texts at volumes that metal type "
       "could not match for another generation. "),
    _d("web-0004", "web", "fineweb_edu",
       "Railway gauge conversion proceeded in phases. Each phase closed a corridor for several months, "
       "which shifted freight onto roads and raised local haulage prices temporarily. Planners "
       "published the schedule two years ahead so that shippers could contract around it. "),
    _d("web-0005", "web", "dclm_baseline",
       "Contact for grievances: write to the district office. Applications are acknowledged within "
       "fifteen working days. Appeals go to the divisional commissioner. "),
    # planted PII - must be caught by the PII screen, not by the firewall
    _d("web-0006", "web", "crawl_misc",
       "Complaint filed by Aditya Singh, resident of 42 Nehru Road, reachable at aditya.singh@example.com "
       "or 9876543210, regarding delayed water connection. The officer noted the file number and "
       "promised a site visit. ", notes="planted PII: name, address, email, phone"),
    # planted near-duplicate of web-0001 - must be caught by dedup
    _d("web-0007", "web", "crawl_misc",
       "The monsoon report changed district irrigation policy because farmers needed earlier warnings. "
       "Officials published revised sowing windows for three crops and asked cooperatives to circulate "
       "them in local languages. Adoption was uneven across districts. ",
       notes="planted near-duplicate of web-0001"),
    # planted repeated-n-gram document - the OLMo 2 spike trigger (arXiv:2501.00656 section 3.1).
    # Their examples are base64 runs and repeated numeric tokens escaping a serialiser; this is the
    # same shape. It must be caught by the repeat-n-gram screen, NOT by dedup: the document is
    # unique across the corpus, so shingle-Jaccard against other documents sees nothing wrong. That
    # distinction is the point of the control - a degenerate document is not a duplicate document.
    _d("web-0008", "web", "crawl_misc",
       "Export log follows. " + "255, " * 60 + "End of export. ",
       notes="planted repeated n-gram run (OLMo 2 spike trigger); unique, so dedup cannot catch it"),
]

# ----------------------------------------------------------------------------------------------
# code  (greedy first-fit: file boundaries preferred but spans tolerated)
# ----------------------------------------------------------------------------------------------
CODE = [
    _d("code-0001", "code", "the_stack_v2", band="B2", text=(
        "def train(loader, model, optimizer):\n"
        "    for batch in loader:\n"
        "        loss = model.loss(batch)\n"
        "        loss.backward()\n"
        "        optimizer.step()\n"
        "        optimizer.zero_grad()\n"
        "    return loss / len(loader)\n")),
    _d("code-0002", "code", "the_stack_v2", band="B2", text=(
        "def pack(documents, context_length):\n"
        "    bins = []\n"
        "    for doc in documents:\n"
        "        for bin_ in bins:\n"
        "            if sum(len(d) for d in bin_) + len(doc) <= context_length:\n"
        "                bin_.append(doc)\n"
        "                break\n"
        "        else:\n"
        "            bins.append([doc])\n"
        "    return bins\n")),
    _d("code-0003", "code", "commitpackft", band="B3", text=(
        "--- a/loader.py\n+++ b/loader.py\n"
        "@@ def resume(state):\n"
        "-    offset = state['rank_offsets'][rank]\n"
        "+    offset = state['global_offset']\n"
        "+    # rank partition is rederived from world_size at load time\n"
        "     return offset\n"), notes="diff-following shape; S5 upsamples CommitPackFT 8x"),
    _d("code-0004", "code", "the_stack_v2", band="B3", text=(
        "class HashChain:\n"
        "    def append(self, payload):\n"
        "        prev = self.head\n"
        "        entry = dict(payload, prev_hash=prev)\n"
        "        entry['entry_hash'] = sha256(canonical(entry))\n"
        "        self.entries.append(entry)\n"
        "        return entry\n")),
    _d("code-0005", "code", "the_stack_v2", band="B2", text=(
        "def position_ids(segment_ids):\n"
        "    out, last, counter = [], None, 0\n"
        "    for seg in segment_ids:\n"
        "        counter = 0 if seg != last else counter + 1\n"
        "        out.append(counter)\n"
        "        last = seg\n"
        "    return out\n")),
]

# ----------------------------------------------------------------------------------------------
# indic  (best_fit; Devanagari, the protected lane)
# ----------------------------------------------------------------------------------------------
INDIC = [
    _d("indic-0001", "indic", "sangraha_verified", language="hi", script="Deva", band="B1", text=(
        "भारत में भाषा और तकनीक का मिलन तेजी से बढ़ रहा है। सरकारी दस्तावेज़ अब कई भाषाओं में "
        "उपलब्ध कराए जाते हैं, जिससे नागरिकों को अपने अधिकारों की जानकारी अपनी भाषा में मिलती है। ")),
    _d("indic-0002", "indic", "sangraha_verified", language="hi", script="Deva", band="B2", text=(
        "शिक्षा की गुणवत्ता का आकलन केवल परीक्षा परिणामों से नहीं किया जा सकता। कक्षा में बिताया गया "
        "समय, शिक्षकों की उपलब्धता और पाठ्यपुस्तकों की भाषा भी उतनी ही महत्वपूर्ण है। ")),
    _d("indic-0003", "indic", "indiccorp_v2", language="mr", script="Deva", band="B1", text=(
        "पाणीपुरवठा योजनेचा आढावा घेण्यात आला. गावातील प्रत्येक घरात नळजोडणी देण्याचे उद्दिष्ट "
        "ठेवण्यात आले आहे. ")),
    _d("indic-0004", "indic", "sangraha_verified", language="hi", script="Deva", band="B3",
       notes="tier-A verified: reserved for anneal per S5", text=(
        "न्यायालय ने कहा कि प्रशासनिक आदेश में कारण दर्ज करना आवश्यक है। कारण दर्ज न करने पर आदेश "
        "की वैधता की जाँच नहीं की जा सकती, और यही कारण अभिलेख का आधार बनते हैं। ")),
    _d("indic-0005", "indic", "indiccorp_v2", language="te", script="Telu", band="B1", text=(
        "గ్రామ పంచాయతీ సమావేశంలో తాగునీటి సమస్యపై చర్చ జరిగింది. కొత్త బోరుబావి కోసం "
        "ప్రతిపాదన పంపాలని నిర్ణయించారు. ")),
    _d("indic-0006", "indic", "sangraha_verified", language="hi", script="Deva", band="B2", text=(
        "मौसम विभाग ने अगले सप्ताह के लिए भारी वर्षा की चेतावनी जारी की है। किसानों को सलाह दी गई "
        "है कि कटाई का कार्य पहले पूरा कर लें। ")),
]

# ----------------------------------------------------------------------------------------------
# stem  (best_fit)
# ----------------------------------------------------------------------------------------------
STEM = [
    _d("stem-0001", "stem", "pes2o", band="B3", text=(
        "The diffusion coefficient scales inversely with viscosity in the Stokes-Einstein relation. "
        "For a spherical particle of radius r in a fluid of viscosity eta, D = kT / (6 pi eta r). "
        "The relation holds when the particle is much larger than the solvent molecules. ")),
    _d("stem-0002", "stem", "proof_pile_2", band="B4", text=(
        "Lemma. If n is prime and n > 2, then n is odd. Proof. Suppose n is even and n > 2. Then "
        "2 divides n and 2 != n, so n has a divisor other than 1 and itself, contradicting primality. ")),
    _d("stem-0003", "stem", "pes2o", band="B3", text=(
        "Sample size determination for a two-proportion test depends on the baseline rate, the "
        "minimum detectable effect, the significance level and the desired power. Reporting only the "
        "final sample size hides which of these was relaxed to reach it. ")),
]

# ----------------------------------------------------------------------------------------------
# reasoning  (best_fit; S5 length bands L0-L3)
# ----------------------------------------------------------------------------------------------
REASONING = [
    _d("reason-0001", "reasoning", "aon_v4", band="B2", notes="L0 short trace", text=(
        "How many integers from 1 to 1000 are divisible by 3 or 5? Inclusion-exclusion: "
        "floor(1000/3) + floor(1000/5) - floor(1000/15) = 333 + 200 - 66 = 467. ")),
    _d("reason-0002", "reasoning", "aon_v4", band="B3", notes="L2 trace with verification", text=(
        "Count multiples of 3 up to 1000: floor(1000/3) = 333. Multiples of 5: floor(1000/5) = 200. "
        "Numbers counted twice are multiples of 15, the least common multiple: floor(1000/15) = 66. "
        "So the union is 333 + 200 - 66 = 467. Verify by complement: the density of numbers divisible "
        "by neither is (2/3)(4/5) = 8/15, so about 533 are excluded and 467 remain. The two agree. ")),
    _d("reason-0003", "reasoning", "aon_v4", band="B4", notes="L3 ultra trace", text=(
        "Restate the goal: count integers in [1,1000] divisible by 3 or by 5. Plan: inclusion-exclusion, "
        "then an independent check by blocks. Derive: |A| = 333, |B| = 200, |A and B| = 66, so "
        "|A or B| = 467. Check by blocks of 15: each block contains 7 qualifying numbers "
        "(3,5,6,9,10,12,15). There are 66 full blocks giving 462, and the remainder 991..1000 "
        "contributes 993,995,996,999,1000 which is 5. Total 467. Both methods agree. ")),
]

# ----------------------------------------------------------------------------------------------
# agentic  (structure_preserving; role-segmented, mask per S5)
# ----------------------------------------------------------------------------------------------
AGENTIC = [
    Document("agent-0001", "agentic", "swe_gym", "safe", "en", "Latn", "B3", "train", (
        Segment("user", "The build fails after the loader refactor. Fix it. "),
        Segment("assistant_plan", "I will read the failing test, then inspect loader.py for the "
                                  "changed offset field. "),
        Segment("tool_call", "read_file(path='tests/test_loader.py') "),
        Segment("tool_observation", "AssertionError: expected global_offset, got rank_offsets "),
        Segment("assistant_plan", "The state schema changed. I will update the resume path. "),
        Segment("tool_call", "edit_file(path='loader.py', old='rank_offsets[rank]', new='global_offset') "),
        Segment("tool_observation", "1 replacement made "),
        Segment("tool_call", "run_tests(path='tests/test_loader.py') "),
        Segment("tool_observation", "2 passed "),
        Segment("final_answer", "Fixed: resume now reads the global offset and rederives the rank "
                                "partition at load time. "),
        Segment("verifier_outcome", "PASS "),
    ), notes="clean trajectory"),
    Document("agent-0002", "agentic", "openhands", "safe", "en", "Latn", "B3", "train", (
        Segment("user", "Find why the shard hash changes between runs. "),
        Segment("assistant_plan", "I will hash the shard twice and diff the manifest. "),
        Segment("tool_call", "hash_shard(id='shard-web-0001') "),
        Segment("tool_observation", "ERROR: shard not found "),
        Segment("assistant_plan", "Wrong id format. Shard ids are prefixed by stage; I will list them. "),
        Segment("tool_call", "list_shards() "),
        Segment("tool_observation", "['v5_web_000', 'v5_code_001'] "),
        Segment("tool_call", "hash_shard(id='v5_web_000') "),
        Segment("tool_observation", "sha256:9f2c... "),
        Segment("final_answer", "The hash is stable; the manifest was being regenerated with a "
                                "non-content-derived field. "),
        Segment("verifier_outcome", "PASS "),
    ), notes="contains a FAILED tool call - S5 requires the recovery to be supervised"),
]

# ----------------------------------------------------------------------------------------------
# longctx  (long_context: one document owns the window)
# ----------------------------------------------------------------------------------------------
LONGCTX = [
    _d("longctx-0001", "longctx", "repo_packed", band="B4", text=(
        "Repository walkthrough. The data system is organised as a pipeline whose stages each emit an "
        "immutable artifact. Stage one reads documents and attaches provenance. Stage two applies the "
        "cleaning pipeline and records its hash, so that the transformation from raw text to admitted "
        "text is reproducible. Stage three deduplicates, first exactly and then by shingle overlap, "
        "and records the cluster id of every near-duplicate group. Stage four screens for personal "
        "information and records the screen status. Stage five scans against the evaluation registry "
        "and records overlap status. Only then is a document eligible for tokenization. Stage six "
        "tokenizes with the frozen tokenizer and writes a shard, whose content hash is taken over the "
        "token bytes rather than over any metadata. Stage seven builds the manifest and submits it to "
        "the admission gate, which blocks on any missing hard field. The scheduler consumes only "
        "admitted shards, and the ledger records every batch it serves. " * 3)),
]

# ----------------------------------------------------------------------------------------------
# validation and test - registered so they can be BLOCKED, plus contamination cases
# ----------------------------------------------------------------------------------------------
EVAL = [
    Document("eval-milu-001", "indic", "milu_benchmark", "safe", "hi", "Deva", "B3", "test", (
        Segment("text", "प्रश्न: भारतीय संविधान की प्रस्तावना में कौन से शब्द बाद में जोड़े गए? "
                        "CANARY-MILU-7f3a9c उत्तर: समाजवादी और पंथनिरपेक्ष। "),
    ), benchmark_id="MILU", canary="CANARY-MILU-7f3a9c"),
    Document("eval-gsm-001", "reasoning", "gsm8k_benchmark", "safe", "en", "Latn", "B2", "test", (
        Segment("text", "Natalia sold clips to 48 friends in April and half as many in May. "
                        "CANARY-GSM-11b204 How many did she sell altogether? Answer: 72. "),
    ), benchmark_id="GSM8K", canary="CANARY-GSM-11b204"),
    Document("eval-swe-001", "code", "swebench_verified", "safe", "en", "Latn", "B4", "test", (
        Segment("text", "Task instance astropy__astropy-12907. Gold patch modifies separability_matrix "
                        "to handle nested CompoundModels. CANARY-SWE-55e1d8 "),
    ), benchmark_id="SWE-bench-Verified", canary="CANARY-SWE-55e1d8"),
    Document("val-web-001", "web", "held_out_web", "safe", "en", "Latn", "B2", "validation", (
        Segment("text", "Held-out web sample used for periodic validation loss. It may be read during "
                        "training for evaluation but must never carry gradient. "),
    ), benchmark_id="internal-val"),
    Document("val-indic-001", "indic", "held_out_indic", "safe", "hi", "Deva", "B2", "validation", (
        Segment("text", "यह पाठ केवल मूल्यांकन के लिए है। इस पर प्रशिक्षण नहीं किया जाएगा। "),
    ), benchmark_id="internal-val"),
    # contamination case: derived from a benchmark, no canary, overlap below the flat threshold.
    # This is the one that leaks if only a single gate is enabled - the measured thin spot in the
    # course firewall widget was exactly this class of document.
    Document("web-0100", "web", "blog_scrape", "safe", "en", "Latn", "B2", "train", (
        Segment("text", "Walkthrough: a seller moved 48 hair clips among her friends during April, "
                        "then only half that number the following month, which leaves 24 for the "
                        "second month and 72 altogether. This is a worked explanation of a "
                        "well-known grade-school word problem. "),
    ), derived_from_benchmark=True,
       notes="benchmark-derived TRUE paraphrase: no canary, Jaccard below threshold, and no shared "
             "13-gram - so only the derived gate can catch it. An earlier version of this document "
             "reused the benchmark's exact wording, which meant n-gram containment caught it too "
             "and the 'derived gate is load-bearing' control was measuring the wrong thing."),
    # contamination case reachable ONLY by n-gram containment. A long, ordinary host article that
    # quotes one benchmark item verbatim. Jaccard against the benchmark is near zero because the
    # host dominates the shingle union; there is no canary; it is not marked benchmark-derived. This
    # is the shape the decontamination literature says is typical - a small passage inside a large
    # document - and it is exactly what a global-similarity gate cannot see.
    Document("web-0102", "web", "news_aggregate", "safe", "en", "Latn", "B2", "train", (
        Segment("text",
                "Education notebook: a round-up of classroom material submitted by readers this "
                "term. Teachers in three districts described how they introduce arithmetic through "
                "everyday transactions, and several sent along the exercises they use in class. "
                "One contributor wrote that word problems land better when the quantities are "
                "small and the situation is familiar to the children. A worked example she "
                "circulates reads: Natalia sold clips to 48 friends in April and half as many in "
                "May. Other teachers preferred problems drawn from local markets, and a few noted "
                "that pupils who struggle with symbolic notation often reason correctly when the "
                "same question is posed aloud. The notebook closes with a request for more "
                "submissions before the end of the term. "),
    ), notes="benchmark passage quoted inside a long host; only n-gram containment catches it"),
    # unsafe license: must be blocked by admission, independently of content
    _d("web-0101", "web", "unlicensed_dump", license_tier="unsafe", text=(
        "Scraped corpus of unknown provenance with no license statement attached. "),
       notes="unsafe license"),
]

# ----------------------------------------------------------------------------------------------
# OPUS golden proxy — the direction the selector scores candidates against.
#
# A separate split, deliberately. It is never trained on (so it cannot leak into the model) and it
# is not registered with the firewall (it is not benchmark data, so calling it "eval" would make the
# firewall's rejection counts meaningless). It is exactly what S5 describes: a small high-quality set
# whose gradient direction defines "useful update".
#
# S5's warning applies directly here: OPUS under an English-heavy proxy drove Indic to 0 tokens in
# 60/60 iterations. So the proxy deliberately spans the capabilities we claim - including Indic and
# agentic - and the protected floors exist because even a balanced proxy did not rescue agentic.
# ----------------------------------------------------------------------------------------------
PROXY = [
    _d("proxy-web-001", "web", "golden_proxy", split="proxy", band="B2", text=(
        "A well-formed explanatory paragraph states its claim, gives the mechanism, and names the "
        "measurement that would refute it. ")),
    _d("proxy-code-001", "code", "golden_proxy", split="proxy", band="B3", text=(
        "def resume(state, world_size):\n"
        "    offset = state['global_offset']\n"
        "    return [i for i in range(offset, offset + 8) if i % world_size == rank]\n")),
    _d("proxy-indic-001", "indic", "golden_proxy", language="hi", script="Deva", split="proxy",
       band="B2", text=(
        "किसी भी नीति का मूल्यांकन उसके उद्देश्य और उसके मापने योग्य परिणाम दोनों से किया जाना चाहिए। ")),
    _d("proxy-reason-001", "reasoning", "golden_proxy", split="proxy", band="B3", text=(
        "Claim, then derivation, then an independent check by a second method. If the two methods "
        "disagree, the claim is not yet established. ")),
    Document("proxy-agent-001", "agentic", "golden_proxy", "safe", "en", "Latn", "B3", "proxy", (
        Segment("user", "Make the failing test pass. "),
        Segment("assistant_plan", "Read the test, locate the assertion, then change the source. "),
        Segment("tool_call", "run_tests(path='tests/') "),
        Segment("tool_observation", "1 failed "),
        Segment("final_answer", "Corrected the field name in the resume path. "),
    )),
]


# ----------------------------------------------------------------------------------------------
# Difficulty spread. S5 bands B0..B5 and maps them to stages, so a corpus that is entirely B2/B3
# starves the seed and anneal stages and the curriculum gate rejects almost everything. These fill
# the ends of the ladder. (Found the hard way - see EXPERIMENT_LOG F1.)
# ----------------------------------------------------------------------------------------------
LADDER = [
    _d("web-0010", "web", "tinystories_like", band="B0", text=(
        "The cat sat on the mat. It was warm. The sun was up. The cat was happy. ")),
    _d("web-0011", "web", "simple_wiki", band="B1", text=(
        "A river is a stream of water that flows to the sea. Rivers begin in hills. People build "
        "towns near rivers because water is needed for farms. ")),
    _d("indic-0010", "indic", "indiccorp_v2", language="hi", script="Deva", band="B0", text=(
        "यह एक घर है। घर बड़ा है। घर में एक बिल्ली है। बिल्ली सो रही है। ")),
    _d("indic-0011", "indic", "indiccorp_v2", language="hi", script="Deva", band="B1", text=(
        "आज मौसम अच्छा है। बच्चे स्कूल गए। शिक्षक ने पाठ पढ़ाया। सब ने ध्यान से सुना। ")),
    _d("code-0010", "code", "the_stack_v2", band="B1", text=(
        "def add(a, b):\n    return a + b\n\n\ndef mean(xs):\n    return sum(xs) / len(xs)\n")),
    _d("stem-0010", "stem", "simple_science", band="B1", text=(
        "Water boils at one hundred degrees Celsius at sea level. At higher altitude the boiling "
        "point is lower because the air presses down with less force. ")),
    _d("reason-0010", "reasoning", "gsm8k_like", band="B1", text=(
        "Natalia has 12 apples. She gives 5 away. How many are left? 12 minus 5 is 7. ")),
    # B5: research-grade, concentrated in the anneal per S5
    _d("stem-0011", "stem", "proof_pile_2", band="B5", text=(
        "Theorem. Every finite integral domain is a field. Proof. Let D be finite and a nonzero. "
        "The map x -> ax is injective by cancellation, hence surjective by finiteness, so there "
        "exists b with ab = 1. ")),
    _d("reason-0011", "reasoning", "aon_v4", band="B5", text=(
        "Suppose the packing utilization u and the loss-mask density m are independent. Then useful "
        "tokens per window is u*m*L. Maximising u alone is therefore not optimal when m falls with "
        "u, which is exactly the structure-preserving case. ")),
    _d("code-0011", "code", "swebench_like", band="B5", text=(
        "--- a/scheduler.py\n+++ b/scheduler.py\n@@\n-    rank = cursor % world\n"
        "+    rank = global_index % world  # position is global; partition is derived at load\n")),
    _d("indic-0012", "indic", "sangraha_verified", language="hi", script="Deva", band="B5", text=(
        "संवैधानिक व्याख्या में उद्देश्यपूर्ण दृष्टिकोण यह मानता है कि प्रावधान का अर्थ उसके "
        "अभिप्रेत परिणाम से निर्धारित होता है, न कि केवल शाब्दिक पाठ से। ")),
]

ALL_DOCUMENTS: list[Document] = (
    WEB + CODE + INDIC + STEM + REASONING + AGENTIC + LONGCTX + LADDER + EVAL + PROXY
)


def load_corpus() -> list[Document]:
    return list(ALL_DOCUMENTS)


def by_split(split: Split) -> list[Document]:
    return [d for d in ALL_DOCUMENTS if d.split == split]


def canaries() -> dict[str, str]:
    """canary string -> benchmark id, for the firewall scanner."""
    return {d.canary: d.benchmark_id for d in ALL_DOCUMENTS if d.canary}
