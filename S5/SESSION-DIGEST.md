# S5 — Data Mixtures and Curriculum: extracted ground truth

Source: `S5.webarchive` (lesson page, axiom.theschoolofai.in, 9 embedded widgets) + `S5 Transcript.txt`
(session 2026-07-25, 2h14m). Widget HTML is preserved under `widgets/`; the numbers below are read out
of the widgets' JS, which is authoritative over anything said verbally in class.

---

## 1. The target the mixture must buy

Excellent at **coding + agentic** (Codex-style: long task, plan, multi-step tool calls, read results,
recover from failures, hold growing history). **Controllable reasoning depth** (low/medium/high/ultra).
**Native Indic** — the differentiator and the reason the project exists.

Central claim: *the mixture is the model*. Same clean corpus + same compute → completely different
models depending on how much of each category, how often, and at what stage.

---

## 2. Mixture composer (widget 1) — lanes, floors, presets

Lane order = budget vector index. Protected floors are hard clamps on the sliders.

| lane | floor |
|---|---|
| Code | 0 |
| Agentic / tool-use | **2%** |
| Reasoning traces | 0 |
| Long-context | 0 |
| Indic | **12%** |
| STEM / math | 0 |
| General web | 0 |

`WEIGHTED_MIN = 8` — a lane needs ≥8% share before the widget treats its benchmarks as "funded".

Presets (each sums to 100):

| preset | code | agentic | reason | longctx | indic | stem | web |
|---|---|---|---|---|---|---|---|
| **pretrain** (main run) | 24 | 2 | 6 | 6 | 16 | 12 | 34 |
| **anneal** (short final phase) | 20 | 8 | 18 | 8 | 28 | 10 | 8 |
| **naive** (crawl what's cheapest) | 18 | 2 | 4 | 2 | 6 | 8 | 60 |

Indic provenance tiers, default split of the Indic slice (Session 3 tiers):
A verified native 40% · B unverified crawl 25% · C translated 20% · D synthetic 15%.

**Supply check.** Real available tokens per lane (T), and the classification rule:

```
SUPPLY_T = {code:1.1, agentic:0.00063, reason:0.085, longctx:0.1, indic:0.276, stem:0.25, web:4.5}
demand = share/100 * runSize          runSize ∈ {1, 2, 5, 10} T
demand <= supply          -> "covered"
demand <= 4 * supply      -> "needs repetition"
else                      -> "must synthesize"
```

Widget's own verdict: *"agentic is the binding constraint, it must be synthesized."*

Benchmark→lane map used to light up chips:
Code → LiveCodeBench, Aider · Agentic → SWE-bench, tau-bench, BFCL, GAIA, BrowseComp ·
Reasoning+STEM → AIME, GPQA (reason+stem), HLE (reason) · Long-context → long-eval ·
Indic → MILU, IndicGenBench · General web → MMLU.

---

## 3. Dataset inventory (widget 9) — the real supply

Sizing must happen in **two currencies**: samples ≠ tokens.

### Code — 1,103B tokens
| dataset | source | samples | tokens | license | tier |
|---|---|---|---|---|---|
| The Stack v2 | BigCode / Software Heritage | 600M | 900B | permissive + opt-out | B |
| CommitPack / CommitPackFT | BigCode | 4M | 4B | mixed / permissive | B |
| D3 Code (V4 corpus) | V4 run (confirmed) | 250M | 199B | V4 lineage | B |

### Agentic & tool-use — **627M tokens total** (the whole lane)
| dataset | samples | tokens | license | tier |
|---|---|---|---|---|
| Glaive function-calling v2 | 113K | 50M | Apache-2.0 | D |
| xLAM / APIGen (Salesforce) | 60K | 25M | CC-BY (mixed) | A/D |
| ToolACE | 110K | 60M | Apache-2.0 | A/D |
| Hermes function-calling (Nous) | 15K | 22M | Apache-2.0 | A/D |
| ToolBench (OpenBMB) | 120K | 80M | Apache-2.0 | D |
| Nexus / NexusRaven | 40K | 30M | CC-BY-4.0 | A |
| **SWE-Gym** | 2.4K | **150M** | task licenses | A |
| SWE-smith | 26K | 120M | task licenses | A |
| OpenHands rollouts | 10K | 90M | mixed | A |

SWE-Gym is the samples-vs-tokens lesson in one row: sorts **last** on samples, **first** on tokens.
Long trajectories are where agentic tokens actually live.

### Reasoning & math — 85.1B tokens (but 78B of it is V4 lineage)
| dataset | samples | tokens | license | tier |
|---|---|---|---|---|
| OpenThoughts2 | 1.1M | 3B | Apache-2.0 | A/D |
| OpenMathReasoning (NVIDIA) | 3.2M | 2B | CC-BY-4.0 | A/D |
| NuminaMath | 860K | 500M | Apache / CC-BY | A |
| OpenR1-Math (R1-distilled) | 220K | 1.6B | Apache-2.0 | D |
| AON (V4 corpus) | 40M | 78B | V4 lineage | A |

→ **open, non-V4 reasoning supply is only ~7.1B tokens.**

### Long-context — 100B tokens
Repo-packed code (32K+) 1.5M / 60B / permissive / B · Book-length corpora (packed) 400K / 40B / mixed / B

### Indic — 275.9B tokens
| dataset | samples | tokens | license | tier |
|---|---|---|---|---|
| Sangraha (verified) | 40M | **64B** | CC-BY-4.0 | A |
| Sangraha (unverified) | 15M | 24B | CC-BY-4.0 | B |
| Sangraha (synthetic) | 90M | **162B** | CC-BY-4.0 | C (synth) |
| IndicCorpV2 | 10M | 20.9B | CC-BY / mixed | B |
| BPCC (parallel) | 22M | 3B | CC-BY-4.0 | C |
| Samanantar | 49.7M | 2B | CC0 / CC-BY | C |

### General web & STEM — ~4,837B tokens
FineWeb-Edu 1.3B/1.3T/ODC-By/B · DCLM-Baseline 2.6B/2.6T/mixed/B · peS2o 40M/42B/ODC-By/A ·
proof-pile-2 5M/55B/mixed/A · D1 Web-Foundation (V4) 200M/164B/B · D2 Web-Diverse (V4) 780M/627B/B ·
D4 STEM (V4) 60M/49B/B

---

## 4. Benchmarks (widget 2) — what each one actually scores, and the loss map

Colour code used throughout: **green = supervised (loss)**, **grey = masked context**, **violet = reward only**.

**Agentic & tool-use**
- **SWE-bench Verified** — 500 human-validated GitHub issues + real repo; write a patch. % resolved (pass@1).
  Green: assistant reasoning + patch. Grey: issue text, repo files, test output. Violet: hidden tests pass.
- **SWE-bench Live / Pro** — fresher, contamination-resistant, harder. Same shape.
- **Terminal-Bench** — real shell. Green: reasoning + commands. Grey: shell output. Violet: harness probe.
- **tau-bench / tau2-bench** — tool-agent-user under a written policy (retail, airline). Metric: pass^k.
  Green: assistant messages + tool calls. Grey: simulated-user turns + every tool return.
- **BFCL v3** — Berkeley Function-Calling Leaderboard, single/parallel/multiple/multi-turn. AST-match accuracy.
  Green: the emitted call only. Grey: schema, user, tool return.
- **WebArena / WorkArena** — self-hosted sites, enterprise workflows, goal-state match.
- **GAIA** — ~466 multi-step general-assistant questions needing tools; exact match; 3 levels.
- **BrowseComp** — hard verifiable web browsing (this is the grant-hunting pattern in widget 5).
- **OSWorld** — ~369 real computer-use tasks across desktop apps; screenshots grey, actions green.

**Coding** — LiveCodeBench (rolling window, pass@1) · Aider Polyglot (~225 exercises, strict search/replace
diff, % solved) · Codeforces (ELO).

**Reasoning & math** — AIME 2024/2025 (15 problems/yr, integer 0–999) · FrontierMath (Epoch AI,
research-level) · GPQA Diamond (198 google-proof grad science MCQ) · HLE (~2500 expert questions).
In distillation data: CoT + final answer green, problem grey. RLVR adds a violet reward, no extra token loss.

**Indic** — MILU (AI4Bharat, multi-task, accuracy per language) · IndicGenBench (Google; **29 Indic
languages, 13 scripts, 4 families**; summarization / MT / cross-lingual QA; ROUGE / chrF / exact-match).

---

## 5. Training lifecycle (widget 4) — five stages, five loss shapes

| # | stage | budget | loss / signal |
|---|---|---|---|
| 1 | Pretraining | ~95% of tokens | next-token CE on **every** token; trillions of tokens |
| 2 | Mid-training / Anneal | ~2% (1–5%) | CE on every token, but **only the reserved best mix**, LR decayed |
| 3 | SFT | <1% | CE on **response only**, prompt masked; ~10K–1M examples |
| 4 | Reasoning training | <1% | long-CoT SFT: CE on CoT+answer. Then **RLVR: no token loss**, verifier reward via policy gradient / GRPO; ~10K–100K prompts |
| 5 | Preference alignment | <1% | (prompt, chosen, rejected) triples |

Reasoning is taught **after the base model exists** — long traces are not simply poured into pretraining.
S5's reserved reasoning data is the raw material for **Sessions 17–18**.

---

## 6. Agentic trajectory (widget 5) — the masking rule, concretely

Task: find US grants matching a cryo-EM motion-correction project, resolve awardees, decide who could buy
a spare high-throughput GPU rig. Ten steps:

1. user request — **masked**
2. assistant plan — **trained**
3. assistant tool call `search_grants({...})` — **trained**
4. tool observation (3 grant ids) — **masked**
5. assistant reasoning + `lookup_awardee(...)` — **trained**
6. tool observation (PI, org, $1.85M) — **masked**
7. assistant `fetch_org_profile(...)` → **ERROR 429 rate_limited** — the **call is trained**, the error text is environment output
8. assistant recovery + `web_search(...)` — **trained**
9. tool observation — **masked**
10. assistant final answer — **trained**

Rule: tool observations are ground truth the model may read and reason over but must **never** be trained to
reproduce. Applying loss to them teaches the model to *invent* tool results instead of calling the tool.
Note step 7: you **do** train the failed call, because recovery behaviour is the thing being taught.

---

## 7. Reasoning-effort dial (widget 6)

One problem — "how many integers 1..1000 are divisible by 3 or 5?", answer **467** — at four depths:
LOW 2 steps (near-direct) · MEDIUM 4 steps · HIGH 6 steps (derive + check) · ULTRA 14 steps
(deliberate, two independent verifications, endpoint re-check).

Accuracy model in the widget: `BASE=0.34`, `PLATEAU=0.955`, `TAU = ULTRA_TOK*0.18`, with the comment
*"high tier already sits near the plateau"* → `ULTRA_GAIN` is small, i.e. **ultra is mostly wasted tokens**.
That is the whole economic argument for a dial: the model must be trained to *obey* the requested effort,
not to always maximise it.

Consequence for the mixture: reasoning is **not one slot** — it is a *distribution of trace lengths*
across math, code and general problem solving. Short traces first, longer traces later.

---

## 8. OPUS — online data selection (widget 3)

Reference given in-widget: **arXiv:2602.05400, ICML 2026 Oral**. Used in the V4 production run.

Mechanism (as taught): keep an exact frozen copy of the model. Run the **golden proxy** (benchmark copies)
through it, compute the loss and back-propagate, but **do not update weights** — record which weights carry
the large gradient magnitude. Then take the candidate batch (e.g. 1024 samples), score each using only its
**first ~512 tokens**, and keep the fraction whose update best aligns with the proxy direction.

Widget implementation — domains as vectors in `[English, Reasoning, Sovereign]` space:

```
web       [0.92, 0.20, 0.08]      code    [0.78, 0.52, 0.06]
reasoning [0.50, 0.82, 0.10]      agentic [0.34, 0.50, 0.30]   (scarce)
indic     [0.18, 0.26, 0.92]                                   (scarce)

PROXY english  [0.70, 0.53, 0.33]      PROXY balanced [0.577, 0.577, 0.577]
score = dot(domain_vec, proxy_vec) + noise      LANE_FLOOR = 0.08
candidate stream sampled at web 34 / code 22 / reasoning 16 / agentic 16 / indic 12
effMult = clamp(6.0 * (0.40/keepFrac)^0.8, 1.4, 14)
```

V4 production numbers: kept **~40%** of candidates → **~6×** effective tokens (**~200B actual → ~1.2T
effective**) at **~4.7%** compute overhead. Its proxy was English-heavy — **cosine 0.876** with the English
web band — so it under-valued Indic. Fix: **Always-On lane injects 8% of every batch, invisible to OPUS.**
V4 protected Indic only; **V5 extends protection to Indic, agentic and reasoning**.

Why agentic needs the floor even though agentic benchmarks exist in the proxy: scoring reads only the
**first ~500 tokens** of a sample, and an agentic trace opens with a user request and tool logs — which look
like low-quality text. The selector cuts it before it ever reaches the valuable part.

Final design: **an aggressive selector operating above a protected capability floor.**

---

## 9. Curriculum (widget 7) — five stage anchors, interpolated

Each profile sums to 100. Note this widget has **no separate agentic band** (it folds into code).

| stage | sub | gen | code | reason | lctx | indic | stem |
|---|---|---|---|---|---|---|---|
| 1 Seed | warm start | 55 | 15 | 3 | 2 | 15 | 10 |
| 2 General | broad base | 45 | 20 | 6 | 3 | 16 | 10 |
| 3 Reasoning | code + logic | 25 | 28 | 18 | 5 | 14 | 10 |
| 4 Long-context | stretch ctx | 18 | 30 | 16 | 18 | 12 | 6 |
| 5 **Anneal** | low-LR cool | 8 | 22 | 20 | 10 | **30** | 10 |

Bands whose Tier-A best is **reserved for the cooldown**: `indic`, `reason`.

Difficulty ladder: **B0** Nursery (simple children's sentences) · **B1** Grade-school · **B2** High-school ·
**B3** Undergraduate · **B4** Graduate · **B5** Research / PhD.

The anneal is the highest-leverage stage: small phase, decayed LR, disproportionate capability gain — but
**only if the reserve survives the main run.** If the selector eats the best Indic/agentic/reasoning data
early, there is nothing special left. So the anneal begins as an S5 *allocation* decision.

---

## 10. Mixture-shift instability (widget 8)

V4 fact: a sudden increase in the Hindi share interacted with **frozen embeddings** → gradient norm jumped
**~150×** over a short stretch. Enough to destroy a run.

Widget model:
```
effSharp  = sharpness / (1 + bandB * C_W)
effFrozen = frozen ? 1 + (F-1)/(1 + bandB * C_F) : 1
peakFactor = 1 + effSharp * A * effFrozen
A = 7, F = 21.4, C_W = 1.5, C_F = 2.0
RED_T = 3.0   (above 3x baseline the run is treated as unstable)
STEPS_PER_B = 26   (steps spanned by 1B tokens of warmup band)
```
Sanity: sharpness 1.0, frozen on, no warmup → `1 + 7*21.4 = 150.8×`. ✓

Mitigation: **never change the mixture in one hard step.** Blend every transition across a warmup band of
several **billion** tokens. This is also why architecture and mixture are frozen before the main run.

In class Rohan added the operational target: **healthy grad-norm ≈ 0.2**; he showed V4 runs where it drifted
up and stabilised around 35, which he called "not a great idea".

---

## 11. The assignment (verbatim requirements)

Draft the **mixture-and-curriculum plan for V5 as a written specification**, specific enough to defend.

Must contain:
1. A defended **share of budget for every capability lane**.
2. The **Indic split across verified / unverified / translated / synthetic** tiers — not a single headline number.
3. **Agentic, reasoning and long-context slots named explicitly**, each pointed at datasets from the inventory.
4. The **protected always-on floor** the selector is not allowed to cross.
5. The **anneal reserve** held back for the cooldown.
6. **Difficulty bands and reasoning-length bands, with a concrete example for each.**
7. Each lane **tied back to the benchmarks it is meant to win**.
8. Every lane **sized against real supply**, stating plainly where a share can only be reached by
   **repetition** or **synthesis**. Wishful accounting is explicitly penalised.
9. A commitment to **proxy runs at 1B and 3B scale** before any number is trusted at full scale.

Grading: reviewed as if a data-curriculum reviewer sat across from you and pushed on every number. Reasoning
quality and evidence carry the grade. **A tightly argued short plan scores well; padding earns nothing.**
Highest marks: the plan written as a **testable hypothesis** with a concrete 1B/3B proxy experiment and the
**named metric that would confirm or refute** the mixture. Very highest: **actually run the proxy and bring
the numbers back.**

Gating: plans are reviewed only once the team has met the **data-gating threshold** — cleaning continues
toward the cumulative target, now aimed at the slots the mixture shows to be starved.

**Submission: a link to a GitHub repo README.md.** (Explicitly *not* a Netlify app or widget this time.)

---

## 12. Where S5 sits

Session 3 → provenance tiers. Session 4 → clean, deduplicated, provenance-stamped shards.
**Session 5 → the recipe.** Session 6 → the system that executes it: sharding, sequence packing,
deterministic shuffling, pause-and-resume, dataloader throughput. Sessions 17–18 → reasoning training.

---

## 13. MEASURED — results from driving the widgets in a real browser

Playwright + Chromium, widgets loaded from `widgets/*.html`. These are computed outputs, not readings
of the source. Scripts: `explore1.py`, `explore3.py`, `explore4.py`, `explore8.py` in the session scratchpad.

### 13.1 Feasibility matrix — supply verdict per lane, preset × run size

`pretrain` preset:

| lane | 1T | 2T | 5T | 10T |
|---|---|---|---|---|
| Code | covered | covered | needs repetition | needs repetition |
| **Agentic** | **must synthesize** | **must synthesize** | **must synthesize** | **must synthesize** |
| Reasoning | covered | needs repetition | needs repetition | must synthesize |
| Long-context | covered | needs repetition | needs repetition | must synthesize |
| Indic | covered | needs repetition | needs repetition | must synthesize |
| STEM | covered | covered | needs repetition | must synthesize |
| General web | covered | covered | covered | covered |

**A 1T run is the only scale at which the pretrain preset is nearly feasible from real supply** — every
lane except agentic is covered. At 2T three lanes need repetition; at 10T only general web survives intact.
Agentic is "must synthesize" at *every* scale, including 1T.

The composer's own funding threshold (8%) reports on the `pretrain` preset:
> Funded: Code, Indic, STEM, General web · **Starved: Agentic, Reasoning traces, Long-context**

i.e. by the widget's own definition the default pretraining mixture starves the three lanes the project
most wants — deliberately, because they are concentrated in the anneal. The `naive` preset additionally
trips **"PROTECTED FLOOR BREACHED"** (Indic 6% < 12%).

⚠️ **Widget artifact:** the supply check computes `demand = share × run size` using the *full* run size even
when the anneal preset is loaded. The anneal is only ~2% of tokens, so its true demand is ~50× smaller than
displayed. At a 2T run the widget shows anneal Indic demanding 560B; the honest figure is 28% × (2% × 2T)
= **11.2B**, comfortably inside the 64B verified pool. Do not quote the anneal column as-is.

### 13.2 OPUS — measured share of trained tokens, 60 iterations per configuration

Keep fraction fixed at the V4 production value of 40%.

| proxy | always-on | Indic share (mean / median) | iterations at 0% | Agentic share (mean / median) | iterations at 0% |
|---|---|---|---|---|---|
| English-heavy (V4) | OFF | **0.00% / 0.00%** | **60 / 60** | 0.91% / 0.00% | 58 / 60 |
| English-heavy (V4) | ON | 14.84% / 15.20% | 0 / 60 | 14.51% / 13.25% | 0 / 60 |
| Balanced | OFF | 32.49% / 27.75% | 5 / 60 | **0.41% / 0.00%** | **59 / 60** |
| Balanced | ON | 26.04% / 19.80% | 0 / 60 | 16.28% / 15.85% | 0 / 60 |

Two findings worth carrying into the plan:

1. **Starvation is total, not gradual.** Under the V4 English-heavy proxy with protection off, Indic received
   **zero** tokens in 60 of 60 iterations. The lesson's phrase "falls toward zero" understates it.
2. **Rebalancing the proxy does not save agentic.** Switching to a balanced proxy rescues Indic
   (0% → 32%) but leaves agentic at 0.41% with 59/60 zero iterations. Dot products against each proxy
   explain why:

   | domain | vs English proxy | vs Balanced proxy |
   |---|---|---|
   | code | 0.841 (1st) | 0.785 (2nd=) |
   | reasoning | 0.818 (2nd) | 0.819 (1st) |
   | web | 0.776 (3rd) | 0.693 (4th) |
   | agentic | 0.602 (4th) | **0.658 (5th, last)** |
   | indic | **0.567 (5th, last)** | 0.785 (2nd=) |

   Agentic ranks **last under the balanced proxy and second-last under the English one**. With keep=40% of
   ~12 tiles only the top ~5 survive, and noise is ±0.07 — not enough to close a 0.13–0.18 gap.
   **Therefore no choice of proxy rescues agentic; only the protected floor does.** This is the measured
   justification for V5 extending always-on protection beyond Indic to agentic.

⚠️ Two caveats on this widget: (a) the always-on lane is *stated* as 8% of every batch but the simulation
forces 2 scarce tiles out of ~11–14 candidates into a kept set of ~5, so it **over-delivers to ~15%** of
trained tokens; (b) the multiplier curve `6.0·(0.40/keep)^0.8` clamped to [1.4, 14] still claims **3.1× at
90% keep**, which is not credible — keeping 90% of your data cannot triple its value. It is a stylized
anchor around the one real datapoint (40% → 6×), not a measurement.

Keep-fraction curve as displayed: 10%→14.0× (hits cap) · 20%→10.4× · 30%→7.6× · **40%→6.0× (V4)** ·
50%→5.0× · 60%→4.3× · 70%→3.8× · 80%→3.4× · 90%→3.1×.

### 13.3 Mixture-shift instability — spike multiplier (verified toggle polarity)

Peak gradient norm as a multiple of baseline; **unstable above 3×**.

| warmup band | frozen ON, sharp 1.0 | frozen ON, sharp 0.7 | frozen OFF, sharp 1.0 | frozen OFF, sharp 0.7 |
|---|---|---|---|---|
| 0B | **151×** | 106× | 8.0× | 5.9× |
| 0.5B | 45.8× | 32.4× | 5.0× | 3.8× |
| 1B | 22.8× | 16.3× | 3.8× | **3.0× ✓** |
| 2B | 9.9× | 7.2× | **2.8× ✓** | 2.2× ✓ |
| 3B | 6.0× | 4.5× | 2.3× ✓ | 1.9× ✓ |
| 4B | 4.3× | 3.3× | 2.0× ✓ | 1.7× ✓ |
| 5B | 3.4× | **2.6× ✓** | 1.8× ✓ | 1.6× ✓ |

**The headline finding reframes the V4 incident.** With embeddings frozen at full sharpness, *no warmup band
the widget allows* (max 5B tokens) brings the shift under the 3× stability threshold — 5B still gives 3.4×.
Unfreezing the embeddings is worth roughly **19×** on its own (151× → 8.0×), after which a 2B band is
sufficient. So the ordering of mitigations is: **(1) do not freeze embeddings across a mixture transition;
(2) then blend over ≥2B tokens.** The warmup band is the second lever, not the first.

### 13.4 Reasoning effort — the cost curve, measured

Same problem (answer 467) at each level:

| level | supervised trace tokens | approx accuracy | widget's own verdict |
|---|---|---|---|
| LOW (near-direct) | 37 | 62% | "thin but cheap" |
| MEDIUM (a few steps) | 74 | 77% | "well spent" |
| HIGH (derive + check) | 161 | 91% | "well spent" |
| ULTRA (deliberate + verify) | 346 | 95% | **"wasted effort"** |

HIGH → ULTRA costs **2.15× the tokens for +4 accuracy points**. That is the quantified case for a dial the
model obeys rather than a model that always maximises effort — and the reason the effort tag must be
*trained in*, since a correctness-only reward makes maximum effort the model's dominant strategy.

### 13.5 Agentic trajectory — the supervision yield

Fully revealed, the grant-research trajectory measures **356 supervised tokens of 668 seen = 53% supervised**.

This is a real budgeting constraint the assignment's "size every lane against real supply" demands:
**roughly half of every agentic token you buy carries no loss.** A 40B agentic allocation purchases
~21B supervised tokens. Against 627M real tokens of supply, the effective shortfall is worse than the raw
ratio suggests.

### 13.6 Indic supply — three different numbers, all defensible

The widget's own scarcity card states: *"Agentic slot holds only about **627M** tokens of real supply, and
Indic's trusted, non-synthetic text is about **114B** against **162B** synthetic."*

So three figures apply depending on the claim being made:
- **64B** — Tier A verified native only (Sangraha verified).
- **114B** — all non-synthetic Indic (adds unverified crawl, IndicCorpV2, BPCC, Samanantar).
- **276B** — total including Sangraha's 162B synthetic expansion.

Per-slot totals as rendered: Code **1.1T** · Agentic **627M** · Reasoning & math **85.1B** ·
Long-context **100B** · Indic **276B** · General web & STEM **4.8T**.

### 13.7 Curriculum sweep — interpolated mixture across the run

| t | stage | gen | code | reason | lctx | indic | stem |
|---|---|---|---|---|---|---|---|
| 0.000 | Seed | 55 | 15 | 3 | 2 | 15 | 10 |
| 0.250 | General | 45 | 20 | 6 | 3 | 16 | 10 |
| 0.500 | Reasoning | 25 | 28 | 18 | 5 | 14 | 10 |
| 0.750 | Long-context | 18 | 30 | 16 | 18 | 12 | 6 |
| 0.875 | **anneal reserve engages** | 13 | 26 | 18 | 14 | 21 | 8 |
| 1.000 | Anneal | 8 | 22 | 20 | 10 | 30 | 10 |

The reserve engages at **t ≈ 0.875**, i.e. the last ~12% of the run, and the widget's note names
"the held-back Tier-A Indic, **agentic** and long-reasoning data" — even though this widget has no agentic
band. Indic roughly doubles (12% → 30%) across the final stretch.

---

## 14. Discrepancies found between the transcript and the widgets

Recorded so a reviewer's push does not catch us out. **The widget JS is authoritative.**

1. **Agentic share.** In class the pretrain mixture was described as "16 is going to be agent tool calls".
   The composer's `pretrain` preset has **agentic = 2**, at its floor. The 16 figure is from widget 3's
   *candidate-stream* distribution (`web 34 / code 22 / reasoning 16 / agentic 16 / indic 12`), which is
   the OPUS sampler, not the budget. Easy conflation — both start with web 34.
2. **OPUS multiplier.** Class said "8x efficiency". Lesson page, widget readout and the cited paper all say
   **~6×** at 40% keep (200B → 1.2T). Use 6×.
3. **Indic tier default vs real supply.** The composer defaults the Indic slice to A 40 / B 25 / C 20 / D 15.
   The actual inventory gives verified-A = 64B of 275.9B ≈ **23%**, and Sangraha-synthetic alone is 162B
   ≈ **59%**. Claiming 40% verified is not supportable from the listed sources — this is exactly the
   "wishful accounting" the assignment penalises. (See §13.6: the widget's own scarcity card frames it as
   114B non-synthetic vs 162B synthetic, a third framing again.)
4. **Reasoning supply is mostly our own V4 corpus.** 78B of the 85B is AON (V4 lineage). Open supply ≈ 7.1B.
5. **Agentic is ~64× short at a 2T run.** 2% of 2T = 40B demanded vs **0.627B** real supply. Even the
   *floor* cannot be met from real data. Synthesis is not optional here, it is the lane.
6. **Curriculum widget has no agentic band**; the composer does. Any plan needs to reconcile the two —
   either fold agentic into code in the curriculum view or add the band explicitly.
7. **Supply totals are approximations.** `SUPPLY_T.web = 4.5T` and `stem = 0.25T` (4.75T combined) vs the
   inventory's "General web & STEM" group summing to ~4.84T; STEM-ish rows (peS2o 42 + proof-pile-2 55 +
   D4 49) total 146B, not 250B. Minor, but do not quote both numbers as if they reconcile exactly.
8. **Grad-norm units.** Widget 8 normalises baseline to 1.0 and reports a multiplier; in class the absolute
   healthy target quoted was ≈0.2. Not a contradiction, but do not mix the two scales in one sentence.
