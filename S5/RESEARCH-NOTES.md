# S5 research pass — proxy-run method and published mixture anchors

Scope: the two questions that shape the plan's structure. Every entry is tied to a number we must choose.
Sources are primary (arXiv abstracts / ar5iv full text) unless marked secondary.

---

## Q1 — How to make a 1B/3B proxy run predictive rather than anecdotal

### DoReMi — arXiv:2305.10429 (Xie et al.)
Group DRO over domains, run on a **small proxy model**, produces domain weights used to train the real model.
- **280M proxy → 8B target** (30× gap); weights transfer.
- **2.6× fewer training steps** to reach baseline accuracy; **+6.5 pp** average few-shot downstream accuracy on The Pile.
- On GLaM, **matched domain weights that had been tuned on downstream tasks** — without any downstream knowledge.
- Improves perplexity across *all* domains, even ones it downweights.

Relevance: the canonical "a small run decides the mixture" result. It is the citation that makes a 1B/3B proxy
a *method* rather than a gesture.

### Data Mixing Laws — arXiv:2403.16952 (Ye et al., ICLR 2025)
Fits performance as an explicit function of mixture proportions, then **nests it with scaling laws over training
steps and model size** to predict large-run performance under mixtures never actually run.
- Validated at **1B params / 100B tokens** on RedPajama.
- Optimised mix matched a default-mix run trained **48% longer**.

Relevance: strictly stronger than DoReMi for our purpose — it predicts *unseen* mixtures, so we do not have to
run one proxy per candidate. This is the tool for defending a specific set of percentages.

### Annealing-as-evaluation — Llama 3, arXiv:2407.21783 (verified against ar5iv full text)
The cheapest usable protocol, and the one Meta explicitly prefers:
> "annealing the learning rate of a 50% trained Llama 3 8B model linearly to 0 on **40B tokens**", with
> **"30% weight to the new dataset and the remaining 70% weight to the default data mix."**

They state annealing lets you judge the value of small domain-specific datasets and that it is **more efficient
than running scaling-law experiments for every small dataset**.
- Measured: **GSM8k +24.0%**, **MATH +6.4%** on the 8B model.
- ⚠️ **"The improvements on the 405B model are negligible"** — Meta attribute this to the flagship's strong
  in-context learning. **Anneal gains are scale-dependent and shrink as the base model gets strong.**

### OLMo 2 — arXiv:2501.00656 (partly secondary)
Open analogue, fully replicable. **Dolmino Mix 1124**: an 843B-token pool sampled into **50B / 100B / 300B**
anneal mixes, each **50% high-quality web + 50% domain-specific** (academic, Q&A, instruction, math workbooks,
synthetic and human). 7B is mid-trained on the 50B mix; 13B/32B anneal on the larger ones. For 7B they anneal
**3× on 50B tokens with different data orders and average the resulting models**.
Sources are assessed independently via **"microannealing"** — the open-source name for the Llama 3 protocol.

### Two-phase pretraining — arXiv:2412.15285 (NVIDIA)
Directly validates designing a blend small and scaling it up:
> blends designed "using downsampled data at a smaller scale of **1T tokens**", then scaled to a
> **15T token horizon and 25B model size**.
- Two-phase ordering beats random data ordering by **+3.4%** and the natural token distribution by **+17%** average accuracy.
- Explicitly notes mixture scalability to longer horizons "remains underexplored due to limited disclosure by
  model developers" — i.e. the gap Rohan described is acknowledged in the literature.

### Recommendation for our plan
**Make the proxy experiment annealing-based, not a from-scratch 1B run.** We cannot afford one full 1B run per
candidate mixture, and Llama 3 states outright that anneal-evaluation is cheaper than per-dataset scaling laws.

Protocol to adopt (adapted from Llama 3 / OLMo 2 microannealing):
1. Take a mid-run checkpoint (≈50% trained) at 1B or 3B.
2. Anneal LR linearly to 0 over a fixed small budget, **30% candidate lane / 70% default mix**.
3. Read the per-lane benchmark delta as the accept/reject signal.
4. Randomise data order and repeat where the signal is marginal (OLMo 2 averages 3 runs).

**State the caveat explicitly**: Llama 3 found anneal gains negligible at 405B. Our target is far smaller, so the
signal should hold — but the plan must say that this protocol measures *anneal-stage* value and that its
transferability degrades as base-model capability rises. A reviewer will push on exactly this.

---

## Q2 — Published mixtures to anchor against

| model | general web / knowledge | math + reasoning | code | multilingual | notes |
|---|---|---|---|---|---|
| **Llama 3** (final mix) | ~50% | 25% | 17% | 8% | arXiv:2407.21783, verbatim |
| **Nemotron-4 15B** (phase 1) | 70% English | — | 15% | 15% | 8T tokens, then **1T continued** on a different blend (secondary) |
| **ERA V5 composer** `pretrain` | 34% web + 12% STEM | 6% reason traces | 24% | 16% Indic | + 2% agentic, 6% long-context |
| **ERA V4 actual** | 72% → 18% | 7% → 39% sci/math | 13% → 35% | 8% pinned always-on | a schedule, not a constant |

### What the comparison actually says

- **Indic at 16% is aggressive but not unprecedented.** It is 2× Llama 3's multilingual (8%) and comparable to
  Nemotron's 15%. But those cover *all* languages with enormous supply; ours is Indic-only against 276B total /
  114B non-synthetic / 64B verified. The defence has to be the differentiator argument plus the protected floor,
  not the precedent.
- **Code at 24% is above Llama 3's 17%**, justified by the coding/agentic target. Defensible.
- **Reasoning is not as under-funded as it looks.** Llama 3's 25% is math+reasoning *content*; ours splits into
  reasoning traces 6% + STEM 12% = **18%**. State this explicitly or a reviewer will read 6% and object.
- **Nobody publishes an agentic lane.** This confirms the session's claim directly. Our 2% floor therefore has
  **no external anchor** and must be justified from supply (627M real tokens) and from the SWE-bench-shaped target.
- **Everyone changes the blend mid-run.** Llama 3, Nemotron-4, OLMo 2 and our own V4 all shift distribution and
  reserve quality for late. A flat mixture is the outlier position and would need defending, not the reverse.

---

## The biggest actionable finding: long-context is over-budgeted by 1–2 orders of magnitude

**Data Engineering for Scaling Language Models to 128K Context — arXiv:2402.10171**
- **"500 million to 5 billion tokens are enough to enable the model to retrieve information anywhere within the
  128K context."** Continual pretraining on 1–5B tokens is effective.
- Mechanism: long-context ability is **"mostly already acquired through large-scale pretraining"**; extension is
  lightweight and is primarily a *data engineering* problem.
- **Naively upsampling long documents from books "gives suboptimal performance."** A balanced domain mixture
  matters more than length upsampling.

**Against our composer**: the `pretrain` preset gives long-context **6%**, which at a 2T run is **120B tokens** —
**24× to 240× more than the literature says is required.**

Implication: cutting long-context to ~0.5–1% frees on the order of **100B tokens** to redeploy into the lanes the
supply check shows starved (agentic synthesis, verified Indic, reasoning). It also removes the "needs repetition"
verdict on long-context, whose real supply is only 100B.

Caveats to state when we use this:
- The result is for extension to 128K by *continued* pretraining from an already-capable base. A different target
  context, or training long-context in from early, changes the cost.
- Batch homogeneity (all sequences in a batch share a length) means long-context is staged anyway, which is
  consistent with a small, late allocation rather than a share spread across the run.

---

## Sources
- DoReMi — https://arxiv.org/abs/2305.10429
- Data Mixing Laws — https://arxiv.org/abs/2403.16952
- Llama 3 Herd of Models — https://arxiv.org/abs/2407.21783 (ar5iv full text)
- OLMo 2 — https://arxiv.org/abs/2501.00656
- Two-Phase Pretraining — https://arxiv.org/abs/2412.15285
- 128K Context Data Engineering — https://arxiv.org/abs/2402.10171
- MiniCPM (WSD scheduler) — https://arxiv.org/abs/2404.06395
- Nemotron-4 15B — https://arxiv.org/pdf/2402.16819 (blend via secondary sources)
