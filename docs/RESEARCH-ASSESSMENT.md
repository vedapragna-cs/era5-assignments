# Literature assessment — how this should be done

A survey of published work bearing on the S5/S6 training-data design, and what it says we should
change. Searched August 2026.

**Epistemic status, stated up front.** Most of this is drawn from search results and abstracts, not
full reads. **Four papers have been read in full** — Luo et al. ICLR 2026 (§4), RHO-1 (the reference-model
section), PaLM §5 (§2), and the Apple mixture scaling law (§3) — because it bears
directly on a decision S5 has already taken; its numbers, tables and caveats below are from the PDF,
not the abstract, and the full read materially sharpened the conclusion. Everywhere else, where a
claim drives a recommendation I give the source and the quoted finding so it can be checked, and
where I am extrapolating I say so. Several sources post-date my training data and are treated as
reported rather than known. **Nothing outside those three should change a number in the plan until the paper
behind it has been read the same way** — the rest of this document exists to say *which* papers are
worth that time.

Reading them changed four things I had written from abstracts, including one where the full read
reversed my recommendation, and each correction is marked in
place. The remaining sections are still abstract-level and should be treated as leads.

---

## The single highest-leverage change: run a reference model

**Read in full:** RHO-1, *Not All Tokens Are What You Need* (NeurIPS 2024,
[2404.07965v4](https://arxiv.org/abs/2404.07965)). The full read is stronger than the abstract on the
point that matters to us, and it corrects a caveat I wrote from the abstract.

### The token-level finding that reshapes the V6 payload

They continually pretrained TinyLlama-1B on 15B OpenWebMath tokens, checkpointing every 1B, and
tracked **individual token loss trajectories** on a 320k-token validation set. Tokens fall into four
categories:

| trajectory | share | meaning |
|---|---:|---|
| **L→L** consistently low | **51%** | already learned — training on them is near-wasted compute |
| **H→L** high then falling | **26%** | the tokens actually being learned |
| **L→H** rising | 12% | getting worse |
| **H→H** persistently high | **11%** | *"likely due to high aleatoric uncertainty"* — irreducible |

**This is the finding S6's V6 payload most needs and does not currently account for.** The plan is:
record per-token perplexity, find the high-perplexity tokens, collect more data like them for V6.

But 11% of tokens are persistently high-loss *because the uncertainty is irreducible* — noise,
ambiguity, arbitrary strings. Collecting more data that looks like those tokens buys nothing. And
they are indistinguishable from the 26% H→L tokens **if you only record the level**. The two have
opposite implications for data collection and the same instantaneous signature.

**So a perplexity snapshot is not sufficient — the trajectory is the signal.** Concretely: the same
tokens must be scored at multiple checkpoints, and the V6 payload should record the *slope*, not just
the value. Our learning ledger already carries `model_age_tokens` and `repeated_pass_number` per
observation, so trajectories are reconstructible — but only if the same tokens are actually re-scored,
which the current demo does not do systematically. That is a cheap change now and impossible later.

### The method, and the correction to what I wrote from the abstract

Selective Language Modeling: train a reference model on curated high-quality data, score every token
in the pretraining corpus by reference loss `L_RM(x_i) = −log P(x_i | x_<i)`, then train **only on
tokens with high excess loss** between the training model and the reference. *"Tokens with high excess
loss are more learnable and better aligned with the desired distribution, naturally excluding tokens
that are either irrelevant or of low quality."*

| setting | result |
|---|---|
| math continual pretrain, 15B OpenWebMath | **+16%** avg few-shot on GSM8k/MATH; baseline accuracy reached **5–10× faster**; RHO-1-7B matches DeepSeekMath-7B using **15B tokens vs 500B** |
| **general continual pretrain, 80B tokens, TinyLlama-1.1B** | **+6.8% average across 15 benchmarks**, *"gains over 10% in code and math"* |
| **self-reference** (no high-quality reference corpus) | **+3.3%**; on Proof-Pile-2, 24.7 → **26.5** while training on **36B instead of 52B tokens** |

**Correction to my earlier note.** I wrote that whether this transfers beyond math *"is not
established."* That was wrong — §3.3 is exactly that experiment, at 80B general tokens across 15
benchmarks. The caveat I should have written is narrower and is stated below.

**And the reference model is not a hard prerequisite.** §3.4 shows self-referencing: train the
reference on the pretraining corpus itself and score with reference loss `L_RM` **intersected with**
next-token entropy `H_RM`. Their framing is the useful one — *"the key factor is not scoring the
desired distribution but filtering out noisy tokens."* This matters for us because S5 has no obvious
high-quality Indic reference corpus; self-reference sidesteps that.

### Why this bears directly on OPUS

OPUS as specified scores a **batch** by gradient alignment and accepts or rejects it whole. RHO-1's
ablation says the useful signal is at the **token** level and that the two are not interchangeable:
*"selected-token pretraining substantially lowers downstream loss, while traditional pretraining's
effect on downstream loss is less pronounced despite initial loss reductions"*, and the loss on
*unselected* tokens correlates **negatively** with downstream accuracy (their Figure 7). A batch is a
mixture of 51% already-learned tokens, 26% learnable ones and 11% noise; accepting or rejecting it
whole cannot express that.

Their conclusion is the blunt version: *"reducing loss across all tokens is not imperative for
improved model performance."* S5's OPUS design implicitly assumes the opposite.

### Limitations — theirs, stated plainly, and they are real for us

- **The model converges toward the reference model's domain.** *"Training exclusively with SLM leads
  to quickly convergence to the domain focused by the reference model, accompanied by a significant
  rise in the loss of unselected tokens."* For a seven-lane model whose whole point is breadth
  (Indic + code + agentic + reasoning), the reference model silently becomes the definition of what
  you become. This is a bigger risk for us than for a math-focused run, and it argues for a
  *per-lane* reference rather than one global reference.
- **Scale is unverified where we need it.** *"We have only verified the effectiveness of our method on
  smaller models (≤7B parameters) and smaller datasets (<100B tokens) … it's possible that very large
  models trained on extensive corpora may naturally develop this inductive bias."* We plan 3T tokens.
  The authors are explicitly uncertain whether the benefit survives.
- Selection ratio matters and has an interior optimum (their Figure 9 sweeps 40–100%); it is a
  hyperparameter, not a free choice.

**Net recommendation.** Adopt the *measurement* immediately — record reference loss and token
trajectories, because neither can be reconstructed after the run. Treat the *intervention* (training
only on selected tokens) as a pre-registered experiment with a per-lane reference, not a settled
design, because the domain-narrowing risk lands precisely on our differentiator.

---

## Findings by subsystem

### 1. Packing and attention — the plan is under-specified, and the fix is measured

**What the plan says.** EOS marks a boundary; the model learns not to attend across it because
back-propagation corrects it.

**What the literature says.** *Analysing The Impact of Sequence Composition on Language Model
Pre-Training* ([2402.13991](https://arxiv.org/html/2402.13991)) measures the alternative directly:
intra-document causal masking (structurally forbidding cross-document attention) scores **44.48%
average vs 43.38%** for standard causal masking, and *"significantly improves language modeling,
in-context learning, and knowledge memorization by eliminating cross-document distractions."*

**Verdict: the plan's answer is not wrong, it is just leaving ~1.1 points on the table.** EOS teaches
the convention; a segment mask enforces it. Both are cheap. We implement the segment mask and the
demo verifies zero cross-document pairs are permitted.

**And there is a further gain the plan does not touch.** The same paper finds that packing *related*
documents together — retrieval-based sequence construction rather than arbitrary concatenation —
yields **ICL +11.6%, knowledge memorization +9.8%, context utilization +7.2%**. Our packer chooses
bins by *length*. Nothing in S5 or S6 mentions choosing them by *relatedness*. This is the largest
unclaimed improvement I found in the whole survey.

**Cost note:** intra-doc masking *"decreases pre-training efficiency"*, mitigable with FlashAttention2.
That trade needs measuring on our stack before committing.

### 2. Loss spikes — an interaction, not a data property, and it can poison the V6 payload

**Read in full:** PaLM ([2204.02311](https://arxiv.org/abs/2204.02311)), §5 *Training Instability*.
It is one paragraph and every clause in it matters.

> *"For the largest model, we observed spikes in the loss roughly 20 times during training, despite
> the fact that gradient clipping was enabled. These spikes occurred at highly irregular intervals,
> sometimes happening late into training, and **were not observed when training the smaller
> models**."*

Their mitigation: restart ~100 steps before the spike and skip ~200–500 batches covering the spike
window. *"With this mitigation, the loss did not spike again at the same point."*

**The ablation is the load-bearing part, and it is what changes our design:**

> *"We do not believe that the spikes were caused by 'bad data' per se, because we ran several
> ablation experiments where we took the batches of data that were surrounding the spike, and then
> trained on those same data batches starting from a different, earlier checkpoint. In these cases,
> we did not see a spike. This implies that spikes only occur due to the combination of specific data
> batches with a particular model parameter state."*

**Consequence for the V6 payload.** Our learning ledger classifies a shard `harmful` when loss rises
across its exposure. PaLM says a spike is an *interaction between batch and parameter state*, so a
shard that happens to coincide with one gets permanently labelled harmful and may be dropped from the
next corpus **on the basis of an artifact**. The most valuable output of this session can be poisoned
by its most dramatic event.

*Action, and PaLM supplies the exact procedure:* record a spike as a `run_event`, never attribute it
to a shard; flag any shard whose harmful classification derives from a spike window as
`attribution_uncertain`; and resolve it by **re-training on those same batches from a different,
earlier checkpoint** — which is precisely the ablation PaLM ran. If the spike does not reproduce, the
data is exonerated.

**Two corrections to what I wrote before reading it.**

1. I said the ledger needs "skip-and-**reinject**". PaLM skipped; they do **not** say they re-injected
   the skipped batches. Re-injection is my inference from "the data is not bad", and it is a design
   choice with its own risk — re-serving a batch that previously interacted badly could reproduce the
   interaction if the parameter state has not moved far enough. The honest specification is
   `batches_skipped` as a recorded event with an explicit, separately-decided re-injection policy —
   not one operation.
2. **Spikes were not observed in smaller models.** Our demo will never exhibit one, so any
   spike-handling we build is *untestable at our scale*. That does not make it wrong to build — the
   ledger schema has to exist before the run that needs it — but it must be labelled as untested
   rather than demonstrated, and it is the one part of this system I cannot put a negative control
   behind.

### 3. The 4-epoch ceiling is the wrong *shape*, but relaxing it is not obviously safe at our scale

**Read in full:** *Scaling Laws for Mixture Pretraining Under Data Constraints*
([2605.12715](https://arxiv.org/abs/2605.12715), Sedova/Seto/Schluter/Ablin, Apple + ITU, May 2026).
**2,000+ training runs.** The setup is structurally identical to ours — a scarce but valuable target
mixed with abundant generic data, tested on multilingual (German, French, Swahili), domain-specific
(peS2o, Wikipedia, Maths) and quality-filtered mixtures.

**This paper corrects me in the opposite direction from what I expected.** I wrote that the 4-epoch
ceiling was "too conservative" and that Indic/reasoning therefore had "comfortable headroom." Half of
that is supported and half is not.

#### What it confirms

**Repetition tolerance really is far higher in a mixture.** Optimal repetition factor `r` reaches
**15–20**, and up to **r ≈ 30** for the scarcest sets — *"substantially exceeding the widely used
rule-of-thumb of <4 epochs for single-source data-constrained training (Muennighoff et al., 2023)."*

The mechanism is the part worth internalising: *"the generic domain … provides a continuous supply of
fresh tokens that effectively regularizes training and allows the model to absorb far more
target-domain repetition before degradation sets in."* In their scaling law this is explicit — the
generic term `(1−h)·D_total` never saturates, while the target contribution saturates at `(1+r₁)·D_target`.

**Overfitting onset is governed by `r` alone.** *"The same target weight can be safe or harmful
depending on the available dataset, but the same `r` consistently triggers degradation regardless of
how it is reached"* — holding across data sizes, model scales and domains.

**This is the finding that reshapes S5's constraint.** S5 applies one epoch ceiling to all seven
lanes. But the controlling variable is each lane's *own* repetition factor, and the scarcer the lane
the higher its optimal `r`: in their multi-domain experiment the 5×-smaller target reached **r ≈ 30**
while the 5×-larger stayed **below r ≈ 7**. Applying `≤4` uniformly to agentic (0.63B supply) and web
(4,691B supply) is the wrong shape regardless of the level.

And usefully: **you do not need to hit `r` precisely.** Their 10% compute-confidence bands are wide —
*"even rough estimates of optimal r yield near-optimal target loss."*

#### What it complicates, and this is the correction

**Optimal repetition *decreases* with model size, and so does the optimal target weight.** For a 50M
target dataset, the weight needed to reach optimal `r` falls from **9.5% at 101M parameters to 1.9%
at 539M** — roughly 5× lower for a 5.3× larger model. Their summary: larger models *"memorize the
data more easily"*, so *"the optimal operating window narrows with model size."*

S5 plans ~30B parameters. Every trend in this paper that matters runs **against** relaxing the ceiling
at our scale, not for it.

**And the authors close the door explicitly:** *"our experiments span ∼100M–900M parameters; whether
the fitted exponents extrapolate to multi-billion-parameter models remains an open question."*

So the honest revision of what I wrote:

| my earlier claim | corrected |
|---|---|
| the 4× ceiling is too conservative | the ceiling is the wrong **shape** — it should be per-lane `r`, not a uniform epoch count |
| Indic and reasoning have comfortable headroom | **unsupported.** Optimal `r` falls with model size, and 30B is 30× beyond their largest run |
| "Indic and reasoning are the tight lanes" may be an artifact | it may be — but this paper cannot settle it, and points the other way on scale |

The one conclusion that survives untouched: **agentic at 104 epochs is infeasible under any reading**,
including r ≈ 30 for the scarcest lanes.

#### Two things we should take regardless

**Broad filters can beat more repetition.** They vary a quality threshold to trade quality for
quantity within a domain and find a crossover where the broader band (Q90-99) overtakes the narrower
one (Q95-99). This is a direct argument for S5's Indic tier broadening (A 28% / B 30% / C 2% / D 40%)
rather than concentrating on verified Tier A — and it supplies the mechanism S5 argued for on
intuition. It also gives a decision rule: the crossover is measurable, not a matter of taste.

**Weight constrained lanes proportionally to their supply when data is scarce.** Among *multiple*
constrained targets, *"proportional weighting outperforms equal weighting when data is scarce, while
equal weighting is better in the data-rich regime"*, and *"proportional weighting becomes increasingly
beneficial as repetition grows."* S5's shares are not proportional to supply — agentic gets 2% on
0.63B while stem gets 12% on 146B. That deviation is defensible (agentic is a *capability
requirement*, not a loss-minimisation choice) but it should be named as a deliberate departure from
what the loss-optimal rule would do, not left implicit.

Best result in their multi-domain setup came from setting **each domain's `r` independently** from the
bilingual law (`r_wiki = 35`, `r_peS2o = 17`), which *"consistently outperforms the best proportional
weighting."* That is exactly the per-lane treatment S5 needs.

#### Limitations, stated by the authors

- *"The scaling law is descriptive: it summarises observed training dynamics but does not prescribe an
  optimal allocation from first principles."*
- Quality-filtering mixtures are the hardest case to predict (test wR² = 0.71, log-error 0.47),
  *"likely because quality tiers interact with the model in ways not fully captured by the effective
  data abstraction."* Our Indic lane is exactly a quality-tiered mixture.
- Size extrapolation is uneven — German wR² = 0.65 at 539M, French 0.93, **Swahili 0.69** and beaten
  by a simpler baseline. Swahili is the closest analogue to our low-resource Indic languages, and it
  is the case that extrapolates worst.

#### What I would actually do with this

Do **not** raise the ceiling on the strength of this paper. Instead:

1. **Re-parameterise the constraint from "epochs ≤ 4 per lane" to "repetition factor `r_lane`"**, which
   is the variable the paper shows actually controls degradation. This is free and strictly better.
2. **Fit their law on our own proxy runs.** They emphasise the practical payoff — *"given only the
   target pool size and compute budget, the law prescribes the mixture ratio … in a single run"* — and
   the fitting procedure is fully specified (Eq. 5.1–5.4). S5's T-series already runs proxy models;
   this adds a fitting target, not new infrastructure.
3. **Treat the model-size trend as the open risk**, and size the Indic lane at 3T with an explicit
   statement that optimal target weight falls with scale and we are extrapolating 30× beyond the
   evidence.

### 4. The anneal design is the one configuration this paper measures as *negative*

**This is the only paper in the survey I read in full** (`arXiv:2511.18903v3`, ICLR 2026, Luo et al.,
Tsinghua/Peng Cheng Lab). It deserved it: the finding lands directly on S5's central bet, and the
abstract understates how specific the result is.

**The mechanism.** The LR is an implicit importance weight. With `θ_{t+1} = θ_t − η_t g_t`, decaying
`η_t` reduces gradient noise *but equally shrinks the step taken in the signal direction*. So data
arriving late arrives when the optimizer is least able to learn from it — **regardless of its
quality**. Putting your best data at the end and decaying the LR to zero are individually sensible
and mechanically opposed.

**The effect size, on uniform-quality data** (their Table 11, 1.5B/30B):

| LR schedule | data order | avg |
|---|---|---|
| Constant | uniform | 47.80 |
| Constant | **ascending quality** | **49.37** (+1.57) |
| Constant | descending | 45.79 (−2.01) |
| Cosine | uniform | 49.13 |
| Cosine | ascending | 49.73 (+0.60) |

The curriculum is worth **+1.57 under constant LR and +0.60 under cosine** — decay removes about
60% of the benefit. (The descending arm at −2.01 is a useful sanity check that the axis is real and
directional, not noise.)

**Now the part that matters for us.** They run a *mid-training* setting — "a large corpus of
average-quality data supplemented by a smaller, high-quality dataset in a later training stage",
citing OLMo 2, which is S5's own anneal anchor. Their Table 2, where `Order` is written as
(main phase, late phase):

| weight avg | order | LR | core | avg |
|---|---|---|---|---|
| — | U,U | WSD | 41.61 | 47.49 *(baseline)* |
| — | U,A | WSD | 41.66 | 47.57 (+0.08) |
| — | A-T | WSD | 42.73 (+1.12) | 48.01 (+0.52) |
| EMA | **U,A** | Const | 41.30 (−0.31) | **47.45 (−0.04)** |
| EMA | **A,A** | Const | **43.61 (+2.00)** | **48.69 (+1.20)** |
| SMA | A-T | Const | 43.90 (+2.29) | 48.69 (+1.20) |

**S5's design is the `U,A` row.** The main run is ordered by *capability stage* (web → reasoning →
long-context), not by quality; the best material — Tier-A Indic, long agentic, long reasoning, B5
difficulty — is held for the anneal. Uniform-ish ordering in phase one, ascending quality in phase
two. That row measures **−0.04 average and −0.31 core**: no benefit, marginally negative.

The authors say it directly: *"applying a curriculum only to the final, high-quality data phase
(EMA + U,A) is not sufficient for optimal results. The superior performance of the A,A schedule over
U,A suggests that applying a curriculum to the initial, lower-quality data phase is also
beneficial."* Their proposed explanation is worth knowing — *"reordering data in the lower-quality
phase can exploit the model's forgetting mechanism to mitigate the adverse effects of toxic samples
that pass through the cleaning pipeline by chance."*

**And CMA's benefit is *larger* in this setting, not smaller**: +1.20 avg / +2.00 core in mid-training
versus smaller gains on uniformly high-quality data, because *"when high-quality data is sparse, each
sample provides a relatively more valuable signal for parameter updates."* Sparse high-quality data
is exactly our situation.

#### The fix is directly implementable and does not disturb the lane budget

Their Appendix D gives a multi-domain algorithm for precisely our problem — several lanes, no shared
quality metric, and mixing ratios that must not move:

```
for each lane i:
    sort ascending by that lane's own quality metric   # within-lane ranking
    assign ordinal rank r_i(x)
    rescale:  R(x) = r_i(x) × N_total / |D_i|          # put lanes on one scale
merge all lanes; sort globally by R(x)
```

Stated properties: within-lane quality ordering is preserved, samples interleave proportionally, and
**the lane mixture ratio stays stable throughout training**. That last property is the one that makes
it adoptable — S5's lane shares are budget-constrained and cannot be perturbed to buy a curriculum.

**Validated at a scale that matters:** 3.2B parameters, 730B base + 156B continual, four domains with
four different quality metrics (DCLM fasttext score, FineWeb score, GitHub stars, duplicate count).
Result **+1.09 core average**, with ARC-C +3.39, BoolQ +3.39, CEval +1.29, MMLU +0.44.

Their working LR configuration at that scale, which is the concrete recommendation:

| | baseline | CDMA |
|---|---|---|
| schedule | WSD, peak 1e-3 | WSD, peak 1e-3 |
| final LR | 1e-5 (final/peak = **0.01**) | **3.67e-4** (final/peak = **0.367**) |
| after decay | — | hold, then **EMA over last 6 checkpoints**, α = 0.2 (~1.67B tokens apart) |
| data order | uniform | multi-domain ascending |

#### What this means for S5, concretely

1. **The anneal reserve is currently placed where the optimizer can least use it.** The reserve is not
   wrong; pairing it with LR → 0 is.
2. **T3 is confounded as designed.** S5's T3 asks whether reserving for the anneal beats spending
   early — but it measures the reserve in the regime where the paper says it cannot pay off. A null
   T3 would be uninterpretable: it cannot distinguish "the reserve earns nothing" from "the LR
   schedule prevented it earning anything". T3 needs an LR arm.
3. **S5's stage design *is* the "folding" strategy**, and the paper tested it: folding helps at low
   peak LR (1e-4) but the benefit *"diminishes — and even reverses — under a higher peak LR (3e-3)"*.
   So the value of our five-stage structure is LR-regime dependent and should not be assumed.
4. **E6 in our own experiment log was testing the right axis.** We pre-registered "does ordering the
   Indic lane internally by difficulty beat a flat shuffle" and reported it under-powered. This paper
   says within-lane ordering is exactly where the benefit lives, and gives the algorithm. E6 should
   be re-run at a scale where it can decide, with the rank-rescaling formula rather than our
   band-sort.

#### Caveats I would not want a reviewer to have to find

- The multi-domain result is **continual pretraining from a base checkpoint**, not from scratch. S5's
  anneal is also continual, so this matches — but the headline 1.5B/30B numbers are from-scratch and
  the two should not be pooled.
- Quality metrics are all cheap proxies (fasttext scores, GitHub stars, duplicate counts). S5 would
  need to nominate one per lane, and for Indic there is no obvious equivalent — this is a real gap
  in adopting the method, not a solved problem.
- *"The benefits of such complex strategies are often confined to smaller-scale experiments with low
  learning rates and can degrade at larger scales and under a high LR regime."* The authors apply
  that caution to prior curriculum work; intellectual honesty says it also applies to theirs, and to
  S5's five-stage design.
- Gains are 1–2% on benchmark averages. Real, reproducible across two scales, and not transformative.
  It is worth a config change and an experiment arm; it is not worth restructuring the plan around.

### 5. Deduplication — the supply figures may be measuring the wrong thing

**What the literature says.** MinHash-LSH at threshold ~0.85 is the production standard. Common Crawl
carries ~50% near-duplicates within a snapshot and 30–40% more across snapshots; dedup *"is typically
the single highest-leverage filtering step … while reducing token volume by 50–70%."* Semantic dedup
catches paraphrases and translated reposts that MinHash misses, but *"also removes legitimate
diversity"* and some ablations show neutral or slightly negative results.

**The question this raises for S5 is sharp and unanswered:** are the audited supply figures
**pre- or post-dedup**? At a typical 50–70% reduction this is not a rounding error. Indic at 157.2B
pre-dedup could be 50–80B post-dedup, which moves the lane from 3.31 epochs to 6–10 — through the
ceiling under any reading, conservative or relaxed.

S5's audit corrected several inventory rows downward for provenance reasons (English inside "Indic"
corpora, transliteration double-counting). It does not state a dedup basis. **Given the audit already
found a rule that "every corpus that looks large is large because something was translated", the
same scepticism should be applied to duplication.**

*Action:* state the dedup basis for every supply figure. This is the highest-value single number to
nail down, because it can invalidate the tightest lane in the plan.

**Also relevant to our tier-D finding.** Sangraha's synthetic tier is one encyclopedia rendered 14
languages × 2 scripts. MinHash operates lexically and will not deduplicate a translation against its
source. Only semantic/cross-lingual dedup would — and that is exactly the case where semantic dedup
earns its cost. S5 identified the repetition-of-content problem; the dedup literature says the standard
tool does not solve it.

### 6. Indic tokenization — the cost is worse than the plan's own numbers imply

**What the literature says.** *The Tokenizer Tax* ([2607.24276](https://arxiv.org/abs/2607.24276))
measures an average **8.0× tokenization tax** for Indian languages relative to English under
`cl100k_base`, **13.0× for Malayalam**, reducing effective context to as little as **12%** of
English's. And the compute consequence: *"a 2× increase in fertility produces a 4× increase in
training time and cost."*

S5 already does the right structural thing — it allocates the Indic lane over **content, not tokens**,
using MUTANT-Indic fertility (1.375 tok/word Hindi, 3.025 Malayalam) and solves α = 0.59. That is
ahead of the practice these papers criticise.

**Two things to add.** First, `IndicSuperTokenizer` (2026) exists as a purpose-built comparison point;
our own tokenizer's per-language fertility should be reported against it, since fertility is the one
number that compounds into every downstream cost. Second, the quadratic remark matters for the
long-context stage specifically: a high-fertility language pushes the *same content* into a longer
window, so Indic long-context material is disproportionately expensive — an interaction between the
Indic lane and the long-context lane that the per-lane budget does not capture.

### 7. Agentic loss masking — the plan's rule is under-specified in a way that matters

**What the plan says.** S5: supervise assistant planning, tool calls and final answer *"including a
call that fails, because recovery is the taught behaviour."*

**What the literature says — and it appears to contradict, until you read it carefully.**

- Supporting S5: *"High-quality datasets should include examples where tool calls fail, return
  incomplete data, or produce unexpected formats — these cases are essential because without them,
  agents learn an unrealistic picture of the world."*
- Against S5, apparently: **error-masked training**, where *"for any turn that triggers an error during
  tool execution, token-level losses are zeroed out to ensure gradient updates are driven exclusively
  by executable and semantically valid trajectories."*
- And universally: *"mask tool execution results during loss calculation to prevent models from
  memorizing deterministic environmental feedback."*

**The reconciliation is the finding.** Two different things are both called "a failed tool call":

1. the model emits a **malformed or invalid call** — training on this teaches the model to emit
   invalid syntax. It should be **masked**.
2. the model emits a **valid call that the environment fails** (404, timeout, missing file) and then
   **recovers** — the recovery is exactly the behaviour we want. It should be **supervised**.

S5's rule does not distinguish them, and our implementation inherits that. Our own corpus contains
case 2 (`agent-0002`: a wrong-id call, then a correct retry) and supervises it, which is right — but
nothing in the mask policy would stop case 1 being supervised too.

*Action:* split the role. `tool_call_valid` → supervised; `tool_call_malformed` → masked;
`tool_observation` → always context. This requires the trajectory to carry a validity verdict per
call, which is a corpus-construction requirement, not a packer one — so it has to be decided before
the agentic synthesis programme runs, not after.

### 8. Resumption across world size — a known production concern with stated limits

**What the literature says.** Mosaic StreamingDataset provides *"deterministic sample ordering
independent of the real number of devices"* — but its guarantee holds under stated conditions:
**global batch size constant, seed unchanged, and the number of logical nodes maintained.** Elastic
training guidance is blunter: *"when your job resumes with a different world size, previously processed
samples may be repeated or skipped, affecting training quality."*

**Verdict: our design removes the logical-node constraint** by making position a single global index
with the rank partition derived at load. The global-batch constraint remains and should be made
explicit — a change of global batch mid-run is a change of experiment and belongs in the ledger as a
branch, not absorbed silently.

---

## Where the plan is already ahead

Stated because a survey that only finds faults is not a survey.

- **Allocating Indic over content rather than tokens**, with a fertility-corrected temperature
  parameter, is more careful than most published multilingual recipes, which the tokenizer-tax
  literature criticises precisely for allocating in tokens.
- **Protected floors as mechanism rather than preference**, justified by a measurement (OPUS drives
  unprotected lanes to 0 in 60/60 iterations) rather than by assertion.
- **The four-gate eval firewall with an independent derived-content gate** matches the contamination
  literature's finding that n-gram overlap *"can miss rephrasing"* — the paraphrase channel is the
  documented weak point, and it has its own gate.
- **Sizing the agentic lane in samples rather than tokens**, and declaring it a production programme
  rather than absorbing it into a percentage.

---

## What the literature does not settle

Honest gaps — these need our own experiments, not more searching.

1. **Whether curriculum helps at 3T.** Evidence is strongest below 1B parameters and explicitly shrinks
   with scale. S5's five-stage design is a hypothesis at our scale.
2. **Whether token-level selection (RHO-1) transfers from math to broad multilingual pretraining.** The
   headline gains are domain-matched; nobody has shown it for a 22-language Indic mixture.
3. **The right repetition ceiling for a scarce lane inside a mixture.** "15–20×" is reported with
   dependence on target size, compute and model scale — none of which match our configuration.
4. **Whether related-document packing helps when documents are multilingual.** The +11.6% ICL result is
   English; "related" across scripts is not obviously the same operation.
5. **Whether the anneal reserve earns its keep once the LR-decay confound is controlled.** S5's T3 as
   designed cannot separate the two.

---

## Recommended changes, by value over cost

| # | change | cost | why |
|---|---|---|---|
| 1 | **Record reference loss AND token loss trajectories (same tokens, multiple checkpoints)** | one extra forward pass + periodic re-scoring | 11% of tokens are persistently hard from *irreducible* uncertainty and 26% are genuinely learnable — indistinguishable from the level alone, opposite implications for V6 collection. Cannot be reconstructed after the run |
| 2 | **Moderate the final LR (final/peak ≈ 0.37, not 0.01) + EMA the last ~6 anneal checkpoints, and order *within* lanes ascending by quality** | config change + one sort | S5's design is the paper's `U,A` row, which measured **−0.04 avg / −0.31 core**. The `A,A` row measured **+1.20 / +2.00**. Validated at 3.2B/886B. Also un-confounds T3 |
| 3 | **State the dedup basis of every supply figure** | analysis only | at 50–70% typical reduction this can invalidate the tightest lane |
| 4 | **Record spikes as run events, not shard properties; resolve `attribution_uncertain` by PaLM's own ablation** | ledger schema | otherwise the V6 payload mislabels data on the basis of an interaction artifact |
| 5 | **Add `batches_skipped` to the ledger, with re-injection as a separate policy decision** | small | the standard spike recovery path. Untestable at demo scale — spikes do not occur in small models |
| 6 | **Split `tool_call` into valid / malformed for masking** | corpus schema | must be decided before agentic synthesis runs |
| 7 | **Segment-masked attention (already implemented here)** | efficiency trade | +1.1 pts average, measured |
| 8 | **Re-parameterise the ceiling from "epochs ≤ 4" to per-lane repetition factor `r`** | free | `r` is the variable that actually controls degradation, and optimal `r` differs ~4× between our scarcest and most abundant lanes. Do **not** raise the level — the model-size trend runs against it at 30B |
| 9 | **Try related-document packing** | retrieval pass over the corpus | largest unclaimed gain found (+11.6% ICL), least certain to transfer |
| 10 | **Token-level selection with a PER-LANE reference** | experiment | +6.8% on 15 general benchmarks at 80B tokens, but it narrows the model toward the reference domain — with one global reference that risk lands on our differentiator |

---

## Second pass (2026-08-07): the systems side

The survey above targets the training *plan*. A second sweep targeted the mechanisms S6 implements.
Full findings and what each changed are in `EXPERIMENT_LOG.md` E9; in brief:

| source | bearing on this system | acted on? |
|---|---|---|
| OLMo 2, arXiv:2501.00656 §3.1 | documents with 32+ repeated n-gram runs are over-represented in spiking batches; filtering flattens the gradient-norm trace | **implemented** — curation filter + trainer-side loss mask + 3 controls |
| GPT-3 / Llama decontamination practice (13-gram, 8–13 token windows) | Jaccard cannot see one benchmark item quoted inside a long host | **implemented** — sixth firewall gate, isolated by a planted case |
| Ding et al., arXiv:2404.10830 (ICML 2024) | our `best_fit` *is* their Best-Fit-Decreasing; truncation has measurable downstream cost | confirmed our choice; plan-level recommendation recorded, not applied |
| Krell et al., arXiv:2107.02027 | cross-contamination in packed attention | confirms the segment-isolation invariant |
| Wortsman et al., arXiv:2309.14322 | slow-divergence instabilities reproduce at small scale — **explicitly not loss spikes** (footnote 1) | corrects our spike account; not claimed as transfer |
| MegaScale-Data, arXiv:2504.09844 | packing composition changes per-rank compute (quadratic attention): 30+70 costs 16% more than 50+50 | V6 only — we do not model rank stragglers |
| Croissant 1.1, MLCommons (Feb 2026) | machine-actionable provenance on PROV-O; EU AI Act enforcement from Aug 2026 | V6 — our manifests carry the facts under private names |

**The correction this pass forced.** We had written that loss spikes are an interaction and not a
data property. OLMo 2 finds a data property; PaLM's ablation finds an interaction. The precise
statement is that spikes are an optimizer–curvature event (adaptive edge of stability) that certain
degenerate data reliably triggers — so filter the trigger, and still never mark a shard harmful for
coinciding with a spike.

## Sources

- [Fewer Truncations Improve Language Modeling (ICML 2024)](https://arxiv.org/abs/2404.10830)
- [Efficient Sequence Packing without Cross-contamination](https://arxiv.org/abs/2107.02027)
- [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)
- [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322)
- [MegaScale-Data: Scaling Dataloader for Multisource Large Foundation Model Training](https://arxiv.org/abs/2504.09844)
- [Croissant 1.1 — MLCommons](https://mlcommons.org/2026/02/croissant-1-1-standard/)
- [D4: Improving LLM Pretraining via Document De-Duplication and Diversification](https://proceedings.neurips.cc/paper_files/paper/2023/file/a8f8cbd7f7a5fb2c837e578c75e5b615-Paper-Datasets_and_Benchmarks.pdf)
- [Deduplicating Training Data Makes Language Models Better](https://aclanthology.org/2022.acl-long.577.pdf)
- [Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://arxiv.org/html/2402.13991)
- [Rho-1: Not All Tokens Are What You Need](https://arxiv.org/html/2404.07965v1) · [repo](https://github.com/microsoft/rho)
- [How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining (ICLR 2026)](https://arxiv.org/abs/2511.18903)
- [Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics](https://arxiv.org/html/2601.21698v2)
- [Scaling Data-Constrained Language Models](https://arxiv.org/pdf/2305.16264)
- [Scaling Laws for Mixture Pretraining Under Data Constraints](https://arxiv.org/pdf/2605.12715)
- [Larger Datasets Can Be Repeated More](https://arxiv.org/pdf/2511.13421)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311)
- [Data Mixture Inference: What do BPE Tokenizers Reveal about their Training Data?](https://arxiv.org/pdf/2407.16607) · [repo](https://github.com/alisawuffles/tokenizer-attack)
- [The Tokenizer Tax: Cross-Lingual Cost of Subword Tokenization for Indian Languages](https://arxiv.org/abs/2607.24276)
- [MUTANT: A Recipe for Multilingual Tokenizer Design](https://arxiv.org/html/2511.03237)
- [Explaining Data Mixing Scaling Laws](https://arxiv.org/html/2606.08167v2)
- [MosaicML StreamingDataset](https://www.databricks.com/blog/mosaicml-streamingdataset) · [docs](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataset.html)
- [Elastic training on SageMaker HyperPod](https://aws.amazon.com/blogs/machine-learning/adaptive-infrastructure-for-foundation-model-training-with-elastic-training-on-sagemaker-hyperpod/)
- [How to Train Long-Context Language Models (Effectively)](https://arxiv.org/html/2410.02660v4)
- [Long Context is Not Long at All: A Prospector of Long-Dependency Data](https://arxiv.org/pdf/2405.17915)
- [MinHash LSH for LLM training data](https://milvus.io/blog/minhash-lsh-in-milvus-the-secret-weapon-for-fighting-duplicates-in-llm-training-data.md)
- [An Open Source Data Contamination Report for LLMs](https://arxiv.org/pdf/2310.17589)
- [Detecting Benchmark Contamination Through Watermarking](https://openreview.net/forum?id=WFGxFzFDmQ)
