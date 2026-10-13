# ERA V5 — Assignments

Session work for the ERA V5 course. One folder per session, each with its own README,
code, and results.

| | Session | What it is |
|---|---|---|
| [S4](S4) | Data cleaning & deduplication | The 8-stage cleaning pipeline run for real on raw Common Crawl — 128M words down to 17M. [Live page](https://vedapragna-cs.github.io/era5-s4-data-cleaning/) |
| [S5](S5) | Agentic synthesis programme | Sizing the agentic data lane: benchmark map, budget proposal, cross-lane analysis |
| [S6](S6) | Training Data Execution System | The data ledger — immutable shards, curriculum mixture, packing, dual ledgers, crash/resume/replay/fork, with a generated evidence bundle |
| [S7-kronecker](S7-kronecker) | Invertible Kronecker embeddings (Problem 5) | Tie the LM output head to the input projection, removing 99.6% of output parameters. [Live page](https://vedapragna-cs.github.io/era-v5-s7-invertible-kronecker/) |
| [S7-rank](S7-rank) | Rank, not window (Problem 3) | The 32-byte Kronecker window is slack at inference and binding at tokenizer construction — rank governs quality |
| [S8](S8) | Modern attention variants | Every attention mechanism from the session, in the order they were actually launched. 34 mechanisms, every date verified against the arXiv API. [Live page](https://vedapragna-cs.github.io/era-v5-s8-attention-timeline/) |
| [S11](S11) | Optimizers & LR schedules | Five required experiments, every number reproduced from code in the folder |
| [S12](S12) | ZeRO stages on 32 virtual GPUs | ZeRO-1/2/3 simulated on CPU in plain PyTorch — memory and communication verified against the paper |
| [S13](S13) | Reversible training | Train with no stored activations. Euler vs midpoint, 42% less memory at 26% lower throughput |
| [S14](S14) | Dense → Mixture of Experts | Upcycle a dense model into an MoE and keep training. Loss-free load balancing, zero dead experts |

## Layout

Sessions 4, 6, 7 and 8 were originally developed in their own repositories. They were
merged here with `git subtree`, so every commit of each project is part of this
repository's history. The originals remain in place and keep serving the GitHub Pages
links above:

| Folder | Original repository |
|---|---|
| `S4` | [era5-s4-data-cleaning](https://github.com/vedapragna-cs/era5-s4-data-cleaning) |
| `S6` | [era-v5-s6-data-ledger](https://github.com/vedapragna-cs/era-v5-s6-data-ledger) |
| `S7-kronecker` | [era-v5-s7-invertible-kronecker](https://github.com/vedapragna-cs/era-v5-s7-invertible-kronecker) |
| `S7-rank` | [era-v5-s7-rank-not-window](https://github.com/vedapragna-cs/era-v5-s7-rank-not-window) |
| `S8` | [era-v5-s8-attention-timeline](https://github.com/vedapragna-cs/era-v5-s8-attention-timeline) |
