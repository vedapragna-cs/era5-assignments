"""Sizing the agentic synthesis programme.
The lane cannot be filled from real data at any scale (see BUDGET-PROPOSAL.md §3).
This converts the token gap into a concrete production target in SAMPLES.
"""
FC = {  # short function-calling: many samples, few tokens each
 "Glaive v2":(113_000, 50e6), "ToolBench":(120_000, 80e6), "ToolACE":(110_000, 60e6),
 "Nexus/NexusRaven":(40_000, 30e6), "xLAM/APIGen":(60_000, 25e6), "Hermes FC":(15_000, 22e6)}
TRAJ = {  # long trajectories: few samples, many tokens each
 "SWE-Gym":(2_400, 150e6), "SWE-smith":(26_000, 120e6), "OpenHands rollouts":(10_000, 90e6)}

def report(name, d):
    s = sum(v[0] for v in d.values()); t = sum(v[1] for v in d.values())
    print(f"\n{name}: {s:,} samples · {t/1e6:.0f}M tokens · mean {t/s:,.0f} tok/sample")
    for k,(ns,nt) in sorted(d.items(), key=lambda x:-x[1][1]/x[1][0]):
        print(f"   {k:<20}{ns:>9,} samples {nt/1e6:>7.0f}M tok   {nt/ns:>8,.0f} tok/sample")
    return s, t

print("="*80); print("EXISTING PUBLIC AGENTIC SUPPLY, split by shape"); print("="*80)
fc_s, fc_t = report("Short function-calling", FC)
tr_s, tr_t = report("Long trajectories", TRAJ)
print(f"\n   TOTAL {(fc_t+tr_t)/1e6:.0f}M tokens — the entire public agentic lane")

EPOCH_CAP = 4        # arXiv:2305.16264
MAIN_DEMAND, ANN_DEMAND = 58.2e9, 7.2e9
SUP_YIELD = 356/668  # measured: fraction of trajectory tokens carrying loss

print(f"\n{'='*80}\nSYNTHESIS TARGETS at the {EPOCH_CAP}-epoch ceiling\n{'='*80}")
for label, demand, have_t, mean_tok, kind in [
    ("MAIN RUN  (function-calling)", MAIN_DEMAND, fc_t, fc_t/fc_s, "calls"),
    ("ANNEAL    (trajectories)",     ANN_DEMAND,  tr_t, tr_t/tr_s, "trajectories")]:
    unique = demand/EPOCH_CAP
    gap = unique - have_t
    print(f"\n{label}")
    print(f"   demand           {demand/1e9:>8.1f}B seen tokens  ({demand*SUP_YIELD/1e9:.1f}B supervised at 53% yield)")
    print(f"   unique needed    {unique/1e9:>8.1f}B   (repeat 4x)")
    print(f"   have             {have_t/1e9:>8.2f}B")
    print(f"   MUST SYNTHESIZE  {gap/1e9:>8.1f}B  = {gap/have_t:>5.1f}x existing public supply of this shape")
    print(f"   => {gap/mean_tok:>12,.0f} new {kind} at the observed mean of {mean_tok:,.0f} tok each")
