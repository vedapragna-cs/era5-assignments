# ERA V5 budget proposal — demand vs real supply, epochs, verdicts.
# Supply (B tokens) from the S5 Dataset Inventory widget.
SUP = {"code":1103, "agentic":0.627, "reason":85.1, "longctx":100, "indic":275.9, "stem":146, "web":4691}
# NOTE: composer states stem 250B / web 4500B; inventory rows give stem 146B (peS2o+proof-pile+D4)
# and web 4691B. Using inventory rows — they are the auditable number.

TOTAL = 3000.0          # B tokens. Course target stated in session: 2.4-4T.
ANNEAL_FRAC = 0.03      # 3% -> 90B, in line with OLMo 2 (50/100/300B anneal mixes)
MAIN = TOTAL*(1-ANNEAL_FRAC); ANN = TOTAL*ANNEAL_FRAC

COMPOSER = {"code":24,"agentic":2,"reason":6,"longctx":6,"indic":16,"stem":12,"web":34}
PROPOSED = {"code":24,"agentic":2,"reason":9,"longctx":2,"indic":17,"stem":12,"web":34}
ANNEAL   = {"code":20,"agentic":8,"reason":18,"longctx":8,"indic":28,"stem":10,"web":8}

def verdict(ep):
    if ep <= 1: return "covered"
    if ep <= 4: return f"repeat {ep:.1f}x (OK, <4)"
    return f"repeat {ep:.0f}x -- MUST SYNTHESIZE"

def table(name, mix, pool):
    print(f"\n{'='*78}\n{name}  ({pool:.0f}B tokens)\n{'='*78}")
    print(f"{'lane':<10}{'share':>7}{'demand':>10}{'supply':>10}{'epochs':>9}   verdict")
    assert abs(sum(mix.values())-100) < 1e-6, sum(mix.values())
    for k,v in mix.items():
        d = v/100*pool; s = SUP[k]; ep = d/s
        print(f"{k:<10}{v:>6}%{d:>9.1f}B{s:>9.1f}B{ep:>8.2f}   {verdict(ep)}")

table("A. COMPOSER DEFAULT applied to main run", COMPOSER, MAIN)
table("B. PROPOSED main pretraining run", PROPOSED, MAIN)
table("C. PROPOSED anneal (separate pool - this is what the widget gets wrong)", ANNEAL, ANN)

# ---- Indic tier accounting ----
TIERS = {"A verified":64, "B unverified":44.9, "C translated/parallel":5.0, "D synthetic":162}
print(f"\n{'='*78}\nINDIC TIER SPLIT — main run, Indic = {PROPOSED['indic']}% of {MAIN:.0f}B = {PROPOSED['indic']/100*MAIN:.0f}B\n{'='*78}")
indic_demand = PROPOSED["indic"]/100*MAIN
for label, split in [("composer default (A40/B25/C20/D15)", {"A verified":40,"B unverified":25,"C translated/parallel":20,"D synthetic":15}),
                     ("PROPOSED (A28/B30/C2/D40)",          {"A verified":28,"B unverified":30,"C translated/parallel":2,"D synthetic":40})]:
    print(f"\n  {label}")
    for t,pc in split.items():
        d = pc/100*indic_demand; s = TIERS[t]; ep = d/s
        flag = "OK" if ep<=4 else "IMPOSSIBLE"
        print(f"    {t:<24}{pc:>4}% -> {d:>7.1f}B vs {s:>6.1f}B supply = {ep:>6.1f} epochs  {flag}")

# ---- agentic reality check ----
print(f"\n{'='*78}\nAGENTIC — the binding constraint\n{'='*78}")
for pool,lbl,share in [(MAIN,"main run",PROPOSED["agentic"]),(ANN,"anneal",ANNEAL["agentic"])]:
    d = share/100*pool
    unique_needed = d/4      # at the 4-epoch ceiling
    print(f"  {lbl:<10} {share}% = {d:>6.1f}B demand | at 4 epochs needs {unique_needed:>5.1f}B unique "
          f"| have {SUP['agentic']}B -> must SYNTHESIZE {unique_needed-SUP['agentic']:>5.1f}B "
          f"({(unique_needed-SUP['agentic'])/SUP['agentic']:.0f}x all public agentic data)")

print(f"\n  Long-context freed by 6%->2%: {(6-2)/100*MAIN:.0f}B tokens redeployed.")
print(f"  Literature requirement for 128K context (arXiv 2402.10171): 0.5-5B. "
      f"Proposed {PROPOSED['longctx']/100*MAIN:.0f}B is still {PROPOSED['longctx']/100*MAIN/5:.0f}x the upper bound.")
