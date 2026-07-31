"""Restate the budget as funded share / trigger / fallback.

Flat percentages hide which parts of the plan rest on data we hold and which rest on
work not yet done. Every share here is one of three things and the plan should say which:

  FUNDED     -- supportable at <=4 epochs from supply we hold today
  PROGRAMME  -- supportable only if a named production programme delivers
  GATED      -- supportable only if a named agreement or measurement lands

A lane whose share exceeds what its supply supports is not wrong; it is CONDITIONAL, and
the condition belongs in the table rather than in a footnote. This also makes the plan
robust to data arriving over the next six months: a trigger fires, a share moves, and no
redesign is needed.

Run:  python3 S5/scripts/conditional.py
"""

MAIN_B = 2910.0
ANNEAL_B = 90.0
EPOCH_CAP = 4.0

# lane -> (requested share, held supply B, note on the supply)
LANES = {
    "web":     (0.34, 4691.0, "DCLM + FineWeb-Edu + V4 web"),
    "code":    (0.24, 1103.0, "The Stack v2 + CommitPack + V4"),
    "indic":   (0.17,  157.2, "native-script, English removed (E2)"),
    "stem":    (0.12,  146.0, "auditable rows; composer claims 250B"),
    "reason":  (0.09,   85.1, "78B of it is V4 AON lineage"),
    "longctx": (0.02,  100.0, "repo-packed code + packed books"),
    "agentic": (0.02,    0.63, "all public function-calling + trajectories"),
}
# supply that would survive losing our own V4 corpora
PUBLIC_ONLY = {"reason": 7.1, "indic": 157.2, "stem": 97.0, "code": 904.0}

TRIGGERS = [
    ("agentic", "synthesis programme delivers 24.5M verified calls",
     "holds 2%", "falls to 0.09% — what 0.63B supports at 4 epochs"),
    ("agentic", "anneal programme delivers 153,600 trajectories",
     "holds 8% of the anneal", "anneal agentic falls to 2.8%"),
    ("indic", "a gated language crosses 1.30B unique tokens",
     "re-enters the allocation automatically", "stays gated; IndicGenBench stays 14/29"),
    ("stem", "NCERT / SCERT / NPTEL / IGNOU agreement lands",
     "Indic-STEM requirement 3.4% -> 15%", "requirement lowered to 3.4%, capability conceded"),
    ("stem", "course confirms the 250B STEM figure",
     "epochs 2.39 -> 1.40, lane comfortable", "stays at 146B and 2.39 epochs"),
    ("code", "more diff-following data than CommitPackFT's 4B",
     "upsample factor drops below 8x", "8x upsample holds; 2 epochs on CommitPackFT"),
    ("indic", "T2 shows tier D holds on India-specific MILU subjects",
     "tier D stays at 40%", "D cut, A raised, Indic total falls below 17%"),
    ("longctx", "T1 confirms 2% loses nothing vs 6%",
     "2% holds, 116B stays reallocated", "restore to 6%, take 116B back from web"),
    ("reason", "court judgments ingested (~8B, CC BY 4.0)",
     "public-rebuild floor roughly doubles", "public rebuild stays at 7.1B"),
]


def main():
    print("=" * 80)
    print("1. WHAT EACH SHARE ACTUALLY RESTS ON")
    print("=" * 80)
    print(f"  {'lane':<9}{'asked':>7}{'max @4ep':>10}{'status':>12}{'gap':>9}  supply note")
    funded_share = 0.0
    for lane, (share, sup, note) in sorted(LANES.items(), key=lambda kv: -kv[1][0]):
        maxshare = sup * EPOCH_CAP / MAIN_B
        if maxshare >= share:
            status, gap = "FUNDED", ""
            funded_share += share
        else:
            status = "PROGRAMME"
            gap = f"{share/maxshare:.0f}x"
        print(f"  {lane:<9}{share:>6.0%}{maxshare:>9.1%}{status:>12}{gap:>9}  {note}")
    print(f"\n  FUNDED from supply we hold today: {funded_share:.0%} of the main run.")
    print(f"  Everything else is one lane: agentic at {LANES['agentic'][0]:.0%}, which rests")
    print("  entirely on a named synthesis programme. That is the honest headline —")
    print("  not '100% allocated' but '98% funded, 2% programme-dependent'.")

    print("\n" + "=" * 80)
    print("2. THE DEPENDENCY NOBODY ASKS ABOUT — what if we lose the V4 corpora?")
    print("=" * 80)
    print(f"  {'lane':<9}{'asked':>7}{'with V4':>10}{'public only':>13}{'fallback share':>16}")
    hole = 0.0
    for lane in ("reason", "stem", "code", "indic"):
        share, sup, _ = LANES[lane]
        pub = PUBLIC_ONLY[lane]
        fb = min(share, pub * EPOCH_CAP / MAIN_B)
        hole += share - fb
        print(f"  {lane:<9}{share:>6.0%}{sup:>9.1f}B{pub:>12.1f}B{fb:>15.1%}")
    print(f"\n  Reasoning is the exposure: 78B of its 85B is V4 AON lineage, so a")
    print(f"  public-only rebuild supports {PUBLIC_ONLY['reason']*EPOCH_CAP/MAIN_B:.1%}, not 9%.")
    print(f"  Total hole if the V4 corpora were unavailable: {hole:.1%} of the run.")
    print("  Worth stating because 'we hold AON' is load-bearing and invisible in a")
    print("  flat percentage table.")

    print("\n" + "=" * 80)
    print("3. TRIGGERS AND FALLBACKS")
    print("=" * 80)
    for lane, trig, then, els in TRIGGERS:
        print(f"  [{lane}]  IF {trig}")
        print(f"           THEN {then}")
        print(f"           ELSE {els}")

    print("\n" + "=" * 80)
    print("4. THE DONOR LANE")
    print("=" * 80)
    web_share, web_sup, _ = LANES["web"]
    print(f"  Web is asked for {web_share:.0%} against {web_sup:,.0f}B — "
          f"{web_share*MAIN_B/web_sup:.2f} epochs.")
    print(f"  It could support {web_sup*EPOCH_CAP/MAIN_B:.0%} of the run on its own.")
    print("  Every fallback above is absorbed here, and every trigger that fires takes")
    print("  from here. That is the lane's structural role and it should be named as such:")
    print("  web is not 34% because 34% is optimal, it is 34% because it is the only lane")
    print("  with the slack to be the adjustment variable.")
    print("\n  Risk of naming it: the session warns that cutting general web too far gives a")
    print("  model whose code compiles and does not work. The floor below which we do not")
    print("  cut web should be stated -- we propose 25%, still 5x any other lane's slack.")


if __name__ == "__main__":
    main()
