"""Per-language allocation inside the Indic lane.

Supply is the AI4Bharat Sangraha per-language token table (HF dataset card), whose
totals -- 64,306.1M verified / 24,307.7M unverified / 162,707.9M synthetic -- reproduce
the S5 inventory rows exactly, so the per-language rows are auditable against a number
the session already uses.

Two corrections fall out of the table itself and are applied here:

  1. 12,759.9M of "verified" Sangraha is ENGLISH. Counting it toward an Indic lane is
     the wishful accounting the plan exists to avoid. Excluded (routed to the web lane,
     which has 3.7T of slack and does not notice).

  2. The synthetic config is 28 splits = 14 languages x 2 scripts, and the native/Latin
     pairs have IDENTICAL row counts (hin_Deva and hin_Latn are both 5,775,143 rows).
     It is one corpus transliterated, not two corpora. Only the native-script half buys
     MILU / IndicGenBench, which are scored in native script.

Allocation mechanism: temperature sampling p_i ~ S_i^alpha (XLM-R, arXiv:1911.02116),
water-filled against a hard per-language cap of 4 x supply (arXiv:2305.16264).
The cap is what makes this non-trivial: the tail cannot be lifted by allocation.

Run:  python3 S5/scripts/indic.py
"""

# Sangraha per-language tokens, millions. (verified, synthetic, unverified)
# https://huggingface.co/datasets/ai4bharat/sangraha
SANGRAHA = {
    "asm": (292.1, 11696.4, 17.5),      "ben": (10604.4, 13814.1, 5608.8),
    "brx": (1.5, 0.0, 0.0),             "doi": (0.06, 0.0, 0.0),
    "eng": (12759.9, 0.0, 0.0),         "gom": (10.1, 0.0, 0.0),
    "guj": (3647.9, 12934.5, 597.0),    "hin": (12617.3, 9578.7, 12348.3),
    "kan": (1778.3, 12087.4, 388.8),    "kas": (0.5, 0.0, 0.0),
    "mai": (14.6, 0.0, 0.0),            "mal": (2730.8, 13130.0, 547.8),
    "mar": (2827.0, 10816.7, 652.1),    "mni": (7.4, 0.0, 0.0),
    "npi": (1822.5, 10588.7, 485.5),    "ori": (1177.1, 11338.0, 23.7),
    "pan": (1075.3, 9969.6, 136.9),     "san": (1329.0, 13553.5, 9.8),
    "sat": (0.3, 0.0, 0.0),             "snd": (258.2, 0.0, 0.0),
    "tam": (3985.1, 11859.3, 1515.9),   "tel": (3706.8, 11924.5, 647.4),
    "urd": (3658.1, 9415.8, 1328.2),
}

NAME = {
    "asm": "Assamese", "ben": "Bengali", "brx": "Bodo", "doi": "Dogri", "gom": "Konkani",
    "guj": "Gujarati", "hin": "Hindi", "kan": "Kannada", "kas": "Kashmiri", "mai": "Maithili",
    "mal": "Malayalam", "mar": "Marathi", "mni": "Manipuri", "npi": "Nepali", "ori": "Odia",
    "pan": "Punjabi", "san": "Sanskrit", "sat": "Santali", "snd": "Sindhi", "tam": "Tamil",
    "tel": "Telugu", "urd": "Urdu",
}

# MILU (arXiv:2411.02538) scores 11 languages, one of which is English -> 10 Indic.
MILU = {"ben", "guj", "hin", "kan", "mal", "mar", "ori", "pan", "tam", "tel"}

# IndicGenBench (arXiv:2404.16816): 29 languages, 13 scripts, 4 families.
IGB_HIGH = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Tamil",
            "Telugu", "Urdu"]
IGB_MED = ["Assamese", "Bhojpuri", "Nepali", "Odia", "Punjabi", "Pashto", "Sanskrit"]
IGB_LOW = ["Awadhi", "Haryanvi", "Tibetan", "Garhwali", "Konkani", "Chhattisgarhi",
           "Rajasthani", "Maithili", "Manipuri", "Malvi", "Marwari", "Santali", "Bodo"]

LANE_B = 519.9          # Indic lane total, B tokens (CURRICULUM.md: main + anneal)
EPOCH_CAP = 4.0         # arXiv:2305.16264
NATIVE_SHARE = 0.50     # synthetic native-script half; rows are exactly 1:1 with Latin


def supply_b():
    """Usable native-script Indic supply per language, in B tokens."""
    out = {}
    for lg, (ver, syn, unv) in SANGRAHA.items():
        if lg == "eng":
            continue
        out[lg] = (ver + unv + NATIVE_SHARE * syn) / 1000.0
    return out


def allocate(S, total, alpha, cap_mult=EPOCH_CAP):
    """Temperature-sampled allocation, water-filled against per-language epoch caps.

    Repeatedly spread the remaining budget over uncapped languages in proportion to
    S_i^alpha, clip anything above cap, and redistribute. Returns (alloc, unplaceable)
    where unplaceable is budget that no language can absorb without breaking the cap.
    """
    cap = {lg: cap_mult * s for lg, s in S.items()}
    alloc = {lg: 0.0 for lg in S}
    free = set(S)
    remaining = total
    for _ in range(200):
        if remaining <= 1e-9 or not free:
            break
        wsum = sum(S[lg] ** alpha for lg in free)
        newly_capped = []
        for lg in free:
            alloc[lg] += remaining * (S[lg] ** alpha) / wsum
            if alloc[lg] >= cap[lg] - 1e-12:
                alloc[lg] = cap[lg]
                newly_capped.append(lg)
        placed = sum(alloc.values())
        remaining = total - placed
        for lg in newly_capped:
            free.discard(lg)
    return alloc, max(0.0, remaining)


def fmt(x):
    return f"{x:,.2f}" if x < 10 else f"{x:,.1f}"


def main():
    S = supply_b()
    tot_ver = sum(v for lg, (v, _, _) in SANGRAHA.items() if lg != "eng")
    print("=" * 78)
    print("SUPPLY")
    print("=" * 78)
    print(f"  Sangraha verified, all configs : {sum(v for v,_,_ in SANGRAHA.values())/1000:6.1f}B")
    print(f"  ...of which ENGLISH            : {SANGRAHA['eng'][0]/1000:6.1f}B  "
          f"({SANGRAHA['eng'][0]/sum(v for v,_,_ in SANGRAHA.values())*100:.1f}% of tier A)")
    print(f"  verified Indic only            : {tot_ver/1000:6.1f}B")
    syn = sum(s for _, s, _ in SANGRAHA.values()) / 1000
    print(f"  synthetic, both scripts        : {syn:6.1f}B  (14 langs x 2 scripts, rows 1:1)")
    print(f"  synthetic, native script only  : {syn*NATIVE_SHARE:6.1f}B  <- what buys the benchmarks")
    print(f"  usable native-script Indic     : {sum(S.values()):6.1f}B  "
          f"vs {LANE_B:.0f}B lane demand -> {LANE_B/sum(S.values()):.2f} epochs")

    # concentration
    ranked = sorted(S.items(), key=lambda kv: -kv[1])
    top2 = sum(v for _, v in ranked[:2])
    print(f"\n  top-2 (hin+ben) hold {top2/sum(S.values())*100:.1f}% of usable supply")
    bottom = [lg for lg, v in ranked if v < 1.0]
    print(f"  {len(bottom)} languages under 1B tokens: "
          f"{', '.join(NAME[l] for l in bottom)}")
    print(f"  their combined supply: {sum(S[l] for l in bottom)*1000:.1f}M "
          f"({sum(S[l] for l in bottom)/sum(S.values())*100:.4f}% of the lane's supply)")

    # benchmark coverage
    have = {NAME[lg] for lg in S}
    print("\n" + "=" * 78)
    print("BENCHMARK COVERAGE vs SUPPLY")
    print("=" * 78)
    for tier, langs in (("high", IGB_HIGH), ("medium", IGB_MED), ("low", IGB_LOW)):
        missing = [l for l in langs if l not in have]
        print(f"  IndicGenBench {tier:6s}: {len(langs)-len(missing):2d}/{len(langs)} have supply"
              + (f"   ZERO: {', '.join(missing)}" if missing else ""))
    all_igb = IGB_HIGH + IGB_MED + IGB_LOW
    zero = [l for l in all_igb if l not in have]
    print(f"  IndicGenBench total : {len(all_igb)-len(zero)}/{len(all_igb)} funded, "
          f"{len(zero)} with ZERO tokens in inventory")
    print(f"  MILU (10 Indic)     : {sum(1 for l in MILU if l in S)}/10 funded — "
          f"weakest is {min((S[l], NAME[l]) for l in MILU)[1]} at "
          f"{min(S[l] for l in MILU):.2f}B")

    # alpha sweep
    print("\n" + "=" * 78)
    print("TEMPERATURE SWEEP  (p_i ~ S_i^alpha, capped at 4 epochs)")
    print("=" * 78)
    print(f"  {'alpha':>5} {'hin %':>7} {'ben %':>7} {'weakest MILU':>13} {'capped':>7} "
          f"{'unplaceable':>12}")
    for alpha in (1.0, 0.7, 0.5, 0.3, 0.2, 0.1):
        a, un = allocate(S, LANE_B, alpha)
        cappd = sum(1 for lg in S if a[lg] >= 4 * S[lg] - 1e-9)
        wk = min(a[lg] for lg in MILU)
        print(f"  {alpha:>5.1f} {a['hin']/LANE_B*100:>6.1f}% {a['ben']/LANE_B*100:>6.1f}% "
              f"{wk:>11.1f}B {cappd:>7d} {un:>11.1f}B")

    # The weakest MILU language pins at its 4-epoch cap somewhere below alpha=1.
    # Past that point, lowering alpha buys MILU nothing and only pins more languages
    # to the ceiling. Find the largest alpha that already achieves the maximum.
    print("\n" + "=" * 78)
    print("FINE SWEEP — where does the weakest MILU language stop gaining?")
    print("=" * 78)
    wk_max = min(4 * S[lg] for lg in MILU)
    print(f"  ceiling for weakest MILU language (Punjabi): 4 x {min(S[lg] for lg in MILU):.2f}B "
          f"= {wk_max:.1f}B")
    star = None
    print(f"  {'alpha':>5} {'weakest MILU':>13} {'capped':>7} {'hin %':>7}")
    for i in range(41):
        alpha = 1.00 - i * 0.01
        a, _ = allocate(S, LANE_B, alpha)
        wk = min(a[lg] for lg in MILU)
        cappd = sum(1 for lg in S if a[lg] >= 4 * S[lg] - 1e-9)
        if wk >= wk_max - 1e-6 and star is None:
            star = alpha
        if abs(alpha * 100 - round(alpha * 100 / 5) * 5) < 1e-6:
            print(f"  {alpha:>5.2f} {wk:>11.1f}B {cappd:>7d} {a['hin']/LANE_B*100:>6.1f}%")
    print(f"\n  alpha* = {star:.2f}  — the largest temperature at which the weakest MILU")
    print("  language already reaches its ceiling. Below this, MILU gains nothing and")
    print("  more languages get pinned at 4 epochs with zero headroom.")

    # sensitivity: what the un-audited corpora would add
    extra = 20.9 + 5.0   # IndicCorpV2 + BPCC + Samanantar, no per-language table
    print(f"\n  Sensitivity: adding IndicCorpV2 (20.9B) + BPCC/Samanantar (5.0B) raises")
    print(f"  usable supply {sum(S.values()):.1f}B -> {sum(S.values())+extra:.1f}B, "
          f"lane epochs {LANE_B/sum(S.values()):.2f} -> {LANE_B/(sum(S.values())+extra):.2f}.")
    print("  Per-language tables for these are not published, so they are not allocated here.")

    for alpha in (1.0, star, 0.3):
        a, un = allocate(S, LANE_B, alpha)
        print("\n" + "=" * 78)
        print(f"ALLOCATION at alpha = {alpha}    (unplaceable {un:.1f}B)")
        print("=" * 78)
        print(f"  {'lang':<11}{'supply B':>10}{'alloc B':>10}{'share':>8}{'epochs':>8}"
              f"{'ver%':>7}  bench")
        for lg, _ in sorted(S.items(), key=lambda kv: -kv[1]):
            ver, syn_, unv = SANGRAHA[lg]
            usable = S[lg]
            verpct = ver / (usable * 1000) * 100 if usable else 0
            ep = a[lg] / usable if usable else float("inf")
            tag = "MILU" if lg in MILU else ""
            flag = "  <-CAP" if ep >= 3.999 else ""
            print(f"  {NAME[lg]:<11}{fmt(usable):>10}{fmt(a[lg]):>10}"
                  f"{a[lg]/LANE_B*100:>7.2f}%{ep:>8.2f}{verpct:>6.0f}%  {tag}{flag}")
        print(f"  {'TOTAL':<11}{sum(S.values()):>10.1f}{sum(a.values()):>10.1f}")

    # tier composition is not a free per-language choice
    print("\n" + "=" * 78)
    print("TIER MIX IS NOT A FREE CHOICE PER LANGUAGE")
    print("=" * 78)
    a, _ = allocate(S, LANE_B, star)
    realized_ver = sum(a[lg] * (SANGRAHA[lg][0] / (S[lg] * 1000)) for lg in S if S[lg] > 0)
    print(f"  If each language draws its own tiers in proportion to what it has, the")
    print(f"  realized global Tier-A (verified) share is {realized_ver/LANE_B*100:.1f}% "
          f"— vs the {28}% the budget proposes.")
    vers = sorted(((SANGRAHA[lg][0] / (S[lg] * 1000) * 100), NAME[lg])
                  for lg in S if S[lg] > 0.5)
    print(f"  Per-language verified share ranges {vers[0][0]:.0f}% ({vers[0][1]}) to "
          f"{vers[-1][0]:.0f}% ({vers[-1][1]}) among languages above 0.5B.")
    print("  A single global tier split therefore cannot be met language by language.")

    # what the tail would need
    print("\n" + "=" * 78)
    print("THE TAIL IS NOT AN ALLOCATION PROBLEM")
    print("=" * 78)
    print("  Max tokens each sub-1B language can absorb at the 4-epoch ceiling:")
    for lg, v in sorted(((l, S[l]) for l in S if S[l] < 1.0), key=lambda kv: -kv[1]):
        pct = 4 * v / LANE_B * 100
        print(f"    {NAME[lg]:<11} supply {v*1000:>8.1f}M  ->  max {4*v*1000:>8.1f}M "
              f"= {pct:.4f}% of the lane")
    print("\n  Even allocating every one of them their full 4-epoch ceiling spends "
          f"{sum(4*S[l] for l in S if S[l]<1.0)*1000:.0f}M")
    print(f"  = {sum(4*S[l] for l in S if S[l]<1.0)/LANE_B*100:.3f}% of the Indic lane. "
          "No share of the budget fixes this.")


if __name__ == "__main__":
    main()
