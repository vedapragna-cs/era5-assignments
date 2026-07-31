"""Fertility-adjust the per-language Indic allocation, and settle the script question.

E2 allocated the Indic lane in TOKENS. S3's MUTANT-Indic fertility ceilings say that is
the wrong unit: the same content costs 1.375 tokens/word in Hindi and 3.025 in Malayalam,
so an equal-token allocation gives Malayalam roughly 45% of Hindi's content. Since MILU
scores comprehension, not token counts, the allocation should be made over CONTENT and
converted back to tokens at the end.

The 4-epoch cap is fertility-invariant -- tokens_i <= 4*S_i and content_i <= 4*S_i/f_i are
the same constraint -- so only the allocation shape moves, not the ceiling.

Second half settles a problem E6 surfaced: several tail languages exist in two scripts
across our corpora, splitting supply that was already negligible.

Run:  python3 S5/scripts/indic_fertility.py
"""

from indic import SANGRAHA, NAME, MILU, supply_b, allocate, LANE_B

# S3 MUTANT-Indic prose fertility ceilings (tokens per whitespace word), group midpoints.
# Stated, not measured here -- they come from ACL 2026 MUTANT-Indic via the S3 proposal.
FERT_GROUP = {
    1.375: ["hin", "pan", "urd", "kas", "snd", "doi"],
    1.725: ["mar", "ori", "mai", "npi", "ben", "guj", "asm"],
    2.050: ["tel", "brx", "tam", "gom", "kan"],
    3.025: ["mni", "mal", "san", "sat"],
}
FERT = {lg: f for f, lgs in FERT_GROUP.items() for lg in lgs}

# Languages present in more than one script across Sangraha / IndicCorpV2 / BPCC (E6).
# benchmark_script: what IndicGenBench scores in, or None where it does not cover the language.
SCRIPTS = {
    "kas": (["Arab", "Deva"], None,   "Perso-Arabic is the official script in J&K"),
    "mni": (["Beng", "Mtei"], "Mtei", "Meetei Mayek is official since 2021; Bengali script is historical"),
    "snd": (["Arab", "Deva"], None,   "Perso-Arabic dominates supply 40.2MB vs 0.2MB in BPCC"),
    "sat": (["Olck"],         "Olck", "Ol Chiki is the official script; the only one carried in BPCC"),
}


def content_supply(S):
    """Supply expressed in word-equivalents rather than tokens."""
    return {lg: S[lg] / FERT[lg] for lg in S}


def main():
    S = supply_b()
    C = content_supply(S)

    print("=" * 78)
    print("1. THE UNIT PROBLEM")
    print("=" * 78)
    tok, _ = allocate(S, LANE_B, 0.72)
    print(f"  {'language':<11}{'fert':>6}{'tokens B':>10}{'content Bw':>12}"
          f"{'content vs Hindi':>18}")
    hin_c = tok["hin"] / FERT["hin"]
    for lg in sorted(MILU, key=lambda k: -tok[k]):
        c = tok[lg] / FERT[lg]
        print(f"  {NAME[lg]:<11}{FERT[lg]:>6.3f}{tok[lg]:>10.1f}{c:>12.1f}"
              f"{c/hin_c*100:>17.0f}%")
    print("\n  Equal-token allocation is not equal-content allocation. Malayalam receives")
    print(f"  {tok['mal']/FERT['mal']/hin_c*100:.0f}% of Hindi's content while receiving "
          f"{tok['mal']/tok['hin']*100:.0f}% of its tokens.")

    print("\n" + "=" * 78)
    print("2. ALLOCATE OVER CONTENT, CONVERT BACK TO TOKENS")
    print("=" * 78)
    # We want content_i ~ C_i^alpha, so tokens_i ~ C_i^alpha * f_i, water-filled against
    # the SAME token cap of 4 x supply. The budget is in tokens, so the fill runs in tokens.
    def fill(alpha):
        w = {lg: (C[lg] ** alpha) * FERT[lg] for lg in S}
        cap = {lg: 4.0 * S[lg] for lg in S}
        alloc = {lg: 0.0 for lg in S}
        free, remaining = set(S), LANE_B
        for _ in range(300):
            if remaining <= 1e-9 or not free:
                break
            wsum = sum(w[lg] for lg in free)
            newly = []
            for lg in free:
                alloc[lg] += remaining * w[lg] / wsum
                if alloc[lg] >= cap[lg] - 1e-12:
                    alloc[lg] = cap[lg]
                    newly.append(lg)
            remaining = LANE_B - sum(alloc.values())
            for lg in newly:
                free.discard(lg)
        return alloc

    wk_max = min(4 * S[lg] for lg in MILU)       # Punjabi's token cap, as in E2
    star = None
    for i in range(101):
        a = 1.00 - i * 0.01
        if min(fill(a)[lg] for lg in MILU) >= wk_max - 1e-6:
            star = a
            break
    newtok = fill(star if star is not None else 0.72)

    print(f"  alpha* in content space = {star:.2f}   (token-space alpha* was 0.72)")
    print(f"\n  {'language':<11}{'fert':>6}{'tok E2':>9}{'tok new':>9}{'delta':>9}"
          f"{'epochs':>8}  MILU")
    for lg in sorted(S, key=lambda k: -S[k]):
        if S[lg] < 0.01:
            continue
        d = newtok[lg] - tok[lg]
        ep = newtok[lg] / S[lg]
        print(f"  {NAME[lg]:<11}{FERT[lg]:>6.3f}{tok[lg]:>9.1f}{newtok[lg]:>9.1f}"
              f"{d:>+9.1f}{ep:>8.2f}  {'Y' if lg in MILU else ''}")
    print(f"  {'TOTAL':<11}{'':>6}{sum(tok.values()):>9.1f}{sum(newtok.values()):>9.1f}")
    over = [NAME[lg] for lg in S if S[lg] > 0.01 and newtok[lg] / S[lg] > 4.001]
    print(f"\n  languages over the 4-epoch cap after adjustment: {over or 'none'}")

    gain = sorted(((newtok[lg] - tok[lg], NAME[lg]) for lg in S if S[lg] > 0.01),
                  reverse=True)
    print(f"  largest gain  {gain[0][1]} {gain[0][0]:+.1f}B   "
          f"largest loss {gain[-1][1]} {gain[-1][0]:+.1f}B")
    capped = [NAME[lg] for lg in S if S[lg] > 0.01 and newtok[lg] >= 4 * S[lg] - 1e-6]
    print(f"\n  pinned at 4 epochs after adjustment: {len(capped)} — {', '.join(capped)}")
    old_par = (tok["mal"] / FERT["mal"]) / (tok["hin"] / FERT["hin"]) * 100
    new_par = (newtok["mal"] / FERT["mal"]) / (newtok["hin"] / FERT["hin"]) * 100
    print(f"\n  Malayalam content as % of Hindi's:  {old_par:.0f}%  ->  {new_par:.0f}%")
    print("  Better, and still nowhere near parity — because Malayalam hits the 4-epoch")
    print("  ceiling before it can absorb the tokens its fertility demands. High-fertility")
    print("  languages need MORE tokens per unit of content and therefore exhaust their")
    print("  repetition budget SOONER. Fertility adjustment improves the unit; it cannot")
    print("  manufacture supply, and content parity is not reachable at this lane size.")

    print("\n  Fertility adjustment moves tokens toward high-fertility languages, which is")
    print("  the opposite of what a token-proportional rule does. It is not a rounding")
    print("  correction: it is the difference between funding words and funding bytes.")

    print("\n" + "=" * 78)
    print("3. SCRIPT DECISION FOR THE FRAGMENTED TAIL")
    print("=" * 78)
    print("  Rule applied, in order:")
    print("   (a) if a claimed benchmark scores the language, use the script it scores in;")
    print("   (b) otherwise use the script with the most supply;")
    print("   (c) report the supply stranded in the unchosen script rather than pooling it.")
    print()
    print(f"  {'language':<11}{'scripts':<14}{'benchmark':<11}{'chosen':<9}basis")
    for lg, (scripts, bench, note) in SCRIPTS.items():
        chosen = bench if bench else scripts[0]
        basis = "benchmark" if bench else "supply"
        print(f"  {NAME[lg]:<11}{'/'.join(scripts):<14}{str(bench or '—'):<11}"
              f"{chosen:<9}{basis}")
        print(f"  {'':<11}{note}")
    print("\n  Manipuri is the live case: IndicGenBench scores Meetei Mayek, BPCC carries")
    print("  BOTH mni_Beng and mni_Mtei, and IndicCorpV2 carries only mni_Mtei. Choosing")
    print("  Mtei is right for the benchmark and strands the Bengali-script material.")
    print("  Kashmiri and Sindhi are NOT in IndicGenBench's 29, so nothing forces a script")
    print("  and we take Perso-Arabic on supply. That choice is unvalidated by any")
    print("  benchmark we claim, which is the honest way to record it.")


if __name__ == "__main__":
    main()
