"""Cost the two arms of T5: English-internal reasoning vs native reasoning.

The plan has no data of the target shape -- Indic question, reasoning, Indic answer --
so both arms have to be synthesized. This sizes what each costs, in translated tokens
and in trained tokens, and shows why they differ by more than intuition suggests.

The asymmetry has two independent sources that compound:
  1. Arm A translates only the question and answer; Arm B also translates the chain of
     thought, which is the largest part of a reasoning example.
  2. The same content costs more tokens in Indic than in English, so translating the CoT
     inflates the trained-token cost as well as the translation cost.

Fertility figures are the S3 MUTANT-Indic prose ceilings (tokens per whitespace word),
weighted by the alpha=0.72 per-language allocation from INDIC-ALLOCATION.md.

Run:  python3 S5/scripts/xlingual.py
"""

# language -> (allocation share % at alpha=0.72, MUTANT-Indic fertility tokens/word)
LANGS = {
    "Hindi":     (14.78, 1.375), "Bengali":  (12.32, 1.725), "Tamil":     (7.42, 2.05),
    "Gujarati":  (7.08, 1.725),  "Telugu":   (6.89, 2.05),   "Malayalam": (6.66, 3.00),
    "Urdu":      (6.59, 1.375),  "Marathi":  (6.19, 1.725),  "Kannada":   (5.85, 2.05),
    "Sanskrit":  (5.80, 3.00),   "Nepali":   (5.53, 1.725),  "Odia":      (5.14, 1.725),
    "Punjabi":   (4.77, 1.375),  "Assamese": (4.74, 1.725),
}
FERT_EN = 1.15                 # S3 ceiling for English

# Reasoning-length bands from SESSION-DIGEST §13 (measured widget traces) and the
# lane distribution proposed in CURRICULUM.md.
BANDS = [("L0 low", 37, 0.30), ("L1 medium", 74, 0.35),
         ("L2 high", 161, 0.25), ("L3 ultra", 346, 0.10)]
Q_TOK, A_TOK = 80, 20          # English question / answer, AIME-GPQA shape

AON_B = 78.0                   # our English reasoning corpus, B tokens
REASON_LANE_B = 278.1          # reasoning lane, B tokens
XLING_SHARE = 0.10             # proposed cross-lingual slice of the reasoning lane


def fertility_ratio():
    wsum = sum(s for s, _ in LANGS.values())
    mean = sum(s * f for s, f in LANGS.values()) / wsum
    return mean, mean / FERT_EN


def main():
    fert, ratio = fertility_ratio()
    cot = sum(t * w for _, t, w in BANDS)
    en_example = Q_TOK + cot + A_TOK

    print("=" * 76)
    print("UNIT ECONOMICS OF ONE REASONING EXAMPLE")
    print("=" * 76)
    print(f"  allocation-weighted Indic fertility  {fert:.3f} tokens/word")
    print(f"  English fertility                    {FERT_EN:.3f} tokens/word")
    print(f"  -> same content costs                {ratio:.3f}x more tokens in Indic")
    print(f"\n  lane-weighted CoT length             {cot:.0f} tokens")
    print("  " + "  ".join(f"{n}={t}@{w:.0%}" for n, t, w in BANDS))
    print(f"  English example (Q {Q_TOK} + CoT {cot:.0f} + A {A_TOK}) = {en_example:.0f} tokens")

    # Arm A: question and answer in Indic, chain of thought stays English
    a_q, a_cot, a_a = Q_TOK * ratio, cot, A_TOK * ratio
    a_tot = a_q + a_cot + a_a
    a_translated = a_q + a_a
    # Arm B: everything in Indic
    b_q, b_cot, b_a = Q_TOK * ratio, cot * ratio, A_TOK * ratio
    b_tot = b_q + b_cot + b_a
    b_translated = b_tot

    print("\n" + "=" * 76)
    print("PER EXAMPLE")
    print("=" * 76)
    print(f"  {'':<34}{'Arm A':>12}{'Arm B':>12}{'B/A':>8}")
    print(f"  {'English-internal / native CoT':<34}{'':>12}{'':>12}")
    print(f"  {'trained tokens':<34}{a_tot:>12.0f}{b_tot:>12.0f}{b_tot/a_tot:>8.2f}x")
    print(f"  {'of which translated':<34}{a_translated:>12.0f}{b_translated:>12.0f}"
          f"{b_translated/a_translated:>8.2f}x")
    print(f"  {'translated share of example':<34}{a_translated/a_tot*100:>11.0f}%"
          f"{b_translated/b_tot*100:>11.0f}%")

    print("\n" + "=" * 76)
    print(f"AT SCALE — cross-lingual slice = {XLING_SHARE:.0%} of the {REASON_LANE_B:.0f}B "
          f"reasoning lane = {REASON_LANE_B*XLING_SHARE:.1f}B tokens")
    print("=" * 76)
    budget = REASON_LANE_B * XLING_SHARE * 1e9
    print(f"  {'':<34}{'Arm A':>14}{'Arm B':>14}{'B/A':>8}")
    for label, tot, tr in (("", a_tot, a_translated), ("", b_tot, b_translated)):
        pass
    na, nb = budget / a_tot, budget / b_tot
    tra, trb = na * a_translated, nb * b_translated
    srca, srcb = na * en_example, nb * en_example
    print(f"  {'examples needed':<34}{na/1e6:>13.0f}M{nb/1e6:>13.0f}M{nb/na:>8.2f}x")
    print(f"  {'tokens to translate + verify':<34}{tra/1e9:>13.1f}B{trb/1e9:>13.1f}B"
          f"{trb/tra:>8.2f}x")
    print(f"  {'English source drawn from AON':<34}{srca/1e9:>13.1f}B{srcb/1e9:>13.1f}B")
    print(f"  {'as % of our 78B AON corpus':<34}{srca/1e9/AON_B*100:>12.0f}%"
          f"{srcb/1e9/AON_B*100:>12.0f}%")

    print("\n" + "=" * 76)
    print("THE COST DIFFERENCE IS NOT THE DECIDING FACTOR")
    print("=" * 76)
    print(f"  Arm B costs {trb/tra:.1f}x the translation volume, but the harder problem is that")
    print("  what it translates is UNVERIFIABLE. Arm A translates questions and short")
    print("  answers -- often integers or multiple-choice options, checkable by exact match,")
    print("  the same 'verifier is the label' rule the agentic programme runs on. Arm B")
    print("  translates free-form reasoning chains, where a fluent mistranslation is")
    print("  indistinguishable from a correct one and errors compound across steps.")
    print()
    print("  So the real question T5 answers is not which arm is cheaper. It is whether")
    print("  Arm B buys anything at all -- because if the model reasons in English")
    print("  internally regardless (cross-lingual collapse), Arm B pays "
          f"{trb/tra:.1f}x for a veneer.")


if __name__ == "__main__":
    main()
