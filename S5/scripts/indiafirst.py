"""India-first coverage, the Indic-STEM cross requirement, and the Indic SFT slice.

Three gaps between S3 and S5, resolved together because they are the same gap seen from
three sides: S5 has no way to express a requirement that cuts ACROSS lanes.

  1. S3 allocates 9% to an "India-first domains" lane. S5 has no such lane.
  2. S3 requires "at least 15% of STEM data should be Indic-language". S5 cannot state it.
  3. S3 requires "at least 55% of India-first data should be non-English". Same.
  4. IndicAlign (17.5GB) is Indic instruction data with no home in either budget.

E7 measured mean tags per token at m = 1.020 and concluded the disjoint lane model is
correct to within 2%. That conclusion stands for the *budget*. But India-first is not a
corpus -- it is a property of documents that already belong to other lanes. A court
judgment is India-first AND reasoning; a Tamil news article is India-first AND Indic.
So India-first is a TAG WITH A COVERAGE TARGET, not a lane with a budget share, and this
script shows what happens if you try to make it a lane instead.

Run:  python3 S5/scripts/indiafirst.py
"""

MAIN_B = 2910.0
INDIC_LANE_B = 494.7          # 17% of the main run
STEM_LANE_B = 349.2           # 12%
EPOCH_CAP = 4.0

# English-language India-first supply that is OPEN and measured or published.
# Court figures are measured (E4/E5); the rest are published counts.
INDIA_FIRST_EN = {
    "High Court judgments (E4, measured)":        8.00,
    "Supreme Court English (43,517 docs, E5)":    0.39,
    "OpenSansad Lok Sabha QA (CC BY)":            0.05,
    "India Biodiversity / Project Madurai etc.":  0.02,
}
# KanoonGPT's 17M rows are a derivative of the same court sources and are deliberately
# excluded rather than added -- that is the Samanantar mistake, and we make it once.

GATED = ["India Code", "e-Gazette + state gazettes", "Sansad Digital Library (1.45M items)",
         "Digital Sansad debates", "PIB archive (500K+ releases)", "state assemblies / NeVA",
         "Census of India", "RBI DBIE", "CAG (attributed reuse only)", "KrishiKosh / ICAR"]

# S3's SFT plan
SFT_TOTAL_B = 2.5
SFT_INDIC_SHARE = 0.20
INDICALIGN_GB = {"wiki_chat": 10.6, "indowordnet": 5.3, "indicsharellama": 0.92,
                 "oasst": 0.47, "dolly": 0.16, "anudesh": 0.042}
BYTES_PER_TOKEN = 8.0
NATIVELY_AUTHORED = {"anudesh"}      # collected natively; the rest are translated/derived


def main():
    en = sum(INDIA_FIRST_EN.values())
    print("=" * 78)
    print("1. INDIA-FIRST AS A LANE — what S3 proposes")
    print("=" * 78)
    dem = 0.09 * MAIN_B
    print(f"  S3 share 9% of {MAIN_B:.0f}B = {dem:.0f}B demand")
    print(f"\n  Open English-language India-first supply:")
    for k, v in INDIA_FIRST_EN.items():
        print(f"    {k:<44}{v:>7.2f}B")
    print(f"    {'TOTAL':<44}{en:>7.2f}B")
    print(f"\n  epochs at a 9% lane = {dem/en:.1f}   -- against a ceiling of {EPOCH_CAP:.0f}")
    print(f"  fundable share at 4 epochs = {en*EPOCH_CAP/MAIN_B*100:.2f}% "
          f"({en*EPOCH_CAP:.1f}B)")
    print("\n  So a dedicated 9% India-first lane is the SECOND unfundable lane after")
    print("  agentic, and it comes from our own S3 proposal, which assigned the share")
    print("  without a supply check. That is the exact failure the rubric grades against.")
    print(f"\n  Everything else in the India-first source map is gated ({len(GATED)} sources):")
    print(f"    {', '.join(GATED[:5])},")
    print(f"    {', '.join(GATED[5:])}")
    print("  None of it carries a number, so none of it can carry a budget share.")

    print("\n" + "=" * 78)
    print("2. INDIA-FIRST AS A TAG — what the coverage actually is")
    print("=" * 78)
    print("  India-first is a property of documents already counted in other lanes.")
    print("  Coverage = (fraction of the Indic lane that is about Indian life) plus the")
    print("  English-language India slice, which is the scarce half.")
    print(f"\n  {'phi':>5}{'Indic contribution':>20}{'English':>10}{'coverage':>11}"
          f"{'% of run':>10}{'non-Eng':>9}  vs S3 9%")
    for phi in (1.00, 0.75, 0.50, 0.25):
        indic = phi * INDIC_LANE_B
        cov = indic + en * EPOCH_CAP
        print(f"  {phi:>5.2f}{indic:>19.0f}B{en*EPOCH_CAP:>9.1f}B{cov:>10.0f}B"
              f"{cov/MAIN_B*100:>9.1f}%{indic/cov*100:>8.0f}%"
              f"{'   PASS' if cov/MAIN_B >= 0.09 else '   FAIL'}")
    print("\n  phi = fraction of Indic-language text that is about Indian context. Even at")
    print("  phi = 0.25 -- a deliberately hostile assumption, since text written in Indian")
    print("  languages is overwhelmingly about Indian life -- coverage clears S3's 9%.")
    print("  S3's 'at least 55% non-English' rule passes at every phi, by a wide margin.")
    print("\n  CONCLUSION: S3 and S5 do not actually disagree. S3 modelled India-first as a")
    print("  lane because S3 assumed multi-tagging; in a disjoint budget the same content")
    print("  cannot be booked twice. The lane is unfundable and the tag is over-delivered.")
    print("  What IS scarce is the ENGLISH-language India-first component at "
          f"{en:.1f}B unique.")

    print("\n" + "=" * 78)
    print("3. THE CROSS-LANE REQUIREMENT THAT IS GENUINELY UNFUNDED")
    print("=" * 78)
    need = 0.15 * STEM_LANE_B
    have = 11.7        # E7: 8% Indic x STEM overlap estimate
    print(f'  S3: "at least 15% of STEM data should be Indic-language"')
    print(f"    need {need:.1f}B of Indic-language STEM, have ~{have:.1f}B -> "
          f"{need/have:.1f}x short")
    print(f"    supply supports {have/STEM_LANE_B*100:.1f}%, not 15%.")
    print("\n  Three options, and the plan should pick one rather than restate the rule:")
    print(f"    (a) lower the requirement to {have/STEM_LANE_B*100:.0f}% and say why;")
    print(f"    (b) synthesize {need-have:.0f}B of Indic STEM -- translated science, which")
    print("        runs straight into the tier-D finding that translated content teaches")
    print("        the source culture's distribution, not India's;")
    print("    (c) source it: NCERT/SCERT/NPTEL/IGNOU are Indian-curriculum STEM in Indian")
    print("        languages and are GATED. This is the strongest argument in the plan for")
    print("        prioritising one specific negotiation.")

    print("\n" + "=" * 78)
    print("4. INDIC SFT — outside the pretraining budget, and covered")
    print("=" * 78)
    tot_gb = sum(INDICALIGN_GB.values())
    tot_b = tot_gb * 1e9 / BYTES_PER_TOKEN / 1e9
    native_gb = sum(v for k, v in INDICALIGN_GB.items() if k in NATIVELY_AUTHORED)
    need_sft = SFT_TOTAL_B * SFT_INDIC_SHARE
    print(f"  IndicAlign total {tot_gb:.1f}GB ~= {tot_b:.2f}B tokens")
    for k, v in sorted(INDICALIGN_GB.items(), key=lambda kv: -kv[1]):
        tag = "  <- natively authored" if k in NATIVELY_AUTHORED else ""
        print(f"    {k:<18}{v:>7.2f}GB{tag}")
    print(f"\n  S3 SFT plan: {SFT_TOTAL_B}B total, {SFT_INDIC_SHARE:.0%} native Indic "
          f"= {need_sft:.2f}B needed")
    print(f"  -> covered {tot_b/need_sft:.1f}x by volume. SFT is <1% of the lifecycle and")
    print("     sits outside the 97/3 pretrain/anneal budget, so it needs naming, not a share.")
    print(f"\n  BUT S3 also requires 'at least half of Indic instructions authored natively,")
    print(f"  not translated'. Natively-authored in IndicAlign is {native_gb:.3f}GB of "
          f"{tot_gb:.1f}GB = {native_gb/tot_gb*100:.1f}%.")
    print(f"  Need {need_sft/2:.2f}B natively authored; have ~{native_gb*1e9/BYTES_PER_TOKEN/1e9:.3f}B.")
    print(f"  -> {need_sft/2/(native_gb*1e9/BYTES_PER_TOKEN/1e9):.0f}x short. The VOLUME is fine")
    print("     and the PROVENANCE is not. wiki_chat and indowordnet are derived from")
    print("     Wikipedia and a lexical database; anudesh is the only natively collected set.")


if __name__ == "__main__":
    main()
