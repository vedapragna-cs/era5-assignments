"""Model lanes as overlapping tag sets rather than disjoint buckets.

S3 asserts a mechanism this plan never modelled: "Documents receive multiple tags, so a
Tamil physics lesson satisfies both Tamil and STEM targets even though it is sampled only
once." Every budget table in S5 assumes lanes are disjoint -- each token belongs to
exactly one lane, demands sum to 100%, and supplies do not intersect.

If documents really carry multiple tags, three things change:

  1. Lane shares no longer have to sum to 100%. Coverage is counted per tag, so a token
     tagged {Indic, STEM} advances two lanes at once.
  2. A lane's usable supply grows: it is every document carrying that tag, including ones
     a disjoint inventory filed under some other lane.
  3. Conversely, CROSS-LANE requirements become expressible -- and checkable. S3 states
     two: "at least 15% of STEM data should be Indic-language" and "at least 55% of
     India-first data should be non-English". A disjoint model cannot even represent
     these, so it never checked them.

The honest danger is (1). Multi-tagging is the most natural way to make an infeasible
budget look feasible -- every scarce token counted twice -- which is precisely the
wishful accounting this plan exists to catch. So the model below is built to answer one
question: how much overlap ACTUALLY exists in our inventory, and what would it have to be
to matter?

Run:  python3 S5/scripts/multitag.py
"""

# lane -> (share of main run, unique supply B tokens) from BUDGET-PROPOSAL/SUPPLY-AUDIT
LANES = {
    "web":     (0.34, 4691.0),
    "code":    (0.24, 1103.0),
    "indic":   (0.17,  157.2),
    "stem":    (0.12,  146.0),
    "reason":  (0.09,   85.1),
    "longctx": (0.02,  100.0),
    "agentic": (0.02,    0.63),
}
MAIN_B = 2910.0

# Pairwise overlap: fraction of the SMALLER lane's supply that also carries the other tag.
# 'inventory' = derivable from our own dataset rows. 'estimate' = reasoned, needs measuring.
OVERLAP = {
    # 60B of the 100B long-context lane IS repo-packed code -- the same underlying corpus
    # as the code lane. This one is not an estimate; it is how the lane was constructed.
    ("longctx", "code"):   (0.60, "inventory"),
    # agentic trajectories are patches, shell sessions and repo files: code by content
    ("agentic", "code"):   (0.50, "inventory"),
    # proof-pile-2 (formal + informal math) overlaps AON's math reasoning
    ("stem", "reason"):    (0.15, "estimate"),
    # book-length packed documents in long-context are web/reference prose
    ("longctx", "web"):    (0.40, "inventory"),
    # the ones that would actually relieve the tight lane, and the ones we cannot verify:
    ("indic", "stem"):     (0.08, "estimate"),   # STEM-ish fraction of Sangraha-style crawl
    ("indic", "code"):     (0.01, "estimate"),
    ("indic", "reason"):   (0.005, "estimate"),
    ("indic", "web"):      (0.00, "inventory"),  # web lane is DCLM/FineWeb: English corpora
}

# Cross-lane requirements asserted in S3 that a disjoint model cannot express.
S3_REQS = [
    ("stem", "indic", 0.15, "at least 15% of STEM data should be Indic-language"),
]


def tag_inclusive_supply():
    """A lane's supply under multi-tagging = its own corpus plus the tagged slices of others."""
    out = {}
    for lane, (_, sup) in LANES.items():
        extra = 0.0
        for (a, b), (frac, _) in OVERLAP.items():
            if lane == a:
                extra += frac * min(sup, LANES[b][1])
            elif lane == b:
                extra += frac * min(sup, LANES[a][1])
        out[lane] = sup + extra
    return out


def main():
    print("=" * 78)
    print("1. WHAT MULTI-TAGGING CHANGES ARITHMETICALLY")
    print("=" * 78)
    tag_instances = sum(s for s, _ in LANES.values())
    print(f"  Disjoint model: shares sum to {tag_instances:.0%} and each token serves one lane.")
    print("  Multi-tag model: shares sum to m x 100%, where m = mean tags per token.")
    print("  Distinct tokens needed for the same coverage = T x (sum of shares) / m.")
    print("\n  So the relief is exactly m. m=1.0 is the disjoint model and buys nothing;")
    print("  m=1.3 would deliver 130% of lane shares in the same run.")
    print("  m is NOT a modelling choice -- it is a property of the corpus, and it is what")
    print("  has to be measured before any share is raised on the strength of it.")

    print("\n" + "=" * 78)
    print("2. OVERLAP ACTUALLY PRESENT IN OUR INVENTORY")
    print("=" * 78)
    print(f"  {'pair':<22}{'overlap':>9}{'shared B':>11}  basis")
    shared_total = 0.0
    for (a, b), (frac, basis) in sorted(OVERLAP.items(), key=lambda kv: -kv[1][0]):
        sh = frac * min(LANES[a][1], LANES[b][1])
        shared_total += sh
        print(f"  {a+' x '+b:<22}{frac:>8.1%}{sh:>11.1f}  {basis}")
    total_unique = sum(s for _, s in LANES.values())
    m = 1 + shared_total / total_unique
    print(f"\n  total unique supply      {total_unique:>9.1f}B")
    print(f"  total shared (tagged 2+) {shared_total:>9.1f}B")
    print(f"  implied mean tags/token  m = {m:.3f}")
    print(f"\n  -> multi-tagging would let shares sum to {m:.1%}, not 100%.")
    print(f"     That is {m-1:.1%} of extra coverage, and almost all of it sits in")
    print("     code/long-context/agentic, which are ALREADY covered lanes.")

    print("\n" + "=" * 78)
    print("3. DOES IT RELIEVE THE TIGHT LANES?  (this is the whole question)")
    print("=" * 78)
    inc = tag_inclusive_supply()
    print(f"  {'lane':<9}{'demand B':>10}{'disjoint':>10}{'ep':>7}"
          f"{'tag-incl':>10}{'ep':>7}{'change':>9}")
    for lane, (share, sup) in sorted(LANES.items(), key=lambda kv: -kv[1][0]):
        dem = share * MAIN_B
        e0, e1 = dem / sup, dem / inc[lane]
        flag = "" if e1 < 3.9 else "  <-still tight"
        print(f"  {lane:<9}{dem:>10.1f}{sup:>10.1f}{e0:>7.2f}{inc[lane]:>10.1f}"
              f"{e1:>7.2f}{(e1-e0):>+9.2f}{flag}")
    print("\n  Indic and reason are the tight lanes. Their overlap with everything else is")
    print("  near zero BY CORPUS CONSTRUCTION: the web lane is DCLM/FineWeb (English), the")
    print("  STEM lane is peS2o/proof-pile-2 (English papers), the reasoning lane is AON")
    print("  (English). Sangraha does not intersect any of them.")
    print("  The lanes that DO overlap heavily -- code, long-context, agentic -- are the")
    print("  ones already sitting at 0.65 epochs or below, where relief is worthless.")

    print("\n" + "=" * 78)
    print("4. THE CONSTRAINT MULTI-TAGGING ADDS (and a disjoint model could not see)")
    print("=" * 78)
    for lane, tag, req, text in S3_REQS:
        dem = LANES[lane][0] * MAIN_B
        need = req * dem
        frac, basis = OVERLAP.get((tag, lane)) or OVERLAP.get((lane, tag))
        have = frac * min(LANES[lane][1], LANES[tag][1])
        print(f'  S3: "{text}"')
        print(f"    {lane} demand {dem:.0f}B  x  {req:.0%}  =  {need:.1f}B of {tag}-tagged {lane}")
        print(f"    available at {frac:.1%} overlap ({basis})        =  {have:.1f}B")
        print(f"    -> shortfall {need-have:.1f}B = {need/have:.1f}x  **UNFUNDED**")
        need_frac = need / min(LANES[lane][1], LANES[tag][1])
        print(f"    would need {need_frac:.0%} of Indic supply to be STEM content.")

    print("\n" + "=" * 78)
    print("5. WHAT WOULD HAVE TO BE TRUE FOR MULTI-TAGGING TO RESCUE INDIC")
    print("=" * 78)
    dem = LANES["indic"][0] * MAIN_B
    for target_ep in (3.0, 2.5, 2.0):
        need_sup = dem / target_ep
        extra = need_sup - LANES["indic"][1]
        frac_of_stem = extra / LANES["stem"][1]
        print(f"  to reach {target_ep:.1f} epochs Indic needs {need_sup:.0f}B tagged supply "
              f"= +{extra:.0f}B,")
        print(f"    i.e. {frac_of_stem:.0%} of the entire STEM corpus would have to be "
              f"Indic-language.")
    print("\n  peS2o is open-access English papers and proof-pile-2 is English mathematics.")
    print("  Neither is plausibly Indic at any such rate. Multi-tagging does not rescue")
    print("  this lane, and a plan that claimed it did would be double-counting.")


if __name__ == "__main__":
    main()
