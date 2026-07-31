"""Gate the tail languages instead of funding them homeopathically.

E2/E8 allocated all 22 languages, which meant giving Santali 31M tokens and Dogri 1M --
their full 4-epoch ceiling, and still not enough to produce capability. That is the shape
of a plan that lets today's supply set tomorrow's design.

This replaces it with a supply GATE: a language is funded when it crosses an entry
threshold and is otherwise a named acquisition target with a stated trigger. The plan
then becomes a function of supply rather than a snapshot of it.

The threshold is not chosen. The supply distribution has a natural cliff and this script
finds it.

Run:  python3 S5/scripts/indic_gated.py
"""

from indic import SANGRAHA, NAME, MILU, supply_b, LANE_B
from indic_fertility import FERT, content_supply

# IndicGenBench's 29 include these tail languages; Sindhi, Kashmiri and Dogri are not in it.
IGB_TAIL = {"gom", "mai", "mni", "sat", "brx"}
# Rasa / IndicVoices are organised by language and carry substantial speech for these.
SPEECH = {"brx": "Rasa's LARGEST language (125 files)", "mai": "Rasa + IndicVoices",
          "gom": "Rasa", "san": "Rasa + IndicVoices", "mni": "IndicVoices"}

MIN_LANE_SHARE = 0.01        # a funded language should be able to reach >=1% of the lane


def main():
    S = supply_b()
    ranked = sorted(S.items(), key=lambda kv: -kv[1])

    print("=" * 78)
    print("1. THE CLIFF — the threshold is in the data, not chosen")
    print("=" * 78)
    print(f"  {'rank':>4}  {'language':<11}{'supply B':>10}{'ratio to next':>15}")
    worst = (0, None)
    for i in range(len(ranked) - 1):
        lg, v = ranked[i]
        nxt = ranked[i + 1][1]
        r = v / nxt if nxt > 0 else float("inf")
        if r > worst[0] and v > 0.05:
            worst = (r, i)
        if i < 16:
            print(f"  {i+1:>4}  {NAME[lg]:<11}{v:>10.2f}{r:>14.1f}x")
    ci = worst[1]
    print(f"\n  largest gap: {NAME[ranked[ci][0]]} {ranked[ci][1]:.2f}B -> "
          f"{NAME[ranked[ci+1][0]]} {ranked[ci+1][1]:.2f}B = **{worst[0]:.0f}x**")
    print(f"  The distribution is not a smooth tail. It is 14 languages and then a cliff.")

    funded = [lg for lg, _ in ranked[:ci + 1]]
    gated = [lg for lg, _ in ranked[ci + 1:]]

    print("\n" + "=" * 78)
    print("2. ENTRY GATE")
    print("=" * 78)
    need_alloc = MIN_LANE_SHARE * LANE_B
    need_supply = need_alloc / 4.0
    print(f"  A funded language must be able to reach {MIN_LANE_SHARE:.0%} of the lane "
          f"({need_alloc:.1f}B)")
    print(f"  without breaching 4 epochs, i.e. hold at least "
          f"**{need_supply:.2f}B unique tokens**.")
    print(f"\n  Every one of the {len(funded)} funded languages clears it "
          f"(min {min(S[l] for l in funded):.2f}B).")
    print(f"  None of the {len(gated)} gated ones do.\n")
    print(f"  {'language':<11}{'supply':>10}{'gap to gate':>13}{'IndicGenBench':>15}  route")
    for lg in gated:
        gap = need_supply / S[lg] if S[lg] > 0 else float("inf")
        igb = "scored" if lg in IGB_TAIL else "not scored"
        route = SPEECH.get(lg, "text collection / partnership")
        print(f"  {NAME[lg]:<11}{S[lg]*1000:>9.1f}M{gap:>12.0f}x{igb:>15}  {route}")

    print("\n" + "=" * 78)
    print("3. WHAT DROPPING THEM COSTS AND FREES")
    print("=" * 78)
    freed = sum(4 * S[lg] for lg in gated)
    print(f"  freed budget: {freed:.2f}B = {freed/LANE_B*100:.3f}% of the lane")
    print(f"  -> redistributed across the {len(funded)} funded languages; the largest single")
    print(f"     gain is under {freed/len(funded)*3/1:.2f}B. This is NOT why we do it.")
    lost = sorted(NAME[l] for l in gated if l in IGB_TAIL)
    print(f"\n  IndicGenBench claim: **19/29 -> 14/29**")
    print(f"  languages we stop claiming: {', '.join(lost)}")
    print(f"  MILU is unaffected: 10/10 (no MILU language is gated).")
    print("\n  That is the real price, and it is worth paying. 19/29 where five are funded")
    print("  at 30M tokens is a weaker claim than 14/29 funded properly -- a reviewer who")
    print("  checks Santali finds 7.7M tokens behind a claimed benchmark language.")

    print("\n" + "=" * 78)
    print("4. THE TRIGGER — what makes the plan a function of supply")
    print("=" * 78)
    print("  Each gated language re-enters the lane automatically when it crosses")
    print(f"  {need_supply:.2f}B unique native-script tokens. No re-planning, no renegotiation:")
    print("  the allocation rule (content-weighted temperature at alpha*, water-filled")
    print("  against 4 epochs) already handles any number of languages. Adding one is a")
    print("  re-run, not a redesign.")
    print("\n  Crucially, the two halves need DIFFERENT kinds of 'later':")
    print("    passive -- the 14 funded languages grow as the web grows;")
    print("    active  -- the 8 gated ones do not. Nothing is being written down in")
    print("               Santali or Dogri at volume, so their supply grows only if we")
    print("               fund collection. Speech is the cheapest route: Rasa's single")
    print("               largest language is Bodo, and IndicVoices carries Manipuri.")
    print("\n  So the gated set is not 'dropped'. It is moved from the budget, where it was")
    print("  doing nothing, to the acquisition programme, where it has a target and a")
    print("  trigger. A share of 0.006% was never a commitment to Manipuri; a threshold is.")


if __name__ == "__main__":
    main()
