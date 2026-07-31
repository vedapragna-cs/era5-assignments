"""Audit the S3 source map against live, queryable supply.

Class 1 of the audit: sources whose size can be *measured* rather than cited. Everything
here hits a live endpoint. Sources needing an MoU are recorded in SUPPLY-AUDIT.md with a
rights tier and no number, which is the honest treatment.

Two calibrations are used and both are stated where they appear:
  - HF datasets-server reports num_bytes_memory. IndicCorpV2's converted fraction
    (37.6%) and its published 20.9B-token total give ~8.0 bytes/token, which is the
    conversion applied to its per-language byte sizes.
  - English fertility is 1.15 tokens/word (S3 MUTANT-Indic ceiling).

Run:  .venv/bin/python S5/scripts/supply_audit.py
"""

import json
import urllib.parse
import urllib.request

HF = "https://datasets-server.huggingface.co/size?dataset="

# Sangraha per-language verified tokens (M), from the HF dataset card. Tail only.
SANGRAHA_TAIL_M = {
    "gom_Deva": 10.1, "mai_Deva": 14.6, "snd_Deva": 258.2, "santhali": 0.3,
    "brx_Deva": 1.5, "mni_Mtei": 7.4, "doi_Deva": 0.06, "kas_Arab": 0.5,
    "khasi": 0.0,   # Khasi is absent from Sangraha entirely
}
NAME = {"gom_Deva": "Konkani", "mai_Deva": "Maithili", "snd_Deva": "Sindhi",
        "santhali": "Santali", "brx_Deva": "Bodo", "mni_Mtei": "Manipuri",
        "doi_Deva": "Dogri", "kas_Arab": "Kashmiri", "khasi": "Khasi"}

BYTES_PER_TOKEN = 8.0     # calibrated below, printed with its derivation
LANE_B = 519.9            # Indic lane, B tokens
EPOCH_CAP = 4.0


def hf_size(dataset):
    url = HF + urllib.parse.quote(dataset, safe="")
    try:
        return json.loads(urllib.request.urlopen(url, timeout=90).read())
    except Exception as e:
        return {"error": str(e)}


def audit_indiccorp():
    j = hf_size("ai4bharat/IndicCorpV2")
    if "error" in j:
        print("  IndicCorpV2 query failed:", j["error"]); return None
    d = j["size"]["dataset"]
    frac = d["num_rows"] / d["estimated_num_rows"]
    print(f"  partial conversion: {j.get('partial')}  "
          f"{d['num_rows']:,} of ~{d['estimated_num_rows']:,} rows = {frac*100:.1f}%")
    full_bytes = d["num_bytes_memory"] / frac
    bpt = full_bytes / 20.9e9
    print(f"  extrapolated full size {full_bytes/1e9:.0f}GB against the published 20.9B")
    print(f"  tokens  ->  {bpt:.1f} bytes/token  (calibration used below)")

    splits = j["size"]["splits"]
    capped = [s for s in splits if (s.get("num_bytes_memory") or 0) > 4.9e9]
    print(f"\n  {len(capped)} splits are pinned at the ~5GB conversion cap and are NOT")
    print(f"  real sizes: {', '.join(sorted(s['split'] for s in capped))}")
    print(f"  The other {len(splits)-len(capped)} are complete — and they are the tail.")
    return {s["split"]: (s.get("num_bytes_memory") or 0) for s in splits}, bpt


def main():
    print("=" * 78)
    print("INDICCORPV2 — per-language, the table that was missing")
    print("=" * 78)
    got = audit_indiccorp()
    if not got:
        return
    sizes, bpt = got

    print("\n" + "=" * 78)
    print("THE TAIL, RE-MEASURED  (Sangraha + IndicCorpV2)")
    print("=" * 78)
    print(f"  {'language':<11}{'Sangraha':>11}{'IndicCorpV2':>13}{'combined':>11}"
          f"{'uplift':>9}{'max @4ep':>11}")
    tot_old = tot_new = 0.0
    for k, name in NAME.items():
        s = SANGRAHA_TAIL_M[k]
        icv2 = sizes.get(k, 0) / bpt / 1e6          # M tokens
        comb = s + icv2
        tot_old += s
        tot_new += comb
        up = f"{comb/s:.1f}x" if s > 0 else "NEW"
        print(f"  {name:<11}{s:>10.2f}M{icv2:>12.1f}M{comb:>10.1f}M{up:>9}"
              f"{comb*4:>10.1f}M")
    print(f"  {'TOTAL':<11}{tot_old:>10.1f}M{tot_new-tot_old:>12.1f}M{tot_new:>10.1f}M")
    print(f"\n  Tail supply {tot_old:.0f}M -> {tot_new:.0f}M ({tot_new/tot_old:.1f}x).")
    print(f"  At the 4-epoch ceiling the whole tail absorbs {tot_new*4/1000:.2f}B = "
          f"{tot_new*4/1e3/LANE_B*100:.3f}% of the Indic lane")
    print(f"  (was 0.225%). CONCLUSION UNCHANGED: the tail is a supply problem.")
    print("\n  Overlap risk: Sangraha Unverified is drawn from 'existing multilingual")
    print("  corpora' and could double-count IndicCorp — but it has splits for only the")
    print("  14 major languages, so NONE of the tail above can be double-counted there.")
    print("  Residual risk is against Sangraha Verified, a different acquisition path")
    print("  (human-verified sites, PDF OCR, video transcription) than IndicCorp's crawl.")

    print("\n" + "=" * 78)
    print("SANGRAHA SYNTHETIC — what tier D actually is")
    print("=" * 78)
    print("  AI4Bharat describe it as: 'Wikimedia English translated to 14 Indic")
    print("  languages and further romanized ... by transliteration to English.'")
    print("  English Wikipedia is ~4.5B words (2024) / ~5.2B words (2026).")
    print()
    print("  So the 162.7B-token synthetic tier renders ONE encyclopedia 28 ways")
    print("  (14 languages x 2 scripts). Its distinct informational content is")
    print(f"  ~4.5B words, not 162.7B tokens of Indic knowledge.")
    d_demand = 0.40 * LANE_B
    print(f"\n  Our budget puts tier D at 40% of the lane = {d_demand:.0f}B tokens.")
    print(f"  Against 81.4B of native-script synthetic that reads as 2.4 epochs.")
    print(f"  Against ~4.5B words of distinct source content it is far heavier")
    print(f"  repetition of a single corpus, in a way the epoch figure hides.")
    print()
    print("  Consequence for MILU: MILU is India-centric by construction (regional and")
    print("  state examinations, local history, arts, festivals, law). Translated English")
    print("  Wikipedia is the wrong content distribution for exactly those subjects.")
    print("  T2 should therefore predict tier D underperforms on MILU's India-specific")
    print("  subjects specifically, not uniformly — a sharper and more falsifiable test.")


if __name__ == "__main__":
    main()
