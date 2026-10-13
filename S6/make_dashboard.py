#!/usr/bin/env python3
"""Render submission_artifacts/ into one self-contained dashboard.html.

    python make_dashboard.py

Scope, stated because the session describes a four-server architecture and then carves it out of the
assignment: this is **not** a service. It is a static rendering of artifacts that already exist, with
no network, no dependencies and no live data. Everything it shows is recomputed from
`submission_artifacts/` at render time, so it cannot display a number the artifacts do not contain.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent / "submission_artifacts"
OUT = Path(__file__).parent / "dashboard.html"


def load(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if p.exists() else []


def esc(x) -> str:
    return html.escape(str(x))


def bar(pct: float, colour: str) -> str:
    return (f'<div class="bar"><span style="width:{max(0.5, min(100, pct)):.2f}%;'
            f'background:{colour}"></span></div>')


def main() -> int:
    ev = load(ROOT / "evidence.json")
    if ev is None:
        print(f"no artifacts under {ROOT} - run run_demo.py first")
        return 2
    perf = load(ROOT / "performance.json", {})
    feas = load(ROOT / "manifests" / "feasibility.json", {})
    plan = load(ROOT / "manifests" / "plan.json", {})
    v6 = load(ROOT / "reports" / "v6_recommendations.json", {})
    ctrls = load(ROOT / "reports" / "negative_controls.json", {})
    exps = load(ROOT / "reports" / "experiments.json", {})
    xo = load(ROOT / "reports" / "exactly_once.json", {})
    vol = load(ROOT / "reports" / "ledger_volume.json", {})
    osum = load(ROOT / "ledgers" / "opus_summary.json", {})
    batches = [e for e in jsonl(ROOT / "ledgers" / "consumption.jsonl")
               if e.get("event") == "batch_committed"]
    # Chart the MAIN branch only: run-b is an intentional fork from an early checkpoint, so
    # splicing its steps onto run-a's would draw a loss curve that never happened.
    all_steps = perf.get("steps_detail", [])
    steps = [s for s in all_steps if s.get("branch", "run-a") == "run-a"]

    meta = ev["meta"]
    summ = ev["summary"]

    # ---- loss curve as inline SVG -----------------------------------------------------------
    losses = [s["mean_loss"] for s in steps]
    svg = ""
    if len(losses) > 1:
        lo, hi = min(losses), max(losses)
        rng = (hi - lo) or 1.0
        pts = " ".join(
            f"{40 + i / (len(losses) - 1) * 620:.1f},{150 - (v - lo) / rng * 120:.1f}"
            for i, v in enumerate(losses))
        svg = f'''<svg viewBox="0 0 680 180" class="chart" role="img"
      aria-label="mean loss per optimizer step, branch run-a">
  <line x1="40" y1="30" x2="40" y2="150" class="ax"/>
  <line x1="40" y1="150" x2="660" y2="150" class="ax"/>
  <polyline points="{pts}" class="line"/>
  <text x="6" y="34" class="tick">{hi:.4f}</text>
  <text x="6" y="154" class="tick">{lo:.4f}</text>
  <text x="40" y="172" class="tick">step 0</text>
  <text x="600" y="172" class="tick">step {len(losses)-1}</text>
</svg>'''

    # ---- lane delivery ------------------------------------------------------------------------
    tally: dict[str, int] = {}
    for e in batches:
        if e.get("branch_id") != "run-a":
            continue
        for ln, n in e["mixture_lanes"].items():
            tally[ln] = tally.get(ln, 0) + n
    tot = sum(tally.values()) or 1
    floors = plan.get("protected_floors", {})
    lane_rows = ""
    for ln in sorted(tally):
        share = tally[ln] / tot
        f = floors.get(ln)
        flag = ""
        if f is not None:
            ok = share >= f - 1e-9
            flag = (f'<span class="pill {"ok" if ok else "bad"}">floor {f:.0%} '
                    f'{"held" if ok else "BREACHED"}</span>')
        lane_rows += (f"<tr><td>{esc(ln)}</td><td class='num'>{share:.2%}</td>"
                      f"<td>{bar(share*100, 'var(--c1)')}</td><td>{flag}</td></tr>")

    # ---- feasibility --------------------------------------------------------------------------
    feas_rows = ""
    for r in feas.get("lanes", []):
        ok = r["within_ceiling"]
        feas_rows += (
            f"<tr><td>{esc(r['lane'])}</td>"
            f"<td class='num'>{r['demand_tokens']/1e9:,.1f}B</td>"
            f"<td class='num'>{r['supply_tokens']/1e9:,.1f}B</td>"
            f"<td class='num'>{r['epochs']:.2f}</td>"
            f"<td><span class='pill {'ok' if ok else 'bad'}'>"
            f"{'within ceiling' if ok else 'BREACH'}</span></td></tr>")

    # ---- lane perplexity ----------------------------------------------------------------------
    ppl_rows = ""
    by_lane = v6.get("by_lane", {})
    if by_lane:
        mx = max(d["geometric_mean_ppl"] for d in by_lane.values()) or 1
        for ln, d in sorted(by_lane.items(), key=lambda kv: -kv[1]["geometric_mean_ppl"]):
            ppl_rows += (f"<tr><td>{esc(ln)}</td>"
                         f"<td class='num'>{d['geometric_mean_ppl']:,.2f}</td>"
                         f"<td>{bar(d['geometric_mean_ppl']/mx*100, 'var(--c2)')}</td>"
                         f"<td class='num'>{d['hot_token_pct']:.1f}%</td>"
                         f"<td class='num'>{d['n_tokens']:,}</td></tr>")

    # ---- V6 recommendations -------------------------------------------------------------------
    v6_rows = ""
    for c in v6.get("shard_cards", []):
        r = c["recommendation"]
        cls = {"protect_or_upsample": "ok", "delay_or_reclean": "bad"}.get(r["action"], "warn")
        v6_rows += (f"<tr><td><code>{esc(c['shard_id'])}</code></td>"
                    f"<td>{esc(c['capability_lane'])}</td>"
                    f"<td><span class='pill {cls}'>{esc(r['action'])}</span></td>"
                    f"<td class='why'>{esc(r['because'])}</td>"
                    f"<td class='num'>{esc(c['best_phase'])}</td></tr>")

    req_rows = "".join(
        f"<tr><td>{esc(r['requirement'])}</td>"
        f"<td><span class='pill {'ok' if r['result']=='PASS' else 'bad'}'>{r['result']}</span></td>"
        f"<td><code>{esc(', '.join(r['evidence_paths']))}</code></td></tr>"
        for r in ev["requirements"])

    pack_rows = "".join(
        f"<tr><td>{esc(r['policy'])}</td><td class='num'>{r['n_sequences']}</td>"
        f"<td class='num'>{r['utilization']:.1%}</td>"
        f"<td>{bar(r['utilization']*100, 'var(--c3)')}</td>"
        f"<td class='num'>{r['loss_bearing_fraction']:.1%}</td>"
        f"<td class='num'>{r['documents_truncated']}</td></tr>"
        for r in perf.get("packing_by_policy", []))

    e1 = exps.get("E1", {})
    e1_rows = "".join(
        f"<tr><td>{esc(k)}</td>"
        f"<td class='num'>{v['indic_share_by_scheduler_lane_tag']:.2%}</td>"
        f"<td class='num'>{v['indic_share_by_actual_document_content']:.2%}</td>"
        f"<td><span class='pill {'ok' if v['auditable_from_ledger'] else 'bad'}'>"
        f"{'auditable' if v['auditable_from_ledger'] else 'NOT auditable'}</span></td></tr>"
        for k, v in sorted(e1.get("arms", {}).items()))

    n_ctrl = ctrls.get("n_controls", 0)
    n_ctrl_ok = ctrls.get("n_passed", 0)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ERA V5 · S6 data system · {esc(ev['run_id'])}</title>
<style>
:root{{--bg:#fbfcfe;--fg:#101828;--mut:#667085;--line:#e4e7ec;--card:#fff;
--c1:#0891b2;--c2:#7c3aed;--c3:#16a34a;--ok:#067647;--okb:#ecfdf3;--bad:#b42318;--badb:#fef3f2;
--warn:#b54708;--warnb:#fffaeb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;
--line:#21262d;--card:#161b22;--okb:#0b2919;--badb:#2d1214;--warnb:#2b1d05;
--ok:#3fb950;--bad:#f85149;--warn:#d29922}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Inter,sans-serif;padding:28px 20px 60px}}
.wrap{{max-width:1120px;margin:0 auto}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;margin:34px 0 10px}}
.sub{{color:var(--mut);font-size:13px;margin-bottom:22px}}
.grid{{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(168px,1fr))}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.card b{{display:block;font-size:22px;letter-spacing:-.02em}}
.card span{{color:var(--mut);font-size:12px}}
.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;min-width:520px}}
th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:middle}}
th{{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
tr:last-child td{{border-bottom:0}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
td.why{{color:var(--mut);font-size:12.5px}}
code{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut)}}
.pill{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11.5px;font-weight:600;
white-space:nowrap}}
.pill.ok{{background:var(--okb);color:var(--ok)}}
.pill.bad{{background:var(--badb);color:var(--bad)}}
.pill.warn{{background:var(--warnb);color:var(--warn)}}
.bar{{background:var(--line);border-radius:999px;height:7px;min-width:90px}}
.bar span{{display:block;height:7px;border-radius:999px}}
.chart{{width:100%;height:auto;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:6px}}
.line{{fill:none;stroke:var(--c1);stroke-width:2}}
.ax{{stroke:var(--line);stroke-width:1}}
.tick{{fill:var(--mut);font-size:10px}}
.note{{color:var(--mut);font-size:12.5px;margin:8px 0 0}}
</style></head><body><div class="wrap">

<h1>Training data execution system</h1>
<div class="sub">run <code>{esc(ev['run_id'])}</code> · plan <code>{esc(meta['plan_id'])}</code>
 · tokenizer <code>{esc(meta['tokenizer_hash'][:16])}</code> (vocab {meta['vocab_size']})
 · {meta['steps']} steps · global batch {meta['global_batch']} × seq {meta['sequence_length']}
 · rendered from <code>submission_artifacts/</code>, no live data</div>

<div class="grid">
 <div class="card"><b>{summ['n_passed']}/{summ['n_requirements']}</b><span>requirements pass</span></div>
 <div class="card"><b>{n_ctrl_ok}/{n_ctrl}</b><span>negative controls fire</span></div>
 <div class="card"><b>{perf.get('packing_utilization',0):.1%}</b><span>packing utilization</span></div>
 <div class="card"><b>{perf.get('useful_fraction_of_raw',0):.1%}</b><span>useful of raw tokens</span></div>
 <div class="card"><b>{osum.get('observed_rejection_rate',0):.1%}</b><span>OPUS rejection (target
   {osum.get('target_rejection_rate',0):.0%})</span></div>
 <div class="card"><b>{len(batches)}</b><span>batches committed</span></div>
</div>

<h2>Mean loss per optimizer step — branch run-a</h2>
{svg}
<p class="note">Branch <code>run-b</code> is an intentional fork from an early checkpoint and is
 charted separately in <code>ledgers/branches.json</code> — splicing it here would draw a curve that
 never happened. Initial loss should sit at <code>ln(vocab)</code> for an untrained model.
 ln({meta['vocab_size']}) = {__import__('math').log(meta['vocab_size']):.4f};
 measured first step {losses[0]:.4f} if present.</p>

<h2>Delivered lane shares — measured from the ledger, not the plan (branch run-a)</h2>
<div class="scroll"><table>
<thead><tr><th>lane</th><th class="num">delivered</th><th>share</th><th>protected floor</th></tr></thead>
<tbody>{lane_rows}</tbody></table></div>

<h2>Production feasibility at 3T — arithmetic only, no tokens trained</h2>
<div class="scroll"><table>
<thead><tr><th>lane</th><th class="num">demand</th><th class="num">supply</th>
<th class="num">epochs</th><th>verdict</th></tr></thead>
<tbody>{feas_rows}</tbody></table></div>
<p class="note">{esc(feas.get('demand_rule',''))} · ceiling {feas.get('epoch_ceiling')} epochs ·
 rejection {feas.get('rejection_rate',0):.0%} ·
 infeasible: <b>{esc(', '.join(feas.get('infeasible_lanes',[])) or 'none')}</b></p>

<h2>Perplexity by capability lane</h2>
<div class="scroll"><table>
<thead><tr><th>lane</th><th class="num">geo-mean ppl</th><th></th>
<th class="num">hot tokens</th><th class="num">observed</th></tr></thead>
<tbody>{ppl_rows}</tbody></table></div>

<h2>V6 corpus feedback — generated from the learning ledger</h2>
<div class="scroll"><table>
<thead><tr><th>shard</th><th>lane</th><th>recommendation</th><th>because</th>
<th class="num">best phase</th></tr></thead>
<tbody>{v6_rows}</tbody></table></div>

<h2>Packing policies (E3)</h2>
<div class="scroll"><table>
<thead><tr><th>policy</th><th class="num">seqs</th><th class="num">util</th><th></th>
<th class="num">loss-bearing</th><th class="num">truncated</th></tr></thead>
<tbody>{pack_rows}</tbody></table></div>

<h2>E1 — mixed vs pure lane shards (the session's open question)</h2>
<div class="scroll"><table>
<thead><tr><th>arm</th><th class="num">Indic by lane tag</th>
<th class="num">by document content</th><th>audit</th></tr></thead>
<tbody>{e1_rows}</tbody></table></div>
<p class="note">{esc(e1.get('finding',''))}</p>

<h2>Exactly-once across a changed world size</h2>
<p class="note">
 {esc(xo.get('run_a',{}).get('label','?'))} vs {esc(xo.get('run_b',{}).get('label','?'))} ·
 {xo.get('run_a',{}).get('samples',0)} positions ·
 order identical <b>{xo.get('order_identical')}</b> ·
 duplicate positions {xo.get('coverage_run_b',{}).get('duplicate_positions','?')} ·
 missing {xo.get('coverage_run_b',{}).get('missing_positions','?')}<br>{esc(xo.get('method',''))}</p>

<h2>Ledger volume — why the full tier is a proxy/debug tier</h2>
<p class="note">
 full (extrapolated) {vol.get('bytes_full_extrapolated',0):,} B ·
 stored [{esc(vol.get('tier_used','?'))}, 1-in-{vol.get('sample_every','?')}]
 {vol.get('bytes_stored_this_run',0):,} B ·
 aggregate {vol.get('bytes_aggregate_only',0):,} B ·
 {vol.get('bytes_per_token_full','?')} B/token →
 <b>{vol.get('extrapolation',{}).get('full_tier_petabytes_at_3T_tokens','?')} PB at 3T
 loss-bearing tokens</b></p>

<h2>Requirements</h2>
<div class="scroll"><table>
<thead><tr><th>requirement</th><th>result</th><th>evidence</th></tr></thead>
<tbody>{req_rows}</tbody></table></div>

<p class="note" style="margin-top:26px">
Static rendering of generated artifacts. Not a service, no network, no live data — the session
describes a four-server architecture (training, dashboard, data-ledger, benchmark) and explicitly
excludes building it from this assignment, so this shows what such a dashboard would serve.
Throughput figures are a CPU float64 toy: ratios transfer, absolute tokens/sec does not.</p>

</div></body></html>"""

    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({len(doc):,} bytes, self-contained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
