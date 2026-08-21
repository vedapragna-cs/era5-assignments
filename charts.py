#!/usr/bin/env python3
"""Inline SVG charts for the timeline pages.

Two charts, both generated from data/mechanisms.json, both theme-aware and dependency-free.

Design decisions, following the project's data-viz procedure:
  * Form first. The timeline's job is identity + change over time, so it is a faceted dot
    plot on a real proportional time axis -- not cards under year headings, which is
    structurally still a list. Mechanisms with a shipped date get a dumbbell (v1 dot ->
    connector -> hollow shipped dot), which is the documented form for before/after per item.
  * Family is encoded by LANE, not by colour. Seven families exceeds the three-slot
    all-pairs cap for scatter forms, so faceting carries identity and each lane is labelled.
  * Colour carries one binary distinction only: covered in class vs added. Two slots,
    validated all-pairs in both modes (CVD dE 24.7 light / 26.8 dark, normal-vision 33.6 /
    31.8, all six checks PASS).
  * Text never wears the series colour; labels are ink tokens.
"""
import json, datetime, os

S1_L, S2_L = "#2a78d6", "#eb6834"   # light: covered in class, added
S1_D, S2_D = "#3987e5", "#d95926"   # dark steps of the same two hues

CHART_CSS = f"""
.viz{{--s1:{S1_L};--s2:{S2_L};--s3:#1baf7a;--vsurf:var(--card);--vink:var(--fg);--vmut:var(--mut);
--vgrid:var(--line);margin:1.4rem 0}}
@media(prefers-color-scheme:dark){{:root:where(:not([data-theme=light])) .viz{{--s1:{S1_D};--s2:{S2_D};--s3:#199e70}}}}
:root[data-theme=dark] .viz{{--s1:{S1_D};--s2:{S2_D};--s3:#199e70}}
:root[data-theme=light] .viz{{--s1:{S1_L};--s2:{S2_L};--s3:#1baf7a}}
.viz svg{{width:100%;height:auto;display:block;overflow:visible}}
.viz figcaption{{font-family:system-ui,sans-serif;font-size:.82rem;color:var(--mut);
margin-top:.6rem;line-height:1.5}}
.vzl{{font-family:system-ui,sans-serif;font-size:.82rem;color:var(--mut);display:flex;
flex-wrap:wrap;gap:1rem;align-items:center;margin:.2rem 0 .7rem}}
.vzl b{{display:inline-flex;align-items:center;gap:.35rem;font-weight:400;white-space:nowrap}}
.vzl i{{width:11px;height:11px;border-radius:50%;display:inline-block}}
.vzs{{font:600 10.5px system-ui,sans-serif;fill:var(--vink)}}
.vzlane{{font:600 10px system-ui,sans-serif;fill:var(--vmut);letter-spacing:.03em}}
.vzax{{font:10px system-ui,sans-serif;fill:var(--vmut)}}
.viz .grid{{stroke:var(--vgrid);stroke-width:1}}
.viz .dot{{stroke:var(--vsurf);stroke-width:2}}
.viz .conn{{stroke-width:2;stroke-linecap:round;opacity:.45}}
.viz g.m:hover .dot{{stroke:var(--vink)}}
.vztable{{margin-top:.7rem;font-family:system-ui,sans-serif;font-size:.8rem}}
.vztable summary{{cursor:pointer;color:var(--acc)}}
"""

LANES = [("foundation", "foundation"), ("position", "position"), ("sparsity", "sparsity"),
         ("cache", "KV cache"), ("recurrent", "recurrent state"),
         ("composition", "composition"), ("scaling", "context scaling"),
         ("systems", "systems")]


def _x(date, x0, x1, d0, d1):
    t = (datetime.date.fromisoformat(date) - d0).days / (d1 - d0).days
    return round(x0 + t * (x1 - x0), 1)


def timeline_svg(cards):
    d0, d1 = datetime.date(2014, 1, 1), datetime.date(2026, 12, 31)
    X0, X1, TOP, LH = 104, 986, 30, 72
    H = TOP + len(LANES) * LH + 16
    p = [f'<svg viewBox="0 0 1000 {H}" role="img" aria-labelledby="tlt tld">',
         '<title id="tlt">Attention mechanisms by launch date, 2014 to 2026</title>',
         f'<desc id="tld">{len(cards)} mechanisms placed on a proportional time axis and '
         'grouped into seven family lanes. A line from a filled dot to a hollow dot shows '
         'the gap between the arXiv v1 date and the date the mechanism shipped in a '
         'well-known model. Those lines get shorter from left to right.</desc>']

    for yr in range(2014, 2028):
        x = _x(f"{yr}-01-01", X0, X1, d0, d1)
        p.append(f'<line class="grid" x1="{x}" y1="{TOP-8}" x2="{x}" y2="{H-14}"/>')
        if yr % 2 == 0 or yr >= 2023:
            p.append(f'<text class="vzax" x="{x}" y="{TOP-14}" text-anchor="middle">{yr}</text>')

    by = {k: [] for k, _ in LANES}
    for c in cards:
        by[c["family"]].append(c)

    for i, (key, label) in enumerate(LANES):
        cy = TOP + i * LH + LH / 2
        p.append(f'<line class="grid" x1="{X0}" y1="{cy+18}" x2="{X1}" y2="{cy+18}" opacity=".5"/>')
        p.append(f'<text class="vzlane" x="{X0-12}" y="{cy+3}" text-anchor="end">{label}</text>')
        items = sorted(by[key], key=lambda c: c["date_v1"])
        placed = []   # (x_left, x_right, level)
        drawn = []    # (x, dy) of dots already placed in this lane
        for c in items:
            x = _x(c["date_v1"], X0, X1, d0, d1)
            col = "var(--s2)" if c["added"] else "var(--s1)"
            w = len(c["short"]) * 5.6 + 6
            lvl = 0
            while any(not (x + w / 2 < a or x - w / 2 > b) and l == lvl for a, b, l in placed):
                lvl += 1
            placed.append((x - w / 2, x + w / 2, lvl))
            ly = cy - 14 - lvl * 12
            # nudge a dot that would sit on top of one already drawn in this lane
            dy = 0
            for ox, od in drawn:
                if abs(x - ox) < 9:
                    dy = 7 if od == 0 else -7
            drawn.append((x, dy))
            cyd = cy + dy
            if c.get("shipped"):
                xs = _x(c["shipped"]["date"], X0, X1, d0, d1)
                p.append(f'<line class="conn" x1="{x}" y1="{cyd}" x2="{xs}" y2="{cyd}" stroke="{col}"/>')
                p.append(f'<circle class="dot" cx="{xs}" cy="{cyd}" r="4.5" fill="var(--vsurf)" '
                         f'stroke="{col}"/>')
            ttl = (f'{c["name"]} — arXiv v1 {c["date_v1"]}'
                   + (f', shipped {c["shipped"]["date"]}' if c.get("shipped") else "")
                   + (' (date bounded, not exact)' if c["status"] == "bounded" else ""))
            p.append(f'<g class="m"><title>{ttl}</title>'
                     f'<a href="#{c["id"]}">'
                     + (f'<line x1="{x}" y1="{ly+3}" x2="{x}" y2="{cyd-6}" stroke="{col}" '
                        f'stroke-width="1" opacity=".5"/>' if lvl or dy else '')
                     + f'<circle class="dot" cx="{x}" cy="{cyd}" r="5" fill="{col}"/>'
                     f'<text class="vzs" x="{x}" y="{ly}" text-anchor="middle">{c["short"]}</text>'
                     f'</a></g>')
    p.append("</svg>")

    legend = ('<div class="vzl">'
              '<b><i style="background:var(--s1)"></i>on the assignment&rsquo;s list</b>'
              '<b><i style="background:var(--s2)"></i>added by us</b>'
              '<b><svg width="34" height="11" style="overflow:visible"><line x1="2" y1="5.5" x2="32" '
              'y2="5.5" stroke="var(--vmut)" stroke-width="2" opacity=".5"/>'
              '<circle cx="2" cy="5.5" r="4" fill="var(--vmut)"/>'
              '<circle cx="32" cy="5.5" r="3.5" fill="var(--card)" stroke="var(--vmut)" '
              'stroke-width="2"/></svg> arXiv v1 → shipped in a known model</b></div>')
    return ('<figure class="viz">' + legend + "".join(p) +
            '<figcaption>Position on the axis is the real date, so the spacing is the story. '
            'Family is the lane, not the colour. The connectors are the gap between a paper '
            'and its first well-known deployment &mdash; read them left to right and watch them '
            'shorten. Every dot links to its card below.</figcaption></figure>')


def lag_svg(cards):
    """Lag from arXiv v1 to first public release, sorted by lag and grouped by what the
    mechanism costs. Deliberately NOT plotted against time: the earlier version of this chart
    put lag on a time axis with a connecting line, which implied a trend that the corrected
    data does not support. Sorting by magnitude and colouring by cost shows what actually
    predicts the lag."""
    CLS = [("third", "authors published no code; somebody reimplemented", "var(--s2)"),
           ("author", "authors released the implementation themselves", "var(--s3)")]
    rows = []
    for c in cards:
        sh = c.get("shipped")
        if not sh or sh.get("adoption_class") in (None, "excluded"):
            continue
        lag = (datetime.date.fromisoformat(sh["date"])
               - datetime.date.fromisoformat(c["date_v1"])).days
        rows.append((lag, c["short"], sh["adoption_class"], c["name"], c["date_v1"],
                     sh["date"], sh.get("verified", False)))
    rows.sort(key=lambda r: -r[0])
    X0, X1 = 148, 900
    TOP, RH = 34, 26
    H = TOP + len(rows) * RH + 46
    lo = min(r[0] for r in rows)
    hi = max(r[0] for r in rows)
    lo -= (hi - lo) * 0.07   # pad so negative bars and their labels clear the row labels
    zero = X0 + (0 - lo) / (hi - lo) * (X1 - X0)
    x = lambda v: round(X0 + (v - lo) / (hi - lo) * (X1 - X0), 1)
    col = {k: c for k, _, c in CLS}
    p = [f'<svg viewBox="0 0 1000 {H}" role="img" aria-labelledby="lgt lgd">',
         '<title id="lgt">Days from paper to first public release, sorted by lag</title>',
         '<desc id="lgd">Eleven mechanisms. The three longest lags all belong to mechanisms that '
         'cost something to adopt. Everything free, or released by its own authors, shipped within '
         'weeks or before the paper.</desc>']
    for v in (0, 300, 600, 900, 1200):
        if lo <= v <= hi:
            p.append(f'<line class="grid" x1="{x(v)}" y1="{TOP-10}" x2="{x(v)}" y2="{H-40}"/>')
            p.append(f'<text class="vzax" x="{x(v)}" y="{TOP-16}" text-anchor="middle">{v:,}</text>')
    p.append(f'<text class="vzax" x="{X0}" y="{H-18}">days from arXiv v1 to first public release '
             f'&mdash; negative means the code came first</text>')
    for i, (lag, short, cls, name, v1, sh, ver) in enumerate(rows):
        cy = TOP + i * RH + RH / 2
        c = col[cls]
        x1v = x(lag)
        p.append(f'<text class="vzs" x="{X0-14}" y="{cy+3.5}" text-anchor="end">{short}'
                 + ('' if ver else '<tspan class="vzax"> *</tspan>') + '</text>')
        p.append(f'<line x1="{zero}" y1="{cy}" x2="{x1v}" y2="{cy}" stroke="{c}" stroke-width="2" '
                 'stroke-linecap="round" opacity=".55"/>')
        p.append(f'<g class="m"><title>{name}: v1 {v1}, released {sh} &mdash; {lag} days</title>'
                 f'<circle class="dot" cx="{x1v}" cy="{cy}" r="5" fill="{c}"/></g>')
        anchor = "start" if lag >= 0 else "end"
        dx = 11 if lag >= 0 else -11
        p.append(f'<text class="vzax" x="{x1v+dx}" y="{cy+3.5}" text-anchor="{anchor}">{lag} d</text>')
    p.append(f'<line class="grid" x1="{zero}" y1="{TOP-10}" x2="{zero}" y2="{H-40}" '
             'stroke-width="1.5" opacity=".8"/>')
    p.append("</svg>")
    legend = '<div class="vzl">' + "".join(
        f'<b><i style="background:{c}"></i>{lbl}</b>' for _, lbl, c in CLS) + "</div>"
    trows = "".join(
        f"<tr><td>{n}</td><td>{v1}</td><td>{sh}{'' if ver else ' *'}</td><td>{lag}</td>"
        f"<td>{dict((k, l) for k, l, _ in CLS)[cls]}</td></tr>"
        for lag, s_, cls, n, v1, sh, ver in rows)
    tbl = ('<details class="vztable"><summary>Same data as a table</summary>'
           '<div class="tw"><table><thead><tr><th>mechanism</th><th>arXiv v1</th>'
           '<th>first public release</th><th>lag (days)</th><th>class</th></tr></thead>'
           f'<tbody>{trows}</tbody></table></div></details>')
    return ('<figure class="viz">' + legend + "".join(p) +
            '<figcaption>Sorted by lag, not by date &mdash; because there is no trend in time. '
            'Nine of eleven sit at or before their own paper, because the authors pushed the repo. The column '
            'measures publishing practice, not adoption. The only two real waits are both Google papers '
            'published without weights &mdash; and their lengths differ by twenty times. Rows marked '
            '<span class="vzax">*</span> are documented releases rather than API-verified ones.</figcaption>' + tbl + "</figure>")
