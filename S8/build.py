#!/usr/bin/env python3
"""Generate the static site from data/*.json.

Fail-closed by design. The build refuses to emit a page if any mechanism card is missing
one of the six required fields, and in particular if `gives_up` is empty -- the assignment
states that a technique written up with no meaningful downside has not been understood, so
a cost-free card is a build error here, not a style note.

    python3 build.py            # validate, then write index.html findings.html corrections.html
    python3 build.py --check    # validate only, emit nothing
"""
import json, os, sys, re, html, datetime, collections
import charts

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda n: os.path.join(HERE, "data", n)
REPO_URL = "https://github.com/vedapragna-cs/era-v5-s8-attention-timeline"
REQUIRED = ["problem", "mechanism", "buys", "gives_up", "choose_when"]
FAMILIES = {"foundation": "foundation", "position": "position", "sparsity": "sparsity",
            "cache": "KV cache", "recurrent": "recurrent state",
            "composition": "composition", "scaling": "context scaling", "systems": "systems"}

# ---------------------------------------------------------------- validation

def validate(cards):
    errs = []
    seen = set()
    for c in cards:
        cid = c.get("id", "<no id>")
        if cid in seen:
            errs.append(f"{cid}: duplicate id")
        seen.add(cid)
        for f in REQUIRED:
            if not (c.get(f) or "").strip():
                errs.append(f"{cid}: '{f}' is empty -- every mechanism is a trade and must state its cost")
        if not c.get("date_v1"):
            errs.append(f"{cid}: no date")
        if not c.get("source_url"):
            errs.append(f"{cid}: no primary source URL")
        if c.get("status") not in ("verified", "bounded"):
            errs.append(f"{cid}: status must be 'verified' or 'bounded', got {c.get('status')!r}")
        if c.get("status") == "verified" and not c.get("arxiv"):
            errs.append(f"{cid}: claims verified with no arXiv id")
    covered = {c["list_item"] for c in cards if c.get("list_item")}
    missing = sorted(set(range(1, 18)) - covered)
    if missing:
        errs.append(f"required list items not covered: {missing}")
    return errs, len(covered)

# ---------------------------------------------------------------- rendering

MIN_LIST = [
    (1, "standard attention"), (2, "absolute learned positions"), (3, "sinusoidal"),
    (4, "RoPE"), (5, "ALiBi"), (6, "MQA"), (7, "GQA"), (8, "sliding window"),
    (9, "attention sinks"), (10, "NTK-aware scaling"), (11, "YaRN"), (12, "linear attention"),
    (13, "the delta rule and Gated DeltaNet"), (14, "MLA"), (15, "sparse and top-k attention"),
    (16, "DeepSeek's compressed sparse attention"), (17, "DroPE"),
]


def coverage_section(cards):
    by = {}
    for c in cards:
        if c.get("list_item"):
            by.setdefault(c["list_item"], []).append(c)
    rows = []
    for n, label in MIN_LIST:
        got = by.get(n, [])
        links = ", ".join(f'<a href="#{c["id"]}">{esc(c["name"])}</a> '
                          f'<span style="color:var(--mut)">{c["date_v1"]}</span>' for c in got)
        rows.append(f"<tr><td>{n}</td><td>{esc(label)}</td><td>{links}</td></tr>")
    extra = [c for c in cards if c["added"]]
    taught = [c for c in extra if c.get("taught_not_listed")]
    return f"""<h2>Coverage, stated as a count</h2>
<p>The assignment names <strong>17</strong> mechanisms as the minimum. All 17 are here, mapped to the
card that covers them. Three list items are covered by more than one card, because the item names a
lineage rather than a single paper &mdash; item 13 is three papers and item 15 is two.</p>
<div class="tw"><table><thead><tr><th>#</th><th>required</th><th>covered by</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p><strong>{len(MIN_LIST)} of {len(MIN_LIST)} required items covered</strong>, across
{sum(len(v) for v in by.values())} cards, plus <strong>{len(extra)}</strong> we added for
{len(cards)} in total. {len(taught)} of the additions are mechanisms the session
<em>taught</em> that the minimum list does not name &mdash; see
<a href="corrections.html">corrections</a>.</p>"""


def esc(s):
    return html.escape(s or "", quote=False)

def md_inline(s):
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s

def md_block(s):
    """Minimal markdown: paragraphs, and pipe tables."""
    out, buf, tbl = [], [], []
    def flush_p():
        if buf:
            out.append("<p>" + md_inline(" ".join(buf)) + "</p>")
            buf.clear()
    def flush_t():
        if tbl:
            head, *rest = tbl
            rest = [r for r in rest if not re.match(r"^[\s|:-]+$", r)]
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
            out.append("<div class='tw'><table><thead><tr>"
                       + "".join(f"<th>{md_inline(c)}</th>" for c in cells(head))
                       + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in cells(r))
                                 + "</tr>" for r in rest)
                       + "</tbody></table></div>")
            tbl.clear()
    for line in (s or "").split("\n"):
        if line.strip().startswith("|"):
            flush_p(); tbl.append(line)
        elif not line.strip():
            flush_p(); flush_t()
        else:
            flush_t(); buf.append(line.strip())
    flush_p(); flush_t()
    return "\n".join(out)

CSS = """
:root{--bg:#fbfaf8;--fg:#1c1a17;--mut:#6b655c;--line:#e2ddd4;--card:#fff;--acc:#8a4b2a;
--warn:#8a2a2a;--ok:#2a6b46;--code:#f2efe9;--shadow:0 1px 2px rgba(0,0,0,.05)}
@media(prefers-color-scheme:dark){:root{--bg:#141311;--fg:#eae6df;--mut:#9b948a;--line:#2e2b27;
--card:#1c1a18;--acc:#d99a6c;--warn:#e08585;--ok:#7fc9a0;--code:#242120;--shadow:none}}
:root[data-theme=dark]{--bg:#141311;--fg:#eae6df;--mut:#9b948a;--line:#2e2b27;--card:#1c1a18;
--acc:#d99a6c;--warn:#e08585;--ok:#7fc9a0;--code:#242120;--shadow:none}
:root[data-theme=light]{--bg:#fbfaf8;--fg:#1c1a17;--mut:#6b655c;--line:#e2ddd4;--card:#fff;
--acc:#8a4b2a;--warn:#8a2a2a;--ok:#2a6b46;--code:#f2efe9;--shadow:0 1px 2px rgba(0,0,0,.05)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.62 ui-serif,Georgia,'Iowan Old Style',serif;-webkit-text-size-adjust:100%}
.wrap{max-width:52rem;margin:0 auto;padding:2rem 1.15rem 6rem}
h1{font-size:2rem;line-height:1.15;margin:.2em 0 .3em;letter-spacing:-.015em}
h2{font-size:1.3rem;margin:2.6em 0 .5em;letter-spacing:-.01em}
h3{font-size:1.05rem;margin:1.8em 0 .3em}
p{margin:.7em 0}
a{color:var(--acc)}
code{font:.87em ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code);
padding:.1em .32em;border-radius:3px}
pre{background:var(--code);padding:.85rem 1rem;border-radius:6px;overflow-x:auto;
font:.83rem/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.sub{color:var(--mut);font-size:1.02rem;margin-top:0}
nav.top{display:flex;flex-wrap:wrap;gap:.5rem;margin:1.4rem 0 0;font-size:.87rem;
font-family:system-ui,sans-serif}
nav.top a{display:inline-block;padding:.32rem .7rem;border:1px solid var(--line);
border-radius:99px;text-decoration:none;background:var(--card)}
nav.top a[aria-current]{border-color:var(--acc);color:var(--acc);font-weight:600}
.tw{overflow-x:auto;margin:1rem 0}
table{border-collapse:collapse;width:100%;font-size:.87rem;
font-family:system-ui,sans-serif}
th,td{text-align:left;padding:.42rem .6rem;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--mut);font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}
.flow{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:1.1rem;
margin:1.2rem 0;box-shadow:var(--shadow)}
.flow pre{background:transparent;padding:0;font-size:.86rem;text-align:center}
.era{font-family:system-ui,sans-serif;font-size:.78rem;letter-spacing:.14em;color:var(--mut);
text-transform:uppercase;margin:3rem 0 .8rem;padding-bottom:.3rem;border-bottom:1px solid var(--line)}
.card{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--line);
border-radius:7px;padding:1.1rem 1.2rem;margin:.9rem 0;box-shadow:var(--shadow)}
.card.added{border-left-color:var(--acc)}
.card h3{margin:0 0 .15rem;font-size:1.12rem}
.meta{font-family:system-ui,sans-serif;font-size:.79rem;color:var(--mut);
display:flex;flex-wrap:wrap;gap:.45rem;align-items:center;margin-bottom:.75rem}
.date{font-weight:700;color:var(--fg);font-variant-numeric:tabular-nums}
.tag{border:1px solid var(--line);border-radius:99px;padding:.05rem .5rem}
.tag.add{border-color:var(--acc);color:var(--acc)}
.tag.bounded{border-color:var(--warn);color:var(--warn)}
.tag.ship{border-color:var(--ok);color:var(--ok)}
.qa{margin:.55rem 0 0}
.qa dt{font-family:system-ui,sans-serif;font-size:.72rem;text-transform:uppercase;
letter-spacing:.07em;color:var(--mut);margin-top:.6rem}
.qa dd{margin:.1rem 0 0}
.qa dd.cost{color:var(--fg)}
.src{font-family:system-ui,sans-serif;font-size:.76rem;color:var(--mut);margin-top:.8rem;
padding-top:.6rem;border-top:1px dotted var(--line);overflow-wrap:anywhere}
.note{font-size:.9rem;background:var(--code);border-radius:5px;padding:.6rem .8rem;margin-top:.8rem}
.note b{color:var(--acc)}
.f{border:1px solid var(--line);border-radius:7px;padding:1.1rem 1.2rem;margin:1.1rem 0;
background:var(--card);box-shadow:var(--shadow)}
.f h3{margin-top:0}
.kind{font-family:system-ui,sans-serif;font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;
color:var(--mut)}
.caveat{font-size:.87rem;color:var(--mut);border-top:1px dotted var(--line);margin-top:.8rem;
padding-top:.55rem}
.sev{display:inline-block;font-family:system-ui,sans-serif;font-size:.7rem;letter-spacing:.06em;
text-transform:uppercase;padding:.12rem .5rem;border-radius:99px;border:1px solid var(--line)}
.sev.correction{border-color:var(--warn);color:var(--warn)}
.sev.negative-control{border-color:var(--ok);color:var(--ok)}
footer{margin-top:4rem;padding-top:1.2rem;border-top:1px solid var(--line);
font-family:system-ui,sans-serif;font-size:.82rem;color:var(--mut)}
#ctl{display:none}
""" + charts.CHART_CSS

JS = """
(function(){
 var t=document.getElementById('ctl'); if(t) t.style.display='flex';
 var cards=[].slice.call(document.querySelectorAll('.card'));
 var eras=[].slice.call(document.querySelectorAll('.era'));
 function apply(f){
   cards.forEach(function(c){
     c.style.display=(f==='all'||c.dataset.family===f||(f==='added'&&c.dataset.added==='1'))?'':'none';});
   eras.forEach(function(h){
     var n=0,e=h.nextElementSibling;
     while(e&&!e.classList.contains('era')){if(e.classList.contains('card')&&e.style.display!=='none')n++;e=e.nextElementSibling;}
     h.style.display=n?'':'none';});
 }
 var sel=document.getElementById('fam');
 if(sel) sel.addEventListener('change',function(){apply(this.value);});
})();
"""

def page(title, desc, body, active):
    nav = [("index.html", "Timeline"), ("findings.html", "What the timeline shows"),
           ("corrections.html", "Corrections"), ("sources.html", "Sources")]
    links = "".join(f'<a href="{h}"{" aria-current=page" if h==active else ""}>{esc(t)}</a>'
                    for h, t in nav)
    links += (f'<a class="repo" href="{REPO_URL}" rel="noopener">Code &amp; data on GitHub</a>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{html.escape(desc, quote=True)}">
<style>{CSS}</style></head>
<body><div class="wrap">
<nav class="top">{links}</nav>
{body}
<footer>ERA V5 &middot; Session 8, Modern Attention Variants. Every date on this site is
re-checkable: run <code>python3 verify_dates.py</code> to re-fetch each arXiv v1 date from the
arXiv API and diff it against the stored value, and <code>python3 analyse.py</code> to recompute
every number on the findings page. Built {datetime.date.today().isoformat()}.</footer>
</div><script>{JS}</script></body></html>
"""

def render_card(c):
    cls = "card added" if c["added"] else "card"
    bits = [f'<span class="date">{c["date_v1"]}</span>']
    if c["status"] == "bounded":
        bits.append('<span class="tag bounded">date bounded, not exact</span>')
    bits.append(f'<span class="tag">{esc(FAMILIES.get(c["family"], c["family"]))}</span>')
    if c["added"]:
        bits.append('<span class="tag add">added &mdash; not on the required list</span>')
    if c.get("shipped"):
        bits.append(f'<span class="tag ship">shipped {c["shipped"]["date"]}</span>')
    ants = ""
    if c.get("antecedents"):
        ants = ("<dt>answers / builds on</dt><dd>"
                + "; ".join(f'{md_inline(a["name"])} <span style="color:var(--mut)">({a["date"]})</span>'
                            for a in c["antecedents"]) + "</dd>")
    nums = ""
    if c.get("numbers"):
        nums = "<dt>numbers</dt>" + "".join(
            f'<dd>{md_inline(n["value"])} &mdash; <em>{esc(n["label"])}</em>, {md_inline(n["source"])}</dd>'
            for n in c["numbers"])
    ship = ""
    if c.get("shipped"):
        ship = (f'<br>Shipped: {esc(c["shipped"]["what"])} &mdash; '
                f'<a href="{esc(c["shipped"]["url"])}">link</a>')
    note = f'<div class="note"><b>Why this card is here.</b> {md_inline(c["note"])}</div>' if c.get("note") else ""
    return f"""<article class="{cls}" data-family="{c['family']}" data-added="{1 if c['added'] else 0}" id="{c['id']}">
<h3>{esc(c['name'])}</h3>
<div class="meta">{''.join(bits)}</div>
<dl class="qa">
<dt>the problem at that moment</dt><dd>{md_inline(c['problem'])}</dd>
<dt>mechanism</dt><dd>{md_inline(c['mechanism'])}</dd>
<dt>what it buys</dt><dd>{md_inline(c['buys'])}</dd>
<dt>what it gives up</dt><dd class="cost">{md_inline(c['gives_up'])}</dd>
<dt>when you would actually pick it</dt><dd>{md_inline(c['choose_when'])}</dd>
{ants}{nums}</dl>
{note}
<div class="src">Date source: {md_inline(c['source_kind'])}<br>
<a href="{esc(c['source_url'])}">{esc(c['source_url'])}</a>{ship}</div>
</article>"""

def build():
    cards = json.load(open(D("mechanisms.json")))
    errs, ncov = validate(cards)
    if errs:
        print("BUILD FAILED -- %d validation error(s):" % len(errs))
        for e in errs:
            print("  x", e)
        return 1
    print(f"validation passed: {len(cards)} cards, {ncov}/17 required list items, "
          f"0 cost-free cards")
    if "--check" in sys.argv:
        return 0

    cards.sort(key=lambda c: (c["date_v1"], c.get("same_day_order", 50), c["id"]))
    ERAS = [(1900, 2016, "2014-2016 &mdash; attention before the Transformer"),
            (2017, 2018, "2017 &mdash; the architecture, and the bill it created"),
            (2019, 2020, "2019-2020 &mdash; the first wave: stop computing every pair"),
            (2021, 2022, "2021-2022 &mdash; position becomes the problem; exactness gets cheap"),
            (2023, 2023, "2023 &mdash; the year the bills came due"),
            (2024, 2025, "2024-2025 &mdash; state returns, sparsity returns rebuilt"),
            (2026, 2100, "2026 &mdash; described in class, not yet in the literature")]
    body = []
    rendered = []
    for lo, hi, label in ERAS:
        grp = [c for c in cards if lo <= int(c["date_v1"][:4]) <= hi]
        if not grp:
            continue
        body.append(f'<h2 class="era">{label}</h2>')
        body += [render_card(c) for c in grp]
        rendered += [c["id"] for c in grp]

    dropped = [c["id"] for c in cards if c["id"] not in set(rendered)]
    if dropped:
        print("BUILD FAILED -- validated cards that no era bucket renders: " + ", ".join(dropped))
        print("  (a card outside every ERAS range is silently invisible; extend ERAS)")
        return 1
    nadd = sum(1 for c in cards if c["added"])
    ntl = sum(1 for c in cards if c.get("taught_not_listed"))
    nnp = sum(1 for c in cards if not c.get("arxiv"))
    nnp_other = nnp - 1
    intro = f"""<h1>How attention works now</h1>
<p class="sub">Every attention mechanism from ERA V5 Session 8, in the order it was launched &mdash;
not the order it was taught, and not grouped by family. {len(cards)} mechanisms, covering all 17
items on the required list plus {nadd} we added &mdash; {ntl} of which the session taught but the list omits. Each one is an answer to a
problem that existed on its date, with what it buys, what it costs, and when you would pick it.</p>

<h2>Start here: what everything else modifies</h2>
<p>Nothing below makes sense without this. Scaled dot-product attention takes a query for the
current token, scores it against the key of every token it is allowed to see, turns those scores
into weights that sum to one, and returns the weighted sum of the values.</p>
<div class="flow"><pre>Q &times; K&#8868;
   &darr;
 scores        one number per (query, key) pair &mdash; this matrix is the compute bill
   &darr;
 scale         divide by &radic;d&#8342;, or softmax saturates at high dimension
   &darr;
 mask          a token may not attend to its own future
   &darr;
 softmax       scores become weights that sum to 1
   &darr;
 &sum; wV         weighted sum of values &mdash; the output</pre></div>
<p>It sends two separate bills, and the whole timeline is people paying one of them down.
<strong>Compute</strong> is the score matrix: <code>n&times;n</code> per head per layer, so doubling
the context quadruples it. <strong>Memory</strong> is the KV cache at decode time: every token you
have already generated leaves its keys and values behind, reloaded for every new token, growing
linearly with length <em>and</em> with the number of concurrent users. Sparsity and linear attention
attack the first bill. MQA, GQA and MLA attack the second. Position encoding is a third problem
entirely &mdash; and, as the timeline shows, it arrived on the same day as the architecture.</p>

<h2>How to read the dates</h2>
<p>Sorted by <strong>arXiv v1 submission date</strong> &mdash; when the idea became public, not when
it was published in a venue and not when it became popular. Where a mechanism shipped in a
well-known model at a materially different time, that <span class="tag ship">shipped</span> date is
shown too. The gap between the two turns out to be the most interesting column on the page; see
<a href="findings.html">what the timeline shows</a>. One mechanism &mdash; NTK-aware scaling &mdash;
has no paper at all and its exact day could not be verified; it and {nnp_other} others are marked
<span class="tag bounded">date bounded, not exact</span> rather than guessed.</p>
<p>The assignment predicts what this timeline will show &mdash; &ldquo;exact global attention,
cheaper decoding memory, better position handling, longer contexts, recurrent state returning,
sparsity returning, compression becoming more aggressive.&rdquo; We treated that as a testable claim
and checked it against the dates. Most of it holds in order; <a href="findings.html#Q2-9">one pair
is inverted by 25 months</a>.</p>
{{COVERAGE}}
{{TIMELINE_CHART}}
<div id="ctl" style="gap:.6rem;align-items:center;font-family:system-ui,sans-serif;font-size:.85rem;margin:1.2rem 0">
<label for="fam">Filter:</label>
<select id="fam"><option value="all">all {len(cards)} mechanisms</option>
<option value="added">the {nadd} we added</option>
{''.join(f'<option value="{k}">{esc(v)}</option>' for k, v in FAMILIES.items())}</select>
<span style="color:var(--mut)">Order stays chronological.</span></div>"""

    open(os.path.join(HERE, "index.html"), "w").write(
        page("How attention works now - a chronological timeline",
             "Every attention mechanism from standard scaled dot-product to DroPE, in launch order, "
             "with honest trade-offs and a primary source for every date.",
             intro.replace("{COVERAGE}", coverage_section(cards)).replace("{TIMELINE_CHART}", charts.timeline_svg(cards)) + "\n".join(body), "index.html"))

    # ---- findings page (Question 2)
    f = json.load(open(D("findings.json")))
    fb = [f'<h1>What the timeline shows</h1><p class="sub">{md_inline(f["intro"])}</p>']
    for x in f["findings"]:
        cav = f'<div class="caveat"><strong>Caveat.</strong> {md_inline(x["caveat"])}</div>' if x.get("caveat") else ""
        fig = charts.lag_svg(cards) if x["id"] == "Q2-1" else ""
        fb.append(f'<div class="f" id="{esc(x["id"])}"><div class="kind">{esc(x["id"])} &middot; {esc(x["kind"])}</div>'
                  f'<h3>{md_inline(x["headline"])}</h3>{fig}{md_block(x["body"])}{cav}</div>')
    u = f["uncovered"]
    fb.append(f'<h2>Mechanisms not covered in class</h2><p>{md_inline(u["intro"])}</p><div class="tw"><table>'
              '<thead><tr><th>mechanism</th><th>v1 date</th><th>source</th><th>why it belongs</th></tr></thead><tbody>'
              + "".join(f'<tr><td><strong>{esc(i["name"])}</strong></td><td>{i["date"]}</td>'
                        f'<td><a href="{esc(i["url"])}">arXiv {esc(i["arxiv"])}</a></td>'
                        f'<td>{md_inline(i["why"])}</td></tr>' for i in u["items"])
              + "</tbody></table></div>")
    open(os.path.join(HERE, "findings.html"), "w").write(
        page("What the timeline shows - ERA V5 Session 8",
             "Question 2: seven things visible in date order that a list of attention mechanisms hides.",
             "\n".join(fb), "findings.html"))

    # ---- sources page (the date table as HTML, so the nav link is not a .md download)
    srows = []
    for c in sorted(cards, key=lambda z: (z["date_v1"], z["id"])):
        src = (f'<a href="{esc(c["source_url"])}">arXiv {c["arxiv"]}</a>' if c.get("arxiv")
               else f'<a href="{esc(c["source_url"])}">no paper &mdash; see note</a>')
        st = "verified" if c["status"] == "verified" else "<strong>bounded</strong>"
        sh = c["shipped"]["date"] + ("" if c["shipped"].get("verified", True) else " *") if c.get("shipped") else "&mdash;"
        srows.append(f'<tr><td>{c["date_v1"]}</td><td><a href="index.html#{c["id"]}">{esc(c["name"])}</a></td>'
                     f'<td>{sh}</td><td>{src}</td><td>{st}</td></tr>')
    ver = json.load(open(D("date-verification.json")))
    sb = f'''<h1>Where every date comes from</h1>
<p class="sub">The assignment says the chronology is the part that is easiest to get wrong and easiest
to check. So here is all of it, checkable.</p>
<h2>The convention</h2>
<p>The sort key is the <strong>arXiv v1 submission date</strong> &mdash; the day the idea became
public, not the conference publication date and not the day it became popular. Two mechanisms that
share a paper share a date, and that is not an error: standard attention and sinusoidal encoding are
both <em>Attention Is All You Need</em>, 2017-06-12.</p>
<p><strong>Shipped</strong> means one thing throughout: the first public release of weights or
reference code implementing the mechanism, read from the GitHub or HuggingFace API where possible.
Rows marked <strong>*</strong> are documented releases we could not read from an API. An earlier
version of this site mixed third-party adoption with authors' own releases in this column; that was
wrong and is <a href="findings.html#Q2-1">documented as a retraction</a>.</p>
<h2>How to re-run the check yourself</h2>
<pre>python3 verify_dates.py     # re-fetch every arXiv v1 date, diff, exit 1 on drift
python3 analyse.py          # recompute every number on the findings page
python3 build.py            # validate all cards, regenerate these pages</pre>
<p>Last run: <strong>{ver['arxiv_dates_matched']} of {ver['cards']} dates re-verified against the
arXiv API, {ver['no_paper_by_design']} with no paper by design, {ver['failures']} failures</strong>
({ver['checked_at']}). The raw API response is committed at <code>data/arxiv-response.xml</code> so
the check also runs offline.</p>
<h2>Every date, with its source</h2>
<div class="tw"><table><thead><tr><th>v1 date</th><th>mechanism</th><th>shipped</th>
<th>primary source</th><th>status</th></tr></thead><tbody>{''.join(srows)}</tbody></table></div>
<p style="color:var(--mut);font-size:.9rem">Three dates carry no paper at all: NTK-aware scaling
(a Reddit post, bounded by GitHub API timestamps), DeepSeek's low-rank indexer (a model release), and
the V4 Memory Stream (described only in the course material). Each says so on its card rather than
being given a plausible-looking date.</p>'''
    open(os.path.join(HERE, "sources.html"), "w").write(
        page("Where every date comes from - ERA V5 Session 8",
             "The full chronology with a primary source for every date, and the script to re-check them.",
             sb, "sources.html"))

    # ---- corrections page
    cs = json.load(open(D("corrections.json")))
    cb = ['<h1>Corrections and limitations</h1><p class="sub">The assignment asks us to check every '
          'date against a primary source, and to say so if the course material got something wrong. '
          'It also warns that our own agent will invent dates confidently. Both directions are '
          'recorded here, including one case where we went looking for an error and found the '
          'material was right.</p>']
    for x in cs:
        cb.append(f'<div class="f"><span class="sev {esc(x["severity"])}">{esc(x["severity"])}</span> '
                  f'<span class="kind">{esc(x["id"])}</span><h3>{md_inline(x["title"])}</h3>'
                  f'<dl class="qa"><dt>claim, and where it comes from</dt><dd>{md_inline(x["claim_source"])}</dd>'
                  f'<dt>what we checked</dt><dd>{md_inline(x["what_we_checked"])}</dd>'
                  f'<dt>finding</dt><dd>{md_inline(x["finding"])}</dd>'
                  f'<dt>what we did about it</dt><dd>{md_inline(x["resolution"])}</dd>'
                  + ("<dt>primary sources</dt><dd>" + "<br>".join(f'<a href="{esc(p)}">{esc(p)}</a>'
                     for p in x["primary"]) + "</dd>" if x.get("primary") else "")
                  + "</dl></div>")
    open(os.path.join(HERE, "corrections.html"), "w").write(
        page("Corrections and limitations - ERA V5 Session 8",
             "Where the course material, our sources, and our own first answers were wrong -- "
             "and one case where they were right.",
             "\n".join(cb), "corrections.html"))

    for n in ("index.html", "findings.html", "corrections.html", "sources.html"):
        print(f"  wrote {n}  ({os.path.getsize(os.path.join(HERE, n)):,} bytes)")
    return 0

if __name__ == "__main__":
    sys.exit(build())
