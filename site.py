"""Generate the interactive page from the artefacts.

Everything shown is read from `results/evidence.json` and `results/analysis.json`
at build time. No number is typed in by hand, so the page cannot drift from the
run that produced it -- if a claim is not in the artefacts, it does not appear.

    python site.py          # -> docs/index.html
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
DOCS = os.path.join(ROOT, "docs")


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


PAGE = r"""<title>Invertible Kronecker Embeddings</title>
<style>
:root{
  --bg:#fbfaf8; --panel:#ffffff; --ink:#1a1a1a; --muted:#5d5d5d;
  --line:#e2ded7; --accent:#8c4a2f; --accent-soft:#f3e6df;
  --good:#2f6b4f; --bad:#a33a2a; --code:#f5f2ee;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#141312; --panel:#1c1b1a; --ink:#ece8e3; --muted:#a29c95;
    --line:#302e2b; --accent:#e0906b; --accent-soft:#2e211a;
    --good:#7fbf9a; --bad:#e0806b; --code:#232120;
  }
}
:root[data-theme="dark"]{
  --bg:#141312; --panel:#1c1b1a; --ink:#ece8e3; --muted:#a29c95;
  --line:#302e2b; --accent:#e0906b; --accent-soft:#2e211a;
  --good:#7fbf9a; --bad:#e0806b; --code:#232120;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.65 ui-serif,Georgia,"Times New Roman",serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:860px;margin:0 auto;padding:0 22px 100px}
header{padding:70px 0 34px;border-bottom:1px solid var(--line);margin-bottom:44px}
h1{font-size:2.5rem;line-height:1.12;margin:0 0 14px;letter-spacing:-.02em}
h2{font-size:1.45rem;margin:56px 0 14px;letter-spacing:-.01em;
   padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:1.05rem;margin:30px 0 10px;color:var(--accent);
   font-family:var(--mono);font-weight:600;letter-spacing:0}
p{margin:0 0 16px}
.lede{font-size:1.16rem;color:var(--muted)}
.tag{display:inline-block;font-family:var(--mono);font-size:.72rem;
  letter-spacing:.09em;text-transform:uppercase;color:var(--accent);
  background:var(--accent-soft);padding:4px 10px;border-radius:3px;margin-bottom:18px}
code{font-family:var(--mono);font-size:.87em;background:var(--code);
  padding:2px 5px;border-radius:3px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  padding:22px;margin:22px 0}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:.83rem}
th,td{padding:8px 11px;text-align:right;border-bottom:1px solid var(--line);
  white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:var(--muted);font-size:.72rem;letter-spacing:.05em;
  text-transform:uppercase}
tr.hi td{background:var(--accent-soft)}
.win{color:var(--good);font-weight:600}
.lose{color:var(--bad)}
input[type=text]{font-family:var(--mono);font-size:1rem;padding:10px 12px;
  border:1px solid var(--line);border-radius:6px;background:var(--bg);
  color:var(--ink);width:100%}
input[type=range]{width:100%;accent-color:var(--accent)}
label{display:block;font-size:.78rem;color:var(--muted);margin-bottom:7px;
  font-family:var(--mono);letter-spacing:.04em;text-transform:uppercase}
.grid-viz{display:flex;gap:3px;flex-wrap:wrap;margin:14px 0}
.cell{width:34px;height:44px;border-radius:4px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;font-family:var(--mono);font-size:.66rem;
  border:1px solid var(--line);background:var(--panel)}
.cell.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.cell.pad{background:var(--code);color:var(--muted)}
.cell b{font-size:.82rem;font-weight:600}
.cell span{font-size:.56rem;opacity:.75}
.kv{display:grid;grid-template-columns:auto 1fr;gap:5px 16px;
  font-family:var(--mono);font-size:.83rem;margin-top:14px}
.kv dt{color:var(--muted)}
.kv dd{margin:0}
.note{font-size:.9rem;color:var(--muted);border-left:2px solid var(--line);
  padding-left:14px;margin:18px 0}
.bad-note{border-left-color:var(--bad)}
footer{margin-top:70px;padding-top:22px;border-top:1px solid var(--line);
  font-size:.85rem;color:var(--muted)}
a{color:var(--accent)}
.missing{color:var(--muted);font-style:italic}
</style>

<div class="wrap">
<header>
  <div class="tag">ERA V5 &middot; Session 7 &middot; Problem 5</div>
  <h1>Making the Kronecker embedding run backwards</h1>
  <p class="lede">The Kronecker codec is exactly invertible, so the output head can
  be replaced by the input projection transposed. That removes the output matrix
  entirely and unties model size from vocabulary size &mdash; at a measured cost,
  reported here rather than rounded off.</p>
</header>

<h2>The idea in one paragraph</h2>
<p>A token is a byte string. The codec writes it as a sum over byte positions,
<code>&kappa;(b) = (1/&radic;L) &Sigma;<sub>p</sub> c<sub>b<sub>p</sub></sub> &otimes; p<sub>p</sub></code>,
which lands in a fixed space of <code>D = 256 &times; 32 = 8192</code> dimensions
regardless of how many tokens exist. Because the byte and the position occupy
separate axes, nothing is mixed: coordinate <code>b&middot;32 + p</code> means
&ldquo;byte <em>b</em> sits at position <em>p</em>&rdquo; and nothing else. So the
map is invertible, and the output side can be a classifier over that same grid
instead of a softmax over a vocabulary.</p>

<h3>Try it</h3>
<div class="panel">
  <label for="probe">any string &mdash; including one no tokenizer contains</label>
  <input type="text" id="probe" value="netwrok" spellcheck="false" autocomplete="off">
  <div class="grid-viz" id="viz"></div>
  <dl class="kv" id="probe-info"></dl>
</div>
<p class="note">The grid is <code>32 &times; 256</code>: one row per byte position,
one column per byte value. Exactly one cell is lit per row. Decoding reads each
row's argmax and stops at the first <code>PAD</code> &mdash; which is why any byte
string is reachable, not just the ones in a vocabulary.</p>

<h2>What was measured</h2>
<div id="results"></div>

<h2>The held-out vocabulary test</h2>
<p>Comparing heads on ordinary text only shows whether the byte head keeps up.
The actual claim of P5 is <em>vocabulary-freedom</em>, so a set of token ids was
withheld from every arm's training loss. Those tokens still appear in the context,
so every arm's shared input pathway sees their bytes; no arm is ever trained to
predict them. They are then scored on exactly those positions.</p>
<div id="heldout"></div>

<h2>One prediction at a time</h2>
<p>The same test, asked more bluntly: not &ldquo;was the token right&rdquo; but
&ldquo;how close was the byte string&rdquo;. Both measures are defined for every
arm &mdash; whatever token the baseline picks is a byte string too &mdash; so this
is not a metric the byte head grades itself on.</p>
<div id="demo-sec"></div>

<h2>Why not nearest neighbour?</h2>
<p>The session's framing is to predict the Kronecker vector and look up the
closest token. Two measurements, one of which came out against the design
argument and is reported anyway.</p>
<div id="decoders"></div>

<h2>Scaling</h2>
<p>The output-side parameter count is where the structural claim lives. Drag to
change the vocabulary.</p>
<div class="panel">
  <label for="vslider">vocabulary size &mdash; <span id="vlabel"></span></label>
  <input type="range" id="vslider" min="0" max="4" step="1" value="0">
  <div class="scroll"><table id="scaling"></table></div>
</div>

<footer id="foot"></footer>
</div>

<script id="evidence" type="application/json">__EVIDENCE__</script>
<script id="analysis" type="application/json">__ANALYSIS__</script>
<script id="demo" type="application/json">__DEMO__</script>
<script>
const EV = JSON.parse(document.getElementById('evidence').textContent);
const AN = JSON.parse(document.getElementById('analysis').textContent);
const DM = JSON.parse(document.getElementById('demo').textContent);
const P = 32, NB = 256, PAD = 0;

const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const num = (x, d=4) => (x === null || x === undefined) ? '&mdash;' : Number(x).toFixed(d);
const comma = n => Number(n).toLocaleString('en-US');

/* ---------- codec explorer ---------- */
function utf8Truncate(s, limit){
  const b = Array.from(new TextEncoder().encode(s));
  if (b.length <= limit) return b;
  let cut = limit;                       // never split a codepoint
  while (cut > 0 && (b[cut] & 0xC0) === 0x80) cut--;
  return b.slice(0, cut);
}
function renderProbe(){
  const s = document.getElementById('probe').value;
  const bytes = utf8Truncate(s, P);
  const viz = document.getElementById('viz');
  viz.innerHTML = '';
  const show = Math.min(P, Math.max(bytes.length + 2, 8));
  for (let p = 0; p < show; p++){
    const d = document.createElement('div');
    if (p < bytes.length){
      const b = bytes[p];
      const ch = (b >= 33 && b <= 126) ? String.fromCharCode(b) : '·';
      d.className = 'cell on';
      d.innerHTML = '<b>' + esc(ch) + '</b><span>' + b + '</span>';
      d.title = 'position ' + p + ', byte ' + b + ', flat index ' + (b * P + p);
    } else {
      d.className = 'cell pad';
      d.innerHTML = '<b>&middot;</b><span>pad</span>';
      d.title = 'position ' + p + ': PAD';
    }
    viz.appendChild(d);
  }
  const inVocab = (AN.reachability.probes.find(r => r.string === s) || {}).is_a_single_vocabulary_token;
  const idx = bytes.map((b, p) => b * P + p);
  document.getElementById('probe-info').innerHTML =
    '<dt>bytes</dt><dd>' + bytes.length + ' of ' + P + ' positions used</dd>' +
    '<dt>flat indices</dt><dd>' + (idx.slice(0, 10).join(', ') || '&mdash;') +
      (idx.length > 10 ? ', &hellip;' : '') + '</dd>' +
    '<dt>norm</dt><dd>' + (bytes.length ? '1.000 (scaled by 1/&radic;' + bytes.length + ')' : '0') + '</dd>' +
    '<dt>round trips</dt><dd class="win">yes &mdash; exactly</dd>' +
    '<dt>nearest-neighbour decoder</dt><dd class="' + (inVocab ? 'win' : 'lose') + '">' +
      (inVocab === undefined ? 'reachable only if this is one of the ' + comma(AN.vocab_size) + ' tokens'
       : inVocab ? 'reachable' : 'unreachable &mdash; not a single vocabulary token') + '</dd>';
}
document.getElementById('probe').addEventListener('input', renderProbe);

/* ---------- results ---------- */
function results(){
  const el = document.getElementById('results');
  if (!EV || !EV.summary){
    el.innerHTML = '<p class="missing">results/evidence.json not found &mdash; run <code>python run.py</code>.</p>';
    return;
  }
  if (EV.smoke){
    el.innerHTML = '<p class="note bad-note"><strong>These are smoke-test numbers, not results.</strong> ' +
      'evidence.json was produced by <code>python run.py --smoke</code>, which trains for a few dozen ' +
      'steps to check the pipeline. Run <code>python run.py</code> and regenerate this page.</p>';
    return;
  }
  const S = EV.summary, C = EV.config;
  const names = {vocab:'vocab (baseline)', kron:'kron', kron_tied:'kron_tied', regress:'regress'};
  let h = '<p>' + C.seeds.length + ' seeds per arm, ' + comma(C.steps) + ' steps, ' +
    'identical trunk and identical Kronecker input in every arm &mdash; only the head differs. ' +
    'Bits per byte is the negative log-probability assigned to the <em>correct next token</em>, ' +
    'divided by that token\'s length in bytes.</p>';
  h += '<div class="scroll"><table><tr><th>head</th><th>output params</th>' +
       '<th>bits/byte</th><th>vs. baseline</th><th>exact token acc.</th></tr>';
  const base = S.vocab.bits_per_byte.mean;
  for (const k of ['vocab','kron','kron_tied','regress']){
    const s = S[k]; if (!s) continue;
    const b = s.bits_per_byte.mean;
    h += '<tr' + (k === 'kron_tied' ? ' class="hi"' : '') + '><td>' + names[k] + '</td>' +
      '<td>' + comma(s.params.output) + '</td>' +
      '<td>' + (b === null ? '<span class="missing">undefined</span>' : num(b, 3) + ' &plusmn; ' + num(s.bits_per_byte.std, 3)) + '</td>' +
      '<td>' + (b === null ? '&mdash;' : (b > base ? '+' : '') + num(b - base, 3)) + '</td>' +
      '<td>' + num(s.exact_token_accuracy.mean, 4) + '</td></tr>';
  }
  h += '</table></div>';
  h += '<p class="note bad-note"><strong>The regress arm has no bits-per-byte at all.</strong> ' +
       'An MSE head defines no normalised distribution over tokens, so a likelihood does not exist ' +
       'for it. It is left blank rather than given a substitute number that would look comparable ' +
       'and would not be.</p>';

  const kt = S.kron_tied, kr = S.kron;
  if (kt && kr && kt.bits_per_byte.mean !== null){
    const d = Math.abs(kt.bits_per_byte.mean - kr.bits_per_byte.mean);
    const sd = Math.max(kt.bits_per_byte.std, kr.bits_per_byte.std);
    h += '<h3>tying costs ' + num(d, 3) + ' bits/byte</h3>' +
      '<p>Against a seed-to-seed spread of ' + num(sd, 3) + '. Tying removes ' +
      comma(kr.params.output - kt.params.output) + ' output parameters &mdash; ' +
      'from a ' + comma(kr.params.output) + '-weight matrix down to a ' +
      comma(kt.params.output) + '-element bias vector, which is the whole output side.</p>';
  }
  // where the gap actually comes from
  const rr = (EV.runs || []).filter(r => r.head === 'kron')
    .map(r => r.final.bits_per_byte_vocab_restricted).filter(x => x != null);
  if (rr.length){
    const restricted = rr.reduce((a, b) => a + b, 0) / rr.length;
    const free = S.kron.bits_per_byte.mean;
    h += '<h3>where the ' + num(free - base, 3) + ' bit/byte gap comes from</h3>' +
      '<p>Renormalising the byte head over just the ' + comma(EV.tokenizer_and_tables.vocab_size) +
      ' real tokens gives <strong>' + num(restricted, 3) + '</strong> bits/byte. So <strong>' +
      num(free - restricted, 3) + '</strong> of the gap is the price of ranging over every byte ' +
      'string instead of a fixed list &mdash; the capability being bought &mdash; and the ' +
      'remaining <strong>' + num(restricted - base, 3) + '</strong> is not. That larger part is ' +
      'the head predicting all 32 byte positions independently from one hidden state: knowing a ' +
      'token starts with <code>th</code> tells it nothing about position 2. It is the honest ' +
      'headline and it is not in the method\'s favour.</p>';
  }
  el.innerHTML = h;
}

/* ---------- held-out ---------- */
function heldout(){
  const el = document.getElementById('heldout');
  if (!EV || !EV.summary || EV.smoke){ el.innerHTML = ''; return; }
  const H = EV.heldout, S = EV.summary;
  let h = '<div class="scroll"><table><tr><th>head</th><th>bits/byte, seen</th>' +
          '<th>bits/byte, held out</th><th>penalty</th></tr>';
  for (const k of ['vocab','kron','kron_tied']){
    const s = S[k]; if (!s) continue;
    const seen = s.bits_per_byte_seen.mean, held = s.bits_per_byte_heldout.mean;
    const pen = (seen === null || held === null) ? null : held - seen;
    h += '<tr' + (k === 'kron_tied' ? ' class="hi"' : '') + '><td>' + k + '</td>' +
      '<td>' + num(seen, 3) + '</td><td>' + num(held, 3) + '</td>' +
      '<td class="' + (pen === null ? '' : pen > 1 ? 'lose' : 'win') + '">' +
      (pen === null ? '&mdash;' : '+' + num(pen, 3)) + '</td></tr>';
  }
  h += '</table></div>';
  h += '<dl class="kv"><dt>tokens withheld</dt><dd>' + H.n_tokens_withheld + '</dd>' +
    '<dt>training positions masked</dt><dd>' + comma(H.train_positions_masked) +
      ' (' + (100 * H.train_coverage).toFixed(2) + '%)</dd>' +
    '<dt>val positions scored</dt><dd>' + comma(H.val_positions_scored) + '</dd>' +
    '<dt>examples</dt><dd>' + (H.examples || []).slice(0, 12).map(e => '<code>' + esc(e) + '</code>').join(' ') + '</dd></dl>';
  el.innerHTML = h;
}

/* ---------- demo ---------- */
function demo(){
  const el = document.getElementById('demo-sec');
  if (!DM || !DM.tally){ el.innerHTML = '<p class="missing">results/demo.json not found &mdash; run <code>python demo.py</code>.</p>'; return; }
  const t = DM.tally;
  let h = '<div class="scroll"><table><tr><th>arm</th><th>exact</th>' +
          '<th>mean edit distance</th><th>first byte correct</th></tr>';
  for (const k of ['vocab','kron','kron_tied']){
    if (!t[k]) continue;
    h += '<tr' + (k === 'kron_tied' ? ' class="hi"' : '') + '><td>' + k + '</td>' +
      '<td>' + t[k].correct + '/' + comma(t[k].total) + '</td>' +
      '<td>' + num(t[k].mean_edit_distance, 3) + '</td>' +
      '<td class="' + (t[k].first_byte_accuracy > t.vocab.first_byte_accuracy ? 'win' : '') + '">' +
      (100 * t[k].first_byte_accuracy).toFixed(1) + '%</td></tr>';
  }
  h += '</table></div>';
  h += '<p class="note">Exact reproduction of a never-trained token is out of reach for ' +
    'every arm at this scale &mdash; ' + t.kron.correct + ' and ' + t.kron_tied.correct +
    ' successes in ' + comma(t.kron_tied.total) + ' are not a capability claim. The signal is ' +
    'the partial credit: the byte head gets the first byte right ' +
    (100 * t.kron_tied.first_byte_accuracy).toFixed(1) + '% of the time against the baseline\'s ' +
    (100 * t.vocab.first_byte_accuracy).toFixed(1) + '%. It composes part of the spelling and ' +
    'rarely finishes it.</p>';
  const hits = (DM.exact_hits || []).slice(0, 5);
  if (hits.length){
    h += '<h3>withheld tokens the tied head spelled exactly</h3><div class="scroll"><table>' +
         '<tr><th>context</th><th>target</th><th>emitted</th></tr>';
    for (const r of hits){
      h += '<tr><td><code>' + esc(r.context.slice(-34).replace(/\n/g,'\\n')) + '</code></td>' +
        '<td><code>' + esc(r.target) + '</code></td>' +
        '<td class="win"><code>' + esc(r.preds.kron_tied.emitted) + '</code></td></tr>';
    }
    h += '</table></div>';
  }
  el.innerHTML = h;
}

/* ---------- decoders ---------- */
function decoders(){
  const R = AN.reachability, N = AN.robustness, C = AN.concentration;
  let h = '<h3>reachability &mdash; the argument that settles it</h3>';
  h += '<p>Nearest neighbour is an <code>argmin</code> over a fixed table, so its range is ' +
    'exactly the ' + comma(R.nearest_neighbour_range) + ' strings in that table. It cannot emit ' +
    'anything else however good the model gets. Per-position argmax ranges over ' +
    R.per_position_argmax_range_note + '.</p>';
  h += '<div class="scroll"><table><tr><th>string</th><th>bytes</th>' +
       '<th>nearest neighbour</th><th>per-position argmax</th></tr>';
  for (const r of R.probes){
    h += '<tr><td><code>' + esc(r.string) + '</code></td><td>' + r.bytes + '</td>' +
      '<td class="' + (r.reachable_by_nearest_neighbour ? 'win' : 'lose') + '">' +
      (r.reachable_by_nearest_neighbour ? 'reachable' : 'unreachable') + '</td>' +
      '<td class="' + (r.reachable_by_per_position_argmax ? 'win' : 'lose') + '">' +
      (r.reachable_by_per_position_argmax ? 'reachable' : 'unreachable') + '</td></tr>';
  }
  h += '</table></div>';
  h += '<p><strong>' + R.reachable_by_nearest_neighbour + ' of ' + R.n_probes + '</strong> versus <strong>' +
    R.reachable_by_per_position_argmax + ' of ' + R.n_probes + '</strong>. ' +
    'A decoder that cannot leave the vocabulary cannot deliver a vocabulary-free model.</p>';

  h += '<h3>noise &mdash; where the easy argument fails</h3>';
  h += '<p>The tempting claim is that nearest neighbour is <em>fragile</em>. It is not. ' +
    'Under equal signal energy it decodes better than per-position argmax at every noise level ' +
    'tested, and the codec vectors are not pathologically crowded either &mdash; the median gap ' +
    'between best and second-best cosine is ' + num(C.median_gap_best_to_second, 3) + '.</p>';
  h += '<div class="scroll"><table><tr><th>&sigma;</th><th>nearest neighbour</th>' +
       '<th>per-position argmax</th></tr>';
  for (const r of N){
    h += '<tr><td>' + r.sigma.toFixed(2) + '</td><td>' + num(r.nearest_neighbour_exact, 3) +
      '</td><td>' + num(r.per_position_argmax_exact, 3) + '</td></tr>';
  }
  h += '</table></div>';
  h += '<p class="note">Kept because it refutes the convenient version of the argument. The ' +
    'equal-energy constraint also favours the codec vector by construction &mdash; it concentrates ' +
    'its norm in L coordinates while the grid spreads the same norm over all 32 rows &mdash; and a ' +
    'real classifier is under no such constraint, since cross-entropy is free to scale its logits. ' +
    'The case against nearest neighbour rests on reachability and on having no likelihood, ' +
    'not on noise.</p>';
  document.getElementById('decoders').innerHTML = h;
}

/* ---------- scaling ---------- */
function scaling(){
  const rows = AN.scaling.rows;
  const sl = document.getElementById('vslider');
  sl.max = rows.length - 1;
  function draw(){
    const r = rows[+sl.value];
    document.getElementById('vlabel').textContent = comma(r.vocab_size);
    const t = document.getElementById('scaling');
    const ratio = r.vocab_head / r.kron_tied_head;
    t.innerHTML =
      '<tr><th>head</th><th>output parameters</th><th>relative</th></tr>' +
      '<tr><td>vocab</td><td>' + comma(r.vocab_head) + '</td><td>' + Math.round(ratio).toLocaleString() + '&times;</td></tr>' +
      '<tr><td>kron</td><td>' + comma(r.kron_head) + '</td><td>' + Math.round(r.kron_head / r.kron_tied_head) + '&times;</td></tr>' +
      '<tr class="hi"><td>kron_tied</td><td>' + comma(r.kron_tied_head) + '</td><td>1&times;</td></tr>';
  }
  sl.addEventListener('input', draw);
  draw();
}

/* ---------- footer ---------- */
function foot(){
  const e = (EV && EV.environment) || {};
  const c = (EV && EV.corpus) || {};
  document.getElementById('foot').innerHTML =
    'Generated from <code>results/evidence.json</code> and <code>results/analysis.json</code>. ' +
    (c.sha256 ? 'Corpus ' + comma(c.bytes) + ' bytes, sha256 <code>' + c.sha256.slice(0, 16) + '</code>. ' : '') +
    (e.torch ? 'torch ' + e.torch + ', device ' + e.device + '. ' : '') +
    (EV && EV.wall_clock_s ? 'Run took ' + (EV.wall_clock_s / 60).toFixed(0) + ' minutes.' : '');
}

renderProbe(); results(); heldout(); demo(); decoders(); scaling(); foot();
</script>
"""


def main() -> int:
    ev = load("evidence.json")
    an = load("analysis.json")
    if an is None:
        print("results/analysis.json missing; run `python analysis.py` first")
        return 1

    dm = load("demo.json")
    html = (PAGE
            .replace("__EVIDENCE__", json.dumps(ev or {}))
            .replace("__ANALYSIS__", json.dumps(an))
            .replace("__DEMO__", json.dumps(dm or {})))

    os.makedirs(DOCS, exist_ok=True)
    out = os.path.join(DOCS, "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("-> {} ({:,} bytes){}".format(
        os.path.relpath(out, ROOT), len(html.encode("utf-8")),
        "" if ev else "   [no evidence.json yet: results section will be empty]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
