"""A one-file labelling page for the compound sheet. It opens from disk, needs no server,
keeps progress in the browser, and exports a CSV that `labels score` reads."""

from __future__ import annotations

import json
from collections.abc import Sequence


def render_page(words: Sequence) -> str:
    """HTML for labelling the given SheetWords. Nothing a splitter produced is shown."""
    data = [{"id": i, "word": w.word, "context": w.context, "stratum": w.stratum} for i, w in enumerate(words, start=1)]
    # "<" is escaped so that no question text can close the script element.
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return _PAGE.replace("__WORDS__", payload)


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compound sheet</title>
<style>
  :root {
    --bg: #f6f4ef; --card: #ffffff; --ink: #1d1c1a; --soft: #6b675f; --line: #d9d4c8;
    --accent: #0b5c4a; --accent-ink: #ffffff; --accent-soft: #dcefe9; --warn: #a13d12; --cut: #c2410c;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #151614; --card: #1f211e; --ink: #ecebe6; --soft: #a3a097; --line: #3a3d38;
      --accent: #5cc4a8; --accent-ink: #0d1f1a; --accent-soft: #1f3a33; --warn: #f0a070; --cut: #fb923c;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif; }
  header { position: sticky; top: 0; z-index: 2; background: var(--bg); border-bottom: 1px solid var(--line); }
  .bar { max-width: 860px; margin: 0 auto; padding: 12px 16px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  h1 { font-size: 18px; margin: 0; }
  .count { color: var(--soft); font-variant-numeric: tabular-nums; }
  .track { flex: 1 1 160px; height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; }
  .fill { height: 100%; width: 0; background: var(--accent); transition: width .2s; }
  button { font: inherit; color: inherit; cursor: pointer; }
  .primary { background: var(--accent); color: var(--accent-ink); border: 0; border-radius: 8px; padding: 8px 14px; font-weight: 600; }
  main { max-width: 860px; margin: 0 auto; padding: 20px 16px 60px; }
  .how { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; margin-bottom: 16px; }
  .how summary { cursor: pointer; font-weight: 600; }
  .how ol { margin: 8px 0 0; padding-left: 20px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 24px 20px; }
  .position { color: var(--soft); font-size: 14px; }
  .word { display: flex; flex-wrap: wrap; align-items: stretch; justify-content: center; margin: 18px 0 6px; font: 600 34px/1 ui-monospace, "SF Mono", Menlo, monospace; }
  .letter { padding: 10px 0; }
  .gap { width: 16px; border: 0; background: transparent; padding: 0; position: relative; border-radius: 4px; }
  .gap::after { content: ""; position: absolute; left: 50%; top: 4px; bottom: 4px; width: 2px; transform: translateX(-50%); background: var(--line); opacity: 0; }
  .gap:hover { background: var(--accent-soft); }
  .gap:hover::after { opacity: 1; }
  .gap.on { width: 26px; }
  .gap.on::after { opacity: 1; width: 4px; background: var(--cut); border-radius: 2px; }
  .gap:focus-visible, .type:focus-visible, .dot:focus-visible, .nav button:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
  .preview { text-align: center; color: var(--soft); min-height: 24px; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  .context { text-align: center; margin: 14px 0 20px; font-size: 17px; }
  .context mark { background: var(--accent-soft); color: inherit; padding: 1px 3px; border-radius: 3px; }
  .types { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
  .type { text-align: left; background: var(--bg); border: 2px solid var(--line); border-radius: 10px; padding: 10px 12px; }
  .type b { display: block; }
  .type small { color: var(--soft); }
  .type kbd { float: right; font: 12px ui-monospace, monospace; color: var(--soft); border: 1px solid var(--line); border-radius: 4px; padding: 0 5px; }
  .type.on { border-color: var(--accent); background: var(--accent-soft); }
  .hint { color: var(--warn); min-height: 24px; margin: 10px 0 0; text-align: center; font-weight: 600; }
  .note { width: 100%; margin-top: 8px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--bg); color: var(--ink); font: inherit; }
  .nav { display: flex; justify-content: space-between; margin-top: 14px; }
  .nav button { background: transparent; border: 1px solid var(--line); border-radius: 8px; padding: 8px 14px; }
  .dots { display: grid; grid-template-columns: repeat(auto-fill, 22px); gap: 5px; margin-top: 22px; }
  .dot { width: 22px; height: 22px; border-radius: 5px; border: 1px solid var(--line); background: var(--card); padding: 0; font-size: 0; }
  .dot.done { background: var(--accent); border-color: var(--accent); }
  .dot.now { outline: 3px solid var(--cut); outline-offset: 1px; }
  .done-banner { display: none; background: var(--accent-soft); border: 1px solid var(--accent); border-radius: 10px; padding: 12px 16px; margin-bottom: 16px; }
  .done-banner.show { display: block; }
</style>
</head>
<body>
<header>
  <div class="bar">
    <h1>Compound sheet</h1>
    <span class="count" id="count"></span>
    <div class="track" aria-hidden="true"><div class="fill" id="fill"></div></div>
    <button class="primary" id="download">Download CSV</button>
  </div>
</header>
<main>
  <div class="done-banner" id="banner">All words are labelled. Click <b>Download CSV</b> and save the file as <code>labels/compounds_labels.csv</code> in the project folder.</div>
  <details class="how" open>
    <summary>How to label (about 20 seconds per word)</summary>
    <ol>
      <li><b>Cut the word.</b> Click between two letters where two parts meet. A linking element stays with the left part: <code>Bundes|gericht</code>, <code>Sonnen|schein</code>. Click again to remove a cut.</li>
      <li><b>Pick the type.</b> The page then moves to the next word.</li>
      <li>Your work is saved in this browser as you go. You can close the page and come back.</li>
    </ol>
  </details>
  <div class="card">
    <div class="position" id="position"></div>
    <div class="word" id="word"></div>
    <div class="preview" id="preview"></div>
    <div class="context" id="context"></div>
    <div class="types" id="types"></div>
    <p class="hint" id="hint" role="status"></p>
    <input class="note" id="note" placeholder="Note (optional)" aria-label="Note">
    <div class="nav"><button id="back">&larr; Back</button><button id="next">Next &rarr;</button></div>
  </div>
  <div class="dots" id="dots" aria-label="All words"></div>
</main>
<script>
const WORDS = __WORDS__;
const TYPES = [
  {name: "compound", text: "Compound", help: "The parts keep their meaning: Tabak|konzern", cuts: true},
  {name: "lexicalised compound", text: "Lexicalised compound", help: "Built from parts, but the meaning is its own: Bahn|hof", cuts: true},
  {name: "name", text: "Name", help: "A proper name. Cut it only if you see real parts.", cuts: false},
  {name: "not a compound", text: "Not a compound", help: "One word, or a derived word: Regierung", cuts: false, clears: true},
];
const KEY = "compound-search-labels-v1";
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }
let current = Math.max(0, WORDS.findIndex(w => !(labels[w.id] && labels[w.id].type)));
const $ = id => document.getElementById(id);
const entry = w => labels[w.id] || (labels[w.id] = {cuts: [], type: "", note: ""});
function save() { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function splitText(w) {
  const cuts = entry(w).cuts.slice().sort((a, b) => a - b);
  let out = [], last = 0;
  for (const c of cuts) { out.push(w.word.slice(last, c)); last = c; }
  out.push(w.word.slice(last));
  return out;
}
function render() {
  const w = WORDS[current], e = entry(w);
  $("position").textContent = "Word " + (current + 1) + " of " + WORDS.length;
  const box = $("word"); box.textContent = "";
  [...w.word].forEach((ch, i) => {
    if (i > 0) {
      const gap = document.createElement("button");
      gap.className = "gap" + (e.cuts.includes(i) ? " on" : "");
      gap.setAttribute("aria-label", "Cut between " + w.word[i - 1] + " and " + ch);
      gap.setAttribute("aria-pressed", e.cuts.includes(i));
      gap.onclick = () => toggleCut(i);
      box.appendChild(gap);
    }
    const span = document.createElement("span");
    span.className = "letter"; span.textContent = ch;
    box.appendChild(span);
  });
  const pieces = splitText(w);
  $("preview").textContent = pieces.length > 1 ? pieces.join(" + ") : "no cut";
  const ctx = $("context"); ctx.textContent = "";
  const at = w.context.indexOf(w.word);
  if (at < 0) { ctx.textContent = w.context; } else {
    ctx.append(w.context.slice(0, at));
    const mark = document.createElement("mark"); mark.textContent = w.word; ctx.append(mark);
    ctx.append(w.context.slice(at + w.word.length));
  }
  const types = $("types"); types.textContent = "";
  TYPES.forEach((t, i) => {
    const b = document.createElement("button");
    b.className = "type" + (e.type === t.name ? " on" : "");
    b.setAttribute("aria-pressed", e.type === t.name);
    const kbd = document.createElement("kbd"); kbd.textContent = i + 1;
    const title = document.createElement("b"); title.textContent = t.text;
    const help = document.createElement("small"); help.textContent = t.help;
    b.append(kbd, title, help);
    b.onclick = () => pickType(t);
    types.appendChild(b);
  });
  $("note").value = e.note || "";
  $("hint").textContent = "";
  const done = WORDS.filter(x => labels[x.id] && labels[x.id].type).length;
  $("count").textContent = done + " of " + WORDS.length + " labelled";
  $("fill").style.width = (100 * done / WORDS.length) + "%";
  $("banner").classList.toggle("show", done === WORDS.length);
  const dots = $("dots"); dots.textContent = "";
  WORDS.forEach((x, i) => {
    const d = document.createElement("button");
    d.className = "dot" + (labels[x.id] && labels[x.id].type ? " done" : "") + (i === current ? " now" : "");
    d.title = (i + 1) + ". " + x.word;
    d.textContent = x.word;
    d.onclick = () => { current = i; render(); };
    dots.appendChild(d);
  });
}
function toggleCut(i) {
  const e = entry(WORDS[current]);
  e.cuts = e.cuts.includes(i) ? e.cuts.filter(c => c !== i) : e.cuts.concat(i).sort((a, b) => a - b);
  if (e.type === "not a compound" && e.cuts.length) e.type = "";
  save(); render();
}
function pickType(t) {
  const e = entry(WORDS[current]);
  if (t.cuts && e.cuts.length === 0) {
    render();
    $("hint").textContent = "First click between the letters to cut the word, then pick this type.";
    return;
  }
  if (t.clears) e.cuts = [];
  e.type = t.name; save(); render();
  setTimeout(() => step(1), 350);
}
function step(by) { current = Math.min(WORDS.length - 1, Math.max(0, current + by)); render(); }
$("back").onclick = () => step(-1);
$("next").onclick = () => step(1);
$("note").oninput = ev => { entry(WORDS[current]).note = ev.target.value; save(); };
document.addEventListener("keydown", ev => {
  if (ev.target === $("note")) return;
  if (ev.key >= "1" && ev.key <= "4") pickType(TYPES[Number(ev.key) - 1]);
  if (ev.key === "ArrowRight") step(1);
  if (ev.key === "ArrowLeft") step(-1);
});
$("download").onclick = () => {
  const quote = v => '"' + String(v).replace(/"/g, '""') + '"';
  const rows = [["id", "word", "split", "type", "note", "stratum", "context"]];
  for (const w of WORDS) {
    const e = entry(w);
    rows.push([w.id, w.word, e.type ? splitText(w).join("+") : "", e.type, e.note || "", w.stratum, w.context]);
  }
  const csv = rows.map(r => r.map(quote).join(",")).join("\n") + "\n";
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
  link.download = "compounds_labels.csv";
  document.body.appendChild(link); link.click(); link.remove();
};
render();
</script>
</body>
</html>
"""
