/* Where to buy groceries — Cambridge, MA.
 *
 * Reads docs/verdicts.json, which stage 3 of the pipeline produces.
 *
 * Every string in that file except the keys is text somebody typed on Reddit.
 * It is set with textContent and never with innerHTML, and the only attribute
 * built from it is a permalink, which is checked to be a reddit path first.
 * Do not "simplify" either of those.
 */
"use strict";

let DATA = null;
let cmpSort = { key: "w", dir: -1 };

/* ── tiny DOM helpers ───────────────────────────────────────────────── */

function el(tag, props = {}, kids = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;      // never innerHTML
    else if (k === "dataset") Object.assign(n.dataset, v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const kid of [].concat(kids)) {
    if (kid) n.append(typeof kid === "string" ? document.createTextNode(kid) : kid);
  }
  return n;
}

const $ = (sel) => document.querySelector(sel);
const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

/** Reddit permalinks only. A claim cannot smuggle in a javascript: URL. */
function redditUrl(path) {
  return /^\/r\/[A-Za-z0-9_]+\//.test(path || "") ? "https://reddit.com" + path : null;
}

const fmt = (x) => (x > 0 ? "+" : "") + x.toFixed(2);
const cls = (s) => (s > 0.15 ? "pos" : s < -0.15 ? "neg" : "mid");
const titleCase = (s) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/* ── scoring ────────────────────────────────────────────────────────── */

/* Enough evidence for the cell to be worth acting on. Below this the
 * shrinkage in stage 3 dominates and every number drifts toward zero. */
const MIN_W = 1.0;

/** Combine quality and price into one number. pref: 0 = cheapest, 1 = best. */
function value(cell, pref) {
  const quality = cell.s;
  // price_level is negative for cheap, so flip it: higher is better value.
  const price = cell.pl === undefined ? 0 : -cell.pl;
  const havePrice = cell.pl !== undefined;
  if (!havePrice) return quality;          // nothing said about cost
  return pref * quality + (1 - pref) * price;
}

/** Split a shopping list into terms. Blank lines and stray commas ignored. */
function parseList(raw) {
  return [...new Set(
    raw.split(/[\n,;]+/).map((s) => s.trim().toLowerCase()).filter(Boolean)
  )].slice(0, 40);
}

/**
 * Resolve one list term to evidence, most specific first:
 *   1. an item the corpus actually discusses ("rotisserie chicken")
 *   2. the department the word belongs to ("milk" -> dairy)
 *   3. nothing — reported as unmatched rather than guessed at
 */
function resolve(term) {
  const perStore = {};
  for (const [store, items] of Object.entries(DATA.items)) {
    for (const [name, cell] of Object.entries(items)) {
      if (name === term || name.includes(term) || term.includes(name)) {
        const prev = perStore[store];
        if (!prev || cell.w > prev.cell.w) perStore[store] = { cell, label: name };
      }
    }
  }
  if (Object.keys(perStore).length) return { kind: "item", perStore };

  const category = DATA.keywords[term];
  if (category) {
    const byStore = {};
    for (const [store, cats] of Object.entries(DATA.stores)) {
      if (cats[category]) byStore[store] = { cell: cats[category], label: category };
    }
    if (Object.keys(byStore).length) return { kind: "category", category, perStore: byStore };
  }
  return { kind: "none", perStore: {} };
}

/** Rank stores across the whole list. */
function rankStores(terms, pref) {
  const resolved = terms.map((t) => ({ term: t, ...resolve(t) }));
  const tally = {};
  for (const store of Object.keys(DATA.stores)) {
    tally[store] = { store, score: 0, covered: 0, wins: [], weak: [] };
  }
  for (const r of resolved) {
    if (r.kind === "none") continue;
    let best = null;
    for (const [store, hit] of Object.entries(r.perStore)) {
      if (hit.cell.w < MIN_W || !tally[store]) continue;
      const v = value(hit.cell, pref);
      tally[store].score += v;
      tally[store].covered += 1;
      if (v < -0.15) tally[store].weak.push(r.term);
      if (!best || v > best.v) best = { store, v, hit };
    }
    if (best) tally[best.store].wins.push(r.term);
  }
  const ranked = Object.values(tally)
    .filter((t) => t.covered > 0)
    // Average, not sum: a store with evidence on two items should not beat
    // one with evidence on eight simply by having fewer chances to be bad.
    .map((t) => ({ ...t, avg: t.score / t.covered }))
    .sort((a, b) => b.covered - a.covered || b.avg - a.avg);
  return { ranked, resolved };
}

/* ── view: shopping list ────────────────────────────────────────────── */

function prefLabel(v) {
  if (v <= 15) return "cheapest";
  if (v <= 40) return "mostly price";
  if (v < 60) return "balanced";
  if (v < 85) return "mostly quality";
  return "best quality";
}

function runList() {
  const terms = parseList($("#list-input").value);
  const out = $("#list-result");
  const detail = $("#list-detail");
  clear(out); clear(detail);
  if (!terms.length) {
    out.append(el("p", { class: "empty", text: "Add a few items and press the button." }));
    return;
  }
  const pref = $("#pref").value / 100;
  const { ranked, resolved } = rankStores(terms, pref);
  const unmatched = resolved.filter((r) => r.kind === "none").map((r) => r.term);

  if (!ranked.length) {
    out.append(el("p", { class: "empty",
      text: "Nothing on that list matches anything the corpus discusses." }));
  }

  ranked.slice(0, 4).forEach((t, i) => {
    const card = el("div", { class: "card" + (i === 0 ? " win" : "") });
    card.append(el("h3", {}, [
      el("span", { text: t.store }),
      el("span", { class: "rank", text: `${fmt(t.avg)}` }),
    ]));
    card.append(el("p", { class: "why",
      text: `evidence for ${t.covered} of ${terms.length} item${terms.length === 1 ? "" : "s"}` }));
    const badges = el("div", { class: "badges" });
    t.wins.slice(0, 6).forEach((w) =>
      badges.append(el("span", { class: "badge good", text: "best for " + w })));
    t.weak.slice(0, 3).forEach((w) =>
      badges.append(el("span", { class: "badge bad", text: "weak on " + w })));
    if (badges.children.length) card.append(badges);
    card.onclick = () => showStore(t.store);
    out.append(card);
  });

  if (unmatched.length) {
    out.append(el("p", { class: "muted", text: "No evidence either way: " + unmatched.join(", ") }));
  }

  // Per-item breakdown: the split shop, if you are willing to make two stops.
  const rows = resolved.filter((r) => r.kind !== "none");
  if (!rows.length) return;
  detail.append(el("h2", { text: "Item by item" }));
  const table = el("table", { class: "data" });
  table.append(el("thead", {}, el("tr", {}, [
    el("th", { text: "Item" }),
    el("th", { text: "Matched" }),
    el("th", { text: "Best here" }),
    el("th", { class: "num", text: "Score" }),
    el("th", { class: "num", text: "Claims" }),
  ])));
  const body = el("tbody");
  for (const r of rows) {
    let best = null;
    for (const [store, hit] of Object.entries(r.perStore)) {
      if (hit.cell.w < MIN_W) continue;
      const v = value(hit.cell, pref);
      if (!best || v > best.v) best = { store, v, hit };
    }
    if (!best) continue;
    const tr = el("tr", {}, [
      el("td", { text: r.term }),
      el("td", { class: "muted", text: r.kind === "item" ? best.hit.label : titleCase(best.hit.label) }),
      el("td", { text: best.store }),
      el("td", { class: "num" }, el("span", { class: "score " + cls(best.v), text: fmt(best.v) })),
      el("td", { class: "num muted", text: String(best.hit.cell.n) }),
    ]);
    tr.onclick = () => showStore(best.store);
    body.append(tr);
  }
  table.append(body);
  detail.append(table);
}

/* ── view: compare ──────────────────────────────────────────────────── */

function compareRows() {
  const cat = $("#cmp-cat").value;
  const minW = Number($("#cmp-min").value);
  const hideThin = $("#cmp-thin").checked;
  const rows = [];
  for (const [store, cats] of Object.entries(DATA.stores)) {
    const totals = DATA.totals[store] || { n: 0, w: 0, s: 0, thin: true };
    if (hideThin && totals.thin) continue;
    const cell = cat ? cats[cat] : null;
    if (cat && !cell) continue;
    const price = cat ? cell.pl : (cats.price_overall ? cats.price_overall.pl : undefined);
    const quality = cat ? cell.s : (cats.quality_overall ? cats.quality_overall.s : undefined);
    const n = cat ? cell.n : totals.n;
    const w = cat ? cell.w : totals.w;
    if (w < minW) continue;
    rows.push({ store, price, quality, sentiment: cat ? cell.s : totals.s, n, w });
  }
  const k = cmpSort.key;
  rows.sort((a, b) => {
    if (k === "store") return cmpSort.dir * a.store.localeCompare(b.store);
    const av = a[k], bv = b[k];
    if (av === undefined) return 1;
    if (bv === undefined) return -1;
    return cmpSort.dir * (av - bv);
  });
  return rows;
}

function renderCompare() {
  const rows = compareRows();
  const body = $("#cmp-table tbody");
  clear(body);
  if (!rows.length) {
    body.append(el("tr", {}, el("td", { colspan: "6", class: "empty",
      text: "No store has that much evidence in this category." })));
  }
  for (const r of rows) {
    const tr = el("tr", {}, [
      el("td", { text: r.store }),
      el("td", { class: "num" }, r.price === undefined
        ? el("span", { class: "muted", text: "–" })
        : el("span", { class: "score " + cls(-r.price), text: fmt(-r.price) })),
      el("td", { class: "num" }, r.quality === undefined
        ? el("span", { class: "muted", text: "–" })
        : el("span", { class: "score " + cls(r.quality), text: fmt(r.quality) })),
      el("td", { class: "num" }, el("span", { class: "score " + cls(r.sentiment), text: fmt(r.sentiment) })),
      el("td", { class: "num muted", text: r.n.toLocaleString() }),
      el("td", { class: "num muted", text: r.w.toLocaleString() }),
    ]);
    tr.onclick = () => showStore(r.store);
    body.append(tr);
  }
  renderScatter(rows);
}

function renderScatter(rows) {
  const svg = $("#scatter");
  clear(svg);
  const W = 640, H = 420, m = { t: 22, r: 20, b: 44, l: 52 };
  const pts = rows.filter((r) => r.price !== undefined && r.quality !== undefined);
  const ns = (t, a) => {
    const n = document.createElementNS("http://www.w3.org/2000/svg", t);
    for (const [k, v] of Object.entries(a)) n.setAttribute(k, v);
    return n;
  };
  const x = (v) => m.l + ((-v + 1) / 2) * (W - m.l - m.r);   // left = expensive
  const y = (v) => m.t + ((1 - v) / 2) * (H - m.t - m.b);    // top  = high quality

  for (const v of [-0.5, 0, 0.5]) {
    svg.append(ns("line", { class: v === 0 ? "axis" : "gridline",
      x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }));
    svg.append(ns("line", { class: v === 0 ? "axis" : "gridline",
      x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
  }
  const label = (t, px, py, cls2, anchor) => {
    const n = ns("text", { class: cls2, x: px, y: py, "text-anchor": anchor || "middle" });
    n.textContent = t;
    svg.append(n);
  };
  label("cheaper →", W - m.r, H - m.b + 30, "axis-label", "end");
  label("← pricier", m.l, H - m.b + 30, "axis-label", "start");
  label("better quality ↑", 12, m.t + 4, "axis-label", "start");
  label("worse quality ↓", 12, H - m.b - 4, "axis-label", "start");
  label("bargain", x(0.55), y(0.85), "quad");
  label("premium", x(-0.55), y(0.85), "quad");
  label("avoid", x(-0.55), y(-0.85), "quad");

  for (const p of pts) {
    const r = Math.max(4, Math.min(16, Math.sqrt(p.w) / 1.6));
    const dot = ns("circle", { class: "dot", cx: x(p.price), cy: y(p.quality), r });
    const t = ns("title", {});
    t.textContent = `${p.store} — ${p.n.toLocaleString()} claims`;
    dot.append(t);
    dot.addEventListener("click", () => showStore(p.store));
    svg.append(dot);
    const name = ns("text", { class: "dot-label", x: x(p.price), y: y(p.quality) - r - 4,
      "text-anchor": "middle" });
    name.textContent = p.store;
    svg.append(name);
  }
  if (!pts.length) label("no store has both price and quality evidence here",
    W / 2, H / 2, "axis-label");
}

/* ── view: store detail ─────────────────────────────────────────────── */

function cellBlock(name, cell) {
  const d = el("details", { class: "cell" });
  const sum = el("summary", {}, [
    el("span", { class: "cat", text: titleCase(name) }),
    el("span", { class: "score " + cls(cell.s), text: fmt(cell.s) }),
    el("span", { class: "muted", text: `${cell.n} claim${cell.n === 1 ? "" : "s"}` }),
  ]);
  if (cell.p) sum.append(el("span", { class: "badge " + (cell.p === "cheap" ? "good" : cell.p === "expensive" ? "warn" : ""), text: cell.p }));
  d.append(sum);
  const quotes = el("div", { class: "quotes" });
  for (const q of cell.e) {
    const meta = el("div", { class: "meta" }, [
      el("span", { text: q.d }),
      q.l ? el("span", { text: "· " + q.l }) : null,
      el("span", { text: "· " + q.c + " confidence" }),
    ]);
    const url = redditUrl(q.u);
    if (url) meta.append(el("a", { href: url, target: "_blank", rel: "noopener nofollow" }, "source ↗"));
    quotes.append(el("div", { class: "quote" }, [el("div", { text: q.t }), meta]));
  }
  d.append(quotes);
  return d;
}

function showStore(store) {
  switchView("store");
  $("#store-pick").value = store;
  renderStore();
}

function renderStore() {
  const store = $("#store-pick").value;
  const pick = $("#branch-pick");
  const branches = DATA.branches[store] || {};
  const chosen = pick.value;
  clear(pick);
  pick.append(el("option", { value: "", text: "all branches (chain level)" }));
  for (const b of Object.keys(branches).sort()) {
    pick.append(el("option", { value: b, text: b }));
  }
  pick.value = branches[chosen] ? chosen : "";

  const body = $("#store-body");
  clear(body);
  const totals = DATA.totals[store];
  const cats = pick.value ? branches[pick.value] : DATA.stores[store] || {};

  const head = el("div", { class: "card" });
  head.append(el("h3", {}, [
    el("span", { text: pick.value ? `${store} — ${pick.value}` : store }),
    el("span", { class: "rank", text: fmt(pick.value ? 0 : totals.s) }),
  ]));
  head.append(el("p", { class: "why",
    text: `${totals.n.toLocaleString()} claims across the chain, ${Object.keys(branches).length} branch${Object.keys(branches).length === 1 ? "" : "es"} with their own evidence` }));
  if (totals.thin) head.append(el("p", { class: "badges" },
    el("span", { class: "badge warn", text: "thin evidence — treat with caution" })));
  body.append(head);

  const ordered = Object.entries(cats).sort((a, b) => b[1].w - a[1].w);
  if (!ordered.length) {
    body.append(el("p", { class: "empty", text: "No claims above the evidence threshold." }));
  }
  body.append(el("h2", { text: "By category" }));
  for (const [name, cell] of ordered) body.append(cellBlock(name, cell));

  const items = DATA.items[store] || {};
  const topItems = Object.entries(items).sort((a, b) => b[1].w - a[1].w).slice(0, 20);
  if (topItems.length) {
    body.append(el("h2", { text: "Specific items people mention" }));
    for (const [name, cell] of topItems) body.append(cellBlock(name, cell));
  }

  const regions = DATA.regions[store] || {};
  if (Object.keys(regions).length) {
    body.append(el("h2", { text: "Mentioned by area, not by branch" }));
    body.append(el("p", { class: "muted",
      text: "These name a region rather than a store, so they are listed apart: "
            + Object.keys(regions).sort().join(", ") }));
  }
}

/* ── view: item search ──────────────────────────────────────────────── */

function renderItems() {
  const q = $("#item-q").value.trim().toLowerCase();
  const onlyStore = $("#item-store").value;
  const body = $("#item-body");
  clear(body);

  const hits = [];
  for (const [store, items] of Object.entries(DATA.items)) {
    if (onlyStore && store !== onlyStore) continue;
    for (const [name, cell] of Object.entries(items)) {
      if (q && !name.includes(q)) continue;
      hits.push({ store, name, cell });
    }
  }
  hits.sort((a, b) => b.cell.w - a.cell.w);

  if (!hits.length) {
    body.append(el("p", { class: "empty",
      text: q ? `Nobody discussed "${q}" in a way the pipeline could pin to a store.`
              : "No items." }));
    return;
  }
  body.append(el("p", { class: "muted",
    text: `${hits.length.toLocaleString()} item${hits.length === 1 ? "" : "s"}` }));

  const table = el("table", { class: "data" });
  table.append(el("thead", {}, el("tr", {}, [
    el("th", { text: "Item" }), el("th", { text: "Store" }),
    el("th", { class: "num", text: "Sentiment" }),
    el("th", { class: "num", text: "Price" }),
    el("th", { class: "num", text: "Claims" }),
  ])));
  const tb = el("tbody");
  for (const h of hits.slice(0, 250)) {
    const tr = el("tr", {}, [
      el("td", { text: h.name }),
      el("td", { text: h.store }),
      el("td", { class: "num" }, el("span", { class: "score " + cls(h.cell.s), text: fmt(h.cell.s) })),
      el("td", { class: "num muted", text: h.cell.p || "–" }),
      el("td", { class: "num muted", text: String(h.cell.n) }),
    ]);
    tr.onclick = () => showStore(h.store);
    tb.append(tr);
  }
  table.append(tb);
  body.append(table);
  if (hits.length > 250) {
    body.append(el("p", { class: "muted", text: `Showing the 250 best-evidenced of ${hits.length}. Narrow the search to see the rest.` }));
  }
}

/* ── boot ───────────────────────────────────────────────────────────── */

function switchView(name) {
  for (const b of document.querySelectorAll(".tabs button")) b.classList.toggle("on", b.dataset.view === name);
  for (const v of document.querySelectorAll(".view")) v.hidden = v.id !== "view-" + name;
  if (location.hash.slice(1) !== name) history.replaceState(null, "", "#" + name);
}

function fillMethod() {
  const c = DATA.corpus || {};
  const dl = $("#m-prov");
  const pairs = [
    ["Generated", DATA.generated_at],
    ["Documents extracted", (c.documents_extracted || 0).toLocaleString()],
    ["Candidate documents", (c.working_set || 0).toLocaleString()],
    ["Stores covered", Object.keys(DATA.stores).length],
    ["Branches", Object.values(DATA.branches).reduce((a, b) => a + Object.keys(b).length, 0)],
    ["Items indexed", Object.values(DATA.items).reduce((a, b) => a + Object.keys(b).length, 0)],
    ["Shrinkage constant", DATA.method.shrinkage_k],
    ["Default half-life", DATA.method.default_half_life_years + " years"],
    ["Transient claims", DATA.method.transient_claims],
  ];
  for (const [k, v] of pairs) { dl.append(el("dt", { text: k }), el("dd", { text: String(v) })); }
}

async function boot() {
  try {
    const res = await fetch("verdicts.json");
    if (!res.ok) throw new Error("HTTP " + res.status);
    DATA = await res.json();
  } catch (err) {
    $("#loading").textContent = "Could not load the data: " + err.message;
    return;
  }
  $("#loading").hidden = true;

  const stores = Object.keys(DATA.stores).sort();
  for (const s of stores) {
    $("#store-pick").append(el("option", { value: s, text: s }));
    $("#item-store").append(el("option", { value: s, text: s }));
  }
  for (const c of DATA.categories) {
    $("#cmp-cat").append(el("option", { value: c, text: titleCase(c) }));
  }

  const c = DATA.corpus || {};
  $("#corpus-line").textContent =
    `${(c.documents_extracted || 0).toLocaleString()} Reddit posts and comments`;
  $("#footer-line").textContent =
    `Generated ${DATA.generated_at} · opinion aggregated from Reddit, not verified prices · `;
  $("#footer-line").append(el("a", { href: "https://github.com/pelednoam/find-groceries" }, "source"));
  const dates = [];
  for (const cats of Object.values(DATA.stores)) {
    for (const cell of Object.values(cats)) for (const e of cell.e) dates.push(e.d);
  }
  if (dates.length) {
    dates.sort();
    $("#m-span").textContent = `${dates[0]} to ${dates[dates.length - 1]}`;
  }
  fillMethod();

  for (const b of document.querySelectorAll("[data-view]")) {
    b.addEventListener("click", () => {
      switchView(b.dataset.view);
      if (b.dataset.view === "compare") renderCompare();
      if (b.dataset.view === "store") renderStore();
      if (b.dataset.view === "items") renderItems();
    });
  }
  $("#list-go").addEventListener("click", runList);
  $("#list-demo").addEventListener("click", () => {
    $("#list-input").value = "milk\nchicken\nproduce\nbread\ncoffee\nbeer";
    runList();
  });
  $("#pref").addEventListener("input", (e) => {
    $("#pref-out").textContent = prefLabel(Number(e.target.value));
    if ($("#list-result").children.length) runList();
  });
  $("#cmp-cat").addEventListener("change", renderCompare);
  $("#cmp-thin").addEventListener("change", renderCompare);
  $("#cmp-min").addEventListener("input", (e) => {
    $("#cmp-min-out").textContent = e.target.value === "0" ? "any" : e.target.value + "+";
    renderCompare();
  });
  for (const th of document.querySelectorAll("#cmp-table th")) {
    th.addEventListener("click", () => {
      const k = th.dataset.sort;
      cmpSort = { key: k, dir: cmpSort.key === k ? -cmpSort.dir : (k === "store" ? 1 : -1) };
      renderCompare();
    });
  }
  $("#store-pick").addEventListener("change", () => { $("#branch-pick").value = ""; renderStore(); });
  $("#branch-pick").addEventListener("change", renderStore);
  let t = null;
  $("#item-q").addEventListener("input", () => { clearTimeout(t); t = setTimeout(renderItems, 120); });
  $("#item-store").addEventListener("change", renderItems);

  const start = location.hash.slice(1);
  const known = ["list", "compare", "store", "items", "method"];
  switchView(known.includes(start) ? start : "list");
  if (start === "compare") renderCompare();
  if (start === "store") renderStore();
  if (start === "items") renderItems();
}

boot();
