/* Where to buy groceries — Cambridge, MA.
 *
 * Compiled to docs/app.js by `npm run build`. Edit this, never the output.
 *
 * Every string in the payload except the object keys is text somebody typed
 * on Reddit. It is set with textContent and never with innerHTML, and the one
 * attribute built from it is a permalink, checked to be a reddit path first.
 * `tests/site_smoke.js` reruns the whole app against a poisoned payload to
 * keep that true. Do not "simplify" either of those.
 */

let DATA: Payload | null = null;
let cmpSort: { key: string; dir: number } = { key: "w", dir: -1 };
let mapDrawn = false;

/** The payload after boot. Every view runs behind `boot()` resolving. */
function data(): Payload {
  if (!DATA) throw new Error("data() before boot");
  return DATA;
}

/* ── tiny DOM helpers ───────────────────────────────────────────────── */

type Props = Record<string, string | number | null | undefined>;
type Kid = Node | string | null | undefined;

function el(tag: string, props: Props = {}, kids: Kid | Kid[] = []): HTMLElement {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === null || v === undefined) continue;
    if (k === "class") n.className = String(v);
    else if (k === "text") n.textContent = String(v);   // never innerHTML
    else n.setAttribute(k, String(v));
  }
  for (const kid of Array.isArray(kids) ? kids : [kids]) {
    if (kid === null || kid === undefined) continue;
    n.append(typeof kid === "string" ? document.createTextNode(kid) : kid);
  }
  return n;
}

function need<T extends HTMLElement>(sel: string): T {
  const n = document.querySelector<T>(sel);
  if (!n) throw new Error("missing element: " + sel);
  return n;
}
const all = <T extends Element>(sel: string): T[] =>
  Array.from(document.querySelectorAll<T>(sel));

function clear(node: Element): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/** Reddit permalinks only. A claim cannot smuggle in a javascript: URL. */
function redditUrl(path: string | undefined): string | null {
  return path && /^\/r\/[A-Za-z0-9_]+\//.test(path) ? "https://reddit.com" + path : null;
}

const fmt = (x: number): string => (x > 0 ? "+" : "") + x.toFixed(2);
const cls = (s: number): string => (s > 0.15 ? "pos" : s < -0.15 ? "neg" : "mid");
const titleCase = (s: string): string =>
  s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

/* ── scoring ────────────────────────────────────────────────────────── */

/* Enough evidence for a cell to be worth acting on. Below this the shrinkage
 * applied in stage 3 dominates and every number drifts toward zero. */
const MIN_W = 1.0;

/** Combine quality and price into one number. pref: 0 = cheapest, 1 = best. */
function value(cell: Cell, pref: number): number {
  if (cell.pl === undefined) return cell.s;      // nothing said about cost
  return pref * cell.s + (1 - pref) * -cell.pl;  // pl is negative for cheap
}

/** Split a shopping list into terms. Blank lines and stray commas ignored. */
function parseList(raw: string): string[] {
  const seen = new Set(
    raw.split(/[\n,;]+/).map((s) => s.trim().toLowerCase()).filter(Boolean),
  );
  return Array.from(seen).slice(0, 40);
}

/**
 * Resolve one list term to evidence, most specific first:
 *   1. an item the corpus actually discusses ("rotisserie chicken")
 *   2. the department the word belongs to ("milk" -> dairy)
 *   3. nothing — reported as unmatched rather than guessed at
 */
function resolve(term: string): Resolution {
  const perStore: Record<string, { cell: Cell; label: string }> = {};
  for (const [store, items] of Object.entries(data().items)) {
    for (const [name, cell] of Object.entries(items)) {
      if (name === term || name.includes(term) || term.includes(name)) {
        const prev = perStore[store];
        if (!prev || cell.w > prev.cell.w) perStore[store] = { cell, label: name };
      }
    }
  }
  if (Object.keys(perStore).length) return { kind: "item", term, perStore };

  const category = data().keywords[term];
  if (category) {
    const byStore: Record<string, { cell: Cell; label: string }> = {};
    for (const [store, cats] of Object.entries(data().stores)) {
      const cell = cats[category];
      if (cell) byStore[store] = { cell, label: category };
    }
    if (Object.keys(byStore).length) return { kind: "category", term, perStore: byStore };
  }
  return { kind: "none", term, perStore: {} };
}

function bestFor(r: Resolution, pref: number): { store: string; v: number; hit: { cell: Cell; label: string } } | null {
  let best: { store: string; v: number; hit: { cell: Cell; label: string } } | null = null;
  for (const [store, hit] of Object.entries(r.perStore)) {
    if (hit.cell.w < MIN_W) continue;
    const v = value(hit.cell, pref);
    if (!best || v > best.v) best = { store, v, hit };
  }
  return best;
}

/** Rank stores across the whole list. */
function rankStores(terms: string[], pref: number): { ranked: Ranked[]; resolved: Resolution[] } {
  const resolved = terms.map(resolve);
  const tally: Record<string, Ranked> = {};
  for (const store of Object.keys(data().stores)) {
    tally[store] = { store, score: 0, covered: 0, avg: 0, wins: [], weak: [] };
  }
  for (const r of resolved) {
    if (r.kind === "none") continue;
    for (const [store, hit] of Object.entries(r.perStore)) {
      const t = tally[store];
      if (!t || hit.cell.w < MIN_W) continue;
      const v = value(hit.cell, pref);
      t.score += v;
      t.covered += 1;
      if (v < -0.15) t.weak.push(r.term);
    }
    const best = bestFor(r, pref);
    if (best) tally[best.store]?.wins.push(r.term);
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

function prefLabel(v: number): string {
  if (v <= 15) return "cheapest";
  if (v <= 40) return "mostly price";
  if (v < 60) return "balanced";
  if (v < 85) return "mostly quality";
  return "best quality";
}

function runList(): void {
  const terms = parseList(need<HTMLTextAreaElement>("#list-input").value);
  const out = need("#list-result");
  const detail = need("#list-detail");
  clear(out);
  clear(detail);
  if (!terms.length) {
    out.append(el("p", { class: "empty", text: "Add a few items and press the button." }));
    return;
  }
  const pref = Number(need<HTMLInputElement>("#pref").value) / 100;
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
      el("span", { class: "rank", text: fmt(t.avg) }),
    ]));
    card.append(el("p", { class: "why",
      text: `evidence for ${t.covered} of ${terms.length} item${terms.length === 1 ? "" : "s"}` }));
    const badges = el("div", { class: "badges" });
    for (const w of t.wins.slice(0, 6)) {
      badges.append(el("span", { class: "badge good", text: "best for " + w }));
    }
    for (const w of t.weak.slice(0, 3)) {
      badges.append(el("span", { class: "badge bad", text: "weak on " + w }));
    }
    if (badges.children.length) card.append(badges);
    card.addEventListener("click", () => showStore(t.store));
    out.append(card);
  });

  if (unmatched.length) {
    out.append(el("p", { class: "muted", text: "No evidence either way: " + unmatched.join(", ") }));
  }

  // Per-item breakdown: the split shop, if you will make two stops.
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
    const best = bestFor(r, pref);
    if (!best) continue;
    const tr = el("tr", {}, [
      el("td", { text: r.term }),
      el("td", { class: "muted",
        text: r.kind === "item" ? best.hit.label : titleCase(best.hit.label) }),
      el("td", { text: best.store }),
      el("td", { class: "num" }, el("span", { class: "score " + cls(best.v), text: fmt(best.v) })),
      el("td", { class: "num muted", text: String(best.hit.cell.n) }),
    ]);
    tr.addEventListener("click", () => showStore(best.store));
    body.append(tr);
  }
  table.append(body);
  detail.append(table);
}

/* ── view: compare ──────────────────────────────────────────────────── */

interface CmpRow {
  store: string;
  price: number | undefined;
  quality: number | undefined;
  sentiment: number;
  n: number;
  w: number;
}

function compareRows(): CmpRow[] {
  const cat = need<HTMLSelectElement>("#cmp-cat").value;
  const minW = Number(need<HTMLInputElement>("#cmp-min").value);
  const hideThin = need<HTMLInputElement>("#cmp-thin").checked;
  const rows: CmpRow[] = [];
  for (const [store, cats] of Object.entries(data().stores)) {
    const totals = data().totals[store] ?? { n: 0, w: 0, s: 0, thin: true };
    if (hideThin && totals.thin) continue;
    const cell = cat ? cats[cat] : undefined;
    if (cat && !cell) continue;
    const price = cell ? cell.pl : cats["price_overall"]?.pl;
    const quality = cell ? cell.s : cats["quality_overall"]?.s;
    const n = cell ? cell.n : totals.n;
    const w = cell ? cell.w : totals.w;
    if (w < minW) continue;
    rows.push({ store, price, quality, sentiment: cell ? cell.s : totals.s, n, w });
  }
  const k = cmpSort.key;
  rows.sort((a, b) => {
    if (k === "store") return cmpSort.dir * a.store.localeCompare(b.store);
    const av = (a as unknown as Record<string, number | undefined>)[k];
    const bv = (b as unknown as Record<string, number | undefined>)[k];
    if (av === undefined) return 1;
    if (bv === undefined) return -1;
    return cmpSort.dir * (av - bv);
  });
  return rows;
}

function renderCompare(): void {
  const rows = compareRows();
  const body = need("#cmp-table tbody");
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
    tr.addEventListener("click", () => showStore(r.store));
    body.append(tr);
  }
  renderScatter(rows);
}

const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag: string, attrs: Record<string, string | number>): SVGElement {
  const n = document.createElementNS(SVGNS, tag) as SVGElement;
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  return n;
}

function renderScatter(rows: CmpRow[]): void {
  const svg = need<HTMLElement>("#scatter");
  clear(svg);
  const W = 640, H = 420, m = { t: 22, r: 20, b: 44, l: 52 };
  const pts = rows.filter(
    (r): r is CmpRow & { price: number; quality: number } =>
      r.price !== undefined && r.quality !== undefined,
  );
  const x = (v: number): number => m.l + ((-v + 1) / 2) * (W - m.l - m.r); // left = pricier
  const y = (v: number): number => m.t + ((1 - v) / 2) * (H - m.t - m.b);  // top  = better

  for (const v of [-0.5, 0, 0.5]) {
    svg.append(svgEl("line", { class: v === 0 ? "axis" : "gridline",
      x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }));
    svg.append(svgEl("line", { class: v === 0 ? "axis" : "gridline",
      x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
  }
  const label = (t: string, px: number, py: number, klass: string, anchor = "middle"): void => {
    const n = svgEl("text", { class: klass, x: px, y: py, "text-anchor": anchor });
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
    const dot = svgEl("circle", { class: "dot", cx: x(p.price), cy: y(p.quality), r });
    const t = svgEl("title", {});
    t.textContent = `${p.store} — ${p.n.toLocaleString()} claims`;
    dot.append(t);
    dot.addEventListener("click", () => showStore(p.store));
    svg.append(dot);
    const name = svgEl("text", { class: "dot-label", x: x(p.price),
      y: y(p.quality) - r - 4, "text-anchor": "middle" });
    name.textContent = p.store;
    svg.append(name);
  }
  if (!pts.length) {
    label("no store has both price and quality evidence here", W / 2, H / 2, "axis-label");
  }
}

/* ── view: map ──────────────────────────────────────────────────────── */

/* Leaflet is loaded from docs/vendor/, not a CDN, and typed by
 * @types/leaflet, which declares the global `L`. It is the only third-party
 * code on the page; tiles are the only third-party request the browser makes.
 * `typeof L` is checked at runtime in renderMap, so a failed script tag
 * degrades to a message rather than a crash. */
let map: L.Map | null = null;
let markerLayer: L.LayerGroup | null = null;

/** Colour a pin by how the store scores, so the map carries the verdict. */
function pinColour(sentiment: number | undefined): string {
  if (sentiment === undefined) return "#8a8a8a";
  if (sentiment > 0.15) return "#1f8a5f";
  if (sentiment < -0.15) return "#c0392b";
  return "#a0862a";
}

function placeScore(place: Place): { cell: Cell | undefined; label: string } {
  const branchCats = place.branch
    ? data().branches[place.store]?.[place.branch]
    : undefined;
  const chainCats = data().stores[place.store];
  const cat = need<HTMLSelectElement>("#map-cat").value;
  if (cat) {
    const cell = branchCats?.[cat] ?? chainCats?.[cat];
    return { cell, label: cat };
  }
  return { cell: undefined, label: "" };
}

function renderMap(): void {
  const wrap = need("#map");
  if (typeof L === "undefined") {
    wrap.textContent = "Map library failed to load.";
    return;
  }
  if (!map) {
    map = L.map("map", { scrollWheelZoom: false }).setView([42.3736, -71.1097], 12);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);
    markerLayer = L.layerGroup().addTo(map);
  }
  if (!markerLayer) return;
  markerLayer.clearLayers();

  const wanted = need<HTMLSelectElement>("#map-store").value;
  const onlyEvidence = need<HTMLInputElement>("#map-evidence").checked;
  const cat = need<HTMLSelectElement>("#map-cat").value;
  let shown = 0;

  for (const place of data().places) {
    if (wanted && place.store !== wanted) continue;
    if (onlyEvidence && !place.branch) continue;
    const totals = data().totals[place.store];
    const { cell } = placeScore(place);
    const sentiment = cat ? cell?.s : totals?.s;
    if (cat && cell === undefined) continue;

    const marker = L.circleMarker([place.lat, place.lon], {
      radius: 7,
      color: "#fff",
      weight: 1.5,
      fillColor: pinColour(sentiment),
      fillOpacity: 0.9,
    });

    // Popups take an HTML string, so build a detached node instead and let
    // Leaflet adopt it — the address and branch are OSM data, and the claim
    // text below is Reddit's. Neither is trusted enough to interpolate.
    const pop = el("div", { class: "pin" });
    pop.append(el("strong", { text: place.store }));
    if (place.branch) pop.append(el("div", { class: "muted", text: place.branch + " branch" }));
    const where = [place.address, place.city].filter(Boolean).join(", ");
    if (where) pop.append(el("div", { class: "muted", text: where }));
    if (sentiment !== undefined) {
      pop.append(el("div", {}, el("span", { class: "score " + cls(sentiment),
        text: fmt(sentiment) + (cat ? " " + titleCase(cat) : " overall") })));
    }
    if (cell) pop.append(el("div", { class: "muted", text: `${cell.n} claims` }));
    const link = el("button", { class: "linkish", type: "button", text: "see the evidence →" });
    link.addEventListener("click", () => showStore(place.store, place.branch));
    pop.append(link);
    marker.bindPopup(pop);

    marker.addTo(markerLayer);
    shown += 1;
  }

  need("#map-count").textContent =
    `${shown} location${shown === 1 ? "" : "s"} of ${data().places.length}`;
  const attribution = data().places_attribution;
  need("#map-attrib").textContent = attribution ? "Locations: " + attribution : "";
  mapDrawn = true;
  // Leaflet measures the container on creation; if that happened while the
  // tab was hidden the tiles lay out against a zero-height box.
  setTimeout(() => map?.invalidateSize(), 0);
}

/* ── view: store detail ─────────────────────────────────────────────── */

function cellBlock(name: string, cell: Cell): HTMLElement {
  const d = el("details", { class: "cell" });
  const sum = el("summary", {}, [
    el("span", { class: "cat", text: titleCase(name) }),
    el("span", { class: "score " + cls(cell.s), text: fmt(cell.s) }),
    el("span", { class: "muted", text: `${cell.n} claim${cell.n === 1 ? "" : "s"}` }),
  ]);
  if (cell.p) {
    sum.append(el("span", {
      class: "badge " + (cell.p === "cheap" ? "good" : cell.p === "expensive" ? "warn" : ""),
      text: cell.p,
    }));
  }
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

function showStore(store: string, branch?: string): void {
  switchView("store");
  need<HTMLSelectElement>("#store-pick").value = store;
  renderStore(branch);
}

function renderStore(wantBranch?: string): void {
  const store = need<HTMLSelectElement>("#store-pick").value;
  const pick = need<HTMLSelectElement>("#branch-pick");
  const branches = data().branches[store] ?? {};
  const chosen = wantBranch ?? pick.value;
  clear(pick);
  pick.append(el("option", { value: "", text: "all branches (chain level)" }));
  for (const b of Object.keys(branches).sort()) {
    pick.append(el("option", { value: b, text: b }));
  }
  pick.value = chosen && branches[chosen] ? chosen : "";

  const body = need("#store-body");
  clear(body);
  const totals = data().totals[store];
  const cats = pick.value ? branches[pick.value] ?? {} : data().stores[store] ?? {};

  const head = el("div", { class: "card" });
  head.append(el("h3", {}, [
    el("span", { text: pick.value ? `${store} — ${pick.value}` : store }),
    el("span", { class: "rank", text: totals ? fmt(totals.s) : "" }),
  ]));
  const nBranch = Object.keys(branches).length;
  head.append(el("p", { class: "why",
    text: `${(totals?.n ?? 0).toLocaleString()} claims across the chain, `
        + `${nBranch} branch${nBranch === 1 ? "" : "es"} with their own evidence` }));
  if (totals?.thin) {
    head.append(el("p", { class: "badges" },
      el("span", { class: "badge warn", text: "thin evidence — treat with caution" })));
  }
  const pins = data().places.filter((p) => p.store === store);
  if (pins.length) {
    const go = el("button", { class: "linkish", type: "button",
      text: `show ${pins.length} location${pins.length === 1 ? "" : "s"} on the map →` });
    go.addEventListener("click", () => {
      need<HTMLSelectElement>("#map-store").value = store;
      switchView("map");
      renderMap();
    });
    head.append(go);
  }
  body.append(head);

  const ordered = Object.entries(cats).sort((a, b) => b[1].w - a[1].w);
  if (!ordered.length) {
    body.append(el("p", { class: "empty", text: "No claims above the evidence threshold." }));
  }
  body.append(el("h2", { text: "By category" }));
  for (const [name, cell] of ordered) body.append(cellBlock(name, cell));

  const items = data().items[store] ?? {};
  const topItems = Object.entries(items).sort((a, b) => b[1].w - a[1].w).slice(0, 20);
  if (topItems.length) {
    body.append(el("h2", { text: "Specific items people mention" }));
    for (const [name, cell] of topItems) body.append(cellBlock(name, cell));
  }

  const regions = data().regions[store] ?? {};
  if (Object.keys(regions).length) {
    body.append(el("h2", { text: "Mentioned by area, not by branch" }));
    body.append(el("p", { class: "muted",
      text: "These name a region rather than a store, so they are listed apart: "
          + Object.keys(regions).sort().join(", ") }));
  }
}

/* ── view: item search ──────────────────────────────────────────────── */

function renderItems(): void {
  const q = need<HTMLInputElement>("#item-q").value.trim().toLowerCase();
  const onlyStore = need<HTMLSelectElement>("#item-store").value;
  const body = need("#item-body");
  clear(body);

  const hits: { store: string; name: string; cell: Cell }[] = [];
  for (const [store, items] of Object.entries(data().items)) {
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
      el("td", { class: "num muted", text: h.cell.p ?? "–" }),
      el("td", { class: "num muted", text: String(h.cell.n) }),
    ]);
    tr.addEventListener("click", () => showStore(h.store));
    tb.append(tr);
  }
  table.append(tb);
  body.append(table);
  if (hits.length > 250) {
    body.append(el("p", { class: "muted",
      text: `Showing the 250 best-evidenced of ${hits.length}. Narrow the search to see the rest.` }));
  }
}

/* ── boot ───────────────────────────────────────────────────────────── */

const VIEWS = ["list", "compare", "map", "store", "items", "method"] as const;
type View = (typeof VIEWS)[number];

function switchView(name: View): void {
  for (const b of all<HTMLElement>(".tabs button")) {
    b.classList.toggle("on", b.dataset["view"] === name);
  }
  for (const v of all<HTMLElement>(".view")) v.hidden = v.id !== "view-" + name;
  if (location.hash.slice(1) !== name) history.replaceState(null, "", "#" + name);
}

function renderView(name: View): void {
  if (name === "compare") renderCompare();
  if (name === "store") renderStore();
  if (name === "items") renderItems();
  if (name === "map" && !mapDrawn) renderMap();
  else if (name === "map") setTimeout(() => map?.invalidateSize(), 0);
}

function fillMethod(): void {
  const c = data().corpus;
  const dl = need("#m-prov");
  const pairs: [string, string | number][] = [
    ["Generated", data().generated_at],
    ["Documents extracted", (c?.documents_extracted ?? 0).toLocaleString()],
    ["Candidate documents", (c?.working_set ?? 0).toLocaleString()],
    ["Stores covered", Object.keys(data().stores).length],
    ["Branches", Object.values(data().branches).reduce((a, b) => a + Object.keys(b).length, 0)],
    ["Items indexed", Object.values(data().items).reduce((a, b) => a + Object.keys(b).length, 0)],
    ["Mapped locations", data().places.length],
    ["Shrinkage constant", data().method.shrinkage_k],
    ["Default half-life", data().method.default_half_life_years + " years"],
    ["Transient claims", data().method.transient_claims],
  ];
  for (const [k, v] of pairs) dl.append(el("dt", { text: k }), el("dd", { text: String(v) }));
}

async function boot(): Promise<void> {
  try {
    const res = await fetch("verdicts.json");
    if (!res.ok) throw new Error("HTTP " + res.status);
    DATA = (await res.json()) as Payload;
  } catch (err) {
    need("#loading").textContent =
      "Could not load the data: " + (err instanceof Error ? err.message : String(err));
    return;
  }
  need("#loading").hidden = true;

  const stores = Object.keys(data().stores).sort();
  for (const s of stores) {
    need("#store-pick").append(el("option", { value: s, text: s }));
    need("#item-store").append(el("option", { value: s, text: s }));
  }
  const mapStores = Array.from(new Set(data().places.map((p) => p.store))).sort();
  for (const s of mapStores) need("#map-store").append(el("option", { value: s, text: s }));
  for (const c of data().categories) {
    need("#cmp-cat").append(el("option", { value: c, text: titleCase(c) }));
    need("#map-cat").append(el("option", { value: c, text: titleCase(c) }));
  }

  const corpus = data().corpus;
  need("#corpus-line").textContent =
    `${(corpus?.documents_extracted ?? 0).toLocaleString()} Reddit posts and comments`;
  const footer = need("#footer-line");
  footer.textContent = `Generated ${data().generated_at} · opinion aggregated from Reddit, not verified prices · `;
  footer.append(el("a", { href: "https://github.com/pelednoam/find-groceries" }, "source"));

  const dates: string[] = [];
  for (const cats of Object.values(data().stores)) {
    for (const cell of Object.values(cats)) for (const e of cell.e) dates.push(e.d);
  }
  if (dates.length) {
    dates.sort();
    need("#m-span").textContent = `${dates[0]} to ${dates[dates.length - 1]}`;
  }
  fillMethod();

  for (const b of all<HTMLElement>("[data-view]")) {
    b.addEventListener("click", () => {
      const v = b.dataset["view"] as View;
      switchView(v);
      renderView(v);
    });
  }
  need("#list-go").addEventListener("click", runList);
  need("#list-demo").addEventListener("click", () => {
    need<HTMLTextAreaElement>("#list-input").value = "milk\nchicken\nproduce\nbread\ncoffee\nbeer";
    runList();
  });
  need("#pref").addEventListener("input", (e) => {
    const v = Number((e.target as HTMLInputElement).value);
    need("#pref-out").textContent = prefLabel(v);
    if (need("#list-result").children.length) runList();
  });
  need("#cmp-cat").addEventListener("change", renderCompare);
  need("#cmp-thin").addEventListener("change", renderCompare);
  need("#cmp-min").addEventListener("input", (e) => {
    const v = (e.target as HTMLInputElement).value;
    need("#cmp-min-out").textContent = v === "0" ? "any" : v + "+";
    renderCompare();
  });
  for (const th of all<HTMLElement>("#cmp-table th")) {
    th.addEventListener("click", () => {
      const k = th.dataset["sort"];
      if (!k) return;
      cmpSort = { key: k, dir: cmpSort.key === k ? -cmpSort.dir : (k === "store" ? 1 : -1) };
      renderCompare();
    });
  }
  need("#map-store").addEventListener("change", renderMap);
  need("#map-cat").addEventListener("change", renderMap);
  need("#map-evidence").addEventListener("change", renderMap);
  need("#store-pick").addEventListener("change", () => {
    need<HTMLSelectElement>("#branch-pick").value = "";
    renderStore();
  });
  need("#branch-pick").addEventListener("change", () => renderStore());
  let t: ReturnType<typeof setTimeout> | null = null;
  need("#item-q").addEventListener("input", () => {
    if (t) clearTimeout(t);
    t = setTimeout(renderItems, 120);
  });
  need("#item-store").addEventListener("change", renderItems);

  const start = location.hash.slice(1);
  const view: View = (VIEWS as readonly string[]).includes(start) ? (start as View) : "list";
  switchView(view);
  if (view !== "list") renderView(view);
}

void boot();
