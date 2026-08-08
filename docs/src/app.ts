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

/** The shopping list. Quantities weight how much each item matters to the
 *  ranking — two of something you buy every week should count for more than
 *  one of something you buy once. */
interface BasketItem { name: string; qty: number; }
let basket: BasketItem[] = [
  { name: "milk", qty: 1 }, { name: "chicken", qty: 2 }, { name: "produce", qty: 1 },
  { name: "bread", qty: 1 }, { name: "coffee", qty: 1 },
];
let sortMode: "value" | "quality" | "evidence" = "value";

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

/** Make a non-button element behave like one for keyboard users.
 *
 * Cards, table rows and scatter dots all navigate on click. Without this they
 * are invisible to anyone not using a mouse — which is most of the ways this
 * page gets read.
 */
function activatable(node: Element, label: string, run: () => void): void {
  node.setAttribute("tabindex", "0");
  node.setAttribute("role", "button");
  node.setAttribute("aria-label", label);
  node.addEventListener("click", run);
  node.addEventListener("keydown", (e) => {
    const k = (e as KeyboardEvent).key;
    if (k === "Enter" || k === " ") { e.preventDefault(); run(); }
  });
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

/** Combine quality and price into one number. pref: 0 = cheapest, 1 = best.
 *
 * A cell with no price evidence used to return its full sentiment, so at
 * "cheapest" it competed at full strength against cells actually being
 * judged on price. It now contributes only the share the reader asked for,
 * which is the honest answer: we know nothing about its cost.
 */
function value(cell: Cell, pref: number): number {
  if (cell.pl === undefined) return pref * cell.s;
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
        // Specificity first, evidence second. An exact match beats a
        // substring however much weight the substring carries, so asking
        // for "key limes" is not answered by "limes".
        const rank = (n: string): number => (n === term ? 2 : n.includes(term) ? 1 : 0);
        const better = !prev
          || rank(name) > rank(prev.label)
          || (rank(name) === rank(prev.label) && cell.w > prev.cell.w);
        if (better) perStore[store] = { cell, label: name };
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

/** Rank stores across the whole basket.
 *
 * Quantity is a weight, not a multiplier on a price: there are no prices
 * here. Asking for two of something says it matters twice as much to you,
 * so a store's score on it counts twice.
 */
function rankStores(items: BasketItem[], pref: number): { ranked: Ranked[]; resolved: Resolution[] } {
  const qty = new Map(items.map((i) => [i.name, i.qty]));
  const resolved = items.map((i) => resolve(i.name));
  const tally: Record<string, Ranked> = {};
  for (const store of Object.keys(data().stores)) {
    tally[store] = { store, score: 0, covered: 0, avg: 0, wins: [], weak: [] };
  }
  for (const r of resolved) {
    if (r.kind === "none") continue;
    const w = qty.get(r.term) ?? 1;
    for (const [store, hit] of Object.entries(r.perStore)) {
      const t = tally[store];
      if (!t || hit.cell.w < MIN_W) continue;
      const v = value(hit.cell, pref);
      t.score += v * w;
      t.covered += w;
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

function basketCount(): string {
  const items = basket.length;
  const units = basket.reduce((a, b) => a + b.qty, 0);
  return `${items} item${items === 1 ? "" : "s"}${units > items ? ` · ${units} units` : ""}`;
}

function addToBasket(name: string): void {
  const clean = name.trim().toLowerCase();
  if (!clean) return;
  const found = basket.find((i) => i.name === clean);
  if (found) found.qty += 1;
  else basket.push({ name: clean, qty: 1 });
  renderBasket();
  runList();
}

function renderBasket(): void {
  const ul = need("#basket-list");
  clear(ul);
  need("#basket-count").textContent = basketCount();

  for (const item of basket) {
    const resolved = resolve(item.name);
    const li = el("li");
    const name = el("span", { class: "name" });
    name.append(el("span", { text: item.name }));
    // Say what the term was actually matched against, so a shopper can see
    // that "milk" was answered by the dairy aisle rather than by milk.
    const label = resolved.kind === "none" ? "no evidence either way"
      : resolved.kind === "category" ? "matched to " + titleCase(
          Object.values(resolved.perStore)[0]?.label ?? "")
      : "matched to an item people named";
    name.append(el("span", { class: "matched", text: label }));
    li.append(name);
    li.append(el("span", { class: "qty", text: "×" + item.qty }));

    const dec = el("button", { class: "step", type: "button",
      "aria-label": `One fewer ${item.name}` }, "−");
    dec.addEventListener("click", () => {
      item.qty = Math.max(1, item.qty - 1); renderBasket(); runList();
    });
    const inc = el("button", { class: "step", type: "button",
      "aria-label": `One more ${item.name}` }, "+");
    inc.addEventListener("click", () => { item.qty += 1; renderBasket(); runList(); });
    const drop = el("button", { class: "drop", type: "button",
      "aria-label": `Remove ${item.name}` }, "×");
    drop.addEventListener("click", () => {
      basket = basket.filter((i) => i !== item); renderBasket(); runList();
    });
    li.append(dec, inc, drop);
    ul.append(li);
  }

  // Suggestions come from the keyword map, so every one of them resolves.
  const chips = need("#suggestions");
  clear(chips);
  const pool = Object.keys(data().keywords)
    .filter((w) => !basket.some((i) => i.name === w))
    .slice(0, 40);
  for (const word of pool.filter((_, i) => i % 5 === 0).slice(0, 7)) {
    const chip = el("button", { class: "chip-add", type: "button", text: "+ " + word });
    chip.addEventListener("click", () => addToBasket(word));
    chips.append(chip);
  }
}

function prefLabelShort(v: number): string {
  return prefLabel(v);
}

function runList(): void {
  const out = need("#list-result");
  const detail = need("#list-detail");
  clear(out);
  clear(detail);
  if (!basket.length) {
    out.append(el("p", { class: "empty",
      text: "Add a few things and Basket will rank the stores on them." }));
    return;
  }
  const pref = Number(need<HTMLInputElement>("#pref").value) / 100;
  const minW = Number(need<HTMLInputElement>("#cmp-min").value);
  const hideThin = need<HTMLInputElement>("#cmp-thin").checked;
  const { ranked, resolved } = rankStores(basket, pref);
  const unmatched = resolved.filter((r) => r.kind === "none").map((r) => r.term);
  const units = basket.reduce((a, b) => a + b.qty, 0);

  let rows = ranked.filter((t) => {
    const totals = data().totals[t.store];
    if (hideThin && totals?.thin) return false;
    return (totals?.w ?? 0) >= minW;
  });
  if (sortMode === "quality") {
    rows = [...rows].sort((a, b) => (data().totals[b.store]?.s ?? 0) - (data().totals[a.store]?.s ?? 0));
  } else if (sortMode === "evidence") {
    rows = [...rows].sort((a, b) => (data().totals[b.store]?.w ?? 0) - (data().totals[a.store]?.w ?? 0));
  }

  need("#results-title").textContent =
    `${rows.length} store${rows.length === 1 ? "" : "s"}, scored on your basket`;

  if (!rows.length) {
    out.append(el("p", { class: "empty",
      text: "Nothing on that list matches anything the corpus discusses." }));
    return;
  }

  const table = el("div", { class: "table" });
  table.append(el("div", { class: "thead grid-row" }, [
    el("div", { text: "Store" }),
    el("div", { text: "On your list" }),
    el("div", { text: "Price" }),
    el("div", { text: "Covered" }),
    el("div", { text: "Overall" }),
  ]));

  rows.forEach((t, i) => {
    const totals = data().totals[t.store];
    const price = data().stores[t.store]?.["price_overall"];
    const row = el("button", {
      class: "trow grid-row" + (i === 0 ? " top" : ""), type: "button",
    });

    const cell = el("div", { class: "store-cell" });
    cell.append(el("span", { class: "rank-badge", text: String(i + 1) }));
    const names = el("span", { style: "min-width:0" });
    names.append(el("span", { class: "store-name", text: t.store }));
    names.append(el("span", { class: "store-area",
      text: `${(totals?.n ?? 0).toLocaleString()} claims` }));
    cell.append(names);
    row.append(cell);

    const score = el("div");
    score.append(el("span", { class: "big-figure score " + cls(t.avg), text: fmt(t.avg) }));
    score.append(el("span", { class: "sub-figure muted",
      text: t.wins.length ? "best for " + t.wins.slice(0, 2).join(", ") : "—" }));
    row.append(score);

    row.append(el("div", {}, price?.p
      ? el("span", { class: "cell-num score " + (price.p === "cheap" ? "pos" : price.p === "expensive" ? "neg" : "mid"),
          text: price.p })
      : el("span", { class: "cell-num muted", text: "—" })));

    const cover = el("div");
    cover.append(el("span", { class: "cell-num", text: `${t.covered} of ${units}` }));
    cover.append(el("span", { class: "sub-figure muted",
      text: t.weak.length ? "weak on " + t.weak[0] : "no gaps found" }));
    row.append(cover);

    row.append(el("div", {}, el("span", {
      class: "cell-num score " + cls(totals?.s ?? 0), text: fmt(totals?.s ?? 0) })));

    row.addEventListener("click", () => openDetail(t.store));
    table.append(row);
  });
  out.append(table);

  if (unmatched.length) {
    out.append(el("p", { class: "hidden-note",
      text: "No evidence either way: " + unmatched.join(", ") }));
  }

  // The two notes from the comp: a verdict, and whether a second stop pays.
  const best = rows[0];
  const second = rows[1];
  const notes = el("div", { class: "notes" });
  if (best) {
    const totals = data().totals[best.store];
    const note = el("div", { class: "note amber" });
    note.append(el("h3", { text: `Shop at ${best.store} this week` }));
    note.append(el("p", { text:
      `Scores ${fmt(best.avg)} across your basket with evidence for `
      + `${best.covered} of ${units} units, on ${(totals?.n ?? 0).toLocaleString()} `
      + `claims. ${totals?.thin ? "Treat it carefully — the evidence is thin." : ""}` }));
    notes.append(note);
  }
  if (best && second) {
    const note = el("div", { class: "note" });
    note.append(el("h3", { text: "Worth a second stop?" }));
    const gap = best.avg - second.avg;
    note.append(el("p", { text: best.weak.length
      ? `${best.store} reads weak on ${best.weak.slice(0, 2).join(" and ")}. `
        + `${second.store} is the next best overall — worth the detour if those matter this week.`
      : gap < 0.08
        ? `Not really. ${second.store} is within ${fmt(gap)} of ${best.store} on the same basket, `
          + `so a second stop buys you very little.`
        : `Probably not. ${best.store} leads ${second.store} by ${fmt(gap)} across your list.` }));
    notes.append(note);
  }
  if (notes.children.length) detail.append(notes);

  // Item by item, so the split shop is visible.
  detail.append(el("h2", { text: "Item by item", style: "font-size:20px;margin:26px 0 12px" }));
  const per = el("div", { class: "table" });
  per.append(el("div", { class: "thead grid-row",
    style: "grid-template-columns:1.4fr 1.2fr 1fr .8fr" }, [
    el("div", { text: "Item" }), el("div", { text: "Matched to" }),
    el("div", { text: "Best here" }), el("div", { text: "Score" }),
  ]));
  for (const r of resolved) {
    if (r.kind === "none") continue;
    const bestHit = bestFor(r, pref);
    if (!bestHit) continue;
    const tr = el("button", { class: "trow grid-row", type: "button",
      style: "grid-template-columns:1.4fr 1.2fr 1fr .8fr" }, [
      el("div", { text: r.term }),
      el("div", { class: "muted",
        text: r.kind === "item" ? bestHit.hit.label : titleCase(bestHit.hit.label) }),
      el("div", { text: bestHit.store }),
      el("div", {}, el("span", { class: "score " + cls(bestHit.v), text: fmt(bestHit.v) })),
    ]);
    tr.addEventListener("click", () => openDetail(bestHit.store));
    per.append(tr);
  }
  detail.append(per);
}

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
  // The Price column shows -pl ("higher is cheaper"), so it must sort on the
  // number displayed. Sorting the raw level put the most expensive store at
  // the top of a column headed by a positive figure.
  const value = (r: CmpRow): number | undefined =>
    k === "price" ? (r.price === undefined ? undefined : -r.price)
      : k === "quality" ? r.quality
      : k === "sentiment" ? r.sentiment
      : k === "n" ? r.n
      : k === "w" ? r.w
      : undefined;
  rows.sort((a, b) => {
    if (k === "store") return cmpSort.dir * a.store.localeCompare(b.store);
    const av = value(a), bv = value(b);
    // Missing values sort last in either direction, and two missing values
    // compare equal — otherwise the comparator is not a valid ordering.
    if (av === undefined && bv === undefined) return a.store.localeCompare(b.store);
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
      el("td", { class: "num" }, (() => {
        const m = data().merged?.stores[r.store];
        return m && m.g !== undefined
          ? el("span", { class: "score " + cls(m.v), text: fmt(m.v) })
          : el("span", { class: "muted", text: "–" });
      })()),
      el("td", { class: "num muted", text: r.n.toLocaleString() }),
      el("td", { class: "num muted", text: r.w.toLocaleString() }),
    ]);
    activatable(tr, `${r.store} details`, () => showStore(r.store));
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
  // x() puts CHEAP on the right (price_level is negative for cheap), so a
  // bargain label belongs at a negative price level. These were mirrored:
  // "bargain" sat over the expensive half and "premium" over the cheap one.
  label("bargain", x(-0.55), y(0.85), "quad");
  label("premium", x(0.55), y(0.85), "quad");
  label("avoid", x(0.55), y(-0.85), "quad");

  for (const p of pts) {
    const r = Math.max(4, Math.min(16, Math.sqrt(p.w) / 1.6));
    const dot = svgEl("circle", { class: "dot", cx: x(p.price), cy: y(p.quality), r });
    const t = svgEl("title", {});
    t.textContent = `${p.store} — ${p.n.toLocaleString()} claims`;
    dot.append(t);
    activatable(dot, `${p.store}, ${p.n} claims`, () => showStore(p.store));
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

/* ── merged estimate ────────────────────────────────────────────────── */

/** Render a combined estimate so the inputs stay visible.
 *
 * A merged number that hides what went into it is a number the reader cannot
 * argue with, and here the mix varies enormously — Reddit holds 93% of the
 * weight at Market Basket Somerville and 13% at a branch with one claim. The
 * bar shows that split.
 */
function mergedBlock(m: MergedValue, label: string): HTMLElement {
  const box = el("div", { class: "merged" });
  const head = el("div", { class: "merged-head" }, [
    el("span", { class: "score " + cls(m.v), text: fmt(m.v) }),
    el("span", { class: "muted", text: `${label} · ±${m.se.toFixed(2)}` }),
  ]);
  if (m.conflict) {
    head.append(el("span", { class: "badge bad", text: "sources conflict" }));
  }
  box.append(head);

  if (m.r !== undefined && m.g !== undefined) {
    box.append(reconciliationRail(m.r, m.g, m.v));
    const bar = el("div", { class: "mixbar" });
    bar.append(el("span", { class: "mix reddit", style: `width:${m.share * 100}%` }));
    bar.append(el("span", { class: "mix google", style: `width:${(1 - m.share) * 100}%` }));
    box.append(bar);
    box.append(el("div", { class: "mixlab" }, [
      el("span", { text: `Reddit ${fmt(m.r)} · ${(m.share * 100).toFixed(0)}% of the weight` }),
      el("span", { text: `Google ${fmt(m.g)} · ${((1 - m.share) * 100).toFixed(0)}%` }),
    ]));
  } else {
    box.append(el("div", { class: "mixlab" },
      el("span", { text: m.r !== undefined ? "Reddit only" : "Google only" })));
  }
  return box;
}

/** Both sources and the combined figure on one -1..+1 scale.
 *
 * The number alone makes you do the comparison in your head; the rail does
 * it for you. The span between the two marks *is* the disagreement, which is
 * the thing this site refuses to hide.
 */
function reconciliationRail(reddit: number, google: number, combined: number): HTMLElement {
  const W = 320, H = 38, pad = 12;
  const x = (v: number): number => pad + ((v + 1) / 2) * (W - pad * 2);
  const y = 19;
  const wrap = el("div", { class: "rail" });
  // No preserveAspectRatio="none": it stretched the circles and the diamond
  // horizontally by whatever the container happened to be.
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`,
    role: "img", "aria-label":
      `Reddit ${fmt(reddit)}, reviews ${fmt(google)}, combined ${fmt(combined)}` });

  svg.append(svgEl("line", { class: "axis", x1: pad, x2: W - pad, y1: y, y2: y }));
  svg.append(svgEl("line", { class: "zero", x1: x(0), x2: x(0), y1: y - 9, y2: y + 9 }));
  // The span between the sources, drawn before the marks so they sit on it.
  svg.append(svgEl("line", { class: "span",
    x1: Math.min(x(reddit), x(google)), x2: Math.max(x(reddit), x(google)),
    y1: y, y2: y }));
  svg.append(svgEl("circle", { class: "tick-a", cx: x(reddit), cy: y, r: 5 }));
  svg.append(svgEl("circle", { class: "tick-b", cx: x(google), cy: y, r: 5 }));
  // The combined figure is a diamond, so it never reads as a third source.
  const d = svgEl("rect", { class: "combined", x: x(combined) - 5, y: y - 5,
    width: 10, height: 10, rx: 1.5,
    transform: `rotate(45 ${x(combined)} ${y})` });
  svg.append(d);
  for (const [v, label] of [[-1, "worse"], [1, "better"]] as [number, string][]) {
    const t = svgEl("text", { class: "scale-label", x: x(v), y: y + 15,
      "text-anchor": v < 0 ? "start" : "end" });
    t.textContent = label;
    svg.append(t);
  }
  wrap.append(svg);
  return wrap;
}

/* ── view: cross-check ──────────────────────────────────────────────── */

/** Which Google population the reader chose: all ratings, or only the ones
 *  with a paragraph. The two differ by 0.36 stars on average and by 1.15 for
 *  one chain, so the choice is the reader's rather than ours. */
function googleValue(r: Rating): number | null {
  switch (need<HTMLSelectElement>("#cross-pop").value) {
    case "norm_long": return r.norm_long;
    case "norm_recent": return r.norm_recent;
    default: return r.norm;
  }
}

interface CrossRow {
  store: string; reddit: number; google: number; gap: number; r: Rating;
  /** the star mean for the population the reader chose, not always all-time */
  stars: number;
}

function crossRows(): CrossRow[] {
  const cc = data().crosscheck;
  if (!cc) return [];
  const rows: CrossRow[] = [];
  for (const [store, r] of Object.entries(cc.stores)) {
    const totals = data().totals[store];
    const google = googleValue(r);
    if (!totals || google === null || r.thin) continue;
    const key = need<HTMLSelectElement>("#cross-pop").value;
    const stars = key === "norm_long" ? (r.mean_long ?? r.mean)
      : key === "norm_recent" ? r.mean_recent
      : r.mean;
    rows.push({ store, reddit: totals.s, google, gap: google - totals.s, r, stars });
  }
  return rows.sort((a, b) => b.gap - a.gap);
}

function renderCross(): void {
  const cc = data().crosscheck;
  const body = need("#cross-table tbody");
  clear(body);
  if (!cc) {
    body.append(el("tr", {}, el("td", { colspan: "7", class: "empty",
      text: "No Google cross-check in this build." })));
    return;
  }
  const rows = crossRows();
  const mean = rows.length
    ? rows.reduce((a, b) => a + b.gap, 0) / rows.length : 0;
  need("#cross-summary").textContent =
    `${cc.n_reviews.toLocaleString()} ratings, ${cc.coverage} · `
    + `Google reads ${fmt(mean)} vs the corpus on average`;
  need("#cross-cite").textContent = cc.citation;

  for (const row of rows) {
    const tr = el("tr", {}, [
      el("td", { text: row.store }),
      el("td", { class: "num" }, el("span", { class: "score " + cls(row.reddit), text: fmt(row.reddit) })),
      el("td", { class: "num" }, el("span", { class: "score " + cls(row.google), text: fmt(row.google) })),
      el("td", { class: "num" }, el("span", {
        class: Math.abs(row.gap) >= 0.5 ? "gap-big" : "muted", text: fmt(row.gap) })),
      el("td", { class: "num muted", text: row.stars.toFixed(2) + "★" }),
      el("td", { class: "num muted", text: (
        need<HTMLSelectElement>("#cross-pop").value === "norm_long"
          ? row.r.n_long : row.r.n).toLocaleString() }),
      el("td", { class: "num muted", text: Math.round(row.r.n_eff).toLocaleString() }),
      el("td", { class: "num muted", text: row.r.median_date }),
    ]);
    activatable(tr, `${row.store} details`, () => showStore(row.store));
    body.append(tr);
  }
  renderCrossChart(rows);
}

/** A dumbbell per store: the two sources joined by a line, so the gap is the
 *  thing you see rather than a number you have to subtract. */
function renderCrossChart(rows: CrossRow[]): void {
  const svg = need<HTMLElement>("#cross-chart");
  clear(svg);
  const W = 640, m = { t: 30, r: 24, b: 16, l: 150 };
  const rowH = 23;
  const H = m.t + m.b + Math.max(1, rows.length) * rowH;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const x = (v: number): number => m.l + ((v + 1) / 2) * (W - m.l - m.r);

  for (const v of [-1, -0.5, 0, 0.5, 1]) {
    svg.append(svgEl("line", { class: v === 0 ? "axis" : "gridline",
      x1: x(v), x2: x(v), y1: m.t - 8, y2: H - m.b }));
    const t = svgEl("text", { class: "axis-label", x: x(v), y: m.t - 14,
      "text-anchor": "middle" });
    t.textContent = v.toFixed(1);
    svg.append(t);
  }
  rows.forEach((row, i) => {
    const y = m.t + i * rowH + rowH / 2;
    const name = svgEl("text", { class: "name", x: m.l - 10, y: y + 4, "text-anchor": "end" });
    name.textContent = row.store;
    svg.append(name);
    svg.append(svgEl("line", { class: "link", x1: x(row.reddit), x2: x(row.google), y1: y, y2: y }));
    const rd = svgEl("circle", { class: "reddit", cx: x(row.reddit), cy: y, r: 5 });
    const rt = svgEl("title", {}); rt.textContent = `Reddit ${fmt(row.reddit)}`; rd.append(rt);
    const gd = svgEl("circle", { class: "google", cx: x(row.google), cy: y, r: 5 });
    const gt = svgEl("title", {});
    gt.textContent = `Google ${fmt(row.google)} (${row.r.n.toLocaleString()} ratings)`;
    gd.append(gt);
    svg.append(rd, gd);
  });
  if (!rows.length) {
    const t = svgEl("text", { class: "axis-label", x: W / 2, y: H / 2, "text-anchor": "middle" });
    t.textContent = "no store has both a verdict and enough Google ratings";
    svg.append(t);
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
    // Say which level the figure came from. Falling back from branch to
    // chain silently labelled a chain-wide number as this branch's.
    const branchCell = branchCats?.[cat];
    if (branchCell) return { cell: branchCell, label: cat };
    const chainCell = chainCats?.[cat];
    if (chainCell) return { cell: chainCell, label: `${cat}, chain-wide` };
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
    const { cell, label: cellLabel } = placeScore(place);
    const rating = data().crosscheck?.locations[place.osm];
    const colourBy = need<HTMLSelectElement>("#map-colour").value;
    const byGoogle = colourBy === "google";
    const byMerged = colourBy === "merged";
    let sentiment = cat ? cell?.s : totals?.s;
    if (cat && cell === undefined) continue;
    const mergedHere = place.branch
      ? data().merged?.branches[place.store]?.[place.branch]
      : data().merged?.stores[place.store];
    if (byGoogle) {
      if (!rating || rating.thin) continue;
      sentiment = rating.norm;
    } else if (byMerged) {
      if (!mergedHere) continue;
      sentiment = mergedHere.v;
    }

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
        text: fmt(sentiment) + " "
            + (byGoogle ? "Google rating"
               : byMerged ? "combined"
               : cellLabel ? titleCase(cellLabel) : "overall") })));
    }
    if (cell) pop.append(el("div", { class: "muted", text: `${cell.n} claims` }));
    if (mergedHere && mergedHere.g !== undefined && mergedHere.r !== undefined) {
      const box = el("div", { class: "cross" });
      box.append(el("span", { class: "score " + cls(mergedHere.v),
        text: `${fmt(mergedHere.v)} combined` }));
      box.append(el("span", { class: "muted",
        text: ` · ${(mergedHere.share * 100).toFixed(0)}% Reddit` }));
      pop.append(box);
    }
    if (rating && !rating.thin) {
      const box = el("div", { class: "cross" });
      box.append(el("span", { class: "score " + cls(rating.norm),
        text: `${rating.mean.toFixed(2)}★ Google` }));
      box.append(el("span", { class: "muted",
        text: ` · ${rating.n.toLocaleString()} ratings to ${rating.last}` }));
      pop.append(box);
    }
    const link = el("button", { class: "linkish", type: "button", text: "see the evidence →" });
    link.addEventListener("click", () => showStore(place.store, place.branch));
    pop.append(link);
    marker.bindPopup(pop);

    marker.addTo(markerLayer);
    shown += 1;
  }

  const list = need("#map-list");
  clear(list);
  const seen = new Set<string>();
  for (const place of data().places) {
    if (wanted && place.store !== wanted) continue;
    const key = place.store + "|" + (place.branch ?? "");
    if (seen.has(key)) continue;
    seen.add(key);
    if (seen.size > 14) break;
    const totals = data().totals[place.store];
    const b = el("button", { type: "button" }, [
      el("span", { class: "grow" }, [
        el("span", { class: "t", text: place.store }),
        el("span", { class: "s",
          text: [place.branch, place.city].filter(Boolean).join(" · ") || place.address }),
      ]),
      el("span", { class: "score " + cls(totals?.s ?? 0), text: fmt(totals?.s ?? 0) }),
      el("span", { class: "chev", text: "›" }),
    ]);
    b.addEventListener("click", () => openDetail(place.store, place.branch));
    list.append(b);
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

/* ── detail drawer ──────────────────────────────────────────────────── */

let drawerOpen: HTMLElement[] = [];
let lastFocus: Element | null = null;

function closeDrawer(): void {
  for (const node of drawerOpen) node.remove();
  drawerOpen = [];
  if (lastFocus instanceof HTMLElement) lastFocus.focus();
}

/** Everything known about one store, in a panel over the page.
 *
 * The comp puts the store's whole story here — both sources, the quotes, and
 * how it does on the basket — rather than sending the reader to a separate
 * view and losing their place. */
function openDetail(store: string, branch?: string): void {
  closeDrawer();
  lastFocus = document.activeElement;
  const totals = data().totals[store];
  const cats = branch ? data().branches[store]?.[branch] ?? {} : data().stores[store] ?? {};
  const rating = data().crosscheck?.stores[store];
  const merged = branch
    ? data().merged?.branches[store]?.[branch]
    : data().merged?.stores[store];

  const scrim = el("button", { class: "scrim", type: "button", "aria-label": "Close" });
  scrim.addEventListener("click", closeDrawer);

  const panel = el("aside", { class: "drawer", role: "dialog", "aria-modal": "true",
    "aria-label": store, tabindex: "-1" });
  const close = el("button", { class: "drawer-close", type: "button",
    "aria-label": "Close" }, "×");
  close.addEventListener("click", closeDrawer);
  panel.append(close);

  const branches = Object.keys(data().branches[store] ?? {});
  panel.append(el("div", { class: "eyebrow",
    text: branch ?? `${branches.length} branch${branches.length === 1 ? "" : "es"} with their own evidence` }));
  panel.append(el("h2", { text: store }));

  const pair = el("div", { class: "stat-pair" });
  const a = el("div", { class: "stat" });
  a.append(el("div", { class: "k", text: "This corpus" }));
  a.append(el("div", { class: "v score " + cls(totals?.s ?? 0), text: fmt(totals?.s ?? 0) }));
  a.append(el("div", { class: "n", text: `${(totals?.n ?? 0).toLocaleString()} claims` }));
  pair.append(a);
  const b = el("div", { class: "stat" });
  b.append(el("div", { class: "k", text: "Google reviews" }));
  b.append(el("div", { class: "v", text: rating ? rating.mean.toFixed(2) + "★" : "—" }));
  b.append(el("div", { class: "n",
    text: rating ? `${rating.n.toLocaleString()} to ${rating.last}` : "no ratings" }));
  pair.append(b);
  panel.append(pair);

  if (merged) {
    panel.append(el("h3", { text: "Combined" }));
    panel.append(mergedBlock(merged, branch ? "this branch" : "chain"));
  }

  if (rating && totals) {
    panel.append(el("h3", { text: "Reddit vs Google" }));
    const box = el("div", { class: "boxed" });
    box.append(sourceBar("Reddit", totals.s, "reddit"));
    box.append(sourceBar("Google", rating.norm, "google"));
    const gap = rating.norm - totals.s;
    box.append(el("div", { class: "gap-note", text:
      Math.abs(gap) < 0.15
        ? "The two sources agree closely here."
        : gap > 0
          ? `Google reads ${fmt(gap)} kinder. Star ratings compress toward the top, `
            + "and reviewers are self-selected customers of this shop."
          : `This corpus reads ${fmt(-gap)} kinder than Google does — unusual.` }));
    panel.append(box);
  }

  const ordered = Object.entries(cats).sort((x, y) => y[1].w - x[1].w);
  const combined = branch
    ? data().reviews?.branches[store]?.[branch]
    : data().reviews?.categories[store];
  if (ordered.length) {
    panel.append(el("h3", { text: "By category" }));
    for (const [name, cell] of ordered.slice(0, 8)) {
      panel.append(cellBlock(name, cell, combined?.[name]));
    }
  }

  const items = Object.entries(data().items[store] ?? {})
    .sort((x, y) => y[1].w - x[1].w).slice(0, 6);
  if (items.length) {
    panel.append(el("h3", { text: "Your basket here" }));
    const lines = el("div", { class: "lines" });
    for (const [name, cell] of items) {
      lines.append(el("div", {}, [
        el("span", { text: name }),
        el("span", { class: "score " + cls(cell.s), text: fmt(cell.s) }),
      ]));
    }
    panel.append(lines);
  }

  panel.append(el("p", { class: "fineprint", text:
    "Figures are weighted opinion, not measurements. Every quote links to the "
    + "comment it came from." }));

  document.body.append(scrim, panel);
  drawerOpen = [scrim, panel];
  panel.focus();
}

function sourceBar(label: string, value: number, kind: string): HTMLElement {
  // -1..+1 mapped onto a 0-100% bar.
  const pct = Math.round(((value + 1) / 2) * 100);
  const row = el("div", { class: "bar-row" });
  row.append(el("span", { class: "bar-label", text: label }));
  const track = el("span", { class: "bar-track" });
  track.append(el("span", { class: `bar-fill ${kind}`, style: `width:${pct}%` }));
  row.append(track);
  row.append(el("span", { class: "bar-value", text: fmt(value) }));
  return row;
}

/** Store cards for the Stores view, straight from the comp. */
function renderStoreCards(): void {
  const wrap = need("#store-cards");
  clear(wrap);
  const stores = Object.keys(data().stores).sort((a, b) =>
    (data().totals[b]?.w ?? 0) - (data().totals[a]?.w ?? 0));
  for (const store of stores) {
    const totals = data().totals[store];
    const rating = data().crosscheck?.stores[store];
    const card = el("button", { class: "store-card", type: "button" });
    const top = el("div", { class: "top" });
    top.append(el("h3", { text: store }));
    top.append(el("span", { class: "score " + cls(totals?.s ?? 0), text: fmt(totals?.s ?? 0) }));
    card.append(top);
    const nBranch = Object.keys(data().branches[store] ?? {}).length;
    card.append(el("div", { class: "meta",
      text: `${(totals?.n ?? 0).toLocaleString()} claims · ${nBranch} branch${nBranch === 1 ? "" : "es"}` }));
    const bars = el("div", { class: "bars" });
    bars.append(sourceBar("Reddit", totals?.s ?? 0, "reddit"));
    if (rating) bars.append(sourceBar("Google", rating.norm, "google"));
    card.append(bars);
    // The most strongly evidenced thing anyone said about this store.
    const best = Object.values(data().stores[store] ?? {})
      .flatMap((c) => c.e).slice(0, 1)[0];
    if (best) card.append(el("p", { class: "quote", text: "“" + best.t + "”" }));
    card.addEventListener("click", () => openDetail(store));
    wrap.append(card);
  }
}

/* ── view: store detail ─────────────────────────────────────────────── */

/** A one-line combined figure to sit inside a category's summary row. */
function combinedChip(m: MergedValue | undefined): HTMLElement | null {
  if (!m || m.g === undefined || m.r === undefined) return null;
  const chip = el("span", {
    class: "chip" + (m.conflict ? " conflict" : ""),
    title: `Reddit ${fmt(m.r)}, reviews ${fmt(m.g)}, `
         + `${(m.share * 100).toFixed(0)}% of the weight from Reddit`,
  });
  chip.append(el("span", { class: "score " + cls(m.v), text: fmt(m.v) }));
  chip.append(el("span", { class: "muted", text: " combined" }));
  if (m.conflict) chip.append(el("span", { class: "muted", text: " ⚠" }));
  return chip;
}

function cellBlock(name: string, cell: Cell, merged?: MergedValue): HTMLElement {
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
  const chip = combinedChip(merged);
  if (chip) sum.append(chip);
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
  // The comp replaces the separate store page with a drawer, so the reader
  // keeps their place in whatever list they clicked from.
  openDetail(store, branch);
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
    activatable(tr, `${h.name} at ${h.store}`, () => showStore(h.store));
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

const VIEWS = ["list", "compare", "cross", "map", "items", "method"] as const;
type View = (typeof VIEWS)[number];

function switchView(name: View): void {
  for (const b of all<HTMLElement>(".tabs button")) {
    const on = b.dataset["view"] === name;
    b.classList.toggle("on", on);
    // The tablist role is a contract: selection state and a roving tabindex
    // are part of it, not decoration on top of it.
    b.setAttribute("aria-selected", on ? "true" : "false");
    b.setAttribute("tabindex", on ? "0" : "-1");
  }
  for (const v of all<HTMLElement>(".view")) v.hidden = v.id !== "view-" + name;
  if (location.hash.slice(1) !== name) history.replaceState(null, "", "#" + name);
}

function renderView(name: View): void {
  if (name === "compare") { renderCompare(); renderStoreCards(); }
  if (name === "items") renderItems();
  if (name === "cross") renderCross();
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
    ["Google ratings (cross-check)", (data().crosscheck?.n_reviews ?? 0).toLocaleString()],
    ["Google half-life", (data().crosscheck?.half_life_years ?? 0) + " years"],
    ["Review-derived claims", (data().reviews?.n_review_claims ?? 0).toLocaleString()],
    ["Claim-merge slope", data().reviews?.calibration.slope ?? "n/a"],
    ["Calibration slope", data().merged?.calibration.slope ?? "n/a"],
    ["Calibration LOO error", data().merged?.calibration.loo_rmse ?? "n/a"],
    ["Shrinkage constant", data().method.shrinkage_k],
    ["Default half-life", data().method.default_half_life_years + " years"],
    ["Transient claims", data().method.transient_claims],
  ];
  for (const [k, v] of pairs) dl.append(el("dt", { text: k }), el("dd", { text: String(v) }));
}

/** Headline figures on the Method page. Real counts, never rounded up. */
function fillTally(): void {
  const d = data();
  // `totals` is shopping-only, so this is the number that actually feeds a
  // ranking — smaller than the number extracted, and the honest one to show.
  const claims = Object.values(d.totals).reduce((a, t) => a + t.n, 0);
  const items = Object.values(d.items).reduce((a, v) => a + Object.keys(v).length, 0);
  const pairs: [string, string][] = [
    ["Claims weighed into the rankings", claims.toLocaleString()],
    ["Google ratings cross-checked", (d.crosscheck?.n_reviews ?? 0).toLocaleString()],
    ["Stores scored", String(Object.keys(d.stores).length)],
    ["Locations mapped", String(d.places.length)],
    ["Specific items you can look up", items.toLocaleString()],
  ];
  const grid = need("#m-stats");
  for (const [k, v] of pairs) {
    grid.append(el("div", {}, [
      el("div", { class: "v", text: v }), el("div", { class: "k", text: k }),
    ]));
  }
}

/** The Method page quotes real numbers rather than remembered ones. */
function fillCalibrationProse(): void {
  const m = data().merged;
  if (!m) return;
  const c = m.calibration;
  need("#m-cal").textContent =
    `reddit = ${c.intercept.toFixed(2)} + ${c.slope.toFixed(2)} x google`;
  need("#m-loo").textContent = c.loo_rmse.toFixed(3);
  need("#m-resid").textContent = `±${c.residual_sd.toFixed(2)}`;
}

/** Check the fetched document really is the payload before trusting it.
 *
 * `as Payload` asserted rather than verified, so a truncated or stale file
 * became a blank page with a message only in the console. Structural, not
 * exhaustive: it confirms the views the app indexes into actually exist.
 */
function isPayload(value: unknown): value is Payload {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  const objects = ["stores", "branches", "regions", "items", "totals", "keywords", "method"];
  for (const key of objects) {
    if (typeof v[key] !== "object" || v[key] === null) return false;
  }
  return Array.isArray(v["places"]) && Array.isArray(v["categories"])
    && typeof v["generated_at"] === "string";
}

async function boot(): Promise<void> {
  try {
    const res = await fetch("verdicts.json");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const parsed: unknown = await res.json();
    if (!isPayload(parsed)) throw new Error("verdicts.json is not the expected shape");
    DATA = parsed;
  } catch (err) {
    need("#loading").textContent =
      "Could not load the data: " + (err instanceof Error ? err.message : String(err));
    return;
  }
  need("#loading").hidden = true;

  for (const s of Object.keys(data().stores).sort()) {
    need("#item-store").append(el("option", { value: s, text: s }));
  }
  const mapStores = Array.from(new Set(data().places.map((p) => p.store))).sort();
  for (const s of mapStores) need("#map-store").append(el("option", { value: s, text: s }));
  for (const c of data().categories) {
    need("#cmp-cat").append(el("option", { value: c, text: titleCase(c) }));
    need("#map-cat").append(el("option", { value: c, text: titleCase(c) }));
  }


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
  fillTally();
  fillCalibrationProse();
  renderBasket();
  runList();

  const go = (v: View): void => { switchView(v); renderView(v); };
  for (const b of all<HTMLElement>("[data-view]")) {
    b.addEventListener("click", () => go(b.dataset["view"] as View));
  }
  // Arrow keys move between tabs, Home/End jump to the ends — what a
  // tablist is required to do once it claims the role.
  const tabs = all<HTMLElement>(".tabs button");
  for (const [i, tab] of tabs.entries()) {
    tab.addEventListener("keydown", (e) => {
      const key = (e as KeyboardEvent).key;
      const step = key === "ArrowRight" ? 1 : key === "ArrowLeft" ? -1 : 0;
      let next = -1;
      if (step) next = (i + step + tabs.length) % tabs.length;
      else if (key === "Home") next = 0;
      else if (key === "End") next = tabs.length - 1;
      if (next < 0) return;
      e.preventDefault();
      const target = tabs[next];
      if (!target) return;
      target.focus();
      go(target.dataset["view"] as View);
    });
  }
  need("#add-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = need<HTMLInputElement>("#list-input");
    addToBasket(input.value);
    input.value = "";
  });
  const sorts: [typeof sortMode, string][] = [
    ["value", "Best on your list"], ["quality", "Best rated"], ["evidence", "Most evidence"],
  ];
  const bar = need("#sorts");
  for (const [id, label] of sorts) {
    const b = el("button", { type: "button", class: id === sortMode ? "on" : "", text: label });
    b.addEventListener("click", () => {
      sortMode = id;
      for (const other of Array.from(bar.children)) other.classList.remove("on");
      b.classList.add("on");
      runList();
    });
    bar.append(b);
  }
  document.addEventListener("keydown", (e) => {
    if ((e as KeyboardEvent).key === "Escape" && drawerOpen.length) closeDrawer();
  });
  need("#pref").addEventListener("input", (e) => {
    need("#pref-out").textContent = prefLabel(Number((e.target as HTMLInputElement).value));
    runList();
  });
  need("#pref-out").textContent = prefLabel(50);
  // These controls serve both the basket ranking and the compare table.
  const refilter = (): void => { renderCompare(); runList(); };
  need("#cmp-cat").addEventListener("change", refilter);
  need("#cmp-thin").addEventListener("change", refilter);
  need("#cmp-min").addEventListener("input", (e) => {
    const v = (e.target as HTMLInputElement).value;
    need("#cmp-min-out").textContent = v === "0" ? "any" : v + "+";
    refilter();
  });
  for (const th of all<HTMLElement>("#cmp-table th")) {
    th.addEventListener("click", () => {
      const k = th.dataset["sort"];
      if (!k) return;
      cmpSort = { key: k, dir: cmpSort.key === k ? -cmpSort.dir : (k === "store" ? 1 : -1) };
      renderCompare();
    });
  }
  need("#cross-pop").addEventListener("change", renderCross);
  need("#map-colour").addEventListener("change", renderMap);
  need("#map-store").addEventListener("change", renderMap);
  need("#map-cat").addEventListener("change", renderMap);
  need("#map-evidence").addEventListener("change", renderMap);
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
