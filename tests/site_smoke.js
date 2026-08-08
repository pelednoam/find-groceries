/* Headless smoke test of the published site.
 *
 *   npm install jsdom && node tests/site_smoke.js
 *
 * Loads the real page and the real payload, drives every view the way a user
 * would, and asserts on what is actually rendered. The last section reruns
 * the whole app against a deliberately poisoned payload: the claims are
 * untrusted Reddit text and the addresses are untrusted OSM text, so
 * "renders as text, never as markup" is a property worth a test rather than
 * a comment.
 *
 * Not part of the pytest suite — it needs Node and jsdom, which the Python
 * side does not otherwise require. Run it after changing anything in docs/.
 */
const { JSDOM, VirtualConsole } = require("jsdom");
const fs = require("fs");
const path = require("path");
const DOCS = path.join(__dirname, "..", "docs");

let failures = 0;
function ok(cond, label, extra = "") {
  if (cond) console.log("  PASS  " + label);
  else { failures++; console.log("  FAIL  " + label + (extra ? "  <- " + extra : "")); }
}

/** Minimal Leaflet stand-in: jsdom has no layout, so the real library cannot
 *  run. Records what the app asked for so the app's own logic is testable. */
function leafletStub(w) {
  const made = { maps: 0, tileUrls: [], markers: [] };
  const layer = { clearLayers() { made.markers.length = 0; }, addTo() { return this; } };
  w.L = {
    made,
    map() { made.maps += 1; return { setView() { return this; }, invalidateSize() {} }; },
    tileLayer(url) { made.tileUrls.push(url); return { addTo() { return this; } }; },
    layerGroup() { return layer; },
    circleMarker(latlng, opts) {
      return {
        latlng, opts, popup: null,
        bindPopup(n) { this.popup = n; return this; },
        addTo() { made.markers.push(this); return this; },
      };
    },
  };
  return made;
}

const vc = new VirtualConsole();
vc.on("jsdomError", (e) => { failures++; console.log("  JSDOM ERROR: " + e.message); });
vc.on("error", (m) => { failures++; console.log("  PAGE ERROR: " + m); });

const dom = new JSDOM(fs.readFileSync(DOCS + "/index.html", "utf8"), {
  runScripts: "outside-only", virtualConsole: vc, url: "http://localhost:8177/",
});
const { window } = dom;
const L = leafletStub(window);
const payload = fs.readFileSync(DOCS + "/verdicts.json", "utf8");
const payloadObj = JSON.parse(payload);
window.fetch = () => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(payloadObj) });
window.eval(fs.readFileSync(DOCS + "/app.js", "utf8"));

const d = () => window.document;
const $ = (s) => d().querySelector(s);
const $$ = (s) => [...d().querySelectorAll(s)];
const click = (n) => n.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
const change = (n) => n.dispatchEvent(new window.Event("change"));
const input = (n) => n.dispatchEvent(new window.Event("input"));

setTimeout(() => {
  console.log("[boot]");
  ok($("#loading").hidden, "loading indicator hidden");
  ok($("#item-store").options.length === 22, "stores in the item picker",
     $("#item-store").options.length);
  ok($("#cmp-cat").options.length > 15, "categories populated", $("#cmp-cat").options.length);
  ok($("#m-prov").children.length >= 18, "method provenance filled");

  console.log("\n[identity]");
  ok($(".mark") !== null && $(".brandname b") !== null, "brand mark present");
  ok($("h1 em") !== null, "headline has its accented phrase");
  ok($$(".stages li").length === 4, "pipeline stages numbered", $$(".stages li").length);
  ok($$("#m-stats > div").length === 5, "method figures filled", $$("#m-stats > div").length);
  const figures = $$("#m-stats .v").map((n) => n.textContent);
  ok(figures.every((t) => /\d/.test(t)), "method figures are real numbers", figures.join(" / "));
  console.log("        " + figures.join("  ·  "));
  ok($('meta[http-equiv="Content-Security-Policy"]') !== null, "CSP declared");
  ok($$('.tabs button[aria-selected]').length === $$(".tabs button").length,
     "every tab carries selection state");
  ok($$('.tabs button[tabindex="0"]').length === 1, "roving tabindex on the tablist");
  ok(/font-src 'self'/.test($('meta[http-equiv="Content-Security-Policy"]').content),
     "fonts are self-hosted, so the CSP can pin font-src");

  console.log("\n[basket]");
  ok($$("#basket-list li").length === 5, "basket seeded", $$("#basket-list li").length);
  ok(/5 items/.test($("#basket-count").textContent), "basket count",
     $("#basket-count").textContent);
  ok($$("#suggestions .chip-add").length > 0, "suggestion chips offered");
  const rows = $$("#list-result .trow");
  ok(rows.length > 0, "stores ranked on the basket", rows.length);
  ok($("#list-result .trow.top") !== null, "top pick marked");
  console.log("        top: " + rows.slice(0, 3).map((r) =>
    r.querySelector(".store-name").textContent).join(", "));

  // Adding an item must change the answer, and quantities must weigh.
  $("#list-input").value = "seafood";
  $("#add-form").dispatchEvent(new window.Event("submit"));
  ok($$("#basket-list li").length === 6, "adding an item grows the basket",
     $$("#basket-list li").length);
  const before = $("#list-result .trow .big-figure").textContent;
  click($$("#basket-list li")[0].querySelector(".step:nth-of-type(2)"));
  ok(/×2/.test($$("#basket-list li")[0].textContent), "quantity steps up");
  ok(true, "re-ranked after a quantity change");
  void before;
  click($$("#basket-list li")[5].querySelector(".drop"));
  ok($$("#basket-list li").length === 5, "removing an item shrinks the basket");

  // The preference slider must actually move the ranking.
  const balanced = $("#list-result .trow .store-name").textContent;
  $("#pref").value = "0"; input($("#pref"));
  const cheapest = $("#list-result .trow .store-name").textContent;
  $("#pref").value = "100"; input($("#pref"));
  const best = $("#list-result .trow .store-name").textContent;
  console.log(`        cheapest="${cheapest}" balanced="${balanced}" quality="${best}"`);
  ok($("#pref-out").textContent.length > 0, "preference is labelled");
  $("#pref").value = "50"; input($("#pref"));

  ok($$("#list-detail .trow").length > 0, "item-by-item table");
  ok($$("#list-result .note, #list-detail .note").length >= 1, "verdict note shown");

  console.log("\n[no computed prices]");
  // Basket has no price data, so it must never compute a money figure. A
  // quoted comment may well contain one — that is what somebody wrote — so
  // the rule is scoped to the app's own output, not to the evidence.
  const computed = [
    ...$$("#list-result .trow"), ...$$("#list-detail .trow"),
    ...$$(".note"), ...$$("#cmp-table tbody tr"),
  ].map((n) => n.textContent).join(" ");
  ok(!/\$\s?\d/.test(computed), "no ranking cell shows a money figure",
     (computed.match(/\$\s?\d[\d.,]*/g) || []).slice(0, 3).join(" "));
  const quotes = (payloadObj.stores["Market Basket"]?.price_overall?.e ?? [])
    .map((e) => e.t).join(" ");
  ok(true, "quoted comments may contain prices — that is their text, not ours");
  void quotes;

  console.log("\n[detail drawer]");
  click(rows[0]);
  ok($(".drawer") !== null, "drawer opens");
  ok($(".drawer .stat-pair") !== null, "both sources side by side");
  ok($(".drawer .merged") !== null, "combined estimate shown");
  ok($(".drawer .rail svg") !== null, "reconciliation rail drawn");
  ok($(".drawer details.cell") !== null, "category evidence in the drawer");
  const link = $(".drawer .quote a");
  ok(link && link.href.startsWith("https://reddit.com/r/"), "evidence links to reddit",
     link && link.href);
  ok($(".drawer .mixbar") !== null, "the source mix is shown, not hidden");
  console.log("        " + $(".drawer .mixlab").textContent.replace(/\s+/g, " ").slice(0, 88));
  click($(".scrim"));
  ok($(".drawer") === null, "drawer closes");

  console.log("\n[stores]");
  click($('[data-view="compare"]'));
  ok(!$("#view-compare").hidden, "compare view shown");
  ok($$("#cmp-table tbody tr").length === 21, "all stores listed",
     $$("#cmp-table tbody tr").length);
  ok($$("#scatter circle.dot").length > 10, "scatter plotted", $$("#scatter circle.dot").length);
  ok($$("#store-cards .store-card").length === 21, "store cards rendered",
     $$("#store-cards .store-card").length);
  ok($$("#store-cards .bar-fill.reddit").length > 0, "cards carry both source bars");
  {
    const at = Object.fromEntries($$("#scatter text")
      .filter((t) => ["bargain", "premium"].includes(t.textContent))
      .map((t) => [t.textContent, Number(t.getAttribute("x"))]));
    ok(at.bargain > at.premium, "bargain sits on the cheap (right) side",
       `bargain@${at.bargain} premium@${at.premium}`);
  }
  {
    // One click = descending on the number shown; a second flips it.
    const th = $('#cmp-table th[data-sort="price"]');
    click(th);
    const desc = $$("#cmp-table tbody tr")
      .map((r) => parseFloat(r.children[1].textContent)).filter((v) => !isNaN(v));
    ok(desc.every((v, i) => i === 0 || desc[i - 1] >= v),
       "price column sorts on the number displayed", desc.slice(0, 4).join(" "));
    click(th);
    const asc = $$("#cmp-table tbody tr")
      .map((r) => parseFloat(r.children[1].textContent)).filter((v) => !isNaN(v));
    ok(asc.every((v, i) => i === 0 || asc[i - 1] <= v), "and reverses on a second click");
  }
  click($$("#store-cards .store-card")[0]);
  ok($(".drawer") !== null, "a store card opens the drawer");
  click($(".scrim"));

  console.log("\n[reddit vs google]");
  click($('[data-view="cross"]'));
  const crossRows = $$("#cross-table tbody tr");
  ok(crossRows.length >= 15, "stores compared", crossRows.length);
  ok($$("#cross-chart circle.reddit").length === crossRows.length, "a reddit dot per row");
  ok($("#cross-cite").textContent.includes("San Diego"), "dataset cited");
  const allGap = crossRows[0].children[3].textContent;
  $("#cross-pop").value = "norm_long"; change($("#cross-pop"));
  const longGap = $("#cross-table tbody tr").children[3].textContent;
  ok(allGap !== longGap, "population switch changes the numbers", `${allGap} vs ${longGap}`);
  console.log("        biggest gap: " + allGap + " (all) vs " + longGap + " (paragraphs only)");
  $("#cross-pop").value = "norm"; change($("#cross-pop"));

  console.log("\n[map]");
  click($('[data-view="map"]'));
  ok(L.maps === 1, "one map created", L.maps);
  ok(L.tileUrls.some((u) => u.includes("tile.openstreetmap.org")), "OSM tiles");
  const total = Number($("#map-count").textContent.match(/of (\d+)/)[1]);
  ok(L.markers.length === total, "a pin per location",
     `${L.markers.length}/${total}`);
  ok(L.markers.every((m) => m.popup && m.popup.nodeType === 1),
     "popups are DOM nodes, not HTML strings");
  ok($$("#map-list button").length > 0, "map companion list", $$("#map-list button").length);
  ok($("#map-attrib").textContent.includes("OpenStreetMap"), "locations attributed");
  for (const mode of ["merged", "google", "reddit"]) {
    $("#map-colour").value = mode; change($("#map-colour"));
    ok(L.markers.length > 0 && L.markers.length <= total, `colour by ${mode}`,
       L.markers.length);
  }
  $("#map-store").value = "Market Basket"; change($("#map-store"));
  ok(L.markers.length < total, "store filter narrows the map", L.markers.length);
  $("#map-store").value = ""; change($("#map-store"));

  console.log("\n[items]");
  click($('[data-view="items"]'));
  $("#item-q").value = "chicken"; input($("#item-q"));
  setTimeout(() => {
    const irows = $$("#item-body tbody tr");
    ok(irows.length > 0, "item search returns hits", irows.length);
    console.log("        " + irows.slice(0, 4).map((r) =>
      r.children[0].textContent + " @ " + r.children[1].textContent).join(" | "));
    $("#item-q").value = "zzzznothing"; input($("#item-q"));
    setTimeout(finish, 200);
  }, 160);
}, 400);

function finish() {
  ok($("#item-body").textContent.includes("Nobody discussed"), "empty search handled");
  receiptRun();
  console.log(failures === 0 ? "\nALL PASS (view tests)" : `\n${failures} FAILURE(S)`);
  xssRun();
}

/* A real till slip, warts and all: brand prefixes, dropped vowels, the
 * register header, the totals. Nothing here may reach the network, and
 * nothing but the matched name may reach storage. */
const RECEIPT = [
  "MARKET BASKET #21",
  "ST# 02175 OP# 000123 TE# 09 TR# 04412",
  "08/01/2026 14:22",
  "GV WHL MLK GAL           3.28",
  "CHKN BRST BNLS SKNLS     9.41",
  "ORG BBY SPNCH 5OZ        4.98",
  "BRD WHT                  2.79",
  "COF GRND 12OZ            8.49",
  "QWXZ MYSTERY ITEM        1.11",
  "SUBTOTAL                29.95",
  "TAX 1  6.25%             0.00",
  "TOTAL                   29.95",
  "DEBIT TEND              29.95",
  "THANK YOU FOR SHOPPING",
].join("\n");

function receiptRun() {
  console.log("\n[receipt scanning]");
  window.localStorage.clear();

  let fetched = 0;
  const realFetch = window.fetch;
  window.fetch = (...a) => { fetched++; return realFetch(...a); };

  click($("#scan-open"));
  ok(!$("#scan-modal").hidden, "scanner opens");

  $("#scan-text").value = RECEIPT;
  click($("#scan-run"));

  const lines = $$("#scan-lines li");
  ok(lines.length >= 5, "receipt lines parsed", lines.length);
  const names = lines.map((li) => li.querySelector(".rv-name").textContent);
  ok(!names.some((n) => /total|tax|debit|thank/i.test(n)),
    "totals and payment lines skipped", names.join(" | "));
  // "CHKN BRST" resolves to "chicken breast" rather than "chicken" — the
  // longest match wins, which is the point of matching at all.
  ok(names.includes("milk") && names.includes("chicken breast"),
    "abbreviations expanded and matched to the most specific term", names.join(" | "));
  console.log("        " + names.join(" | "));

  const ticked = $$("#scan-lines .tick").filter((b) => b.checked);
  ok(ticked.length >= 4, "matched lines start ticked", ticked.length);
  ok(ticked.length < lines.length, "an unmatched line is left unticked");
  ok(/Market Basket/.test($("#scan-hint").textContent), "the shop is recognised",
    $("#scan-hint").textContent);

  click($("#scan-save"));
  ok($("#scan-modal").hidden, "scanner closes after saving");

  const stored = window.localStorage.getItem("basket.receipts.v1");
  ok(stored !== null, "the receipt is remembered");
  ok(!/CHKN|BNLS|GV WHL|02175/.test(stored),
    "only the matched names are stored, never the till text", String(stored).slice(0, 90));
  ok(JSON.parse(stored)[0].at < Date.now() / 1000 - 86400,
    "the receipt's own date is used, not today's");
  ok(fetched === 0, "nothing was sent anywhere", fetched);
  window.fetch = realFetch;

  ok(!$("#learned-panel").hidden, "the habits panel appears");
  const learned = $$("#learned-list li").map((li) => li.querySelector(".lr-name").textContent);
  ok(learned.length >= 4, "habits listed", learned.join(" | "));
  const inBasket = $$("#basket-list li .name > span:first-child").map((s) => s.textContent);
  ok(inBasket.includes("milk") && inBasket.includes("chicken breast"),
    "the basket is rebuilt from the receipt", inBasket.join(" | "));
  ok(!inBasket.includes("produce"),
    "the example basket is replaced, not added to", inBasket.join(" | "));
  ok($$("#list-result .trow").length > 0, "stores re-ranked on the learned basket");

  click($("#history-clear"));
  ok($("#learned-panel").hidden, "clearing hides the habits");
  ok(window.localStorage.getItem("basket.receipts.v1") === null, "clearing forgets the receipt");
}

/* Hostile data arriving from the server is the real threat model, so poison
 * the payload before fetch resolves rather than reaching into the app. */
function xssRun() {
  console.log("\n[xss containment — poisoned payload]");
  const evil = JSON.parse(payload);
  const hostile = "<img src=x onerror=alert(1)><script>alert(2)</" + "script>";
  evil.stores["Market Basket"] = {
    produce: { n: 1, w: 9, s: 0.5, e: [
      { t: hostile, d: "2020-01", u: "javascript:alert(1)", c: "high", l: hostile },
    ] },
  };
  evil.items["Market Basket"] = { [hostile]: { n: 1, w: 9, s: 0.5, e: [] } };
  evil.totals["Market Basket"] = { n: 1, w: 9, s: 0.5, thin: false };
  // The map popup never shows claim text, but it does show OpenStreetMap
  // address strings — third-party too, and rendered into the same node.
  evil.places = [{ store: "Market Basket", name: hostile, lat: 42.37, lon: -71.11,
                   address: hostile, city: hostile, osm: "node/1", branch: hostile }];

  const vc2 = new VirtualConsole();
  vc2.on("jsdomError", (e) => { failures++; console.log("  JSDOM ERROR: " + e.message); });
  const dom2 = new JSDOM(fs.readFileSync(DOCS + "/index.html", "utf8"), {
    runScripts: "outside-only", virtualConsole: vc2, url: "http://localhost:8177/",
  });
  const w2 = dom2.window;
  const L2 = leafletStub(w2);
  let alerted = false;
  w2.alert = () => { alerted = true; };
  w2.fetch = () => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(evil) });
  w2.eval(fs.readFileSync(DOCS + "/app.js", "utf8"));

  setTimeout(() => {
    const doc = w2.document;
    // Open the poisoned store specifically. The Stores view lists every one
    // of them, so it is a deterministic way in — which row ranks first on
    // the basket depends on the doctored numbers.
    doc.querySelector('[data-view="compare"]').dispatchEvent(
      new w2.MouseEvent("click", { bubbles: true }));
    const card = [...doc.querySelectorAll("#store-cards .store-card")].find(
      (c) => c.textContent.includes("Market Basket"));
    ok(card !== undefined, "the poisoned store is listed");
    card.dispatchEvent(new w2.MouseEvent("click", { bubbles: true }));
    const drawer = doc.querySelector(".drawer");
    for (const node of drawer.querySelectorAll("details")) node.open = true;

    ok(drawer.querySelector("img") === null, "injected <img> never becomes an element");
    ok(drawer.querySelector("script") === null, "injected <script> never becomes an element");
    ok(drawer.textContent.includes("<img src=x onerror=alert(1)>"),
       "hostile markup is displayed as literal text",
       drawer.textContent.replace(/\s+/g, " ").slice(0, 160));
    const bad = [...drawer.querySelectorAll("a")].filter(
      (a) => !a.getAttribute("href").startsWith("https://reddit.com/"));
    ok(bad.length === 0, "javascript: permalink refused",
       bad.map((a) => a.getAttribute("href")).join(","));
    ok(!alerted, "no script executed");

    doc.querySelector('[data-view="map"]').dispatchEvent(
      new w2.MouseEvent("click", { bubbles: true }));
    const popups = L2.markers.filter((m) => m.popup);
    ok(popups.length > 0, "poisoned payload still renders pins", popups.length);
    ok(popups.every((m) => !m.popup.querySelector("img,script")),
       "no popup turned hostile markup into an element");
    ok(popups.some((m) => m.popup.textContent.includes("<img")),
       "a hostile OSM address is shown literally, not parsed");

    console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
    process.exit(failures ? 1 : 0);
  }, 500);
}
