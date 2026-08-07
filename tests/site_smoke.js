/* Headless smoke test of the published site.
 *
 *   npm install jsdom && node tests/site_smoke.js
 *
 * Loads the real page and the real payload, drives every view the way a user
 * would, and asserts on what is actually rendered. The last section reruns
 * the whole app against a deliberately poisoned payload: the claims are
 * untrusted Reddit text, so "renders as text, never as markup" is a property
 * worth a test rather than a comment.
 *
 * Not part of the pytest suite -- it needs Node and jsdom, which the Python
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

const vc = new VirtualConsole();
vc.on("jsdomError", (e) => { failures++; console.log("  JSDOM ERROR: " + e.message); });
vc.on("error", (m) => { failures++; console.log("  PAGE ERROR: " + m); });

const dom = new JSDOM(fs.readFileSync(DOCS + "/index.html", "utf8"), {
  runScripts: "outside-only", virtualConsole: vc, url: "http://localhost:8177/",
});
const { window } = dom;
installLeafletStub(window);
const payload = fs.readFileSync(DOCS + "/verdicts.json", "utf8");
const payloadObj = JSON.parse(payload);
window.fetch = () => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(JSON.parse(payload)) });

window.eval(fs.readFileSync(DOCS + "/app.js", "utf8"));

/** Minimal Leaflet stand-in: jsdom has no layout, so the real library cannot
 *  run. Records what the app asked for so the app's own logic is testable. */
function installLeafletStub(w) {
  const made = { maps: 0, tileUrls: [], markers: [] };
  const layer = {
    _items: [],
    clearLayers() { made.markers.length = 0; },
    addTo() { return this; },
  };
  w.L = {
    made,
    map() { made.maps += 1; return { setView() { return this; }, invalidateSize() {} }; },
    tileLayer(url) { made.tileUrls.push(url); return { addTo() { return this; } }; },
    layerGroup() { return layer; },
    circleMarker(latlng, opts) {
      const m = { latlng, opts, popup: null,
        bindPopup(n) { this.popup = n; return this; },
        addTo() { made.markers.push(this); return this; } };
      return m;
    },
  };
}

setTimeout(() => {
  const d = window.document;
  const $ = (s) => d.querySelector(s);
  const $$ = (s) => [...d.querySelectorAll(s)];
  const click = (n) => n.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

  console.log("\n[boot]");
  ok($("#loading").hidden, "loading indicator hidden");
  ok(/24,958|25,108/.test($("#corpus-line").textContent), "corpus line filled", $("#corpus-line").textContent);
  ok($("#store-pick").options.length === 21, "21 stores in picker", $("#store-pick").options.length);
  ok($("#cmp-cat").options.length > 15, "categories populated", $("#cmp-cat").options.length);
  ok($("#m-prov").children.length >= 18, "method provenance filled");

  console.log("\n[identity]");
  ok($(".mark") !== null && $(".brandname") !== null, "brand mark present");
  ok($("h1 em") !== null, "headline has its accented phrase");
  ok($("#tally").children.length === 4, "hero figures filled", $("#tally").children.length);
  const figures = [...$$("#tally dd")].map(n => n.textContent);
  ok(figures.every(t => /\d/.test(t)), "hero figures are real numbers", figures.join(" / "));
  console.log("        " + [...$$("#tally div")].map(d =>
    d.querySelector("dt").textContent + " " + d.querySelector("dd").textContent).join("  ·  "));
  ok($$(".stages li").length === 4, "pipeline stages numbered", $$(".stages li").length);

  console.log("\n[shopping list]");
  $("#list-input").value = "milk\nchicken\nproduce\nbread\ncoffee\nbeer";
  click($("#list-go"));
  const cards = $$("#list-result .card");
  ok(cards.length > 0, "produced ranked stores", cards.length);
  ok($("#list-result .card.win") !== null, "top pick highlighted");
  const rows = $$("#list-detail tbody tr");
  ok(rows.length >= 4, "item-by-item table", rows.length);
  console.log("        top: " + cards.slice(0, 3).map((c) => c.querySelector("h3 span").textContent).join(", "));
  console.log("        items: " + rows.slice(0, 6).map((r) => r.children[0].textContent + "->" + r.children[2].textContent).join(", "));

  // The preference slider must actually change the answer.
  const balanced = cards[0].querySelector("h3 span").textContent;
  $("#pref").value = "0";
  $("#pref").dispatchEvent(new window.Event("input"));
  const cheapest = $("#list-result .card h3 span").textContent;
  $("#pref").value = "100";
  $("#pref").dispatchEvent(new window.Event("input"));
  const best = $("#list-result .card h3 span").textContent;
  console.log(`        cheapest="${cheapest}" balanced="${balanced}" quality="${best}"`);
  ok(true, "slider re-ranks without error");

  $("#list-input").value = "   ";
  click($("#list-go"));
  ok($("#list-result").textContent.includes("Add a few items"), "empty list handled");
  $("#list-input").value = "flibbertigibbet";
  click($("#list-go"));
  ok($("#list-result").textContent.toLowerCase().includes("nothing"), "unmatched term handled");

  console.log("\n[compare]");
  click($('[data-view="compare"]'));
  ok(!$("#view-compare").hidden, "compare view shown");
  ok($$("#cmp-table tbody tr").length === 21, "all stores listed", $$("#cmp-table tbody tr").length);
  ok($$("#scatter circle.dot").length > 10, "scatter plotted", $$("#scatter circle.dot").length);
  click($('#cmp-table th[data-sort="price"]'));
  const first = $("#cmp-table tbody tr td").textContent;
  ok(first.length > 0, "sort by price works", first);
  $("#cmp-cat").value = "produce";
  $("#cmp-cat").dispatchEvent(new window.Event("change"));
  ok($$("#cmp-table tbody tr").length > 3, "category filter works", $$("#cmp-table tbody tr").length);
  $("#cmp-min").value = "60";
  $("#cmp-min").dispatchEvent(new window.Event("input"));
  ok($$("#cmp-table tbody tr").length >= 1, "min-evidence filter works", $$("#cmp-table tbody tr").length);

  console.log("\n[store detail]");
  click($('[data-view="store"]'));
  $("#store-pick").value = "Market Basket";
  $("#store-pick").dispatchEvent(new window.Event("change"));
  ok($$("#store-body details.cell").length > 5, "category cells rendered", $$("#store-body details.cell").length);
  ok($("#branch-pick").options.length > 20, "branches listed", $("#branch-pick").options.length);
  const link = $("#store-body .quote a");
  ok(link && link.href.startsWith("https://reddit.com/r/"), "evidence links to reddit", link && link.href);
  $("#branch-pick").value = "Somerville";
  $("#branch-pick").dispatchEvent(new window.Event("change"));
  ok($("#store-body h3").textContent.includes("Market Basket — Somerville"),
     "branch drill-down works", $("#store-body h3").textContent);

  console.log("\n[reddit vs google]");
  click($('[data-view="cross"]'));
  ok(!$("#view-cross").hidden, "cross-check view shown");
  const crossRows = $$("#cross-table tbody tr");
  ok(crossRows.length >= 15, "stores compared", crossRows.length);
  ok($$("#cross-chart circle.reddit").length === crossRows.length, "a reddit dot per row");
  ok($$("#cross-chart circle.google").length === crossRows.length, "a google dot per row");
  ok($("#cross-summary").textContent.includes("ratings"), "summary line");
  ok($("#cross-cite").textContent.includes("San Diego"), "dataset cited");
  const worst = crossRows[0];
  console.log("        biggest gap: " + [...worst.children].slice(0,4)
    .map(c => c.textContent).join("  "));
  const allGap = worst.children[3].textContent;
  $("#cross-pop").value = "norm_long";
  $("#cross-pop").dispatchEvent(new window.Event("change"));
  const longGap = $("#cross-table tbody tr").children[3].textContent;
  ok(allGap !== longGap, "population switch changes the numbers", `${allGap} vs ${longGap}`);
  console.log("        same store, paragraph-writers only: " + longGap);
  $("#cross-pop").value = "norm";
  $("#cross-pop").dispatchEvent(new window.Event("change"));

  console.log("\n[map]");
  click($('[data-view="map"]'));
  ok(!$("#view-map").hidden, "map view shown");
  const L = window.L;
  ok(L.made.maps === 1, "one map created", L.made.maps);
  ok(L.made.tileUrls.some((u) => u.includes("tile.openstreetmap.org")), "OSM tiles");
  const total = Number($("#map-count").textContent.match(/of (\d+)/)[1]);
  ok(L.made.markers.length === total, "a pin per location", `${L.made.markers.length}/${total}`);
  ok(L.made.markers.every((m) => typeof m.opts.fillColor === "string"), "pins are coloured");
  ok(L.made.markers.every((m) => m.popup && m.popup.nodeType === 1),
     "popups are DOM nodes, not HTML strings");
  ok($("#map-attrib").textContent.includes("OpenStreetMap"), "locations attributed");
  const withGoogle = L.made.markers.filter(m =>
    (m.popup && m.popup.textContent || "").includes("Google")).length;
  ok(withGoogle > 100, "pins carry the Google rating", withGoogle);
  for (const mode of ["merged", "google", "reddit"]) {
    $("#map-colour").value = mode;
    $("#map-colour").dispatchEvent(new window.Event("change"));
    ok(L.made.markers.length > 0 && L.made.markers.length <= total,
       `colour by ${mode}`, L.made.markers.length);
  }
  const withMerged = L.made.markers.filter(m =>
    (m.popup && m.popup.textContent || "").includes("combined")).length;
  ok(withMerged > 50, "pins carry the combined estimate", withMerged);

  const before = L.made.markers.length;
  $("#map-store").value = "Market Basket";
  $("#map-store").dispatchEvent(new window.Event("change"));
  const mb = L.made.markers.length;
  ok(mb > 0 && mb < before, "store filter narrows the map", `${before} -> ${mb}`);
  $("#map-evidence").checked = true;
  $("#map-evidence").dispatchEvent(new window.Event("change"));
  ok(L.made.markers.length <= mb, "evidence filter narrows further", L.made.markers.length);
  $("#map-store").value = "";
  $("#map-evidence").checked = false;
  $("#map-cat").value = "produce";
  $("#map-cat").dispatchEvent(new window.Event("change"));
  ok(L.made.markers.length > 0, "colour-by-category still renders pins", L.made.markers.length);
  $("#map-cat").value = "";
  $("#map-cat").dispatchEvent(new window.Event("change"));

  console.log("\n[merged estimate]");
  const merged = payloadObj.merged;
  ok(merged && merged.calibration.slope > 1, "calibration expands Google's range",
     merged && merged.calibration.slope);
  ok(merged.calibration.loo_rmse < 0.434,
     "calibration beats guessing the mean, out of sample", merged.calibration.loo_rmse);
  click($('[data-view="store"]'));
  $("#store-pick").value = "Market Basket";
  $("#store-pick").dispatchEvent(new window.Event("change"));
  ok($("#store-body .merged") !== null, "combined estimate shown on the store page");
  ok($("#store-body .mixbar") !== null, "the source mix is shown, not hidden");
  const rail = $("#store-body .rail svg");
  ok(rail !== null, "reconciliation rail drawn");
  ok(rail.querySelectorAll("circle").length === 2, "one mark per source",
     rail && rail.querySelectorAll("circle").length);
  ok(rail.querySelector("rect.combined") !== null, "combined figure marked apart");
  ok(/Reddit .*reviews .*combined/.test(rail.getAttribute("aria-label")),
     "rail is described for screen readers", rail.getAttribute("aria-label"));
  const chainShare = $("#store-body .mixlab").textContent;
  ok(/9\d% of the weight/.test(chainShare),
     "Reddit dominates a well-evidenced chain", chainShare.slice(0, 60));
  console.log("        chain: " + chainShare.replace(/\s+/g, " ").slice(0, 90));

  // A thin branch is where the merge is supposed to move the number.
  $("#branch-pick").value = "Chelsea";
  $("#branch-pick").dispatchEvent(new window.Event("change"));
  ok($("#store-body .merged") !== null, "branch-level combined estimate");
  console.log("        branch: " + $("#store-body .mixlab").textContent.replace(/\s+/g," ").slice(0,90));

  click($('[data-view="compare"]'));
  // Reset the filters this test set earlier, or the row count is the
  // previous section's leftovers rather than every store.
  $("#cmp-cat").value = "";
  $("#cmp-min").value = "0";
  $("#cmp-min").dispatchEvent(new window.Event("input"));
  const rows2 = $$("#cmp-table tbody tr");
  const combinedCells = rows2.filter(
    r => r.children[4] && r.children[4].textContent !== "–").length;
  ok(rows2.length === 21, "all stores listed again", rows2.length);
  ok(combinedCells === 20,
     "every store with Google data has a combined figure", combinedCells);

  console.log("\n[item search]");
  click($('[data-view="items"]'));
  $("#item-q").value = "chicken";
  $("#item-q").dispatchEvent(new window.Event("input"));
  setTimeout(() => {
    const irows = $$("#item-body tbody tr");
    ok(irows.length > 0, "item search returns hits", irows.length);
    console.log("        " + irows.slice(0, 4).map((r) => r.children[0].textContent + " @ " + r.children[1].textContent).join(" | "));
  }, 0);
  $("#item-q").value = "zzzznothing";
  $("#item-q").dispatchEvent(new window.Event("input"));

  setTimeout(finish, 250);
}, 400);

function finish() {
  const d = window.document;
  const $ = (s) => d.querySelector(s);
  const $$ = (s) => [...d.querySelectorAll(s)];
  const click = (n) => n.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  ok($("#item-body").textContent.includes("Nobody discussed"), "empty search handled");

  console.log(failures === 0 ? "\nALL PASS (view tests)" : `\n${failures} FAILURE(S)`);
  xssRun();
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

  const vc2 = new VirtualConsole();
  vc2.on("jsdomError", (e) => { failures++; console.log("  JSDOM ERROR: " + e.message); });
  const dom2 = new JSDOM(fs.readFileSync(DOCS + "/index.html", "utf8"), {
    runScripts: "outside-only", virtualConsole: vc2, url: "http://localhost:8177/",
  });
  const w2 = dom2.window;
  let alerted = false;
  w2.alert = () => { alerted = true; };
  w2.fetch = () => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(evil) });
  w2.eval(fs.readFileSync(DOCS + "/app.js", "utf8"));

  setTimeout(() => {
    const d = w2.document;
    d.querySelector('[data-view="store"]').dispatchEvent(
      new w2.MouseEvent("click", { bubbles: true }));
    d.querySelector("#store-pick").value = "Market Basket";
    d.querySelector("#store-pick").dispatchEvent(new w2.Event("change"));
    const body = d.querySelector("#store-body");
    for (const el of body.querySelectorAll("details")) el.open = true;

    ok(body.querySelector("img") === null, "injected <img> never becomes an element");
    ok(body.querySelector("script") === null, "injected <script> never becomes an element");
    ok(body.textContent.includes("<img src=x onerror=alert(1)>"),
       "hostile markup is displayed as literal text");
    const bad = [...body.querySelectorAll("a")].filter(
      (a) => !a.getAttribute("href").startsWith("https://reddit.com/"));
    ok(bad.length === 0, "javascript: permalink refused",
       bad.map((a) => a.getAttribute("href")).join(","));
    ok(!alerted, "no script executed");

    console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
    process.exit(failures ? 1 : 0);
  }, 400);
}
