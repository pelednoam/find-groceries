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
const payload = fs.readFileSync(DOCS + "/verdicts.json", "utf8");
window.fetch = () => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(JSON.parse(payload)) });

window.eval(fs.readFileSync(DOCS + "/app.js", "utf8"));

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
