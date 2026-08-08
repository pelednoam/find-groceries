/* Reading a receipt, and remembering what it said.
 *
 * Everything here runs in the browser and stays there. A receipt is a record
 * of where someone was and what they eat; it is not this project's to
 * collect. There is no upload, no request, and no telemetry — the CSP pins
 * connect-src to 'self', so the page could not send one if it tried. History
 * lives in localStorage and the reader can wipe it in one click.
 *
 * The vocabulary comes from groceries/receipts.py so it can be tested against
 * real till lines; the parsing lives here so the image never has to move.
 */

interface ReceiptVocabulary {
  abbreviations: Record<string, string>;
  brands: string[];
  units: string[];
  noise: string[];
  price: string;
  quantity: string;
  weighed: string;
  codes: string;
  sizes: string;
}

/** One line the parser believes was a purchase. */
interface ParsedLine {
  /** the raw till text, kept so the reader can check the parse */
  raw: string;
  /** expanded into ordinary words */
  text: string;
  qty: number;
  /** what the shopper paid, when the line carried a price */
  price: number | null;
  /** the term this maps to in the corpus, when one matches */
  match: string | null;
}

interface Receipt {
  /** epoch seconds; the receipt's own date if we found one, else today */
  at: number;
  store: string | null;
  lines: { name: string; qty: number; price: number | null }[];
}

/** What repeated shopping has taught us about one item. */
interface Learned {
  name: string;
  /** receipts it appeared on */
  seen: number;
  /** units bought in total */
  units: number;
  /** epoch seconds of the most recent purchase */
  last: number;
  /** frequency discounted by age — how much it should matter now */
  weight: number;
}

const HISTORY_KEY = "basket.receipts.v1";
/** Shopping habits turn over faster than store reputations do. Eight weeks
 *  halves the weight of a purchase, so a thing bought monthly stays in the
 *  basket and a one-off drops out of it within a season. */
const HABIT_HALF_LIFE_DAYS = 56;
const DAY = 86400;

function vocab(): ReceiptVocabulary {
  return data().receipts;
}

/* ── parsing ────────────────────────────────────────────────────────── */

function isNoise(line: string): boolean {
  const lowered = line.toLowerCase();
  return vocab().noise.some((p) => new RegExp(p, "i").test(lowered));
}

/** Expand a till line into ordinary words. Mirrors `receipts.expand`. */
function expandLine(text: string): string {
  const v = vocab();
  let line = text.replace(new RegExp(v.price, "i"), "");
  line = line.replace(new RegExp(v.weighed, "gi"), " ");
  line = line.replace(new RegExp(v.sizes, "gi"), " ");
  line = line.replace(new RegExp(v.codes, "g"), " ");
  const brands = new Set(v.brands);
  const units = new Set(v.units);
  const words: string[] = [];
  for (const raw of line.split(/[^A-Za-z]+/)) {
    const word = raw.toLowerCase();
    if (!word || brands.has(word) || units.has(word)) continue;
    const expanded = v.abbreviations[word] ?? word;
    if (expanded.length < 2) continue;
    for (const part of expanded.split(" ")) {
      if (words[words.length - 1] !== part) words.push(part);
    }
  }
  return words.join(" ");
}

/** The term in our vocabulary this line is about, if any.
 *
 * Longest match wins, so "peanut butter" beats "butter" — a till line is
 * terse enough that a short accidental match is a real risk.
 */
function matchTerm(text: string): string | null {
  if (!text) return null;
  let best: string | null = null;
  for (const word of Object.keys(data().keywords)) {
    if (text.includes(word) && (!best || word.length > best.length)) best = word;
  }
  for (const items of Object.values(data().items)) {
    for (const name of Object.keys(items)) {
      if (text.includes(name) && (!best || name.length > best.length)) best = name;
    }
  }
  return best;
}

function parseLines(receiptText: string): ParsedLine[] {
  const v = vocab();
  const priceRe = new RegExp(v.price, "i");
  const qtyRe = new RegExp(v.quantity, "i");
  const out: ParsedLine[] = [];
  for (const raw of receiptText.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.length < 3 || isNoise(line)) continue;
    const priceMatch = priceRe.exec(line);
    const price = priceMatch?.[1]
      ? Number(priceMatch[1].replace(/[$\s]/g, "").replace(",", "."))
      : null;
    const qtyMatch = qtyRe.exec(line);
    const qty = qtyMatch?.[1] ? Math.min(20, Number(qtyMatch[1])) : 1;
    const text = expandLine(line);
    // A line with neither a price nor a recognisable word is a header.
    if (!text || (price === null && !matchTerm(text))) continue;
    out.push({ raw: line, text, qty, price, match: matchTerm(text) });
  }
  return out;
}

/** Which shop printed this, if its name is on the slip. */
function detectStore(receiptText: string): string | null {
  const head = receiptText.slice(0, 400).toLowerCase();
  let best: string | null = null;
  for (const store of Object.keys(data().stores)) {
    const key = store.toLowerCase().replace(/[^a-z]/g, "");
    const flat = head.replace(/[^a-z]/g, "");
    if (key.length > 3 && flat.includes(key) && (!best || store.length > best.length)) {
      best = store;
    }
  }
  return best;
}

/** The date on the slip, so an old receipt is weighted as old. */
function detectDate(receiptText: string): number | null {
  const m = /\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b/.exec(receiptText);
  if (!m || !m[1] || !m[2] || !m[3]) return null;
  const year = Number(m[3].length === 2 ? "20" + m[3] : m[3]);
  const stamp = Date.UTC(year, Number(m[1]) - 1, Number(m[2])) / 1000;
  const now = Date.now() / 1000;
  // A misread digit can produce 2085; refuse anything not plausibly a shop.
  return stamp > now - 5 * 365 * DAY && stamp < now + DAY ? stamp : null;
}

/* ── history ────────────────────────────────────────────────────────── */

function loadHistory(): Receipt[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((r): r is Receipt =>
      typeof r === "object" && r !== null && Array.isArray((r as Receipt).lines));
  } catch {
    // A corrupt or full localStorage must not take the site down with it.
    return [];
  }
}

function saveHistory(history: Receipt[]): boolean {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-60)));
    return true;
  } catch {
    return false;
  }
}

/** Fold every receipt into a per-item habit, discounted by age.
 *
 * Frequency alone would keep something bought weekly two years ago ahead of
 * something bought weekly since March. The same exponential decay the corpus
 * uses on claims applies here, just far faster: habits turn over in months,
 * store reputations in years.
 */
function learn(history: Receipt[], now = Date.now() / 1000): Learned[] {
  const by = new Map<string, Learned>();
  for (const receipt of history) {
    const ageDays = Math.max(0, (now - receipt.at) / DAY);
    const decay = Math.pow(0.5, ageDays / HABIT_HALF_LIFE_DAYS);
    for (const line of receipt.lines) {
      const found = by.get(line.name) ?? {
        name: line.name, seen: 0, units: 0, last: 0, weight: 0,
      };
      found.seen += 1;
      found.units += line.qty;
      found.last = Math.max(found.last, receipt.at);
      found.weight += line.qty * decay;
      by.set(line.name, found);
    }
  }
  return [...by.values()].sort((a, b) => b.weight - a.weight);
}

/** Turn what we have learned into a basket.
 *
 * Quantity is the rounded habit weight, so something bought every week
 * carries more of the ranking than something bought once in April.
 */
function basketFromHistory(learned: Learned[], limit = 12): BasketItem[] {
  return learned
    .filter((l) => l.weight >= 0.35)
    .slice(0, limit)
    .map((l) => ({ name: l.name, qty: Math.max(1, Math.min(5, Math.round(l.weight))) }));
}

/* ── OCR ────────────────────────────────────────────────────────────── */

/* Tesseract is loaded on demand, from this origin, the first time someone
 * scans an image. It is 3MB of WebAssembly plus a language model, which is a
 * lot to hand to every reader who only wanted to know where to buy milk — and
 * running it here rather than on a server is the whole point: the photograph
 * never leaves the device. */
declare const Tesseract: {
  recognize(
    image: unknown, lang: string,
    opts: { logger?: (m: { status: string; progress: number }) => void;
            workerPath?: string; corePath?: string; langPath?: string;
            workerBlobURL?: boolean },
  ): Promise<{ data: { text: string } }>;
} | undefined;

let ocrLoading: Promise<void> | null = null;

function loadOcr(): Promise<void> {
  if (typeof Tesseract !== "undefined") return Promise.resolve();
  if (ocrLoading) return ocrLoading;
  ocrLoading = new Promise((resolve, reject) => {
    const tag = document.createElement("script");
    tag.src = "vendor/tesseract/tesseract.min.js";
    tag.onload = () => resolve();
    tag.onerror = () => reject(new Error("could not load the reader"));
    document.head.append(tag);
  });
  return ocrLoading;
}

/** Flatten a photograph into something OCR can read.
 *
 * Thermal paper photographed under a kitchen light is low-contrast, curved
 * and warm-tinted. Greyscale plus a hard contrast stretch is crude but it is
 * the difference between a usable parse and noise, and it costs nothing.
 */
function preprocess(img: HTMLImageElement): HTMLCanvasElement {
  const maxWidth = 1400;
  const scale = Math.min(1, maxWidth / img.naturalWidth);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const px = frame.data;
  let total = 0;
  for (let i = 0; i < px.length; i += 4) {
    const grey = 0.299 * (px[i] ?? 0) + 0.587 * (px[i + 1] ?? 0) + 0.114 * (px[i + 2] ?? 0);
    px[i] = px[i + 1] = px[i + 2] = grey;
    total += grey;
  }
  // Threshold about the image's own mean rather than a fixed value, so a
  // dim photo and a bright one both come out legible.
  const mean = total / (px.length / 4);
  for (let i = 0; i < px.length; i += 4) {
    const v = px[i] ?? 0;
    const stretched = v < mean * 0.86 ? 0 : v > mean * 1.02 ? 255 : (v - mean * 0.86) * 6;
    px[i] = px[i + 1] = px[i + 2] = Math.max(0, Math.min(255, stretched));
  }
  ctx.putImageData(frame, 0, 0);
  return canvas;
}

async function readImage(
  file: File, onProgress: (pct: number, label: string) => void,
): Promise<string> {
  onProgress(2, "Loading the reader…");
  await loadOcr();
  if (typeof Tesseract === "undefined") throw new Error("the reader did not load");

  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error("that file is not an image"));
      el.src = url;
    });
    onProgress(12, "Cleaning up the photo…");
    const canvas = preprocess(img);
    const result = await Tesseract.recognize(canvas, "eng", {
      workerPath: "vendor/tesseract/worker.min.js",
      // Load the worker from this origin rather than the blob: URL the
      // library defaults to — one fewer thing the CSP has to allow.
      workerBlobURL: false,
      corePath: "vendor/tesseract/core",
      langPath: "vendor/tesseract/lang",
      logger: (m) => {
        if (m.status === "recognizing text") {
          onProgress(15 + Math.round(m.progress * 80), "Reading the receipt…");
        }
      },
    });
    return result.data.text;
  } finally {
    URL.revokeObjectURL(url);
  }
}

/* ── the scanner UI ─────────────────────────────────────────────────── */

/** The file waiting to be read, if the reader chose a photo. */
let staged: File | null = null;
/** The last parse, awaiting confirmation. Nothing is remembered until the
 *  reader has seen what we think the receipt said and agreed to it. */
let pending: ParsedLine[] = [];
/** Which of those lines are ticked. */
let accepted = new Set<number>();
let pendingStore: string | null = null;
let pendingAt: number | null = null;

function scanProgress(pct: number, label: string): void {
  const box = need<HTMLElement>("#scan-progress");
  box.hidden = false;
  need<HTMLElement>("#scan-bar").style.width = Math.max(0, Math.min(100, pct)) + "%";
  need("#scan-progress-label").textContent = label;
}

function scanHint(text: string): void {
  need("#scan-hint").textContent = text;
}

function openScanner(): void {
  const modal = need<HTMLElement>("#scan-modal");
  modal.hidden = false;
  document.body.classList.add("modal-open");
  need<HTMLElement>("#scan-close").focus();
}

function closeScanner(): void {
  need<HTMLElement>("#scan-modal").hidden = true;
  document.body.classList.remove("modal-open");
  need<HTMLElement>("#scan-open").focus();
}

/** Stage a photo without reading it — the reader still has to press the
 *  button. Loading 3MB of OCR because someone dropped a file by accident
 *  would be rude on a phone. */
function stageFile(file: File): void {
  if (!file.type.startsWith("image/")) {
    scanHint("That is not an image. A photo of the slip, or paste the text.");
    return;
  }
  staged = file;
  need(".dz-title").textContent = file.name;
  need(".dz-sub").textContent = Math.round(file.size / 1024) + " KB · ready to read";
  need<HTMLElement>("#dropzone").classList.add("loaded");
  scanHint("");
}

function renderReview(): void {
  const list = need("#scan-lines");
  clear(list);
  need<HTMLElement>("#scan-review").hidden = pending.length === 0;
  need<HTMLElement>("#scan-save").hidden = pending.length === 0;

  const matched = pending.filter((l) => l.match).length;
  need("#scan-found").textContent = pending.length
    ? `${pending.length} line${pending.length === 1 ? "" : "s"}, ${matched} of them `
      + "something people talk about. Untick anything we misread."
    : "";

  for (const [i, line] of pending.entries()) {
    const li = el("li", { class: line.match ? "hit" : "miss" });
    const box = el("input", { type: "checkbox", class: "tick",
      "aria-label": `Remember ${line.match ?? line.text}` }) as HTMLInputElement;
    box.checked = accepted.has(i);
    box.addEventListener("change", () => {
      if (box.checked) accepted.add(i);
      else accepted.delete(i);
      renderSaveCount();
    });
    const body = el("span", { class: "rv-body" });
    body.append(el("span", { class: "rv-name", text: line.match ?? line.text }));
    body.append(el("span", { class: "rv-raw", text: line.raw }));
    li.append(box, body);
    if (line.qty > 1) li.append(el("span", { class: "rv-qty", text: "×" + line.qty }));
    // The price is the reader's own, from their own receipt. We show it back
    // and never do anything else with it — this project has no price data.
    if (line.price !== null) {
      li.append(el("span", { class: "rv-price", text: "$" + line.price.toFixed(2) }));
    }
    list.append(li);
  }
  renderSaveCount();
}

function renderSaveCount(): void {
  const n = accepted.size;
  need("#scan-save").textContent = n ? `Remember ${n} item${n === 1 ? "" : "s"}` : "Nothing ticked";
  need<HTMLButtonElement>("#scan-save").disabled = n === 0;
}

function ingest(text: string): void {
  pending = parseLines(text);
  // Default to remembering the lines we could match; an unmatched line is
  // usually a misread, and the reader can always tick it back on.
  accepted = new Set(pending.flatMap((l, i) => (l.match ? [i] : [])));
  pendingStore = detectStore(text);
  pendingAt = detectDate(text);
  renderReview();
  if (!pending.length) {
    scanHint("Nothing on that looked like a grocery line. If it was a photo, "
      + "try a straighter, brighter one, or paste the text.");
    return;
  }
  const where = pendingStore ? ` at ${pendingStore}` : "";
  scanHint(`Read ${pending.length} lines${where}. Nothing has left this device.`);
}

async function runScan(): Promise<void> {
  const typed = need<HTMLTextAreaElement>("#scan-text").value.trim();
  if (!staged && !typed) {
    scanHint("Choose a photo, or paste the text of the slip.");
    return;
  }
  const button = need<HTMLButtonElement>("#scan-run");
  button.disabled = true;
  try {
    if (staged) {
      const text = await readImage(staged, scanProgress);
      scanProgress(100, "Done");
      need<HTMLTextAreaElement>("#scan-text").value = text;
      ingest(text);
    } else {
      ingest(typed);
    }
  } catch (err) {
    need<HTMLElement>("#scan-progress").hidden = true;
    scanHint("Could not read that: " + (err instanceof Error ? err.message : String(err))
      + ". You can paste the text instead.");
  } finally {
    button.disabled = false;
  }
}

/** Commit the ticked lines to history, then let them move the basket. */
function commitScan(): void {
  const lines = pending
    .filter((_, i) => accepted.has(i))
    .map((l) => ({ name: l.match ?? l.text, qty: l.qty, price: l.price }));
  if (!lines.length) return;

  const history = loadHistory();
  history.push({
    at: pendingAt ?? Math.floor(Date.now() / 1000),
    store: pendingStore,
    lines,
  });
  const saved = saveHistory(history);
  applyHistory();
  closeScanner();
  resetScanner();
  const where = pendingStore ? ` from ${pendingStore}` : "";
  need("#scan-status").textContent = saved
    ? `Remembered ${lines.length} item${lines.length === 1 ? "" : "s"}${where}.`
    : "Read the receipt, but this browser would not let us save it "
      + "(private mode blocks storage), so it is in your basket for now only.";
}

function resetScanner(): void {
  staged = null;
  pending = [];
  accepted = new Set();
  pendingStore = null;
  pendingAt = null;
  need<HTMLTextAreaElement>("#scan-text").value = "";
  need<HTMLElement>("#scan-review").hidden = true;
  need<HTMLElement>("#scan-save").hidden = true;
  need<HTMLElement>("#scan-progress").hidden = true;
  need<HTMLElement>("#dropzone").classList.remove("loaded");
  need(".dz-title").textContent = "Drop a photo, or choose one";
  need(".dz-sub").textContent = "JPG or PNG · read on your device, never uploaded";
  scanHint("");
}

/* ── what history does to the page ──────────────────────────────────── */

const ago = (seconds: number): string => {
  const days = Math.round((Date.now() / 1000 - seconds) / DAY);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 14) return days + " days ago";
  if (days < 60) return Math.round(days / 7) + " weeks ago";
  return Math.round(days / 30) + " months ago";
};

function renderLearned(learned: Learned[], receipts: number): void {
  const panel = need<HTMLElement>("#learned-panel");
  panel.hidden = learned.length === 0;
  if (!learned.length) return;

  const list = need("#learned-list");
  clear(list);
  const top = learned.slice(0, 10);
  const most = top[0]?.weight ?? 1;
  for (const item of top) {
    const li = el("li");
    li.append(el("span", { class: "lr-name", text: item.name }));
    const track = el("span", { class: "lr-track" });
    track.append(el("span", { class: "lr-fill",
      style: `width:${Math.max(6, Math.round((item.weight / most) * 100))}%` }));
    li.append(track);
    li.append(el("span", { class: "lr-meta",
      text: `${item.seen}× · ${ago(item.last)}` }));
    list.append(li);
  }
  need("#learned-note").textContent =
    `From ${receipts} receipt${receipts === 1 ? "" : "s"} on this device. `
    + "Recent shopping counts for more — a purchase is worth half as much "
    + `after ${HABIT_HALF_LIFE_DAYS} days. Nothing here has been sent anywhere.`;
}

/** Rebuild the basket and the habit panel from stored receipts.
 *
 * Called on load and after every scan, so the ranking below always reflects
 * what this shopper actually buys rather than the example list.
 */
function applyHistory(): void {
  const history = loadHistory();
  const learned = learn(history);
  renderLearned(learned, history.length);
  const built = basketFromHistory(learned);
  if (built.length) {
    basket = built;
    renderBasket();
    runList();
  }
}

function initScanner(): void {
  need("#scan-open").addEventListener("click", openScanner);
  need("#scan-close").addEventListener("click", closeScanner);
  need("#scan-run").addEventListener("click", () => void runScan());
  need("#scan-save").addEventListener("click", commitScan);

  const modal = need<HTMLElement>("#scan-modal");
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeScanner();
  });
  document.addEventListener("keydown", (e) => {
    if ((e as KeyboardEvent).key === "Escape" && !modal.hidden) closeScanner();
  });

  const zone = need<HTMLElement>("#dropzone");
  const file = need<HTMLInputElement>("#scan-file");
  activatable(zone, "Choose a receipt photo", () => file.click());
  file.addEventListener("change", () => {
    const chosen = file.files?.[0];
    if (chosen) stageFile(chosen);
  });
  for (const type of ["dragenter", "dragover"]) {
    zone.addEventListener(type, (e) => {
      e.preventDefault();
      zone.classList.add("over");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    zone.addEventListener(type, () => zone.classList.remove("over"));
  }
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    const dropped = (e as DragEvent).dataTransfer?.files?.[0];
    if (dropped) stageFile(dropped);
  });
  // A screenshot on the clipboard is how most people have a receipt to hand.
  document.addEventListener("paste", (e) => {
    if (modal.hidden) return;
    const item = Array.from((e as ClipboardEvent).clipboardData?.items ?? [])
      .find((i) => i.type.startsWith("image/"));
    const asFile = item?.getAsFile();
    if (asFile) {
      e.preventDefault();
      stageFile(asFile);
    }
  });

  need("#history-clear").addEventListener("click", () => {
    try {
      localStorage.removeItem(HISTORY_KEY);
    } catch {
      // Nothing was stored, so nothing to forget.
    }
    need<HTMLElement>("#learned-panel").hidden = true;
    need("#scan-status").textContent = "Forgotten. Nothing about your shopping is kept.";
  });

  applyHistory();
}
