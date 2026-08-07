/* The shape of docs/verdicts.json.
 *
 * This mirrors what `groceries/site.py:build_payload` emits. The two are kept
 * honest by `tests/test_site.py::TestPayloadContract`, which asserts the real
 * payload has exactly the keys declared here — a field renamed on the Python
 * side and not here would otherwise surface as a blank panel in the browser
 * rather than as an error anywhere.
 *
 * Every string below except the object keys is text a stranger typed on
 * Reddit. `Cell.e[].t` in particular is untrusted: it reaches the DOM through
 * textContent only.
 */

/** Evidence quote backing a cell. */
interface Quote {
  /** claim text — UNTRUSTED */
  t: string;
  /** "2025-04" */
  d: string;
  /** reddit permalink path, validated before use as an href */
  u: string;
  /** high | medium | low */
  c: string;
  /** branch, when the claim named one */
  l?: string;
}

/** Accumulated evidence for one (store, [branch,] category-or-item). */
interface Cell {
  /** claim count */
  n: number;
  /** weighted evidence */
  w: number;
  /** sentiment, -1..+1, shrunk toward 0 */
  s: number;
  /** cheap | fair | expensive — absent when there is too little to say */
  p?: string;
  /** price level, -1 cheap .. +1 expensive */
  pl?: number;
  /** raw counts behind the level */
  pd?: Record<string, number>;
  e: Quote[];
}

interface Totals {
  n: number;
  w: number;
  s: number;
  /** every cell fell below the evidence threshold */
  thin: boolean;
}

/** A physical store, from OpenStreetMap. */
interface Place {
  store: string;
  name: string;
  lat: number;
  lon: number;
  address: string;
  city: string;
  /** "node/473641811" */
  osm: string;
  /** branch key with its own evidence, when one matched confidently */
  branch?: string;
}

interface Corpus {
  claims_file: string;
  working_set?: number;
  documents_extracted?: number;
  note?: string;
}

interface Method {
  shrinkage_k: number;
  default_half_life_years: number;
  min_weight: number;
  max_claims_per_document: number;
  max_claims_per_author_cell: number;
  transient_claims: string;
  half_life_years: Record<string, number>;
  note: string;
}

interface Payload {
  generated_at: string;
  method: Method;
  corpus: Corpus | null;
  places: Place[];
  places_attribution: string;
  totals: Record<string, Totals>;
  /** store -> category -> cell */
  stores: Record<string, Record<string, Cell>>;
  /** store -> branch -> category -> cell */
  branches: Record<string, Record<string, Record<string, Cell>>>;
  /** store -> region -> category -> cell (regions are not branches) */
  regions: Record<string, Record<string, Record<string, Cell>>>;
  /** store -> item -> cell */
  items: Record<string, Record<string, Cell>>;
  /** shopping-list word -> category */
  keywords: Record<string, string>;
  categories: string[];
}

/** One resolved shopping-list term. */
type Resolution =
  | { kind: "none"; term: string; perStore: Record<string, never> }
  | {
      kind: "item" | "category";
      term: string;
      perStore: Record<string, { cell: Cell; label: string }>;
    };

interface Ranked {
  store: string;
  score: number;
  covered: number;
  avg: number;
  wins: string[];
  weak: string[];
}
