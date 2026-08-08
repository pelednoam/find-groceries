#!/usr/bin/env python3
"""Extract the Massachusetts grocery slice of the UCSD Google Local dataset.

    .venv/bin/python scripts/fetch_google_local.py

The reviews file is 785MB gzipped and holds 10.4M reviews for every kind of
business in the state; we want 135k of them. It is therefore streamed and
filtered rather than downloaded — only the matched subset (~8MB) is written.

Two passes:
  1. metadata (20MB) -> the businesses that are a known grocery store
  2. reviews (785MB, streamed) -> the reviews belonging to those businesses

Data: Google Local review data, McAuley Lab, UC San Diego. Research use with
citation; see groceries/crosscheck.py. Only ratings and timestamps are used
downstream — review text is never published.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groceries.locations import BBOX, match_store  # noqa: E402
from groceries.paths import DATA  # noqa: E402

BASE: Final = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal"
STATE: Final = "Massachusetts"
OUT_DIR = DATA / "googlelocal"

# Name matching alone is not enough: "Costco Tire Center", "Haymarket
# Garage", "Shaw's Pharmacy" and an Eastern Bank branch trading as "Shaw's
# East Boston" all match a store pattern. The primary category is the gate,
# exactly as grocery context gates "Target" and "Haymarket" in prose.
GROCERY_CATEGORIES: Final[frozenset[str]] = frozenset({
    "Supermarket", "Warehouse club", "Warehouse store", "Produce market",
    "Pasta shop", "Health food store", "Cheese shop", "Department store",
    "Discount store", "Natural goods store", "Butcher shop", "Fish market",
    "Farmers' market",
})
# "Japanese grocery store", "Korean grocery store", ... — match the shape
# rather than enumerating nationalities.
GROCERY_SUFFIX: Final = re.compile(r"(grocery store|grocer|supermarket|food market)$", re.I)


def is_grocery(primary: str) -> bool:
    return primary in GROCERY_CATEGORIES or bool(GROCERY_SUFFIX.search(primary))


def in_area(business: dict[str, Any]) -> bool:
    lat, lon = business.get("latitude"), business.get("longitude")
    if not isinstance(lat, float | int) or not isinstance(lon, float | int):
        return False
    south, west, north, east = BBOX
    return south <= lat <= north and west <= lon <= east


def stream_lines(url: str, timeout: float) -> Iterator[str]:
    """Yield decompressed lines without holding the file on disk."""
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with gzip.open(response.raw, "rt", encoding="utf-8", errors="replace") as fh:
            yield from fh


def select_businesses(lines: Iterator[str]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for line in lines:
        try:
            b = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not in_area(b):
            continue
        store = match_store(b.get("name") or "")
        if store is None:
            continue
        primary = (b.get("category") or ["<none>"])[0]
        if not is_grocery(primary):
            continue
        kept.append({**b, "store": store})
    return kept


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--state", default=STATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "matched_meta.json"
    reviews_path = args.out_dir / "reviews-matched.jsonl.gz"

    print(f"pass 1: metadata for {args.state} …")
    try:
        businesses = select_businesses(
            stream_lines(f"{BASE}/meta-{args.state}.json.gz", args.timeout)
        )
    except requests.RequestException as exc:
        print(f"download failed: {exc}")
        return 1
    if not businesses:
        print("refusing to continue: matched no known grocery stores.")
        return 1
    meta_path.write_text(json.dumps(businesses, indent=1), encoding="utf-8")
    counts = collections.Counter(b["store"] for b in businesses)
    print(f"  {len(businesses)} locations across {len(counts)} stores")

    wanted = {b["gmap_id"]: b["store"] for b in businesses}
    print(f"\npass 2: streaming reviews (785MB, ~4 min) …")
    per: collections.Counter[str] = collections.Counter()
    scanned = kept = 0
    # Write to a partial file and rename only on success. A connection that
    # drops at 80% otherwise leaves a perfectly readable gzip that every
    # later stage treats as the whole corpus.
    partial = reviews_path.with_suffix(reviews_path.suffix + ".part")
    try:
        with gzip.open(partial, "wt", encoding="utf-8") as out:
            for line in stream_lines(f"{BASE}/review-{args.state}.json.gz", args.timeout):
                scanned += 1
                if scanned % 2_000_000 == 0:
                    print(f"  scanned {scanned:,}, kept {kept:,}", flush=True)
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                store = wanted.get(r.get("gmap_id"))
                if store is None:
                    continue
                out.write(json.dumps({**r, "store": store}, ensure_ascii=False) + "\n")
                kept += 1
                per[store] += 1
    except requests.RequestException as exc:
        partial.unlink(missing_ok=True)
        print(f"download failed after {kept:,} reviews: {exc}")
        print("Partial output discarded; rerun to start again.")
        return 1
    if kept == 0:
        partial.unlink(missing_ok=True)
        print("refusing to write: the stream matched no known store.")
        return 1
    partial.replace(reviews_path)

    print(f"\nscanned {scanned:,}, kept {kept:,}")
    for store, n in per.most_common():
        print(f"  {store:<22}{n:>8,}")
    print(f"\nwrote {meta_path}\n      {reviews_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
