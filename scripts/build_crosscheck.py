#!/usr/bin/env python3
"""Aggregate the Google Local reviews into a cross-check block.

    .venv/bin/python scripts/build_crosscheck.py

Reads the extracted Massachusetts slice and writes data/crosscheck.json,
which build_site.py folds into the payload.

Statistics only. No review text, user name or user id is read out of the
source file, let alone published — see groceries/crosscheck.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groceries.crosscheck import build  # noqa: E402
from groceries.jsonl import write_atomic  # noqa: E402
from groceries.paths import DATA  # noqa: E402

GL = DATA / "googlelocal"
REVIEWS = GL / "reviews-matched.jsonl.gz"
META = GL / "matched_meta.json"
LOCATIONS = DATA / "locations.json"
OUT = DATA / "crosscheck.json"

# The only fields that leave the source file. `text` is replaced by its
# length at the boundary, so no review prose is held in memory beyond the
# line being parsed, let alone written anywhere.
KEEP = ("store", "gmap_id", "rating", "time")


def read_reviews(path: Path) -> list[dict[str, Any]]:
    """Read ratings and timestamps. Review text is never loaded."""
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("rating") is None or raw.get("time") is None:
                continue
            row = {k: raw[k] for k in KEEP if k in raw}
            row["text_len"] = len((raw.get("text") or "").strip())
            rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=REVIEWS)
    parser.add_argument("--meta", type=Path, default=META)
    parser.add_argument("--locations", type=Path, default=LOCATIONS)
    parser.add_argument("--out", type=Path, default=OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path, hint in (
        (args.reviews, "scripts/fetch_google_local.py"),
        (args.meta, "scripts/fetch_google_local.py"),
        (args.locations, "scripts/fetch_store_locations.py"),
    ):
        if not path.exists():
            print(f"missing {path}")
            print(f"Run {hint} first.")
            return 2

    reviews = read_reviews(args.reviews)
    google_places = json.loads(args.meta.read_text(encoding="utf-8"))
    osm_places = json.loads(args.locations.read_text(encoding="utf-8"))["places"]

    block = build(reviews, google_places, osm_places)
    write_atomic(args.out, [json.dumps(block, indent=1)])

    print(f"{block['n_reviews']:,} ratings over {block['n_locations']} locations")
    print(f"{block['n_matched_to_map']} matched to an OSM pin")
    print(f"coverage {block['coverage']}, median {block['median_date']}\n")
    print(f"{'store':<22}{'google':>8}{'n':>8}{'thin':>6}")
    print("-" * 44)
    for store, r in sorted(block["stores"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"{store:<22}{r['mean']:>8.2f}{r['n']:>8,}{'  yes' if r['thin'] else '':>6}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
