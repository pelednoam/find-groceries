#!/usr/bin/env python3
"""Build the published site's data payload from the stage-3 verdicts.

    .venv/bin/python scripts/build_site.py

Writes docs/verdicts.json, which GitHub Pages serves alongside docs/index.html.
Note the filename: the repo gitignores `data/`, so a `docs/data/` directory
would be silently dropped from the commit and the published site would 404.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groceries.paths import DATA, ROOT, VERDICTS  # noqa: E402
from groceries.site import build_payload, write_payload  # noqa: E402

SITE_DATA = ROOT / "docs" / "verdicts.json"
LOCATIONS = DATA / "locations.json"
CROSSCHECK = DATA / "crosscheck.json"
REVIEW_VERDICTS = DATA / "extraction" / "review_verdicts.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", type=Path, default=VERDICTS)
    parser.add_argument("--out", type=Path, default=SITE_DATA)
    parser.add_argument("--locations", type=Path, default=LOCATIONS)
    parser.add_argument("--crosscheck", type=Path, default=CROSSCHECK)
    parser.add_argument("--review-verdicts", type=Path, default=REVIEW_VERDICTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.verdicts.exists():
        print(f"no verdicts at {args.verdicts}")
        print("Run scripts/aggregate_claims.py first.")
        return 2

    verdicts = json.loads(args.verdicts.read_text(encoding="utf-8"))
    locations = None
    if args.locations.exists():
        locations = json.loads(args.locations.read_text(encoding="utf-8"))
    else:
        # The site is useful without a map; say so rather than failing.
        print(f"no {args.locations} — building without the map view.")
        print("Run scripts/fetch_store_locations.py to add it.")
    crosscheck = None
    if args.crosscheck.exists():
        crosscheck = json.loads(args.crosscheck.read_text(encoding="utf-8"))
    else:
        print(f"no {args.crosscheck} — building without the Google cross-check.")
    reviews = None
    if args.review_verdicts.exists():
        reviews = json.loads(args.review_verdicts.read_text(encoding="utf-8"))
    else:
        print(f"no {args.review_verdicts} — building without review-derived claims.")
    payload = build_payload(verdicts, locations, crosscheck, reviews)
    written = write_payload(payload, args.out)

    before = args.verdicts.stat().st_size
    print(f"{len(payload['stores'])} stores, "
          f"{sum(len(v) for v in payload['branches'].values())} branches, "
          f"{sum(len(v) for v in payload['regions'].values())} regions, "
          f"{sum(len(v) for v in payload['items'].values())} items, "
          f"{len(payload['places'])} mapped locations "
          f"({sum(1 for p in payload['places'] if p.get('branch'))} linked to a branch)")
    rv = payload.get("reviews")
    if rv:
        cats = sum(len(v) for v in rv["categories"].values())
        print(f"review claims merged: {len(rv['stores'])} stores, {cats} store-categories, "
              f"calibration slope {rv['calibration']['slope']}")
    cc = payload.get("crosscheck")
    if cc:
        print(f"cross-check: {cc['n_reviews']:,} Google ratings, "
              f"{len(cc['stores'])} stores, {cc['n_matched_to_map']} pins")
    print(f"{before / 1e6:.1f}MB verdicts -> {written / 1e6:.1f}MB payload "
          f"({written / before:.0%})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
