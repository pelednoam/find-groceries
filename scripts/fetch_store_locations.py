#!/usr/bin/env python3
"""Fetch store coordinates from OpenStreetMap.

    .venv/bin/python scripts/fetch_store_locations.py

Writes data/locations.json, which build_site.py folds into the payload. Run it
rarely — stores do not move — and commit nothing: the file is regenerable and
lives under the gitignored data/ directory.

Data © OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groceries.jsonl import write_atomic  # noqa: E402
from groceries.locations import (  # noqa: E402
    ATTRIBUTION,
    OVERPASS_URL,
    QUERY,
    extract_places,
)
from groceries.paths import DATA  # noqa: E402

LOCATIONS = DATA / "locations.json"
USER_AGENT = "find-groceries/1.0 (+https://github.com/pelednoam/find-groceries)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=LOCATIONS)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--from-file", type=Path, default=None,
        help="parse a saved Overpass response instead of calling the API",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.from_file:
        raw = json.loads(args.from_file.read_text(encoding="utf-8"))
    else:
        print(f"querying {OVERPASS_URL} …")
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": QUERY},
                headers={"User-Agent": USER_AGENT},
                timeout=args.timeout,
            )
            response.raise_for_status()
            raw = response.json()
        except requests.RequestException as exc:
            print(f"Overpass request failed: {exc}")
            print("Overpass is rate-limited and sometimes busy; try again shortly.")
            return 1

    places = extract_places(raw)
    if not places:
        print("refusing to write: matched no known stores.")
        print("Either the query changed or Overpass returned an error document.")
        return 1

    by_store: dict[str, int] = {}
    for place in places:
        by_store[place["store"]] = by_store.get(place["store"], 0) + 1
    for store, n in sorted(by_store.items(), key=lambda kv: -kv[1]):
        print(f"  {store:<22}{n:>4}")

    payload = {"attribution": ATTRIBUTION, "places": places}
    write_atomic(args.out, [json.dumps(payload, indent=1)])
    print(f"\n{len(places)} locations across {len(by_store)} stores -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
