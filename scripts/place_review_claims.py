#!/usr/bin/env python3
"""Put each review-derived claim back at the shop it came from.

    .venv/bin/python scripts/place_review_claims.py

Stage 2 guesses a `location` out of the prose, because that is all a Reddit
comment offers. A review is attached to one listing as a matter of record, so
the guess is strictly worse than the truth — and a wrong branch silently
splits one store's evidence in two.

Branch names are taken from the OpenStreetMap pin the listing matches, so
they are the *same* names the Reddit side uses. Without that the two sources
would each have their own spelling of every branch and nothing would join.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groceries.crosscheck import match_to_places  # noqa: E402
from groceries.locations import attach_branches  # noqa: E402
from groceries.jsonl import read_jsonl, write_atomic  # noqa: E402
from groceries.paths import DATA, EXTRACT_DIR  # noqa: E402

GL = DATA / "googlelocal"
CLAIMS = EXTRACT_DIR / "reviews_claims.jsonl"
PLACES = EXTRACT_DIR / "reviews_places.json"
OUT = EXTRACT_DIR / "reviews_claims_placed.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=CLAIMS)
    parser.add_argument("--places", type=Path, default=PLACES)
    parser.add_argument("--meta", type=Path, default=GL / "matched_meta.json")
    parser.add_argument("--locations", type=Path, default=DATA / "locations.json")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verdicts", type=Path,
                        default=EXTRACT_DIR / "store_verdicts.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.claims, args.places, args.meta, args.locations):
        if not path.exists():
            print(f"missing {path}")
            return 2

    claims, bad = read_jsonl(args.claims)
    doc_to_place: dict[str, str] = json.loads(args.places.read_text(encoding="utf-8"))
    google_places = json.loads(args.meta.read_text(encoding="utf-8"))
    osm_places = json.loads(args.locations.read_text(encoding="utf-8"))["places"]

    # gmap_id -> osm id -> branch name, so both sources name a branch alike.
    # The branch names are the Reddit side's own, linked to pins the same way
    # build_payload does it — data/locations.json carries no branch field of
    # its own, that association is derived.
    verdicts = json.loads(args.verdicts.read_text(encoding="utf-8"))
    branch_keys = {
        store: list(by_branch)
        for store, by_branch in verdicts.get("branch_totals", {}).items()
    }
    osm_to_branch = attach_branches(osm_places, branch_keys)
    gmap_to_osm = match_to_places(google_places, osm_places)
    gmap_to_branch = {
        g: osm_to_branch.get(o, "") for g, o in gmap_to_osm.items()
    }

    placed = 0
    overruled = 0
    out = []
    for claim in claims:
        gmap = doc_to_place.get(str(claim.get("source_id", "")), "")
        branch = gmap_to_branch.get(gmap, "")
        if claim.get("location") and claim["location"] != branch:
            overruled += 1
        claim["location"] = branch
        claim["gmap_id"] = gmap
        if branch:
            placed += 1
        out.append(claim)

    n = write_atomic(
        args.out, (json.dumps(c, ensure_ascii=False) + "\n" for c in out)
    )
    print(f"{n:,} claims  ({bad} unparseable lines skipped)")
    print(f"  placed at a named branch      {placed:,} ({placed / max(n, 1):.0%})")
    print(f"  model's guess overruled       {overruled:,}")
    print(f"  listings matched to a pin     {len(gmap_to_osm)} of {len(google_places)}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
