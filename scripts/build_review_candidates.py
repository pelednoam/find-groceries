#!/usr/bin/env python3
"""Turn the Google review slice into a stage-2 working set.

    .venv/bin/python scripts/build_review_candidates.py

Writes data/extraction/reviews_working_set.jsonl — the same Candidate shape
stage 2 already consumes, so the extractor, runner, cost ceiling and resume
logic all work unchanged.

Only reviews with real prose are included; a four-star tap with no text costs
a request and yields nothing.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groceries.paths import DATA, EXTRACT_DIR  # noqa: E402
from groceries.reviews import MIN_CHARS, candidates, review_id  # noqa: E402
from groceries.select import write_candidates  # noqa: E402

GL = DATA / "googlelocal"
OUT = EXTRACT_DIR / "reviews_working_set.jsonl"
# doc id -> gmap_id, so extracted claims can be put back at their own shop.
PLACES_OUT = EXTRACT_DIR / "reviews_places.json"


def read(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=GL / "reviews-matched.jsonl.gz")
    parser.add_argument("--meta", type=Path, default=GL / "matched_meta.json")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--places-out", type=Path, default=PLACES_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.reviews, args.meta):
        if not path.exists():
            print(f"missing {path}\nRun scripts/fetch_google_local.py first.")
            return 2

    reviews = read(args.reviews)
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    stores = {b["gmap_id"]: b["store"] for b in meta}
    docs = candidates(reviews, stores)
    if not docs:
        print("refusing to write: no review met the length floor.")
        return 1

    n = write_candidates(docs, args.out)
    # Keep the doc -> place mapping beside it: the branch is a fact of the
    # record here, and stage 2's guess at it will be overwritten.
    mapping = {review_id(r): str(r["gmap_id"]) for r in reviews}
    args.places_out.write_text(json.dumps(mapping), encoding="utf-8")

    by_store: dict[str, int] = {}
    for d in docs:
        by_store[d["stores"][0]] = by_store.get(d["stores"][0], 0) + 1
    for store, count in sorted(by_store.items(), key=lambda kv: -kv[1]):
        print(f"  {store:<22}{count:>7,}")
    print(f"\n{len(reviews):,} reviews, {n:,} with at least {MIN_CHARS} characters")
    print(f"wrote {args.out}\n      {args.places_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
