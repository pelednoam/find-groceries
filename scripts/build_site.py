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

from groceries.paths import ROOT, VERDICTS  # noqa: E402
from groceries.site import build_payload, write_payload  # noqa: E402

SITE_DATA = ROOT / "docs" / "verdicts.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", type=Path, default=VERDICTS)
    parser.add_argument("--out", type=Path, default=SITE_DATA)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.verdicts.exists():
        print(f"no verdicts at {args.verdicts}")
        print("Run scripts/aggregate_claims.py first.")
        return 2

    verdicts = json.loads(args.verdicts.read_text(encoding="utf-8"))
    payload = build_payload(verdicts)
    written = write_payload(payload, args.out)

    before = args.verdicts.stat().st_size
    print(f"{len(payload['stores'])} stores, "
          f"{sum(len(v) for v in payload['branches'].values())} branches, "
          f"{sum(len(v) for v in payload['regions'].values())} regions, "
          f"{sum(len(v) for v in payload['items'].values())} items")
    print(f"{before / 1e6:.1f}MB verdicts -> {written / 1e6:.1f}MB payload "
          f"({written / before:.0%})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
