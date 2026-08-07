#!/usr/bin/env python3
"""CLI for stage 3 — roll claims into per-store verdicts.

    .venv/bin/python scripts/aggregate_claims.py
    .venv/bin/python scripts/aggregate_claims.py --store "Market Basket"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groceries.aggregate import (  # noqa: E402
    aggregate,
    format_store,
    format_totals,
    read_claims,
    write_verdicts,
)

ROOT = Path(__file__).resolve().parent.parent
EXTRACT_DIR = ROOT / "data" / "extraction"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=EXTRACT_DIR / "claims.jsonl")
    parser.add_argument("--out", type=Path, default=EXTRACT_DIR / "store_verdicts.json")
    parser.add_argument("--store", default=None)
    parser.add_argument("--min-weight", type=float, default=1.0)
    args = parser.parse_args(argv)

    claims, dropped = read_claims(args.claims)
    if dropped:
        print(f"warning: dropped {dropped:,} malformed claim rows")
    if dropped and not claims:
        # Every row was rejected — writing now would replace a good verdict
        # file with an empty one and still report success.
        print(f"refusing to write: all {dropped:,} rows were rejected.")
        print("The claims file is probably from an older schema.")
        return 1
    summary = aggregate(claims, min_weight=args.min_weight)
    write_verdicts(summary, args.out)

    if args.store:
        print(format_store(summary, args.store))
    else:
        print(format_totals(summary))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
