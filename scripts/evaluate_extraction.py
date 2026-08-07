#!/usr/bin/env python3
"""Score stage 2 against the hand-labelled gold set.

    .venv/bin/python scripts/evaluate_extraction.py

Costs one request per case (24 as of writing, well under a cent). Run it after
any change to the prompt, the schema, or the model — those are exactly the
changes whose effect is invisible in the output and only shows up as a shifted
aggregate three stages later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groceries.client import build_client  # noqa: E402
from groceries.evaluate import (  # noqa: E402
    evaluate,
    format_score,
    read_gold,
    write_report,
)
from groceries.extract import DEFAULT_MODEL, DEFAULT_REGION, Extractor  # noqa: E402
from groceries.paths import EXTRACT_DIR, ROOT  # noqa: E402

GOLD = ROOT / "gold" / "extraction_gold.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--out", type=Path, default=EXTRACT_DIR / "gold_report.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument(
        "--min-exact-match",
        type=float,
        default=0.0,
        help="exit non-zero below this exact-match rate, for use as a gate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.gold.exists():
        print(f"no gold set at {args.gold}")
        return 2
    cases = read_gold(args.gold)
    if not cases:
        print(f"refusing to run: {args.gold} has no enabled cases")
        return 2
    print(f"scoring {len(cases)} cases against {args.model}\n")

    extractor = Extractor(
        client=build_client(args.region),
        model=args.model,
        thinking=not args.no_thinking,
    )
    score = evaluate(extractor, cases)
    print(format_score(score))
    write_report(score, args.out)
    print(f"\nwrote {args.out}")
    if score.exact_match() < args.min_exact_match:
        print(
            f"\nFAIL: exact match {score.exact_match():.0%} is below the "
            f"{args.min_exact_match:.0%} gate"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
