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
    DEFAULT_MIN_WEIGHT,
    aggregate,
    format_store,
    format_totals,
    read_claims,
    write_verdicts,
)
from groceries.paths import CLAIMS, DONE, VERDICTS, WORKING_SET  # noqa: E402


def corpus_provenance(claims_path: Path) -> dict[str, object]:
    """What the verdicts were computed from.

    Without this the output is a set of confident numbers with no way to tell
    whether they came from the full corpus or from a 150-document calibration
    sample — a distinction that changes how much any of them is worth.

    The working-set and done-file counts describe the *default* pipeline run.
    Attaching them to a hand-picked `--claims` file would assert a provenance
    that file does not have, which is worse than recording none.
    """
    provenance: dict[str, object] = {"claims_file": claims_path.name}
    if claims_path.resolve() != CLAIMS.resolve():
        provenance["note"] = "custom claims file; pipeline counts not applicable"
        return provenance
    for label, path in (("working_set", WORKING_SET), ("documents_extracted", DONE)):
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                provenance[label] = sum(1 for line in fh if line.strip())
    return provenance


def verdicts_path_for(claims: Path) -> Path:
    """Where a claims file's verdicts belong.

    The canonical corpus keeps the canonical name; anything else gets its
    own, named after it. Defaulting to the canonical file meant
    `--claims reviews_claims.jsonl` would have replaced the Reddit verdicts
    with review-derived ones and reported success.
    """
    if claims.resolve() == CLAIMS.resolve():
        return VERDICTS
    stem = claims.stem.replace("_claims", "").replace("_placed", "")
    return claims.parent / f"{stem}_verdicts.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=CLAIMS)
    # No default: derived from --claims below, so pointing this at a
    # secondary corpus cannot silently overwrite the canonical verdicts.
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--store", default=None)
    parser.add_argument("--min-weight", type=float, default=DEFAULT_MIN_WEIGHT)
    parser.add_argument("--no-provenance", action="store_true",
                        help="skip pipeline counts (for a secondary corpus)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out is None:
        args.out = verdicts_path_for(args.claims)

    if not args.claims.exists():
        print(f"no claims file at {args.claims}")
        print("Run scripts/extract_claims.py first.")
        return 2
    claims, dropped = read_claims(args.claims)
    if dropped:
        print(f"warning: dropped {dropped:,} malformed claim rows")
    if not claims:
        # Writing now would replace a good verdict file with an empty one and
        # still report success.
        print(f"refusing to write: no usable claims in {args.claims}.")
        if dropped:
            print(f"All {dropped:,} rows were rejected — probably an older schema.")
        return 1
    summary = aggregate(
        claims,
        min_weight=args.min_weight,
        corpus=None if args.no_provenance else corpus_provenance(args.claims),
    )
    write_verdicts(summary, args.out)

    if args.store:
        print(format_store(summary, args.store))
    else:
        print(format_totals(summary))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
