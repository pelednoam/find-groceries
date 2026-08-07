#!/usr/bin/env python3
"""CLI for stage 1 — build the candidate working set from the Reddit corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groceries.select import (  # noqa: E402
    SelectionReport,
    iter_candidates,
    write_candidates,
)

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "reddit"
DEFAULT_OUT = ROOT / "data" / "extraction" / "working_set.jsonl"


def format_report(report: SelectionReport) -> str:
    pct = 100.0 * report.kept / report.scanned if report.scanned else 0.0
    lines = [
        f"scanned {report.scanned:,} documents, kept {report.kept:,} ({pct:.2f}%)",
        "",
        f"{'store':<22}{'candidates':>11}",
        "-" * 33,
    ]
    lines += [f"{s:<22}{c:>11,}" for s, c in report.per_store.most_common()]
    lines.append(f"\nby subreddit: {dict(report.per_subreddit.most_common())}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--subreddits",
        nargs="+",
        default=None,
        help="restrict to these subreddits (e.g. boston Somerville CambridgeMA)",
    )
    args = parser.parse_args(argv)

    shards = sorted(args.corpus.glob("*/*/*.ndjson.gz"))
    print(f"{len(shards)} shard files under {args.corpus}")
    report = SelectionReport()
    n = write_candidates(
        iter_candidates(shards, report, limit=args.limit, subreddits=args.subreddits),
        args.out,
    )
    print(format_report(report))
    print(f"\nwrote {n:,} candidates -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
