#!/usr/bin/env python3
"""CLI for stage 1 — build the candidate working set from the Reddit corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groceries.paths import CORPUS, WORKING_SET  # noqa: E402
from groceries.select import (  # noqa: E402
    SelectionReport,
    iter_candidates,
    write_candidates,
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=WORKING_SET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--subreddits",
        nargs="+",
        default=None,
        help="restrict to these subreddits (e.g. boston Somerville CambridgeMA)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    shards = sorted(args.corpus.glob("*/*/*.ndjson.gz"))
    print(f"{len(shards)} shard files under {args.corpus}")
    # A typo'd --corpus used to glob nothing, write an empty working set over
    # the good one, and exit 0. Stage 2 then reported "0 docs to process" and
    # also exited 0, so the whole pipeline succeeded at doing nothing.
    if not shards:
        print(f"refusing to run: no shards match {args.corpus}/*/*/*.ndjson.gz")
        return 2
    if args.subreddits is not None:
        available = {s.parent.name for s in shards}
        unknown = sorted(set(args.subreddits) - available)
        if unknown:
            print(f"refusing to run: no shards for {', '.join(unknown)}")
            print(f"available: {', '.join(sorted(available))}")
            return 2

    # Write to a staging path first. `write_candidates` publishes atomically,
    # so writing straight to --out replaced a good working set with an empty
    # one and only *then* reported failure — the destructive half of the
    # operation had already committed.
    staged = args.out.with_suffix(args.out.suffix + ".staged")
    report = SelectionReport()
    n = write_candidates(
        iter_candidates(shards, report, limit=args.limit, subreddits=args.subreddits),
        staged,
    )
    print(format_report(report))
    if n == 0:
        staged.unlink(missing_ok=True)
        print(f"\nrefusing to overwrite {args.out}: the selection is empty.")
        return 1
    staged.replace(args.out)
    print(f"\nwrote {n:,} candidates -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
