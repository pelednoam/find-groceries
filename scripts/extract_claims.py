#!/usr/bin/env python3
"""CLI for stage 2 — extract structured store claims via Claude on Bedrock.

    .venv/bin/python scripts/extract_claims.py --sample 150   # calibrate
    .venv/bin/python scripts/extract_claims.py                # full run
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groceries.client import build_client  # noqa: E402
from groceries.extract import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REGION,
    Extractor,
    format_report,
    pricing_for,
)
from groceries.paths import (  # noqa: E402
    CLAIMS,
    DONE,
    EXTRACT_DIR,
    FAILED,
    LOCK,
    WORKING_SET,
)
from groceries.runner import (  # noqa: E402
    Limits,
    Paths,
    make_progress,
    pending,
    read_done,
    run,
    sample,
)
from groceries.select import read_candidates  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-set", type=Path, default=WORKING_SET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="concurrent requests; network-bound, so this sets total runtime",
    )
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument(
        "--max-cost",
        type=float,
        default=150.0,
        help=(
            "stop the run once estimated spend exceeds this (USD). Estimated "
            "from Anthropic first-party rates; Bedrock bills at AWS rates, so "
            "treat this as an order-of-magnitude guard, not an exact budget"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    lock = LOCK
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"another extraction appears to be running ({lock}).")
        print("If that is stale, delete the file and retry.")
        return 2
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        return _run(args)
    finally:
        lock.unlink(missing_ok=True)


def _run(args: argparse.Namespace) -> int:
    if not args.working_set.exists():
        print(f"no working set at {args.working_set}")
        print("Run scripts/select_evaluative.py first.")
        return 2
    docs, dropped = read_candidates(args.working_set)
    if dropped:
        print(f"warning: dropped {dropped:,} malformed rows from the working set")
    if not docs:
        # Paying per document to discover the file is unreadable is the
        # expensive way to learn this.
        print(f"refusing to run: {args.working_set} yielded no usable candidates.")
        print("Regenerate it with scripts/select_evaluative.py.")
        return 2
    paths = Paths(claims=CLAIMS, done=DONE, failed=FAILED)

    done = read_done(paths.done)
    if done:
        print(f"resuming: {len(done):,} docs already processed")
    # Sample first, then filter: sampling the shrinking pending list would
    # redraw a different cohort on every resume.
    todo = pending(sample(docs, args.sample, args.seed), done)
    if args.sample:
        print(f"sampling {len(todo)} docs (seed={args.seed})")

    thinking = not args.no_thinking
    print(
        f"{len(todo):,} docs to process, model={args.model}, region={args.region}, "
        f"workers={args.workers}, thinking={'adaptive' if thinking else 'disabled'}"
    )
    if not todo:
        return 0

    if args.max_cost is not None and not math.isfinite(args.max_cost):
        print(f"refusing to run: --max-cost must be a finite number, got {args.max_cost}")
        return 2
    pricing = pricing_for(args.model)
    if args.max_cost is not None and pricing is None:
        print(f"refusing to run: --max-cost is set but no rate card exists for")
        print(f"  {args.model}")
        print("Without one the ceiling cannot be enforced and spend is unbounded.")
        return 2

    extractor = Extractor(
        client=build_client(args.region), model=args.model, thinking=thinking
    )
    stats, elapsed = run(
        extractor,
        todo,
        paths,
        workers=args.workers,
        progress=make_progress(),
        limits=Limits(max_cost=args.max_cost, pricing=pricing),
    )
    print("\n" + format_report(stats, args.model, elapsed, total_docs=len(docs)))
    print(f"\nclaims -> {paths.claims}")
    if stats.failed:
        print(f"{stats.failed:,} documents failed — see {paths.failed}")
    return 1 if (stats.failed or stats.stopped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
