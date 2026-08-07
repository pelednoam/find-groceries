#!/usr/bin/env python3
"""
Verify the fetched Reddit corpus: which month-windows are missing, whether every
file is readable NDJSON, and how the row counts compare to what Arctic Shift
reports for each subreddit.

Run after fetch_reddit_dumps.py. Exits non-zero if anything is missing or corrupt,
so it doubles as a completion check.
"""

import glob
import gzip
import json
import os
import sys
import time
from collections import defaultdict

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_reddit_dumps import API, KINDS, OUT_ROOT, SUBS, month_windows  # noqa: E402

# Every window present and readable is not the same as every row fetched: a
# window that paginated to exhaustion against a throttled endpoint can be a
# tenth of its true size and still look fine. Arctic Shift's reported totals
# are approximate, so this is a floor, not an equality check.
MIN_COVERAGE = float(os.environ.get("MIN_COVERAGE", 0.95))


def expected_windows() -> tuple[dict[tuple[str, str], set[str]], dict[str, dict[str, int]]]:
    session = requests.Session()
    now = int(time.time())
    exp: dict[tuple[str, str], set[str]] = defaultdict(set)
    totals: dict[str, dict[str, int]] = {}
    for sub in SUBS:
        meta = session.get(f"{API}/subreddits/search",
                           params={"subreddit": sub, "limit": "1"}, timeout=60
                           ).json()["data"][0]["_meta"]
        totals[sub] = {"posts": meta["num_posts"], "comments": meta["num_comments"]}
        for kind in KINDS:
            start = meta["earliest_post"] if kind == "posts" else meta["earliest_comment"]
            for label, _, _ in month_windows(start, now):
                exp[(sub, kind)].add(label)
    return exp, totals


def main() -> int:
    exp, totals = expected_windows()
    missing: list[str] = []
    corrupt: list[str] = []
    rows: dict[tuple[str, str], int] = defaultdict(int)
    ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    dupes = 0

    for (sub, kind), labels in sorted(exp.items()):
        d = os.path.join(OUT_ROOT, kind, sub)
        have = {os.path.basename(p).replace(".ndjson.gz", "")
                for p in glob.glob(os.path.join(d, "*.ndjson.gz"))}
        for lab in sorted(labels - have):
            missing.append(f"{sub}/{kind}/{lab}")
        for p in sorted(glob.glob(os.path.join(d, "*.ndjson.gz"))):
            try:
                with gzip.open(p, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        o = json.loads(line)
                        rows[(sub, kind)] += 1
                        if o["id"] in ids[(sub, kind)]:
                            dupes += 1
                        ids[(sub, kind)].add(o["id"])
            except Exception as e:
                corrupt.append(f"{p}: {e!r}"[:160])

    short: list[str] = []
    print(f"{'subreddit':<14}{'kind':<10}{'fetched':>12}{'reported':>12}{'coverage':>10}")
    print("-" * 58)
    for sub in SUBS:
        for kind in KINDS:
            got, rep = rows[(sub, kind)], totals[sub][kind]
            pct = f"{100.0 * got / rep:.1f}%" if rep else "n/a"
            print(f"{sub:<14}{kind:<10}{got:>12,}{rep:>12,}{pct:>10}")
            if rep and got < rep * MIN_COVERAGE:
                short.append(f"{sub}/{kind}: {got:,} of ~{rep:,} ({pct})")
    total = sum(rows.values())
    print("-" * 58)
    print(f"{'TOTAL':<24}{total:>12,}")
    print(f"\nmissing windows: {len(missing)}   corrupt files: {len(corrupt)}   "
          f"duplicate ids: {dupes}   short of target: {len(short)}")
    for m in missing[:25]:
        print("  MISSING", m)
    if len(missing) > 25:
        print(f"  ... and {len(missing) - 25} more")
    for c in corrupt[:10]:
        print("  CORRUPT", c)
    for s in short:
        print("  SHORT", s)

    manifest_path = os.path.join(OUT_ROOT, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    if manifest.get("failed_windows"):
        print(f"\n  {len(manifest['failed_windows'])} windows failed on the last fetch "
              "(see manifest.json)")

    if missing or corrupt or short or manifest.get("failed_windows"):
        print("\nRe-run to fill gaps:  WORKERS=2 MAX_RETRIES=15 python3 scripts/fetch_reddit_dumps.py")
        return 1
    print(f"\nCorpus complete and readable (every slice at or above "
          f"{MIN_COVERAGE:.0%} of the reported total).")
    if manifest.get("incomplete_month"):
        print(f"Note: {manifest['incomplete_month']} is still accumulating; "
              "rerun the fetch to top it up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
