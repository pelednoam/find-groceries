#!/usr/bin/env python3
"""
How much grocery signal is actually in the fetched corpus?

Counts documents mentioning each store, and how many of those also carry a
price/quality word — a rough proxy for "this comment says something evaluative
about a store" rather than just naming it as a landmark.
"""

import collections
import glob
import gzip
import json
import os
import re
import sys

OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reddit")

STORES = {
    "Market Basket": r"market\s*basket|\bmarket\s*basket\b|\bMB\b",
    "Trader Joe's": r"trader\s*joe",
    "Star Market": r"star\s*market",
    "Whole Foods": r"whole\s*foods|\bWF\b",
    "Stop & Shop": r"stop\s*(&|and|n)?\s*shop",
    "H Mart": r"h[\s-]?mart",
    "Wegmans": r"wegmans",
    "Aldi": r"\baldi\b",
    "Costco": r"costco",
    "Haymarket": r"haymarket",
    "Target": r"\btarget\b",
}
PATS = {k: re.compile(v, re.I) for k, v in STORES.items()}
QUAL = re.compile(r"\b(produce|quality|fresh|rotten|cheap|cheaper|expensive|price|prices|\$\d)", re.I)


def main() -> int:
    cnt: collections.Counter[str] = collections.Counter()
    both: collections.Counter[str] = collections.Counter()
    per_sub: collections.Counter[str] = collections.Counter()
    n = 0
    files = sorted(glob.glob(os.path.join(OUT_ROOT, "*", "*", "*.ndjson.gz")))
    for i, f in enumerate(files):
        sub = os.path.basename(os.path.dirname(f))
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                n += 1
                t = o.get("body") or ((o.get("title") or "") + " " + (o.get("selftext") or ""))
                if not t or len(t) < 10:
                    continue
                hits = [k for k, p in PATS.items() if p.search(t)]
                if not hits:
                    continue
                per_sub[sub] += 1
                evaluative = bool(QUAL.search(t))
                for k in hits:
                    cnt[k] += 1
                    if evaluative:
                        both[k] += 1
        if i % 200 == 0:
            print(f"  ...{i}/{len(files)} files, {n:,} docs", flush=True)

    print(f"\nscanned {n:,} documents\n")
    print(f"{'store':<15}{'mentions':>10}{'evaluative':>13}{'ratio':>8}")
    print("-" * 46)
    for k, v in cnt.most_common():
        r = f"{100.0 * both[k] / v:.0f}%" if v else "-"
        print(f"{k:<15}{v:>10,}{both[k]:>13,}{r:>8}")
    print("-" * 46)
    print(f"{'TOTAL hits':<15}{sum(cnt.values()):>10,}{sum(both.values()):>13,}")
    print("\ndocuments mentioning any store, by subreddit:")
    for s, v in per_sub.most_common():
        print(f"  {s:<14}{v:>9,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
