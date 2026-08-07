#!/usr/bin/env python3
"""
Pull full Reddit dumps for the grocery-research subreddits from Arctic Shift.

Arctic Shift caps pages at 100 items, so a subreddit is fetched as many
independent month-windows (after/before on created_utc) that paginate in
parallel. Each window lands in its own gzipped NDJSON file, which makes the
whole job resumable: rerunning skips any window already written.

Output layout:
    data/reddit/<kind>/<subreddit>/<YYYY-MM>.ndjson.gz
    data/reddit/_logs/fetch.log
    data/reddit/manifest.json
"""

import calendar
from collections.abc import Iterator
from typing import Any, TextIO
import gzip
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

API = "https://arctic-shift.photon-reddit.com/api"
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reddit")
SUBS = ["CambridgeMA", "Somerville", "traderjoes", "boston"]
KINDS = ["posts", "comments"]
PAGE = 100
# Overridable so the gap-fill pass can run slower and more stubbornly than the
# bulk pass: WORKERS=2 MAX_RETRIES=15 python3 scripts/fetch_reddit_dumps.py
WORKERS = int(os.environ.get("WORKERS", 5))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 8))
PER_REQUEST_PAUSE = float(os.environ.get("PER_REQUEST_PAUSE", 0.15))
USER_AGENT = (
    "grocery-research/1.0 (personal price+quality research; "
    "+https://github.com/pelednoam/find-groceries)"
)

_print_lock = threading.Lock()
_log_fh: TextIO | None = None
_stats_lock = threading.Lock()
STATS: dict[str, int] = {"requests": 0, "items": 0, "windows_done": 0, "windows_skipped": 0, "retries": 0, "gaps": 0}


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    with _print_lock:
        print(line, flush=True)
        if _log_fh:
            _log_fh.write(line + "\n")
            _log_fh.flush()


def bump(**kw: int) -> None:
    with _stats_lock:
        for k, v in kw.items():
            STATS[k] += v


def month_windows(start_utc: int, end_utc: int) -> Iterator[tuple[str, int, int]]:
    """Yield (label, after, before) covering [start_utc, end_utc] one month at a time."""
    d = datetime.fromtimestamp(start_utc, timezone.utc)
    y, m = d.year, d.month
    while True:
        first = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())
        last_day = calendar.monthrange(y, m)[1]
        last = int(datetime(y, m, last_day, 23, 59, 59, tzinfo=timezone.utc).timestamp())
        if first > end_utc:
            return
        yield f"{y:04d}-{m:02d}", first, last
        m += 1
        if m > 12:
            y, m = y + 1, 1


def get(session: requests.Session, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """GET with backoff on throttle/overload responses. Returns the decoded 'data' list.

    Arctic Shift signals overload two ways: a normal 429, and a 422 whose body
    reads "Timeout. Maybe slow down a bit". The 422 is a request to back off,
    not a bad-request, so it has to be retried rather than raised.
    """
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(PER_REQUEST_PAUSE)
            r = session.get(f"{API}/{path}", params=params, timeout=120)
            bump(requests=1)
            if r.status_code == 200:
                return r.json().get("data") or []

            body = r.text[:200]
            overloaded = (
                r.status_code == 429
                or r.status_code >= 500
                or (r.status_code == 422 and ("slow down" in body.lower() or "timeout" in body.lower()))
            )
            if overloaded:
                reset = r.headers.get("x-ratelimit-reset")
                wait = float(reset) if reset is not None and reset.isdigit() else delay
                wait = min(wait, 60) + random.uniform(0, 1.5)
                log(f"  HTTP {r.status_code}, backing off {wait:.1f}s "
                    f"({params.get('subreddit')}/{path} after={params.get('after')})")
                time.sleep(wait)
                bump(retries=1)
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {body}")
        except requests.RequestException as e:
            log(f"  network error ({e.__class__.__name__}), retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s")
            time.sleep(delay + random.uniform(0, 1))
            bump(retries=1)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"giving up after {MAX_RETRIES} retries: {path} {params}")


def fetch_window(sub: str, kind: str, label: str, after: int, before: int) -> int:
    """Paginate one month-window to exhaustion and write it out."""
    out_dir = os.path.join(OUT_ROOT, kind, sub)
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"{label}.ndjson.gz")
    if os.path.exists(final):
        bump(windows_skipped=1)
        return 0
    part = final + ".part"

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    seen = set()
    n = 0
    # `after` is exclusive, so step back one second to include the boundary item.
    cursor = after - 1

    with gzip.open(part, "wt", encoding="utf-8") as fh:
        while True:
            data = get(session, f"{kind}/search", {
                "subreddit": sub, "limit": PAGE, "sort": "asc",
                "after": cursor, "before": before,
            })
            if not data:
                break
            fresh = [x for x in data if x.get("id") not in seen]
            for x in fresh:
                seen.add(x["id"])
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
            n += len(fresh)
            bump(items=len(fresh))

            max_utc = max(x["created_utc"] for x in data)
            if max_utc <= cursor:
                # >100 items share one timestamp; step past it and note the gap.
                bump(gaps=1)
                log(f"  ! {sub}/{kind}/{label}: >100 items at utc={max_utc}, some may be skipped")
                cursor = max_utc + 1
            else:
                cursor = max_utc
            if cursor > before:
                break
            if len(data) < PAGE:
                break

    os.replace(part, final)
    bump(windows_done=1)
    return n


def main() -> int:
    os.makedirs(os.path.join(OUT_ROOT, "_logs"), exist_ok=True)
    global _log_fh
    _log_fh = open(os.path.join(OUT_ROOT, "_logs", "fetch.log"), "a", encoding="utf-8")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    now = int(time.time())

    tasks = []
    log("planning windows...")
    for sub in SUBS:
        meta = get(session, "subreddits/search", {"subreddit": sub, "limit": 1})[0]["_meta"]
        for kind in KINDS:
            start = meta["earliest_post"] if kind == "posts" else meta["earliest_comment"]
            wins = list(month_windows(start, now))
            tasks += [(sub, kind, lab, a, b) for lab, a, b in wins]
        log(f"  {sub:14} posts={meta['num_posts']:>8,}  comments={meta['num_comments']:>10,}")

    # Smallest subreddits first so partial results are useful early.
    order = {s: i for i, s in enumerate(SUBS)}
    tasks.sort(key=lambda t: (order[t[0]], t[1], t[2]))
    log(f"{len(tasks)} month-windows queued across {len(SUBS)} subreddits, {WORKERS} workers")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(fetch_window, *t): t for t in tasks}
        for f in as_completed(futs):
            sub, kind, label = futs[f][:3]
            done += 1
            try:
                f.result()
            except Exception as e:
                log(f"  FAILED {sub}/{kind}/{label}: {e}")
            if done % 25 == 0 or done == len(tasks):
                el = time.time() - t0
                rate = STATS["requests"] / el if el else 0
                eta = (len(tasks) - done) * (el / done) if done else 0
                log(f"{done}/{len(tasks)} windows | {STATS['items']:,} items | "
                    f"{rate:.1f} req/s | ETA {eta / 60:.0f}m")

    manifest = {
        "source": "Arctic Shift (arctic-shift.photon-reddit.com)",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "subreddits": SUBS,
        "kinds": KINDS,
        "stats": STATS,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"DONE {json.dumps(STATS)} in {(time.time() - t0) / 60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
