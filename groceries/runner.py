"""Concurrent, resumable, interruptible driver for stage 2.

Resume works by appending every completed document key to a done-file. A rerun
reads that file and skips those documents, so an interrupted run costs at most
the in-flight requests.

Three things make this safe to leave unattended:

* **It stops when told to.** Futures are submitted eagerly, and a bare
  `ThreadPoolExecutor` context manager drains every queued task before letting
  an exception out — so Ctrl-C would otherwise appear to hang while the run
  billed to completion. A shared stop flag plus `cancel_futures` fixes that.
* **It stops when the run has clearly gone wrong.** A systematic fault
  (expired credentials, a bad model id) fails documents instantly, and without
  a circuit breaker the whole queue burns through in seconds reporting success.
* **It stops when it has spent enough.** An optional cost ceiling bounds the
  damage from a cache-hit collapse or a pathological document.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .extract import Extractor, doc_key, record
from .types import Candidate, Pricing, RunStats, SourcedClaim, Usage

DEFAULT_MAX_CONSECUTIVE_FAILURES = 50


@dataclass(frozen=True)
class Paths:
    """Where a run reads and writes."""

    claims: Path
    done: Path
    failed: Path

    def ensure_parent(self) -> None:
        """Create every output directory, not just the claims one."""
        for path in (self.claims, self.done, self.failed):
            path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Limits:
    """Conditions under which a run should stop early."""

    max_cost: float | None = None
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    pricing: Pricing | None = None


def read_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def pending(docs: Sequence[Candidate], done: set[str]) -> list[Candidate]:
    return [d for d in docs if doc_key(d) not in done]


def sample(docs: Sequence[Candidate], n: int | None, seed: int = 0) -> list[Candidate]:
    """Draw a stable subset.

    Sample *before* filtering by the done-set, never after: sampling the
    shrinking pending list redraws a different cohort on every resume, so
    rerunning the same command would process fresh documents instead of
    finishing the ones it started.
    """
    if n is None:
        return list(docs)
    if n <= 0:
        return []
    rng = random.Random(seed)
    return rng.sample(list(docs), min(n, len(docs)))


class Sink:
    """Thread-safe append-only writer for claims, done-keys, and failures."""

    def __init__(self, claims: TextIO, done: TextIO, failed: TextIO) -> None:
        self._claims = claims
        self._done = done
        self._failed = failed
        self._lock = threading.Lock()

    def write(self, key: str, claims: Iterable[SourcedClaim]) -> None:
        payload = [json.dumps(c, ensure_ascii=False) for c in claims]
        with self._lock:
            if payload:
                self._claims.write("\n".join(payload) + "\n")
                self._claims.flush()
            self._done.write(key + "\n")
            self._done.flush()

    def fail(self, key: str, error: str) -> None:
        with self._lock:
            self._failed.write(json.dumps({"id": key, "error": error[:300]}) + "\n")
            self._failed.flush()


def process_one(
    extractor: Extractor,
    doc: Candidate,
    sink: Sink,
    stats: RunStats,
    lock: threading.Lock,
    stop: threading.Event,
    limits: Limits = Limits(),
) -> None:
    """Extract one document, persist it, and fold it into the run totals.

    Every failure is contained here — including a failed *write*, which used to
    sit outside the boundary. By the time such an exception reached the caller
    the executor would already have billed the rest of the queue.
    """
    if stop.is_set():
        return
    key = doc_key(doc)
    try:
        claims, response = extractor.extract(doc)
        sink.write(key, claims)
    except Exception as exc:  # noqa: BLE001 - deliberate containment boundary
        sink.fail(key, f"{type(exc).__name__}: {exc}")
        with lock:
            stats.failed += 1
            stats.consecutive_failures += 1
            # A response that arrived and then failed to parse was billed.
            stats.usage = stats.usage + getattr(exc, "usage", Usage())
            if stats.consecutive_failures >= limits.max_consecutive_failures:
                stats.stopped = (
                    f"{stats.consecutive_failures} consecutive failures — "
                    "the run looks systematically broken"
                )
                stop.set()
            elif _over_budget(stats, limits):
                stop.set()
        return
    with lock:
        record(stats, claims, response)
        stats.consecutive_failures = 0
        if _over_budget(stats, limits):
            stop.set()


def _over_budget(stats: RunStats, limits: Limits) -> bool:
    """Record the stop reason if spend has passed the ceiling. Caller holds the lock."""
    if limits.max_cost is None or limits.pricing is None:
        return False
    spent = limits.pricing.cost(stats.usage)
    if spent <= limits.max_cost:
        return False
    stats.stopped = f"cost ceiling reached (${spent:.2f})"
    return True


def run(
    extractor: Extractor,
    docs: Sequence[Candidate],
    paths: Paths,
    workers: int = 8,
    progress: Callable[[int, int, RunStats, float], None] | None = None,
    now: Callable[[], float] = time.monotonic,
    limits: Limits = Limits(),
) -> tuple[RunStats, float]:
    """Process every document concurrently; returns stats and elapsed seconds."""
    paths.ensure_parent()
    stats = RunStats()
    lock = threading.Lock()
    stop = threading.Event()
    start = now()
    with (
        paths.claims.open("a", encoding="utf-8") as claims_fh,
        paths.done.open("a", encoding="utf-8") as done_fh,
        paths.failed.open("a", encoding="utf-8") as failed_fh,
    ):
        sink = Sink(claims_fh, done_fh, failed_fh)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [
                pool.submit(
                    process_one, extractor, doc, sink, stats, lock, stop, limits
                )
                for doc in docs
            ]
            try:
                for i, fut in enumerate(as_completed(futures), 1):
                    fut.result()
                    if progress:
                        progress(i, len(docs), stats, now() - start)
                    if stop.is_set():
                        break
            finally:
                # Without cancel_futures the pool drains every queued document
                # on the way out, so Ctrl-C would bill the whole remaining run.
                stop.set()
                pool.shutdown(wait=True, cancel_futures=True)
    return stats, now() - start


def make_progress(every: int = 100) -> Callable[[int, int, RunStats, float], None]:
    """A progress callback that prints every `every` documents and at the end."""

    def report(i: int, total: int, stats: RunStats, elapsed: float) -> None:
        if i % every and i != total:
            return
        rate = i / elapsed if elapsed > 0 else 0.0
        eta = (total - i) / rate if rate > 0 else 0.0
        # `failed` belongs on this line: without it a run whose every call is
        # failing looks exactly like a healthy one that is finding nothing.
        print(
            f"  {i}/{total}  claims={stats.claims:,}  failed={stats.failed:,}  "
            f"{rate:.1f} docs/s  ETA {eta / 60:.0f}m",
            flush=True,
        )

    return report
