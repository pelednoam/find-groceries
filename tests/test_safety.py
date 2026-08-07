"""Tests for the behaviours that make an unattended paid run safe.

These target the mutants a prior audit showed the suite could not kill: run
interruption, the circuit breaker, the cost ceiling, weighted-mean
normalisation, the cross-stage provenance contract, and real concurrency.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from groceries import aggregate, extract, runner, select
from groceries.extract import doc_key
from groceries.extract import Extractor
from groceries.types import Candidate, Pricing, RunStats, Usage
from tests.conftest import BadRequestError, FakeClient, message_with_claims


def docs(n: int) -> list[Candidate]:
    return [
        Candidate(
            id=f"id{i}", subreddit="boston", kind="comments",
            created_utc=1_700_000_000 + i, score=1, author="alice",
            permalink=f"/r/boston/comments/p/_/id{i}/",
            stores=["Aldi"], text="Aldi is cheap", truncated=False,
        )
        for i in range(n)
    ]


def _claim(sentiment: str) -> Any:
    return {
        "store": "Aldi", "location": "", "category": "produce", "item": "",
        "claim": "c", "sentiment": sentiment, "price_signal": "none",
        "confidence": "high", "source_id": "s", "source_key": "c_boston_s",
        "subreddit": "boston", "kind": "comments",
        "created_utc": 1_700_000_000, "permalink": "/p/", "score": 1,
    }


@pytest.fixture
def paths(tmp_path: Path) -> runner.Paths:
    """Three *different* parents, so `ensure_parent` is actually exercised."""
    return runner.Paths(
        claims=tmp_path / "a" / "claims.jsonl",
        done=tmp_path / "b" / "done.txt",
        failed=tmp_path / "c" / "failed.jsonl",
    )


class TestEnsureParent:
    def test_creates_all_three_directories(self, paths: runner.Paths) -> None:
        paths.ensure_parent()
        assert paths.claims.parent.is_dir()
        assert paths.done.parent.is_dir()
        assert paths.failed.parent.is_dir()


class TestCircuitBreaker:
    def test_systematic_failure_stops_the_run(self, paths: runner.Paths) -> None:
        # An expired credential fails every document instantly; without a
        # breaker the whole queue burns through and reports success.
        ex = Extractor(
            client=FakeClient([BadRequestError() for _ in range(100)]),
            sleep=lambda _: None,
        )
        stats, _ = runner.run(
            ex, docs(100), paths, workers=1,
            limits=runner.Limits(max_consecutive_failures=5),
        )
        assert stats.stopped is not None
        assert "consecutive failures" in stats.stopped
        assert stats.failed < 100, "the breaker should stop the run early"

    def test_a_success_resets_the_streak(self, paths: runner.Paths, raw_claim: Any) -> None:
        results: list[Any] = [BadRequestError(), BadRequestError(),
                              message_with_claims([raw_claim]),
                              BadRequestError(), BadRequestError()]
        ex = Extractor(client=FakeClient(results), sleep=lambda _: None)
        stats, _ = runner.run(
            ex, docs(5), paths, workers=1,
            limits=runner.Limits(max_consecutive_failures=3),
        )
        assert stats.stopped is None
        assert stats.failed == 4 and stats.docs == 1


class TestCostCeiling:
    def test_run_stops_once_the_ceiling_is_passed(
        self, paths: runner.Paths, raw_claim: Any
    ) -> None:
        ex = Extractor(
            client=FakeClient([message_with_claims([raw_claim]) for _ in range(50)]),
            sleep=lambda _: None,
        )
        # Each fake response reports 50 output tokens; price them absurdly so
        # the ceiling trips after a couple of documents.
        stats, _ = runner.run(
            ex, docs(50), paths, workers=1,
            limits=runner.Limits(
                max_cost=0.001, pricing=Pricing(0.0, 1_000_000.0, 0.0, 0.0)
            ),
        )
        assert stats.stopped is not None and "cost ceiling" in stats.stopped
        assert stats.docs < 50

    def test_ceiling_without_a_rate_card_cannot_stop_the_run(
        self, paths: runner.Paths, raw_claim: Any
    ) -> None:
        # An unknown model has no Pricing, so the ceiling is unenforceable —
        # it must not silently behave as if spend were zero *or* infinite.
        ex = Extractor(
            client=FakeClient([message_with_claims([raw_claim]) for _ in range(3)]),
            sleep=lambda _: None,
        )
        stats, _ = runner.run(
            ex, docs(3), paths, workers=1,
            limits=runner.Limits(max_cost=0.0, pricing=None),
        )
        assert stats.stopped is None and stats.docs == 3

    def test_spend_below_the_ceiling_runs_to_completion(
        self, paths: runner.Paths, raw_claim: Any
    ) -> None:
        ex = Extractor(
            client=FakeClient([message_with_claims([raw_claim]) for _ in range(4)]),
            sleep=lambda _: None,
        )
        stats, _ = runner.run(
            ex, docs(4), paths, workers=1,
            limits=runner.Limits(max_cost=1_000.0, pricing=Pricing(3.0, 15.0, 0.3, 3.75)),
        )
        assert stats.stopped is None and stats.docs == 4

    def test_no_ceiling_means_no_early_stop(
        self, paths: runner.Paths, raw_claim: Any
    ) -> None:
        ex = Extractor(
            client=FakeClient([message_with_claims([raw_claim]) for _ in range(5)]),
            sleep=lambda _: None,
        )
        stats, _ = runner.run(ex, docs(5), paths, workers=1)
        assert stats.stopped is None and stats.docs == 5


class TestConcurrency:
    def test_parallel_workers_do_not_tear_output(
        self, paths: runner.Paths, raw_claim: Any
    ) -> None:
        """Force genuine simultaneity, then check every write survived intact."""
        workers = 8
        n = 64
        barrier = threading.Barrier(workers)
        lock = threading.Lock()
        released = {"count": 0}

        class Blocking:
            @property
            def messages(self) -> Any:
                return self

            def create(self, **kwargs: Any) -> Any:
                # Hold each worker until `workers` of them are inside, so the
                # Sink is genuinely contended rather than incidentally serial.
                with lock:
                    released["count"] += 1
                    should_wait = released["count"] <= (n // workers) * workers
                if should_wait:
                    try:
                        barrier.wait(timeout=5)
                    except threading.BrokenBarrierError:  # pragma: no cover
                        pass
                return message_with_claims([raw_claim, raw_claim, raw_claim])

        ex = Extractor(client=Blocking(), sleep=lambda _: None)
        stats, _ = runner.run(ex, docs(n), paths, workers=workers)

        assert stats.docs == n, "an unlocked stats update would lose increments"
        assert stats.claims == n * 3
        keys = paths.done.read_text().split()
        assert len(keys) == n and len(set(keys)) == n
        lines = paths.claims.read_text().splitlines()
        assert len(lines) == n * 3
        for line in lines:
            json.loads(line)  # a torn interleaved write would not parse


class TestInterruption:
    def test_worker_exception_does_not_leak_out_of_run(
        self, paths: runner.Paths
    ) -> None:
        class Exploding:
            @property
            def messages(self) -> Any:
                raise RuntimeError("boom")

        ex = Extractor(client=Exploding(), sleep=lambda _: None)
        stats, _ = runner.run(ex, docs(4), paths, workers=2,
                              limits=runner.Limits(max_consecutive_failures=99))
        assert stats.failed == 4

    def test_stop_flag_short_circuits_remaining_documents(
        self, paths: runner.Paths, raw_claim: Any
    ) -> None:
        ex = Extractor(client=FakeClient([BadRequestError()] * 20), sleep=lambda _: None)
        stats, _ = runner.run(
            ex, docs(20), paths, workers=1,
            limits=runner.Limits(max_consecutive_failures=2),
        )
        # Documents skipped after the stop flag are neither done nor failed.
        assert stats.failed == 2
        assert len(paths.done.read_text().split()) == 0


class TestWeightedMean:
    def test_sentiment_is_normalised_by_weight(self) -> None:
        """A mutant returning raw score instead of score/weight must fail."""
        cell = aggregate.Cell()
        cell.add(_claim("positive"), 3.0)
        cell.add(_claim("negative"), 1.0)
        # raw score would be +2.0; the weighted mean is +0.5
        assert cell.sentiment() == pytest.approx(0.5)
        assert -1.0 <= cell.sentiment() <= 1.0

    def test_totals_sentiment_is_normalised(self) -> None:
        totals = aggregate.Totals(n=2, weight=4.0, score=2.0)
        assert totals.sentiment() == pytest.approx(0.5)


class TestCrossStageContract:
    def test_stage2_output_satisfies_stage3_required_fields(
        self, raw_claim: Any, candidate: Candidate
    ) -> None:
        """The single highest-value test: dropping a provenance field in
        stage 2 would make stage 3 reject 100% of rows, silently."""
        row = extract.attach_provenance([raw_claim], candidate)[0]
        missing = aggregate.REQUIRED_FIELDS - row.keys()
        assert not missing, f"stage 2 omits fields stage 3 requires: {missing}"

    def test_schema_enum_tracks_the_stage1_matchers(self) -> None:
        # The previous version compared SCHEMA's enum to extract.STORES, which
        # is the same object — a tautology. Compare across modules instead.
        assert set(extract.STORES) == set(select.STORE_PATTERNS) | {"other"}


class TestSampleEdges:
    def test_non_positive_sample_yields_nothing(self) -> None:
        assert runner.sample(docs(5), 0) == []
        assert runner.sample(docs(5), -1) == []

    def test_sample_is_taken_before_the_done_filter(self) -> None:
        """Sampling the shrinking pending list would redraw on every resume."""
        population = docs(40)
        first = runner.sample(population, 10, seed=1)
        done = {doc_key(d) for d in first[:4]}
        # Same command again: the cohort is stable, only the finished ones drop.
        again = runner.pending(runner.sample(population, 10, seed=1), done)
        assert [d["id"] for d in again] == [d["id"] for d in first[4:]]


class TestReportSurfacesStop:
    def test_early_stop_is_the_first_thing_reported(self) -> None:
        stats = RunStats(docs=3, usage=Usage(), stopped="cost ceiling reached ($9.99)")
        text = extract.format_report(stats, "us.anthropic.claude-sonnet-4-6", 1.0)
        assert text.splitlines()[0].startswith("RUN STOPPED EARLY")
