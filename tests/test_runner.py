"""Tests for the concurrent, resumable stage-2 driver."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from groceries import runner
from groceries.extract import Extractor, doc_key
from groceries.types import Candidate, RunStats
from tests.conftest import (
    BadRequestError,
    FakeBlock,
    FakeClient,
    FakeMessage,
    RateLimitError,
    message_with_claims,
)


@pytest.fixture
def paths(tmp_path: Path) -> runner.Paths:
    return runner.Paths(
        claims=tmp_path / "out" / "claims.jsonl",
        done=tmp_path / "out" / "done.txt",
        failed=tmp_path / "out" / "failed.jsonl",
    )


def make_docs(n: int) -> list[Candidate]:
    return [
        Candidate(
            id=f"id{i}",
            subreddit="boston",
            kind="comments",
            created_utc=1_700_000_000 + i,
            score=1,
            author="alice",
            parent_body="",
            permalink=f"/r/boston/comments/p/_/id{i}/",
            stores=["Aldi"],
            text="Aldi is cheap",
            truncated=False,
        )
        for i in range(n)
    ]


class TestDoneFile:
    def test_missing_file_is_empty_set(self, tmp_path: Path) -> None:
        assert runner.read_done(tmp_path / "nope.txt") == set()

    def test_reads_keys_and_ignores_blanks(self, tmp_path: Path) -> None:
        p = tmp_path / "done.txt"
        p.write_text("a\n\nb\n", encoding="utf-8")
        assert runner.read_done(p) == {"a", "b"}


class TestPending:
    def test_filters_completed(self) -> None:
        docs = make_docs(3)
        done = {doc_key(docs[0])}
        assert [d["id"] for d in runner.pending(docs, done)] == ["id1", "id2"]

    def test_nothing_done(self) -> None:
        docs = make_docs(2)
        assert runner.pending(docs, set()) == docs


class TestSample:
    def test_none_returns_everything(self) -> None:
        docs = make_docs(5)
        assert runner.sample(docs, None) == docs

    def test_is_deterministic_for_a_seed(self) -> None:
        docs = make_docs(20)
        assert runner.sample(docs, 5, seed=7) == runner.sample(docs, 5, seed=7)

    def test_different_seeds_differ(self) -> None:
        docs = make_docs(50)
        assert runner.sample(docs, 5, seed=1) != runner.sample(docs, 5, seed=2)

    def test_caps_at_population_size(self) -> None:
        assert len(runner.sample(make_docs(3), 10)) == 3


class TestSink:
    def test_writes_claims_and_done(self, tmp_path: Path, sourced_claim: Any) -> None:
        c, d, f = (tmp_path / n for n in ("c.jsonl", "d.txt", "f.jsonl"))
        with c.open("w") as cf, d.open("w") as df, f.open("w") as ff:
            runner.Sink(cf, df, ff).write("k1", [sourced_claim])
        assert json.loads(c.read_text())["store"] == "Market Basket"
        assert d.read_text() == "k1\n"

    def test_done_written_even_with_no_claims(self, tmp_path: Path) -> None:
        c, d, f = (tmp_path / n for n in ("c.jsonl", "d.txt", "f.jsonl"))
        with c.open("w") as cf, d.open("w") as df, f.open("w") as ff:
            runner.Sink(cf, df, ff).write("k1", [])
        assert c.read_text() == ""
        assert d.read_text() == "k1\n"

    def test_failure_is_recorded_and_truncated(self, tmp_path: Path) -> None:
        c, d, f = (tmp_path / n for n in ("c.jsonl", "d.txt", "f.jsonl"))
        with c.open("w") as cf, d.open("w") as df, f.open("w") as ff:
            runner.Sink(cf, df, ff).fail("k1", "x" * 500)
        row = json.loads(f.read_text())
        assert row["id"] == "k1"
        assert len(row["error"]) == 300


class TestProcessOne:
    def test_records_success(self, paths: runner.Paths, raw_claim: Any) -> None:
        paths.ensure_parent()
        docs = make_docs(1)
        ex = Extractor(client=FakeClient([message_with_claims([raw_claim])]), sleep=lambda _: None)
        stats = RunStats()
        with (
            paths.claims.open("a") as cf,
            paths.done.open("a") as df,
            paths.failed.open("a") as ff,
        ):
            runner.process_one(ex, docs[0], runner.Sink(cf, df, ff), stats,
                               threading.Lock(), threading.Event())
        assert stats.docs == 1 and stats.claims == 1 and stats.failed == 0

    def test_records_failure(self, paths: runner.Paths) -> None:
        paths.ensure_parent()
        docs = make_docs(1)
        ex = Extractor(client=FakeClient([BadRequestError()]), sleep=lambda _: None)
        stats = RunStats()
        with (
            paths.claims.open("a") as cf,
            paths.done.open("a") as df,
            paths.failed.open("a") as ff,
        ):
            runner.process_one(ex, docs[0], runner.Sink(cf, df, ff), stats,
                               threading.Lock(), threading.Event())
        assert stats.failed == 1 and stats.docs == 0
        assert "BadRequestError" in paths.failed.read_text()


class TestRun:
    def _extractor(self, n: int, raw_claim: Any) -> Extractor:
        return Extractor(
            client=FakeClient([message_with_claims([raw_claim]) for _ in range(n)]),
            sleep=lambda _: None,
        )

    def test_serial_path(self, paths: runner.Paths, raw_claim: Any) -> None:
        docs = make_docs(3)
        stats, elapsed = runner.run(self._extractor(3, raw_claim), docs, paths, workers=1)
        assert stats.docs == 3 and stats.claims == 3
        assert elapsed >= 0
        assert len(paths.done.read_text().strip().split("\n")) == 3

    def test_concurrent_path(self, paths: runner.Paths, raw_claim: Any) -> None:
        docs = make_docs(6)
        stats, _ = runner.run(self._extractor(6, raw_claim), docs, paths, workers=3)
        assert stats.docs == 6
        assert len(paths.claims.read_text().strip().split("\n")) == 6

    def test_progress_callback_is_invoked(self, paths: runner.Paths, raw_claim: Any) -> None:
        seen: list[int] = []
        runner.run(
            self._extractor(2, raw_claim),
            make_docs(2),
            paths,
            workers=1,
            progress=lambda i, total, stats, el: seen.append(i),
        )
        assert seen == [1, 2]

    def test_progress_callback_on_concurrent_path(
        self, paths: runner.Paths, raw_claim: Any
    ) -> None:
        seen: list[int] = []
        runner.run(
            self._extractor(4, raw_claim),
            make_docs(4),
            paths,
            workers=2,
            progress=lambda i, total, stats, el: seen.append(i),
        )
        assert sorted(seen) == [1, 2, 3, 4]

    def test_creates_output_directory(self, paths: runner.Paths, raw_claim: Any) -> None:
        assert not paths.claims.parent.exists()
        runner.run(self._extractor(1, raw_claim), make_docs(1), paths, workers=1)
        assert paths.claims.exists()

    def test_appends_on_resume(self, paths: runner.Paths, raw_claim: Any) -> None:
        runner.run(self._extractor(1, raw_claim), make_docs(1), paths, workers=1)
        first = paths.done.read_text()
        docs2 = make_docs(2)[1:]
        runner.run(self._extractor(1, raw_claim), docs2, paths, workers=1)
        assert paths.done.read_text().startswith(first)
        assert len(paths.done.read_text().strip().split("\n")) == 2

    def test_empty_document_list(self, paths: runner.Paths) -> None:
        stats, _ = runner.run(Extractor(client=FakeClient([])), [], paths, workers=1)
        assert stats.docs == 0

    def test_injected_clock_controls_elapsed(self, paths: runner.Paths, raw_claim: Any) -> None:
        ticks = iter([100.0, 105.0])
        stats, elapsed = runner.run(
            self._extractor(1, raw_claim), make_docs(1), paths, workers=1,
            now=lambda: next(ticks),
        )
        assert elapsed == 5.0
        assert stats.docs == 1

    def test_retryable_then_success_still_records(self, paths: runner.Paths, raw_claim: Any) -> None:
        ex = Extractor(
            client=FakeClient([RateLimitError(), message_with_claims([raw_claim])]),
            sleep=lambda _: None,
            jitter=lambda: 0.0,
        )
        stats, _ = runner.run(ex, make_docs(1), paths, workers=1)
        assert stats.docs == 1 and stats.failed == 0

    def test_unparseable_response_does_not_abort_the_run(
        self, paths: runner.Paths, raw_claim: Any
    ) -> None:
        # A truncated response used to raise JSONDecodeError straight through
        # fut.result() and kill the whole run.
        bad = FakeMessage(content=[FakeBlock("text", "{trunc")], stop_reason="max_tokens")
        ex = Extractor(
            client=FakeClient([bad, message_with_claims([raw_claim])]),
            sleep=lambda _: None,
        )
        stats, _ = runner.run(ex, make_docs(2), paths, workers=1)
        assert stats.failed == 1
        assert stats.docs == 1
        assert "unparseable" in paths.failed.read_text()

    def test_unexpected_exception_is_contained(
        self, paths: runner.Paths
    ) -> None:
        class Exploding:
            @property
            def messages(self) -> Any:
                raise OSError("disk full")

        ex = Extractor(client=Exploding(), sleep=lambda _: None)
        stats, _ = runner.run(ex, make_docs(2), paths, workers=1)
        assert stats.failed == 2
        assert "OSError" in paths.failed.read_text()

    def test_refusal_is_counted(self, paths: runner.Paths) -> None:
        ex = Extractor(
            client=FakeClient([FakeMessage(content=[], stop_reason="refusal")]),
            sleep=lambda _: None,
        )
        stats, _ = runner.run(ex, make_docs(1), paths, workers=1)
        assert stats.refusals == 1


class TestMakeProgress:
    def test_prints_on_interval_and_final(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = runner.make_progress(every=2)
        stats = RunStats(claims=5)
        report(1, 3, stats, 1.0)
        assert capsys.readouterr().out == ""
        report(2, 3, stats, 1.0)
        assert "2/3" in capsys.readouterr().out
        report(3, 3, stats, 1.0)
        assert "3/3" in capsys.readouterr().out

    def test_zero_elapsed_is_safe(self, capsys: pytest.CaptureFixture[str]) -> None:
        runner.make_progress(every=1)(1, 1, RunStats(), 0.0)
        assert "0.0 docs/s" in capsys.readouterr().out

