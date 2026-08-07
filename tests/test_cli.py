"""Tests for the three stage CLIs.

The value here is entirely in the refusals. Each stage used to have a path
where it did nothing and exited 0 — a mistyped `--corpus`, a stale working
set, a claims file from an older schema — and because every stage reported
success, the pipeline as a whole reported success at producing nothing.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import aggregate_claims  # noqa: E402
import evaluate_extraction  # noqa: E402
import extract_claims  # noqa: E402
import select_evaluative  # noqa: E402

from groceries import paths  # noqa: E402
from groceries.types import Candidate  # noqa: E402
from tests.conftest import FakeClient, message_with_claims  # noqa: E402

GOOD_BODY = "Market Basket is so much cheaper for produce than anywhere else"


def write_shard(root: Path, sub: str, bodies: list[str]) -> None:
    d = root / "comments" / sub
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "2020-01.ndjson.gz", "wt", encoding="utf-8") as fh:
        for i, body in enumerate(bodies):
            fh.write(json.dumps({"id": f"x{i}", "created_utc": 1, "body": body}) + "\n")


class TestSelectCLI:
    def test_writes_a_working_set(self, tmp_path: Path, capsys: Any) -> None:
        write_shard(tmp_path / "corpus", "boston", [GOOD_BODY])
        out = tmp_path / "ws.jsonl"
        code = select_evaluative.main(
            ["--corpus", str(tmp_path / "corpus"), "--out", str(out)]
        )
        assert code == 0
        assert len(out.read_text().splitlines()) == 1
        assert "Market Basket" in capsys.readouterr().out

    def test_a_corpus_path_with_no_shards_is_refused(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        out = tmp_path / "ws.jsonl"
        code = select_evaluative.main(
            ["--corpus", str(tmp_path / "typo"), "--out", str(out)]
        )
        assert code == 2
        # Critically, the previous working set is left alone.
        assert not out.exists()
        assert "refusing to run" in capsys.readouterr().out

    def test_an_unknown_subreddit_is_refused(self, tmp_path: Path, capsys: Any) -> None:
        write_shard(tmp_path / "corpus", "boston", [GOOD_BODY])
        code = select_evaluative.main(
            ["--corpus", str(tmp_path / "corpus"),
             "--out", str(tmp_path / "ws.jsonl"),
             "--subreddits", "Cambridge"]  # the real one is CambridgeMA
        )
        assert code == 2
        assert "no shards for Cambridge" in capsys.readouterr().out

    def test_a_known_subreddit_is_accepted(self, tmp_path: Path) -> None:
        write_shard(tmp_path / "corpus", "boston", [GOOD_BODY])
        code = select_evaluative.main(
            ["--corpus", str(tmp_path / "corpus"),
             "--out", str(tmp_path / "ws.jsonl"),
             "--subreddits", "boston"]
        )
        assert code == 0

    def test_an_empty_result_is_not_a_success(self, tmp_path: Path, capsys: Any) -> None:
        write_shard(tmp_path / "corpus", "boston", ["nothing of interest here at all"])
        code = select_evaluative.main(
            ["--corpus", str(tmp_path / "corpus"), "--out", str(tmp_path / "ws.jsonl")]
        )
        assert code == 1
        assert "refusing to call this a success" in capsys.readouterr().out

    def test_report_handles_a_zero_document_scan(self) -> None:
        from groceries.select import SelectionReport

        assert "0 documents" in select_evaluative.format_report(SelectionReport())


class TestExtractCLI:
    def test_a_missing_working_set_is_refused(self, tmp_path: Path, capsys: Any) -> None:
        code = extract_claims.main(["--working-set", str(tmp_path / "nope.jsonl")])
        assert code == 2
        assert "select_evaluative" in capsys.readouterr().out

    def test_an_unusable_working_set_is_refused_before_spending(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        ws = tmp_path / "ws.jsonl"
        ws.write_text(json.dumps({"id": "a"}) + "\n", encoding="utf-8")
        code = extract_claims.main(["--working-set", str(ws)])
        assert code == 2
        out = capsys.readouterr().out
        assert "dropped 1 malformed rows" in out
        assert "no usable candidates" in out

    def test_an_unpriced_model_with_a_ceiling_is_refused(
        self, tmp_path: Path, candidate: Candidate, capsys: Any
    ) -> None:
        ws = tmp_path / "ws.jsonl"
        ws.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
        code = extract_claims.main(
            ["--working-set", str(ws), "--model", "us.anthropic.made-up-v9"]
        )
        assert code == 2
        assert "no rate card" in capsys.readouterr().out

    def test_a_non_finite_ceiling_is_refused(
        self, tmp_path: Path, candidate: Candidate, capsys: Any
    ) -> None:
        ws = tmp_path / "ws.jsonl"
        ws.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
        code = extract_claims.main(
            ["--working-set", str(ws), "--max-cost", "inf"]
        )
        assert code == 2
        assert "finite" in capsys.readouterr().out

    def test_the_lock_prevents_a_concurrent_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        monkeypatch.setattr(extract_claims, "EXTRACT_DIR", tmp_path)
        monkeypatch.setattr(extract_claims, "LOCK", tmp_path / ".extract.lock")
        (tmp_path / ".extract.lock").write_text("999")
        assert extract_claims.main(["--working-set", str(tmp_path / "n.jsonl")]) == 2
        assert "another extraction appears to be running" in capsys.readouterr().out

    def test_the_lock_is_released_on_the_way_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lock = tmp_path / ".extract.lock"
        monkeypatch.setattr(extract_claims, "EXTRACT_DIR", tmp_path)
        monkeypatch.setattr(extract_claims, "LOCK", lock)
        extract_claims.main(["--working-set", str(tmp_path / "nope.jsonl")])
        assert not lock.exists()

    def test_nothing_left_to_do_is_a_success(
        self, tmp_path: Path, candidate: Candidate, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = tmp_path / "ws.jsonl"
        ws.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
        monkeypatch.setattr(extract_claims, "DONE", tmp_path / "done.txt")
        (tmp_path / "done.txt").write_text("c_boston_abc123\n", encoding="utf-8")
        assert extract_claims.main(["--working-set", str(ws)]) == 0


class TestAggregateCLI:
    def test_a_missing_claims_file_is_refused(self, tmp_path: Path, capsys: Any) -> None:
        code = aggregate_claims.main(["--claims", str(tmp_path / "nope.jsonl")])
        assert code == 2
        assert "extract_claims" in capsys.readouterr().out

    def test_a_wholly_rejected_claims_file_does_not_overwrite(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        claims = tmp_path / "claims.jsonl"
        claims.write_text(json.dumps({"store": "Aldi"}) + "\n", encoding="utf-8")
        out = tmp_path / "verdicts.json"
        out.write_text('{"good": true}', encoding="utf-8")
        assert aggregate_claims.main(["--claims", str(claims), "--out", str(out)]) == 1
        assert json.loads(out.read_text()) == {"good": True}
        assert "older schema" in capsys.readouterr().out

    def test_an_empty_claims_file_is_refused(self, tmp_path: Path, capsys: Any) -> None:
        claims = tmp_path / "claims.jsonl"
        claims.write_text("", encoding="utf-8")
        assert aggregate_claims.main(["--claims", str(claims)]) == 1
        assert "older schema" not in capsys.readouterr().out

    def test_writes_verdicts_with_corpus_provenance(
        self, tmp_path: Path, sourced_claim: Any, capsys: Any
    ) -> None:
        claims = tmp_path / "claims.jsonl"
        claims.write_text(json.dumps(sourced_claim) + "\n", encoding="utf-8")
        out = tmp_path / "verdicts.json"
        code = aggregate_claims.main(
            ["--claims", str(claims), "--out", str(out), "--min-weight", "0"]
        )
        assert code == 0
        summary = json.loads(out.read_text())
        assert summary["corpus"]["claims_file"] == "claims.jsonl"
        assert "Market Basket" in capsys.readouterr().out

    def test_a_named_store_prints_its_readout(
        self, tmp_path: Path, sourced_claim: Any, capsys: Any
    ) -> None:
        claims = tmp_path / "claims.jsonl"
        claims.write_text(json.dumps(sourced_claim) + "\n", encoding="utf-8")
        aggregate_claims.main(
            ["--claims", str(claims), "--out", str(tmp_path / "v.json"),
             "--store", "Market Basket", "--min-weight", "0"]
        )
        assert "produce" in capsys.readouterr().out

    def test_a_partly_malformed_file_still_aggregates(
        self, tmp_path: Path, sourced_claim: Any, capsys: Any
    ) -> None:
        claims = tmp_path / "claims.jsonl"
        claims.write_text(
            json.dumps(sourced_claim) + "\n" + json.dumps({"store": "Aldi"}) + "\n",
            encoding="utf-8",
        )
        code = aggregate_claims.main(
            ["--claims", str(claims), "--out", str(tmp_path / "v.json"),
             "--min-weight", "0"]
        )
        assert code == 0
        assert "dropped 1 malformed" in capsys.readouterr().out

    def test_corpus_provenance_counts_the_pipeline_inputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws, done = tmp_path / "ws.jsonl", tmp_path / "done.txt"
        ws.write_text("a\nb\n\nc\n", encoding="utf-8")
        done.write_text("k1\nk2\n", encoding="utf-8")
        monkeypatch.setattr(aggregate_claims, "WORKING_SET", ws)
        monkeypatch.setattr(aggregate_claims, "DONE", done)
        got = aggregate_claims.corpus_provenance(tmp_path / "claims.jsonl")
        assert got == {
            "claims_file": "claims.jsonl",
            "working_set": 3,
            "documents_extracted": 2,
        }

    def test_corpus_provenance_omits_files_that_do_not_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(aggregate_claims, "WORKING_SET", tmp_path / "nope.jsonl")
        monkeypatch.setattr(aggregate_claims, "DONE", tmp_path / "nope.txt")
        assert aggregate_claims.corpus_provenance(tmp_path / "c.jsonl") == {
            "claims_file": "c.jsonl"
        }


class TestEvaluateCLI:
    def test_a_missing_gold_set_is_refused(self, tmp_path: Path, capsys: Any) -> None:
        code = evaluate_extraction.main(["--gold", str(tmp_path / "nope.jsonl")])
        assert code == 2
        assert "no gold set" in capsys.readouterr().out

    def test_a_gold_set_with_every_case_parked_is_refused(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        gold = tmp_path / "g.jsonl"
        gold.write_text(
            json.dumps({"id": "#parked", "why": "", "text": "", "parent_body": "",
                        "stores": [], "expected": []}) + "\n",
            encoding="utf-8",
        )
        assert evaluate_extraction.main(["--gold", str(gold)]) == 2
        assert "no enabled cases" in capsys.readouterr().out

    def test_scores_and_gates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        gold = tmp_path / "g.jsonl"
        gold.write_text(
            json.dumps({
                "id": "c1", "why": "a documented failure mode", "parent_body": "",
                "text": "Aldi produce is cheap.", "stores": ["Aldi"],
                "expected": [{"store": "Aldi", "category": "produce",
                              "sentiment": "positive", "price_signal": "cheap",
                              "confidence": "high", "transient": False,
                              "comparator_store": ""}],
            }) + "\n",
            encoding="utf-8",
        )
        # The model returns nothing, so the gate must fail the run.
        monkeypatch.setattr(
            evaluate_extraction,
            "build_client",
            lambda region: FakeClient([message_with_claims([])]),
        )
        out = tmp_path / "report.json"
        code = evaluate_extraction.main(
            ["--gold", str(gold), "--out", str(out), "--min-exact-match", "0.9"]
        )
        assert code == 1
        assert "FAIL" in capsys.readouterr().out
        assert json.loads(out.read_text())["recall"] == 0.0

    def test_a_passing_run_exits_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gold = tmp_path / "g.jsonl"
        gold.write_text(
            json.dumps({"id": "c1", "why": "silence is the main risk",
                        "text": "I walked past Aldi.", "parent_body": "",
                        "stores": ["Aldi"], "expected": []}) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            evaluate_extraction,
            "build_client",
            lambda region: FakeClient([message_with_claims([])]),
        )
        code = evaluate_extraction.main(
            ["--gold", str(gold), "--out", str(tmp_path / "r.json"),
             "--min-exact-match", "1.0", "--no-thinking"]
        )
        assert code == 0


def default(module: Any, flag: str) -> Any:
    """The default the CLI would actually use, read off its own parser."""
    return module.build_parser().parse_args([]).__dict__[flag]


class TestSharedPaths:
    """Each stage's default output must be the next stage's default input.

    The bug this guards: stage 1 defaulted to `working_set.jsonl` while stage 2
    defaulted to `working_set_local.jsonl`, so regenerating the working set
    changed nothing stage 2 saw — and both stages exited 0.
    """

    def test_stage1_writes_where_stage2_reads(self) -> None:
        assert default(select_evaluative, "out") == default(
            extract_claims, "working_set"
        )

    def test_stage2_writes_where_stage3_reads(self) -> None:
        # Stage 2's claims path is not a flag; it is fixed by the shared module.
        assert paths.CLAIMS == default(aggregate_claims, "claims")

    def test_stage1_reads_the_fetched_corpus(self) -> None:
        assert default(select_evaluative, "corpus") == paths.CORPUS

    def test_every_output_lives_under_the_data_directory(self) -> None:
        for path in (paths.WORKING_SET, paths.CLAIMS, paths.DONE,
                     paths.FAILED, paths.VERDICTS, paths.LOCK):
            assert paths.DATA in path.parents
