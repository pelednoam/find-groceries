"""Tests for the gold-set scorer.

The scorer is the only thing that can tell us stage 2 regressed, so a scorer
that quietly reports 100% is worse than none at all. These target that: the
metrics must move in the right direction, and the gold set itself must stay
consistent with the schema stage 2 can actually emit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from groceries import evaluate, extract
from groceries.extract import Extractor
from groceries.paths import ROOT
from tests.conftest import BadRequestError, FakeClient, message_with_claims

GOLD_PATH = ROOT / "gold" / "extraction_gold.jsonl"


def case(**over: Any) -> Any:
    base: dict[str, Any] = {
        "id": "c1",
        "why": "a reason",
        "text": "Market Basket produce is cheap.",
        "parent_body": "",
        "stores": ["Market Basket"],
        "expected": [],
    }
    base.update(over)
    return base


def expected(store: str = "Market Basket", **over: Any) -> dict[str, Any]:
    base = {
        "store": store,
        "category": "produce",
        "sentiment": "positive",
        "price_signal": "cheap",
        "confidence": "high",
        "transient": False,
        "comparator_store": "",
    }
    base.update(over)
    return base


def emitted(**over: Any) -> dict[str, Any]:
    base = {**expected(), "location": "", "item": "", "claim": "Produce is cheap."}
    base.update(over)
    return base


def extractor_returning(*payloads: Any) -> Extractor:
    return Extractor(
        client=FakeClient([message_with_claims(p) for p in payloads]),
        sleep=lambda _: None,
    )


class TestTriples:
    def test_projects_the_scored_fields(self) -> None:
        assert evaluate.triples([emitted()]) == {
            ("Market Basket", "produce", "positive")
        }

    def test_flags_index_the_unscored_ones(self) -> None:
        got = evaluate.flags([emitted(transient=True, comparator_store="Aldi")])
        assert got[("Market Basket", "produce", "positive")] == (True, "Aldi")

    def test_absent_flags_default_rather_than_raise(self) -> None:
        bare = {"store": "Aldi", "category": "meat", "sentiment": "neutral"}
        assert evaluate.flags([bare])[("Aldi", "meat", "neutral")] == (False, "")


class TestAsCandidate:
    def test_produces_something_stage2_can_consume(self) -> None:
        cand = evaluate.as_candidate(case(parent_body="p"))
        # The real prompt builder must accept it without a KeyError.
        assert "Market Basket produce is cheap." in extract.user_message(cand)
        assert cand["parent_body"] == "p"


class TestScoring:
    def test_a_perfect_run_scores_one(self) -> None:
        score = evaluate.evaluate(
            extractor_returning([emitted()]), [case(expected=[expected()])]
        )
        assert score.exact_match() == 1.0
        assert score.precision() == score.recall() == score.f1() == 1.0

    def test_a_spurious_claim_costs_precision_not_recall(self) -> None:
        score = evaluate.evaluate(
            extractor_returning([emitted(), emitted(store="Aldi")]),
            [case(expected=[expected()])],
        )
        assert score.recall() == 1.0
        assert score.precision() == pytest.approx(0.5)
        assert score.exact_match() == 0.0

    def test_a_missed_claim_costs_recall_not_precision(self) -> None:
        score = evaluate.evaluate(
            extractor_returning([emitted()]),
            [case(expected=[expected(), expected(store="Aldi")])],
        )
        assert score.precision() == 1.0
        assert score.recall() == pytest.approx(0.5)

    def test_the_wrong_sentiment_is_both_a_miss_and_a_spurious(self) -> None:
        score = evaluate.evaluate(
            extractor_returning([emitted(sentiment="neutral")]),
            [case(expected=[expected()])],
        )
        assert score.precision() == 0.0 and score.recall() == 0.0

    def test_silence_is_scored_separately(self) -> None:
        score = evaluate.evaluate(
            extractor_returning([emitted()], []),
            [case(id="loud", expected=[]), case(id="quiet", expected=[])],
        )
        assert score.silence_accuracy() == pytest.approx(0.5)

    def test_silence_accuracy_with_no_silent_cases(self) -> None:
        score = evaluate.evaluate(
            extractor_returning([emitted()]), [case(expected=[expected()])]
        )
        assert score.silence_accuracy() == 1.0

    def test_flags_are_scored_over_matched_claims(self) -> None:
        # Right claim, wrong `transient` — a closing-down sale entering the
        # aggregate as a durable property of the store.
        score = evaluate.evaluate(
            extractor_returning([emitted(transient=False)]),
            [case(expected=[expected(transient=True)])],
        )
        assert score.precision() == 1.0 and score.recall() == 1.0
        assert score.flag_accuracy() == 0.0
        assert score.exact_match() == 0.0, "a flag disagreement is not an exact match"

    def test_flag_accuracy_ignores_unmatched_claims(self) -> None:
        score = evaluate.evaluate(
            extractor_returning([emitted(store="Aldi")]),
            [case(expected=[expected()])],
        )
        assert score.flag_accuracy() == 1.0

    def test_an_extraction_error_fails_the_case_without_ending_the_run(self) -> None:
        ex = Extractor(
            client=FakeClient([BadRequestError(), message_with_claims([emitted()])]),
            sleep=lambda _: None,
        )
        score = evaluate.evaluate(
            ex, [case(id="boom", expected=[expected()]), case(id="ok", expected=[expected()])]
        )
        assert score.errors == 1
        assert score.exact_match() == pytest.approx(0.5)

    def test_an_empty_score_is_vacuously_perfect(self) -> None:
        score = evaluate.Score()
        assert score.exact_match() == score.f1() == 1.0
        assert score.flag_accuracy() == 1.0

    def test_f1_is_zero_when_nothing_matches(self) -> None:
        score = evaluate.evaluate(
            extractor_returning([emitted(store="Aldi")]),
            [case(expected=[expected()])],
        )
        assert score.f1() == 0.0


class TestReadGold:
    def test_reads_the_shipped_set(self) -> None:
        assert len(evaluate.read_gold(GOLD_PATH)) >= 20

    def test_commented_cases_are_parked_not_read(self, tmp_path: Path) -> None:
        p = tmp_path / "g.jsonl"
        p.write_text(
            json.dumps(case()) + "\n" + json.dumps(case(id="#parked")) + "\n",
            encoding="utf-8",
        )
        assert [c["id"] for c in evaluate.read_gold(p)] == ["c1"]

    def test_a_corrupt_gold_file_is_fatal(self, tmp_path: Path) -> None:
        # Unlike the corpus, the gold set is small and hand-maintained; a line
        # that silently vanishes would quietly weaken the measurement.
        p = tmp_path / "g.jsonl"
        p.write_text(json.dumps(case()) + "\n{trunc", encoding="utf-8")
        with pytest.raises(ValueError, match="unparseable"):
            evaluate.read_gold(p)


class TestGoldSetIntegrity:
    """The gold set is data, and data rots. These keep it honest."""

    @pytest.fixture
    def cases(self) -> list[Any]:
        return evaluate.read_gold(GOLD_PATH)

    def test_ids_are_unique(self, cases: list[Any]) -> None:
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids))

    def test_every_case_says_why_it_exists(self, cases: list[Any]) -> None:
        assert all(len(c["why"]) > 20 for c in cases)

    def test_every_label_is_expressible_in_the_schema(self, cases: list[Any]) -> None:
        """A label stage 2 structurally cannot emit is an unwinnable case."""
        for c in cases:
            for e in c["expected"]:
                assert e["store"] in extract.STORES, c["id"]
                assert e["category"] in extract.CATEGORIES, c["id"]
                assert e["sentiment"] in {"positive", "negative", "mixed", "neutral"}
                assert e["price_signal"] in {"cheap", "expensive", "fair", "none"}
                assert e["confidence"] in {"high", "medium", "low"}
                assert e["comparator_store"] in {*extract.STORES, ""}, c["id"]

    def test_labels_are_internally_consistent(self, cases: list[Any]) -> None:
        for c in cases:
            for e in c["expected"]:
                # A store cannot be its own comparator.
                assert e["comparator_store"] != e["store"], c["id"]

    def test_the_prefilter_stores_cover_the_labels(self, cases: list[Any]) -> None:
        """Stage 1 only ever sends stores it matched; a label naming a store
        outside that list is asking the model to do stage 1's job."""
        for c in cases:
            for e in c["expected"]:
                assert e["store"] in c["stores"], c["id"]

    def test_silence_is_well_represented(self, cases: list[Any]) -> None:
        # Most of the corpus supports no claim; if the gold set does not
        # reflect that, a claim-happy model scores well and floods stage 3.
        quiet = [c for c in cases if not c["expected"]]
        assert len(quiet) >= len(cases) // 4

    def test_the_hard_rules_each_have_a_case(self, cases: list[Any]) -> None:
        labels = [e for c in cases for e in c["expected"]]
        assert any(e["transient"] for e in labels), "no transient case"
        assert any(e["comparator_store"] for e in labels), "no comparator case"
        assert any(e["store"] == "other" for e in labels), "no unlisted-store case"
        assert any(c["parent_body"] for c in cases), "no inherited-referent case"
        assert any(e["sentiment"] == "neutral" for e in labels), "no neutral case"


class TestFormatting:
    def test_a_clean_report_lists_no_disagreements(self) -> None:
        score = evaluate.evaluate(
            extractor_returning([emitted()]), [case(expected=[expected()])]
        )
        text = evaluate.format_score(score)
        assert "exact match      100%" in text
        assert "disagreements" not in text

    def test_disagreements_name_the_failure_mode(self) -> None:
        score = evaluate.evaluate(
            extractor_returning([emitted(store="Aldi")]),
            [case(why="sarcasm is common", expected=[expected()])],
        )
        text = evaluate.format_score(score)
        assert "sarcasm is common" in text
        assert "missed" in text and "spurious" in text

    def test_errors_are_surfaced(self) -> None:
        ex = Extractor(client=FakeClient([BadRequestError()]), sleep=lambda _: None)
        text = evaluate.format_score(evaluate.evaluate(ex, [case()]))
        assert "errors" in text and "ERROR" in text

    def test_flag_disagreements_show_both_sides(self) -> None:
        score = evaluate.evaluate(
            extractor_returning([emitted(transient=False)]),
            [case(expected=[expected(transient=True)])],
        )
        assert "want (True, '')" in evaluate.format_score(score)


class TestReport:
    def test_writes_a_machine_readable_report(self, tmp_path: Path) -> None:
        score = evaluate.evaluate(
            extractor_returning([emitted(transient=False)]),
            [case(expected=[expected(transient=True)])],
        )
        out = tmp_path / "nested" / "report.json"
        evaluate.write_report(score, out)
        payload = json.loads(out.read_text())
        assert payload["precision"] == 1.0
        assert payload["cases"][0]["flag_disagreements"][0]["want"] == [True, ""]
        assert payload["cases"][0]["exact"] is False


class TestClaimsFor:
    def test_selects_by_source_id(self, sourced_claim: Any) -> None:
        other: Any = {**sourced_claim, "source_id": "zzz"}
        got = evaluate.claims_for(case(id="abc123"), [sourced_claim, other])
        assert got == [sourced_claim]
