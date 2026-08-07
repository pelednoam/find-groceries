"""Tests for stage 3 aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from groceries import aggregate
from groceries.types import SourcedClaim

NOW = 1_800_000_000
YEAR = int(aggregate.SECONDS_PER_YEAR)


def claim(**over: Any) -> SourcedClaim:
    base: dict[str, Any] = {
        "store": "Market Basket",
        "location": "",
        "category": "produce",
        "item": "",
        "claim": "Produce is good.",
        "sentiment": "positive",
        "price_signal": "cheap",
        "confidence": "high",
        "source_id": "s1",
        "source_key": "c_boston_s1",
        "subreddit": "boston",
        "kind": "comments",
        "created_utc": NOW,
        "permalink": "/r/boston/comments/a/_/s1/",
        "score": 3,
    }
    base.update(over)
    return base  # type: ignore[return-value]


class TestRecencyWeight:
    def test_now_is_full_weight(self) -> None:
        assert aggregate.recency_weight(NOW, NOW) == 1.0

    def test_one_half_life_halves(self) -> None:
        assert aggregate.recency_weight(NOW - 4 * YEAR, NOW) == pytest.approx(0.5, rel=1e-3)

    def test_two_half_lives_quarter(self) -> None:
        assert aggregate.recency_weight(NOW - 8 * YEAR, NOW) == pytest.approx(0.25, rel=1e-3)

    def test_future_timestamps_clamp(self) -> None:
        assert aggregate.recency_weight(NOW + YEAR, NOW) == 1.0

    def test_custom_half_life(self) -> None:
        got = aggregate.recency_weight(NOW - YEAR, NOW, half_life_years=1.0)
        assert got == pytest.approx(0.5, rel=1e-3)


class TestClaimWeight:
    def test_confidence_scales_weight(self) -> None:
        high = aggregate.claim_weight(claim(confidence="high"), NOW)
        med = aggregate.claim_weight(claim(confidence="medium"), NOW)
        low = aggregate.claim_weight(claim(confidence="low"), NOW)
        assert high == 1.0
        assert med == pytest.approx(0.6)
        assert low == pytest.approx(0.3)

    def test_unknown_confidence_uses_default(self) -> None:
        assert aggregate.claim_weight(claim(confidence="bogus"), NOW) == pytest.approx(0.6)

    def test_age_and_confidence_compose(self) -> None:
        got = aggregate.claim_weight(claim(confidence="medium", created_utc=NOW - 4 * YEAR), NOW)
        assert got == pytest.approx(0.3, rel=1e-2)

    def test_old_claims_lose_weight_rather_than_gaining_it(self) -> None:
        # Regression guard: created_utc used to default to `now` when absent,
        # which promoted corrupt rows to the freshest evidence in the cell.
        old = aggregate.claim_weight(claim(created_utc=NOW - 8 * YEAR), NOW)
        fresh = aggregate.claim_weight(claim(), NOW)
        assert old < fresh


class TestCell:
    def test_accumulates_counts_and_score(self) -> None:
        cell = aggregate.Cell()
        cell.add(claim(sentiment="positive"), 1.0)
        cell.add(claim(sentiment="negative"), 1.0)
        assert cell.n == 2
        assert cell.weight == 2.0
        assert cell.sentiment() == 0.0

    def test_sentiment_with_no_weight(self) -> None:
        assert aggregate.Cell().sentiment() == 0.0

    def test_dominant_price_picks_heaviest(self) -> None:
        cell = aggregate.Cell()
        cell.add(claim(price_signal="cheap"), 1.0)
        cell.add(claim(price_signal="expensive"), 3.0)
        assert cell.dominant_price() == "expensive"

    def test_price_none_is_ignored(self) -> None:
        cell = aggregate.Cell()
        cell.add(claim(price_signal="none"), 1.0)
        assert cell.dominant_price() is None

    def test_top_examples_are_weight_ordered(self) -> None:
        cell = aggregate.Cell()
        cell.add(claim(claim="light"), 0.1)
        cell.add(claim(claim="heavy"), 5.0)
        assert [c["claim"] for c in cell.top_examples(2)] == ["heavy", "light"]

    def test_top_examples_truncates(self) -> None:
        cell = aggregate.Cell()
        for i in range(10):
            cell.add(claim(claim=f"c{i}"), float(i))
        assert len(cell.top_examples(3)) == 3

    def test_unknown_sentiment_scores_zero(self) -> None:
        cell = aggregate.Cell()
        cell.add(claim(sentiment="bogus"), 1.0)
        assert cell.sentiment() == 0.0


class TestAggregate:
    def test_groups_by_store_and_category(self) -> None:
        out = aggregate.aggregate([claim(), claim(category="price_overall")], now=NOW)
        assert set(out["stores"]["Market Basket"]) == {"produce", "price_overall"}

    def test_min_weight_suppresses_thin_cells(self) -> None:
        out = aggregate.aggregate([claim(confidence="low")], now=NOW, min_weight=1.0)
        assert out["stores"] == {}

    def test_min_weight_zero_keeps_everything(self) -> None:
        out = aggregate.aggregate([claim(confidence="low")], now=NOW, min_weight=0.0)
        assert "Market Basket" in out["stores"]

    def test_store_totals_span_categories(self) -> None:
        out = aggregate.aggregate([claim(), claim(category="meat")], now=NOW)
        assert out["store_totals"]["Market Basket"]["n_claims"] == 2

    def test_totals_include_suppressed_cells(self) -> None:
        out = aggregate.aggregate([claim(confidence="low")], now=NOW, min_weight=99.0)
        assert out["stores"] == {}
        assert out["store_totals"]["Market Basket"]["n_claims"] == 1

    def test_evidence_carries_permalink_and_date(self) -> None:
        out = aggregate.aggregate([claim()], now=NOW)
        ev = out["stores"]["Market Basket"]["produce"]["evidence"][0]
        assert ev["permalink"].startswith("/r/boston/")
        assert ev["date"] == "2027-01"
        assert ev["confidence"] == "high"

    def test_max_examples_is_respected(self) -> None:
        claims = [claim(claim=f"c{i}") for i in range(8)]
        out = aggregate.aggregate(claims, now=NOW, max_examples=2)
        assert len(out["stores"]["Market Basket"]["produce"]["evidence"]) == 2

    def test_empty_input(self) -> None:
        out = aggregate.aggregate([], now=NOW)
        assert out["stores"] == {} and out["store_totals"] == {}

    def test_defaults_to_wall_clock(self) -> None:
        out = aggregate.aggregate([claim()])
        assert out["generated_at"].endswith("Z")

    def test_negative_sentiment_reflected(self) -> None:
        out = aggregate.aggregate([claim(sentiment="negative")], now=NOW)
        assert out["stores"]["Market Basket"]["produce"]["sentiment"] == -1.0

    def test_unknown_confidence_still_scores(self) -> None:
        out = aggregate.aggregate([claim(confidence="bogus")], now=NOW, min_weight=0.0)
        ev = out["stores"]["Market Basket"]["produce"]["evidence"][0]
        assert ev["confidence"] == "bogus"


class TestDedupe:
    def test_identical_claims_from_one_document_collapse(self) -> None:
        assert len(aggregate.dedupe([claim(), claim()])) == 1

    def test_distinct_claims_from_one_document_are_kept(self) -> None:
        pair = [claim(), claim(category="meat")]
        assert len(aggregate.dedupe(pair)) == 2

    def test_same_text_from_different_documents_is_kept(self) -> None:
        pair = [claim(source_key="c_boston_a"), claim(source_key="c_boston_b")]
        assert len(aggregate.dedupe(pair)) == 2

    def test_post_and_comment_sharing_an_id_are_distinct(self) -> None:
        # Reddit base-36 ids collide across the post/comment namespaces, so
        # source_id alone would wrongly merge these two.
        pair = [
            claim(source_id="abc", source_key="c_boston_abc", kind="comments"),
            claim(source_id="abc", source_key="p_boston_abc", kind="posts"),
        ]
        assert len(aggregate.dedupe(pair)) == 2

    def test_crash_resume_duplicates_do_not_double_count(self) -> None:
        # Sink flushes claims before the done-key, so a kill between the two
        # re-extracts the document and appends its claims again.
        out = aggregate.aggregate([claim(), claim()], now=NOW)
        assert out["store_totals"]["Market Basket"]["n_claims"] == 1


class TestMonth:
    def test_formats_year_month(self) -> None:
        assert aggregate.month(NOW) == "2027-01"


class TestIO:
    def test_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "claims.jsonl"
        p.write_text(json.dumps(claim()) + "\n\n", encoding="utf-8")
        claims, dropped = aggregate.read_claims(p)
        assert len(claims) == 1 and dropped == 0

    def test_rows_missing_provenance_are_dropped_not_defaulted(
        self, tmp_path: Path
    ) -> None:
        bare: dict[str, Any] = dict(claim())
        bare.pop("created_utc")
        p = tmp_path / "claims.jsonl"
        p.write_text(
            json.dumps(claim()) + "\n" + json.dumps(bare) + "\n", encoding="utf-8"
        )
        claims, dropped = aggregate.read_claims(p)
        assert len(claims) == 1 and dropped == 1

    def test_write_creates_parents(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "v.json"
        aggregate.write_verdicts(aggregate.aggregate([claim()], now=NOW), out)
        assert json.loads(out.read_text())["half_life_years"] == 4.0


class TestCorruptInput:
    def test_a_truncated_line_does_not_lose_the_good_rows(self, tmp_path: Path) -> None:
        # A SIGKILL mid-flush leaves a partial final line; raising there would
        # hold every paid-for row hostage to one truncated one.
        p = tmp_path / "claims.jsonl"
        p.write_text(
            json.dumps(claim()) + "\n" + json.dumps(claim(source_key="c_b_x"))[:40],
            encoding="utf-8",
        )
        claims, dropped = aggregate.read_claims(p)
        assert len(claims) == 1 and dropped == 1

    @pytest.mark.parametrize("junk", ["null", "[1,2]", '"hello"', "42"])
    def test_non_object_rows_are_rejected_not_crashed_on(
        self, tmp_path: Path, junk: str
    ) -> None:
        p = tmp_path / "claims.jsonl"
        p.write_text(json.dumps(claim()) + "\n" + junk + "\n", encoding="utf-8")
        claims, dropped = aggregate.read_claims(p)
        assert len(claims) == 1 and dropped == 1

    def test_wrongly_typed_created_utc_is_rejected(self, tmp_path: Path) -> None:
        # Presence was checked but not type, so a string timestamp reached the
        # scoring math and raised TypeError there instead.
        bad: dict[str, Any] = dict(claim())
        bad["created_utc"] = "1700000000"
        p = tmp_path / "claims.jsonl"
        p.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        claims, dropped = aggregate.read_claims(p)
        assert claims == [] and dropped == 1


class TestRowValidation:
    @pytest.mark.parametrize("junk", ["null", "[1,2]", '"hi"', "42"])
    def test_non_object_rows_rejected(self, tmp_path: Path, junk: str) -> None:
        p = tmp_path / "c.jsonl"
        p.write_text(json.dumps(claim()) + "\n" + junk + "\n", encoding="utf-8")
        claims, dropped = aggregate.read_claims(p)
        assert len(claims) == 1 and dropped == 1

    def test_wrongly_typed_fields_rejected(self, tmp_path: Path) -> None:
        # Presence-only checking let {"store": []} reach dedupe and raise
        # there on an unhashable key instead of being rejected here.
        bad: dict[str, Any] = dict(claim())
        bad["store"] = []
        worse: dict[str, Any] = dict(claim())
        worse["created_utc"] = "1700000000"
        p = tmp_path / "c.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in (claim(), bad, worse)),
                     encoding="utf-8")
        claims, dropped = aggregate.read_claims(p)
        assert len(claims) == 1 and dropped == 2


class TestFormatting:
    def test_totals_table_lists_stores(self) -> None:
        summary = aggregate.aggregate([claim(), claim(store="Aldi")], now=NOW)
        text = aggregate.format_totals(summary)
        assert "Market Basket" in text and "Aldi" in text

    def test_store_readout(self) -> None:
        summary = aggregate.aggregate([claim()], now=NOW)
        text = aggregate.format_store(summary, "Market Basket")
        assert "produce" in text and "Produce is good." in text

    def test_store_readout_for_unknown_store(self) -> None:
        summary = aggregate.aggregate([claim()], now=NOW)
        assert "no claims above" in aggregate.format_store(summary, "Nonexistent")

    @pytest.mark.parametrize(
        ("sentiment", "mark"), [("positive", "+"), ("negative", "-"), ("neutral", "~")]
    )
    def test_sentiment_marks(self, sentiment: str, mark: str) -> None:
        summary = aggregate.aggregate([claim(sentiment=sentiment)], now=NOW)
        line = next(
            l for l in aggregate.format_store(summary, "Market Basket").splitlines()
            if "produce" in l
        )
        assert f" {mark} sentiment" in line

    def test_evidence_truncated_to_max(self) -> None:
        claims = [claim(claim=f"claim number {i}") for i in range(6)]
        summary = aggregate.aggregate(claims, now=NOW)
        text = aggregate.format_store(summary, "Market Basket", max_evidence=2)
        assert text.count("[2027-01]") == 2
