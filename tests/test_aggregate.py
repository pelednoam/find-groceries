"""Tests for stage 3 aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from groceries import aggregate
from groceries.types import SourcedClaim

NOW = 1_770_000_000  # 2026-02
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
        "comparator_store": "",
        "transient": False,
        "source_id": "s1",
        "source_key": "c_boston_s1",
        "subreddit": "boston",
        "kind": "comments",
        "created_utc": NOW,
        "permalink": "/r/boston/comments/a/_/s1/",
        "score": None,
        "author": "",
    }
    base.update(over)
    return base  # type: ignore[return-value]


class TestDecay:
    def test_now_is_full_weight(self) -> None:
        assert aggregate.recency_weight(NOW, NOW, 4.0) == 1.0

    def test_one_half_life_halves(self) -> None:
        got = aggregate.recency_weight(NOW - 4 * YEAR, NOW, 4.0)
        assert got == pytest.approx(0.5, rel=1e-3)

    def test_future_clamps(self) -> None:
        assert aggregate.recency_weight(NOW + YEAR, NOW, 4.0) == 1.0

    def test_monotonic_in_age(self) -> None:
        ws = [aggregate.recency_weight(NOW - y * YEAR, NOW, 4.0) for y in range(12)]
        assert all(a >= b for a, b in zip(ws, ws[1:], strict=False))
        assert all(0.0 < w <= 1.0 for w in ws)

    def test_volatile_categories_age_faster(self) -> None:
        assert aggregate.half_life_for("cleanliness") < aggregate.half_life_for(
            "price_overall"
        )

    def test_unknown_category_uses_default(self) -> None:
        assert aggregate.half_life_for("nope") == aggregate.DEFAULT_HALF_LIFE_YEARS

    def test_category_half_life_reaches_claim_weight(self) -> None:
        old = NOW - 4 * YEAR
        fast = aggregate.claim_weight(claim(category="cleanliness", created_utc=old), NOW)
        slow = aggregate.claim_weight(claim(category="price_overall", created_utc=old), NOW)
        assert fast < slow

    def test_confidence_scales_weight(self) -> None:
        hi = aggregate.claim_weight(claim(confidence="high"), NOW)
        lo = aggregate.claim_weight(claim(confidence="low"), NOW)
        assert hi == 1.0 and lo == pytest.approx(0.3)

    def test_unknown_confidence_uses_default(self) -> None:
        assert aggregate.claim_weight(claim(confidence="?"), NOW) == pytest.approx(0.6)


class TestVoteWeight:
    def test_missing_score_is_neutral(self) -> None:
        assert aggregate.vote_weight(None) == 1.0

    def test_downvoted_is_discounted(self) -> None:
        assert aggregate.vote_weight(-5) == 0.5

    def test_upvotes_help_but_are_capped(self) -> None:
        assert aggregate.vote_weight(0) == 1.0
        assert 1.0 < aggregate.vote_weight(50) <= 2.0
        assert aggregate.vote_weight(100_000) == 2.0

    def test_monotonic(self) -> None:
        ws = [aggregate.vote_weight(s) for s in (0, 5, 50, 500)]
        assert all(a <= b for a, b in zip(ws, ws[1:], strict=False))


class TestCell:
    def test_sentiment_is_weight_normalised_and_shrunk(self) -> None:
        cell = aggregate.Cell()
        cell.add(claim(sentiment="positive"), 3.0)
        cell.add(claim(sentiment="negative"), 1.0)
        assert cell.sentiment() == pytest.approx(2.0 / (4.0 + aggregate.SHRINKAGE_K))
        assert abs(cell.sentiment()) < 0.5

    def test_thin_evidence_reads_closer_to_neutral(self) -> None:
        thin, thick = aggregate.Cell(), aggregate.Cell()
        thin.add(claim(), 1.0)
        for _ in range(40):
            thick.add(claim(), 1.0)
        assert thin.sentiment() < thick.sentiment()
        assert thin.sentiment() < 0.4

    def test_neutral_is_evidence_but_not_in_the_mean(self) -> None:
        cell = aggregate.Cell()
        cell.add(claim(sentiment="positive"), 1.0)
        cell.add(claim(sentiment="neutral"), 1.0)
        only_positive = aggregate.Cell()
        only_positive.add(claim(sentiment="positive"), 1.0)
        assert cell.sentiment() == only_positive.sentiment()
        assert cell.weight > only_positive.weight

    def test_no_valenced_claims(self) -> None:
        cell = aggregate.Cell()
        cell.add(claim(sentiment="neutral"), 1.0)
        assert cell.sentiment() == 0.0

    def test_price_counts_every_signal(self) -> None:
        cell = aggregate.Cell()
        cell.add(claim(price_signal="fair"), 0.53)
        cell.add(claim(price_signal="cheap"), 0.51)
        for w in (0.29, 0.09, 0.04):
            cell.add(claim(price_signal="expensive"), w)
        assert cell.price_counts == {"fair": 1, "cheap": 1, "expensive": 3}
        assert cell.price_level() is not None

    @pytest.mark.parametrize(
        ("signal", "label"),
        [("cheap", "cheap"), ("expensive", "expensive"), ("fair", "fair")],
    )
    def test_price_label_follows_the_level(self, signal: str, label: str) -> None:
        cell = aggregate.Cell()
        for _ in range(20):
            cell.add(claim(price_signal=signal), 1.0)
        assert cell.price_label() == label

    def test_no_price_claims(self) -> None:
        cell = aggregate.Cell()
        cell.add(claim(price_signal="none"), 1.0)
        assert cell.price_level() is None and cell.price_label() is None

    def test_top_examples_are_weight_ordered_and_truncated(self) -> None:
        cell = aggregate.Cell()
        for i in range(10):
            cell.add(claim(claim=f"c{i}"), float(i))
        assert [c["claim"] for c in cell.top_examples(3)] == ["c9", "c8", "c7"]


class TestTotals:
    def test_sentiment_normalised(self) -> None:
        t = aggregate.Totals(n=2, weight=4.0, score=2.0, valenced_weight=4.0)
        assert t.sentiment() == pytest.approx(2.0 / (4.0 + aggregate.SHRINKAGE_K))

    def test_no_valenced_weight(self) -> None:
        assert aggregate.Totals().sentiment() == 0.0


class TestScrub:
    def test_removes_control_bidi_and_zero_width(self) -> None:
        assert aggregate.scrub("a\x1b[31mb") == "a[31mb"
        assert aggregate.scrub("safe‮gnirts") == "safegnirts"
        assert aggregate.scrub("zero​width") == "zerowidth"
        assert aggregate.scrub("line\nbreak\ttab") == "linebreaktab"

    def test_leaves_ordinary_text(self) -> None:
        assert aggregate.scrub("Market Basket — cheap!") == "Market Basket — cheap!"


class TestValidation:
    def test_accepts_a_good_row(self) -> None:
        assert aggregate._is_valid(dict(claim()))

    @pytest.mark.parametrize("junk", [None, [1, 2], "hi", 42])
    def test_rejects_non_objects(self, junk: Any) -> None:
        assert not aggregate._is_valid(junk)

    def test_rejects_missing_provenance(self) -> None:
        bad = dict(claim())
        bad.pop("source_key")
        assert not aggregate._is_valid(bad)

    def test_rejects_wrongly_typed_fields(self) -> None:
        assert not aggregate._is_valid({**claim(), "store": []})
        assert not aggregate._is_valid({**claim(), "created_utc": "1700000000"})

    def test_rejects_out_of_range_timestamps(self) -> None:
        assert not aggregate._is_valid({**claim(), "created_utc": 10**18})
        assert not aggregate._is_valid({**claim(), "created_utc": 0})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("score", "12"), ("score", True), ("transient", "yes"),
            ("location", 5), ("item", []), ("comparator_store", None),
            ("author", 7),
        ],
    )
    def test_rejects_fields_that_crash_the_maths(self, field: str, value: Any) -> None:
        """`location` reaches `.strip()` and `score` reaches a numeric
        comparison, so a bad value there was a crash at the trust boundary
        rather than a rejected row."""
        assert not aggregate._is_valid({**claim(), field: value})

    @pytest.mark.parametrize("field", ["location", "item", "comparator_store", "author"])
    def test_absent_optional_fields_are_fine(self, field: str) -> None:
        row = dict(claim())
        row.pop(field)
        assert aggregate._is_valid(row)

    def test_a_null_score_is_allowed(self) -> None:
        assert aggregate._is_valid({**claim(), "score": None})

    def test_rejects_bool_timestamp(self) -> None:
        assert not aggregate._is_valid({**claim(), "created_utc": True})


class TestReadClaims:
    def test_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "c.jsonl"
        p.write_text(json.dumps(claim()) + "\n\n", encoding="utf-8")
        claims, dropped = aggregate.read_claims(p)
        assert len(claims) == 1 and dropped == 0

    def test_truncated_line_does_not_lose_good_rows(self, tmp_path: Path) -> None:
        p = tmp_path / "c.jsonl"
        p.write_text(json.dumps(claim()) + "\n{trunc", encoding="utf-8")
        claims, dropped = aggregate.read_claims(p)
        assert len(claims) == 1 and dropped == 1

    def test_invalid_rows_are_dropped_and_counted(self, tmp_path: Path) -> None:
        p = tmp_path / "c.jsonl"
        bad = dict(claim())
        bad.pop("store")
        p.write_text(
            f"{json.dumps(claim())}\n{json.dumps(bad)}\n", encoding="utf-8"
        )
        claims, dropped = aggregate.read_claims(p)
        assert len(claims) == 1 and dropped == 1

    def test_rows_missing_optional_text_fields_survive(self, tmp_path: Path) -> None:
        # A claims file from an older pipeline version has no `comparator_store`;
        # scrubbing must skip the absent field rather than reject the row.
        thin = dict(claim())
        for f in ("location", "item", "comparator_store"):
            thin.pop(f)
        (tmp_path / "c.jsonl").write_text(json.dumps(thin), encoding="utf-8")
        claims, dropped = aggregate.read_claims(tmp_path / "c.jsonl")
        assert len(claims) == 1 and dropped == 0

    def test_scrubs_on_the_read_path_too(self, tmp_path: Path) -> None:
        p = tmp_path / "c.jsonl"
        p.write_text(
            json.dumps(claim(claim="cheap\x1b[2J", location="Som​e")),
            encoding="utf-8",
        )
        claims, _ = aggregate.read_claims(p)
        assert "\x1b" not in claims[0]["claim"]
        assert claims[0]["location"] == "Some"


class TestDedupe:
    def test_identical_claims_collapse(self) -> None:
        assert len(aggregate.dedupe([claim(), claim()])) == 1

    def test_distinct_categories_are_kept(self) -> None:
        assert len(aggregate.dedupe([claim(), claim(category="meat")])) == 2

    def test_post_and_comment_sharing_an_id_stay_distinct(self) -> None:
        pair = [claim(source_key="c_boston_abc"), claim(source_key="p_boston_abc")]
        assert len(aggregate.dedupe(pair)) == 2


class TestReciprocals:
    def test_both_halves_of_one_comparison_are_marked(self) -> None:
        pair = [
            claim(store="Market Basket", comparator_store="Shaw's"),
            claim(store="Shaw's", comparator_store="Market Basket", category="meat"),
        ]
        out = aggregate.collapse_reciprocals(pair)
        assert all(c.get("_reciprocal") for c in out)

    def test_a_comparison_to_an_absent_store_is_not_marked(self) -> None:
        out = aggregate.collapse_reciprocals([claim(comparator_store="Wegmans")])
        assert not out[0].get("_reciprocal")

    def test_non_comparative_claims_untouched(self) -> None:
        assert not aggregate.collapse_reciprocals([claim()])[0].get("_reciprocal")

    def test_reciprocals_are_half_weighted(self) -> None:
        full, _ = aggregate.build([claim()], NOW)
        marked = dict(claim())
        marked["_reciprocal"] = True
        half, _ = aggregate.build([marked], NOW)  # type: ignore[list-item]
        assert next(iter(half.values())).weight == pytest.approx(
            next(iter(full.values())).weight / 2
        )


class TestIndependenceCaps:
    def test_one_document_cannot_dominate(self) -> None:
        many = [claim(category=f"c{i}") for i in range(10)]
        assert len(aggregate.limit_per_source(many)) == aggregate.MAX_CLAIMS_PER_DOCUMENT

    def test_one_author_is_capped_per_cell(self) -> None:
        many = [
            claim(author="regular", source_key=f"c_boston_{i}", claim=f"c{i}")
            for i in range(6)
        ]
        assert (
            len(aggregate.limit_per_source(many))
            == aggregate.MAX_CLAIMS_PER_AUTHOR_CELL
        )

    def test_the_author_cap_is_per_cell_not_global(self) -> None:
        many = [
            claim(author="regular", source_key=f"c_boston_{i}", category=cat)
            for i, cat in enumerate(("produce", "meat", "dairy", "bakery"))
        ]
        assert len(aggregate.limit_per_source(many)) == 4

    @pytest.mark.parametrize("author", ["", "[deleted]", "[removed]"])
    def test_anonymous_authors_are_not_capped_together(self, author: str) -> None:
        many = [claim(author=author, source_key=f"c_boston_{i}") for i in range(5)]
        assert len(aggregate.limit_per_source(many)) == 5

    def test_automoderator_is_capped_like_anyone_else(self) -> None:
        # It is exactly one prolific poster, which is what the cap is for.
        many = [
            claim(author="AutoModerator", source_key=f"c_boston_{i}")
            for i in range(5)
        ]
        assert (
            len(aggregate.limit_per_source(many))
            == aggregate.MAX_CLAIMS_PER_AUTHOR_CELL
        )

    def test_higher_confidence_survives_the_cap(self) -> None:
        mixed = [claim(confidence="low", claim=f"lo{i}") for i in range(3)] + [
            claim(confidence="high", claim="hi")
        ]
        assert "hi" in [c["claim"] for c in aggregate.limit_per_source(mixed)]


class TestPrepare:
    def test_transient_claims_are_dropped_before_the_caps(self) -> None:
        """Filtering them inside `build`, after the caps, let three
        closing-down-sale claims fill a document's cap and starve the durable
        claim in the same comment."""
        one_doc = [
            claim(claim="sale1", transient=True, category="deals_loyalty"),
            claim(claim="sale2", transient=True, category="price_overall"),
            claim(claim="sale3", transient=True, category="quality_overall"),
            claim(claim="durable", category="produce"),
        ]
        kept = [c["claim"] for c in aggregate.prepare(one_doc)]
        assert "durable" in kept
        assert not any(k.startswith("sale") for k in kept)

    def test_transient_can_be_kept_explicitly(self) -> None:
        assert len(aggregate.prepare([claim(transient=True)], False)) == 1


class TestBuild:
    def test_transient_claims_are_excluded(self) -> None:
        assert aggregate.build([claim(transient=True)], NOW)[0] == {}

    def test_transient_can_be_included_explicitly(self) -> None:
        cells, _ = aggregate.build([claim(transient=True)], NOW, exclude_transient=False)
        assert len(cells) == 1

    def test_cells_are_keyed_by_branch(self) -> None:
        cells, _ = aggregate.build(
            [claim(location="Somerville"), claim(location="Chelsea", claim="x")], NOW
        )
        assert {k[1] for k in cells} == {"somerville", "chelsea"}

    def test_spelling_variants_are_one_branch(self) -> None:
        """Unconstrained model prose used as a dict key fragments a branch
        into several, each individually too thin to clear the threshold."""
        variants = ["Somerville", "somerville", " Somerville ", "Somerville, MA",
                    "the Somerville one", "Somerville  store"]
        cells, _ = aggregate.build(
            [claim(location=v, claim=f"c{i}") for i, v in enumerate(variants)], NOW
        )
        assert len(cells) == 1
        assert next(iter(cells)).__getitem__(1) == "somerville"
        assert next(iter(cells.values())).n == 6

    def test_street_type_variants_are_one_branch(self) -> None:
        """Measured on the real corpus: 6% of branch keys differed only by
        the street-type suffix, each half too thin to clear the threshold."""
        variants = ["Beacon", "Beacon St", "Beacon Street", "beacon st."]
        cells, _ = aggregate.build(
            [claim(location=v, claim=f"c{i}") for i, v in enumerate(variants[:3])], NOW
        )
        assert len(cells) == 1
        assert aggregate.branch_key("Porter Square") == aggregate.branch_key("Porter")

    def test_a_town_qualifier_still_separates_two_branches(self) -> None:
        # There is a Star Market on Beacon St in Somerville and another on
        # Beacon Street in Washington Square. Merging them is worse than not.
        assert aggregate.branch_key("Beacon St, Somerville") != aggregate.branch_key(
            "Beacon Street, Washington Square"
        )
        assert aggregate.branch_key("Beacon St, Somerville") == aggregate.branch_key(
            "beacon street, somerville"
        )

    @pytest.mark.parametrize("junk", ["labor_ethics", "produce", "Market Basket"])
    def test_a_location_that_is_not_a_place_is_dropped(self, junk: str) -> None:
        """The model occasionally puts the category or the store's own name in
        `location`. Rare, but each one becomes a visible branch heading."""
        cells, _ = aggregate.build([claim(location=junk)], NOW)
        assert next(iter(cells))[1] == "", "junk location became a branch"
        assert next(iter(cells.values())).n == 1, "the claim itself must survive"

    def test_the_non_location_list_tracks_the_stage2_vocabulary(self) -> None:
        from groceries.extract import CATEGORIES

        for category in CATEGORIES:
            assert aggregate.branch_key(category) == ""

    def test_the_readable_spelling_survives_normalisation(self) -> None:
        cells, _ = aggregate.build(
            [claim(location="Somerville"), claim(location="somerville", claim="x")],
            NOW,
        )
        assert next(iter(cells.values())).label() == "Somerville"

    def test_item_variants_are_one_entry(self) -> None:
        _, items = aggregate.build(
            [claim(item="Key Limes", category="specific_item"),
             claim(item="key limes", category="specific_item", claim="x")],
            NOW,
        )
        assert list(items) == ["Market Basket|key limes"]
        assert items["Market Basket|key limes"].label() == "Key Limes"

    def test_items_are_indexed_separately(self) -> None:
        _, items = aggregate.build(
            [claim(item="Key limes", category="specific_item")], NOW
        )
        assert "Market Basket|key limes" in items

    def test_claims_without_an_item_are_not_indexed(self) -> None:
        assert aggregate.build([claim(item="  ")], NOW)[1] == {}


class TestChainRollup:
    def test_branches_fold_into_the_chain(self) -> None:
        cells, _ = aggregate.build(
            [claim(location="Somerville"), claim(location="Chelsea", claim="x")], NOW
        )
        chain = aggregate.chain_rollup(cells)
        assert list(chain) == [("Market Basket", "produce")]
        assert chain[("Market Basket", "produce")].n == 2

    def test_price_distribution_survives_the_rollup(self) -> None:
        cells, _ = aggregate.build(
            [
                claim(location="A", price_signal="cheap"),
                claim(location="B", price_signal="expensive", claim="x"),
            ],
            NOW,
        )
        chain = aggregate.chain_rollup(cells)
        assert chain[("Market Basket", "produce")].price_counts == {
            "cheap": 1,
            "expensive": 1,
        }


class TestHeadlineHalfLife:
    """Anything compared against the headline number must age at its rate."""

    def test_mixes_the_category_half_lives_by_evidence(self) -> None:
        cells, _ = aggregate.build(
            [claim(category="price_overall"),          # 7y
             claim(category="crowding_hours", claim="x")],  # 2y
            NOW,
        )
        got = aggregate.headline_half_life(aggregate.chain_rollup(cells))
        assert 2.0 < got < 7.0

    def test_evidence_moves_the_mix(self) -> None:
        heavy_slow = [claim(category="price_overall", source_key=f"c_b_{i}",
                            author=f"a{i}") for i in range(10)]
        one_fast = [claim(category="crowding_hours", claim="x")]
        cells, _ = aggregate.build(heavy_slow + one_fast, NOW)
        got = aggregate.headline_half_life(aggregate.chain_rollup(cells))
        assert got > 6.0, "ten price claims should dominate one crowding claim"

    def test_company_claims_do_not_set_the_rate(self) -> None:
        cells, _ = aggregate.build(
            [claim(category="store_lifecycle")], NOW      # 1y, non-shopping
        )
        got = aggregate.headline_half_life(aggregate.chain_rollup(cells))
        assert got == aggregate.DEFAULT_HALF_LIFE_YEARS

    def test_no_evidence_falls_back_to_the_default(self) -> None:
        assert aggregate.headline_half_life({}) == aggregate.DEFAULT_HALF_LIFE_YEARS

    def test_it_reaches_the_verdict_document(self) -> None:
        out = aggregate.aggregate([claim()], now=NOW, min_weight=0.0)
        assert out["headline_half_life_years"] == aggregate.DEFAULT_HALF_LIFE_YEARS


class TestBranchTotals:
    def test_rolls_a_branch_up_across_categories(self) -> None:
        cells, _ = aggregate.build(
            [claim(location="Somerville"),
             claim(location="Somerville", category="meat", claim="x")], NOW
        )
        got = aggregate.branch_totals_from(cells)
        assert got[("Market Basket", "somerville")].n == 2

    def test_unlocated_claims_are_not_a_branch(self) -> None:
        cells, _ = aggregate.build([claim(location="")], NOW)
        assert aggregate.branch_totals_from(cells) == {}

    def test_company_claims_are_excluded_like_the_chain_rollup(self) -> None:
        cells, _ = aggregate.build(
            [claim(location="Somerville", category="labor_ethics"),
             claim(location="Somerville", category="produce", claim="x")], NOW
        )
        assert aggregate.branch_totals_from(cells)[("Market Basket", "somerville")].n == 1
        assert aggregate.branch_totals_from(cells, shopping_only=False)[
            ("Market Basket", "somerville")].n == 2

    def test_reaches_the_verdict_document_with_a_readable_name(self) -> None:
        out = aggregate.aggregate(
            [claim(location="Somerville")], now=NOW, min_weight=0.0
        )
        assert out["branch_totals"]["Market Basket"]["Somerville"]["n_claims"] == 1


class TestTotalsFrom:
    def test_company_claims_are_excluded_from_the_headline(self) -> None:
        cells, _ = aggregate.build(
            [claim(category="labor_ethics"), claim(category="produce", claim="x")], NOW
        )
        chain = aggregate.chain_rollup(cells)
        assert aggregate.totals_from(chain)["Market Basket"].n == 1
        assert aggregate.totals_from(chain, shopping_only=False)["Market Basket"].n == 2


class TestAggregate:
    def test_produces_all_four_views(self) -> None:
        out = aggregate.aggregate(
            [
                claim(location="Somerville"),
                claim(item="key limes", category="specific_item", claim="limes"),
            ],
            now=NOW,
            min_weight=0.0,
        )
        assert "Market Basket" in out["stores"]
        assert "Somerville" in out["branches"]["Market Basket"]
        assert "key limes" in out["items"]["Market Basket"]
        assert out["store_totals"]["Market Basket"]["n_claims"] == 2

    def test_evidence_free_stores_are_labelled(self) -> None:
        out = aggregate.aggregate([claim(confidence="low")], now=NOW, min_weight=99.0)
        assert out["stores"] == {}
        assert out["store_totals"]["Market Basket"]["insufficient_evidence"] is True

    def test_a_store_with_evidence_is_not_labelled(self) -> None:
        out = aggregate.aggregate([claim()], now=NOW, min_weight=0.0)
        assert out["store_totals"]["Market Basket"]["insufficient_evidence"] is False

    def test_method_block_records_the_parameters(self) -> None:
        out = aggregate.aggregate([claim()], now=NOW)
        assert out["method"]["shrinkage_k"] == aggregate.SHRINKAGE_K
        assert out["method"]["transient_claims"] == "excluded"

    def test_corpus_provenance_is_recorded(self) -> None:
        out = aggregate.aggregate([claim()], now=NOW, corpus={"docs": 20332})
        assert out["corpus"] == {"docs": 20332}

    def test_corpus_absent_when_not_supplied(self) -> None:
        assert aggregate.aggregate([claim()], now=NOW)["corpus"] is None

    def test_empty_input(self) -> None:
        out = aggregate.aggregate([], now=NOW)
        assert out["stores"] == {} and out["store_totals"] == {}

    def test_defaults_to_wall_clock(self) -> None:
        assert aggregate.aggregate([claim()])["generated_at"].endswith("Z")

    def test_evidence_carries_branch_and_link(self) -> None:
        out = aggregate.aggregate(
            [claim(location="Somerville")], now=NOW, min_weight=0.0
        )
        ev = out["stores"]["Market Basket"]["produce"]["evidence"][0]
        assert ev["location"] == "Somerville"
        assert ev["permalink"].startswith("/r/boston/")
        assert ev["date"] == "2026-02"

    def test_unlocated_claims_do_not_create_branches(self) -> None:
        assert aggregate.aggregate([claim()], now=NOW, min_weight=0.0)["branches"] == {}

    def test_thin_items_are_suppressed(self) -> None:
        out = aggregate.aggregate(
            [claim(item="x", category="specific_item", confidence="low")],
            now=NOW,
            min_weight=99.0,
        )
        assert out["items"] == {}

    def test_max_examples_is_respected(self) -> None:
        claims = [
            claim(source_key=f"c_boston_{i}", author=f"a{i}", claim=f"c{i}")
            for i in range(8)
        ]
        out = aggregate.aggregate(claims, now=NOW, min_weight=0.0, max_examples=2)
        assert len(out["stores"]["Market Basket"]["produce"]["evidence"]) == 2


class TestMonth:
    def test_formats_year_month(self) -> None:
        assert aggregate.month(NOW) == "2026-02"


class TestIO:
    def test_write_creates_parents(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "v.json"
        aggregate.write_verdicts(aggregate.aggregate([claim()], now=NOW), out)
        assert json.loads(out.read_text())["method"]["shrinkage_k"] == 2.0


class TestFormatting:
    def test_totals_table_marks_thin_stores(self) -> None:
        out = aggregate.aggregate([claim(confidence="low")], now=NOW, min_weight=99.0)
        assert "insufficient evidence" in aggregate.format_totals(out)

    def test_totals_table_lists_stores(self) -> None:
        out = aggregate.aggregate(
            [claim(), claim(store="Aldi", claim="x")], now=NOW, min_weight=0.0
        )
        text = aggregate.format_totals(out)
        assert "Market Basket" in text and "Aldi" in text

    def test_store_readout_includes_branches_and_items(self) -> None:
        out = aggregate.aggregate(
            [
                claim(location="Somerville"),
                claim(item="key limes", category="specific_item", claim="limes"),
            ],
            now=NOW,
            min_weight=0.0,
        )
        text = aggregate.format_store(out, "Market Basket")
        assert "Somerville" in text and "key limes" in text

    def test_store_readout_for_unknown_store(self) -> None:
        out = aggregate.aggregate([claim()], now=NOW)
        assert "no claims above" in aggregate.format_store(out, "Nonexistent")

    @pytest.mark.parametrize(
        ("sentiment", "mark"), [("positive", "+"), ("negative", "-"), ("neutral", "~")]
    )
    def test_sentiment_marks(self, sentiment: str, mark: str) -> None:
        claims = [
            claim(sentiment=sentiment, source_key=f"c_b_{i}", author=f"a{i}")
            for i in range(20)
        ]
        out = aggregate.aggregate(claims, now=NOW, min_weight=0.0)
        line = next(
            line
            for line in aggregate.format_store(out, "Market Basket").splitlines()
            if "produce" in line
        )
        assert f" {mark} sentiment" in line

    def test_store_with_no_branch_or_item_data(self) -> None:
        text = aggregate.format_store(
            aggregate.aggregate([claim()], now=NOW, min_weight=0.0), "Market Basket"
        )
        assert "branches with their own evidence" not in text
        assert "items:" not in text
