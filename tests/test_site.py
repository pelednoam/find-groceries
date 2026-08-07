"""Tests for the published-site payload builder.

The payload is the only part of this pipeline a stranger can read, so the
things that matter here are: it says what the verdicts say, it does not
quietly drop evidence, and it carries nothing that should stay private.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from groceries import site
from groceries.extract import CATEGORIES


def cell(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "n_claims": 12,
        "weighted_evidence": 8.25,
        "sentiment": 0.6123,
        "price_level": -0.42,
        "price_signal": "cheap",
        "price_distribution": {"cheap": 9, "expensive": 1},
        "evidence": [
            {"claim": f"claim {i}", "date": "2025-0{}".format(i + 1),
             "permalink": f"/r/boston/comments/a/_/c{i}/", "confidence": "high",
             "location": "Somerville" if i == 0 else ""}
            for i in range(5)
        ],
    }
    base.update(over)
    return base


def verdicts(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "generated_at": "2026-08-07T00:00:00Z",
        "method": {"shrinkage_k": 2.0, "default_half_life_years": 4.0,
                   "transient_claims": "excluded"},
        "corpus": {"documents_extracted": 24958, "working_set": 25108},
        "stores": {"Market Basket": {"produce": cell(), "meat": cell()}},
        "branches": {"Market Basket": {"Somerville": {"produce": cell()},
                                       "NH": {"produce": cell()}}},
        "items": {"Market Basket": {"donuts": cell(),
                                    "thin thing": cell(weighted_evidence=0.2)}},
        "store_totals": {"Market Basket": {"n_claims": 40, "weighted_evidence": 22.5,
                                           "sentiment": 0.5,
                                           "insufficient_evidence": False}},
    }
    base.update(over)
    return base


class TestSlimCell:
    def test_keeps_the_numbers_the_ui_shows(self) -> None:
        got = site.slim_cell(cell())
        assert got["n"] == 12 and got["w"] == 8.2 and got["s"] == 0.612
        assert got["p"] == "cheap" and got["pl"] == -0.42

    def test_caps_evidence(self) -> None:
        assert len(site.slim_cell(cell())["e"]) == site.SITE_EXAMPLES
        assert len(site.slim_cell(cell(), examples=1)["e"]) == 1

    def test_evidence_keeps_its_provenance(self) -> None:
        e = site.slim_cell(cell())["e"][0]
        assert e["t"] == "claim 0" and e["u"].startswith("/r/boston/")
        assert e["c"] == "high" and e["l"] == "Somerville"

    def test_a_blank_location_is_omitted_not_empty(self) -> None:
        # 7,390 of 28,225 claims have a location; carrying "" for the rest is
        # ~340KB of nothing.
        assert "l" not in site.slim_cell(cell())["e"][1]

    def test_a_cell_with_no_price_says_nothing_about_price(self) -> None:
        got = site.slim_cell(cell(price_signal=None, price_level=None,
                                  price_distribution={}))
        assert "p" not in got and "pl" not in got and "pd" not in got


class TestRegions:
    @pytest.mark.parametrize("name", ["Somerville", "Central Square", "Chelsea",
                                      "Beacon St, Somerville"])
    def test_real_branches(self, name: str) -> None:
        assert site.is_branch(name)

    @pytest.mark.parametrize("name", ["NH", "New Hampshire", "Southern NH",
                                      "the suburbs", "inside 128", "North Shore",
                                      "Cape Cod"])
    def test_regions_are_not_branches(self, name: str) -> None:
        """"NH" and "Somerville" are both answers to "where", but only one is
        a store, and listing them together implies a precision "NH" lacks."""
        assert not site.is_branch(name)

    def test_regions_are_kept_but_separated(self) -> None:
        payload = site.build_payload(verdicts())
        assert "Somerville" in payload["branches"]["Market Basket"]
        assert "NH" in payload["regions"]["Market Basket"]
        assert "NH" not in payload["branches"]["Market Basket"]


class TestPayload:
    def test_has_every_view_the_ui_needs(self) -> None:
        payload = site.build_payload(verdicts())
        for key in ("generated_at", "method", "corpus", "totals", "stores",
                    "branches", "regions", "items", "keywords", "categories"):
            assert key in payload, key

    def test_thin_items_are_dropped(self) -> None:
        items = site.build_payload(verdicts())["items"]["Market Basket"]
        assert "donuts" in items and "thin thing" not in items

    def test_totals_carry_the_thin_flag(self) -> None:
        payload = site.build_payload(verdicts())
        assert payload["totals"]["Market Basket"]["thin"] is False

    def test_categories_are_derived_from_the_data(self) -> None:
        assert site.build_payload(verdicts())["categories"] == ["meat", "produce"]

    def test_corpus_provenance_survives(self) -> None:
        """Without it the site cannot say whether it is showing the full
        corpus or a 150-document sample."""
        assert site.build_payload(verdicts())["corpus"]["documents_extracted"] == 24958

    def test_no_author_names_anywhere(self) -> None:
        """Reddit usernames are public, but the UI has no use for them and
        not republishing them is free."""
        blob = json.dumps(site.build_payload(verdicts()))
        assert "author" not in blob

    def test_an_empty_verdict_file_produces_an_empty_payload(self) -> None:
        empty = verdicts(stores={}, branches={}, items={}, store_totals={})
        payload = site.build_payload(empty)
        assert payload["stores"] == {} and payload["categories"] == []


class TestKeywords:
    def test_every_mapped_category_exists_in_stage_two(self) -> None:
        """A keyword pointing at a category the pipeline never emits sends a
        shopping-list term to a cell that can never exist."""
        for category in site.LIST_CATEGORIES:
            assert category in CATEGORIES, category

    def test_the_index_is_inverted_not_duplicated(self) -> None:
        index = site.keyword_index()
        assert index["milk"] == "dairy"
        assert index["rotisserie"] == "prepared_food"
        assert index["seltzer"] == "alcohol"

    def test_no_word_maps_to_two_categories(self) -> None:
        seen: dict[str, str] = {}
        for category, words in site.LIST_CATEGORIES.items():
            for word in words:
                assert word not in seen, f"{word}: {seen.get(word)} vs {category}"
                seen[word] = category

    def test_an_ordinary_shopping_list_is_fully_covered(self) -> None:
        index = site.keyword_index()
        everyday = ["milk", "eggs", "bread", "chicken", "coffee", "rice",
                    "cheese", "bananas", "pasta", "beer", "salmon", "ice cream"]
        missing = [w for w in everyday if w not in index]
        assert not missing, f"no category for {missing}"


class TestWrite:
    def test_writes_compact_json(self, tmp_path: Path) -> None:
        out = tmp_path / "verdicts.json"
        n = site.write_payload(site.build_payload(verdicts()), out)
        assert n == out.stat().st_size
        text = out.read_text(encoding="utf-8")
        assert ", " not in text[:200], "should be separator-compact"
        assert json.loads(text)["stores"]

    def test_non_ascii_survives_the_round_trip(self, tmp_path: Path) -> None:
        v = verdicts()
        v["stores"]["Market Basket"]["produce"]["evidence"][0]["claim"] = "café — 5€"
        out = tmp_path / "v.json"
        site.write_payload(site.build_payload(v), out)
        assert json.loads(out.read_text(encoding="utf-8"))[
            "stores"]["Market Basket"]["produce"]["e"][0]["t"] == "café — 5€"


class TestPublishedPayload:
    """Guards on the file that is actually served, when it has been built."""

    @pytest.fixture
    def payload(self) -> Any:
        path = Path(__file__).resolve().parent.parent / "docs" / "verdicts.json"
        if not path.exists():
            pytest.skip("docs/verdicts.json not built")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_carries_no_contact_details(self, payload: Any) -> None:
        assert "findneuro" not in json.dumps(payload)

    def test_every_permalink_is_a_reddit_path(self, payload: Any) -> None:
        """The UI turns these into hrefs; anything else is a link-injection."""
        for cats in payload["stores"].values():
            for c in cats.values():
                for e in c["e"]:
                    assert e["u"].startswith("/r/"), e["u"]

    def test_is_small_enough_to_serve(self, payload: Any) -> None:
        path = Path(__file__).resolve().parent.parent / "docs" / "verdicts.json"
        assert path.stat().st_size < 4_000_000
