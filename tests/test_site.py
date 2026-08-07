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


def locations(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "attribution": "© OpenStreetMap contributors (ODbL)",
        "places": [
            {"store": "Market Basket", "name": "Market Basket", "lat": 42.3875,
             "lon": -71.0995, "address": "400 Somerville Ave", "city": "Somerville",
             "osm": "node/1"},
            {"store": "Market Basket", "name": "Market Basket", "lat": 42.4710,
             "lon": -71.2085, "address": "43 Middlesex Turnpike", "city": "Burlington",
             "osm": "node/2"},
            {"store": "Nowhere Grocers", "name": "Nowhere", "lat": 42.0, "lon": -71.0,
             "address": "", "city": "", "osm": "node/3"},
        ],
    }
    base.update(over)
    return base


class TestLocationsInPayload:
    def test_places_are_carried_through(self) -> None:
        payload = site.build_payload(verdicts(), locations())
        assert [p["osm"] for p in payload["places"]] == ["node/1", "node/2"]

    def test_a_store_nobody_discusses_gets_no_pin(self) -> None:
        """A pin with no evidence behind it is a pin the map cannot explain."""
        payload = site.build_payload(verdicts(), locations())
        assert all(p["store"] in payload["stores"] for p in payload["places"])

    def test_a_pin_is_linked_to_a_branch_that_has_evidence(self) -> None:
        payload = site.build_payload(verdicts(), locations())
        by_osm = {p["osm"]: p for p in payload["places"]}
        assert by_osm["node/1"]["branch"] == "Somerville"

    def test_a_pin_with_no_matching_branch_is_left_unlinked(self) -> None:
        # Burlington has no branch-level evidence in the fixture, so the pin
        # falls back to the chain rather than borrowing another branch's score.
        payload = site.build_payload(verdicts(), locations())
        by_osm = {p["osm"]: p for p in payload["places"]}
        assert "branch" not in by_osm["node/2"]

    def test_attribution_survives(self) -> None:
        payload = site.build_payload(verdicts(), locations())
        assert "OpenStreetMap" in payload["places_attribution"]

    def test_no_locations_means_no_map_not_no_site(self) -> None:
        payload = site.build_payload(verdicts(), None)
        assert payload["places"] == [] and payload["places_attribution"] == ""
        assert payload["stores"], "the rest of the site must still build"


class TestCrossCheckInPayload:
    def test_carried_whole_and_kept_separate(self) -> None:
        block = {"source": "Google Maps reviews", "stores": {"Market Basket": {"n": 9}}}
        payload = site.build_payload(verdicts(), None, block)
        assert payload["crosscheck"] == block
        # and nothing of it reached the verdict
        assert payload["totals"]["Market Basket"]["s"] == 0.5

    def test_absent_by_default(self) -> None:
        assert site.build_payload(verdicts())["crosscheck"] is None


class TestMergeInPayload:
    def _cc(self) -> dict[str, Any]:
        # Enough stores for the calibration to have degrees of freedom.
        return {"stores": {
            f"S{i}": {"n": 500, "norm": 0.5 + 0.02 * i, "thin": False}
            for i in range(8)
        }, "locations": {}}

    def _verdicts(self) -> dict[str, Any]:
        v = verdicts()
        v["store_totals"] = {
            f"S{i}": {"n_claims": 100, "weighted_evidence": 40.0,
                      "sentiment": -0.5 + 0.15 * i, "insufficient_evidence": False}
            for i in range(8)
        }
        v["stores"] = {f"S{i}": {"produce": cell()} for i in range(8)}
        v["branches"] = {}
        v["branch_totals"] = {}
        v["items"] = {}
        return v

    def test_absent_without_a_crosscheck(self) -> None:
        assert site.build_payload(verdicts())["merged"] is None

    def test_produces_a_calibration_and_per_store_values(self) -> None:
        payload = site.build_payload(self._verdicts(), None, self._cc())
        merged = payload["merged"]
        assert merged is not None
        assert merged["calibration"]["n_stores"] == 8
        assert len(merged["stores"]) == 8

    def test_the_per_category_cells_are_left_alone(self) -> None:
        """Google has one number per shop and no opinion about produce.
        The merge must not touch a category cell."""
        v = self._verdicts()
        before = site.build_payload(v)["stores"]
        after = site.build_payload(v, None, self._cc())["stores"]
        assert before == after

    def test_the_reddit_totals_are_left_alone(self) -> None:
        payload = site.build_payload(self._verdicts(), None, self._cc())
        assert payload["totals"]["S0"]["s"] == -0.5

    def test_a_malformed_rating_degrades_rather_than_raising(self) -> None:
        """The cross-check is a separately generated file; one written by an
        older version must mean "no Google here", not a build failure."""
        cc = self._cc()
        cc["stores"]["S9"] = {"n": 5}          # no norm, no thin
        payload = site.build_payload(self._verdicts(), None, cc)
        assert payload["merged"] is not None

    def _with_branches(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        v = self._verdicts()
        v["branches"] = {"S0": {"Somerville": {"produce": cell()}}}
        v["branch_totals"] = {"S0": {"Somerville": {
            "n_claims": 2, "weighted_evidence": 0.6, "sentiment": -0.4}}}
        cc = self._cc()
        cc["locations"] = {"node/1": {"n": 400, "norm": 0.8},
                           "node/2": {"n": 200, "norm": 0.6}}
        loc = {"attribution": "osm", "places": [
            {"store": "S0", "name": "S0", "lat": 42.3875, "lon": -71.0995,
             "address": "1 A St", "city": "Somerville", "osm": "node/1"},
            {"store": "S0", "name": "S0", "lat": 42.3876, "lon": -71.0996,
             "address": "2 B St", "city": "Somerville", "osm": "node/2"},
        ]}
        return v, cc, loc

    def test_branches_are_merged_and_pins_pooled(self) -> None:
        """Two shops serving one branch name pool by rating count rather than
        one of them being picked."""
        v, cc, loc = self._with_branches()
        merged = site.build_payload(v, loc, cc)["merged"]
        b = merged["branches"]["S0"]["Somerville"]
        assert b["r"] == -0.4 and b["g"] is not None
        # pooled norm = (400*0.8 + 200*0.6)/600 = 0.733
        assert b["share"] < 0.5, "a 0.6-weight branch should not outvote 600 ratings"

    def test_a_branch_with_no_reddit_claims_still_gets_an_answer(self) -> None:
        v, cc, loc = self._with_branches()
        v["branch_totals"] = {}
        for p in loc["places"]:
            p["city"] = "Somerville"
        merged = site.build_payload(v, loc, cc)["merged"]
        b = merged["branches"]["S0"]["Somerville"]
        assert "r" not in b and b["share"] == 0.0

    def test_a_pin_with_no_rating_contributes_nothing(self) -> None:
        v, cc, loc = self._with_branches()
        cc["locations"] = {}
        merged = site.build_payload(v, loc, cc)["merged"]
        assert merged["branches"]["S0"]["Somerville"]["share"] == 1.0

    def test_the_merge_note_states_its_own_scope(self) -> None:
        v, cc, loc = self._with_branches()
        assert "category" in site.build_payload(v, loc, cc)["merged"]["note"]

    def test_too_few_stores_to_calibrate_yields_no_merge(self) -> None:
        cc = {"stores": {"S0": {"n": 500, "norm": 0.6, "thin": False}},
              "locations": {}}
        assert site.build_payload(self._verdicts(), None, cc)["merged"] is None


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


class TestPayloadContract:
    """The TypeScript `Payload` interface must describe what Python emits.

    These are two hand-written descriptions of one wire format in two
    languages, and nothing else connects them. A field renamed on the Python
    side would otherwise surface as a blank panel in the browser rather than
    as an error anywhere — tsc cannot see the Python, and mypy cannot see the
    TypeScript.
    """

    @pytest.fixture
    def declared(self) -> set[str]:
        src = (Path(__file__).resolve().parent.parent
               / "docs" / "src" / "types.ts").read_text(encoding="utf-8")
        body = src.split("interface Payload {", 1)[1].split("\n}", 1)[0]
        fields = set()
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith(("/", "*")):
                continue
            fields.add(line.split(":", 1)[0].rstrip("?"))
        return fields

    def test_typescript_declares_every_python_field(self, declared: set[str]) -> None:
        emitted = set(site.build_payload(verdicts(), locations()))
        assert emitted - declared == set(), "Python emits fields TypeScript does not declare"

    def test_python_emits_every_typescript_field(self, declared: set[str]) -> None:
        emitted = set(site.build_payload(verdicts(), locations()))
        assert declared - emitted == set(), "TypeScript declares fields Python does not emit"

    def _interface(self, name: str) -> set[str]:
        src = (Path(__file__).resolve().parent.parent
               / "docs" / "src" / "types.ts").read_text(encoding="utf-8")
        body = src.split(f"interface {name} {{", 1)[1].split("\n}", 1)[0]
        return {
            line.strip().split(":", 1)[0].rstrip("?")
            for line in body.splitlines()
            if line.strip() and not line.strip().startswith(("/", "*"))
        }

    def test_the_rating_shape_matches_too(self) -> None:
        """Rating gained n_eff/mean_recent/norm_recent when decay landed; the
        UI reads them, so TypeScript has to know they exist."""
        from groceries.crosscheck import summarise

        emitted = summarise([{"rating": 4, "time": 1_600_000_000_000,
                              "text_len": 200}], 1_785_000_000, 4.72)
        assert emitted is not None
        assert set(emitted) == self._interface("Rating")

    def test_the_place_shape_matches_too(self, declared: set[str]) -> None:
        src = (Path(__file__).resolve().parent.parent
               / "docs" / "src" / "types.ts").read_text(encoding="utf-8")
        body = src.split("interface Place {", 1)[1].split("\n}", 1)[0]
        ts_fields = {
            line.strip().split(":", 1)[0].rstrip("?")
            for line in body.splitlines()
            if line.strip() and not line.strip().startswith(("/", "*"))
        }
        built = site.build_payload(verdicts(), locations())["places"]
        assert built, "fixture should produce at least one place"
        for place in built:
            assert set(place) <= ts_fields, set(place) - ts_fields


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
