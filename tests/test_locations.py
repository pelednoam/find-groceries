"""Tests for the OpenStreetMap location step.

Pins go on a public map next to real businesses, so the bar here is: never
place a store somewhere it is not, and never attach a verdict to a location
it was not about. Both failure modes are silent — a wrong pin looks exactly
like a right one — so they get tests rather than care.
"""

from __future__ import annotations

from typing import Any

import pytest

from groceries import locations
from groceries.select import STORE_PATTERNS


def node(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": "node",
        "id": 1,
        "lat": 42.3875,
        "lon": -71.0995,
        "tags": {
            "name": "Market Basket",
            "shop": "supermarket",
            "addr:housenumber": "400",
            "addr:street": "Somerville Ave",
            "addr:city": "Somerville",
        },
    }
    base.update(over)
    return base


class TestMatchStore:
    @pytest.mark.parametrize(
        ("osm_name", "expected"),
        [
            ("Market Basket", "Market Basket"),
            ("Whole Foods Market", "Whole Foods"),
            ("Trader Joe's", "Trader Joe's"),
            ("Stop & Shop", "Stop & Shop"),
            ("BJ's Wholesale Club", "BJ's"),
            ("H Mart", "H Mart"),
            ("Shaw's", "Shaw's"),
        ],
    )
    def test_matches_the_chains_the_corpus_talks_about(
        self, osm_name: str, expected: str
    ) -> None:
        assert locations.match_store(osm_name) == expected

    @pytest.mark.parametrize(
        "osm_name",
        ["7-Eleven", "Speedway", "Roche Bros.", "Hudson", "Mobil Mart", "Star Bakery"],
    )
    def test_unknown_shops_are_dropped_not_guessed(self, osm_name: str) -> None:
        assert locations.match_store(osm_name) is None

    def test_uses_the_stage_one_vocabulary(self) -> None:
        """Not a second copy of the store list: a chain added to stage 1 must
        become mappable without anyone remembering to edit this file."""
        for store in STORE_PATTERNS:
            assert locations.match_store(store) is not None, store

    def test_bjs_restaurant_is_still_excluded(self) -> None:
        # The stage-1 pattern carries this exclusion; matching here inherits it.
        assert locations.match_store("BJ's Restaurant & Brewhouse") is None


class TestExtractPlaces:
    def test_extracts_a_node(self) -> None:
        [place] = locations.extract_places({"elements": [node()]})
        assert place["store"] == "Market Basket"
        assert place["address"] == "400 Somerville Ave"
        assert place["city"] == "Somerville"
        assert place["osm"] == "node/1"

    def test_a_way_uses_its_centre(self) -> None:
        way = node(type="way", id=7, center={"lat": 42.1, "lon": -71.1})
        del way["lat"], way["lon"]
        [place] = locations.extract_places({"elements": [way]})
        assert (place["lat"], place["lon"]) == (42.1, -71.1)
        assert place["osm"] == "way/7"

    def test_unnamed_features_are_skipped(self) -> None:
        assert locations.extract_places({"elements": [node(tags={"shop": "deli"})]}) == []

    def test_features_without_coordinates_are_skipped(self) -> None:
        bare = node()
        del bare["lat"]
        assert locations.extract_places({"elements": [bare]}) == []

    def test_unknown_chains_are_skipped(self) -> None:
        other = node(tags={"name": "7-Eleven", "shop": "convenience"})
        assert locations.extract_places({"elements": [other]}) == []

    def test_the_same_feature_twice_yields_one_place(self) -> None:
        assert len(locations.extract_places({"elements": [node(), node()]})) == 1

    def test_a_missing_address_is_empty_not_absent(self) -> None:
        [place] = locations.extract_places(
            {"elements": [node(tags={"name": "Aldi", "shop": "supermarket"})]}
        )
        assert place["address"] == "" and place["city"] == ""

    def test_an_empty_response_is_not_an_error(self) -> None:
        assert locations.extract_places({}) == []
        assert locations.extract_places({"elements": []}) == []


class TestAttachBranches:
    def _places(self) -> list[locations.Place]:
        return locations.extract_places({
            "elements": [
                node(id=1),
                node(id=2, tags={"name": "Market Basket", "shop": "supermarket",
                                 "addr:city": "Burlington",
                                 "addr:street": "Middlesex Turnpike"}),
                node(id=3, tags={"name": "Market Basket", "shop": "supermarket",
                                 "addr:city": "Cambridge",
                                 "addr:street": "Union Square"}),
            ]
        })

    def test_links_a_pin_to_a_branch_named_in_its_address(self) -> None:
        linked = locations.attach_branches(
            self._places(), {"Market Basket": ["Somerville", "Chelsea"]}
        )
        assert linked == {"node/1": "Somerville"}

    def test_a_branch_nobody_can_place_is_left_unlinked(self) -> None:
        """"the Acre" is a real thing people say and not a thing on a map.
        Attaching it to the nearest pin would invent a fact."""
        linked = locations.attach_branches(
            self._places(), {"Market Basket": ["the Acre", "inside 128"]}
        )
        assert linked == {}

    def test_every_word_of_the_branch_must_be_present(self) -> None:
        # "Union Square" must not match a store merely on a street called
        # Union, nor "Square" alone.
        linked = locations.attach_branches(
            self._places(), {"Market Basket": ["Union Square"]}
        )
        assert linked == {"node/3": "Union Square"}
        assert locations.attach_branches(
            self._places(), {"Market Basket": ["Davis Square"]}
        ) == {}

    def test_the_longest_matching_branch_wins(self) -> None:
        linked = locations.attach_branches(
            self._places(), {"Market Basket": ["Union", "Union Square"]}
        )
        assert linked["node/3"] == "Union Square"

    def test_branches_of_another_store_are_not_borrowed(self) -> None:
        linked = locations.attach_branches(
            self._places(), {"Star Market": ["Somerville"]}
        )
        assert linked == {}

    def test_a_pin_with_no_address_is_skipped(self) -> None:
        bare = locations.extract_places(
            {"elements": [node(id=9, tags={"name": "Aldi", "shop": "supermarket"})]}
        )
        assert locations.attach_branches(bare, {"Aldi": ["Somerville"]}) == {}


class TestQuery:
    def test_the_bbox_covers_cambridge(self) -> None:
        south, west, north, east = locations.BBOX
        assert south < 42.3736 < north and west < -71.1097 < east

    def test_the_query_is_well_formed(self) -> None:
        assert locations.QUERY.startswith("[out:json]")
        assert "supermarket" in locations.QUERY
        assert str(locations.BBOX[0]) in locations.QUERY

    def test_attribution_names_the_source_and_licence(self) -> None:
        assert "OpenStreetMap" in locations.ATTRIBUTION
        assert "ODbL" in locations.ATTRIBUTION
