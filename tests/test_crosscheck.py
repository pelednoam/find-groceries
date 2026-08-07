"""Tests for the Google cross-check.

The bar here is narrower than elsewhere and mostly about restraint: the
numbers must stay separate from the verdict, and no review prose may leak
into anything published. Both failure modes are silent.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from groceries import crosscheck

DAY = 86_400_000  # the dataset stores milliseconds
Y2018 = 1_514_764_800_000
Y2021 = 1_609_459_200_000


def review(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "store": "Market Basket",
        "gmap_id": "0xabc",
        "rating": 5,
        "time": Y2018,
        "text_len": 200,
    }
    base.update(over)
    return base


class TestNormalise:
    @pytest.mark.parametrize(
        ("stars", "expected"), [(5.0, 1.0), (4.0, 0.5), (3.0, 0.0), (1.0, -1.0)]
    )
    def test_maps_stars_onto_the_sentiment_scale(
        self, stars: float, expected: float
    ) -> None:
        assert crosscheck.normalise(stars) == expected

    def test_the_scales_are_comparable(self) -> None:
        """The whole cross-check rests on this: a 4.0 mean and a +0.5
        sentiment must mean the same distance from neutral."""
        assert -1.0 <= crosscheck.normalise(1.0) <= crosscheck.normalise(5.0) <= 1.0


class TestSummarise:
    def test_basic_aggregate(self) -> None:
        got = crosscheck.summarise([review(rating=5), review(rating=3)])
        assert got is not None
        assert got["n"] == 2 and got["mean"] == 4.0 and got["norm"] == 0.5

    def test_empty_bucket(self) -> None:
        assert crosscheck.summarise([]) is None

    def test_thin_is_flagged_not_hidden(self) -> None:
        few = crosscheck.summarise([review()] * 5)
        many = crosscheck.summarise([review()] * crosscheck.MIN_RATINGS)
        assert few is not None and many is not None
        assert few["thin"] is True and many["thin"] is False

    def test_long_reviews_are_summarised_separately(self) -> None:
        """People tap five stars on the way out and write prose when annoyed;
        measured on the real corpus that is worth 0.36 stars."""
        mixed = [review(rating=5, text_len=10)] * 8 + [review(rating=1, text_len=300)] * 2
        got = crosscheck.summarise(mixed)
        assert got is not None
        assert got["mean"] == 4.2
        assert got["n_long"] == 2 and got["mean_long"] == 1.0

    def test_no_long_reviews_yields_null_not_zero(self) -> None:
        got = crosscheck.summarise([review(text_len=5)])
        assert got is not None
        assert got["mean_long"] is None and got["norm_long"] is None
        assert got["n_long"] == 0

    def test_dates_bound_the_evidence(self) -> None:
        got = crosscheck.summarise(
            [review(time=Y2018), review(time=Y2021), review(time=Y2021)]
        )
        assert got is not None
        assert got["first"] == "2018-01" and got["last"] == "2021-01"
        assert got["median_date"] == "2021-01"

    def test_carries_nothing_but_numbers(self) -> None:
        """A review's text must not reach the output even by accident."""
        got = crosscheck.summarise([{**review(), "text": "SECRET", "user_id": "u1",
                                     "name": "Alice"}])
        assert got is not None
        blob = json.dumps(got)
        for leak in ("SECRET", "u1", "Alice", "text", "user_id", "name"):
            assert leak not in blob, leak


class TestBuckets:
    def test_by_store(self) -> None:
        got = crosscheck.by_store([review(store="Aldi"), review(store="Costco")])
        assert set(got) == {"Aldi", "Costco"}

    def test_by_location(self) -> None:
        got = crosscheck.by_location([review(gmap_id="a"), review(gmap_id="b"),
                                      review(gmap_id="b")])
        assert got["b"]["n"] == 2


class TestGeoMatching:
    def test_distance_is_sane(self) -> None:
        # Harvard Square to Central Square is a little over a kilometre.
        d = crosscheck.metres(42.3736, -71.1190, 42.3653, -71.1037)
        assert 1_200 < d < 1_600

    def test_zero_distance(self) -> None:
        assert crosscheck.metres(42.0, -71.0, 42.0, -71.0) == 0.0

    def _pair(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        google = [{"gmap_id": "g1", "store": "Aldi", "latitude": 42.3875,
                   "longitude": -71.0995}]
        osm = [{"osm": "node/1", "store": "Aldi", "lat": 42.3876, "lon": -71.0996}]
        return google, osm

    def test_matches_the_same_shop(self) -> None:
        google, osm = self._pair()
        assert crosscheck.match_to_places(google, osm) == {"g1": "node/1"}

    def test_a_different_chain_at_the_same_spot_does_not_match(self) -> None:
        google, osm = self._pair()
        osm[0]["store"] = "Costco"
        assert crosscheck.match_to_places(google, osm) == {}

    def test_too_far_apart_does_not_match(self) -> None:
        google, osm = self._pair()
        osm[0]["lat"] = 42.50
        assert crosscheck.match_to_places(google, osm) == {}

    def test_matching_is_one_to_one(self) -> None:
        """Two Google records must not both claim one pin, or a branch would
        show one shop's rating twice."""
        google = [
            {"gmap_id": "near", "store": "Aldi", "latitude": 42.3875, "longitude": -71.0995},
            {"gmap_id": "far", "store": "Aldi", "latitude": 42.3880, "longitude": -71.0999},
        ]
        osm = [{"osm": "node/1", "store": "Aldi", "lat": 42.3875, "lon": -71.0995}]
        linked = crosscheck.match_to_places(google, osm)
        assert linked == {"near": "node/1"}, "the closer record should win, alone"

    def test_the_closest_pin_wins(self) -> None:
        google = [{"gmap_id": "g1", "store": "Aldi", "latitude": 42.3875,
                   "longitude": -71.0995}]
        osm = [
            {"osm": "node/far", "store": "Aldi", "lat": 42.3884, "lon": -71.0995},
            {"osm": "node/near", "store": "Aldi", "lat": 42.38751, "lon": -71.0995},
        ]
        assert crosscheck.match_to_places(google, osm) == {"g1": "node/near"}


class TestBuild:
    def _inputs(self) -> tuple[list[Any], list[Any], list[Any]]:
        reviews = [review(gmap_id="g1", store="Aldi", rating=4) for _ in range(30)]
        google = [{"gmap_id": "g1", "store": "Aldi", "latitude": 42.3875,
                   "longitude": -71.0995}]
        osm = [{"osm": "node/1", "store": "Aldi", "lat": 42.3875, "lon": -71.0995}]
        return reviews, google, osm

    def test_produces_both_views(self) -> None:
        block = crosscheck.build(*self._inputs())
        assert block["stores"]["Aldi"]["mean"] == 4.0
        assert block["locations"]["node/1"]["n"] == 30

    def test_locations_are_keyed_by_osm_id(self) -> None:
        """So a map pin can look itself up without a second join."""
        block = crosscheck.build(*self._inputs())
        assert list(block["locations"]) == ["node/1"]

    def test_an_unmatched_location_is_dropped_from_the_map_view(self) -> None:
        reviews, google, osm = self._inputs()
        osm[0]["lat"] = 42.50
        block = crosscheck.build(reviews, google, osm)
        assert block["locations"] == {}
        assert block["stores"]["Aldi"]["n"] == 30, "the store view still has it"

    def test_reports_its_own_coverage(self) -> None:
        block = crosscheck.build(*self._inputs())
        assert block["coverage"] == "2018-01 to 2018-01"
        assert block["n_reviews"] == 30 and block["n_matched_to_map"] == 1

    def test_cites_the_dataset(self) -> None:
        block = crosscheck.build(*self._inputs())
        assert "UC San Diego" in block["citation"]

    def test_empty_input(self) -> None:
        block = crosscheck.build([], [], [])
        assert block["n_reviews"] == 0 and block["coverage"] == ""

    def test_the_whole_block_is_free_of_review_text(self) -> None:
        reviews, google, osm = self._inputs()
        reviews[0] = {**reviews[0], "text": "SECRET REVIEW", "user_id": "u1"}
        blob = json.dumps(crosscheck.build(reviews, google, osm))
        assert "SECRET" not in blob and "u1" not in blob


class TestPublishedCrossCheck:
    """Guards on what actually ships, when it has been built."""

    @pytest.fixture
    def payload(self) -> Any:
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "docs" / "verdicts.json"
        if not path.exists():
            pytest.skip("docs/verdicts.json not built")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_verdict_is_untouched_by_google(self, payload: Any) -> None:
        """The point of the cross-check is that it is *beside* the verdict.
        If a rating ever leaks into `stores` or `totals`, the site is quietly
        averaging two sources it tells the reader it keeps apart."""
        cc = payload.get("crosscheck")
        if not cc:
            pytest.skip("no cross-check in this build")
        for store, rating in cc["stores"].items():
            totals = payload["totals"].get(store)
            if totals is None:
                continue
            assert totals["s"] != rating["norm"] or abs(rating["norm"]) < 1e-9

    def test_published_block_has_no_text_fields(self, payload: Any) -> None:
        cc = payload.get("crosscheck")
        if not cc:
            pytest.skip("no cross-check in this build")
        allowed = {"n", "mean", "norm", "n_long", "mean_long", "norm_long",
                   "thin", "first", "last", "median_date"}
        for rating in list(cc["stores"].values()) + list(cc["locations"].values()):
            assert set(rating) <= allowed, set(rating) - allowed
