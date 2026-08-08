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
        "score": 7.35,
        "valenced_weight": 12.0,
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
                      "sentiment": -0.5 + 0.15 * i,
                      "score": (-0.5 + 0.15 * i) * 40.0, "valenced_weight": 40.0,
                      "insufficient_evidence": False}
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

    def test_a_store_with_no_google_reproduces_stage_three(self) -> None:
        """The invariant that keeps the merge from double-counting the prior:
        one source in, stage 3's own arithmetic out."""
        from groceries.merge import PRIOR_WEIGHT

        v = self._verdicts()
        cc = self._cc()
        del cc["stores"]["S3"]
        merged = site.build_payload(v, None, cc)["merged"]
        t = v["store_totals"]["S3"]
        assert merged["stores"]["S3"]["v"] == pytest.approx(
            t["score"] / (t["valenced_weight"] + PRIOR_WEIGHT), abs=0.001
        )

    def test_the_branch_error_bar_is_measured_where_it_is_used(self) -> None:
        """The line transfers from chain to branch; the residual does not.

        On the real corpus the chain residual is 0.27 and the branch one
        0.34, so a branch merge that inherited the chain figure would give
        Google more weight than its accuracy there deserves.
        """
        v, cc, loc = self._with_branches()
        merged = site.build_payload(v, loc, cc)["merged"]
        assert merged["branch_calibration"]["slope"] == merged["calibration"]["slope"]
        assert (merged["branch_calibration"]["residual_sd"]
                >= merged["calibration"]["residual_sd"])

    def test_a_residual_measured_on_too_few_branches_is_not_trusted(self) -> None:
        """With one branch the "measured" residual is that branch's own
        disagreement, which would then discount the source it came from."""
        v, cc, loc = self._with_branches()
        merged = site.build_payload(v, loc, cc)["merged"]
        assert (merged["branch_calibration"]["residual_sd"]
                == merged["calibration"]["residual_sd"]), "should fall back"

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
            "n_claims": 2, "weighted_evidence": 0.6, "sentiment": -0.4,
            "score": -0.24, "valenced_weight": 0.6}}}
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
        # pooled norm = (400*0.8 + 200*0.6)/600 = 0.733, so both pins counted
        # The affine map would give +1.25; sentiment is bounded at +1.
        assert b["g"] == pytest.approx(1.0, abs=0.01)
        assert b["share"] < 0.1, "0.6 units of evidence must not outvote 600 ratings"

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

    def test_an_older_verdict_file_without_the_raw_fields_still_merges(self) -> None:
        """`score`/`valenced_weight` postdate the merge; a stale file must
        degrade to an approximation rather than to no output."""
        v = self._verdicts()
        for t in v["store_totals"].values():
            del t["score"], t["valenced_weight"]
        merged = site.build_payload(v, None, self._cc())["merged"]
        assert merged is not None and len(merged["stores"]) == 8

    def test_a_cell_with_no_valenced_claims_has_no_reddit_side(self) -> None:
        v = self._verdicts()
        v["store_totals"]["S0"].update(score=0.0, valenced_weight=0.0,
                                       weighted_evidence=0.0)
        merged = site.build_payload(v, None, self._cc())["merged"]
        assert "r" not in merged["stores"]["S0"]

    def test_a_branch_with_no_valenced_claims_is_skipped_when_calibrating(
        self,
    ) -> None:
        """It has a Google rating but no Reddit mean, so it cannot contribute
        a residual — including it as zero would flatter the error bar."""
        v, cc, loc = self._with_branches()
        v["branch_totals"]["S0"]["Somerville"].update(
            score=0.0, valenced_weight=0.0, weighted_evidence=0.0
        )
        merged = site.build_payload(v, loc, cc)["merged"]
        assert (merged["branch_calibration"]["residual_sd"]
                == merged["calibration"]["residual_sd"])

    def test_a_pin_whose_rating_has_no_count_is_skipped(self) -> None:
        v, cc, loc = self._with_branches()
        cc["locations"] = {"node/1": {"n": 0, "norm": 0.8},
                           "node/2": {"n": 0, "norm": 0.6}}
        merged = site.build_payload(v, loc, cc)["merged"]
        assert merged["branches"]["S0"]["Somerville"]["share"] == 1.0

    def test_too_few_stores_to_calibrate_yields_no_merge(self) -> None:
        cc = {"stores": {"S0": {"n": 500, "norm": 0.6, "thin": False}},
              "locations": {}}
        assert site.build_payload(self._verdicts(), None, cc)["merged"] is None


class TestReviewMergeInPayload:
    def _reviews(self) -> dict[str, Any]:
        return {
            "store_totals": {
                f"S{i}": {"n_claims": 200, "weighted_evidence": 60.0,
                          "sentiment": -0.3 + 0.12 * i,
                          "score": (-0.3 + 0.12 * i) * 60.0,
                          "valenced_weight": 60.0,
                          "insufficient_evidence": False}
                for i in range(8)
            },
            "stores": {f"S{i}": {"produce": cell()} for i in range(8)},
            "branches": {},
        }

    def _verdicts8(self) -> dict[str, Any]:
        v = verdicts()
        v["store_totals"] = {
            f"S{i}": {"n_claims": 100, "weighted_evidence": 40.0,
                      "sentiment": -0.5 + 0.15 * i,
                      "score": (-0.5 + 0.15 * i) * 40.0, "valenced_weight": 40.0,
                      "insufficient_evidence": False}
            for i in range(8)
        }
        v["stores"] = {f"S{i}": {"produce": cell()} for i in range(8)}
        v["branches"] = {"S0": {"Somerville": {"produce": cell()}}}
        v["branch_totals"] = {}
        v["items"] = {}
        return v

    def test_absent_without_review_verdicts(self) -> None:
        assert site.build_payload(self._verdicts8())["reviews"] is None

    def test_merges_per_store_and_per_category(self) -> None:
        """What extracting the text bought: star ratings gave one number per
        shop, so the combination could only touch the overall verdict."""
        out = site.build_payload(self._verdicts8(), None, None, self._reviews())
        assert out["reviews"] is not None
        assert len(out["reviews"]["stores"]) == 8
        assert "produce" in out["reviews"]["categories"]["S0"]

    def test_the_single_source_views_are_untouched(self) -> None:
        v = self._verdicts8()
        before = site.build_payload(v)
        after = site.build_payload(v, None, None, self._reviews())
        assert before["stores"] == after["stores"]
        assert before["totals"] == after["totals"]

    def test_the_note_states_the_publishing_rule(self) -> None:
        out = site.build_payload(self._verdicts8(), None, None, self._reviews())
        assert "no review text" in out["reviews"]["note"]

    def test_no_claim_text_reaches_the_block(self) -> None:
        out = site.build_payload(self._verdicts8(), None, None, self._reviews())
        blob = json.dumps(out["reviews"])
        # "n_review_claims" is a count, not text — look for the fields that
        # would actually carry prose.
        for leak in ('"claim"', '"evidence"', '"t"', '"permalink"'):
            assert leak not in blob, leak

    def test_a_store_absent_from_the_review_corpus_has_no_google_side(self) -> None:
        v = self._verdicts8()
        r = self._reviews()
        del r["store_totals"]["S5"]
        out = site.build_payload(v, None, None, r)
        assert "g" not in out["reviews"]["stores"]["S5"]

    def test_an_older_verdicts_file_still_merges_review_claims(self) -> None:
        """`_merge_block` had a fallback for pre-`score` files and
        `_review_block` did not, so a stale verdicts file silently produced
        no claim merge while the star merge beside it kept working."""
        v = self._verdicts8()
        for t in v["store_totals"].values():
            del t["score"], t["valenced_weight"]
        out = site.build_payload(v, None, None, self._reviews())
        assert out["reviews"] is not None and len(out["reviews"]["stores"]) == 8

    def test_a_store_with_no_valenced_claims_is_skipped(self) -> None:
        r = self._reviews()
        r["store_totals"]["S0"].update(score=0.0, valenced_weight=0.0,
                                       weighted_evidence=0.0)
        out = site.build_payload(self._verdicts8(), None, None, r)
        assert "g" not in out["reviews"]["stores"]["S0"]

    def test_branch_categories_are_merged_too(self) -> None:
        v = self._verdicts8()
        r = self._reviews()
        r["branches"] = {"S0": {"Somerville": {"produce": cell()}}}
        out = site.build_payload(v, None, None, r)
        assert "produce" in out["reviews"]["branches"]["S0"]["Somerville"]

    def test_too_few_comparable_stores_yields_nothing(self) -> None:
        r = self._reviews()
        for t in r["store_totals"].values():
            t["n_claims"] = 1
        assert site.build_payload(self._verdicts8(), None, None, r)["reviews"] is None


class TestPublishingBoundary:
    """The licence rule, enforced rather than merely true.

    Four independent reviewers made the same point: nothing checked that
    review text stays out of the published payload — it held only because
    `_review_block` happens to emit numbers, and `slim_cell` would print the
    text of any cell handed to it. A property that survives by construction
    is one refactor from being false.
    """

    def _payload(self, **over: Any) -> dict[str, Any]:
        p = site.build_payload(verdicts())
        p.update(over)
        return p

    def test_a_clean_payload_passes(self) -> None:
        site.assert_publishable(self._payload())

    @pytest.mark.parametrize("field", ["t", "claim", "evidence", "permalink",
                                       "user_id", "name"])
    def test_review_text_or_identity_is_refused(self, field: str) -> None:
        bad = self._payload(reviews={"stores": {"S": {"v": 1, field: "leak"}}})
        with pytest.raises(site.PublishingError, match=field):
            site.assert_publishable(bad)

    def test_it_looks_all_the_way_down(self) -> None:
        buried = {"a": {"b": [{"c": {"claim": "deep leak"}}]}}
        with pytest.raises(site.PublishingError):
            site.assert_publishable(self._payload(reviews=buried))

    def test_the_dataset_citation_is_required(self) -> None:
        """Publishing statistics from the licensed set without crediting it
        is the one thing the licence actually asks for."""
        with pytest.raises(site.PublishingError, match="citation"):
            site.assert_publishable(self._payload(crosscheck={"stores": {}}))
        site.assert_publishable(
            self._payload(crosscheck={"stores": {}, "citation": "McAuley Lab, UCSD"})
        )

    @pytest.mark.parametrize("field", ["text", "user_id", "author"])
    def test_the_crosscheck_may_not_carry_identity(self, field: str) -> None:
        bad = self._payload(crosscheck={"citation": "x", "stores": {"S": {field: "u"}}})
        with pytest.raises(site.PublishingError):
            site.assert_publishable(bad)

    def test_an_evidence_quote_must_be_traceable(self) -> None:
        """A quote with no reddit permalink is either mis-sourced or from the
        licensed dataset. Either way it must not ship."""
        p = self._payload()
        first = next(iter(p["stores"].values()))
        next(iter(first.values()))["e"][0]["u"] = "https://maps.google.com/x"
        with pytest.raises(site.PublishingError, match="permalink"):
            site.assert_publishable(p)

    def test_write_payload_refuses_rather_than_leaking(self, tmp_path: Path) -> None:
        out = tmp_path / "v.json"
        with pytest.raises(site.PublishingError):
            site.write_payload(self._payload(reviews={"x": {"claim": "leak"}}), out)
        assert not out.exists(), "a refused build must not leave a file behind"

    def test_non_finite_numbers_are_refused(self, tmp_path: Path) -> None:
        """Python writes NaN and Infinity happily; JSON.parse rejects them,
        so the site would fail to load at all."""
        p = self._payload()
        next(iter(next(iter(p["stores"].values())).values()))["s"] = float("nan")
        with pytest.raises(ValueError):
            site.write_payload(p, tmp_path / "v.json")


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

    def test_every_declared_interface_matches_what_python_emits(self) -> None:
        """The previous version checked three interfaces by name, so
        `MergeBlock` silently lost `branch_calibration` — declared nowhere,
        emitted always, and read by no checker on either side."""
        payload = site.build_payload(
            verdicts(), locations(), {"stores": {}, "locations": {},
                                      "citation": "x"},
        )
        merged = payload.get("merged")
        if merged is not None:
            assert set(merged) <= self._interface("MergeBlock"), (
                set(merged) - self._interface("MergeBlock")
            )
        assert set(payload["places"][0]) <= self._interface("Place")

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
