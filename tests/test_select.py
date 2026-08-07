"""Tests for stage 1 selection."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from groceries import select
from groceries.types import Candidate, RawDoc


class TestDocText:
    def test_prefers_comment_body(self) -> None:
        raw: RawDoc = {"body": "a comment", "title": "ignored"}
        assert select.doc_text(raw) == "a comment"

    def test_joins_post_title_and_selftext(self) -> None:
        raw: RawDoc = {"title": "Title", "selftext": "Body"}
        assert select.doc_text(raw) == "Title\n\nBody"

    def test_post_with_no_selftext(self) -> None:
        assert select.doc_text({"title": "Only title"}) == "Only title"

    def test_empty_doc(self) -> None:
        assert select.doc_text({}) == ""

    def test_empty_body_is_not_treated_as_missing(self) -> None:
        # "" is a real body, distinct from absent — must not fall through to title.
        assert select.doc_text({"body": "", "title": "T"}) == ""


class TestPermalink:
    def test_post_uses_existing_permalink(self) -> None:
        raw: RawDoc = {"id": "p1", "permalink": "/r/boston/comments/p1/title/"}
        assert select.permalink(raw, "boston", "posts") == "/r/boston/comments/p1/title/"

    def test_post_without_permalink_is_reconstructed(self) -> None:
        assert select.permalink({"id": "p1"}, "boston", "posts") == "/r/boston/comments/p1/"

    def test_comment_uses_link_id(self) -> None:
        raw: RawDoc = {"id": "c1", "link_id": "t3_p1"}
        assert select.permalink(raw, "boston", "comments") == "/r/boston/comments/p1/_/c1/"

    def test_comment_without_link_id(self) -> None:
        assert select.permalink({"id": "c1"}, "boston", "comments") == "/r/boston/comments//_/c1/"


class TestMatchedStores:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Market Basket is great", ["Market Basket"]),
            ("marketbasket rules", ["Market Basket"]),
            ("Trader Joes has it", ["Trader Joe's"]),
            ("Trader Joe's has it", ["Trader Joe's"]),
            ("stop and shop", ["Stop & Shop"]),
            ("Stop & Shop", ["Stop & Shop"]),
            ("stop n shop", ["Stop & Shop"]),
            ("H-Mart", ["H Mart"]),
            ("hmart", ["H Mart"]),
            ("Shaws", ["Shaw's"]),
            ("whole food", ["Whole Foods"]),
        ],
    )
    def test_name_variants(self, text: str, expected: list[str]) -> None:
        assert select.matched_stores(text) == expected

    def test_no_match(self) -> None:
        assert select.matched_stores("just a normal sentence") == []

    def test_multiple_stores(self) -> None:
        got = select.matched_stores("Aldi vs Costco")
        assert set(got) == {"Aldi", "Costco"}

    def test_bjs_restaurant_is_excluded(self) -> None:
        assert "BJ's" not in select.matched_stores("BJ's Restaurant and Brewhouse")

    def test_bjs_wholesale_matches(self) -> None:
        assert "BJ's" in select.matched_stores("BJ's has bulk paper towels")

    def test_target_alone_needs_grocery_context(self) -> None:
        assert select.matched_stores("my target audience") == []

    def test_target_alone_with_grocery_context(self) -> None:
        assert select.matched_stores("Target has cheap milk") == ["Target"]

    def test_target_dropped_beside_real_store_without_context(self) -> None:
        got = select.matched_stores("Aldi beat my target for the quarter")
        assert got == ["Aldi"]

    def test_target_kept_beside_real_store_with_context(self) -> None:
        got = select.matched_stores("Aldi and Target both sell milk")
        assert set(got) == {"Aldi", "Target"}


class TestStoreLiteralGate:
    def test_gate_never_rejects_a_real_store_mention(self) -> None:
        """The pre-gate must be a true necessary condition for every pattern."""
        probes = {
            "Market Basket": "market basket", "Trader Joe's": "trader joes",
            "Star Market": "star market", "Whole Foods": "whole foods",
            "Stop & Shop": "stop and shop", "Shaw's": "shaws", "H Mart": "hmart",
            "Wegmans": "wegmans", "Aldi": "aldi", "Costco": "costco",
            "BJ's": "bjs", "Haymarket": "haymarket", "Russo's": "russos",
            "Broadway Marketplace": "broadway marketplace",
            "Dave's Fresh Pasta": "daves fresh pasta",
            "Formaggio Kitchen": "formaggio", "Cardullo's": "cardullo",
            "Reliable Market": "reliable market", "Ebisuya": "ebisuya",
            "Target": "target",
        }
        assert set(probes) == set(select.STORE_PATTERNS), "probe list drifted"
        for store, text in probes.items():
            assert select.STORES[store].search(text), f"probe missed {store}"
            assert select.may_mention_store(text), f"gate would drop {store}"

    def test_gate_rejects_unrelated_text(self) -> None:
        assert not select.may_mention_store("the prices here are terrible")


class TestEvaluative:
    @pytest.mark.parametrize(
        "text",
        ["prices are high", "so cheap", "the quality is bad", "$4.99", "worth it",
         "I recommend it", "best produce", "avoid this place"],
    )
    def test_positive_cases(self, text: str) -> None:
        assert select.is_evaluative(text)

    def test_negative_case(self) -> None:
        assert not select.is_evaluative("I walked past the building yesterday")


class TestEvaluate:
    def _raw(self, body: str) -> RawDoc:
        return {"id": "x", "created_utc": 1, "body": body, "link_id": "t3_p"}

    def test_accepts_qualifying_document(self) -> None:
        raw = self._raw("Market Basket has the cheapest produce in the whole area")
        cand = select.evaluate(raw, "boston", "comments")
        assert cand is not None
        assert cand["stores"] == ["Market Basket"]
        assert cand["truncated"] is False

    def test_rejects_short_text(self) -> None:
        assert select.evaluate(self._raw("cheap MB"), "boston", "comments") is None

    def test_rejects_empty_text(self) -> None:
        assert select.evaluate({"id": "x", "created_utc": 1}, "boston", "comments") is None

    @pytest.mark.parametrize("placeholder", ["[deleted]", "[removed]"])
    def test_rejects_placeholders(self, placeholder: str) -> None:
        # Rejected by the length floor, not by a name check — an explicit
        # placeholder membership test was dead code for exactly this reason.
        raw: RawDoc = {"id": "x", "created_utc": 1, "body": placeholder}
        assert len(placeholder) < select.MIN_CHARS
        assert select.evaluate(raw, "boston", "comments") is None

    def test_rejects_non_evaluative(self) -> None:
        raw = self._raw("I drove past the Market Basket on my way home from work today")
        assert select.evaluate(raw, "boston", "comments") is None

    def test_rejects_no_store(self) -> None:
        raw = self._raw("The prices around here are absolutely terrible these days")
        assert select.evaluate(raw, "boston", "comments") is None

    def test_rejects_when_gate_passes_but_matcher_rejects(self) -> None:
        # Contains the "target" literal and is evaluative, so it clears both
        # cheap filters — but Target is context-gated and there is no food
        # context, so the full matcher still yields nothing.
        raw = self._raw("My sales target this quarter was terrible, worst ever")
        assert select.may_mention_store(raw["body"].lower())
        assert select.is_evaluative(raw["body"])
        assert select.evaluate(raw, "boston", "comments") is None

    def test_truncates_long_text(self) -> None:
        body = "Market Basket is cheap. " + ("x" * select.MAX_CHARS)
        cand = select.evaluate(self._raw(body), "boston", "comments")
        assert cand is not None
        assert cand["truncated"] is True
        assert len(cand["text"]) == select.MAX_CHARS

    def test_truncation_keeps_the_store_mention_in_view(self) -> None:
        # The store is matched against the full document, so a head-truncation
        # could otherwise hand the model text naming no store at all.
        body = ("filler about prices and quality. " * 400) + " Market Basket is cheapest."
        assert len(body) > select.MAX_CHARS
        cand = select.evaluate(self._raw(body), "boston", "comments")
        assert cand is not None
        assert cand["truncated"] is True
        assert len(cand["text"]) <= select.MAX_CHARS
        assert select.matched_stores(cand["text"]) == ["Market Basket"]

    def test_truncation_keeps_the_head_when_the_store_is_early(self) -> None:
        body = "Market Basket is cheapest. " + ("filler about prices. " * 400)
        cand = select.evaluate(self._raw(body), "boston", "comments")
        assert cand is not None
        assert cand["text"].startswith("Market Basket")

    def test_inherits_a_store_named_only_by_the_parent(self) -> None:
        raw = self._raw("Their produce is genuinely cheap compared to anywhere else")
        assert select.evaluate(raw, "boston", "comments") is None
        cand = select.evaluate(
            raw, "boston", "comments", parent_body="Market Basket is my go-to."
        )
        assert cand is not None
        assert cand["stores"] == ["Market Basket"]
        assert cand["parent_body"].startswith("Market Basket")

    def test_an_ambiguous_parent_is_not_inherited(self) -> None:
        # Two stores in the parent leaves "their produce" without a referent.
        raw = self._raw("Their produce is genuinely cheap compared to anywhere else")
        parent = "I go to Market Basket, though Aldi is close too."
        assert select.evaluate(raw, "boston", "comments", parent_body=parent) is None

    def test_a_storeless_parent_is_not_inherited(self) -> None:
        raw = self._raw("Their produce is genuinely cheap compared to anywhere else")
        assert select.evaluate(raw, "boston", "comments", parent_body="I agree.") is None

    def test_inheritance_requires_grocery_context(self) -> None:
        # Evaluative, but about the parking lot rather than about shopping.
        raw = self._raw("Honestly the parking lot there is the worst I have seen")
        assert select.is_evaluative(raw["body"])
        parent = "Market Basket in Somerville."
        assert select.evaluate(raw, "boston", "comments", parent_body=parent) is None

    def test_parent_body_is_truncated(self) -> None:
        raw = self._raw("Their produce is genuinely cheap compared to anywhere else")
        parent = "Market Basket. " + "x" * (select.PARENT_CONTEXT_CHARS * 2)
        cand = select.evaluate(raw, "boston", "comments", parent_body=parent)
        assert cand is not None
        assert len(cand["parent_body"]) == select.PARENT_CONTEXT_CHARS

    def test_first_mention_offset(self) -> None:
        assert select.first_mention("xx Aldi", ["Aldi"]) == 3
        assert select.first_mention("nothing here", ["Aldi"]) == 0

    def test_score_is_optional(self) -> None:
        cand = select.evaluate(
            self._raw("Market Basket has the cheapest produce anywhere"), "boston", "comments"
        )
        assert cand is not None
        assert cand["score"] is None


def _write_shard(root: Path, kind: str, sub: str, name: str, rows: list[RawDoc]) -> Path:
    d = root / kind / sub
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.ndjson.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


class TestReadShardAndSelect:
    def test_read_shard_reports_subreddit_and_kind(self, tmp_path: Path) -> None:
        path = _write_shard(tmp_path, "comments", "boston", "2020-01", [{"id": "a"}])
        assert select.read_shard(path) == ([{"id": "a"}], "boston", "comments")

    def test_read_shard_skips_a_corrupt_line(self, tmp_path: Path) -> None:
        path = _write_shard(tmp_path, "comments", "boston", "2020-01", [{"id": "a"}])
        with gzip.open(path, "at", encoding="utf-8") as fh:
            fh.write("{truncated\n")
        rows, _, _ = select.read_shard(path)
        assert rows == [{"id": "a"}]

    def test_parent_index_skips_rows_without_a_body(self) -> None:
        rows: list[RawDoc] = [
            {"id": "a", "body": "hello"}, {"id": "b"}, {"body": "orphan"}
        ]
        assert select.parent_index(rows) == {"a": "hello"}

    def test_parent_of_resolves_a_comment_parent(self) -> None:
        index = {"a": "Market Basket is my go-to."}
        raw: RawDoc = {"id": "b", "parent_id": "t1_a"}
        assert select.parent_of(raw, index) == "Market Basket is my go-to."

    @pytest.mark.parametrize("parent_id", ["t3_a", "", "t1_missing"])
    def test_parent_of_returns_empty_when_unresolvable(self, parent_id: str) -> None:
        # t3_ is the post itself, not a comment; a t1_ parent outside this
        # shard is simply absent.
        raw: RawDoc = {"id": "b", "parent_id": parent_id}
        assert select.parent_of(raw, {"a": "x"}) == ""

    def test_select_resolves_parents_within_a_shard(self, tmp_path: Path) -> None:
        rows: list[RawDoc] = [
            # The reply precedes its parent in the file, so a streaming index
            # would miss it.
            {"id": "b", "created_utc": 1, "parent_id": "t1_a",
             "body": "Their produce is genuinely cheap compared to anywhere else"},
            {"id": "a", "created_utc": 1, "body": "I shop at Market Basket."},
        ]
        path = _write_shard(tmp_path, "comments", "boston", "2020-01", rows)
        cands, _ = select.select([path])
        assert [c["id"] for c in cands] == ["b"]
        assert cands[0]["stores"] == ["Market Basket"]

    def test_select_counts_and_filters(self, tmp_path: Path) -> None:
        good: RawDoc = {"id": "a", "created_utc": 1, "body": "Market Basket is so cheap for produce around here"}
        bad: RawDoc = {"id": "b", "created_utc": 1, "body": "nothing relevant at all in this particular comment"}
        path = _write_shard(tmp_path, "comments", "boston", "2020-01", [good, bad])
        cands, report = select.select([path])
        assert report.scanned == 2
        assert report.kept == 1
        assert cands[0]["id"] == "a"
        assert report.per_store["Market Basket"] == 1
        assert report.per_subreddit["boston"] == 1

    def test_limit_zero_yields_nothing(self, tmp_path: Path) -> None:
        rows: list[RawDoc] = [
            {"id": str(i), "created_utc": 1,
             "body": f"Market Basket is so cheap for produce here {i}"}
            for i in range(3)
        ]
        path = _write_shard(tmp_path, "comments", "boston", "2020-01", rows)
        cands, report = select.select([path], limit=0)
        assert cands == [] and report.kept == 0

    def test_select_honours_limit(self, tmp_path: Path) -> None:
        rows: list[RawDoc] = [
            {"id": str(i), "created_utc": 1,
             "body": f"Market Basket is so cheap for produce around here {i}"}
            for i in range(5)
        ]
        path = _write_shard(tmp_path, "comments", "boston", "2020-01", rows)
        cands, report = select.select([path], limit=2)
        assert len(cands) == 2
        assert report.kept == 2

    def test_select_across_multiple_shards(self, tmp_path: Path) -> None:
        a = _write_shard(tmp_path, "comments", "boston", "2020-01",
                         [{"id": "a", "created_utc": 1, "body": "Aldi is cheap and the produce is good here"}])
        b = _write_shard(tmp_path, "posts", "Somerville", "2020-02",
                         [{"id": "b", "created_utc": 1, "title": "Costco prices are great value for bulk goods"}])
        cands, report = select.select([a, b])
        assert report.kept == 2
        assert {c["subreddit"] for c in cands} == {"boston", "Somerville"}


class TestCorruptShard:
    def test_one_bad_line_does_not_abort_the_scan(self, tmp_path: Path) -> None:
        import gzip as _gzip

        d = tmp_path / "comments" / "boston"
        d.mkdir(parents=True)
        path = d / "2020-01.ndjson.gz"

        def row(rid: str, extra: str) -> str:
            return json.dumps({
                "id": rid, "created_utc": 1,
                "body": f"Market Basket is so cheap for produce around here {extra}",
            })

        with _gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(row("a", "one") + "\n{truncated\n" + row("b", "two") + "\n")
        cands, _ = select.select([path])
        assert [c["id"] for c in cands] == ["a", "b"]


class TestVisibleStores:
    def test_truncation_narrows_the_advertised_store_list(self) -> None:
        # Stores are matched on the full text; the prompt must only name the
        # ones still visible in the excerpt actually sent to the model.
        body = ("Aldi is cheap. " + "filler about prices. " * 400
                + " Market Basket has the best produce.")
        raw: RawDoc = {"id": "x", "created_utc": 1, "body": body}
        cand = select.evaluate(raw, "boston", "comments")
        assert cand is not None
        assert set(cand["stores"]) <= set(select.matched_stores(cand["text"]))
        assert cand["stores"] == select.matched_stores(cand["text"])


class TestBoilerplateAndBots:
    def test_identical_texts_are_kept_once(self, tmp_path: Path) -> None:
        # One nightlife copypasta appears 947 times in the real corpus and
        # matches a store name; without this it would dominate that store.
        body = "Market Basket is so cheap for produce around here"
        rows: list[RawDoc] = [
            {"id": f"d{i}", "created_utc": 1, "body": body} for i in range(5)
        ]
        path = _write_shard(tmp_path, "comments", "boston", "2020-01", rows)
        cands, report = select.select([path])
        assert len(cands) == 1
        assert report.duplicates == 4

    def test_bot_authors_are_dropped(self, tmp_path: Path) -> None:
        body = "Market Basket is so cheap for produce around here"
        rows: list[RawDoc] = [
            {"id": "a", "created_utc": 1, "body": body, "author": "AutoModerator"},
            {"id": "b", "created_utc": 1, "body": body + " really", "author": "alice"},
        ]
        path = _write_shard(tmp_path, "comments", "boston", "2020-01", rows)
        cands, _ = select.select([path])
        assert [c["id"] for c in cands] == ["b"]

    def test_author_is_carried_for_independence_control(self) -> None:
        raw: RawDoc = {
            "id": "x", "created_utc": 1, "author": "alice",
            "body": "Market Basket has the cheapest produce in the whole area",
        }
        cand = select.evaluate(raw, "boston", "comments")
        assert cand is not None and cand["author"] == "alice"

    def test_haymarket_needs_grocery_context(self) -> None:
        assert select.matched_stores("I got off at Haymarket and walked north") == []
        assert select.matched_stores("Haymarket has cheap produce") == ["Haymarket"]


class TestSubredditFilter:
    def test_restricts_to_named_subreddits(self, tmp_path: Path) -> None:
        body = "Market Basket is so cheap for produce around here"
        a = _write_shard(tmp_path, "comments", "boston", "2020-01",
                         [{"id": "a", "created_utc": 1, "body": body}])
        b = _write_shard(tmp_path, "comments", "traderjoes", "2020-01",
                         [{"id": "b", "created_utc": 1, "body": body}])
        cands, report = select.select([a, b], subreddits={"boston"})
        assert [c["id"] for c in cands] == ["a"]
        # Filtered-out documents are not counted as scanned.
        assert report.scanned == 1

    def test_none_means_every_subreddit(self, tmp_path: Path) -> None:
        body = "Market Basket is so cheap for produce around here"
        a = _write_shard(tmp_path, "comments", "boston", "2020-01",
                         [{"id": "a", "created_utc": 1, "body": body}])
        cands, _ = select.select([a], subreddits=None)
        assert len(cands) == 1


class TestRoundTrip:
    def test_write_then_read(self, tmp_path: Path, candidate: Candidate) -> None:
        path = tmp_path / "nested" / "ws.jsonl"
        assert select.write_candidates([candidate], path) == 1
        assert select.read_candidates(path) == ([candidate], 0)

    def test_read_skips_blank_lines(self, tmp_path: Path, candidate: Candidate) -> None:
        path = tmp_path / "ws.jsonl"
        path.write_text(json.dumps(candidate) + "\n\n", encoding="utf-8")
        assert select.read_candidates(path) == ([candidate], 0)

    def test_a_truncated_line_is_counted_not_fatal(
        self, tmp_path: Path, candidate: Candidate
    ) -> None:
        path = tmp_path / "ws.jsonl"
        path.write_text(json.dumps(candidate) + "\n{trunc", encoding="utf-8")
        assert select.read_candidates(path) == ([candidate], 1)

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda c: c.pop("text"), id="missing-field"),
            pytest.param(lambda c: c.update(created_utc="1700000000"), id="ts-string"),
            pytest.param(lambda c: c.update(created_utc=True), id="ts-bool"),
            pytest.param(lambda c: c.update(stores="Aldi"), id="stores-not-a-list"),
            pytest.param(lambda c: c.update(stores=[7]), id="stores-not-strings"),
            pytest.param(lambda c: c.update(score="12"), id="score-string"),
            pytest.param(lambda c: c.update(score=True), id="score-bool"),
            pytest.param(lambda c: c.update(text=None), id="text-null"),
        ],
    )
    def test_rows_stage2_would_choke_on_are_rejected(
        self, tmp_path: Path, candidate: Candidate, mutate: Any
    ) -> None:
        """Paying per document to discover the working set is stale is the
        expensive way to learn it."""
        bad = dict(candidate)
        mutate(bad)
        path = tmp_path / "ws.jsonl"
        path.write_text(json.dumps(bad), encoding="utf-8")
        assert select.read_candidates(path) == ([], 1)

    def test_a_null_score_is_allowed(self, tmp_path: Path, candidate: Candidate) -> None:
        # Posts genuinely carry no score; that is not corruption.
        row = {**candidate, "score": None}
        path = tmp_path / "ws.jsonl"
        path.write_text(json.dumps(row), encoding="utf-8")
        assert select.read_candidates(path)[1] == 0

    def test_a_non_object_row_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "ws.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        assert select.read_candidates(path) == ([], 1)


class TestSelectionReport:
    def test_counters_default_to_empty(self) -> None:
        report = select.SelectionReport()
        assert report.per_store == {}
        assert report.per_subreddit == {}

    def test_supplied_counters_are_kept(self) -> None:
        from collections import Counter

        report = select.SelectionReport(per_store=Counter({"Aldi": 3}))
        assert report.per_store["Aldi"] == 3
