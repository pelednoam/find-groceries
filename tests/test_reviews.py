"""Tests for turning Google reviews into stage-2 documents.

The failure that matters is a claim landing at the wrong shop: reviews are
attached to one listing as a matter of record, so a misplaced claim is worse
than no claim at all — it splits one store's evidence in two.
"""

from __future__ import annotations

from typing import Any

import pytest

from groceries import reviews
from groceries.extract import GOOGLE_SOURCE, user_message
from groceries.types import Candidate

LONG = "The produce here is consistently fresh and the prices are better " \
       "than anywhere else nearby, though the queues get long on Sundays."


def review(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "gmap_id": "0xabc", "user_id": "u1", "time": 1_600_000_000_000,
        "rating": 5, "text": LONG,
    }
    base.update(over)
    return base


class TestReviewId:
    def test_is_stable(self) -> None:
        assert reviews.review_id(review()) == reviews.review_id(review())

    def test_distinguishes_reviews(self) -> None:
        assert reviews.review_id(review()) != reviews.review_id(review(user_id="u2"))
        assert reviews.review_id(review()) != reviews.review_id(review(time=1))

    def test_carries_no_author_identity(self) -> None:
        """It is a digest, so a user id cannot be read back out of a claims
        file that quotes it."""
        assert "u1" not in reviews.review_id(review(user_id="u1"))


class TestToCandidate:
    def test_builds_a_stage_two_document(self) -> None:
        c = reviews.to_candidate(review(), "Market Basket")
        assert c is not None
        assert c["stores"] == ["Market Basket"]
        assert c["subreddit"] == GOOGLE_SOURCE and c["kind"] == "review"
        assert c["text"] == LONG

    def test_milliseconds_become_seconds(self) -> None:
        c = reviews.to_candidate(review(time=1_600_000_000_000), "Aldi")
        assert c is not None and c["created_utc"] == 1_600_000_000

    def test_short_reviews_are_not_worth_a_request(self) -> None:
        assert reviews.to_candidate(review(text="Great!"), "Aldi") is None
        assert reviews.to_candidate(review(text=None), "Aldi") is None

    def test_no_score_rather_than_zero(self) -> None:
        """0 would read as "downvoted" to the vote weighting."""
        c = reviews.to_candidate(review(), "Aldi")
        assert c is not None and c["score"] is None

    def test_long_reviews_are_truncated_and_flagged(self) -> None:
        c = reviews.to_candidate(review(text="x" * 9000), "Aldi")
        assert c is not None
        assert len(c["text"]) == reviews.MAX_CHARS and c["truncated"] is True

    def test_the_prompt_states_the_store_is_known(self) -> None:
        """Reddit documents carry a guess the model is told to distrust; a
        review carries a fact, and telling the model to second-guess it
        would invite reassigning claims to shops nobody visited."""
        c = reviews.to_candidate(review(), "Market Basket")
        assert c is not None
        msg = user_message(c)
        assert "Google Maps review of Market Basket" in msg
        assert "pre-filter" not in msg


class TestCandidates:
    def test_only_known_places(self) -> None:
        rows = [review(gmap_id="known"), review(gmap_id="unknown", user_id="u2")]
        got = reviews.candidates(rows, {"known": "Aldi"})
        assert [c["stores"][0] for c in got] == ["Aldi"]

    def test_empty_input(self) -> None:
        assert reviews.candidates([], {}) == []


class TestAttachPlace:
    def _claim(self, **over: Any) -> Any:
        base = {"source_id": "d1", "location": "guessed", "store": "Aldi"}
        base.update(over)
        return base

    def test_the_known_branch_overrides_the_guess(self) -> None:
        got = reviews.attach_place(
            [self._claim()], {"d1": "g1"}, {"g1": "Somerville"}
        )
        assert got[0]["location"] == "Somerville"

    def test_an_unmatched_place_clears_the_guess(self) -> None:
        """Better no branch than a branch the model imagined."""
        got = reviews.attach_place([self._claim()], {"d1": "g9"}, {})
        assert got[0]["location"] == ""

    def test_the_place_is_recorded(self) -> None:
        got = reviews.attach_place([self._claim()], {"d1": "g1"}, {"g1": "X"})
        assert got[0]["gmap_id"] == "g1"
