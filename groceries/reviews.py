"""Turn Google reviews into documents stage 2 can extract from.

Star ratings gave one number per shop, which is why the merge could only
touch the overall verdict. Running the review *text* through the same
extractor as Reddit gives per-category claims instead — and, because it is
the same prompt and the same schema, those land on the same sentiment scale
by construction. The affine calibration in `merge.py` exists to undo star
compression; it has no business anywhere near a claim-derived number.

What remains is a *population* difference: Google reviewers are self-selected
customers of the shop they are rating, Reddit threads are comparative and
skew toward complaint. Extraction does not remove that, but it does make it
measurable for the first time, because both sides are finally the same
instrument pointed at different crowds.

Three differences from a Reddit document, all of which make this easier:

* The store is known, not guessed. It comes from the gmap_id, so stage 1's
  matching problem does not arise.
* The branch is known too. Stage 2 will still guess a `location` from the
  prose; `attach_place` overwrites it with the truth afterwards.
* There is no thread. No parent, no comparative context, so `comparator_store`
  will rarely fire and that is correct rather than a miss.

**No review text is published.** These documents exist on disk and go to the
model; what comes back is aggregated to numbers. Neither the review prose nor
the model's paraphrase of it reaches the site — see `scripts/build_site.py`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any, Final

from .types import Candidate, SourcedClaim

# Matches the cross-check's definition, so "substantive" means one thing in
# both places: below this a review is "Great store!" and costs a request to
# learn nothing.
MIN_CHARS: Final = 120
# Long reviews exist; the Reddit path caps at 6000 for the same reason.
MAX_CHARS: Final = 6000
# `subreddit` and `kind` are stage 2's provenance fields. Reusing them keeps
# one Candidate shape across sources; these values make the origin obvious in
# a claims file and keep doc keys from colliding with Reddit's.
SOURCE: Final = "google"
KIND: Final = "review"


def review_id(review: Mapping[str, Any]) -> str:
    """Stable id for one review.

    The dataset has no review id. gmap_id plus author plus timestamp is
    unique in practice and, being a digest, carries no author identity into
    the claims file.
    """
    material = f"{review.get('gmap_id')}|{review.get('user_id')}|{review.get('time')}"
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def to_candidate(review: Mapping[str, Any], store: str) -> Candidate | None:
    """One review as a stage-2 document, or None if it is not worth a request."""
    text = (review.get("text") or "").strip()
    if len(text) < MIN_CHARS:
        return None
    # The dataset stores milliseconds.
    created = int(review["time"]) // 1000
    return Candidate(
        id=review_id(review),
        subreddit=SOURCE,
        kind=KIND,
        created_utc=created,
        # Reviews carry no community score. None is honest; 0 would read as
        # "downvoted" to `vote_weight`.
        score=None,
        author="",
        parent_body="",
        # A review has no public permalink of its own. The place URL is the
        # nearest true thing, and nothing here is published anyway.
        permalink=f"https://www.google.com/maps/place/?q=place_id:{review['gmap_id']}",
        stores=[store],
        text=text[:MAX_CHARS],
        truncated=len(text) > MAX_CHARS,
    )


def candidates(
    reviews: Iterable[Mapping[str, Any]], stores: Mapping[str, str]
) -> list[Candidate]:
    """Every substantive review whose place maps to a known store."""
    out: list[Candidate] = []
    for review in reviews:
        store = stores.get(str(review.get("gmap_id")))
        if store is None:
            continue
        candidate = to_candidate(review, store)
        if candidate is not None:
            out.append(candidate)
    return out


def attach_place(
    claims: Iterable[SourcedClaim],
    doc_to_place: Mapping[str, str],
    branches: Mapping[str, str],
    reviewed: Mapping[str, str],
) -> tuple[list[SourcedClaim], int, int]:
    """Replace the model's guessed `location` with the branch we already know.

    Stage 2 reads a branch out of the prose because that is all a Reddit
    comment offers. Here the shop is a fact of the record, so the guess is
    strictly worse than the truth — and a wrong branch silently splits one
    store's evidence in two.

    `reviewed` maps a document to the store it is a review *of*. A review
    says several things about its own shop and occasionally one about a
    rival; 1,399 such claims exist in this corpus, and giving them the
    reviewed shop's address filed Trader Joe's evidence at a Whole Foods
    one. Those are left unplaced instead.

    Returns (claims, n_placed, n_about_another_chain).
    """
    out: list[SourcedClaim] = []
    placed = foreign = 0
    for claim in claims:
        doc = claim["source_id"]
        updated = dict(claim)
        if claim.get("store") != reviewed.get(doc):
            foreign += 1
            updated["location"] = ""
            updated["gmap_id"] = ""
        else:
            place = doc_to_place.get(doc, "")
            branch = branches.get(place, "")
            updated["location"] = branch
            updated["gmap_id"] = place
            if branch:
                placed += 1
        out.append(updated)  # type: ignore[arg-type]
    return out, placed, foreign
