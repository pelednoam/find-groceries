"""Google ratings, held next to the Reddit verdict rather than merged into it.

Two sources disagree about these stores, and the disagreement is not noise.
Measured over all 135,345 ratings, Google reads **+0.41** more positive than
the Reddit verdict on the same -1..+1 scale — and not evenly. The big chains
people complain about on Reddit are rated fine on Google (Stop & Shop -0.52
vs +0.51, Shaw's -0.49 vs +0.49, Star Market -0.32 vs +0.54) while the
specialty shops agree within a few hundredths (Dave's Fresh Pasta +0.88 vs
+0.91, Reliable Market +0.75 vs +0.77).

Averaging the two would selectively rehabilitate exactly the stores the
corpus is most negative about, and the result would read as new evidence
rather than as a change in source mix. So this module computes both and
leaves them side by side.

The gap has a second layer worth publishing. Restricting Google to reviews
whose author wrote a paragraph — the population closest to a Reddit comment —
drops the mean 0.36 stars overall and up to 1.15 for one chain, halving the
gap to +0.23. People tap five stars on the way out and write prose when
annoyed, so *which* Google number you quote is itself a choice. Both are
carried here rather than one being picked silently.

**Recency.** The Reddit side decays old claims; the Google side must too, or
the comparison quietly favours whichever source is aged more gently. Both are
decayed at the same rate — the evidence-weighted half-life of the Reddit
headline itself, ~4.7y on this corpus, rather than a number picked here.

Decay does two things, and the second matters far more. It re-weights the mean
toward recent reviews, which barely moves anything: ratings drifted only +0.10
stars across 2016-2021, so most stores shift by hundredths. And it shrinks the
*effective* sample, which matters a great deal — this data stops in September
2021, so a location with 100 ratings is carrying nowhere near 100 ratings'
worth of evidence about today, and its error bar should say so.

**Statistics only — no review text.** The underlying data is a research
dataset (McAuley Lab, UCSD) offered for research with citation. Aggregate
counts and means derived from it are a different thing from republishing
strangers' review prose on a public website, and only the former is done
here. Nothing in the output contains a review, a user name, or a user id.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final, TypedDict

from .aggregate import DEFAULT_HALF_LIFE_YEARS, recency_weight

DEFAULT_HALF_LIFE: Final = DEFAULT_HALF_LIFE_YEARS

CITATION: Final = (
    "Google Local review data (to Sept 2021), McAuley Lab, UC San Diego — "
    "Li et al., UCTopic (ACL 2022); Yan et al., Personalized Showcases (2023)"
)

# A mean over fewer than this is a mood, not a rating.
MIN_RATINGS: Final = 20
# A review with a paragraph in it is a different act from tapping four stars.
# Measured here, long reviews run 0.36 stars below the overall mean and up to
# 1.15 below it for one chain: people write at length when annoyed. Both
# numbers are published, because the distance between them says something
# neither says alone.
LONG_REVIEW_CHARS: Final = 120
# Two records of the same store this close together are the same shop. The
# median Google-to-OSM distance is 24m; 150m clears car parks and mall
# entrances without reaching the next branch.
MATCH_METRES: Final = 150.0


class Rating(TypedDict):
    """Aggregate rating for one store or one location. No text, ever.

    `mean_long` covers only reviews whose author wrote a paragraph — the
    population closest to a Reddit comment, and the fairer comparison.
    """

    n: int
    mean: float
    norm: float          # -1..+1, comparable with our sentiment
    #: Sample size after recency decay — what this evidence is worth *now*.
    n_eff: float
    mean_recent: float
    norm_recent: float
    n_long: int
    mean_long: float | None
    norm_long: float | None
    thin: bool
    first: str           # "2016-04"
    last: str
    median_date: str


def normalise(stars: float) -> float:
    """Map a 1..5 mean onto the -1..+1 line our sentiment already uses."""
    return round((stars - 3.0) / 2.0, 3)


def month(epoch_seconds: float) -> str:
    return time.strftime("%Y-%m", time.gmtime(epoch_seconds))


def summarise(
    reviews: Sequence[Mapping[str, Any]],
    now: int | None = None,
    half_life: float = DEFAULT_HALF_LIFE,
) -> Rating | None:
    """Aggregate one bucket of reviews. None when the bucket is empty."""
    return _rating(reviews, now, half_life) if reviews else None


def _rating(
    reviews: Sequence[Mapping[str, Any]],
    now: int | None = None,
    half_life: float = DEFAULT_HALF_LIFE,
) -> Rating:
    """Aggregate a bucket already known to be non-empty."""
    stars = [float(r["rating"]) for r in reviews]
    # The dataset stores milliseconds.
    times = sorted(int(r["time"]) / 1000 for r in reviews)
    mean = statistics.fmean(stars)

    stamp = int(time.time()) if now is None else now
    weights = [
        recency_weight(int(r["time"]) // 1000, stamp, half_life) for r in reviews
    ]
    total_w = sum(weights)
    # Effective sample size, not a count. 27,000 ratings with a 2018 median
    # are not 27,000 ratings' worth of evidence about 2026.
    n_eff = total_w
    mean_recent = (
        sum(w * s for w, s in zip(weights, stars, strict=True)) / total_w
        if total_w > 0 else mean
    )
    long_stars = [
        float(r["rating"]) for r in reviews
        if int(r.get("text_len", 0)) >= LONG_REVIEW_CHARS
    ]
    mean_long = statistics.fmean(long_stars) if long_stars else None
    return Rating(
        n=len(stars),
        mean=round(mean, 2),
        norm=normalise(mean),
        n_eff=round(n_eff, 1),
        mean_recent=round(mean_recent, 2),
        norm_recent=normalise(mean_recent),
        n_long=len(long_stars),
        mean_long=None if mean_long is None else round(mean_long, 2),
        norm_long=None if mean_long is None else normalise(mean_long),
        thin=len(stars) < MIN_RATINGS,
        first=month(times[0]),
        last=month(times[-1]),
        median_date=month(times[len(times) // 2]),
    )


def _bucket(
    reviews: Iterable[Mapping[str, Any]], key: str,
    now: int | None = None, half_life: float = DEFAULT_HALF_LIFE,
) -> dict[str, Rating]:
    """Group by one field and aggregate. Buckets are non-empty by
    construction, so `_rating` is used directly rather than the None-checking
    wrapper — a guard that can never fire reads as if it might."""
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for r in reviews:
        buckets.setdefault(str(r[key]), []).append(r)
    return {
        name: _rating(rows, now, half_life) for name, rows in buckets.items()
    }


def by_store(reviews: Iterable[Mapping[str, Any]], now: int | None = None,
             half_life: float = DEFAULT_HALF_LIFE) -> dict[str, Rating]:
    return _bucket(reviews, "store", now, half_life)


def by_location(reviews: Iterable[Mapping[str, Any]], now: int | None = None,
                half_life: float = DEFAULT_HALF_LIFE) -> dict[str, Rating]:
    return _bucket(reviews, "gmap_id", now, half_life)


def metres(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance. Haversine is ample at city scale."""
    radius = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def match_to_places(
    google: Sequence[Mapping[str, Any]],
    places: Sequence[Mapping[str, Any]],
    limit: float = MATCH_METRES,
) -> dict[str, str]:
    """Map gmap_id -> OSM id for locations that are the same shop.

    Same chain and within `limit` metres. Matching is greedy by distance and
    one-to-one: two Google records must not both claim the same OSM pin, or a
    branch would show one shop's rating twice.
    """
    candidates: list[tuple[float, str, str]] = []
    for g in google:
        for p in places:
            if p["store"] != g["store"]:
                continue
            d = metres(
                float(g["latitude"]), float(g["longitude"]),
                float(p["lat"]), float(p["lon"]),
            )
            if d <= limit:
                candidates.append((d, str(g["gmap_id"]), str(p["osm"])))
    candidates.sort()
    linked: dict[str, str] = {}
    taken: set[str] = set()
    for _d, gmap_id, osm in candidates:
        if gmap_id in linked or osm in taken:
            continue
        linked[gmap_id] = osm
        taken.add(osm)
    return linked


def build(
    reviews: Sequence[Mapping[str, Any]],
    google_places: Sequence[Mapping[str, Any]],
    osm_places: Sequence[Mapping[str, Any]],
    now: int | None = None,
    half_life: float = DEFAULT_HALF_LIFE,
) -> dict[str, Any]:
    """The whole cross-check block, ready to publish."""
    locations = by_location(reviews, now, half_life)
    linked = match_to_places(google_places, osm_places)
    # Keyed by OSM id so the map can look a pin up directly.
    by_osm = {
        osm: locations[gmap_id] for gmap_id, osm in linked.items() if gmap_id in locations
    }
    times = sorted(int(r["time"]) / 1000 for r in reviews)
    return {
        "source": "Google Maps reviews",
        "citation": CITATION,
        "n_reviews": len(reviews),
        "n_locations": len(locations),
        "n_matched_to_map": len(by_osm),
        "coverage": month(times[0]) + " to " + month(times[-1]) if times else "",
        "median_date": month(times[len(times) // 2]) if times else "",
        "half_life_years": round(half_life, 3),
        "stores": by_store(reviews, now, half_life),
        "locations": by_osm,
    }
