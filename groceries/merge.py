"""Combine the Reddit verdict with the Google rating, on one scale.

The two sources rank these stores nearly the same way (r = +0.86) but on very
different scales: Google's spread is 31% of Reddit's, squeezed into the top of
the range by the leniency and ceiling effects that afflict star ratings. So
the correction is not an offset, it is an *affine rescale*:

    reddit_scale(google) = -1.687 + 2.825 x google_norm

Fitted on the 20 stores that have both. Two parameters over twenty points,
leave-one-out RMSE 0.246 against 0.434 for predicting the mean — a 43%
out-of-sample error reduction, so the map is real and not memorisation.

**Why not a literal per-store offset.** Estimating one offset per store from
that store's own disagreement has twenty parameters for twenty points. It
would fit perfectly, drive every residual to zero, and produce a "merged"
number identical to whatever it was anchored on — a merge that cannot
disagree with Reddit is not evidence, it is Reddit with extra steps. The
affine map still corrects every store by a different amount, because the
amount depends on where the store sits; it just spends two degrees of
freedom doing it instead of twenty.

**Where the merge actually earns its keep.** Not at chain level: Reddit has
thousands of claims per chain and the combination barely moves. At *branch*
level only 26% of Reddit claims name a branch at all, while Google has 20+
ratings for 153 of 168 locations. The inverse-variance weighting below
arrives at that on its own — the calibration residual puts a hard floor of
+-0.25 on Google's precision no matter how many ratings it has, so Reddit
wins wherever it is well evidenced and loses where it is thin.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

# Spread of a single claim on the -1..+1 sentiment line. Claims are mostly
# +-1 with some neutrals, so ~1.0; it sets the scale of Reddit's error bar,
# not its centre.
CLAIM_SPREAD: Final = 1.0
# Matches the shrinkage in stage 3, so a cell with no evidence has a wide
# error bar rather than an undefined one.
PRIOR_WEIGHT: Final = 2.0
# Below this many ratings a Google mean is a mood, not a measurement.
MIN_GOOGLE_RATINGS: Final = 20
# Flag when the two sources are further apart than this many combined
# standard errors. At 2.0 it fires on genuine conflict, not on noise.
DISAGREEMENT_SIGMAS: Final = 2.0


@dataclass(frozen=True)
class Calibration:
    """Affine map from Google's normalised rating onto the Reddit scale."""

    intercept: float
    slope: float
    #: Out-of-sample residual sd. This is the floor on Google's precision:
    #: no number of ratings makes a calibrated value better than this.
    residual_sd: float
    n_stores: int
    r2: float
    loo_rmse: float

    def apply(self, google_norm: float) -> float:
        return self.intercept + self.slope * google_norm

    def as_dict(self) -> dict[str, Any]:
        return {
            "intercept": round(self.intercept, 4),
            "slope": round(self.slope, 4),
            "residual_sd": round(self.residual_sd, 4),
            "n_stores": self.n_stores,
            "r2": round(self.r2, 4),
            "loo_rmse": round(self.loo_rmse, 4),
        }


def _ols(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom
    return my - slope * mx, slope


def fit_calibration(pairs: Sequence[tuple[float, float]]) -> Calibration | None:
    """Fit google -> reddit from (google_norm, reddit_sentiment) pairs.

    None below four points: two parameters need degrees of freedom to spare,
    and a line through three points says nothing about the fourth.
    """
    if len(pairs) < 4:
        return None
    goo = [p[0] for p in pairs]
    red = [p[1] for p in pairs]
    intercept, slope = _ols(goo, red)
    resid = [r - (intercept + slope * g) for g, r in zip(goo, red, strict=True)]

    # Leave-one-out, so the reported error is out-of-sample.
    loo: list[float] = []
    for i in range(len(pairs)):
        xs = [goo[j] for j in range(len(pairs)) if j != i]
        ys = [red[j] for j in range(len(pairs)) if j != i]
        a, b = _ols(xs, ys)
        loo.append(red[i] - (a + b * goo[i]))

    var_red = statistics.pvariance(red)
    var_res = statistics.pvariance(resid)
    return Calibration(
        intercept=intercept,
        slope=slope,
        # The honest error bar is the out-of-sample one.
        residual_sd=statistics.pstdev(loo),
        n_stores=len(pairs),
        r2=(1.0 - var_res / var_red) if var_red else 0.0,
        loo_rmse=math.sqrt(statistics.fmean([e * e for e in loo])),
    )


def reddit_variance(weight: float) -> float:
    """Squared standard error of a Reddit cell, from its weighted evidence."""
    return (CLAIM_SPREAD**2) / max(weight + PRIOR_WEIGHT, 1e-9)


def google_variance(rating: Mapping[str, Any], cal: Calibration) -> float:
    """Squared standard error of a calibrated Google value.

    Two terms: the sampling error of the mean, scaled by the calibration
    slope onto the Reddit scale, and the calibration's own residual — which
    dominates everywhere. 27,000 ratings buy a sampling error near zero and
    still leave +-0.25, because what is uncertain is the translation between
    the two scales, not Google's own average.
    """
    # Effective, not raw. This data ends in September 2021, so a location
    # with 100 ratings is not carrying 100 ratings' worth of evidence about
    # today, and the error bar has to say so. At chain scale it changes
    # nothing (27,000 decays to thousands and the sampling term stays
    # negligible); at a branch with 60 old ratings it roughly doubles.
    n = max(float(rating.get("n_eff", rating["n"])), 1.0)
    # Star sd is ~1.2 across this corpus; /2 puts it on the -1..+1 scale.
    sampling = (cal.slope * (1.2 / 2.0) / math.sqrt(n)) ** 2
    return sampling + cal.residual_sd**2


@dataclass(frozen=True)
class Merged:
    """One combined estimate, with both inputs kept visible."""

    value: float
    se: float
    reddit: float | None
    google: float | None
    reddit_share: float
    disagreement: float
    conflicted: bool

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "v": round(self.value, 3),
            "se": round(self.se, 3),
            "share": round(self.reddit_share, 3),
            "gap": round(self.disagreement, 3),
            "conflict": self.conflicted,
        }
        if self.reddit is not None:
            out["r"] = round(self.reddit, 3)
        if self.google is not None:
            out["g"] = round(self.google, 3)
        return out


def combine(
    reddit: float | None,
    reddit_weight: float,
    rating: Mapping[str, Any] | None,
    cal: Calibration | None,
) -> Merged | None:
    """Inverse-variance combination of whichever sources are present."""
    have_google = (
        rating is not None
        and cal is not None
        and float(rating.get("n_eff", rating["n"])) >= MIN_GOOGLE_RATINGS
    )
    if reddit is None and not have_google:
        return None

    if have_google:
        assert rating is not None and cal is not None  # narrowed above
        # The decayed value where it exists: recency should move the number
        # as well as the confidence, even though on this corpus it barely does.
        g_value = cal.apply(float(rating.get("norm_recent", rating["norm"])))
        g_var = google_variance(rating, cal)
    else:
        g_value = g_var = 0.0

    if reddit is None:
        return Merged(g_value, math.sqrt(g_var), None, g_value, 0.0, 0.0, False)
    r_var = reddit_variance(reddit_weight)
    if not have_google:
        return Merged(reddit, math.sqrt(r_var), reddit, None, 1.0, 0.0, False)

    wr, wg = 1.0 / r_var, 1.0 / g_var
    value = (reddit * wr + g_value * wg) / (wr + wg)
    se = math.sqrt(1.0 / (wr + wg))
    gap = g_value - reddit
    # Conflict is measured against the *inputs'* spread, not the combined
    # error bar, which is by construction smaller than either.
    conflicted = abs(gap) > DISAGREEMENT_SIGMAS * math.sqrt(r_var + g_var)
    return Merged(value, se, reddit, g_value, wr / (wr + wg), gap, conflicted)
