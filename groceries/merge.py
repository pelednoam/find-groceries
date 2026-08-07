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

**One prior, applied once.** Stage 3 publishes `sentiment = score / (vw + k)`
— already pulled toward neutral, hardest where evidence is thinnest. Feeding
that damped number into an inverse-variance combination against an undamped
Google value penalises thin evidence twice, and does it worst exactly where
the merge is supposed to help: measured across the 83 merged branches it
biased the result by up to 0.20. So `combine` works from the *raw* weighted
mean `score / vw` and applies the pull once, to the combined estimate, using
the combined precision. With Google absent the arithmetic collapses back to
`score / (vw + k)` — the merge is a strict generalisation of the shrinkage
stage 3 already does, not a second helping of it.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

# Spread of a single claim on the -1..+1 sentiment line. Claims are mostly
# +-1 with some neutrals, so ~1.0; it sets the scale of Reddit's error bar,
# not its centre.
CLAIM_SPREAD: Final = 1.0
# The pull toward neutral, applied once to the *combination*. Same constant
# as stage 3's, so that with Google absent this reduces to exactly the
# sentiment stage 3 already publishes — see `combine`.
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


def _ols(
    xs: Sequence[float], ys: Sequence[float], ws: Sequence[float]
) -> tuple[float, float]:
    """Weighted least squares."""
    total = sum(ws)
    if total <= 0:
        return 0.0, 0.0
    mx = sum(x * w for x, w in zip(xs, ws, strict=True)) / total
    my = sum(y * w for y, w in zip(ys, ws, strict=True)) / total
    denom = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws, strict=True))
    if denom == 0:
        return my, 0.0
    slope = sum(
        w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws, strict=True)
    ) / denom
    return my - slope * mx, slope


def fit_calibration(
    pairs: Sequence[tuple[float, float]],
    weights: Sequence[float] | None = None,
) -> Calibration | None:
    """Fit google -> reddit from (google_norm, reddit_mean) pairs.

    `weights` should be each store's Reddit precision. Without them one
    barely-evidenced store swings the line: Broadway Marketplace has a
    valenced weight of 1.9 against Market Basket's 2,290, and unweighted it
    moved the slope from 2.80 to 3.05 on its own. The same effect is larger
    at branch level, where unweighted gives 1.75 and weighted 2.85.

    None below four points: two parameters need degrees of freedom to spare,
    and a line through three points says nothing about the fourth.
    """
    if len(pairs) < 4:
        return None
    goo = [p[0] for p in pairs]
    red = [p[1] for p in pairs]
    ws = list(weights) if weights is not None else [1.0] * len(pairs)
    if len(ws) != len(pairs) or sum(ws) <= 0:
        return None
    intercept, slope = _ols(goo, red, ws)
    resid = [r - (intercept + slope * g) for g, r in zip(goo, red, strict=True)]

    # Leave-one-out, so the reported error is out-of-sample.
    loo: list[float] = []
    for i in range(len(pairs)):
        keep = [j for j in range(len(pairs)) if j != i]
        a, b = _ols([goo[j] for j in keep], [red[j] for j in keep],
                    [ws[j] for j in keep])
        loo.append(red[i] - (a + b * goo[i]))

    total = sum(ws)
    wmean = sum(r * w for r, w in zip(red, ws, strict=True)) / total
    var_red = sum(w * (r - wmean) ** 2 for r, w in zip(red, ws, strict=True)) / total
    var_res = sum(w * e**2 for e, w in zip(resid, ws, strict=True)) / total
    # Weighted too: the error that matters is the error at stores whose true
    # value is actually known.
    loo_mse = sum(w * e**2 for e, w in zip(loo, ws, strict=True)) / total
    return Calibration(
        intercept=intercept,
        slope=slope,
        # The honest error bar is the out-of-sample one.
        residual_sd=math.sqrt(loo_mse),
        n_stores=len(pairs),
        r2=(1.0 - var_res / var_red) if var_red else 0.0,
        loo_rmse=math.sqrt(loo_mse),
    )


def reddit_variance(valenced_weight: float) -> float:
    """Squared standard error of the *raw* Reddit mean.

    No prior term: the pull toward neutral is applied to the combination in
    `combine`, so including it here would apply it twice.
    """
    return (CLAIM_SPREAD**2) / max(valenced_weight, 1e-9)


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
    score: float | None,
    valenced_weight: float,
    rating: Mapping[str, Any] | None,
    cal: Calibration | None,
) -> Merged | None:
    """Combine whichever sources are present, then shrink once.

    `score` and `valenced_weight` are stage 3's raw ingredients, not its
    published `sentiment`: the raw mean is `score / valenced_weight`, and the
    pull toward neutral is applied at the end so it lands on the combination
    rather than on one input.
    """
    have_reddit = score is not None and valenced_weight > 0
    have_google = (
        rating is not None
        and cal is not None
        and float(rating.get("n_eff", rating["n"])) >= MIN_GOOGLE_RATINGS
    )
    if not have_reddit and not have_google:
        return None

    precision = 0.0
    weighted_sum = 0.0
    r_value: float | None = None
    g_value: float | None = None
    r_var = g_var = math.inf

    if have_reddit:
        assert score is not None
        r_value = score / valenced_weight
        r_var = reddit_variance(valenced_weight)
        precision += 1.0 / r_var
        weighted_sum += r_value / r_var
    if have_google:
        assert rating is not None and cal is not None
        # The decayed value where it exists: recency should move the number
        # as well as the confidence, even though on this corpus it barely does.
        g_value = cal.apply(float(rating.get("norm_recent", rating["norm"])))
        g_var = google_variance(rating, cal)
        precision += 1.0 / g_var
        weighted_sum += g_value / g_var

    raw = weighted_sum / precision
    # The one prior. With Google absent this is score/(vw + k) exactly.
    shrunk = raw * precision / (precision + PRIOR_WEIGHT)
    se = math.sqrt(1.0 / (precision + PRIOR_WEIGHT))
    share = (1.0 / r_var) / precision if have_reddit else 0.0

    gap = 0.0
    conflicted = False
    if have_reddit and have_google:
        assert r_value is not None and g_value is not None
        gap = g_value - r_value
        # Measured against the inputs' own spread, not the combined error
        # bar, which is by construction smaller than either.
        conflicted = abs(gap) > DISAGREEMENT_SIGMAS * math.sqrt(r_var + g_var)
    return Merged(shrunk, se, r_value, g_value, share, gap, conflicted)
