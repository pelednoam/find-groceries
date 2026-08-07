"""Tests for combining the two sources.

The dangerous failure here is not a crash, it is a merged number that looks
authoritative and is really one source wearing the other's coat. Most of
these check that the combination stays honest: that the calibration has
degrees of freedom to spare, that weight follows evidence, and that a
disagreement survives rather than being averaged into silence.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from groceries import merge


def rating(n: int = 500, norm: float = 0.6) -> dict[str, Any]:
    return {"n": n, "norm": norm}


def line(slope: float, intercept: float, n: int = 20, noise: float = 0.0
         ) -> list[tuple[float, float]]:
    rng = random.Random(7)
    return [
        (g, intercept + slope * g + rng.uniform(-noise, noise))
        for g in [0.4 + 0.025 * i for i in range(n)]
    ]


class TestFitCalibration:
    def test_recovers_a_clean_line(self) -> None:
        cal = merge.fit_calibration(line(2.8, -1.7))
        assert cal is not None
        assert cal.slope == pytest.approx(2.8, abs=1e-6)
        assert cal.intercept == pytest.approx(-1.7, abs=1e-6)
        assert cal.r2 == pytest.approx(1.0, abs=1e-9)

    def test_refuses_to_fit_too_few_points(self) -> None:
        """Two parameters need degrees of freedom to spare; a line through
        three points says nothing about the fourth."""
        assert merge.fit_calibration(line(2.0, 0.0, n=3)) is None
        assert merge.fit_calibration([]) is None
        assert merge.fit_calibration(line(2.0, 0.0, n=4)) is not None

    def test_reports_out_of_sample_error(self) -> None:
        cal = merge.fit_calibration(line(2.8, -1.7, noise=0.3))
        assert cal is not None
        # LOO is the honest number and must not flatter the in-sample fit.
        assert cal.loo_rmse > 0
        assert cal.residual_sd > 0

    def test_noise_shows_up_as_a_worse_fit(self) -> None:
        clean = merge.fit_calibration(line(2.8, -1.7))
        noisy = merge.fit_calibration(line(2.8, -1.7, noise=0.4))
        assert clean is not None and noisy is not None
        assert noisy.r2 < clean.r2
        assert noisy.loo_rmse > clean.loo_rmse

    def test_a_vertical_input_does_not_divide_by_zero(self) -> None:
        cal = merge.fit_calibration([(0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4)])
        assert cal is not None and cal.slope == 0.0

    def test_apply_is_the_fitted_line(self) -> None:
        cal = merge.fit_calibration(line(2.0, -1.0))
        assert cal is not None
        assert cal.apply(0.5) == pytest.approx(0.0, abs=1e-9)

    def test_weights_stop_one_thin_store_tilting_the_line(self) -> None:
        """Broadway Marketplace has 1.9 units of evidence against Market
        Basket's 2,290; unweighted it moved the real slope 2.80 -> 3.05."""
        pairs = line(2.8, -1.7, n=10) + [(0.9, -3.0)]   # one wild outlier
        heavy = [100.0] * 10 + [0.01]
        loose = merge.fit_calibration(pairs)
        tight = merge.fit_calibration(pairs, heavy)
        assert loose is not None and tight is not None
        assert abs(tight.slope - 2.8) < abs(loose.slope - 2.8)
        assert tight.slope == pytest.approx(2.8, abs=0.05)

    def test_one_weighted_point_does_not_crash_the_leave_one_out(self) -> None:
        """Dropping the only point with weight leaves nothing to fit; the
        fold has to return something rather than divide by zero."""
        pairs = line(2.8, -1.7, n=5)
        cal = merge.fit_calibration(pairs, [1.0] + [0.0] * 4)
        assert cal is not None and math.isfinite(cal.loo_rmse)

    def test_zero_total_weight_is_refused(self) -> None:
        assert merge.fit_calibration(line(2.8, -1.7), [0.0] * 20) is None

    def test_mismatched_weights_are_refused(self) -> None:
        assert merge.fit_calibration(line(2.8, -1.7), [1.0, 1.0]) is None

    def test_serialises_for_the_site(self) -> None:
        cal = merge.fit_calibration(line(2.8, -1.7))
        assert cal is not None
        d = cal.as_dict()
        assert set(d) == {"intercept", "slope", "residual_sd", "n_stores",
                          "r2", "loo_rmse"}
        assert d["n_stores"] == 20


class TestVariance:
    def test_more_evidence_narrows_the_reddit_error_bar(self) -> None:
        assert merge.reddit_variance(1000) < merge.reddit_variance(10)
        assert merge.reddit_variance(10) < merge.reddit_variance(1)

    def test_no_evidence_is_wide_not_undefined(self) -> None:
        assert 0 < merge.reddit_variance(0) < math.inf

    def test_google_precision_has_a_floor(self) -> None:
        """This is the load-bearing property. However many ratings back a
        Google mean, the translation onto the Reddit scale is only good to
        the calibration residual — so Google can never swamp a
        well-evidenced Reddit cell."""
        cal = merge.fit_calibration(line(2.8, -1.7, noise=0.3))
        assert cal is not None
        huge = merge.google_variance(rating(n=10_000_000), cal)
        assert huge == pytest.approx(cal.residual_sd**2, rel=0.01)

    def test_more_ratings_still_help_a_little(self) -> None:
        cal = merge.fit_calibration(line(2.8, -1.7, noise=0.3))
        assert cal is not None
        assert merge.google_variance(rating(n=10_000), cal) < merge.google_variance(
            rating(n=25), cal
        )


class TestCombine:
    def _cal(self) -> merge.Calibration:
        cal = merge.fit_calibration(line(2.8, -1.7, noise=0.25))
        assert cal is not None
        return cal

    def test_well_evidenced_reddit_keeps_the_weight(self) -> None:
        # score/valenced_weight = 0.63
        m = merge.combine(1620.0, 2573.0, rating(n=19_764, norm=0.73), self._cal())
        assert m is not None
        assert m.reddit_share > 0.95
        assert m.value == pytest.approx(0.63, abs=0.02)

    def test_a_thin_branch_lets_google_lead(self) -> None:
        m = merge.combine(-0.105, 0.5, rating(n=800, norm=0.79), self._cal())
        assert m is not None
        assert m.reddit_share < 0.3
        assert m.value > 0.0, "the combination should move off the single claim"

    def test_the_share_moves_monotonically_with_evidence(self) -> None:
        cal = self._cal()
        shares = [
            merge.combine(0.2 * w, w, rating(), cal).reddit_share  # type: ignore[union-attr]
            for w in (0.5, 5, 50, 500, 5000)
        ]
        assert shares == sorted(shares)

    def test_reddit_alone_reproduces_stage_three_exactly(self) -> None:
        """The load-bearing invariant. With no second source the combination
        must be the shrunk mean stage 3 already publishes — otherwise the
        merge is applying the neutral prior a second time."""
        score, vw = 40.0, 100.0
        m = merge.combine(score, vw, None, self._cal())
        assert m is not None
        assert m.reddit_share == 1.0 and m.google is None
        assert m.value == pytest.approx(score / (vw + merge.PRIOR_WEIGHT))

    @pytest.mark.parametrize("vw", [0.4, 2.0, 25.0, 400.0, 5000.0])
    def test_the_reduction_holds_at_every_evidence_level(self, vw: float) -> None:
        m = merge.combine(0.7 * vw, vw, None, self._cal())
        assert m is not None
        assert m.value == pytest.approx(0.7 * vw / (vw + merge.PRIOR_WEIGHT))

    def test_google_alone(self) -> None:
        """A branch nobody on Reddit mentioned still has an answer."""
        m = merge.combine(None, 0.0, rating(n=300, norm=0.8), self._cal())
        assert m is not None
        assert m.reddit_share == 0.0 and m.reddit is None and m.google is not None

    def test_neither_source(self) -> None:
        assert merge.combine(None, 0.0, None, self._cal()) is None

    def test_an_all_neutral_cell_counts_as_no_reddit_evidence(self) -> None:
        """valenced_weight 0 means every claim was neutral: there is no mean
        to take, and dividing by it would be a crash or a lie."""
        m = merge.combine(0.0, 0.0, rating(n=300), self._cal())
        assert m is not None and m.reddit is None

    def test_too_few_ratings_are_ignored(self) -> None:
        m = merge.combine(4.0, 10.0, rating(n=merge.MIN_GOOGLE_RATINGS - 1), self._cal())
        assert m is not None and m.google is None and m.reddit_share == 1.0

    def test_no_calibration_means_reddit_only(self) -> None:
        m = merge.combine(4.0, 10.0, rating(), None)
        assert m is not None and m.google is None

    def test_the_combination_lies_between_its_inputs(self) -> None:
        """Shrinkage can pull it toward zero, so the bound is the wider of
        [inputs] and [0, inputs]."""
        cal = self._cal()
        for w in (0.5, 5.0, 500.0):
            m = merge.combine(-0.5 * w, w, rating(n=1000, norm=0.85), cal)
            assert m is not None
            lo = min(m.reddit or 0.0, m.google or 0.0, 0.0)
            hi = max(m.reddit or 0.0, m.google or 0.0, 0.0)
            assert lo - 1e-9 <= m.value <= hi + 1e-9

    def test_combining_never_widens_the_error_bar(self) -> None:
        cal = self._cal()
        m = merge.combine(4.0, 20.0, rating(n=500), cal)
        alone = merge.combine(4.0, 20.0, None, cal)
        assert m is not None and alone is not None
        assert m.se < alone.se

    def test_a_real_conflict_is_flagged_not_hidden(self) -> None:
        """Averaging is only honest when the inputs are compatible. When they
        are not, the number stays but the reader is told."""
        cal = self._cal()
        m = merge.combine(-240.0, 400.0, rating(n=5000, norm=0.9), cal)
        assert m is not None and m.conflicted

    def test_agreement_is_not_flagged(self) -> None:
        cal = self._cal()
        agreeing = cal.apply(0.7)
        m = merge.combine(agreeing * 400.0, 400.0, rating(n=5000, norm=0.7), cal)
        assert m is not None and not m.conflicted

    def test_serialises_with_both_inputs_visible(self) -> None:
        """A merged number that hides its sources cannot be argued with."""
        m = merge.combine(15.0, 50.0, rating(), self._cal())
        assert m is not None
        d = m.as_dict()
        assert {"v", "se", "share", "gap", "conflict", "r", "g"} == set(d)

    def test_absent_inputs_are_omitted_not_zeroed(self) -> None:
        d = merge.combine(15.0, 50.0, None, self._cal()).as_dict()  # type: ignore[union-attr]
        assert "g" not in d and d["r"] == 0.3


class TestCalibrationIsNotCircular:
    def test_a_per_store_offset_would_be_degenerate(self) -> None:
        """The reason the affine map is used instead of one offset per store.

        Fitting an offset from each store's own disagreement drives every
        residual to zero and reproduces the anchor exactly — a merge that
        cannot disagree with Reddit is Reddit with extra steps. Demonstrated
        here so the choice is not merely asserted in a docstring.
        """
        pairs = line(2.8, -1.7, noise=0.4)
        per_store_residuals = [0.0 for _ in pairs]     # offset = gap, by definition
        cal = merge.fit_calibration(pairs)
        assert cal is not None
        affine_residuals = [r - cal.apply(g) for g, r in pairs]
        assert all(e == 0.0 for e in per_store_residuals)
        assert any(abs(e) > 1e-6 for e in affine_residuals), (
            "the affine map must leave residual disagreement; that residual "
            "is the only independent signal the second source contributes"
        )

    def test_the_fit_generalises_to_unseen_stores(self) -> None:
        cal = merge.fit_calibration(line(2.8, -1.7, noise=0.25))
        assert cal is not None
        spread = 2.8 * 0.5 / math.sqrt(12)  # sd of the generated reddit values
        assert cal.loo_rmse < spread, "must beat predicting the mean"
