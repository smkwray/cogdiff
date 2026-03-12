from __future__ import annotations

import math

import pytest

from nls_pipeline.stats import (
    canonical_d_g,
    canonical_log_vr_g,
    iq_points_from_d,
    mean_diff_ci_iq,
    mean_diff_forest_summary,
    variance_ratio_ci,
    variance_ratio_forest_summary,
)


def test_mean_diff_conversion_edge_cases() -> None:
    assert math.isclose(iq_points_from_d(2.0, iq_sd=10.0), 20.0)
    assert math.isclose(iq_points_from_d(-0.4, iq_sd=12.5), -5.0)
    assert math.isclose(iq_points_from_d(0.0), 0.0)
    with pytest.raises(ValueError, match="finite"):
        iq_points_from_d(math.nan)
    with pytest.raises(ValueError, match="iq_sd"):
        iq_points_from_d(0.1, iq_sd=0.0)


def test_mean_diff_forest_summary_returns_consistent_ci() -> None:
    row = mean_diff_forest_summary("nlsy79", d_g=0.4, se_d_g=0.1, iq_sd=15.0, z=1.96)
    assert row["cohort"] == "nlsy79"
    assert math.isclose(row["estimate"], 6.0)
    assert math.isclose(row["ci_low"], 3.06, rel_tol=1e-3)
    assert math.isclose(row["ci_high"], 8.94, rel_tol=1e-3)


def test_variance_ratio_ci_via_log_matches_formula() -> None:
    low, high = variance_ratio_ci(4.0, 1.0, male_n=25, female_n=49, z=1.96)
    assert math.isclose(low, 2.0, rel_tol=2e-4)
    assert math.isclose(high, 8.0, rel_tol=2e-4)


def test_variance_ratio_ci_forest_summary_with_seeded_se() -> None:
    row = variance_ratio_forest_summary("nlsy97", 9.0, 4.0, se_log_vr=0.2, z=1.96)
    assert row["cohort"] == "nlsy97"
    assert math.isclose(row["estimate"], 2.25)
    assert "se" in row


def test_variance_ratio_bad_inputs_raise_explicit_errors() -> None:
    with pytest.raises(ValueError, match="male_n"):
        variance_ratio_ci(4.0, 1.0, male_n=1, female_n=30, se_log_vr=None)
    with pytest.raises(TypeError, match="male_n"):
        variance_ratio_ci(4.0, 1.0, male_n=1.0, female_n=30)
    with pytest.raises(ValueError, match="Either se_log_vr"):
        variance_ratio_ci(4.0, 1.0)


def test_canonical_d_g_is_orientation_locked_and_reciprocal() -> None:
    male = 100.0
    female = 92.0
    male_var = 16.0
    female_var = 9.0
    forward = canonical_d_g(male, female, male_var, female_var)
    reversed_values = canonical_d_g(female, male, female_var, male_var)
    assert math.isclose(forward, -reversed_values)


def test_canonical_log_vr_g_reciprocal_sign_with_swapped_groups() -> None:
    male_var = 25.0
    female_var = 16.0
    forward = canonical_log_vr_g(male_var, female_var)
    reversed_values = canonical_log_vr_g(female_var, male_var)
    assert math.isclose(forward, -reversed_values)


def test_canonical_estimands_reject_non_finite_or_nonpositive_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_d_g(100.0, 92.0, math.nan, 9.0)
    with pytest.raises(ValueError, match="male_var"):
        canonical_log_vr_g(-1.0, 9.0)
