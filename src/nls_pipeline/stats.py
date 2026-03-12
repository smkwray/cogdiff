from __future__ import annotations

import math
from typing import Any


def _require_finite(value: float | int, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _require_positive(value: float | int, name: str) -> float:
    out = _require_finite(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return out


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int > 1.")
    if value <= 1:
        raise ValueError(f"{name} must be > 1.")
    return value


def canonical_d_g(male_mean: float, female_mean: float, male_var: float, female_var: float) -> float:
    male_mean_f = _require_finite(male_mean, "male_mean")
    female_mean_f = _require_finite(female_mean, "female_mean")
    male_var_f = _require_positive(male_var, "male_var")
    female_var_f = _require_positive(female_var, "female_var")
    pooled_sd = math.sqrt((male_var_f + female_var_f) / 2.0)
    return (male_mean_f - female_mean_f) / pooled_sd


def canonical_log_vr_g(male_var: float, female_var: float) -> float:
    male_var_f = _require_positive(male_var, "male_var")
    female_var_f = _require_positive(female_var, "female_var")
    return math.log(male_var_f / female_var_f)


def iq_points_from_d(d_g: float, iq_sd: float = 15.0) -> float:
    d_g_f = _require_finite(d_g, "d_g")
    iq_sd_f = _require_positive(iq_sd, "iq_sd")
    return d_g_f * iq_sd_f


def mean_diff_ci_iq(d_g: float, se_d_g: float, iq_sd: float = 15.0, z: float = 1.96) -> tuple[float, float]:
    d_g_f = _require_finite(d_g, "d_g")
    se_d_g_f = _require_positive(se_d_g, "se_d_g")
    iq_sd_f = _require_positive(iq_sd, "iq_sd")
    z_f = _require_positive(z, "z")
    low = (d_g_f - z_f * se_d_g_f) * iq_sd_f
    high = (d_g_f + z_f * se_d_g_f) * iq_sd_f
    return low, high


def mean_diff_forest_summary(
    cohort: str,
    d_g: float,
    se_d_g: float,
    iq_sd: float = 15.0,
    z: float = 1.96,
) -> dict[str, float | str]:
    d_g_f = _require_finite(d_g, "d_g")
    se_d_g_f = _require_positive(se_d_g, "se_d_g")
    iq_sd_f = _require_positive(iq_sd, "iq_sd")
    ci_low, ci_high = mean_diff_ci_iq(d_g_f, se_d_g_f, iq_sd=iq_sd_f, z=z)
    return {
        "cohort": cohort,
        "estimate": d_g_f * iq_sd_f,
        "se": se_d_g_f * iq_sd_f,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _resolve_se_log_vr(
    *,
    male_n: int | None,
    female_n: int | None,
    se_log_vr: float | None,
) -> float:
    if se_log_vr is not None:
        return _require_positive(se_log_vr, "se_log_vr")
    if male_n is None or female_n is None:
        raise ValueError("Either se_log_vr or both male_n and female_n must be provided.")
    male_n_i = _require_positive_int(male_n, "male_n")
    female_n_i = _require_positive_int(female_n, "female_n")
    return math.sqrt(2.0 / (male_n_i - 1) + 2.0 / (female_n_i - 1))


def variance_ratio_ci(
    male_var: float,
    female_var: float,
    *,
    male_n: int | None = None,
    female_n: int | None = None,
    se_log_vr: float | None = None,
    z: float = 1.96,
) -> tuple[float, float]:
    male_var_f = _require_positive(male_var, "male_var")
    female_var_f = _require_positive(female_var, "female_var")
    z_f = _require_positive(z, "z")
    se = _resolve_se_log_vr(male_n=male_n, female_n=female_n, se_log_vr=se_log_vr)
    center = math.log(male_var_f / female_var_f)
    return math.exp(center - z_f * se), math.exp(center + z_f * se)


def variance_ratio_forest_summary(
    cohort: str,
    male_var: float,
    female_var: float,
    *,
    male_n: int | None = None,
    female_n: int | None = None,
    se_log_vr: float | None = None,
    z: float = 1.96,
) -> dict[str, float | str]:
    male_var_f = _require_positive(male_var, "male_var")
    female_var_f = _require_positive(female_var, "female_var")
    se = _resolve_se_log_vr(male_n=male_n, female_n=female_n, se_log_vr=se_log_vr)
    ci_low, ci_high = variance_ratio_ci(
        male_var_f,
        female_var_f,
        male_n=male_n,
        female_n=female_n,
        se_log_vr=se,
        z=z,
    )
    return {
        "cohort": cohort,
        "estimate": male_var_f / female_var_f,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }

