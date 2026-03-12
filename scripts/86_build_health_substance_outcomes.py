#!/usr/bin/env python3
"""Build g_proxy → health and substance use outcome association tables.

NLSY79: self-rated health (age 40+), smoking (2018), marijuana use (1984),
        obesity (height 1985 / weight 2022), alcohol days (2018).
NLSY97: self-rated health (2023), marijuana use (2015), binge drinking (2023),
        obesity (2011), alcohol days (2023).

Binary outcomes are modelled as logistic(outcome ~ intercept + g_proxy + age).
Continuous outcomes are modelled as OLS(outcome ~ intercept + g_proxy + age).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.exploratory import g_proxy
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
}

OUTPUT_COLUMNS = [
    "cohort",
    "outcome",
    "label",
    "status",
    "reason",
    "n_total",
    "n_used",
    "n_positive",
    "prevalence",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "odds_ratio_g",
    "beta_age",
    "SE_beta_age",
    "p_value_beta_age",
    "pseudo_r2",
    "source_data",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _empty_row(
    cohort: str,
    outcome: str,
    label: str,
    reason: str,
    source_data: str,
    *,
    n_total: int = 0,
    n_used: int = 0,
    n_positive: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "outcome": outcome,
        "label": label,
        "status": "not_feasible",
        "reason": reason,
        "n_total": int(n_total),
        "n_used": int(n_used),
        "n_positive": int(n_positive),
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _logistic_fit(
    y: pd.Series, x: pd.DataFrame, *, max_iter: int = 100, tol: float = 1e-8
) -> tuple[dict[str, Any] | None, str | None]:
    """IRLS logistic regression.  Returns (result_dict, None) or (None, reason)."""
    y_num = pd.to_numeric(y, errors="coerce")
    x_num = x.apply(pd.to_numeric, errors="coerce")
    mask = y_num.notna() & x_num.notna().all(axis=1)
    if int(mask.sum()) == 0:
        return None, "empty_after_numeric_cast"

    yv = y_num[mask].to_numpy(dtype=float)
    xv = x_num[mask].to_numpy(dtype=float)
    n, p = xv.shape
    if n <= p:
        return None, "insufficient_rows_for_model"
    if np.unique(yv).size < 2:
        return None, "single_outcome_class"

    beta = np.zeros(p, dtype=float)
    converged = False
    for _ in range(max_iter):
        eta = np.clip(xv @ beta, -30.0, 30.0)
        p_hat = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p_hat * (1.0 - p_hat), 1e-8, None)
        xtwx = xv.T @ (w[:, None] * xv)
        grad = xv.T @ (yv - p_hat)
        try:
            step = np.linalg.pinv(xtwx) @ grad
        except np.linalg.LinAlgError:
            return None, "logit_xtwx_pinv_failed"
        beta_next = beta + step
        if float(np.max(np.abs(step))) < tol:
            beta = beta_next
            converged = True
            break
        beta = beta_next

    if not converged:
        return None, "logit_failed_to_converge"

    eta = np.clip(xv @ beta, -30.0, 30.0)
    p_hat = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-8, 1.0 - 1e-8)
    w = np.clip(p_hat * (1.0 - p_hat), 1e-8, None)
    xtwx = xv.T @ (w[:, None] * xv)
    try:
        cov = np.linalg.pinv(xtwx)
    except np.linalg.LinAlgError:
        return None, "logit_covariance_failed"
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_stats = beta / se
    p_vals = np.full(shape=(p,), fill_value=np.nan, dtype=float)
    for i in range(p):
        if math.isfinite(float(z_stats[i])) and math.isfinite(float(se[i])) and float(se[i]) > 0.0:
            p_vals[i] = float(2.0 * norm.sf(abs(float(z_stats[i]))))

    ll_full = float(np.sum(yv * np.log(p_hat) + (1.0 - yv) * np.log(1.0 - p_hat)))
    y_bar = float(np.mean(yv))
    if y_bar <= 0.0 or y_bar >= 1.0:
        pseudo_r2 = float("nan")
    else:
        ll_null = float(np.sum(yv * math.log(y_bar) + (1.0 - yv) * math.log(1.0 - y_bar)))
        pseudo_r2 = float(1.0 - (ll_full / ll_null)) if ll_null != 0.0 else float("nan")

    return {
        "beta": beta,
        "se": se,
        "p": p_vals,
        "pseudo_r2": pseudo_r2,
        "n_used": int(n),
    }, None


def _ols_fit(
    y: pd.Series, x: pd.DataFrame
) -> tuple[dict[str, Any] | None, str | None]:
    """OLS regression via normal equations.  Returns (result_dict, None) or (None, reason)."""
    y_num = pd.to_numeric(y, errors="coerce")
    x_num = x.apply(pd.to_numeric, errors="coerce")
    mask = y_num.notna() & x_num.notna().all(axis=1)
    if int(mask.sum()) == 0:
        return None, "empty_after_numeric_cast"

    yv = y_num[mask].to_numpy(dtype=float)
    xv = x_num[mask].to_numpy(dtype=float)
    n, p = xv.shape
    if n <= p:
        return None, "insufficient_rows_for_model"

    # Normal equations: beta = (X'X)^-1 X'y
    xtx = xv.T @ xv
    try:
        xtx_inv = np.linalg.pinv(xtx)
    except np.linalg.LinAlgError:
        return None, "ols_xtx_pinv_failed"

    beta = xtx_inv @ (xv.T @ yv)
    residuals = yv - xv @ beta
    sigma2 = float(np.sum(residuals ** 2) / (n - p))

    # Covariance matrix: sigma^2 * (X'X)^-1
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    # t-statistics and p-values
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
    df = n - p
    p_vals = np.full(shape=(p,), fill_value=np.nan, dtype=float)
    for i in range(p):
        if math.isfinite(float(t_stats[i])) and math.isfinite(float(se[i])) and float(se[i]) > 0.0:
            p_vals[i] = float(2.0 * t_dist.sf(abs(float(t_stats[i])), df))

    # R-squared
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0.0 else float("nan")

    return {
        "beta": beta,
        "se": se,
        "p": p_vals,
        "r2": r2,
        "n_used": int(n),
    }, None


def _fit_and_row(
    cohort: str,
    outcome: str,
    label: str,
    y: pd.Series,
    g: pd.Series,
    age: pd.Series,
    source_data: str,
    n_total: int,
    min_class_n: int,
) -> dict[str, Any]:
    """Fit logistic model and return a result row."""
    work = pd.DataFrame({"y": y, "g": g, "age": age}).dropna()
    if work.empty:
        return _empty_row(cohort, outcome, label, "no_valid_rows", source_data, n_total=n_total)

    unique_vals = sorted(set(float(v) for v in work["y"].dropna().unique()))
    if unique_vals != [0.0, 1.0]:
        return _empty_row(cohort, outcome, label, "outcome_not_binary", source_data, n_total=n_total, n_used=len(work))

    n_used = len(work)
    n_positive = int(work["y"].sum())
    n_negative = n_used - n_positive
    if n_positive < min_class_n or n_negative < min_class_n:
        return _empty_row(
            cohort, outcome, label, "insufficient_class_counts", source_data,
            n_total=n_total, n_used=n_used, n_positive=n_positive,
        )

    x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age": work["age"]}, index=work.index)
    fit, reason = _logistic_fit(work["y"], x)
    if fit is None:
        return _empty_row(
            cohort, outcome, label, f"logit_failed:{reason or 'unknown'}", source_data,
            n_total=n_total, n_used=n_used, n_positive=n_positive,
        )

    beta, se, p_vals = fit["beta"], fit["se"], fit["p"]
    row: dict[str, Any] = {
        "cohort": cohort,
        "outcome": outcome,
        "label": label,
        "status": "computed",
        "reason": pd.NA,
        "n_total": n_total,
        "n_used": n_used,
        "n_positive": n_positive,
        "prevalence": float(n_positive / n_used) if n_used > 0 else pd.NA,
        "beta_g": float(beta[1]),
        "SE_beta_g": float(se[1]),
        "p_value_beta_g": float(p_vals[1]) if math.isfinite(float(p_vals[1])) else pd.NA,
        "odds_ratio_g": float(math.exp(float(beta[1]))) if math.isfinite(float(beta[1])) else pd.NA,
        "beta_age": float(beta[2]),
        "SE_beta_age": float(se[2]),
        "p_value_beta_age": float(p_vals[2]) if math.isfinite(float(p_vals[2])) else pd.NA,
        "pseudo_r2": float(fit["pseudo_r2"]) if math.isfinite(float(fit["pseudo_r2"])) else pd.NA,
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _ols_fit_and_row(
    cohort: str,
    outcome: str,
    label: str,
    y: pd.Series,
    g: pd.Series,
    age: pd.Series,
    source_data: str,
    n_total: int,
) -> dict[str, Any]:
    """Fit OLS model and return a result row."""
    work = pd.DataFrame({"y": y, "g": g, "age": age}).dropna()
    if work.empty:
        return _empty_row(cohort, outcome, label, "no_valid_rows", source_data, n_total=n_total)

    n_used = len(work)

    x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age": work["age"]}, index=work.index)
    fit, reason = _ols_fit(work["y"], x)
    if fit is None:
        return _empty_row(
            cohort, outcome, label, f"ols_failed:{reason or 'unknown'}", source_data,
            n_total=n_total, n_used=n_used,
        )

    beta, se, p_vals = fit["beta"], fit["se"], fit["p"]
    row: dict[str, Any] = {
        "cohort": cohort,
        "outcome": outcome,
        "label": label,
        "status": "computed",
        "reason": pd.NA,
        "n_total": n_total,
        "n_used": n_used,
        "n_positive": pd.NA,
        "prevalence": pd.NA,
        "beta_g": float(beta[1]),
        "SE_beta_g": float(se[1]),
        "p_value_beta_g": float(p_vals[1]) if math.isfinite(float(p_vals[1])) else pd.NA,
        "odds_ratio_g": pd.NA,
        "beta_age": float(beta[2]),
        "SE_beta_age": float(se[2]),
        "p_value_beta_age": float(p_vals[2]) if math.isfinite(float(p_vals[2])) else pd.NA,
        "pseudo_r2": float(fit["r2"]) if math.isfinite(float(fit["r2"])) else pd.NA,
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


# ---------------------------------------------------------------------------
# Binarization helpers
# ---------------------------------------------------------------------------

def _clean_missing(series: pd.Series, missing_codes: list[int] | None = None) -> pd.Series:
    """Convert to numeric and set missing codes to NaN."""
    num = pd.to_numeric(series, errors="coerce")
    if missing_codes:
        num = num.where(~num.isin(missing_codes))
    return num


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def run_health_substance_outcomes(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path = Path("outputs/tables/g_health_substance_outcomes.csv"),
    min_class_n: int = 20,
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    cohort_cfgs = {c: load_yaml(root / COHORT_CONFIGS[c]) for c in cohorts if c in COHORT_CONFIGS}
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    rows: list[dict[str, Any]] = []
    missing_codes = [-1, -2, -3, -4, -5]

    for cohort in cohorts:
        if cohort not in COHORT_CONFIGS:
            rows.append(_empty_row(cohort, "all", "", "cohort_not_configured", ""))
            continue

        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{cohort}_cfa_resid_or_cfa.csv"
        if not source_path.exists():
            rows.append(_empty_row(cohort, "all", "", "missing_source_data", source_data))
            continue

        df = pd.read_csv(source_path, low_memory=False)
        n_total = len(df)

        # Get missing codes from cohort config
        cfg = cohort_cfgs.get(cohort, {})
        missing_codes = cfg.get("sample_construct", {}).get("missing_codes", [-1, -2, -3, -4, -5])

        # Compute g_proxy
        indicators = hierarchical_subtests(models_cfg)
        df["__g_proxy"] = g_proxy(df, indicators)

        # ------- NLSY79 -------
        if cohort == "nlsy79":
            birth_year = pd.to_numeric(df.get("birth_year", pd.Series(dtype=float)), errors="coerce")

            # --- poor_health: health_status_40plus, 4-5 → 1, 1-3 → 0 ---
            outcome_name = "poor_health"
            label = "Fair/poor self-rated health (age 40+)"
            src_col = "health_status_40plus"
            if src_col not in df.columns:
                rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{src_col}", source_data, n_total=n_total))
            else:
                raw = _clean_missing(df[src_col], missing_codes)
                y = raw.map(lambda v: 1.0 if v >= 4 else (0.0 if 1 <= v <= 3 else float("nan")))
                age_series = 2000 - birth_year
                rows.append(_fit_and_row(cohort, outcome_name, label, y, df["__g_proxy"], age_series, source_data, n_total, min_class_n))

            # --- current_smoker: smoking_daily_2018, 1-2 → 1, 3 → 0 ---
            outcome_name = "current_smoker"
            label = "Current smoker (2018)"
            src_col = "smoking_daily_2018"
            if src_col not in df.columns:
                rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{src_col}", source_data, n_total=n_total))
            else:
                raw = _clean_missing(df[src_col], missing_codes)
                y = raw.map(lambda v: 1.0 if v in (1, 2) else (0.0 if v == 3 else float("nan")))
                age_series = 2018 - birth_year
                rows.append(_fit_and_row(cohort, outcome_name, label, y, df["__g_proxy"], age_series, source_data, n_total, min_class_n))

            # --- any_marijuana: marijuana_use_30_1984, >0 → 1, 0 → 0 ---
            outcome_name = "any_marijuana"
            label = "Any marijuana use past 30 days (1984)"
            src_col = "marijuana_use_30_1984"
            if src_col not in df.columns:
                rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{src_col}", source_data, n_total=n_total))
            else:
                raw = _clean_missing(df[src_col], missing_codes)
                y = raw.map(lambda v: 1.0 if v > 0 else (0.0 if v == 0 else float("nan")))
                # Use age column directly
                if "age" in df.columns:
                    age_series = pd.to_numeric(df["age"], errors="coerce")
                else:
                    age_series = pd.Series(float("nan"), index=df.index)
                rows.append(_fit_and_row(cohort, outcome_name, label, y, df["__g_proxy"], age_series, source_data, n_total, min_class_n))

            # --- obese: BMI from height_inches_1985 and weight_pounds_2022, >=30 → 1 ---
            outcome_name = "obese"
            label = "Obese BMI >= 30 (height 1985, weight 2022)"
            h_col = "height_inches_1985"
            w_col = "weight_pounds_2022"
            if h_col not in df.columns or w_col not in df.columns:
                missing = [c for c in (h_col, w_col) if c not in df.columns]
                rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{','.join(missing)}", source_data, n_total=n_total))
            else:
                height = _clean_missing(df[h_col], missing_codes)
                weight = _clean_missing(df[w_col], missing_codes)
                # Filter implausible values
                height = height.where((height >= 48) & (height <= 84))
                weight = weight.where((weight >= 70) & (weight <= 600))
                bmi = (weight * 703) / (height ** 2)
                y = pd.Series(float("nan"), index=df.index)
                valid_bmi = bmi.dropna()
                y.loc[valid_bmi.index] = valid_bmi.apply(lambda v: 1.0 if v >= 30 else 0.0)
                age_series = 2022 - birth_year
                rows.append(_fit_and_row(cohort, outcome_name, label, y, df["__g_proxy"], age_series, source_data, n_total, min_class_n))

            # --- alcohol_days_30: continuous OLS ---
            outcome_name = "alcohol_days_30"
            label = "Alcohol days past 30 (2018, continuous)"
            src_col = "alcohol_days_30_2018"
            if src_col not in df.columns:
                rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{src_col}", source_data, n_total=n_total))
            else:
                raw = _clean_missing(df[src_col], missing_codes)
                age_series = 2018 - birth_year
                rows.append(_ols_fit_and_row(cohort, outcome_name, label, raw, df["__g_proxy"], age_series, source_data, n_total))

        # ------- NLSY97 -------
        elif cohort == "nlsy97":
            birth_year = pd.to_numeric(df.get("birth_year", pd.Series(dtype=float)), errors="coerce")

            # --- poor_health: health_status_2023, 4-5 → 1, 1-3 → 0 ---
            outcome_name = "poor_health"
            label = "Fair/poor self-rated health (2023)"
            src_col = "health_status_2023"
            if src_col not in df.columns:
                rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{src_col}", source_data, n_total=n_total))
            else:
                raw = _clean_missing(df[src_col], missing_codes)
                y = raw.map(lambda v: 1.0 if v >= 4 else (0.0 if 1 <= v <= 3 else float("nan")))
                age_series = 2023 - birth_year
                rows.append(_fit_and_row(cohort, outcome_name, label, y, df["__g_proxy"], age_series, source_data, n_total, min_class_n))

            # --- any_marijuana: marijuana_days_30_2015, >0 → 1, 0 → 0 ---
            outcome_name = "any_marijuana"
            label = "Any marijuana use past 30 days (2015)"
            src_col = "marijuana_days_30_2015"
            if src_col not in df.columns:
                rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{src_col}", source_data, n_total=n_total))
            else:
                raw = _clean_missing(df[src_col], missing_codes)
                y = raw.map(lambda v: 1.0 if v > 0 else (0.0 if v == 0 else float("nan")))
                age_series = 2015 - birth_year
                rows.append(_fit_and_row(cohort, outcome_name, label, y, df["__g_proxy"], age_series, source_data, n_total, min_class_n))

            # --- binge_drinker: binge_days_30_2023, >0 → 1, 0 → 0 ---
            outcome_name = "binge_drinker"
            label = "Any binge drinking past 30 days (2023)"
            src_col = "binge_days_30_2023"
            if src_col not in df.columns:
                rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{src_col}", source_data, n_total=n_total))
            else:
                raw = _clean_missing(df[src_col], missing_codes)
                y = raw.map(lambda v: 1.0 if v > 0 else (0.0 if v == 0 else float("nan")))
                age_series = 2023 - birth_year
                rows.append(_fit_and_row(cohort, outcome_name, label, y, df["__g_proxy"], age_series, source_data, n_total, min_class_n))

            # --- obese: BMI from height_feet_2011, height_inches_2011, weight_pounds_2011 ---
            outcome_name = "obese"
            label = "Obese BMI >= 30 (2011)"
            hf_col = "height_feet_2011"
            hi_col = "height_inches_2011"
            w_col = "weight_pounds_2011"
            needed = [hf_col, hi_col, w_col]
            missing_cols = [c for c in needed if c not in df.columns]
            if missing_cols:
                rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{','.join(missing_cols)}", source_data, n_total=n_total))
            else:
                h_feet = _clean_missing(df[hf_col], missing_codes)
                h_inches = _clean_missing(df[hi_col], missing_codes)
                weight = _clean_missing(df[w_col], missing_codes)
                total_height = h_feet * 12 + h_inches
                # Filter implausible values
                total_height = total_height.where((total_height >= 48) & (total_height <= 84))
                weight = weight.where((weight >= 70) & (weight <= 600))
                bmi = (weight * 703) / (total_height ** 2)
                y = pd.Series(float("nan"), index=df.index)
                valid_bmi = bmi.dropna()
                y.loc[valid_bmi.index] = valid_bmi.apply(lambda v: 1.0 if v >= 30 else 0.0)
                age_series = 2011 - birth_year
                rows.append(_fit_and_row(cohort, outcome_name, label, y, df["__g_proxy"], age_series, source_data, n_total, min_class_n))

            # --- alcohol_days_30: continuous OLS ---
            outcome_name = "alcohol_days_30"
            label = "Alcohol days past 30 (2023, continuous)"
            src_col = "alcohol_days_30_2023"
            if src_col not in df.columns:
                rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{src_col}", source_data, n_total=n_total))
            else:
                raw = _clean_missing(df[src_col], missing_codes)
                age_series = 2023 - birth_year
                rows.append(_ols_fit_and_row(cohort, outcome_name, label, raw, df["__g_proxy"], age_series, source_data, n_total))

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=OUTPUT_COLUMNS)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS].copy()

    output_full = root / output_path
    output_full.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_full, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build g_proxy → health/substance outcome association tables.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/g_health_substance_outcomes.csv"))
    parser.add_argument("--min-class-n", type=int, default=20)
    args = parser.parse_args(argv)

    root = Path(args.project_root)
    cohorts = _cohorts_from_args(args)
    out = run_health_substance_outcomes(root=root, cohorts=cohorts, output_path=args.output_path, min_class_n=args.min_class_n)
    computed = int((out["status"] == "computed").sum()) if "status" in out.columns else 0
    print(f"[ok] health/substance outcome rows computed: {computed}")
    print(f"[ok] wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
