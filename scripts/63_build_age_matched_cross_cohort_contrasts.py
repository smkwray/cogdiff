#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.exploratory import g_proxy, ols_fit
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
}

OVERLAP_TARGETS: tuple[tuple[str, str, str, str], ...] = (
    ("household_income", "age_2000", "age_2019", "household_income_2019"),
    ("household_income", "age_2000", "age_2021", "household_income_2021"),
    ("employment", "age_2000", "age_2019", "employment_2019"),
    ("employment", "age_2000", "age_2021", "employment_2021"),
    ("annual_earnings", "age_2000", "age_2019", "annual_earnings_2019"),
    ("annual_earnings", "age_2000", "age_2021", "annual_earnings_2021"),
)

ESTIMATE_COLUMNS = [
    "outcome",
    "cohort",
    "age_col",
    "outcome_col",
    "status",
    "reason",
    "overlap_min",
    "overlap_max",
    "n_total",
    "n_age_window",
    "n_used",
    "model_type",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "r2_or_pseudo_r2",
    "source_data",
]

CONTRAST_COLUMNS = [
    "outcome",
    "cohort_a",
    "age_col_a",
    "cohort_b",
    "age_col_b",
    "status",
    "reason",
    "overlap_min",
    "overlap_max",
    "beta_a",
    "beta_b",
    "diff_b_minus_a",
    "SE_diff",
    "z_diff",
    "p_value_diff",
    "source_estimates",
]


def _logistic_fit(y: pd.Series, x: pd.DataFrame, *, max_iter: int = 100, tol: float = 1e-8) -> tuple[dict[str, Any] | None, str | None]:
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
    cov = np.linalg.pinv(xtwx)
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


def _empty_estimate(
    outcome: str,
    cohort: str,
    age_col: str,
    outcome_col: str,
    reason: str,
    overlap_min: float | None,
    overlap_max: float | None,
    source_data: str,
    *,
    n_total: int = 0,
    n_age_window: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "outcome": outcome,
        "cohort": cohort,
        "age_col": age_col,
        "outcome_col": outcome_col,
        "status": "not_feasible",
        "reason": reason,
        "overlap_min": overlap_min,
        "overlap_max": overlap_max,
        "n_total": int(n_total),
        "n_age_window": int(n_age_window),
        "n_used": 0,
        "source_data": source_data,
    }
    for col in ESTIMATE_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _fit_one(
    *,
    df: pd.DataFrame,
    cohort: str,
    age_col: str,
    outcome: str,
    outcome_col: str,
    overlap_min: float,
    overlap_max: float,
    source_data: str,
) -> dict[str, Any]:
    if age_col not in df.columns or outcome_col not in df.columns or "__g_proxy" not in df.columns:
        return _empty_estimate(outcome, cohort, age_col, outcome_col, "missing_required_columns", overlap_min, overlap_max, source_data, n_total=len(df))

    ages = pd.to_numeric(df[age_col], errors="coerce")
    in_window = ages.ge(overlap_min) & ages.le(overlap_max)
    window = df.loc[in_window].copy()
    if window.empty:
        return _empty_estimate(outcome, cohort, age_col, outcome_col, "empty_age_window", overlap_min, overlap_max, source_data, n_total=len(df))

    work = pd.DataFrame(
        {
            "age": pd.to_numeric(window[age_col], errors="coerce"),
            "g": pd.to_numeric(window["__g_proxy"], errors="coerce"),
            "outcome": pd.to_numeric(window[outcome_col], errors="coerce"),
        }
    ).dropna()
    if work.empty:
        return _empty_estimate(outcome, cohort, age_col, outcome_col, "no_valid_rows_after_cleaning", overlap_min, overlap_max, source_data, n_total=len(df), n_age_window=len(window))

    if outcome == "employment":
        unique_outcomes = sorted(set(float(v) for v in work["outcome"].unique().tolist()))
        if unique_outcomes != [0.0, 1.0]:
            return _empty_estimate(outcome, cohort, age_col, outcome_col, "outcome_not_binary_zero_one", overlap_min, overlap_max, source_data, n_total=len(df), n_age_window=len(window))
        x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age": work["age"]}, index=work.index)
        fit, reason = _logistic_fit(work["outcome"], x)
        if fit is None:
            return _empty_estimate(outcome, cohort, age_col, outcome_col, f"logit_failed:{reason or 'unknown'}", overlap_min, overlap_max, source_data, n_total=len(df), n_age_window=len(window))
        row = {
            "outcome": outcome,
            "cohort": cohort,
            "age_col": age_col,
            "outcome_col": outcome_col,
            "status": "computed",
            "reason": pd.NA,
            "overlap_min": overlap_min,
            "overlap_max": overlap_max,
            "n_total": int(len(df)),
            "n_age_window": int(len(window)),
            "n_used": int(fit["n_used"]),
            "model_type": "logit",
            "beta_g": float(fit["beta"][1]),
            "SE_beta_g": float(fit["se"][1]),
            "p_value_beta_g": float(fit["p"][1]),
            "r2_or_pseudo_r2": float(fit["pseudo_r2"]),
            "source_data": source_data,
        }
    else:
        x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age": work["age"]}, index=work.index)
        fit, reason = ols_fit(work["outcome"], x)
        if fit is None:
            return _empty_estimate(outcome, cohort, age_col, outcome_col, f"ols_failed:{reason or 'unknown'}", overlap_min, overlap_max, source_data, n_total=len(df), n_age_window=len(window))
        row = {
            "outcome": outcome,
            "cohort": cohort,
            "age_col": age_col,
            "outcome_col": outcome_col,
            "status": "computed",
            "reason": pd.NA,
            "overlap_min": overlap_min,
            "overlap_max": overlap_max,
            "n_total": int(len(df)),
            "n_age_window": int(len(window)),
            "n_used": int(fit["n_used"]),
            "model_type": "ols",
            "beta_g": float(fit["beta"][1]),
            "SE_beta_g": float(fit["se"][1]),
            "p_value_beta_g": float(fit["p"][1]),
            "r2_or_pseudo_r2": float(fit["r2"]),
            "source_data": source_data,
        }
    for col in ESTIMATE_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_age_matched_cross_cohort_contrasts(
    *,
    root: Path,
    estimates_output_path: Path = Path("outputs/tables/age_matched_outcome_validity.csv"),
    contrasts_output_path: Path = Path("outputs/tables/age_matched_cross_cohort_contrasts.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    dfs: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, str] = {}
    for cohort in COHORT_CONFIGS:
        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        source_paths[cohort] = str(source_path.relative_to(root)) if source_path.exists() else f"{cohort}_cfa_resid_or_cfa.csv"
        if not source_path.exists():
            dfs[cohort] = pd.DataFrame()
            continue
        df = pd.read_csv(source_path, low_memory=False)
        indicators = hierarchical_subtests(models_cfg)
        df = df.copy()
        df["__g_proxy"] = g_proxy(df, indicators)
        dfs[cohort] = df

    estimate_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for outcome, age_col_79, age_col_97, outcome_col_97 in OVERLAP_TARGETS:
        df79 = dfs.get("nlsy79", pd.DataFrame())
        df97 = dfs.get("nlsy97", pd.DataFrame())
        source79 = source_paths["nlsy79"]
        source97 = source_paths["nlsy97"]
        if df79.empty or df97.empty or age_col_79 not in df79.columns or age_col_97 not in df97.columns:
            est79 = _empty_estimate(outcome, "nlsy79", age_col_79, outcome if outcome != "employment" else "employment_2000", "missing_source_or_age_column", None, None, source79)
            est97 = _empty_estimate(outcome, "nlsy97", age_col_97, outcome_col_97, "missing_source_or_age_column", None, None, source97)
            estimate_rows.extend([est79, est97])
            continue

        ages79 = pd.to_numeric(df79[age_col_79], errors="coerce")
        ages97 = pd.to_numeric(df97[age_col_97], errors="coerce")
        overlap_min = max(float(ages79.min()), float(ages97.min()))
        overlap_max = min(float(ages79.max()), float(ages97.max()))
        if overlap_min > overlap_max:
            est79 = _empty_estimate(outcome, "nlsy79", age_col_79, outcome if outcome != "employment" else "employment_2000", "no_age_overlap", overlap_min, overlap_max, source79, n_total=len(df79))
            est97 = _empty_estimate(outcome, "nlsy97", age_col_97, outcome_col_97, "no_age_overlap", overlap_min, overlap_max, source97, n_total=len(df97))
            estimate_rows.extend([est79, est97])
            continue

        outcome_col_79 = {
            "household_income": "household_income",
            "employment": "employment_2000",
            "annual_earnings": "annual_earnings",
        }[outcome]
        est79 = _fit_one(df=df79, cohort="nlsy79", age_col=age_col_79, outcome=outcome, outcome_col=outcome_col_79, overlap_min=overlap_min, overlap_max=overlap_max, source_data=source79)
        est97 = _fit_one(df=df97, cohort="nlsy97", age_col=age_col_97, outcome=outcome, outcome_col=outcome_col_97, overlap_min=overlap_min, overlap_max=overlap_max, source_data=source97)
        estimate_rows.extend([est79, est97])

        if est79["status"] != "computed" or est97["status"] != "computed":
            contrast_rows.append(
                {
                    "outcome": outcome,
                    "cohort_a": "nlsy79",
                    "age_col_a": age_col_79,
                    "cohort_b": "nlsy97",
                    "age_col_b": age_col_97,
                    "status": "not_feasible",
                    "reason": "missing_computed_estimate",
                    "overlap_min": overlap_min,
                    "overlap_max": overlap_max,
                    "source_estimates": f"{source79},{source97}",
                }
            )
            continue

        se_diff = math.sqrt(float(est79["SE_beta_g"]) ** 2 + float(est97["SE_beta_g"]) ** 2)
        diff = float(est97["beta_g"]) - float(est79["beta_g"])
        z = diff / se_diff if se_diff > 0.0 else float("nan")
        p = float(2.0 * norm.sf(abs(z))) if math.isfinite(z) else float("nan")
        contrast_rows.append(
            {
                "outcome": outcome,
                "cohort_a": "nlsy79",
                "age_col_a": age_col_79,
                "cohort_b": "nlsy97",
                "age_col_b": age_col_97,
                "status": "computed",
                "reason": pd.NA,
                "overlap_min": overlap_min,
                "overlap_max": overlap_max,
                "beta_a": float(est79["beta_g"]),
                "beta_b": float(est97["beta_g"]),
                "diff_b_minus_a": diff,
                "SE_diff": se_diff,
                "z_diff": z,
                "p_value_diff": p,
                "source_estimates": f"{source79},{source97}",
            }
        )

    estimates = pd.DataFrame(estimate_rows)
    if estimates.empty:
        estimates = pd.DataFrame(columns=ESTIMATE_COLUMNS)
    for col in ESTIMATE_COLUMNS:
        if col not in estimates.columns:
            estimates[col] = pd.NA
    estimates = estimates[ESTIMATE_COLUMNS].copy()

    contrasts = pd.DataFrame(contrast_rows)
    if contrasts.empty:
        contrasts = pd.DataFrame(columns=CONTRAST_COLUMNS)
    for col in CONTRAST_COLUMNS:
        if col not in contrasts.columns:
            contrasts[col] = pd.NA
    contrasts = contrasts[CONTRAST_COLUMNS].copy()

    estimates_full = root / estimates_output_path
    estimates_full.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(estimates_full, index=False)

    contrasts_full = root / contrasts_output_path
    contrasts_full.parent.mkdir(parents=True, exist_ok=True)
    contrasts.to_csv(contrasts_full, index=False)
    return estimates, contrasts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build age-matched cross-cohort contrasts using overlapping NLSY79/NLSY97 outcome windows.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--estimates-output-path", type=Path, default=Path("outputs/tables/age_matched_outcome_validity.csv"))
    parser.add_argument("--contrasts-output-path", type=Path, default=Path("outputs/tables/age_matched_cross_cohort_contrasts.csv"))
    args = parser.parse_args(argv)

    root = Path(args.project_root)
    estimates, contrasts = run_age_matched_cross_cohort_contrasts(
        root=root,
        estimates_output_path=args.estimates_output_path,
        contrasts_output_path=args.contrasts_output_path,
    )
    print(f"[ok] age-matched estimate rows computed: {int((estimates['status'] == 'computed').sum())}")
    print(f"[ok] age-matched contrast rows computed: {int((contrasts['status'] == 'computed').sum())}")
    print(f"[ok] wrote {args.estimates_output_path}")
    print(f"[ok] wrote {args.contrasts_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
