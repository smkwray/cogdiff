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

COHORT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "cohort": "nlsy79",
        "age_col": "age_2000",
        "continuous": (
            {"outcome": "log_annual_earnings", "source_col": "annual_earnings"},
            {"outcome": "log_household_income", "source_col": "household_income"},
        ),
        "binary": (
            {"outcome": "employment_2000", "source_col": "employment_2000", "binary_kind": "existing_zero_one"},
            {"outcome": "ba_or_more_explicit", "source_col": "highest_degree_ever", "binary_kind": "degree_nlsy79"},
        ),
    },
    {
        "cohort": "nlsy97",
        "age_col": "age_2021",
        "continuous": (
            {"outcome": "log_annual_earnings_2021", "source_col": "annual_earnings_2021"},
            {"outcome": "log_household_income_2021", "source_col": "household_income_2021"},
        ),
        "binary": (
            {"outcome": "employment_2021", "source_col": "employment_2021", "binary_kind": "existing_zero_one"},
            {"outcome": "ba_or_more_explicit", "source_col": "degree_2021", "binary_kind": "degree_nlsy97"},
        ),
    },
)

DETAIL_COLUMNS = [
    "cohort",
    "outcome",
    "outcome_type",
    "model",
    "status",
    "reason",
    "source_col",
    "age_col",
    "n_total",
    "n_used",
    "n_positive",
    "prevalence",
    "threshold_share",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "odds_ratio_g",
    "beta_g_sq",
    "SE_beta_g_sq",
    "p_value_g_sq",
    "beta_threshold",
    "SE_beta_threshold",
    "p_value_threshold",
    "odds_ratio_threshold",
    "fit_stat",
    "mean_outcome",
    "source_data",
]
SUMMARY_COLUMNS = [
    "cohort",
    "outcome",
    "outcome_type",
    "status",
    "reason",
    "source_col",
    "age_col",
    "n_linear",
    "n_quadratic",
    "n_threshold",
    "beta_g_linear",
    "beta_g_quadratic",
    "beta_g_sq",
    "p_value_g_sq",
    "delta_fit_linear_to_quadratic",
    "threshold_share",
    "threshold_beta",
    "threshold_odds_ratio",
    "p_value_threshold",
    "source_data",
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
    p_hat = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-8, 1.0 - 1.0e-8)
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
        "fit_stat": pseudo_r2,
        "n_used": int(n),
    }, None


def _empty_detail(*, cohort: str, outcome: str, outcome_type: str, model: str, source_col: str, age_col: str, reason: str, source_data: str, n_total: int = 0) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "outcome": outcome,
        "outcome_type": outcome_type,
        "model": model,
        "status": "not_feasible",
        "reason": reason,
        "source_col": source_col,
        "age_col": age_col or pd.NA,
        "n_total": int(n_total),
        "source_data": source_data,
    }
    for col in DETAIL_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _empty_summary(*, cohort: str, outcome: str, outcome_type: str, source_col: str, age_col: str, reason: str, source_data: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "outcome": outcome,
        "outcome_type": outcome_type,
        "status": "not_feasible",
        "reason": reason,
        "source_col": source_col,
        "age_col": age_col or pd.NA,
        "source_data": source_data,
    }
    for col in SUMMARY_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _derive_binary(series: pd.Series, kind: str) -> pd.Series:
    raw = pd.to_numeric(series, errors="coerce")
    if kind == "existing_zero_one":
        return raw
    if kind == "degree_nlsy79":
        return raw.isin({3, 4, 5, 6, 7}).astype(float)
    if kind == "degree_nlsy97":
        return raw.isin({5, 6, 7, 8}).astype(float)
    raise ValueError(f"Unsupported binary kind: {kind}")


def _derive_continuous(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.where(values >= 0.0)
    return np.log1p(values)


def _fit_outcome(
    *,
    work: pd.DataFrame,
    outcome_type: str,
    cohort: str,
    outcome: str,
    source_col: str,
    age_col: str,
    source_data: str,
    min_n_continuous: int,
    min_class_n_binary: int,
    threshold_quantile: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    threshold_cut = float(work["g_z"].quantile(threshold_quantile))
    work = work.copy()
    work["g_top"] = (work["g_z"] >= threshold_cut).astype(float)
    threshold_share = float(work["g_top"].mean()) if len(work) > 0 else pd.NA

    if outcome_type == "binary":
        values = sorted(set(float(v) for v in work["outcome"].dropna().tolist()))
        if values != [0.0, 1.0]:
            summary = _empty_summary(cohort=cohort, outcome=outcome, outcome_type=outcome_type, source_col=source_col, age_col=age_col, reason="outcome_not_binary_zero_one", source_data=source_data)
            for model in ("linear", "quadratic", "top_quintile_threshold"):
                row = _empty_detail(cohort=cohort, outcome=outcome, outcome_type=outcome_type, model=model, source_col=source_col, age_col=age_col, reason="outcome_not_binary_zero_one", source_data=source_data, n_total=len(work))
                row["threshold_share"] = threshold_share
                rows.append(row)
            return rows, summary
        n_positive = int(work["outcome"].sum())
        n_negative = int(len(work) - n_positive)
        if n_positive < min_class_n_binary or n_negative < min_class_n_binary:
            summary = _empty_summary(cohort=cohort, outcome=outcome, outcome_type=outcome_type, source_col=source_col, age_col=age_col, reason="insufficient_class_counts", source_data=source_data)
            for model in ("linear", "quadratic", "top_quintile_threshold"):
                row = _empty_detail(cohort=cohort, outcome=outcome, outcome_type=outcome_type, model=model, source_col=source_col, age_col=age_col, reason="insufficient_class_counts", source_data=source_data, n_total=len(work))
                row["n_used"] = int(len(work))
                row["n_positive"] = n_positive
                row["prevalence"] = float(n_positive / len(work)) if len(work) > 0 else pd.NA
                row["threshold_share"] = threshold_share
                rows.append(row)
            return rows, summary
    else:
        if len(work) < min_n_continuous:
            summary = _empty_summary(cohort=cohort, outcome=outcome, outcome_type=outcome_type, source_col=source_col, age_col=age_col, reason="insufficient_rows", source_data=source_data)
            for model in ("linear", "quadratic", "top_quintile_threshold"):
                row = _empty_detail(cohort=cohort, outcome=outcome, outcome_type=outcome_type, model=model, source_col=source_col, age_col=age_col, reason="insufficient_rows", source_data=source_data, n_total=len(work))
                row["n_used"] = int(len(work))
                row["threshold_share"] = threshold_share
                rows.append(row)
            return rows, summary

    model_specs = (
        ("linear", ["intercept", "g_z"]),
        ("quadratic", ["intercept", "g_z", "g_sq"]),
        ("top_quintile_threshold", ["intercept", "g_top"]),
    )
    if age_col:
        model_specs = tuple((name, cols + ["age"]) for name, cols in model_specs)

    for model_name, cols in model_specs:
        x = pd.DataFrame({"intercept": 1.0}, index=work.index)
        if "g_z" in cols:
            x["g_z"] = work["g_z"]
        if "g_sq" in cols:
            x["g_sq"] = work["g_sq"]
        if "g_top" in cols:
            x["g_top"] = work["g_top"]
        if "age" in cols:
            x["age"] = work["age"]

        if outcome_type == "binary":
            fit, reason = _logistic_fit(work["outcome"], x)
        else:
            fit, reason = ols_fit(work["outcome"], x)

        if fit is None:
            row = _empty_detail(cohort=cohort, outcome=outcome, outcome_type=outcome_type, model=model_name, source_col=source_col, age_col=age_col, reason=f"model_failed:{reason or 'unknown'}", source_data=source_data, n_total=len(work))
            row["n_used"] = int(len(work))
            row["threshold_share"] = threshold_share
            rows.append(row)
            continue

        row = {
            "cohort": cohort,
            "outcome": outcome,
            "outcome_type": outcome_type,
            "model": model_name,
            "status": "computed",
            "reason": pd.NA,
            "source_col": source_col,
            "age_col": age_col or pd.NA,
            "n_total": int(len(work)),
            "n_used": int(fit["n_used"]),
            "threshold_share": threshold_share,
            "fit_stat": float(fit["fit_stat"] if "fit_stat" in fit else fit["r2"]),
            "mean_outcome": float(work["outcome"].mean()),
            "source_data": source_data,
        }
        if outcome_type == "binary":
            n_positive = int(work["outcome"].sum())
            row["n_positive"] = n_positive
            row["prevalence"] = float(n_positive / len(work)) if len(work) > 0 else pd.NA
        if model_name == "linear":
            row["beta_g"] = float(fit["beta"][1])
            row["SE_beta_g"] = float(fit["se"][1])
            row["p_value_beta_g"] = float(fit["p"][1]) if math.isfinite(float(fit["p"][1])) else pd.NA
            if outcome_type == "binary":
                row["odds_ratio_g"] = float(math.exp(float(fit["beta"][1])))
        elif model_name == "quadratic":
            row["beta_g"] = float(fit["beta"][1])
            row["SE_beta_g"] = float(fit["se"][1])
            row["p_value_beta_g"] = float(fit["p"][1]) if math.isfinite(float(fit["p"][1])) else pd.NA
            row["beta_g_sq"] = float(fit["beta"][2])
            row["SE_beta_g_sq"] = float(fit["se"][2])
            row["p_value_g_sq"] = float(fit["p"][2]) if math.isfinite(float(fit["p"][2])) else pd.NA
            if outcome_type == "binary":
                row["odds_ratio_g"] = float(math.exp(float(fit["beta"][1])))
        else:
            row["beta_threshold"] = float(fit["beta"][1])
            row["SE_beta_threshold"] = float(fit["se"][1])
            row["p_value_threshold"] = float(fit["p"][1]) if math.isfinite(float(fit["p"][1])) else pd.NA
            if outcome_type == "binary":
                row["odds_ratio_threshold"] = float(math.exp(float(fit["beta"][1])))

        for col in DETAIL_COLUMNS:
            row.setdefault(col, pd.NA)
        rows.append(row)

    details = pd.DataFrame(rows)
    linear = details.loc[details["model"] == "linear"].iloc[0] if (details["model"] == "linear").any() else None
    quadratic = details.loc[details["model"] == "quadratic"].iloc[0] if (details["model"] == "quadratic").any() else None
    threshold = details.loc[details["model"] == "top_quintile_threshold"].iloc[0] if (details["model"] == "top_quintile_threshold").any() else None
    if linear is None or quadratic is None or threshold is None or not all(str(x.get("status", "")) == "computed" for x in (linear, quadratic, threshold)):
        summary = _empty_summary(cohort=cohort, outcome=outcome, outcome_type=outcome_type, source_col=source_col, age_col=age_col, reason="one_or_more_models_not_computed", source_data=source_data)
    else:
        summary = {
            "cohort": cohort,
            "outcome": outcome,
            "outcome_type": outcome_type,
            "status": "computed",
            "reason": pd.NA,
            "source_col": source_col,
            "age_col": age_col or pd.NA,
            "n_linear": int(linear["n_used"]),
            "n_quadratic": int(quadratic["n_used"]),
            "n_threshold": int(threshold["n_used"]),
            "beta_g_linear": float(linear["beta_g"]),
            "beta_g_quadratic": float(quadratic["beta_g"]),
            "beta_g_sq": float(quadratic["beta_g_sq"]),
            "p_value_g_sq": float(quadratic["p_value_g_sq"]) if pd.notna(quadratic["p_value_g_sq"]) else pd.NA,
            "delta_fit_linear_to_quadratic": float(quadratic["fit_stat"]) - float(linear["fit_stat"]),
            "threshold_share": float(threshold["threshold_share"]),
            "threshold_beta": threshold.get("beta_threshold", pd.NA),
            "threshold_odds_ratio": threshold.get("odds_ratio_threshold", pd.NA),
            "p_value_threshold": threshold.get("p_value_threshold", pd.NA),
            "source_data": source_data,
        }
        for col in SUMMARY_COLUMNS:
            summary.setdefault(col, pd.NA)
    return rows, summary


def run_nonlinear_threshold_outcome_models(
    *,
    root: Path,
    detail_output_path: Path = Path("outputs/tables/nonlinear_threshold_outcome_models.csv"),
    summary_output_path: Path = Path("outputs/tables/nonlinear_threshold_outcome_summary.csv"),
    min_n_continuous: int = 200,
    min_class_n_binary: int = 40,
    threshold_quantile: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for cohort_spec in COHORT_SPECS:
        cohort = cohort_spec["cohort"]
        age_col = str(cohort_spec["age_col"] or "")
        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{cohort}_cfa_resid_or_cfa.csv"
        if not source_path.exists():
            for spec in cohort_spec["continuous"]:
                for model in ("linear", "quadratic", "top_quintile_threshold"):
                    detail_rows.append(_empty_detail(cohort=cohort, outcome=spec["outcome"], outcome_type="continuous", model=model, source_col=spec["source_col"], age_col=age_col, reason="missing_source_data", source_data=source_data))
                summary_rows.append(_empty_summary(cohort=cohort, outcome=spec["outcome"], outcome_type="continuous", source_col=spec["source_col"], age_col=age_col, reason="missing_source_data", source_data=source_data))
            for spec in cohort_spec["binary"]:
                for model in ("linear", "quadratic", "top_quintile_threshold"):
                    detail_rows.append(_empty_detail(cohort=cohort, outcome=spec["outcome"], outcome_type="binary", model=model, source_col=spec["source_col"], age_col=age_col, reason="missing_source_data", source_data=source_data))
                summary_rows.append(_empty_summary(cohort=cohort, outcome=spec["outcome"], outcome_type="binary", source_col=spec["source_col"], age_col=age_col, reason="missing_source_data", source_data=source_data))
            continue

        df = pd.read_csv(source_path, low_memory=False)
        indicators = hierarchical_subtests(models_cfg)
        df = df.copy()
        df["__g_proxy"] = g_proxy(df, indicators)

        for spec in cohort_spec["continuous"]:
            source_col = spec["source_col"]
            outcome = spec["outcome"]
            if source_col not in df.columns or (age_col and age_col not in df.columns):
                reason = "missing_required_columns"
                for model in ("linear", "quadratic", "top_quintile_threshold"):
                    detail_rows.append(_empty_detail(cohort=cohort, outcome=outcome, outcome_type="continuous", model=model, source_col=source_col, age_col=age_col, reason=reason, source_data=source_data, n_total=len(df)))
                summary_rows.append(_empty_summary(cohort=cohort, outcome=outcome, outcome_type="continuous", source_col=source_col, age_col=age_col, reason=reason, source_data=source_data))
                continue
            work = pd.DataFrame(
                {
                    "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
                    "age": pd.to_numeric(df[age_col], errors="coerce") if age_col else pd.NA,
                    "outcome": _derive_continuous(df[source_col]),
                }
            ).dropna()
            if work.empty:
                reason = "no_valid_rows"
                for model in ("linear", "quadratic", "top_quintile_threshold"):
                    detail_rows.append(_empty_detail(cohort=cohort, outcome=outcome, outcome_type="continuous", model=model, source_col=source_col, age_col=age_col, reason=reason, source_data=source_data, n_total=len(df)))
                summary_rows.append(_empty_summary(cohort=cohort, outcome=outcome, outcome_type="continuous", source_col=source_col, age_col=age_col, reason=reason, source_data=source_data))
                continue
            g_sd = float(work["g"].std(ddof=1))
            if not math.isfinite(g_sd) or g_sd <= 0:
                reason = "degenerate_g_proxy"
                for model in ("linear", "quadratic", "top_quintile_threshold"):
                    detail_rows.append(_empty_detail(cohort=cohort, outcome=outcome, outcome_type="continuous", model=model, source_col=source_col, age_col=age_col, reason=reason, source_data=source_data, n_total=len(df)))
                summary_rows.append(_empty_summary(cohort=cohort, outcome=outcome, outcome_type="continuous", source_col=source_col, age_col=age_col, reason=reason, source_data=source_data))
                continue
            work["g_z"] = (work["g"] - float(work["g"].mean())) / g_sd
            work["g_sq"] = work["g_z"] ** 2
            rows, summary = _fit_outcome(
                work=work,
                outcome_type="continuous",
                cohort=cohort,
                outcome=outcome,
                source_col=source_col,
                age_col=age_col,
                source_data=source_data,
                min_n_continuous=min_n_continuous,
                min_class_n_binary=min_class_n_binary,
                threshold_quantile=threshold_quantile,
            )
            detail_rows.extend(rows)
            summary_rows.append(summary)

        for spec in cohort_spec["binary"]:
            source_col = spec["source_col"]
            outcome = spec["outcome"]
            if source_col not in df.columns or (age_col and age_col not in df.columns):
                reason = "missing_required_columns"
                for model in ("linear", "quadratic", "top_quintile_threshold"):
                    detail_rows.append(_empty_detail(cohort=cohort, outcome=outcome, outcome_type="binary", model=model, source_col=source_col, age_col=age_col, reason=reason, source_data=source_data, n_total=len(df)))
                summary_rows.append(_empty_summary(cohort=cohort, outcome=outcome, outcome_type="binary", source_col=source_col, age_col=age_col, reason=reason, source_data=source_data))
                continue
            work = pd.DataFrame(
                {
                    "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
                    "age": pd.to_numeric(df[age_col], errors="coerce") if age_col else pd.NA,
                    "outcome": _derive_binary(df[source_col], spec["binary_kind"]),
                }
            ).dropna()
            if work.empty:
                reason = "no_valid_rows"
                for model in ("linear", "quadratic", "top_quintile_threshold"):
                    detail_rows.append(_empty_detail(cohort=cohort, outcome=outcome, outcome_type="binary", model=model, source_col=source_col, age_col=age_col, reason=reason, source_data=source_data, n_total=len(df)))
                summary_rows.append(_empty_summary(cohort=cohort, outcome=outcome, outcome_type="binary", source_col=source_col, age_col=age_col, reason=reason, source_data=source_data))
                continue
            g_sd = float(work["g"].std(ddof=1))
            if not math.isfinite(g_sd) or g_sd <= 0:
                reason = "degenerate_g_proxy"
                for model in ("linear", "quadratic", "top_quintile_threshold"):
                    detail_rows.append(_empty_detail(cohort=cohort, outcome=outcome, outcome_type="binary", model=model, source_col=source_col, age_col=age_col, reason=reason, source_data=source_data, n_total=len(df)))
                summary_rows.append(_empty_summary(cohort=cohort, outcome=outcome, outcome_type="binary", source_col=source_col, age_col=age_col, reason=reason, source_data=source_data))
                continue
            work["g_z"] = (work["g"] - float(work["g"].mean())) / g_sd
            work["g_sq"] = work["g_z"] ** 2
            rows, summary = _fit_outcome(
                work=work,
                outcome_type="binary",
                cohort=cohort,
                outcome=outcome,
                source_col=source_col,
                age_col=age_col,
                source_data=source_data,
                min_n_continuous=min_n_continuous,
                min_class_n_binary=min_class_n_binary,
                threshold_quantile=threshold_quantile,
            )
            detail_rows.extend(rows)
            summary_rows.append(summary)

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    for col in DETAIL_COLUMNS:
        if col not in detail.columns:
            detail[col] = pd.NA
    for col in SUMMARY_COLUMNS:
        if col not in summary.columns:
            summary[col] = pd.NA
    detail = detail[DETAIL_COLUMNS].copy()
    summary = summary[SUMMARY_COLUMNS].copy()

    detail_target = detail_output_path if detail_output_path.is_absolute() else root / detail_output_path
    summary_target = summary_output_path if summary_output_path.is_absolute() else root / summary_output_path
    detail_target.parent.mkdir(parents=True, exist_ok=True)
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_target, index=False)
    summary.to_csv(summary_target, index=False)
    return detail, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build nonlinear and threshold versions of core adult outcome models.")
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--detail-output-path", type=Path, default=Path("outputs/tables/nonlinear_threshold_outcome_models.csv"))
    parser.add_argument("--summary-output-path", type=Path, default=Path("outputs/tables/nonlinear_threshold_outcome_summary.csv"))
    parser.add_argument("--min-n-continuous", type=int, default=200)
    parser.add_argument("--min-class-n-binary", type=int, default=40)
    parser.add_argument("--threshold-quantile", type=float, default=0.8)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        detail, summary = run_nonlinear_threshold_outcome_models(
            root=root,
            detail_output_path=args.detail_output_path,
            summary_output_path=args.summary_output_path,
            min_n_continuous=int(args.min_n_continuous),
            min_class_n_binary=int(args.min_class_n_binary),
            threshold_quantile=float(args.threshold_quantile),
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.detail_output_path if args.detail_output_path.is_absolute() else root / args.detail_output_path}")
    print(f"[ok] wrote {args.summary_output_path if args.summary_output_path.is_absolute() else root / args.summary_output_path}")
    print(f"[ok] computed detail rows: {int((detail['status'] == 'computed').sum()) if 'status' in detail.columns else 0}")
    print(f"[ok] computed summary rows: {int((summary['status'] == 'computed').sum()) if 'status' in summary.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
