#!/usr/bin/env python3
"""Build g_proxy → mental health/psychological outcome association tables.

NLSY79: CES-D depression (2022), Rosenberg self-esteem (1980),
        Rotter locus of control (1979).
NLSY97: CES-D depression (2023).

Binary outcomes modelled as logistic(outcome ~ intercept + g_proxy + age).
Continuous outcomes modelled as OLS(outcome ~ intercept + g_proxy + age).
"""
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

from nls_pipeline.exploratory import g_proxy
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
}

# ---------------------------------------------------------------------------
# Outcome definitions
# ---------------------------------------------------------------------------

MISSING_CODES = [-1, -2, -3, -4, -5]

# NLSY79 binary outcomes
NLSY79_BINARY_OUTCOMES: dict[str, dict[str, Any]] = {
    "depressed": {
        "source_col": "cesd_score_2022",
        "label": "Depressed (CES-D >= 8, 2022)",
        "threshold": 8,
        "direction": ">=",
        "age_expr": "2022 - birth_year",
    },
    "low_self_esteem": {
        "source_col": "rosenberg_self_esteem_1980",
        "label": "Low self-esteem (Rosenberg <= 15, 1980)",
        "threshold": 15,
        "direction": "<=",
        "age_col": "age",
    },
    "external_locus": {
        "source_col": "rotter_locus_control_1979",
        "label": "External locus of control (Rotter >= 12, 1979)",
        "threshold": 12,
        "direction": ">=",
        "age_col": "age",
    },
}

# NLSY79 continuous outcomes
NLSY79_CONTINUOUS_OUTCOMES: dict[str, dict[str, Any]] = {
    "cesd_score": {
        "source_col": "cesd_score_2022",
        "label": "CES-D score (continuous, 2022)",
        "age_expr": "2022 - birth_year",
    },
    "rosenberg_score": {
        "source_col": "rosenberg_self_esteem_1980",
        "label": "Rosenberg self-esteem score (continuous, 1980)",
        "age_col": "age",
    },
    "rotter_score": {
        "source_col": "rotter_locus_control_1979",
        "label": "Rotter locus of control score (continuous, 1979)",
        "age_col": "age",
    },
}

# NLSY97 binary outcomes
NLSY97_BINARY_OUTCOMES: dict[str, dict[str, Any]] = {
    "depressed": {
        "source_col": "cesd_score_2023",
        "label": "Depressed (CES-D >= 8, 2023)",
        "threshold": 8,
        "direction": ">=",
        "age_expr": "2023 - birth_year",
    },
}

# NLSY97 continuous outcomes
NLSY97_CONTINUOUS_OUTCOMES: dict[str, dict[str, Any]] = {
    "cesd_score": {
        "source_col": "cesd_score_2023",
        "label": "CES-D score (continuous, 2023)",
        "age_expr": "2023 - birth_year",
    },
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
    """OLS regression.  Returns (result_dict, None) or (None, reason)."""
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

    try:
        xtx_inv = np.linalg.pinv(xv.T @ xv)
    except np.linalg.LinAlgError:
        return None, "ols_xtx_pinv_failed"

    beta = xtx_inv @ (xv.T @ yv)
    residuals = yv - xv @ beta
    sigma2 = float(np.sum(residuals**2) / (n - p))
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((yv - np.mean(yv))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se

    from scipy.stats import t as t_dist

    p_vals = np.full(p, np.nan)
    for i in range(p):
        if math.isfinite(float(t_stats[i])) and float(se[i]) > 0:
            p_vals[i] = float(2.0 * t_dist.sf(abs(float(t_stats[i])), n - p))

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


def _clean_continuous(series: pd.Series, missing_codes: list[int] | None = None) -> pd.Series:
    """Convert to numeric and set missing codes to NaN."""
    num = pd.to_numeric(series, errors="coerce")
    if missing_codes:
        num = num.where(~num.isin(missing_codes))
    return num


def _binarize_threshold(
    series: pd.Series,
    threshold: int | float,
    direction: str,
    missing_codes: list[int] | None = None,
) -> pd.Series:
    """Binarize a continuous variable at a threshold.

    direction: ">=" means >= threshold → 1, else 0.
               "<=" means <= threshold → 1, else 0.
    """
    num = _clean_continuous(series, missing_codes)
    if direction == ">=":
        return num.map(lambda v: 1.0 if v >= threshold else (0.0 if pd.notna(v) else float("nan")))
    elif direction == "<=":
        return num.map(lambda v: 1.0 if v <= threshold else (0.0 if pd.notna(v) else float("nan")))
    else:
        raise ValueError(f"Unknown direction: {direction}")


def _resolve_age(df: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    """Resolve age series from a spec that has either 'age_col' or 'age_expr'."""
    if "age_col" in spec:
        col = spec["age_col"]
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(float("nan"), index=df.index)
    if "age_expr" in spec:
        expr = spec["age_expr"]
        # Expected form: "YEAR - birth_year"
        parts = expr.split(" - ")
        if len(parts) == 2 and "birth_year" in df.columns:
            year = int(parts[0].strip())
            return year - pd.to_numeric(df["birth_year"], errors="coerce")
        return pd.Series(float("nan"), index=df.index)
    return pd.Series(float("nan"), index=df.index)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def run_mental_health_outcomes(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path = Path("outputs/tables/g_mental_health_outcomes.csv"),
    min_class_n: int = 20,
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    cohort_cfgs = {c: load_yaml(root / COHORT_CONFIGS[c]) for c in cohorts if c in COHORT_CONFIGS}
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    rows: list[dict[str, Any]] = []

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
        missing_codes = cfg.get("sample_construct", {}).get("missing_codes", MISSING_CODES)

        # Compute g_proxy
        indicators = hierarchical_subtests(models_cfg)
        df["__g_proxy"] = g_proxy(df, indicators)

        # ------- NLSY79: CES-D, Rosenberg, Rotter -------
        if cohort == "nlsy79":
            # Binary outcomes
            for outcome_name, spec in NLSY79_BINARY_OUTCOMES.items():
                src_col = spec["source_col"]
                label = spec["label"]
                if src_col not in df.columns:
                    rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{src_col}", source_data, n_total=n_total))
                    continue
                y = _binarize_threshold(df[src_col], spec["threshold"], spec["direction"], missing_codes)
                age_series = _resolve_age(df, spec)
                rows.append(_fit_and_row(cohort, outcome_name, label, y, df["__g_proxy"], age_series, source_data, n_total, min_class_n))

            # Continuous outcomes
            for outcome_name, spec in NLSY79_CONTINUOUS_OUTCOMES.items():
                src_col = spec["source_col"]
                label = spec["label"]
                if src_col not in df.columns:
                    rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{src_col}", source_data, n_total=n_total))
                    continue
                y = _clean_continuous(df[src_col], missing_codes)
                age_series = _resolve_age(df, spec)
                rows.append(_ols_fit_and_row(cohort, outcome_name, label, y, df["__g_proxy"], age_series, source_data, n_total))

        # ------- NLSY97: CES-D only -------
        elif cohort == "nlsy97":
            # Binary outcomes
            for outcome_name, spec in NLSY97_BINARY_OUTCOMES.items():
                src_col = spec["source_col"]
                label = spec["label"]
                if src_col not in df.columns:
                    rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{src_col}", source_data, n_total=n_total))
                    continue
                y = _binarize_threshold(df[src_col], spec["threshold"], spec["direction"], missing_codes)
                age_series = _resolve_age(df, spec)
                rows.append(_fit_and_row(cohort, outcome_name, label, y, df["__g_proxy"], age_series, source_data, n_total, min_class_n))

            # Continuous outcomes
            for outcome_name, spec in NLSY97_CONTINUOUS_OUTCOMES.items():
                src_col = spec["source_col"]
                label = spec["label"]
                if src_col not in df.columns:
                    rows.append(_empty_row(cohort, outcome_name, label, f"missing_column:{src_col}", source_data, n_total=n_total))
                    continue
                y = _clean_continuous(df[src_col], missing_codes)
                age_series = _resolve_age(df, spec)
                rows.append(_ols_fit_and_row(cohort, outcome_name, label, y, df["__g_proxy"], age_series, source_data, n_total))

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
    parser = argparse.ArgumentParser(description="Build g_proxy → mental health outcome association tables.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/g_mental_health_outcomes.csv"))
    parser.add_argument("--min-class-n", type=int, default=20)
    args = parser.parse_args(argv)

    root = Path(args.project_root)
    cohorts = _cohorts_from_args(args)
    out = run_mental_health_outcomes(root=root, cohorts=cohorts, output_path=args.output_path, min_class_n=args.min_class_n)
    computed = int((out["status"] == "computed").sum()) if "status" in out.columns else 0
    print(f"[ok] mental health outcome rows computed: {computed}")
    print(f"[ok] wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
