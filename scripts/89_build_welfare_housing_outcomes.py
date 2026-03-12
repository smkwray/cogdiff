#!/usr/bin/env python3
"""Build g_proxy → welfare receipt and homeownership outcome association tables.

NLSY79: year-2000 snapshot (respondents ~36-43) — food stamps, AFDC, any welfare,
        and homeownership.
NLSY97: 2019 snapshot (respondents ~35-39) — government program income receipt,
        and homeownership at ~age 40.

Each outcome is modelled as logistic(outcome ~ intercept + g_proxy + age).
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

# NLSY79 — year 2000 snapshot (respondents ~36-43)
NLSY79_OUTCOMES: dict[str, dict[str, Any]] = {
    "received_food_stamps": {
        "source_col": "food_stamps_amount_2000",
        "label": "Received food stamps (2000)",
        "binarize": "positive_amount",  # >0 → 1, 0 → 0
    },
    "received_afdc": {
        "source_col": "afdc_amount_2000",
        "label": "Received AFDC (2000)",
        "binarize": "positive_amount",
    },
    "any_welfare": {
        "source_cols": ["food_stamps_amount_2000", "afdc_amount_2000"],
        "label": "Any welfare receipt (food stamps or AFDC, 2000)",
        "binarize": "any_positive_amount",
    },
    "homeowner": {
        "source_col": "homeowner_2000",
        "label": "Homeowner (2000)",
        "binarize": "already_binary",  # 1=Yes, 0=No
    },
}

# NLSY97 — 2019 snapshot (respondents ~35-39)
NLSY97_OUTCOMES: dict[str, dict[str, Any]] = {
    "received_govt_program_income": {
        "source_col": "govt_program_income_2019",
        "label": "Received govt program income (TANF/SSI/SSDI/etc., 2019)",
        "binarize": "already_binary",  # 1=Yes, 0=No
    },
    "homeowner": {
        "source_col": "house_type_40",
        "label": "Homeowner (~age 40)",
        "binarize": "house_type_40",  # 1 → 1, 6 → 0, else NaN
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


# ---------------------------------------------------------------------------
# Binarization helpers
# ---------------------------------------------------------------------------

def _binarize_positive_amount(series: pd.Series, missing_codes: list[int]) -> pd.Series:
    """Convert continuous dollar amount to binary: >0 → 1, 0 → 0.  Missing codes → NaN."""
    num = pd.to_numeric(series, errors="coerce")
    num = num.where(~num.isin(missing_codes))
    return num.map(lambda v: 1.0 if v > 0 else (0.0 if v == 0 else float("nan")))


def _binarize_already_binary(series: pd.Series, missing_codes: list[int]) -> pd.Series:
    """Clean an already-binary 1/0 variable.  Missing codes → NaN."""
    num = pd.to_numeric(series, errors="coerce")
    num = num.where(~num.isin(missing_codes))
    return num.map(lambda v: 1.0 if v == 1 else (0.0 if v == 0 else float("nan")))


def _binarize_house_type_40(series: pd.Series, missing_codes: list[int]) -> pd.Series:
    """Binarize house_type_40: 1 → 1 (homeowner), 6 → 0 (not owner), else NaN."""
    num = pd.to_numeric(series, errors="coerce")
    num = num.where(~num.isin(missing_codes))
    return num.map(lambda v: 1.0 if v == 1 else (0.0 if v == 6 else float("nan")))


def _binarize_any_positive_amount(
    df: pd.DataFrame, cols: list[str], missing_codes: list[int]
) -> pd.Series:
    """1 if any of the given dollar-amount columns is > 0, 0 if all are 0, NaN if all missing."""
    binaries = [_binarize_positive_amount(df[c], missing_codes) for c in cols if c in df.columns]
    if not binaries:
        return pd.Series(float("nan"), index=df.index)
    combined = pd.concat(binaries, axis=1)
    has_one = (combined == 1.0).any(axis=1)
    all_nan = combined.isna().all(axis=1)
    result = pd.Series(0.0, index=df.index)
    result[has_one] = 1.0
    result[all_nan] = float("nan")
    return result


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def run_welfare_housing_outcomes(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path = Path("outputs/tables/g_welfare_housing_outcomes.csv"),
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

        # ------- NLSY79: year-2000 welfare and homeownership -------
        if cohort == "nlsy79":
            # Derive age at 2000 snapshot
            if "age_2000" in df.columns:
                age_series = pd.to_numeric(df["age_2000"], errors="coerce")
            elif "birth_year" in df.columns:
                age_series = 2000 - pd.to_numeric(df["birth_year"], errors="coerce")
            else:
                age_series = pd.Series(float("nan"), index=df.index)

            for outcome_name, spec in NLSY79_OUTCOMES.items():
                label = spec["label"]
                binarize = spec["binarize"]

                if binarize == "any_positive_amount":
                    # Composite: any of multiple amount columns > 0
                    cols = spec["source_cols"]
                    available = [c for c in cols if c in df.columns]
                    if not available:
                        rows.append(_empty_row(
                            cohort, outcome_name, label,
                            f"missing_columns:{','.join(cols)}", source_data,
                            n_total=n_total,
                        ))
                        continue
                    y = _binarize_any_positive_amount(df, available, missing_codes)
                else:
                    src_col = spec["source_col"]
                    if src_col not in df.columns:
                        rows.append(_empty_row(
                            cohort, outcome_name, label,
                            f"missing_column:{src_col}", source_data,
                            n_total=n_total,
                        ))
                        continue
                    if binarize == "positive_amount":
                        y = _binarize_positive_amount(df[src_col], missing_codes)
                    elif binarize == "already_binary":
                        y = _binarize_already_binary(df[src_col], missing_codes)
                    else:
                        rows.append(_empty_row(
                            cohort, outcome_name, label,
                            f"unknown_binarize:{binarize}", source_data,
                            n_total=n_total,
                        ))
                        continue

                rows.append(_fit_and_row(
                    cohort, outcome_name, label, y,
                    df["__g_proxy"], age_series, source_data,
                    n_total, min_class_n,
                ))

        # ------- NLSY97: 2019 govt program income + homeownership at ~40 -------
        elif cohort == "nlsy97":
            for outcome_name, spec in NLSY97_OUTCOMES.items():
                label = spec["label"]
                binarize = spec["binarize"]
                src_col = spec["source_col"]

                if src_col not in df.columns:
                    rows.append(_empty_row(
                        cohort, outcome_name, label,
                        f"missing_column:{src_col}", source_data,
                        n_total=n_total,
                    ))
                    continue

                # Binarize
                if binarize == "already_binary":
                    y = _binarize_already_binary(df[src_col], missing_codes)
                elif binarize == "house_type_40":
                    y = _binarize_house_type_40(df[src_col], missing_codes)
                else:
                    rows.append(_empty_row(
                        cohort, outcome_name, label,
                        f"unknown_binarize:{binarize}", source_data,
                        n_total=n_total,
                    ))
                    continue

                # Derive age
                if "birth_year" in df.columns:
                    if binarize == "house_type_40":
                        # Approximately age 40 for everyone
                        age_series = 40.0 * pd.Series(1.0, index=df.index)
                        # But still derive from birth_year for variation
                        age_series = 2019 - pd.to_numeric(df["birth_year"], errors="coerce")
                    else:
                        age_series = 2019 - pd.to_numeric(df["birth_year"], errors="coerce")
                else:
                    age_series = pd.Series(float("nan"), index=df.index)

                rows.append(_fit_and_row(
                    cohort, outcome_name, label, y,
                    df["__g_proxy"], age_series, source_data,
                    n_total, min_class_n,
                ))

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
    parser = argparse.ArgumentParser(description="Build g_proxy → welfare/housing outcome association tables.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/g_welfare_housing_outcomes.csv"))
    parser.add_argument("--min-class-n", type=int, default=20)
    args = parser.parse_args(argv)

    root = Path(args.project_root)
    cohorts = _cohorts_from_args(args)
    out = run_welfare_housing_outcomes(root=root, cohorts=cohorts, output_path=args.output_path, min_class_n=args.min_class_n)
    computed = int((out["status"] == "computed").sum()) if "status" in out.columns else 0
    print(f"[ok] welfare/housing outcome rows computed: {computed}")
    print(f"[ok] wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
