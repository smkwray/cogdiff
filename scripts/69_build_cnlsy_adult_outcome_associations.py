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

OUTPUT_COLUMNS = [
    "cohort",
    "outcome",
    "model_type",
    "status",
    "reason",
    "outcome_col",
    "age_col",
    "n_total",
    "n_used",
    "n_positive",
    "prevalence",
    "mean_outcome",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "odds_ratio_g",
    "beta_age",
    "SE_beta_age",
    "p_value_beta_age",
    "r2_or_pseudo_r2",
    "source_data",
]

CONTINUOUS_OUTCOMES: tuple[tuple[str, str], ...] = (
    ("education_years_2014", "education_years_2014"),
    ("wage_income_2014", "wage_income_2014"),
    ("family_income_2014", "family_income_2014"),
    ("total_hours_2014", "total_hours_2014"),
    ("num_current_jobs_2014", "num_current_jobs_2014"),
)
BINARY_OUTCOMES: tuple[tuple[str, str], ...] = (
    ("enrolled_2014", "enrolled_2014"),
)


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


def _empty_row(
    *,
    outcome: str,
    model_type: str,
    outcome_col: str,
    age_col: str,
    reason: str,
    source_data: str,
    n_total: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": "cnlsy",
        "outcome": outcome,
        "model_type": model_type,
        "status": "not_feasible",
        "reason": reason,
        "outcome_col": outcome_col,
        "age_col": age_col,
        "n_total": int(n_total),
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_cnlsy_adult_outcome_associations(
    *,
    root: Path,
    output_path: Path = Path("outputs/tables/cnlsy_adult_outcome_associations.csv"),
    min_n_continuous: int = 60,
    min_class_n_binary: int = 20,
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir
    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    source_path = processed_dir / "cnlsy_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / "cnlsy_cfa.csv"
    source_data = str(source_path.relative_to(root)) if source_path.exists() else "data/processed/cnlsy_cfa_resid_or_cfa.csv"

    rows: list[dict[str, Any]] = []
    if not source_path.exists():
        for outcome, outcome_col in CONTINUOUS_OUTCOMES:
            rows.append(_empty_row(outcome=outcome, model_type="ols", outcome_col=outcome_col, age_col="age_2014", reason="missing_source_data", source_data=source_data))
        for outcome, outcome_col in BINARY_OUTCOMES:
            rows.append(_empty_row(outcome=outcome, model_type="logit", outcome_col=outcome_col, age_col="age_2014", reason="missing_source_data", source_data=source_data))
        out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        out.to_csv(target, index=False)
        return out

    df = pd.read_csv(source_path, low_memory=False)
    if "age_2014" not in df.columns:
        for outcome, outcome_col in CONTINUOUS_OUTCOMES:
            rows.append(_empty_row(outcome=outcome, model_type="ols", outcome_col=outcome_col, age_col="age_2014", reason="missing_age_2014", source_data=source_data, n_total=len(df)))
        for outcome, outcome_col in BINARY_OUTCOMES:
            rows.append(_empty_row(outcome=outcome, model_type="logit", outcome_col=outcome_col, age_col="age_2014", reason="missing_age_2014", source_data=source_data, n_total=len(df)))
        out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        out.to_csv(target, index=False)
        return out

    indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", [])]
    df = df.copy()
    df["__g_proxy"] = g_proxy(df, indicators)

    for outcome, outcome_col in CONTINUOUS_OUTCOMES:
        if outcome_col not in df.columns:
            rows.append(_empty_row(outcome=outcome, model_type="ols", outcome_col=outcome_col, age_col="age_2014", reason="missing_outcome_column", source_data=source_data, n_total=len(df)))
            continue
        work = pd.DataFrame(
            {
                "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
                "age": pd.to_numeric(df["age_2014"], errors="coerce"),
                "outcome": pd.to_numeric(df[outcome_col], errors="coerce"),
            }
        ).dropna()
        if len(work) < min_n_continuous:
            rows.append(_empty_row(outcome=outcome, model_type="ols", outcome_col=outcome_col, age_col="age_2014", reason="insufficient_rows", source_data=source_data, n_total=len(df)))
            continue
        x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age": work["age"]}, index=work.index)
        fit, reason = ols_fit(work["outcome"], x)
        if fit is None:
            rows.append(_empty_row(outcome=outcome, model_type="ols", outcome_col=outcome_col, age_col="age_2014", reason=f"ols_failed:{reason or 'unknown'}", source_data=source_data, n_total=len(df)))
            continue
        rows.append(
            {
                "cohort": "cnlsy",
                "outcome": outcome,
                "model_type": "ols",
                "status": "computed",
                "reason": pd.NA,
                "outcome_col": outcome_col,
                "age_col": "age_2014",
                "n_total": int(len(df)),
                "n_used": int(fit["n_used"]),
                "n_positive": pd.NA,
                "prevalence": pd.NA,
                "mean_outcome": float(work["outcome"].mean()),
                "beta_g": float(fit["beta"][1]),
                "SE_beta_g": float(fit["se"][1]),
                "p_value_beta_g": float(fit["p"][1]),
                "odds_ratio_g": pd.NA,
                "beta_age": float(fit["beta"][2]),
                "SE_beta_age": float(fit["se"][2]),
                "p_value_beta_age": float(fit["p"][2]),
                "r2_or_pseudo_r2": float(fit["r2"]),
                "source_data": source_data,
            }
        )

    for outcome, outcome_col in BINARY_OUTCOMES:
        if outcome_col not in df.columns:
            rows.append(_empty_row(outcome=outcome, model_type="logit", outcome_col=outcome_col, age_col="age_2014", reason="missing_outcome_column", source_data=source_data, n_total=len(df)))
            continue
        work = pd.DataFrame(
            {
                "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
                "age": pd.to_numeric(df["age_2014"], errors="coerce"),
                "outcome": pd.to_numeric(df[outcome_col], errors="coerce"),
            }
        ).dropna()
        if work.empty:
            rows.append(_empty_row(outcome=outcome, model_type="logit", outcome_col=outcome_col, age_col="age_2014", reason="no_valid_rows", source_data=source_data, n_total=len(df)))
            continue
        values = sorted(set(float(v) for v in work["outcome"].tolist()))
        if values != [0.0, 1.0]:
            rows.append(_empty_row(outcome=outcome, model_type="logit", outcome_col=outcome_col, age_col="age_2014", reason="outcome_not_binary_zero_one", source_data=source_data, n_total=len(df)))
            continue
        n_positive = int(work["outcome"].eq(1.0).sum())
        n_negative = int(work["outcome"].eq(0.0).sum())
        if n_positive < min_class_n_binary or n_negative < min_class_n_binary:
            rows.append(_empty_row(outcome=outcome, model_type="logit", outcome_col=outcome_col, age_col="age_2014", reason="insufficient_class_counts", source_data=source_data, n_total=len(df)))
            continue
        x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age": work["age"]}, index=work.index)
        fit, reason = _logistic_fit(work["outcome"], x)
        if fit is None:
            rows.append(_empty_row(outcome=outcome, model_type="logit", outcome_col=outcome_col, age_col="age_2014", reason=f"logit_failed:{reason or 'unknown'}", source_data=source_data, n_total=len(df)))
            continue
        rows.append(
            {
                "cohort": "cnlsy",
                "outcome": outcome,
                "model_type": "logit",
                "status": "computed",
                "reason": pd.NA,
                "outcome_col": outcome_col,
                "age_col": "age_2014",
                "n_total": int(len(df)),
                "n_used": int(fit["n_used"]),
                "n_positive": n_positive,
                "prevalence": float(n_positive / len(work)),
                "mean_outcome": float(work["outcome"].mean()),
                "beta_g": float(fit["beta"][1]),
                "SE_beta_g": float(fit["se"][1]),
                "p_value_beta_g": float(fit["p"][1]),
                "odds_ratio_g": float(np.exp(fit["beta"][1])),
                "beta_age": float(fit["beta"][2]),
                "SE_beta_age": float(fit["se"][2]),
                "p_value_beta_age": float(fit["p"][2]),
                "r2_or_pseudo_r2": float(fit["pseudo_r2"]),
                "source_data": source_data,
            }
        )

    out = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS].sort_values(["model_type", "outcome"]).reset_index(drop=True)
    out.to_csv(target, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bounded CNLSY 2014 adult-outcome associations.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/cnlsy_adult_outcome_associations.csv"))
    parser.add_argument("--min-n-continuous", type=int, default=60)
    parser.add_argument("--min-class-n-binary", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output_path = args.output_path if args.output_path.is_absolute() else Path(args.output_path)
    out = run_cnlsy_adult_outcome_associations(
        root=root,
        output_path=output_path,
        min_n_continuous=int(args.min_n_continuous),
        min_class_n_binary=int(args.min_class_n_binary),
    )
    computed = int((out["status"] == "computed").sum()) if not out.empty else 0
    print(f"[ok] cnlsy adult outcome rows computed: {computed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
