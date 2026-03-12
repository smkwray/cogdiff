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

from nls_pipeline.exploratory import g_proxy
from nls_pipeline.io import load_yaml, project_root

OUTPUT_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "outcome_col",
    "age_col",
    "n_total",
    "n_used",
    "n_employed",
    "prevalence",
    "age_min_used",
    "age_max_used",
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


def _empty_row(reason: str, source_data: str, *, n_total: int = 0, n_used: int = 0, n_employed: int = 0) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": "cnlsy",
        "status": "not_feasible",
        "reason": reason,
        "outcome_col": "employment_2014",
        "age_col": "csage",
        "n_total": int(n_total),
        "n_used": int(n_used),
        "n_employed": int(n_employed),
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


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


def run_cnlsy_employment_2014(
    *,
    root: Path,
    output_path: Path = Path("outputs/tables/cnlsy_employment_2014.csv"),
    min_class_n: int = 20,
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    source_path = processed_dir / "cnlsy_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / "cnlsy_cfa.csv"
    source_data = str(source_path.relative_to(root)) if source_path.exists() else "cnlsy_cfa_resid_or_cfa.csv"

    if not source_path.exists():
        out = pd.DataFrame([_empty_row("missing_source_data", source_data)])
    else:
        df = pd.read_csv(source_path, low_memory=False)
        required = {"employment_2014", "csage"}
        missing = sorted(col for col in required if col not in df.columns)
        if missing:
            out = pd.DataFrame([_empty_row(f"missing_required_columns:{','.join(missing)}", source_data, n_total=len(df))])
        else:
            indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", [])]
            df = df.copy()
            df["__g_proxy"] = g_proxy(df, indicators)
            work = pd.DataFrame(
                {
                    "employment": pd.to_numeric(df["employment_2014"], errors="coerce"),
                    "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
                    "age": pd.to_numeric(df["csage"], errors="coerce"),
                }
            ).dropna()

            if work.empty:
                out = pd.DataFrame([_empty_row("no_valid_rows_after_cleaning", source_data, n_total=len(df))])
            else:
                unique_outcomes = sorted(set(float(v) for v in work["employment"].dropna().unique().tolist()))
                if unique_outcomes != [0.0, 1.0]:
                    out = pd.DataFrame([_empty_row("employment_not_binary_zero_one", source_data, n_total=len(df), n_used=len(work))])
                else:
                    n_used = int(len(work))
                    n_employed = int(work["employment"].sum())
                    n_unemployed = int(n_used - n_employed)
                    if n_employed < min_class_n or n_unemployed < min_class_n:
                        out = pd.DataFrame(
                            [
                                _empty_row(
                                    "insufficient_outcome_class_counts",
                                    source_data,
                                    n_total=len(df),
                                    n_used=n_used,
                                    n_employed=n_employed,
                                )
                            ]
                        )
                    else:
                        x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age": work["age"]}, index=work.index)
                        fit, reason = _logistic_fit(work["employment"], x)
                        if fit is None:
                            out = pd.DataFrame(
                                [
                                    _empty_row(
                                        f"logit_failed:{reason or 'unknown'}",
                                        source_data,
                                        n_total=len(df),
                                        n_used=n_used,
                                        n_employed=n_employed,
                                    )
                                ]
                            )
                        else:
                            beta = fit["beta"]
                            se = fit["se"]
                            p_vals = fit["p"]
                            row = {
                                "cohort": "cnlsy",
                                "status": "computed",
                                "reason": pd.NA,
                                "outcome_col": "employment_2014",
                                "age_col": "csage",
                                "n_total": int(len(df)),
                                "n_used": n_used,
                                "n_employed": n_employed,
                                "prevalence": float(n_employed / n_used) if n_used > 0 else pd.NA,
                                "age_min_used": float(work["age"].min()),
                                "age_max_used": float(work["age"].max()),
                                "beta_g": float(beta[1]),
                                "SE_beta_g": float(se[1]),
                                "p_value_beta_g": float(p_vals[1]) if math.isfinite(float(p_vals[1])) else pd.NA,
                                "odds_ratio_g": float(math.exp(beta[1])),
                                "beta_age": float(beta[2]),
                                "SE_beta_age": float(se[2]),
                                "p_value_beta_age": float(p_vals[2]) if math.isfinite(float(p_vals[2])) else pd.NA,
                                "pseudo_r2": float(fit["pseudo_r2"]),
                                "source_data": source_data,
                            }
                            for col in OUTPUT_COLUMNS:
                                row.setdefault(col, pd.NA)
                            out = pd.DataFrame([row], columns=OUTPUT_COLUMNS)

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
    parser = argparse.ArgumentParser(description="Build CNLSY 2014 employment-status association with g_proxy.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/cnlsy_employment_2014.csv"))
    parser.add_argument("--min-class-n", type=int, default=20)
    args = parser.parse_args(argv)

    out = run_cnlsy_employment_2014(root=Path(args.project_root), output_path=args.output_path, min_class_n=args.min_class_n)
    print(f"[ok] wrote {args.output_path}")
    print(f"[ok] computed rows: {int((out['status'] == 'computed').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
