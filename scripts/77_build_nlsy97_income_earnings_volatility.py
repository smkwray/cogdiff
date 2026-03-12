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

COHORT = "nlsy97"
VOLATILITY_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    ("household_income", "household_income_2019", "household_income_2021", "age_2019", "age_2021"),
    ("annual_earnings", "annual_earnings_2019", "annual_earnings_2021", "age_2019", "age_2021"),
)
OUTPUT_COLUMNS = [
    "cohort",
    "outcome",
    "model",
    "status",
    "reason",
    "baseline_col",
    "followup_col",
    "age_start_col",
    "age_end_col",
    "n_total",
    "n_two_wave",
    "n_used",
    "n_positive",
    "prevalence",
    "instability_cutoff",
    "mean_age_gap",
    "mean_abs_annualized_log_change",
    "mean_signed_annualized_log_change",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "odds_ratio_g",
    "beta_baseline",
    "SE_beta_baseline",
    "p_value_beta_baseline",
    "r2_or_pseudo_r2",
    "source_data",
]


def _logistic_fit(
    y: pd.Series,
    x: pd.DataFrame,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[dict[str, Any] | None, str | None]:
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
    cov = np.linalg.pinv(xv.T @ (w[:, None] * xv))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_stats = beta / se
    p_vals = np.full(shape=(p,), fill_value=np.nan, dtype=float)
    for i in range(p):
        if math.isfinite(float(z_stats[i])) and math.isfinite(float(se[i])) and float(se[i]) > 0.0:
            p_vals[i] = float(2.0 * norm.sf(abs(float(z_stats[i]))))

    ll_full = float(np.sum(yv * np.log(p_hat) + (1.0 - yv) * np.log(1.0 - p_hat)))
    y_bar = float(np.mean(yv))
    pseudo_r2 = float("nan")
    if 0.0 < y_bar < 1.0:
        ll_null = float(np.sum(yv * math.log(y_bar) + (1.0 - yv) * math.log(1.0 - y_bar)))
        if ll_null != 0.0:
            pseudo_r2 = float(1.0 - (ll_full / ll_null))
    return {"beta": beta, "se": se, "p": p_vals, "fit_stat": pseudo_r2, "n_used": int(n)}, None


def _empty_row(
    *,
    outcome: str,
    model: str,
    baseline_col: str,
    followup_col: str,
    age_start_col: str,
    age_end_col: str,
    reason: str,
    source_data: str,
    n_total: int = 0,
    n_two_wave: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": COHORT,
        "outcome": outcome,
        "model": model,
        "status": "not_feasible",
        "reason": reason,
        "baseline_col": baseline_col,
        "followup_col": followup_col,
        "age_start_col": age_start_col,
        "age_end_col": age_end_col,
        "n_total": int(n_total),
        "n_two_wave": int(n_two_wave),
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _prepare_work(
    df: pd.DataFrame,
    *,
    baseline_col: str,
    followup_col: str,
    age_start_col: str,
    age_end_col: str,
) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
            "baseline": pd.to_numeric(df[baseline_col], errors="coerce"),
            "followup": pd.to_numeric(df[followup_col], errors="coerce"),
            "age_start": pd.to_numeric(df[age_start_col], errors="coerce"),
            "age_end": pd.to_numeric(df[age_end_col], errors="coerce"),
        }
    ).dropna()
    if work.empty:
        return work
    work = work.loc[work["age_end"] > work["age_start"]].copy()
    work = work.loc[(work["baseline"] >= 0.0) & (work["followup"] >= 0.0)].copy()
    if work.empty:
        return work
    work["age_gap"] = work["age_end"] - work["age_start"]
    work["baseline_log1p"] = np.log1p(work["baseline"])
    work["followup_log1p"] = np.log1p(work["followup"])
    work["signed_annualized_log_change"] = (work["followup_log1p"] - work["baseline_log1p"]) / work["age_gap"]
    work["abs_annualized_log_change"] = work["signed_annualized_log_change"].abs()
    return work


def run_nlsy97_income_earnings_volatility(
    *,
    root: Path,
    output_path: Path = Path("outputs/tables/nlsy97_income_earnings_volatility.csv"),
    min_n: int = 200,
    min_class_n: int = 50,
    high_instability_quantile: float = 0.75,
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir
    source_path = processed_dir / f"{COHORT}_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / f"{COHORT}_cfa.csv"
    source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{COHORT}_cfa_resid_or_cfa.csv"

    rows: list[dict[str, Any]] = []
    if not source_path.exists():
        for outcome, baseline_col, followup_col, age_start_col, age_end_col in VOLATILITY_SPECS:
            for model in ("abs_annualized_log_change", "high_instability_top_quartile"):
                rows.append(
                    _empty_row(
                        outcome=outcome,
                        model=model,
                        baseline_col=baseline_col,
                        followup_col=followup_col,
                        age_start_col=age_start_col,
                        age_end_col=age_end_col,
                        reason="missing_source_data",
                        source_data=source_data,
                    )
                )
        out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        out.to_csv(root / output_path, index=False)
        return out

    df = pd.read_csv(source_path, low_memory=False)
    indicators = hierarchical_subtests(models_cfg)
    df = df.copy()
    df["__g_proxy"] = g_proxy(df, indicators)

    for outcome, baseline_col, followup_col, age_start_col, age_end_col in VOLATILITY_SPECS:
        if any(col not in df.columns for col in (baseline_col, followup_col, age_start_col, age_end_col)):
            for model in ("abs_annualized_log_change", "high_instability_top_quartile"):
                rows.append(
                    _empty_row(
                        outcome=outcome,
                        model=model,
                        baseline_col=baseline_col,
                        followup_col=followup_col,
                        age_start_col=age_start_col,
                        age_end_col=age_end_col,
                        reason="missing_required_columns",
                        source_data=source_data,
                        n_total=len(df),
                    )
                )
            continue

        work = _prepare_work(
            df,
            baseline_col=baseline_col,
            followup_col=followup_col,
            age_start_col=age_start_col,
            age_end_col=age_end_col,
        )
        if len(work) < min_n:
            for model in ("abs_annualized_log_change", "high_instability_top_quartile"):
                rows.append(
                    _empty_row(
                        outcome=outcome,
                        model=model,
                        baseline_col=baseline_col,
                        followup_col=followup_col,
                        age_start_col=age_start_col,
                        age_end_col=age_end_col,
                        reason="insufficient_two_wave_rows",
                        source_data=source_data,
                        n_total=len(df),
                        n_two_wave=len(work),
                    )
                )
            continue

        common = {
            "cohort": COHORT,
            "outcome": outcome,
            "baseline_col": baseline_col,
            "followup_col": followup_col,
            "age_start_col": age_start_col,
            "age_end_col": age_end_col,
            "n_total": int(len(df)),
            "n_two_wave": int(len(work)),
            "mean_age_gap": float(work["age_gap"].mean()),
            "mean_abs_annualized_log_change": float(work["abs_annualized_log_change"].mean()),
            "mean_signed_annualized_log_change": float(work["signed_annualized_log_change"].mean()),
            "source_data": source_data,
        }

        x_ols = pd.DataFrame(
            {"intercept": 1.0, "g": work["g"], "baseline_log1p": work["baseline_log1p"], "age_start": work["age_start"]},
            index=work.index,
        )
        fit_ols, reason_ols = ols_fit(work["abs_annualized_log_change"], x_ols)
        if fit_ols is None:
            rows.append(
                {
                    **_empty_row(
                        outcome=outcome,
                        model="abs_annualized_log_change",
                        baseline_col=baseline_col,
                        followup_col=followup_col,
                        age_start_col=age_start_col,
                        age_end_col=age_end_col,
                        reason=f"ols_failed:{reason_ols or 'unknown'}",
                        source_data=source_data,
                        n_total=len(df),
                        n_two_wave=len(work),
                    ),
                    **common,
                }
            )
        else:
            rows.append(
                {
                    **common,
                    "model": "abs_annualized_log_change",
                    "status": "computed",
                    "reason": pd.NA,
                    "n_used": int(fit_ols["n_used"]),
                    "beta_g": float(fit_ols["beta"][1]),
                    "SE_beta_g": float(fit_ols["se"][1]),
                    "p_value_beta_g": float(fit_ols["p"][1]),
                    "beta_baseline": float(fit_ols["beta"][2]),
                    "SE_beta_baseline": float(fit_ols["se"][2]),
                    "p_value_beta_baseline": float(fit_ols["p"][2]),
                    "r2_or_pseudo_r2": float(fit_ols["r2"]),
                }
            )

        cutoff = float(work["abs_annualized_log_change"].quantile(high_instability_quantile))
        work["high_instability"] = (work["abs_annualized_log_change"] >= cutoff).astype(float)
        n_positive = int(work["high_instability"].sum())
        n_negative = int(len(work) - n_positive)
        if min(n_positive, n_negative) < min_class_n:
            rows.append(
                {
                    **_empty_row(
                        outcome=outcome,
                        model="high_instability_top_quartile",
                        baseline_col=baseline_col,
                        followup_col=followup_col,
                        age_start_col=age_start_col,
                        age_end_col=age_end_col,
                        reason="insufficient_high_instability_class_rows",
                        source_data=source_data,
                        n_total=len(df),
                        n_two_wave=len(work),
                    ),
                    **common,
                    "instability_cutoff": cutoff,
                    "n_positive": n_positive,
                    "prevalence": float(work["high_instability"].mean()),
                }
            )
            continue

        x_logit = pd.DataFrame(
            {"intercept": 1.0, "g": work["g"], "baseline_log1p": work["baseline_log1p"], "age_start": work["age_start"]},
            index=work.index,
        )
        fit_logit, reason_logit = _logistic_fit(work["high_instability"], x_logit)
        if fit_logit is None:
            rows.append(
                {
                    **_empty_row(
                        outcome=outcome,
                        model="high_instability_top_quartile",
                        baseline_col=baseline_col,
                        followup_col=followup_col,
                        age_start_col=age_start_col,
                        age_end_col=age_end_col,
                        reason=f"logit_failed:{reason_logit or 'unknown'}",
                        source_data=source_data,
                        n_total=len(df),
                        n_two_wave=len(work),
                    ),
                    **common,
                    "instability_cutoff": cutoff,
                    "n_positive": n_positive,
                    "prevalence": float(work["high_instability"].mean()),
                }
            )
        else:
            rows.append(
                {
                    **common,
                    "model": "high_instability_top_quartile",
                    "status": "computed",
                    "reason": pd.NA,
                    "n_used": int(fit_logit["n_used"]),
                    "n_positive": n_positive,
                    "prevalence": float(work["high_instability"].mean()),
                    "instability_cutoff": cutoff,
                    "beta_g": float(fit_logit["beta"][1]),
                    "SE_beta_g": float(fit_logit["se"][1]),
                    "p_value_beta_g": float(fit_logit["p"][1]),
                    "odds_ratio_g": float(math.exp(float(fit_logit["beta"][1]))),
                    "beta_baseline": float(fit_logit["beta"][2]),
                    "SE_beta_baseline": float(fit_logit["se"][2]),
                    "p_value_beta_baseline": float(fit_logit["p"][2]),
                    "r2_or_pseudo_r2": float(fit_logit["fit_stat"]),
                }
            )

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_path = root / output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build bounded NLSY97 two-wave income and earnings volatility models.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/nlsy97_income_earnings_volatility.csv"))
    parser.add_argument("--min-n", type=int, default=200)
    parser.add_argument("--min-class-n", type=int, default=50)
    parser.add_argument("--high-instability-quantile", type=float, default=0.75)
    args = parser.parse_args(argv)

    out = run_nlsy97_income_earnings_volatility(
        root=Path(args.project_root),
        output_path=args.output_path,
        min_n=int(args.min_n),
        min_class_n=int(args.min_class_n),
        high_instability_quantile=float(args.high_instability_quantile),
    )
    print(f"[ok] volatility rows computed: {int((out['status'] == 'computed').sum())}")
    print(f"[ok] wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
