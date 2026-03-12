#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.exploratory import g_proxy, ols_fit
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT = "nlsy97"
TRAJECTORY_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
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
    "mean_age_gap",
    "mean_baseline",
    "mean_followup",
    "mean_annualized_log_change",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "beta_baseline",
    "SE_beta_baseline",
    "p_value_beta_baseline",
    "r2",
    "source_data",
]


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
    if work.empty:
        return work
    work = work.loc[(work["baseline"] >= 0.0) & (work["followup"] >= 0.0)].copy()
    if work.empty:
        return work
    work["age_gap"] = work["age_end"] - work["age_start"]
    work["baseline_log1p"] = np.log1p(work["baseline"])
    work["followup_log1p"] = np.log1p(work["followup"])
    work["annualized_log_change"] = (work["followup_log1p"] - work["baseline_log1p"]) / work["age_gap"]
    return work


def run_nlsy97_income_earnings_trajectories(
    *,
    root: Path,
    output_path: Path = Path("outputs/tables/nlsy97_income_earnings_trajectories.csv"),
    min_n: int = 200,
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
        for outcome, baseline_col, followup_col, age_start_col, age_end_col in TRAJECTORY_SPECS:
            rows.append(
                _empty_row(
                    outcome=outcome,
                    model="annualized_log_change",
                    baseline_col=baseline_col,
                    followup_col=followup_col,
                    age_start_col=age_start_col,
                    age_end_col=age_end_col,
                    reason="missing_source_data",
                    source_data=source_data,
                )
            )
            rows.append(
                _empty_row(
                    outcome=outcome,
                    model="followup_conditional_on_baseline",
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

    for outcome, baseline_col, followup_col, age_start_col, age_end_col in TRAJECTORY_SPECS:
        if any(col not in df.columns for col in (baseline_col, followup_col, age_start_col, age_end_col)):
            reason = "missing_required_columns"
            rows.append(
                _empty_row(
                    outcome=outcome,
                    model="annualized_log_change",
                    baseline_col=baseline_col,
                    followup_col=followup_col,
                    age_start_col=age_start_col,
                    age_end_col=age_end_col,
                    reason=reason,
                    source_data=source_data,
                    n_total=len(df),
                )
            )
            rows.append(
                _empty_row(
                    outcome=outcome,
                    model="followup_conditional_on_baseline",
                    baseline_col=baseline_col,
                    followup_col=followup_col,
                    age_start_col=age_start_col,
                    age_end_col=age_end_col,
                    reason=reason,
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
            reason = "insufficient_two_wave_rows"
            rows.append(
                _empty_row(
                    outcome=outcome,
                    model="annualized_log_change",
                    baseline_col=baseline_col,
                    followup_col=followup_col,
                    age_start_col=age_start_col,
                    age_end_col=age_end_col,
                    reason=reason,
                    source_data=source_data,
                    n_total=len(df),
                    n_two_wave=len(work),
                )
            )
            rows.append(
                _empty_row(
                    outcome=outcome,
                    model="followup_conditional_on_baseline",
                    baseline_col=baseline_col,
                    followup_col=followup_col,
                    age_start_col=age_start_col,
                    age_end_col=age_end_col,
                    reason=reason,
                    source_data=source_data,
                    n_total=len(df),
                    n_two_wave=len(work),
                )
            )
            continue

        summary = {
            "cohort": COHORT,
            "outcome": outcome,
            "baseline_col": baseline_col,
            "followup_col": followup_col,
            "age_start_col": age_start_col,
            "age_end_col": age_end_col,
            "n_total": int(len(df)),
            "n_two_wave": int(len(work)),
            "mean_age_gap": float(work["age_gap"].mean()),
            "mean_baseline": float(work["baseline"].mean()),
            "mean_followup": float(work["followup"].mean()),
            "mean_annualized_log_change": float(work["annualized_log_change"].mean()),
            "source_data": source_data,
        }

        x_change = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age_start": work["age_start"]}, index=work.index)
        fit_change, reason_change = ols_fit(work["annualized_log_change"], x_change)
        if fit_change is None:
            rows.append(
                {
                    **_empty_row(
                        outcome=outcome,
                        model="annualized_log_change",
                        baseline_col=baseline_col,
                        followup_col=followup_col,
                        age_start_col=age_start_col,
                        age_end_col=age_end_col,
                        reason=f"ols_failed:{reason_change or 'unknown'}",
                        source_data=source_data,
                        n_total=len(df),
                        n_two_wave=len(work),
                    ),
                    **summary,
                }
            )
        else:
            rows.append(
                {
                    **summary,
                    "model": "annualized_log_change",
                    "status": "computed",
                    "reason": pd.NA,
                    "n_used": int(fit_change["n_used"]),
                    "beta_g": float(fit_change["beta"][1]),
                    "SE_beta_g": float(fit_change["se"][1]),
                    "p_value_beta_g": float(fit_change["p"][1]),
                    "beta_baseline": pd.NA,
                    "SE_beta_baseline": pd.NA,
                    "p_value_beta_baseline": pd.NA,
                    "r2": float(fit_change["r2"]),
                }
            )

        x_follow = pd.DataFrame(
            {
                "intercept": 1.0,
                "g": work["g"],
                "age_start": work["age_start"],
                "baseline_log1p": work["baseline_log1p"],
            },
            index=work.index,
        )
        fit_follow, reason_follow = ols_fit(work["followup_log1p"], x_follow)
        if fit_follow is None:
            rows.append(
                {
                    **_empty_row(
                        outcome=outcome,
                        model="followup_conditional_on_baseline",
                        baseline_col=baseline_col,
                        followup_col=followup_col,
                        age_start_col=age_start_col,
                        age_end_col=age_end_col,
                        reason=f"ols_failed:{reason_follow or 'unknown'}",
                        source_data=source_data,
                        n_total=len(df),
                        n_two_wave=len(work),
                    ),
                    **summary,
                }
            )
        else:
            rows.append(
                {
                    **summary,
                    "model": "followup_conditional_on_baseline",
                    "status": "computed",
                    "reason": pd.NA,
                    "n_used": int(fit_follow["n_used"]),
                    "beta_g": float(fit_follow["beta"][1]),
                    "SE_beta_g": float(fit_follow["se"][1]),
                    "p_value_beta_g": float(fit_follow["p"][1]),
                    "beta_baseline": float(fit_follow["beta"][3]),
                    "SE_beta_baseline": float(fit_follow["se"][3]),
                    "p_value_beta_baseline": float(fit_follow["p"][3]),
                    "r2": float(fit_follow["r2"]),
                }
            )

    out = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS].sort_values(["outcome", "model"]).reset_index(drop=True)
    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bounded NLSY97 two-wave income and earnings trajectory models.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables/nlsy97_income_earnings_trajectories.csv"),
        help="Output CSV path, relative to project root unless absolute.",
    )
    parser.add_argument("--min-n", type=int, default=200, help="Minimum two-wave usable rows per outcome.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output_path = args.output_path if args.output_path.is_absolute() else Path(args.output_path)
    out = run_nlsy97_income_earnings_trajectories(root=root, output_path=output_path, min_n=int(args.min_n))
    computed = int((out["status"] == "computed").sum()) if not out.empty else 0
    print(f"[ok] trajectory rows computed: {computed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
