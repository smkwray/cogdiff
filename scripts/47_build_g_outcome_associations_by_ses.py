#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import chi2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.exploratory import build_ses_bins, g_proxy, ols_fit, pick_col, safe_corr
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

OUTCOME_CANDIDATES: dict[str, tuple[str, ...]] = {
    "earnings": ("annual_earnings", "earnings", "labor_income", "wage_income"),
    "household_income": ("household_income", "family_income", "hh_income", "income"),
    "net_worth": ("net_worth", "wealth", "net_assets", "assets_net"),
    "education": ("education_years", "years_education", "highest_grade_completed", "education"),
}

SUMMARY_COLUMNS = [
    "cohort",
    "outcome",
    "status",
    "reason",
    "ses_col",
    "ses_mode",
    "n_total",
    "n_bins_used",
    "heterogeneity_Q",
    "heterogeneity_df",
    "heterogeneity_p_value",
    "min_beta_g",
    "max_beta_g",
    "source_data",
]

DETAIL_COLUMNS = [
    "cohort",
    "outcome",
    "ses_bin",
    "status",
    "reason",
    "ses_col",
    "ses_mode",
    "n_total",
    "n_used",
    "corr_all",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "r2",
    "source_data",
]


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _empty_summary(cohort: str, outcome: str, reason: str, source_data: str, ses_col: str = "", ses_mode: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "outcome": outcome,
        "status": "not_feasible",
        "reason": reason,
        "ses_col": ses_col,
        "ses_mode": ses_mode,
        "n_total": 0,
        "n_bins_used": 0,
        "source_data": source_data,
    }
    for col in SUMMARY_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _empty_detail(
    cohort: str,
    outcome: str,
    ses_bin: str,
    reason: str,
    source_data: str,
    ses_col: str = "",
    ses_mode: str = "",
    n_total: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "outcome": outcome,
        "ses_bin": ses_bin,
        "status": "not_feasible",
        "reason": reason,
        "ses_col": ses_col,
        "ses_mode": ses_mode,
        "n_total": int(n_total),
        "n_used": 0,
        "source_data": source_data,
    }
    for col in DETAIL_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_g_outcome_by_ses(
    *,
    root: Path,
    cohorts: list[str],
    summary_output_path: Path = Path("outputs/tables/g_outcome_associations_by_ses_summary.csv"),
    detail_output_path: Path = Path("outputs/tables/g_outcome_associations_by_ses.csv"),
    min_group_n: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{cohort}_cfa_resid_or_cfa.csv"
        if not source_path.exists():
            for outcome in OUTCOME_CANDIDATES:
                summary_rows.append(_empty_summary(cohort, outcome, "missing_source_data", source_data))
                detail_rows.append(_empty_detail(cohort, outcome, "", "missing_source_data", source_data))
            continue

        df = pd.read_csv(source_path, low_memory=False)
        indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", [])] if cohort == "cnlsy" else hierarchical_subtests(models_cfg)
        df = df.copy()
        df["__g_proxy"] = g_proxy(df, indicators)
        ses_col = "parent_education" if "parent_education" in df.columns else pick_col(df, ("mother_education", "father_education"))
        if ses_col is None:
            for outcome in OUTCOME_CANDIDATES:
                summary_rows.append(_empty_summary(cohort, outcome, "missing_ses_column", source_data))
                detail_rows.append(_empty_detail(cohort, outcome, "", "missing_ses_column", source_data))
            continue

        ses_bins, ses_mode, _ = build_ses_bins(df[ses_col])
        df["__ses_bin"] = ses_bins.astype("string")
        for outcome, candidates in OUTCOME_CANDIDATES.items():
            out_col = pick_col(df, candidates)
            if out_col is None:
                summary_rows.append(_empty_summary(cohort, outcome, "missing_outcome_column", source_data, ses_col=ses_col, ses_mode=ses_mode))
                detail_rows.append(_empty_detail(cohort, outcome, "", "missing_outcome_column", source_data, ses_col=ses_col, ses_mode=ses_mode, n_total=len(df)))
                continue

            work = df[["__ses_bin", "__g_proxy", out_col]].copy()
            work["outcome"] = pd.to_numeric(work[out_col], errors="coerce")
            work = work.dropna(subset=["__ses_bin", "__g_proxy", "outcome"]).copy()
            if work.empty:
                summary_rows.append(_empty_summary(cohort, outcome, "no_valid_rows_after_cleaning", source_data, ses_col=ses_col, ses_mode=ses_mode))
                detail_rows.append(_empty_detail(cohort, outcome, "", "no_valid_rows_after_cleaning", source_data, ses_col=ses_col, ses_mode=ses_mode, n_total=len(df)))
                continue

            for ses_bin, grp in work.groupby("__ses_bin", sort=True):
                if len(grp) < min_group_n:
                    detail_rows.append(
                        _empty_detail(
                            cohort,
                            outcome,
                            str(ses_bin),
                            "insufficient_group_n",
                            source_data,
                            ses_col=ses_col,
                            ses_mode=ses_mode,
                            n_total=len(df),
                        )
                        | {"n_used": int(len(grp))}
                    )
                    continue
                x = pd.DataFrame({"intercept": 1.0, "g": pd.to_numeric(grp["__g_proxy"], errors="coerce")}, index=grp.index)
                fit, reason = ols_fit(grp["outcome"], x)
                if fit is None:
                    detail_rows.append(
                        _empty_detail(
                            cohort,
                            outcome,
                            str(ses_bin),
                            f"ols_failed:{reason or 'unknown'}",
                            source_data,
                            ses_col=ses_col,
                            ses_mode=ses_mode,
                            n_total=len(df),
                        )
                        | {"n_used": int(len(grp))}
                    )
                    continue

                beta = fit["beta"]
                se = fit["se"]
                p = fit["p"]
                row = {
                    "cohort": cohort,
                    "outcome": outcome,
                    "ses_bin": str(ses_bin),
                    "status": "computed",
                    "reason": pd.NA,
                    "ses_col": ses_col,
                    "ses_mode": ses_mode,
                    "n_total": int(len(df)),
                    "n_used": int(fit["n_used"]),
                    "corr_all": safe_corr(grp["__g_proxy"], grp["outcome"]),
                    "beta_g": float(beta[1]),
                    "SE_beta_g": float(se[1]),
                    "p_value_beta_g": float(p[1]) if math.isfinite(float(p[1])) else pd.NA,
                    "r2": float(fit["r2"]) if math.isfinite(float(fit["r2"])) else pd.NA,
                    "source_data": source_data,
                }
                for col in DETAIL_COLUMNS:
                    row.setdefault(col, pd.NA)
                detail_rows.append(row)

            cohort_outcome_details = pd.DataFrame(
                [
                    r
                    for r in detail_rows
                    if str(r.get("cohort")) == cohort and str(r.get("outcome")) == outcome
                ]
            )
            computed = cohort_outcome_details[cohort_outcome_details["status"] == "computed"].copy() if not cohort_outcome_details.empty else pd.DataFrame()
            if computed.shape[0] < 2:
                summary_rows.append(
                    _empty_summary(
                        cohort,
                        outcome,
                        "insufficient_ses_bins_for_heterogeneity_test",
                        source_data,
                        ses_col=ses_col,
                        ses_mode=ses_mode,
                    )
                    | {"n_total": int(len(work)), "n_bins_used": int(computed.shape[0])}
                )
                continue

            valid = computed[["beta_g", "SE_beta_g"]].apply(pd.to_numeric, errors="coerce").dropna().copy()
            valid = valid[valid["SE_beta_g"] > 0.0]
            if valid.shape[0] < 2:
                summary_rows.append(
                    _empty_summary(
                        cohort,
                        outcome,
                        "invalid_group_standard_errors",
                        source_data,
                        ses_col=ses_col,
                        ses_mode=ses_mode,
                    )
                    | {"n_total": int(len(work)), "n_bins_used": int(computed.shape[0])}
                )
                continue

            weights = 1.0 / (valid["SE_beta_g"] ** 2)
            beta_bar = float((weights * valid["beta_g"]).sum() / weights.sum())
            q_stat = float((weights * ((valid["beta_g"] - beta_bar) ** 2)).sum())
            df_q = int(valid.shape[0] - 1)
            p_val = float(chi2.sf(q_stat, df_q)) if df_q > 0 else pd.NA
            row = {
                "cohort": cohort,
                "outcome": outcome,
                "status": "computed",
                "reason": pd.NA,
                "ses_col": ses_col,
                "ses_mode": ses_mode,
                "n_total": int(len(work)),
                "n_bins_used": int(computed.shape[0]),
                "heterogeneity_Q": q_stat,
                "heterogeneity_df": df_q,
                "heterogeneity_p_value": p_val,
                "min_beta_g": float(valid["beta_g"].min()),
                "max_beta_g": float(valid["beta_g"].max()),
                "source_data": source_data,
            }
            for col in SUMMARY_COLUMNS:
                row.setdefault(col, pd.NA)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
    if detail_df.empty:
        detail_df = pd.DataFrame(columns=DETAIL_COLUMNS)
    for col in SUMMARY_COLUMNS:
        if col not in summary_df.columns:
            summary_df[col] = pd.NA
    for col in DETAIL_COLUMNS:
        if col not in detail_df.columns:
            detail_df[col] = pd.NA
    summary_df = summary_df[SUMMARY_COLUMNS].copy()
    detail_df = detail_df[DETAIL_COLUMNS].copy()

    summary_target = summary_output_path if summary_output_path.is_absolute() else root / summary_output_path
    detail_target = detail_output_path if detail_output_path.is_absolute() else root / detail_output_path
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    detail_target.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_target, index=False)
    detail_df.to_csv(detail_target, index=False)
    return summary_df, detail_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Build g_proxy outcome-validity tables by SES bins.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--min-group-n", type=int, default=60, help="Minimum SES-bin n to compute a row.")
    parser.add_argument("--summary-output-path", type=Path, default=Path("outputs/tables/g_outcome_associations_by_ses_summary.csv"))
    parser.add_argument("--detail-output-path", type=Path, default=Path("outputs/tables/g_outcome_associations_by_ses.csv"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        summary_df, detail_df = run_g_outcome_by_ses(
            root=root,
            cohorts=_cohorts_from_args(args),
            summary_output_path=args.summary_output_path,
            detail_output_path=args.detail_output_path,
            min_group_n=int(args.min_group_n),
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.summary_output_path if args.summary_output_path.is_absolute() else root / args.summary_output_path}")
    print(f"[ok] wrote {args.detail_output_path if args.detail_output_path.is_absolute() else root / args.detail_output_path}")
    print(f"[ok] computed summary rows: {int((summary_df['status'] == 'computed').sum()) if 'status' in summary_df.columns else 0}")
    print(f"[ok] computed detail rows: {int((detail_df['status'] == 'computed').sum()) if 'status' in detail_df.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
