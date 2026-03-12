#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.exploratory import g_proxy, ols_fit, pick_col, safe_corr
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

ASSOC_COLUMNS = [
    "cohort",
    "outcome",
    "status",
    "reason",
    "outcome_col",
    "n_total",
    "n_used",
    "corr_all",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "r2",
    "source_data",
]

CONTRAST_COLUMNS = [
    "outcome",
    "cohort_a",
    "cohort_b",
    "status",
    "reason",
    "beta_a",
    "beta_b",
    "diff_b_minus_a",
    "SE_diff",
    "z_diff",
    "p_value_diff",
    "ci_low_diff",
    "ci_high_diff",
    "source_estimates",
]

STABILITY_COLUMNS = [
    "estimand",
    "status",
    "reason",
    "n_cohorts",
    "cohorts",
    "mean_estimate",
    "sd_estimate",
    "range_estimate",
    "cv_estimate",
    "source_estimates",
]


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _empty_assoc(cohort: str, outcome: str, reason: str, source_data: str, *, outcome_col: str = "", n_total: int = 0) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "outcome": outcome,
        "status": "not_feasible",
        "reason": reason,
        "outcome_col": outcome_col,
        "n_total": int(n_total),
        "n_used": 0,
        "source_data": source_data,
    }
    for col in ASSOC_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _pairwise_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    computed = frame[frame["status"] == "computed"].copy()
    for outcome, grp in computed.groupby("outcome", sort=True):
        grp = grp.dropna(subset=["beta_g", "SE_beta_g"]).copy()
        if grp.shape[0] < 2:
            continue
        for idx_a, idx_b in combinations(range(grp.shape[0]), 2):
            row_a = grp.iloc[idx_a]
            row_b = grp.iloc[idx_b]
            beta_a = float(row_a["beta_g"])
            beta_b = float(row_b["beta_g"])
            se_diff = math.sqrt(float(row_a["SE_beta_g"]) ** 2 + float(row_b["SE_beta_g"]) ** 2)
            if se_diff <= 0.0 or not math.isfinite(se_diff):
                rows.append(
                    {
                        "outcome": outcome,
                        "cohort_a": row_a["cohort"],
                        "cohort_b": row_b["cohort"],
                        "status": "not_feasible",
                        "reason": "nonpositive_se_diff",
                        "source_estimates": f"{row_a['source_data']},{row_b['source_data']}",
                    }
                )
                continue
            diff = beta_b - beta_a
            z = diff / se_diff
            p = float(2.0 * norm.sf(abs(z)))
            rows.append(
                {
                    "outcome": outcome,
                    "cohort_a": row_a["cohort"],
                    "cohort_b": row_b["cohort"],
                    "status": "computed",
                    "reason": pd.NA,
                    "beta_a": beta_a,
                    "beta_b": beta_b,
                    "diff_b_minus_a": diff,
                    "SE_diff": se_diff,
                    "z_diff": z,
                    "p_value_diff": p,
                    "ci_low_diff": diff - 1.96 * se_diff,
                    "ci_high_diff": diff + 1.96 * se_diff,
                    "source_estimates": f"{row_a['source_data']},{row_b['source_data']}",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=CONTRAST_COLUMNS)
    for col in CONTRAST_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[CONTRAST_COLUMNS].copy()


def _stability_rows(root: Path) -> pd.DataFrame:
    tables: list[tuple[str, Path, str]] = [
        ("d_g", root / "outputs/tables/g_mean_diff.csv", "estimate"),
        ("vr_g", root / "outputs/tables/g_variance_ratio.csv", "estimate"),
    ]
    rows: list[dict[str, Any]] = []
    for estimand, path, _ in tables:
        if not path.exists():
            rows.append({"estimand": estimand, "status": "not_feasible", "reason": "missing_source_table", "source_estimates": str(path.relative_to(root))})
            continue
        df = pd.read_csv(path, low_memory=False)
        est_col = "estimate" if "estimate" in df.columns else pick_col(df, ("d_g", "VR_g", "vr_g", "variance_ratio", "log_vr_g"))
        cohort_col = "cohort" if "cohort" in df.columns else None
        if est_col is None or cohort_col is None:
            rows.append({"estimand": estimand, "status": "not_feasible", "reason": "missing_required_columns", "source_estimates": str(path.relative_to(root))})
            continue
        work = df[[cohort_col, est_col]].copy()
        work["estimate"] = pd.to_numeric(work[est_col], errors="coerce")
        work = work.dropna(subset=["estimate"]).copy()
        if work.empty:
            rows.append({"estimand": estimand, "status": "not_feasible", "reason": "no_valid_rows", "source_estimates": str(path.relative_to(root))})
            continue
        mean_est = float(work["estimate"].mean())
        sd_est = float(work["estimate"].std(ddof=1)) if work.shape[0] >= 2 else 0.0
        range_est = float(work["estimate"].max() - work["estimate"].min()) if work.shape[0] >= 2 else 0.0
        cv_est = abs(sd_est / mean_est) if mean_est != 0.0 else pd.NA
        rows.append(
            {
                "estimand": estimand,
                "status": "computed",
                "reason": pd.NA,
                "n_cohorts": int(work.shape[0]),
                "cohorts": ",".join(work[cohort_col].astype(str).tolist()),
                "mean_estimate": mean_est,
                "sd_estimate": sd_est,
                "range_estimate": range_est,
                "cv_estimate": cv_est,
                "source_estimates": str(path.relative_to(root)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=STABILITY_COLUMNS)
    for col in STABILITY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[STABILITY_COLUMNS].copy()


def run_cross_cohort_suite(
    *,
    root: Path,
    cohorts: list[str],
    assoc_output_path: Path = Path("outputs/tables/overall_outcome_validity.csv"),
    contrasts_output_path: Path = Path("outputs/tables/cross_cohort_predictive_validity_contrasts.csv"),
    stability_output_path: Path = Path("outputs/tables/cross_cohort_pattern_stability.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    assoc_rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{cohort}_cfa_resid_or_cfa.csv"
        if not source_path.exists():
            for outcome in OUTCOME_CANDIDATES:
                assoc_rows.append(_empty_assoc(cohort, outcome, "missing_source_data", source_data))
            continue

        df = pd.read_csv(source_path, low_memory=False)
        indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", [])] if cohort == "cnlsy" else hierarchical_subtests(models_cfg)
        df = df.copy()
        df["__g_proxy"] = g_proxy(df, indicators)

        for outcome, candidates in OUTCOME_CANDIDATES.items():
            out_col = pick_col(df, candidates)
            if out_col is None:
                assoc_rows.append(_empty_assoc(cohort, outcome, "missing_outcome_column", source_data, n_total=len(df)))
                continue
            work = pd.DataFrame({"g": pd.to_numeric(df["__g_proxy"], errors="coerce"), "outcome": pd.to_numeric(df[out_col], errors="coerce")}).dropna().copy()
            if len(work) < 40:
                assoc_rows.append(_empty_assoc(cohort, outcome, "insufficient_rows", source_data, outcome_col=out_col, n_total=len(df)) | {"n_used": int(len(work))})
                continue
            x = pd.DataFrame({"intercept": 1.0, "g": work["g"]}, index=work.index)
            fit, reason = ols_fit(work["outcome"], x)
            if fit is None:
                assoc_rows.append(_empty_assoc(cohort, outcome, f"ols_failed:{reason or 'unknown'}", source_data, outcome_col=out_col, n_total=len(df)) | {"n_used": int(len(work))})
                continue
            beta = fit["beta"]
            se = fit["se"]
            p = fit["p"]
            assoc_rows.append(
                {
                    "cohort": cohort,
                    "outcome": outcome,
                    "status": "computed",
                    "reason": pd.NA,
                    "outcome_col": out_col,
                    "n_total": int(len(df)),
                    "n_used": int(fit["n_used"]),
                    "corr_all": safe_corr(work["g"], work["outcome"]),
                    "beta_g": float(beta[1]),
                    "SE_beta_g": float(se[1]),
                    "p_value_beta_g": float(p[1]) if math.isfinite(float(p[1])) else pd.NA,
                    "r2": float(fit["r2"]) if math.isfinite(float(fit["r2"])) else pd.NA,
                    "source_data": source_data,
                }
            )

    assoc_df = pd.DataFrame(assoc_rows)
    if assoc_df.empty:
        assoc_df = pd.DataFrame(columns=ASSOC_COLUMNS)
    for col in ASSOC_COLUMNS:
        if col not in assoc_df.columns:
            assoc_df[col] = pd.NA
    assoc_df = assoc_df[ASSOC_COLUMNS].copy()
    contrast_df = _pairwise_contrasts(assoc_df)
    stability_df = _stability_rows(root)

    for frame, out_path in (
        (assoc_df, assoc_output_path),
        (contrast_df, contrasts_output_path),
        (stability_df, stability_output_path),
    ):
        target = out_path if out_path.is_absolute() else root / out_path
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
    return assoc_df, contrast_df, stability_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cross-cohort overall validity, contrasts, and pattern-stability tables.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--assoc-output-path", type=Path, default=Path("outputs/tables/overall_outcome_validity.csv"))
    parser.add_argument("--contrasts-output-path", type=Path, default=Path("outputs/tables/cross_cohort_predictive_validity_contrasts.csv"))
    parser.add_argument("--stability-output-path", type=Path, default=Path("outputs/tables/cross_cohort_pattern_stability.csv"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        assoc_df, contrast_df, stability_df = run_cross_cohort_suite(
            root=root,
            cohorts=_cohorts_from_args(args),
            assoc_output_path=args.assoc_output_path,
            contrasts_output_path=args.contrasts_output_path,
            stability_output_path=args.stability_output_path,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.assoc_output_path if args.assoc_output_path.is_absolute() else root / args.assoc_output_path}")
    print(f"[ok] wrote {args.contrasts_output_path if args.contrasts_output_path.is_absolute() else root / args.contrasts_output_path}")
    print(f"[ok] wrote {args.stability_output_path if args.stability_output_path.is_absolute() else root / args.stability_output_path}")
    print(f"[ok] computed overall validity rows: {int((assoc_df['status'] == 'computed').sum()) if 'status' in assoc_df.columns else 0}")
    print(f"[ok] computed contrast rows: {int((contrast_df['status'] == 'computed').sum()) if 'status' in contrast_df.columns else 0}")
    print(f"[ok] computed stability rows: {int((stability_df['status'] == 'computed').sum()) if 'status' in stability_df.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
