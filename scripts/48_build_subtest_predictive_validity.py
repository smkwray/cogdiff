#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.exploratory import factor_composites, ols_fit, pick_col, safe_corr, zscore
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

OUTCOME_CANDIDATES: dict[str, tuple[str, ...]] = {
    "sat_math_2007": ("sat_math_2007_bin",),
    "sat_verbal_2007": ("sat_verbal_2007_bin",),
    "act_2007": ("act_2007_bin",),
    "earnings": ("annual_earnings", "earnings", "labor_income", "wage_income"),
    "household_income": ("household_income", "family_income", "hh_income", "income"),
    "net_worth": ("net_worth", "wealth", "net_assets", "assets_net"),
    "education": ("education_years", "years_education", "highest_grade_completed", "education"),
}

OUTPUT_COLUMNS = [
    "cohort",
    "outcome",
    "predictor_type",
    "predictor",
    "status",
    "reason",
    "outcome_col",
    "n_total",
    "n_used",
    "pearson",
    "spearman",
    "beta_predictor",
    "SE_beta_predictor",
    "p_value_beta_predictor",
    "r2",
    "source_data",
]


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _empty_row(
    cohort: str,
    outcome: str,
    predictor_type: str,
    predictor: str,
    reason: str,
    source_data: str,
    *,
    outcome_col: str = "",
    n_total: int = 0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "cohort": cohort,
        "outcome": outcome,
        "predictor_type": predictor_type,
        "predictor": predictor,
        "status": "not_feasible",
        "reason": reason,
        "outcome_col": outcome_col,
        "n_total": int(n_total),
        "n_used": 0,
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_subtest_predictive_validity(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path = Path("outputs/tables/subtest_predictive_validity.csv"),
    min_n: int = 40,
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    factor_cfg = models_cfg.get("hierarchical_factors", {})
    factor_map = {str(k): [str(v) for v in vals] for k, vals in factor_cfg.items()} if isinstance(factor_cfg, dict) else {}

    rows: list[dict[str, object]] = []
    for cohort in cohorts:
        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{cohort}_cfa_resid_or_cfa.csv"
        if not source_path.exists():
            rows.append(_empty_row(cohort, "", "", "", "missing_source_data", source_data))
            continue

        df = pd.read_csv(source_path, low_memory=False)
        predictor_series: dict[tuple[str, str], pd.Series] = {}
        if cohort == "cnlsy":
            for subtest in [str(x) for x in models_cfg.get("cnlsy_single_factor", [])]:
                if subtest in df.columns:
                    predictor_series[("subtest", subtest)] = zscore(df[subtest])
        else:
            for subtest in hierarchical_subtests(models_cfg):
                if subtest in df.columns:
                    predictor_series[("subtest", subtest)] = zscore(df[subtest])
            for factor, series in factor_composites(df, factor_map).items():
                predictor_series[("factor", factor)] = series

        if not predictor_series:
            rows.append(_empty_row(cohort, "", "", "", "missing_predictors", source_data, n_total=len(df)))
            continue

        for outcome, candidates in OUTCOME_CANDIDATES.items():
            out_col = pick_col(df, candidates)
            if out_col is None:
                for predictor_type, predictor in predictor_series:
                    rows.append(_empty_row(cohort, outcome, predictor_type, predictor, "missing_outcome_column", source_data, n_total=len(df)))
                continue

            outcome_vals = pd.to_numeric(df[out_col], errors="coerce")
            if outcome.startswith("sat_") or outcome == "act_2007":
                outcome_vals = outcome_vals.mask(outcome_vals <= 0)

            for (predictor_type, predictor), predictor_vals in predictor_series.items():
                work = pd.DataFrame({"predictor": pd.to_numeric(predictor_vals, errors="coerce"), "outcome": outcome_vals}).dropna().copy()
                if len(work) < min_n:
                    rows.append(_empty_row(cohort, outcome, predictor_type, predictor, "insufficient_rows", source_data, outcome_col=out_col, n_total=len(df)) | {"n_used": int(len(work))})
                    continue

                x = pd.DataFrame({"intercept": 1.0, "predictor": work["predictor"]}, index=work.index)
                fit, reason = ols_fit(work["outcome"], x)
                if fit is None:
                    rows.append(_empty_row(cohort, outcome, predictor_type, predictor, f"ols_failed:{reason or 'unknown'}", source_data, outcome_col=out_col, n_total=len(df)) | {"n_used": int(len(work))})
                    continue

                beta = fit["beta"]
                se = fit["se"]
                p = fit["p"]
                row = {
                    "cohort": cohort,
                    "outcome": outcome,
                    "predictor_type": predictor_type,
                    "predictor": predictor,
                    "status": "computed",
                    "reason": pd.NA,
                    "outcome_col": out_col,
                    "n_total": int(len(df)),
                    "n_used": int(fit["n_used"]),
                    "pearson": safe_corr(work["predictor"], work["outcome"], method="pearson"),
                    "spearman": safe_corr(work["predictor"], work["outcome"], method="spearman"),
                    "beta_predictor": float(beta[1]),
                    "SE_beta_predictor": float(se[1]),
                    "p_value_beta_predictor": float(p[1]) if math.isfinite(float(p[1])) else pd.NA,
                    "r2": float(fit["r2"]) if math.isfinite(float(fit["r2"])) else pd.NA,
                    "source_data": source_data,
                }
                for col in OUTPUT_COLUMNS:
                    row.setdefault(col, pd.NA)
                rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=OUTPUT_COLUMNS)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS].copy()
    target = output_path if output_path.is_absolute() else root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build subtest- and factor-level predictive-validity tables for available outcomes.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--min-n", type=int, default=40, help="Minimum usable rows for a predictor/outcome fit.")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/subtest_predictive_validity.csv"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        out = run_subtest_predictive_validity(
            root=root,
            cohorts=_cohorts_from_args(args),
            output_path=args.output_path,
            min_n=int(args.min_n),
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.output_path if args.output_path.is_absolute() else root / args.output_path}")
    print(f"[ok] computed rows: {int((out['status'] == 'computed').sum()) if 'status' in out.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
