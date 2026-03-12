#!/usr/bin/env python3
"""For each outcome, fit a pooled regression (outcome ~ g_proxy + age) and
compute mean residuals by race, sex, and race×sex groups.

A positive residual means the group earns/achieves MORE than their g_proxy
score would predict. A negative residual means they earn/achieve LESS.

Outputs: outputs/tables/outcome_residuals_by_group.csv
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

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
    "cnlsy": "config/cnlsy.yml",
}

RACE_COL = "race_ethnicity_3cat"
SEX_COL = "sex"

# (outcome_label, outcome_col, age_col_or_None, model_type)
OUTCOME_SPECS: dict[str, list[tuple[str, str, str | None, str]]] = {
    "nlsy79": [
        ("earnings", "annual_earnings", "age_2000", "ols"),
        ("household_income", "household_income", "age_2000", "ols"),
        ("net_worth", "net_worth", "age_2000", "ols"),
        ("education_years", "education_years", "age_2000", "ols"),
    ],
    "nlsy97": [
        ("earnings", "annual_earnings_2021", "age_2021", "ols"),
        ("household_income", "household_income_2021", "age_2021", "ols"),
        ("net_worth", "net_worth", "age_2021", "ols"),
        ("education_years", "education_years", "age_2021", "ols"),
    ],
    "cnlsy": [
        ("education_years", "education_years", "age_2014", "ols"),
    ],
}

OUTPUT_COLUMNS = [
    "cohort",
    "outcome",
    "group_kind",
    "group_value",
    "status",
    "reason",
    "n_pooled",
    "n_group",
    "mean_actual",
    "mean_predicted",
    "mean_residual",
    "se_residual",
    "p_value_residual",
    "pct_over_under",
    "mean_g_proxy",
    "pooled_beta_g",
    "pooled_r2",
    "source_data",
]


def _normalize_sex(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"m", "male", "1", "man", "boy"}:
        return "male"
    if token in {"f", "female", "2", "woman", "girl"}:
        return "female"
    return ""


def _ols_fit(y: np.ndarray, X: np.ndarray) -> dict[str, Any] | None:
    """Simple OLS: returns beta, predicted, residuals, r2."""
    n, p = X.shape
    if n <= p:
        return None
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    predicted = X @ beta
    residuals = y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "beta": beta,
        "predicted": predicted,
        "residuals": residuals,
        "r2": r2,
    }


def run(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path,
    min_group_n: int = 20,
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    rows: list[dict[str, Any]] = []

    for cohort in cohorts:
        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = (
            str(source_path.relative_to(root))
            if source_path.exists()
            else f"{cohort}_cfa_resid_or_cfa.csv"
        )
        if not source_path.exists():
            continue

        df = pd.read_csv(source_path, low_memory=False)

        # Compute g_proxy
        indicators = (
            [str(x) for x in models_cfg.get("cnlsy_single_factor", [])]
            if cohort == "cnlsy"
            else hierarchical_subtests(models_cfg)
        )
        df = df.copy()
        df["__g_proxy"] = g_proxy(df, indicators)

        # Normalise sex
        if SEX_COL in df.columns:
            df["__sex"] = df[SEX_COL].map(_normalize_sex)
        else:
            df["__sex"] = ""

        # Race groups
        race_groups: list[str] = []
        if RACE_COL in df.columns:
            race_groups = sorted(df[RACE_COL].dropna().unique().tolist())

        outcomes = OUTCOME_SPECS.get(cohort, [])

        for outcome_name, outcome_col, age_col, model_type in outcomes:
            if outcome_col not in df.columns:
                continue

            # Build work dataframe
            work_cols: dict[str, pd.Series] = {
                "y": pd.to_numeric(df[outcome_col], errors="coerce"),
                "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
            }
            if age_col and age_col in df.columns:
                work_cols["age"] = pd.to_numeric(df[age_col], errors="coerce")

            work = pd.DataFrame(work_cols).dropna()
            if len(work) < 30:
                continue

            # Fit pooled OLS: y ~ intercept + g (+ age)
            y_arr = work["y"].to_numpy(dtype=float)
            if "age" in work.columns:
                X_arr = np.column_stack([
                    np.ones(len(work)),
                    work["g"].to_numpy(dtype=float),
                    work["age"].to_numpy(dtype=float),
                ])
                g_idx = 1
            else:
                X_arr = np.column_stack([
                    np.ones(len(work)),
                    work["g"].to_numpy(dtype=float),
                ])
                g_idx = 1

            fit = _ols_fit(y_arr, X_arr)
            if fit is None:
                continue

            n_pooled = len(work)
            pooled_beta_g = float(fit["beta"][g_idx])
            pooled_r2 = float(fit["r2"])
            predicted = fit["predicted"]
            residuals = fit["residuals"]

            # Store residuals and predicted back into work
            work = work.copy()
            work["predicted"] = predicted
            work["residual"] = residuals

            # Map group columns into work
            work["__race"] = df.loc[work.index, RACE_COL] if RACE_COL in df.columns else ""
            work["__sex"] = df.loc[work.index, "__sex"]

            def _compute_group_row(
                group_kind: str,
                group_value: str,
                mask: pd.Series,
            ) -> dict[str, Any]:
                sub = work[mask]
                n_group = len(sub)
                if n_group < min_group_n:
                    return {
                        "cohort": cohort,
                        "outcome": outcome_name,
                        "group_kind": group_kind,
                        "group_value": group_value,
                        "status": "not_feasible",
                        "reason": "insufficient_group_n",
                        "n_pooled": n_pooled,
                        "n_group": n_group,
                        "source_data": source_data,
                        **{c: pd.NA for c in OUTPUT_COLUMNS if c not in [
                            "cohort", "outcome", "group_kind", "group_value",
                            "status", "reason", "n_pooled", "n_group", "source_data",
                        ]},
                    }

                mean_actual = float(sub["y"].mean())
                mean_predicted = float(sub["predicted"].mean())
                mean_residual = float(sub["residual"].mean())
                mean_g = float(sub["g"].mean())

                resid_arr = sub["residual"].to_numpy(dtype=float)
                se_resid = float(np.std(resid_arr, ddof=1) / np.sqrt(n_group))

                # t-test: is the mean residual significantly different from zero?
                if n_group >= 2:
                    t_stat, p_val = ttest_1samp(resid_arr, 0.0)
                    p_val = float(p_val)
                else:
                    p_val = float("nan")

                # Percentage over/under: residual as % of mean predicted
                if abs(mean_predicted) > 1e-6:
                    pct = float(mean_residual / abs(mean_predicted) * 100.0)
                else:
                    pct = float("nan")

                return {
                    "cohort": cohort,
                    "outcome": outcome_name,
                    "group_kind": group_kind,
                    "group_value": group_value,
                    "status": "computed",
                    "reason": pd.NA,
                    "n_pooled": n_pooled,
                    "n_group": n_group,
                    "mean_actual": round(mean_actual, 4),
                    "mean_predicted": round(mean_predicted, 4),
                    "mean_residual": round(mean_residual, 4),
                    "se_residual": round(se_resid, 4),
                    "p_value_residual": p_val if math.isfinite(p_val) else pd.NA,
                    "pct_over_under": round(pct, 2) if math.isfinite(pct) else pd.NA,
                    "mean_g_proxy": round(mean_g, 4),
                    "pooled_beta_g": round(pooled_beta_g, 4),
                    "pooled_r2": round(pooled_r2, 4),
                    "source_data": source_data,
                }

            # By race
            for rg in race_groups:
                rows.append(_compute_group_row("race", str(rg), work["__race"] == rg))

            # By sex
            for sx in ["male", "female"]:
                rows.append(_compute_group_row("sex", sx, work["__sex"] == sx))

            # By race × sex
            for rg in race_groups:
                for sx in ["male", "female"]:
                    mask = (work["__race"] == rg) & (work["__sex"] == sx)
                    rows.append(_compute_group_row("race_sex", f"{rg}|{sx}", mask))

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute outcome residuals (actual - g-predicted) by race, sex, and race×sex."
    )
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--min-group-n", type=int, default=20)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables/outcome_residuals_by_group.csv"),
    )
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    cohorts = list(COHORT_CONFIGS) if (args.all or not args.cohort) else args.cohort
    out = run(
        root=root,
        cohorts=cohorts,
        output_path=args.output_path,
        min_group_n=args.min_group_n,
    )
    computed = int((out["status"] == "computed").sum()) if "status" in out.columns else 0
    print(f"[ok] {computed} rows computed")
    print(f"[ok] wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
