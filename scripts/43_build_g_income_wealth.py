#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

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
}

AGE_CONTROL_CANDIDATES: dict[str, dict[str, tuple[str, ...]]] = {
    "nlsy79": {
        "earnings": ("age_2000", "birth_year"),
        "household_income": ("age_2000", "birth_year"),
        "net_worth": ("age_2000", "birth_year"),
    },
    "nlsy97": {
        "earnings": ("age_2021", "age_2019", "birth_year"),
        "household_income": ("age_2010", "birth_year"),
        "net_worth": ("age_2021", "age_2019", "birth_year"),
    },
    "cnlsy": {
        "earnings": ("age_2014", "birth_year"),
        "household_income": ("age_2014", "birth_year"),
        "net_worth": ("age_2014", "birth_year"),
    },
}

EDUCATION_COL_CANDIDATES = (
    "education_years",
    "years_education",
    "highest_grade_completed",
    "educ_years",
    "education",
)

OUTPUT_COLUMNS = [
    "cohort",
    "outcome",
    "status",
    "reason",
    "sex_col",
    "outcome_col",
    "education_col",
    "n_total",
    "n_used",
    "n_male",
    "n_female",
    "corr_all",
    "corr_male",
    "corr_female",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "beta_gxsex",
    "SE_beta_gxsex",
    "p_value_beta_gxsex",
    "r2",
    "source_data",
]


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _normalize_sex(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"m", "male", "1", "man", "boy"}:
        return "male"
    if token in {"f", "female", "2", "woman", "girl"}:
        return "female"
    return "unknown"


def _zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mean = vals.mean(skipna=True)
    sd = vals.std(skipna=True, ddof=1)
    if pd.isna(sd) or float(sd) <= 0.0:
        return pd.Series([pd.NA] * len(vals), index=vals.index, dtype="float64")
    return (vals - mean) / sd


def _g_proxy(df: pd.DataFrame, indicators: list[str]) -> pd.Series:
    existing = [col for col in indicators if col in df.columns]
    if not existing:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
    z = pd.DataFrame({col: _zscore(df[col]) for col in existing}, index=df.index)
    return z.mean(axis=1, skipna=False)


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _pick_age_control_col(df: pd.DataFrame, cohort: str, outcome: str) -> str | None:
    cohort_map = AGE_CONTROL_CANDIDATES.get(cohort, {})
    return _pick_col(df, cohort_map.get(outcome, ("birth_year",)))


def _safe_corr(x: pd.Series, y: pd.Series) -> float | None:
    xnum = pd.to_numeric(x, errors="coerce")
    ynum = pd.to_numeric(y, errors="coerce")
    mask = xnum.notna() & ynum.notna()
    if int(mask.sum()) < 3:
        return None
    xv = xnum[mask]
    yv = ynum[mask]
    if float(xv.std(ddof=1)) <= 0.0 or float(yv.std(ddof=1)) <= 0.0:
        return None
    return float(xv.corr(yv))


def _ols_fit(y: pd.Series, x: pd.DataFrame) -> tuple[dict[str, Any] | None, str | None]:
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
        beta, _, _, _ = np.linalg.lstsq(xv, yv, rcond=None)
    except np.linalg.LinAlgError:
        return None, "ols_singular_matrix"

    yhat = xv @ beta
    resid = yv - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((yv - float(np.mean(yv))) ** 2))
    dof = n - p
    if dof <= 0:
        return None, "nonpositive_residual_dof"

    sigma2 = sse / float(dof)
    xtx = xv.T @ xv
    try:
        xtx_inv = np.linalg.pinv(xtx)
    except np.linalg.LinAlgError:
        return None, "ols_xtx_pinv_failed"

    var_beta = sigma2 * xtx_inv
    se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))
    p_vals = np.full(shape=(p,), fill_value=np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
    for i in range(p):
        if math.isfinite(float(t_stats[i])) and math.isfinite(float(se[i])) and float(se[i]) > 0.0:
            p_vals[i] = float(2.0 * t.sf(abs(float(t_stats[i])), dof))

    r2 = np.nan
    if sst > 0.0 and math.isfinite(sse):
        r2 = float(1.0 - (sse / sst))

    return {
        "beta": beta,
        "se": se,
        "p": p_vals,
        "r2": r2,
        "n_used": int(n),
    }, None


def _empty_row(
    cohort: str,
    outcome: str,
    reason: str,
    source_data: str,
    sex_col: str = "",
    outcome_col: str = "",
    education_col: str = "",
    n_total: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "outcome": outcome,
        "status": "not_feasible",
        "reason": reason,
        "sex_col": sex_col,
        "outcome_col": outcome_col,
        "education_col": education_col,
        "n_total": int(n_total),
        "n_used": 0,
        "n_male": 0,
        "n_female": 0,
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_g_income_wealth(
    *,
    root: Path,
    cohorts: list[str] | None = None,
    output_path: Path = Path("outputs/tables/g_income_wealth_associations.csv"),
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = _resolve_path(paths_cfg.get("processed_dir", "data/processed"), root)
    selected = [c for c in COHORT_CONFIGS if cohorts is None or c in set(cohorts)]

    rows: list[dict[str, Any]] = []
    for cohort in selected:
        cohort_cfg = load_yaml(root / COHORT_CONFIGS[cohort])
        sample_cfg = cohort_cfg.get("sample_construct", {}) if isinstance(cohort_cfg.get("sample_construct", {}), dict) else {}
        sex_col = str(sample_cfg.get("sex_col", "sex"))

        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{cohort}_cfa_resid_or_cfa.csv"

        if not source_path.exists():
            for outcome in OUTCOME_CANDIDATES:
                rows.append(_empty_row(cohort, outcome, "missing_source_data", source_data, sex_col=sex_col))
            continue

        df = pd.read_csv(source_path, low_memory=False)
        if sex_col not in df.columns:
            for outcome in OUTCOME_CANDIDATES:
                rows.append(_empty_row(cohort, outcome, "missing_sex_column", source_data, sex_col=sex_col, n_total=len(df)))
            continue

        indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", [])] if cohort == "cnlsy" else hierarchical_subtests(models_cfg)
        df = df.copy()
        df["__g_proxy"] = _g_proxy(df, indicators)
        df["__sex_label"] = df[sex_col].map(_normalize_sex)

        edu_col = _pick_col(df, EDUCATION_COL_CANDIDATES)
        if edu_col is not None:
            df["__edu_z"] = _zscore(df[edu_col])

        for outcome, outcome_cols in OUTCOME_CANDIDATES.items():
            out_col = _pick_col(df, outcome_cols)
            age_col = _pick_age_control_col(df, cohort, outcome)
            if out_col is None:
                rows.append(
                    _empty_row(
                        cohort,
                        outcome,
                        "missing_outcome_column",
                        source_data,
                        sex_col=sex_col,
                        education_col=edu_col or "",
                        n_total=len(df),
                    )
                )
                continue

            selected_cols = ["__g_proxy", "__sex_label", out_col]
            if age_col is not None:
                selected_cols.append(age_col)
            work = df.loc[df["__sex_label"].isin({"male", "female"}), selected_cols].copy()
            if edu_col is not None:
                work["__edu_z"] = df.loc[work.index, "__edu_z"]
            work["outcome"] = pd.to_numeric(work[out_col], errors="coerce")
            if age_col is not None:
                work["age_control"] = pd.to_numeric(work[age_col], errors="coerce")
            work["sex_male"] = (work["__sex_label"] == "male").astype(int)
            work["g_x_sex"] = pd.to_numeric(work["__g_proxy"], errors="coerce") * work["sex_male"].astype(float)
            work = work.dropna(subset=["__g_proxy", "outcome", "sex_male", "g_x_sex"])
            if age_col is not None:
                work = work.dropna(subset=["age_control"])
            if edu_col is not None:
                work = work.dropna(subset=["__edu_z"])

            n_male = int((work["sex_male"] == 1).sum())
            n_female = int((work["sex_male"] == 0).sum())
            if len(work) < 25 or n_male < 8 or n_female < 8:
                rows.append(
                    _empty_row(
                        cohort,
                        outcome,
                        "insufficient_complete_rows",
                        source_data,
                        sex_col=sex_col,
                        outcome_col=out_col,
                        education_col=edu_col or "",
                        n_total=len(df),
                    )
                    | {"n_used": int(len(work)), "n_male": n_male, "n_female": n_female}
                )
                continue

            x_cols = ["intercept", "__g_proxy", "sex_male", "g_x_sex"]
            design = pd.DataFrame({"intercept": 1.0}, index=work.index)
            design["__g_proxy"] = pd.to_numeric(work["__g_proxy"], errors="coerce")
            design["sex_male"] = work["sex_male"].astype(float)
            design["g_x_sex"] = work["g_x_sex"].astype(float)
            if age_col is not None:
                design["age_control"] = pd.to_numeric(work["age_control"], errors="coerce")
                x_cols.append("age_control")
            if edu_col is not None:
                design["__edu_z"] = pd.to_numeric(work["__edu_z"], errors="coerce")
                x_cols.append("__edu_z")
            design = design[x_cols].copy()

            fit, fit_reason = _ols_fit(work["outcome"], design)
            if fit is None:
                rows.append(
                    _empty_row(
                        cohort,
                        outcome,
                        f"ols_failed:{fit_reason or 'unknown'}",
                        source_data,
                        sex_col=sex_col,
                        outcome_col=out_col,
                        education_col=edu_col or "",
                        n_total=len(df),
                    )
                    | {"n_used": int(len(work)), "n_male": n_male, "n_female": n_female}
                )
                continue

            beta = fit["beta"]
            se = fit["se"]
            p_vals = fit["p"]
            name_to_idx = {name: idx for idx, name in enumerate(x_cols)}

            row = {
                "cohort": cohort,
                "outcome": outcome,
                "status": "computed",
                "reason": pd.NA,
                "sex_col": sex_col,
                "outcome_col": out_col,
                "education_col": edu_col if edu_col is not None else pd.NA,
                "n_total": int(len(df)),
                "n_used": int(fit["n_used"]),
                "n_male": n_male,
                "n_female": n_female,
                "corr_all": _safe_corr(work["__g_proxy"], work["outcome"]),
                "corr_male": _safe_corr(work.loc[work["sex_male"] == 1, "__g_proxy"], work.loc[work["sex_male"] == 1, "outcome"]),
                "corr_female": _safe_corr(
                    work.loc[work["sex_male"] == 0, "__g_proxy"], work.loc[work["sex_male"] == 0, "outcome"]
                ),
                "beta_g": float(beta[name_to_idx["__g_proxy"]]),
                "SE_beta_g": float(se[name_to_idx["__g_proxy"]]),
                "p_value_beta_g": float(p_vals[name_to_idx["__g_proxy"]])
                if math.isfinite(float(p_vals[name_to_idx["__g_proxy"]]))
                else pd.NA,
                "beta_gxsex": float(beta[name_to_idx["g_x_sex"]]),
                "SE_beta_gxsex": float(se[name_to_idx["g_x_sex"]]),
                "p_value_beta_gxsex": float(p_vals[name_to_idx["g_x_sex"]])
                if math.isfinite(float(p_vals[name_to_idx["g_x_sex"]]))
                else pd.NA,
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

    output_file = output_path if output_path.is_absolute() else root / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build g-to-income/wealth outcome association diagnostics by cohort.")
    parser.add_argument("--cohort", action="append", choices=tuple(COHORT_CONFIGS.keys()), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables/g_income_wealth_associations.csv"),
        help="Output CSV path (relative to project-root if not absolute).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    cohorts = _cohorts_from_args(args)
    try:
        out = run_g_income_wealth(root=root, cohorts=cohorts, output_path=args.output_path)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.output_path} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
