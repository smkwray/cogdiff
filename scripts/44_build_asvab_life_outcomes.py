#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, t

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
    "employment": ("employed", "employment_status", "employed_indicator", "labor_force_participation"),
    "wages": ("hourly_wage", "wage", "annual_earnings", "earnings"),
    "education": ("education_years", "years_education", "highest_grade_completed", "education"),
    "health": ("self_rated_health", "health_index", "sf12_physical", "sf12_mental"),
}

OUTPUT_COLUMNS = [
    "cohort",
    "outcome",
    "status",
    "reason",
    "sex_col",
    "outcome_col",
    "n_total",
    "n_used",
    "n_male",
    "n_female",
    "beta_g_male",
    "SE_beta_g_male",
    "p_value_beta_g_male",
    "r2_male",
    "beta_g_female",
    "SE_beta_g_female",
    "p_value_beta_g_female",
    "r2_female",
    "delta_r2_male_minus_female",
    "z_delta_beta",
    "p_value_delta_beta",
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


def _ols_simple(y: pd.Series, x: pd.Series) -> tuple[dict[str, Any] | None, str | None]:
    y_num = pd.to_numeric(y, errors="coerce")
    x_num = pd.to_numeric(x, errors="coerce")
    mask = y_num.notna() & x_num.notna()
    if int(mask.sum()) < 3:
        return None, "insufficient_rows"

    yv = y_num[mask].to_numpy(dtype=float)
    xv = x_num[mask].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones_like(xv), xv])
    n, p = x_design.shape
    if n <= p:
        return None, "insufficient_rows_for_model"

    try:
        beta, _, _, _ = np.linalg.lstsq(x_design, yv, rcond=None)
    except np.linalg.LinAlgError:
        return None, "ols_singular_matrix"

    yhat = x_design @ beta
    resid = yv - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((yv - float(np.mean(yv))) ** 2))
    dof = n - p
    if dof <= 0:
        return None, "nonpositive_residual_dof"

    sigma2 = sse / float(dof)
    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    var_beta = sigma2 * xtx_inv
    se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))
    t_slope = float(beta[1] / se[1]) if float(se[1]) > 0.0 else np.nan
    p_slope = float(2.0 * t.sf(abs(t_slope), dof)) if math.isfinite(t_slope) else np.nan
    r2 = np.nan
    if sst > 0.0 and math.isfinite(sse):
        r2 = float(1.0 - (sse / sst))

    return {
        "n": int(n),
        "beta_slope": float(beta[1]),
        "se_slope": float(se[1]),
        "p_slope": p_slope,
        "r2": r2,
    }, None


def _empty_row(
    cohort: str,
    outcome: str,
    reason: str,
    source_data: str,
    sex_col: str = "",
    outcome_col: str = "",
    n_total: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "outcome": outcome,
        "status": "not_feasible",
        "reason": reason,
        "sex_col": sex_col,
        "outcome_col": outcome_col,
        "n_total": int(n_total),
        "n_used": 0,
        "n_male": 0,
        "n_female": 0,
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_asvab_life_outcomes(
    *,
    root: Path,
    cohorts: list[str] | None = None,
    output_path: Path = Path("outputs/tables/asvab_life_outcomes_by_sex.csv"),
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

        for outcome, outcome_cols in OUTCOME_CANDIDATES.items():
            out_col = _pick_col(df, outcome_cols)
            if out_col is None:
                rows.append(_empty_row(cohort, outcome, "missing_outcome_column", source_data, sex_col=sex_col, n_total=len(df)))
                continue

            work = df.loc[df["__sex_label"].isin({"male", "female"}), ["__g_proxy", "__sex_label", out_col]].copy()
            work["outcome"] = pd.to_numeric(work[out_col], errors="coerce")
            work = work.dropna(subset=["__g_proxy", "outcome"])

            male = work.loc[work["__sex_label"] == "male"].copy()
            female = work.loc[work["__sex_label"] == "female"].copy()
            n_male = int(len(male))
            n_female = int(len(female))
            n_used = int(len(work))
            if n_male < 10 or n_female < 10:
                rows.append(
                    _empty_row(cohort, outcome, "insufficient_group_rows", source_data, sex_col=sex_col, outcome_col=out_col, n_total=len(df))
                    | {"n_used": n_used, "n_male": n_male, "n_female": n_female}
                )
                continue

            fit_male, reason_male = _ols_simple(male["outcome"], male["__g_proxy"])
            fit_female, reason_female = _ols_simple(female["outcome"], female["__g_proxy"])
            if fit_male is None or fit_female is None:
                reason = "male_model_failed" if fit_male is None else "female_model_failed"
                detail = reason_male if fit_male is None else reason_female
                rows.append(
                    _empty_row(cohort, outcome, f"{reason}:{detail or 'unknown'}", source_data, sex_col=sex_col, outcome_col=out_col, n_total=len(df))
                    | {"n_used": n_used, "n_male": n_male, "n_female": n_female}
                )
                continue

            delta_beta = float(fit_male["beta_slope"]) - float(fit_female["beta_slope"])
            se_delta = math.sqrt(float(fit_male["se_slope"]) ** 2 + float(fit_female["se_slope"]) ** 2)
            z_delta = np.nan
            p_delta = np.nan
            if se_delta > 0.0 and math.isfinite(se_delta):
                z_delta = float(delta_beta / se_delta)
                p_delta = float(2.0 * norm.sf(abs(z_delta)))

            row = {
                "cohort": cohort,
                "outcome": outcome,
                "status": "computed",
                "reason": pd.NA,
                "sex_col": sex_col,
                "outcome_col": out_col,
                "n_total": int(len(df)),
                "n_used": n_used,
                "n_male": n_male,
                "n_female": n_female,
                "beta_g_male": float(fit_male["beta_slope"]),
                "SE_beta_g_male": float(fit_male["se_slope"]),
                "p_value_beta_g_male": float(fit_male["p_slope"]) if math.isfinite(float(fit_male["p_slope"])) else pd.NA,
                "r2_male": float(fit_male["r2"]) if math.isfinite(float(fit_male["r2"])) else pd.NA,
                "beta_g_female": float(fit_female["beta_slope"]),
                "SE_beta_g_female": float(fit_female["se_slope"]),
                "p_value_beta_g_female": float(fit_female["p_slope"])
                if math.isfinite(float(fit_female["p_slope"]))
                else pd.NA,
                "r2_female": float(fit_female["r2"]) if math.isfinite(float(fit_female["r2"])) else pd.NA,
                "delta_r2_male_minus_female": (
                    float(fit_male["r2"]) - float(fit_female["r2"])
                    if math.isfinite(float(fit_male["r2"])) and math.isfinite(float(fit_female["r2"]))
                    else pd.NA
                ),
                "z_delta_beta": z_delta if math.isfinite(float(z_delta)) else pd.NA,
                "p_value_delta_beta": p_delta if math.isfinite(float(p_delta)) else pd.NA,
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
    parser = argparse.ArgumentParser(description="Build ASVAB/g predictive-validity outcome summaries by sex.")
    parser.add_argument("--cohort", action="append", choices=tuple(COHORT_CONFIGS.keys()), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables/asvab_life_outcomes_by_sex.csv"),
        help="Output CSV path (relative to project-root if not absolute).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    cohorts = _cohorts_from_args(args)
    try:
        out = run_asvab_life_outcomes(root=root, cohorts=cohorts, output_path=args.output_path)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.output_path} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
