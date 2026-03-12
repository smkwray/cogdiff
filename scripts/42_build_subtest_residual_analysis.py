#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

OUTPUT_COLUMNS = [
    "cohort",
    "subtest",
    "status",
    "reason",
    "n_total",
    "n_male",
    "n_female",
    "raw_d_subtest",
    "resid_d_subtest",
    "delta_d_raw_minus_resid",
    "raw_mean_male",
    "raw_mean_female",
    "resid_mean_male",
    "resid_mean_female",
    "predictors_used",
    "source_data",
]


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _as_label(value: Any) -> str:
    if isinstance(value, bool):
        return "NO" if value is False else "YES"
    return str(value)


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
        return pd.Series([np.nan] * len(vals), index=vals.index, dtype="float64")
    return (vals - mean) / sd


def _factor_groups(models_cfg: dict[str, Any]) -> dict[str, list[str]]:
    groups = models_cfg.get("hierarchical_factors", {})
    if not isinstance(groups, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key in ("speed", "math", "verbal", "technical"):
        vals = groups.get(key, [])
        if not isinstance(vals, list):
            vals = []
        out[key] = [_as_label(v) for v in vals]
    return out


def _compute_d(values: pd.Series, sex: pd.Series) -> tuple[float | None, float | None, float | None]:
    clean = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "sex": sex.map(_normalize_sex)}).dropna()
    clean = clean[clean["sex"].isin({"male", "female"})].copy()
    male = clean.loc[clean["sex"] == "male", "value"]
    female = clean.loc[clean["sex"] == "female", "value"]
    if len(male) < 2 or len(female) < 2:
        return None, None, None
    m_mean = float(male.mean())
    f_mean = float(female.mean())
    m_var = float(male.var(ddof=1))
    f_var = float(female.var(ddof=1))
    if m_var <= 0.0 or f_var <= 0.0:
        return None, m_mean, f_mean
    pooled = math.sqrt((m_var + f_var) / 2.0)
    if pooled <= 0.0:
        return None, m_mean, f_mean
    return float((m_mean - f_mean) / pooled), m_mean, f_mean


def _ols_residuals(y: pd.Series, predictors: pd.DataFrame) -> pd.Series:
    y_num = pd.to_numeric(y, errors="coerce")
    x_num = predictors.apply(pd.to_numeric, errors="coerce")
    keep = y_num.notna()
    for c in x_num.columns:
        keep &= x_num[c].notna()
    if int(keep.sum()) < max(8, x_num.shape[1] + 2):
        return pd.Series([np.nan] * len(y_num), index=y_num.index, dtype="float64")

    y_fit = y_num.loc[keep].to_numpy(dtype=float)
    x_fit = x_num.loc[keep].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x_fit)), x_fit])
    beta, *_ = np.linalg.lstsq(x_design, y_fit, rcond=None)
    fitted = x_design @ beta
    resid = y_fit - fitted
    out = pd.Series([np.nan] * len(y_num), index=y_num.index, dtype="float64")
    out.loc[keep] = resid
    return out


def _empty_row(cohort: str, subtest: str, reason: str, source_data: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "subtest": subtest,
        "status": "not_feasible",
        "reason": reason,
        "n_total": 0,
        "n_male": 0,
        "n_female": 0,
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_subtest_residual_analysis(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path = Path("outputs/tables/subtest_residual_analysis.csv"),
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config" / "paths.yml")
    models_cfg = load_yaml(root / "config" / "models.yml")
    processed_dir = _resolve_path(paths_cfg.get("processed_dir", "data/processed"), root)
    factor_groups = _factor_groups(models_cfg)

    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        cohort_cfg = load_yaml(root / COHORT_CONFIGS[cohort])
        sample_cfg = cohort_cfg.get("sample_construct", {}) if isinstance(cohort_cfg.get("sample_construct", {}), dict) else {}
        sex_col = str(sample_cfg.get("sex_col", "sex"))
        subtests = [_as_label(x) for x in sample_cfg.get("subtests", [])]
        if not subtests:
            rows.append(_empty_row(cohort, "", "missing_subtests_config", ""))
            continue

        data_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not data_path.exists():
            data_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(data_path.relative_to(root)) if data_path.exists() else ""
        if not data_path.exists():
            rows.append(_empty_row(cohort, "", "missing_source_data", source_data))
            continue

        df = pd.read_csv(data_path, low_memory=False)
        if sex_col not in df.columns:
            rows.append(_empty_row(cohort, "", "missing_sex_column", source_data))
            continue

        available = [s for s in subtests if s in df.columns]
        if not available:
            rows.append(_empty_row(cohort, "", "no_subtests_present", source_data))
            continue

        z_all = pd.DataFrame({s: _zscore(df[s]) for s in available}, index=df.index)
        sex = df[sex_col]

        for target in available:
            y = pd.to_numeric(df[target], errors="coerce")
            others = [s for s in available if s != target]
            predictors: dict[str, pd.Series] = {}

            if others:
                predictors["g_other"] = z_all[others].mean(axis=1, skipna=False)
            for factor_name, tests in factor_groups.items():
                tests_usable = [t for t in tests if t in available and t != target]
                if tests_usable:
                    predictors[f"{factor_name}_other"] = z_all[tests_usable].mean(axis=1, skipna=False)

            if not predictors:
                rows.append(_empty_row(cohort, target, "no_predictors_for_residualization", source_data))
                continue

            x = pd.DataFrame(predictors, index=df.index)
            resid = _ols_residuals(y, x)
            raw_d, raw_m, raw_f = _compute_d(y, sex)
            resid_d, resid_m, resid_f = _compute_d(resid, sex)

            n_male = int((sex.map(_normalize_sex) == "male").sum())
            n_female = int((sex.map(_normalize_sex) == "female").sum())
            n_total = int((sex.map(_normalize_sex).isin({"male", "female"})).sum())

            if raw_d is None or resid_d is None:
                rows.append(_empty_row(cohort, target, "insufficient_group_n_or_variance", source_data) | {
                    "n_total": n_total,
                    "n_male": n_male,
                    "n_female": n_female,
                    "predictors_used": ",".join(x.columns.tolist()),
                })
                continue

            row = {
                "cohort": cohort,
                "subtest": target,
                "status": "computed",
                "reason": pd.NA,
                "n_total": n_total,
                "n_male": n_male,
                "n_female": n_female,
                "raw_d_subtest": float(raw_d),
                "resid_d_subtest": float(resid_d),
                "delta_d_raw_minus_resid": float(raw_d - resid_d),
                "raw_mean_male": float(raw_m) if raw_m is not None else pd.NA,
                "raw_mean_female": float(raw_f) if raw_f is not None else pd.NA,
                "resid_mean_male": float(resid_m) if resid_m is not None else pd.NA,
                "resid_mean_female": float(resid_f) if resid_f is not None else pd.NA,
                "predictors_used": ",".join(x.columns.tolist()),
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
    parser = argparse.ArgumentParser(
        description="Build subtest sex-difference analysis after residualizing out g/group-factor proxy structure."
    )
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables/subtest_residual_analysis.csv"),
        help="Output CSV path (relative to project-root if not absolute).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    cohorts = _cohorts_from_args(args)
    try:
        out = run_subtest_residual_analysis(
            root=root,
            cohorts=cohorts,
            output_path=args.output_path,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    output_file = args.output_path if args.output_path.is_absolute() else root / args.output_path
    computed = int((out["status"] == "computed").sum()) if "status" in out.columns else 0
    print(f"[ok] wrote {output_file}")
    print(f"[ok] computed rows: {computed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
