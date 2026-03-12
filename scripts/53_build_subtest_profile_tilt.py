#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.exploratory import g_proxy, normalize_sex, ols_fit, safe_corr, zscore
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
}

OUTPUT_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "n_total",
    "n_used_education",
    "d_tilt",
    "se_d_tilt",
    "tilt_g_corr",
    "tilt_incremental_r2_education",
    "p_tilt_incremental",
    "source_data",
]


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _empty_row(cohort: str, reason: str, source_data: str, *, n_total: int = 0, n_used_education: int = 0) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "status": "not_feasible",
        "reason": reason,
        "n_total": int(n_total),
        "n_used_education": int(n_used_education),
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _tilt_effect_size(tilt: pd.Series, sex: pd.Series) -> tuple[float | None, float | None]:
    work = pd.DataFrame({"tilt": pd.to_numeric(tilt, errors="coerce"), "sex": sex.map(normalize_sex)}).dropna()
    male = work.loc[work["sex"] == "male", "tilt"]
    female = work.loc[work["sex"] == "female", "tilt"]
    n_male = int(len(male))
    n_female = int(len(female))
    if n_male < 2 or n_female < 2:
        return None, None
    var_male = float(male.var(ddof=1))
    var_female = float(female.var(ddof=1))
    if var_male <= 0.0 or var_female <= 0.0:
        return None, None
    pooled_var = (((n_male - 1) * var_male) + ((n_female - 1) * var_female)) / float(n_male + n_female - 2)
    if pooled_var <= 0.0:
        return None, None
    pooled_sd = math.sqrt(pooled_var)
    d = float((male.mean() - female.mean()) / pooled_sd)
    se = math.sqrt(((n_male + n_female) / float(n_male * n_female)) + ((d * d) / float(2 * (n_male + n_female - 2))))
    return d, se


def run_subtest_profile_tilt(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path = Path("outputs/tables/subtest_profile_tilt.csv"),
    min_n: int = 40,
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
        source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{cohort}_cfa_resid_or_cfa.csv"
        if not source_path.exists():
            rows.append(_empty_row(cohort, "missing_source_data", source_data))
            continue

        df = pd.read_csv(source_path, low_memory=False)
        required = ["WK", "PC", "AR", "MK", "sex"]
        if any(col not in df.columns for col in required):
            rows.append(_empty_row(cohort, "missing_tilt_inputs", source_data, n_total=len(df)))
            continue

        work = df.copy()
        work["__g_proxy"] = g_proxy(work, hierarchical_subtests(models_cfg))
        work["__verbal"] = pd.concat([zscore(work["WK"]), zscore(work["PC"])], axis=1).mean(axis=1, skipna=False)
        work["__quant"] = pd.concat([zscore(work["AR"]), zscore(work["MK"])], axis=1).mean(axis=1, skipna=False)
        work["__tilt"] = work["__verbal"] - work["__quant"]

        d_tilt, se_d_tilt = _tilt_effect_size(work["__tilt"], work["sex"])
        tilt_g_corr = safe_corr(work["__tilt"], work["__g_proxy"])

        edu_col = "education_years" if "education_years" in work.columns else None
        if edu_col is None:
            rows.append(_empty_row(cohort, "missing_education_outcome", source_data, n_total=len(df)))
            continue

        model_df = pd.DataFrame(
            {
                "education": pd.to_numeric(work[edu_col], errors="coerce"),
                "g": pd.to_numeric(work["__g_proxy"], errors="coerce"),
                "tilt": pd.to_numeric(work["__tilt"], errors="coerce"),
            }
        ).dropna()
        if len(model_df) < min_n:
            rows.append(_empty_row(cohort, "insufficient_rows", source_data, n_total=len(df), n_used_education=len(model_df)))
            continue

        base_x = pd.DataFrame({"intercept": 1.0, "g": model_df["g"]}, index=model_df.index)
        full_x = pd.DataFrame({"intercept": 1.0, "g": model_df["g"], "tilt": model_df["tilt"]}, index=model_df.index)
        base_fit, base_reason = ols_fit(model_df["education"], base_x)
        full_fit, full_reason = ols_fit(model_df["education"], full_x)
        if base_fit is None or full_fit is None:
            reason = base_reason if base_fit is None else full_reason
            rows.append(_empty_row(cohort, f"ols_failed:{reason or 'unknown'}", source_data, n_total=len(df), n_used_education=len(model_df)))
            continue

        p_vals = full_fit["p"]
        row = {
            "cohort": cohort,
            "status": "computed",
            "reason": pd.NA,
            "n_total": int(len(df)),
            "n_used_education": int(full_fit["n_used"]),
            "d_tilt": d_tilt if d_tilt is not None and math.isfinite(float(d_tilt)) else pd.NA,
            "se_d_tilt": se_d_tilt if se_d_tilt is not None and math.isfinite(float(se_d_tilt)) else pd.NA,
            "tilt_g_corr": tilt_g_corr if tilt_g_corr is not None and math.isfinite(float(tilt_g_corr)) else pd.NA,
            "tilt_incremental_r2_education": float(full_fit["r2"] - base_fit["r2"]) if math.isfinite(float(full_fit["r2"])) and math.isfinite(float(base_fit["r2"])) else pd.NA,
            "p_tilt_incremental": float(p_vals[2]) if math.isfinite(float(p_vals[2])) else pd.NA,
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
    parser = argparse.ArgumentParser(description="Build verbal-minus-quantitative subtest profile tilt outputs.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all supported cohorts.")
    parser.add_argument("--min-n", type=int, default=40, help="Minimum usable rows for the education models.")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/subtest_profile_tilt.csv"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        out = run_subtest_profile_tilt(
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
