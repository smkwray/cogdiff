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

from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

RACE_COL_CANDIDATES = (
    "race_ethnicity",
    "race_ethnicity_3cat",
    "race_ethnicity_4cat",
    "race_ethnic",
    "race_group",
    "race_cat",
    "race",
    "ethnicity",
    "hispanic",
)

SUMMARY_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "g_construct",
    "race_col",
    "n_total",
    "n_groups_total",
    "n_groups_used",
    "heterogeneity_Q",
    "heterogeneity_df",
    "heterogeneity_p_value",
    "mean_d_across_race_groups",
    "mean_d_g_proxy_across_race_groups",
    "min_d_race_group",
    "max_d_race_group",
    "source_data",
]

DETAIL_COLUMNS = [
    "cohort",
    "race_group",
    "status",
    "reason",
    "g_construct",
    "n_total",
    "n_male",
    "n_female",
    "mean_all",
    "var_all",
    "mean_male",
    "mean_female",
    "var_male",
    "var_female",
    "d_g",
    "d_g_proxy",
    "SE_d_g",
    "ci_low_d_g",
    "ci_high_d_g",
    "VR_g_proxy",
    "log_VR_g_proxy",
    "SE_log_VR",
    "ci_low_VR",
    "ci_high_VR",
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
        return pd.Series([pd.NA] * len(vals), index=vals.index, dtype="float64")
    return (vals - mean) / sd


def _g_proxy(df: pd.DataFrame, indicators: list[str]) -> pd.Series:
    existing = [col for col in indicators if col in df.columns]
    if not existing:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
    z = pd.DataFrame({col: _zscore(df[col]) for col in existing}, index=df.index)
    return z.mean(axis=1, skipna=False)


def _pick_race_col(df: pd.DataFrame) -> str | None:
    for col in RACE_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _empty_summary_row(cohort: str, reason: str, source_data: str, race_col: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "status": "not_feasible",
        "reason": reason,
        "g_construct": "g_proxy",
        "race_col": race_col,
        "n_total": 0,
        "n_groups_total": 0,
        "n_groups_used": 0,
        "source_data": source_data,
    }
    for col in SUMMARY_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _empty_detail_row(cohort: str, reason: str, source_data: str, race_group: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "race_group": race_group,
        "status": "not_feasible",
        "reason": reason,
        "g_construct": "g_proxy",
        "source_data": source_data,
    }
    for col in DETAIL_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _compute_group_detail(cohort: str, race_group: str, values: pd.Series, sex: pd.Series, source_data: str) -> dict[str, Any]:
    clean = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "sex": sex.map(_normalize_sex)}).dropna()
    clean = clean[clean["sex"].isin({"male", "female"})].copy()

    male = clean.loc[clean["sex"] == "male", "value"]
    female = clean.loc[clean["sex"] == "female", "value"]
    n_male = int(len(male))
    n_female = int(len(female))
    n_total = int(len(clean))

    mean_all = float(clean["value"].mean()) if n_total > 0 else float("nan")
    var_all = float(clean["value"].var(ddof=1)) if n_total > 1 else float("nan")
    if n_male < 2 or n_female < 2:
        return _empty_detail_row(cohort, "insufficient_group_n", source_data, race_group=race_group) | {
            "n_total": n_total,
            "n_male": n_male,
            "n_female": n_female,
            "mean_all": mean_all,
            "var_all": var_all,
        }

    mean_male = float(male.mean())
    mean_female = float(female.mean())
    var_male = float(male.var(ddof=1))
    var_female = float(female.var(ddof=1))
    if var_male <= 0.0 or var_female <= 0.0:
        return _empty_detail_row(cohort, "nonpositive_group_variance", source_data, race_group=race_group) | {
            "n_total": n_total,
            "n_male": n_male,
            "n_female": n_female,
            "mean_all": mean_all,
            "var_all": var_all,
            "mean_male": mean_male,
            "mean_female": mean_female,
            "var_male": var_male,
            "var_female": var_female,
        }

    pooled_sd = math.sqrt((var_male + var_female) / 2.0)
    if pooled_sd <= 0.0:
        return _empty_detail_row(cohort, "nonpositive_pooled_sd", source_data, race_group=race_group) | {
            "n_total": n_total,
            "n_male": n_male,
            "n_female": n_female,
            "mean_all": mean_all,
            "var_all": var_all,
            "mean_male": mean_male,
            "mean_female": mean_female,
            "var_male": var_male,
            "var_female": var_female,
        }

    d_g = (mean_male - mean_female) / pooled_sd
    n_combined = n_male + n_female
    se_d = math.sqrt((n_combined / (n_male * n_female)) + (d_g**2 / (2.0 * (n_combined - 2))))
    ci_low = d_g - 1.96 * se_d
    ci_high = d_g + 1.96 * se_d

    vr = var_male / var_female
    log_vr = math.log(vr)
    se_log_vr = math.sqrt(2.0 / (n_male - 1) + 2.0 / (n_female - 1))
    ci_low_vr = math.exp(log_vr - 1.96 * se_log_vr)
    ci_high_vr = math.exp(log_vr + 1.96 * se_log_vr)

    row = {
        "cohort": cohort,
        "race_group": race_group,
        "status": "computed",
        "reason": pd.NA,
        "g_construct": "g_proxy",
        "n_total": n_total,
        "n_male": n_male,
        "n_female": n_female,
        "mean_all": mean_all,
        "var_all": var_all,
        "mean_male": mean_male,
        "mean_female": mean_female,
        "var_male": var_male,
        "var_female": var_female,
        "d_g": d_g,
        "d_g_proxy": d_g,
        "SE_d_g": se_d,
        "ci_low_d_g": ci_low,
        "ci_high_d_g": ci_high,
        "VR_g_proxy": vr,
        "log_VR_g_proxy": log_vr,
        "SE_log_VR": se_log_vr,
        "ci_low_VR": ci_low_vr,
        "ci_high_VR": ci_high_vr,
        "source_data": source_data,
    }
    for col in DETAIL_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_race_sex_interactions(
    *,
    root: Path,
    cohorts: list[str],
    summary_output_path: Path = Path("outputs/tables/race_sex_interaction_summary.csv"),
    detail_output_path: Path = Path("outputs/tables/race_sex_group_estimates.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config" / "paths.yml")
    models_cfg = load_yaml(root / "config" / "models.yml")
    processed_dir = _resolve_path(paths_cfg.get("processed_dir", "data/processed"), root)

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for cohort in cohorts:
        config_rel = COHORT_CONFIGS.get(cohort)
        if config_rel is None:
            summary_rows.append(_empty_summary_row(cohort, "unknown_cohort", ""))
            continue

        cohort_cfg = load_yaml(root / config_rel)
        sample_cfg = cohort_cfg.get("sample_construct", {}) if isinstance(cohort_cfg.get("sample_construct", {}), dict) else {}
        sex_col = str(sample_cfg.get("sex_col", "sex"))
        indicators = [_as_label(x) for x in sample_cfg.get("subtests", [])]
        if not indicators:
            indicators = [_as_label(x) for x in (models_cfg.get("cnlsy_single_factor", []) if cohort == "cnlsy" else hierarchical_subtests(models_cfg))]

        data_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not data_path.exists():
            data_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(data_path.relative_to(root)) if data_path.exists() else ""
        if not data_path.exists():
            summary_rows.append(_empty_summary_row(cohort, "missing_source_data", source_data))
            detail_rows.append(_empty_detail_row(cohort, "missing_source_data", source_data))
            continue

        df = pd.read_csv(data_path, low_memory=False)
        if sex_col not in df.columns:
            summary_rows.append(_empty_summary_row(cohort, "missing_sex_column", source_data))
            detail_rows.append(_empty_detail_row(cohort, "missing_sex_column", source_data))
            continue

        race_col = _pick_race_col(df)
        if race_col is None:
            summary_rows.append(_empty_summary_row(cohort, "missing_race_column", source_data))
            detail_rows.append(_empty_detail_row(cohort, "missing_race_column", source_data))
            continue

        g = _g_proxy(df, indicators)
        work = pd.DataFrame(
            {
                "race": df[race_col].astype("string").str.strip(),
                "sex": df[sex_col],
                "g": pd.to_numeric(g, errors="coerce"),
            }
        ).dropna(subset=["race", "g"]).copy()
        work = work[~work["race"].isin({"", "NA", "NaN", "nan", "unknown", "Unknown"})].copy()

        if work.empty:
            summary_rows.append(_empty_summary_row(cohort, "no_valid_rows_after_cleaning", source_data, race_col=race_col))
            detail_rows.append(_empty_detail_row(cohort, "no_valid_rows_after_cleaning", source_data))
            continue

        race_groups = sorted(work["race"].astype(str).unique().tolist())
        for rg in race_groups:
            group_slice = work[work["race"].astype(str) == rg].copy()
            detail_rows.append(
                _compute_group_detail(
                    cohort=cohort,
                    race_group=str(rg),
                    values=group_slice["g"],
                    sex=group_slice["sex"],
                    source_data=source_data,
                )
            )

        details_df = pd.DataFrame([r for r in detail_rows if str(r.get("cohort")) == cohort])
        computed = details_df[details_df["status"] == "computed"].copy() if not details_df.empty else pd.DataFrame()
        n_groups_total = len(race_groups)
        n_groups_used = int(computed.shape[0]) if not computed.empty else 0

        if n_groups_used < 2:
            summary_rows.append(
                _empty_summary_row(
                    cohort,
                    "insufficient_race_groups_for_interaction",
                    source_data,
                    race_col=race_col,
                )
                | {
                    "n_total": int(work.shape[0]),
                    "n_groups_total": int(n_groups_total),
                    "n_groups_used": int(n_groups_used),
                }
            )
            continue

        d_vals = pd.to_numeric(computed["d_g"], errors="coerce")
        se_vals = pd.to_numeric(computed["SE_d_g"], errors="coerce")
        valid = pd.DataFrame({"d": d_vals, "se": se_vals}).dropna()
        valid = valid[valid["se"] > 0.0].copy()
        if valid.shape[0] < 2:
            summary_rows.append(
                _empty_summary_row(
                    cohort,
                    "invalid_group_standard_errors",
                    source_data,
                    race_col=race_col,
                )
                | {
                    "n_total": int(work.shape[0]),
                    "n_groups_total": int(n_groups_total),
                    "n_groups_used": int(n_groups_used),
                }
            )
            continue

        weights = 1.0 / (valid["se"] ** 2)
        d_bar = float((weights * valid["d"]).sum() / weights.sum())
        q_stat = float((weights * ((valid["d"] - d_bar) ** 2)).sum())
        df_q = int(valid.shape[0] - 1)
        p_val = float(chi2.sf(q_stat, df_q)) if df_q > 0 else pd.NA

        row = {
            "cohort": cohort,
            "status": "computed",
            "reason": pd.NA,
            "g_construct": "g_proxy",
            "race_col": race_col,
            "n_total": int(work.shape[0]),
            "n_groups_total": int(n_groups_total),
            "n_groups_used": int(n_groups_used),
            "heterogeneity_Q": q_stat,
            "heterogeneity_df": df_q,
            "heterogeneity_p_value": p_val,
            "mean_d_across_race_groups": d_bar,
            "mean_d_g_proxy_across_race_groups": d_bar,
            "min_d_race_group": float(valid["d"].min()),
            "max_d_race_group": float(valid["d"].max()),
            "source_data": source_data,
        }
        for col in SUMMARY_COLUMNS:
            row.setdefault(col, pd.NA)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
    for col in SUMMARY_COLUMNS:
        if col not in summary_df.columns:
            summary_df[col] = pd.NA
    summary_df = summary_df[SUMMARY_COLUMNS].copy()

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        detail_df = pd.DataFrame(columns=DETAIL_COLUMNS)
    for col in DETAIL_COLUMNS:
        if col not in detail_df.columns:
            detail_df[col] = pd.NA
    detail_df = detail_df[DETAIL_COLUMNS].copy()

    summary_output = summary_output_path if summary_output_path.is_absolute() else root / summary_output_path
    detail_output = detail_output_path if detail_output_path.is_absolute() else root / detail_output_path
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    detail_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output, index=False)
    detail_df.to_csv(detail_output, index=False)
    return summary_df, detail_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build race×sex interaction diagnostics for latent g proxy by cohort."
    )
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=Path("outputs/tables/race_sex_interaction_summary.csv"),
        help="Summary output CSV path (relative to project-root if not absolute).",
    )
    parser.add_argument(
        "--detail-output-path",
        type=Path,
        default=Path("outputs/tables/race_sex_group_estimates.csv"),
        help="Detail output CSV path (relative to project-root if not absolute).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    cohorts = _cohorts_from_args(args)
    try:
        summary_df, detail_df = run_race_sex_interactions(
            root=root,
            cohorts=cohorts,
            summary_output_path=args.summary_output_path,
            detail_output_path=args.detail_output_path,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    summary_output = args.summary_output_path if args.summary_output_path.is_absolute() else root / args.summary_output_path
    detail_output = args.detail_output_path if args.detail_output_path.is_absolute() else root / args.detail_output_path
    computed = int((summary_df["status"] == "computed").sum()) if "status" in summary_df.columns else 0
    print(f"[ok] wrote {summary_output}")
    print(f"[ok] wrote {detail_output}")
    print(f"[ok] computed cohort rows: {computed}")
    print(f"[ok] detail rows: {len(detail_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
