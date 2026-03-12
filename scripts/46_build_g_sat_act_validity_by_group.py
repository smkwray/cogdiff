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

from nls_pipeline.exploratory import (
    build_ses_bins,
    g_proxy,
    normalize_sex,
    ols_fit,
    safe_corr,
)
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT_CONFIGS = {"nlsy97": "config/nlsy97.yml"}

OUTCOME_CANDIDATES: dict[str, tuple[str, ...]] = {
    "sat_math_2007": ("sat_math_2007_bin",),
    "sat_verbal_2007": ("sat_verbal_2007_bin",),
    "act_2007": ("act_2007_bin",),
}

OUTPUT_COLUMNS = [
    "cohort",
    "outcome",
    "group_kind",
    "group_col",
    "group_value",
    "status",
    "reason",
    "outcome_col",
    "n_total",
    "n_used",
    "pearson",
    "spearman",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "r2",
    "source_data",
]


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _empty_row(
    cohort: str,
    outcome: str,
    group_kind: str,
    group_col: str,
    group_value: str,
    reason: str,
    source_data: str,
    *,
    outcome_col: str = "",
    n_total: int = 0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "cohort": cohort,
        "outcome": outcome,
        "group_kind": group_kind,
        "group_col": group_col,
        "group_value": group_value,
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


def _compute_group_rows(
    *,
    cohort: str,
    df: pd.DataFrame,
    group_kind: str,
    group_col: str,
    outcome: str,
    out_col: str,
    source_data: str,
    min_group_n: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    work = df[[group_col, "__g_proxy", out_col]].copy()
    work["group_value"] = work[group_col].astype("string").str.strip()
    work["outcome"] = pd.to_numeric(work[out_col], errors="coerce").mask(lambda s: s <= 0)
    work = work.dropna(subset=["group_value", "__g_proxy", "outcome"]).copy()
    work = work[~work["group_value"].isin({"", "NA", "NaN", "nan", "unknown", "Unknown"})].copy()

    if work.empty:
        rows.append(_empty_row(cohort, outcome, group_kind, group_col, "", "no_valid_rows_after_cleaning", source_data, outcome_col=out_col, n_total=len(df)))
        return rows

    for group_value, grp in work.groupby("group_value", sort=True):
        grp = grp.copy()
        if len(grp) < min_group_n:
            rows.append(
                _empty_row(
                    cohort,
                    outcome,
                    group_kind,
                    group_col,
                    str(group_value),
                    "insufficient_group_n",
                    source_data,
                    outcome_col=out_col,
                    n_total=len(df),
                )
                | {"n_used": int(len(grp))}
            )
            continue

        x = pd.DataFrame({"intercept": 1.0, "g": pd.to_numeric(grp["__g_proxy"], errors="coerce")}, index=grp.index)
        fit, reason = ols_fit(grp["outcome"], x)
        if fit is None:
            rows.append(
                _empty_row(
                    cohort,
                    outcome,
                    group_kind,
                    group_col,
                    str(group_value),
                    f"ols_failed:{reason or 'unknown'}",
                    source_data,
                    outcome_col=out_col,
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
            "group_kind": group_kind,
            "group_col": group_col,
            "group_value": str(group_value),
            "status": "computed",
            "reason": pd.NA,
            "outcome_col": out_col,
            "n_total": int(len(df)),
            "n_used": int(fit["n_used"]),
            "pearson": safe_corr(grp["__g_proxy"], grp["outcome"], method="pearson"),
            "spearman": safe_corr(grp["__g_proxy"], grp["outcome"], method="spearman"),
            "beta_g": float(beta[1]),
            "SE_beta_g": float(se[1]),
            "p_value_beta_g": float(p[1]) if math.isfinite(float(p[1])) else pd.NA,
            "r2": float(fit["r2"]) if math.isfinite(float(fit["r2"])) else pd.NA,
            "source_data": source_data,
        }
        for col in OUTPUT_COLUMNS:
            row.setdefault(col, pd.NA)
        rows.append(row)
    return rows


def run_grouped_validity(
    *,
    root: Path,
    min_group_n: int = 60,
    race_output_path: Path = Path("outputs/tables/g_sat_act_validity_by_race.csv"),
    ses_output_path: Path = Path("outputs/tables/g_sat_act_validity_by_ses.csv"),
    race_sex_output_path: Path = Path("outputs/tables/g_sat_act_validity_by_race_sex.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    cohort = "nlsy97"
    cohort_cfg = load_yaml(root / COHORT_CONFIGS[cohort])
    sample_cfg = cohort_cfg.get("sample_construct", {}) if isinstance(cohort_cfg.get("sample_construct", {}), dict) else {}

    source_path = processed_dir / f"{cohort}_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / f"{cohort}_cfa.csv"
    if not source_path.exists():
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        for out_path in (race_output_path, ses_output_path, race_sex_output_path):
            target = out_path if out_path.is_absolute() else root / out_path
            target.parent.mkdir(parents=True, exist_ok=True)
            empty.to_csv(target, index=False)
        return empty, empty.copy(), empty.copy()

    df = pd.read_csv(source_path, low_memory=False)
    source_data = str(source_path.relative_to(root))

    indicators = hierarchical_subtests(models_cfg)
    df = df.copy()
    df["__g_proxy"] = g_proxy(df, indicators)
    df["__sex_norm"] = df[str(sample_cfg.get("sex_col", "sex"))].map(normalize_sex)
    if "race_ethnicity_3cat" in df.columns:
        df["__race_group"] = df["race_ethnicity_3cat"].astype("string")
    else:
        df["__race_group"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    ses_bins, _, _ = build_ses_bins(df["parent_education"] if "parent_education" in df.columns else pd.Series([pd.NA] * len(df)))
    df["__ses_bin"] = ses_bins.astype("string")
    df["__race_sex_group"] = (
        df["__race_group"].astype("string").fillna("")
        + "|"
        + df["__sex_norm"].astype("string").fillna("")
    ).astype("string")

    race_rows: list[dict[str, object]] = []
    ses_rows: list[dict[str, object]] = []
    race_sex_rows: list[dict[str, object]] = []
    for outcome, candidates in OUTCOME_CANDIDATES.items():
        out_col = _pick_col(df, candidates)
        if out_col is None:
            race_rows.append(_empty_row(cohort, outcome, "race", "race_ethnicity_3cat", "", "missing_outcome_column", source_data, n_total=len(df)))
            ses_rows.append(_empty_row(cohort, outcome, "ses", "parent_education", "", "missing_outcome_column", source_data, n_total=len(df)))
            race_sex_rows.append(_empty_row(cohort, outcome, "race_sex", "race_ethnicity_3cat|sex", "", "missing_outcome_column", source_data, n_total=len(df)))
            continue

        race_rows.extend(
            _compute_group_rows(
                cohort=cohort,
                df=df,
                group_kind="race",
                group_col="__race_group",
                outcome=outcome,
                out_col=out_col,
                source_data=source_data,
                min_group_n=min_group_n,
            )
        )
        ses_rows.extend(
            _compute_group_rows(
                cohort=cohort,
                df=df,
                group_kind="ses",
                group_col="__ses_bin",
                outcome=outcome,
                out_col=out_col,
                source_data=source_data,
                min_group_n=min_group_n,
            )
        )
        race_sex_rows.extend(
            _compute_group_rows(
                cohort=cohort,
                df=df,
                group_kind="race_sex",
                group_col="__race_sex_group",
                outcome=outcome,
                out_col=out_col,
                source_data=source_data,
                min_group_n=min_group_n,
            )
        )

    race_df = pd.DataFrame(race_rows)
    ses_df = pd.DataFrame(ses_rows)
    race_sex_df = pd.DataFrame(race_sex_rows)
    if race_df.empty:
        race_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if ses_df.empty:
        ses_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if race_sex_df.empty:
        race_sex_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    for out in (race_df, ses_df, race_sex_df):
        for col in OUTPUT_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
    race_df = race_df[OUTPUT_COLUMNS].copy()
    ses_df = ses_df[OUTPUT_COLUMNS].copy()
    race_sex_df = race_sex_df[OUTPUT_COLUMNS].copy()

    for frame, out_path in (
        (race_df, race_output_path),
        (ses_df, ses_output_path),
        (race_sex_df, race_sex_output_path),
    ):
        target = out_path if out_path.is_absolute() else root / out_path
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
    return race_df, ses_df, race_sex_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NLSY97 SAT/ACT validity tables by race, SES, and race-by-sex groups.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--min-group-n", type=int, default=60, help="Minimum subgroup n to compute a row.")
    parser.add_argument("--race-output-path", type=Path, default=Path("outputs/tables/g_sat_act_validity_by_race.csv"))
    parser.add_argument("--ses-output-path", type=Path, default=Path("outputs/tables/g_sat_act_validity_by_ses.csv"))
    parser.add_argument("--race-sex-output-path", type=Path, default=Path("outputs/tables/g_sat_act_validity_by_race_sex.csv"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        race_df, ses_df, race_sex_df = run_grouped_validity(
            root=root,
            min_group_n=int(args.min_group_n),
            race_output_path=args.race_output_path,
            ses_output_path=args.ses_output_path,
            race_sex_output_path=args.race_sex_output_path,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.race_output_path if args.race_output_path.is_absolute() else root / args.race_output_path}")
    print(f"[ok] wrote {args.ses_output_path if args.ses_output_path.is_absolute() else root / args.ses_output_path}")
    print(f"[ok] wrote {args.race_sex_output_path if args.race_sex_output_path.is_absolute() else root / args.race_sex_output_path}")
    print(f"[ok] computed race rows: {int((race_df['status'] == 'computed').sum()) if 'status' in race_df.columns else 0}")
    print(f"[ok] computed ses rows: {int((ses_df['status'] == 'computed').sum()) if 'status' in ses_df.columns else 0}")
    print(f"[ok] computed race-sex rows: {int((race_sex_df['status'] == 'computed').sum()) if 'status' in race_sex_df.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
