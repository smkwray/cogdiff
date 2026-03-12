#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Any

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

AGE_OUTCOME_SPECS: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "nlsy79": {
        "age_2000": {
            "education": ("education_years",),
            "household_income": ("household_income",),
            "net_worth": ("net_worth",),
            "employment": ("employment_2000",),
            "annual_earnings": ("annual_earnings",),
        }
    },
    "nlsy97": {
        "age_2010": {
            "education": ("education_years",),
            "household_income": ("household_income",),
            "net_worth": ("net_worth",),
            "employment": ("employment_2011",),
            "annual_earnings": ("annual_earnings",),
        },
        "age_2011": {
            "education": ("education_years",),
            "household_income": ("household_income",),
            "net_worth": ("net_worth",),
            "employment": ("employment_2011",),
            "annual_earnings": ("annual_earnings",),
        },
        "age_2019": {
            "household_income": ("household_income_2019",),
            "employment": ("employment_2019",),
            "annual_earnings": ("annual_earnings_2019",),
        },
        "age_2021": {
            "household_income": ("household_income_2021",),
            "employment": ("employment_2021",),
            "annual_earnings": ("annual_earnings_2021",),
        },
    },
}

WINDOW_COLUMNS = [
    "cohort",
    "age_col",
    "outcome",
    "status",
    "reason",
    "outcome_col",
    "n_total",
    "n_used",
    "age_min",
    "age_p10",
    "age_median",
    "age_p90",
    "age_max",
    "source_data",
]

OVERLAP_COLUMNS = [
    "outcome",
    "cohort_a",
    "age_col_a",
    "cohort_b",
    "age_col_b",
    "status",
    "reason",
    "overlap_min",
    "overlap_max",
    "overlap_width_years",
    "n_a_in_overlap",
    "n_b_in_overlap",
    "source_windows",
]


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _empty_window(
    cohort: str,
    age_col: str,
    outcome: str,
    reason: str,
    source_data: str,
    *,
    outcome_col: str = "",
    n_total: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "age_col": age_col,
        "outcome": outcome,
        "status": "not_feasible",
        "reason": reason,
        "outcome_col": outcome_col,
        "n_total": int(n_total),
        "n_used": 0,
        "source_data": source_data,
    }
    for col in WINDOW_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _empty_overlap(
    outcome: str,
    cohort_a: str,
    age_col_a: str,
    cohort_b: str,
    age_col_b: str,
    reason: str,
    *,
    source_windows: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "outcome": outcome,
        "cohort_a": cohort_a,
        "age_col_a": age_col_a,
        "cohort_b": cohort_b,
        "age_col_b": age_col_b,
        "status": "not_feasible",
        "reason": reason,
        "source_windows": source_windows,
    }
    for col in OVERLAP_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _window_row(cohort: str, age_col: str, outcome: str, outcome_col: str, ages: pd.Series, source_data: str, n_total: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "age_col": age_col,
        "outcome": outcome,
        "status": "computed",
        "reason": pd.NA,
        "outcome_col": outcome_col,
        "n_total": int(n_total),
        "n_used": int(ages.shape[0]),
        "age_min": float(ages.min()),
        "age_p10": float(ages.quantile(0.10)),
        "age_median": float(ages.median()),
        "age_p90": float(ages.quantile(0.90)),
        "age_max": float(ages.max()),
        "source_data": source_data,
    }
    for col in WINDOW_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _build_overlap_rows(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    computed = windows.loc[windows["status"] == "computed"].copy()
    for outcome in sorted(computed["outcome"].dropna().astype(str).unique().tolist()):
        subset = computed.loc[computed["outcome"].astype(str) == outcome].copy()
        if subset.empty:
            continue
        nlsy79_rows = subset.loc[subset["cohort"].astype(str) == "nlsy79"].copy()
        nlsy97_rows = subset.loc[subset["cohort"].astype(str) == "nlsy97"].copy()
        if nlsy79_rows.empty or nlsy97_rows.empty:
            continue
        for _, row_a in nlsy79_rows.iterrows():
            for _, row_b in nlsy97_rows.iterrows():
                age_min = max(float(row_a["age_min"]), float(row_b["age_min"]))
                age_max = min(float(row_a["age_max"]), float(row_b["age_max"]))
                source_windows = f"{row_a['source_data']}::{row_a['age_col']}|{row_b['source_data']}::{row_b['age_col']}"
                if age_min > age_max:
                    rows.append(
                        _empty_overlap(
                            outcome,
                            str(row_a["cohort"]),
                            str(row_a["age_col"]),
                            str(row_b["cohort"]),
                            str(row_b["age_col"]),
                            "no_age_overlap",
                            source_windows=source_windows,
                        )
                    )
                    continue
                rows.append(
                    {
                        "outcome": outcome,
                        "cohort_a": str(row_a["cohort"]),
                        "age_col_a": str(row_a["age_col"]),
                        "cohort_b": str(row_b["cohort"]),
                        "age_col_b": str(row_b["age_col"]),
                        "status": "computed",
                        "reason": pd.NA,
                        "overlap_min": age_min,
                        "overlap_max": age_max,
                        "overlap_width_years": float(age_max - age_min),
                        "n_a_in_overlap": 0,
                        "n_b_in_overlap": 0,
                        "source_windows": source_windows,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=OVERLAP_COLUMNS)
    for col in OVERLAP_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[OVERLAP_COLUMNS].copy()


def run_age_matched_feasibility(
    *,
    root: Path,
    cohorts: list[str],
    windows_output_path: Path = Path("outputs/tables/age_matched_feasibility.csv"),
    overlap_output_path: Path = Path("outputs/tables/age_matched_overlap_summary.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{cohort}_cfa_resid_or_cfa.csv"
        age_specs = AGE_OUTCOME_SPECS.get(cohort, {})
        age_cols = tuple(age_specs.keys())
        if not source_path.exists():
            for age_col, outcome in product(age_cols, age_specs.get(age_col, {})):
                rows.append(_empty_window(cohort, age_col, outcome, "missing_source_data", source_data))
            continue
        if not age_cols:
            continue

        df = pd.read_csv(source_path, low_memory=False)
        for age_col in age_cols:
            outcomes = age_specs.get(age_col, {})
            for outcome, candidates in outcomes.items():
                outcome_col = _pick_col(df, candidates)
                if age_col not in df.columns:
                    rows.append(_empty_window(cohort, age_col, outcome, "missing_age_column", source_data, outcome_col=outcome_col or "", n_total=len(df)))
                    continue
                if outcome_col is None:
                    rows.append(_empty_window(cohort, age_col, outcome, "missing_outcome_column", source_data, n_total=len(df)))
                    continue
                ages = pd.to_numeric(df[age_col], errors="coerce")
                outcome_vals = pd.to_numeric(df[outcome_col], errors="coerce")
                used = pd.DataFrame({"age": ages, "outcome": outcome_vals}).dropna()
                if used.empty:
                    rows.append(_empty_window(cohort, age_col, outcome, "no_complete_rows", source_data, outcome_col=outcome_col, n_total=len(df)))
                    continue
                rows.append(_window_row(cohort, age_col, outcome, outcome_col, used["age"], source_data, len(df)))

    windows = pd.DataFrame(rows)
    if windows.empty:
        windows = pd.DataFrame(columns=WINDOW_COLUMNS)
    for col in WINDOW_COLUMNS:
        if col not in windows.columns:
            windows[col] = pd.NA
    windows = windows[WINDOW_COLUMNS].copy()
    overlaps = _build_overlap_rows(windows)

    windows_full = root / windows_output_path
    windows_full.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(windows_full, index=False)

    overlap_full = root / overlap_output_path
    overlap_full.parent.mkdir(parents=True, exist_ok=True)
    overlaps.to_csv(overlap_full, index=False)
    return windows, overlaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize age-window feasibility for cross-cohort outcome comparisons.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--windows-output-path", type=Path, default=Path("outputs/tables/age_matched_feasibility.csv"))
    parser.add_argument("--overlap-output-path", type=Path, default=Path("outputs/tables/age_matched_overlap_summary.csv"))
    args = parser.parse_args(argv)

    root = Path(args.project_root)
    windows, overlaps = run_age_matched_feasibility(
        root=root,
        cohorts=_cohorts_from_args(args),
        windows_output_path=args.windows_output_path,
        overlap_output_path=args.overlap_output_path,
    )
    print(f"[ok] age feasibility rows computed: {int((windows['status'] == 'computed').sum())}")
    print(f"[ok] overlap rows computed: {int((overlaps['status'] == 'computed').sum())}")
    print(f"[ok] wrote {args.windows_output_path}")
    print(f"[ok] wrote {args.overlap_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
