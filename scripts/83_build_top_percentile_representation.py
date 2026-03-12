#!/usr/bin/env python3
"""Compute representation of race/ethnicity and sex groups in the top 10%
(and top 1%) of cognitive and life-outcome distributions.

For each outcome × group, reports:
  - n_in_top: count in the top percentile
  - pct_of_top: share of the top percentile held by this group
  - pct_of_sample: share of the full sample held by this group
  - representation_ratio: pct_of_top / pct_of_sample (>1 = overrepresented)

Outputs: outputs/tables/top_percentile_representation.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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

# Outcomes to analyse per cohort
OUTCOME_SPECS: dict[str, list[tuple[str, str]]] = {
    "nlsy79": [
        ("g_proxy", "__g_proxy"),
        ("earnings", "annual_earnings"),
        ("household_income", "household_income"),
        ("net_worth", "net_worth"),
        ("education_years", "education_years"),
    ],
    "nlsy97": [
        ("g_proxy", "__g_proxy"),
        ("household_income", "household_income_2021"),
        ("net_worth", "net_worth"),
        ("education_years", "education_years"),
        ("earnings", "annual_earnings_2021"),
        ("sat_math", "sat_math_2007_bin"),
        ("sat_verbal", "sat_verbal_2007_bin"),
        ("act", "act_2007_bin"),
    ],
    "cnlsy": [
        ("g_proxy", "__g_proxy"),
        ("education_years", "education_years"),
    ],
}

PERCENTILE_THRESHOLDS = [90, 99]

OUTPUT_COLUMNS = [
    "cohort",
    "outcome",
    "percentile_threshold",
    "group_kind",
    "group_value",
    "status",
    "reason",
    "n_outcome_valid",
    "n_in_top",
    "n_group_total",
    "pct_of_top",
    "pct_of_sample",
    "representation_ratio",
    "threshold_value",
    "source_data",
]


def _normalize_sex(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"m", "male", "1", "man", "boy"}:
        return "male"
    if token in {"f", "female", "2", "woman", "girl"}:
        return "female"
    return ""


def run(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path,
    min_top_n: int = 10,
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

        for outcome_name, outcome_col in outcomes:
            if outcome_col not in df.columns:
                continue

            vals = pd.to_numeric(df[outcome_col], errors="coerce")
            valid_mask = vals.notna()
            n_valid = int(valid_mask.sum())
            if n_valid < 30:
                continue

            for pct in PERCENTILE_THRESHOLDS:
                threshold_val = float(np.percentile(vals[valid_mask], pct))
                top_mask = valid_mask & (vals >= threshold_val)
                n_top = int(top_mask.sum())

                if n_top < min_top_n:
                    continue

                # By race
                for rg in race_groups:
                    race_mask = df[RACE_COL] == rg
                    n_group_valid = int((valid_mask & race_mask).sum())
                    n_in_top = int((top_mask & race_mask).sum())
                    pct_of_top = float(n_in_top / n_top) if n_top > 0 else 0.0
                    pct_of_sample = float(n_group_valid / n_valid) if n_valid > 0 else 0.0
                    rep_ratio = float(pct_of_top / pct_of_sample) if pct_of_sample > 0 else float("nan")

                    rows.append({
                        "cohort": cohort,
                        "outcome": outcome_name,
                        "percentile_threshold": pct,
                        "group_kind": "race",
                        "group_value": str(rg),
                        "status": "computed",
                        "reason": pd.NA,
                        "n_outcome_valid": n_valid,
                        "n_in_top": n_in_top,
                        "n_group_total": n_group_valid,
                        "pct_of_top": round(pct_of_top, 6),
                        "pct_of_sample": round(pct_of_sample, 6),
                        "representation_ratio": round(rep_ratio, 4),
                        "threshold_value": threshold_val,
                        "source_data": source_data,
                    })

                # By sex
                for sx in ["male", "female"]:
                    sex_mask = df["__sex"] == sx
                    n_group_valid = int((valid_mask & sex_mask).sum())
                    n_in_top = int((top_mask & sex_mask).sum())
                    pct_of_top = float(n_in_top / n_top) if n_top > 0 else 0.0
                    pct_of_sample = float(n_group_valid / n_valid) if n_valid > 0 else 0.0
                    rep_ratio = float(pct_of_top / pct_of_sample) if pct_of_sample > 0 else float("nan")

                    rows.append({
                        "cohort": cohort,
                        "outcome": outcome_name,
                        "percentile_threshold": pct,
                        "group_kind": "sex",
                        "group_value": sx,
                        "status": "computed",
                        "reason": pd.NA,
                        "n_outcome_valid": n_valid,
                        "n_in_top": n_in_top,
                        "n_group_total": n_group_valid,
                        "pct_of_top": round(pct_of_top, 6),
                        "pct_of_sample": round(pct_of_sample, 6),
                        "representation_ratio": round(rep_ratio, 4),
                        "threshold_value": threshold_val,
                        "source_data": source_data,
                    })

                # By race × sex
                for rg in race_groups:
                    for sx in ["male", "female"]:
                        combo_mask = (df[RACE_COL] == rg) & (df["__sex"] == sx)
                        n_group_valid = int((valid_mask & combo_mask).sum())
                        n_in_top = int((top_mask & combo_mask).sum())
                        pct_of_top = float(n_in_top / n_top) if n_top > 0 else 0.0
                        pct_of_sample = float(n_group_valid / n_valid) if n_valid > 0 else 0.0
                        rep_ratio = float(pct_of_top / pct_of_sample) if pct_of_sample > 0 else float("nan")

                        rows.append({
                            "cohort": cohort,
                            "outcome": outcome_name,
                            "percentile_threshold": pct,
                            "group_kind": "race_sex",
                            "group_value": f"{rg}|{sx}",
                            "status": "computed",
                            "reason": pd.NA,
                            "n_outcome_valid": n_valid,
                            "n_in_top": n_in_top,
                            "n_group_total": n_group_valid,
                            "pct_of_top": round(pct_of_top, 6),
                            "pct_of_sample": round(pct_of_sample, 6),
                            "representation_ratio": round(rep_ratio, 4),
                            "threshold_value": threshold_val,
                            "source_data": source_data,
                        })

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
        description="Compute top-percentile representation by race, sex, and race×sex."
    )
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--min-top-n", type=int, default=10)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables/top_percentile_representation.csv"),
    )
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    cohorts = list(COHORT_CONFIGS) if (args.all or not args.cohort) else args.cohort
    out = run(
        root=root,
        cohorts=cohorts,
        output_path=args.output_path,
        min_top_n=args.min_top_n,
    )
    computed = int((out["status"] == "computed").sum()) if "status" in out.columns else 0
    print(f"[ok] {computed} rows computed")
    print(f"[ok] wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
