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

from nls_pipeline.io import load_yaml, project_root

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

OCCUPATION_SPECS: dict[str, list[str]] = {
    "nlsy79": ["occupation_code_2000"],
    "nlsy97": [
        "occupation_code_2011",
        "occupation_code_2013",
        "occupation_code_2015",
        "occupation_code_2017",
        "occupation_code_2019",
        "occupation_code_2021",
    ],
}

OUTPUT_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "occupation_col",
    "n_total",
    "n_nonmissing",
    "pct_nonmissing",
    "n_unique_codes",
    "min_code",
    "max_code",
    "top_codes",
    "source_data",
]


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _empty_row(cohort: str, reason: str, source_data: str, *, occupation_col: str = "", n_total: int = 0) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "status": "not_feasible",
        "reason": reason,
        "occupation_col": occupation_col,
        "n_total": int(n_total),
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_occupation_code_availability(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path = Path("outputs/tables/occupation_code_availability.csv"),
    top_n: int = 5,
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{cohort}_cfa_resid_or_cfa.csv"
        occupation_cols = OCCUPATION_SPECS.get(cohort, [])
        if not source_path.exists():
            if occupation_cols:
                for occupation_col in occupation_cols:
                    rows.append(_empty_row(cohort, "missing_source_data", source_data, occupation_col=occupation_col))
            else:
                rows.append(_empty_row(cohort, "missing_source_data", source_data))
            continue
        if not occupation_cols:
            rows.append(_empty_row(cohort, "occupation_spec_not_configured", source_data, n_total=0))
            continue

        df = pd.read_csv(source_path, low_memory=False)
        for occupation_col in occupation_cols:
            if occupation_col not in df.columns:
                rows.append(
                    _empty_row(
                        cohort,
                        "missing_occupation_column",
                        source_data,
                        occupation_col=occupation_col,
                        n_total=len(df),
                    )
                )
                continue

            codes = pd.to_numeric(df[occupation_col], errors="coerce")
            nonmissing = codes.dropna()
            if nonmissing.empty:
                rows.append(
                    _empty_row(
                        cohort,
                        "no_nonmissing_codes",
                        source_data,
                        occupation_col=occupation_col,
                        n_total=len(df),
                    )
                )
                continue
            counts = nonmissing.astype(int).value_counts().head(top_n)
            top_codes = ";".join(f"{int(code)}:{int(count)}" for code, count in counts.items())
            row = {
                "cohort": cohort,
                "status": "computed",
                "reason": pd.NA,
                "occupation_col": occupation_col,
                "n_total": int(len(df)),
                "n_nonmissing": int(nonmissing.shape[0]),
                "pct_nonmissing": float(nonmissing.shape[0] / len(df)) if len(df) > 0 else pd.NA,
                "n_unique_codes": int(nonmissing.astype(int).nunique()),
                "min_code": int(nonmissing.min()) if math.isfinite(float(nonmissing.min())) else pd.NA,
                "max_code": int(nonmissing.max()) if math.isfinite(float(nonmissing.max())) else pd.NA,
                "top_codes": top_codes,
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
    output_full = root / output_path
    output_full.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_full, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize occupation-code availability in processed cohort files.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/occupation_code_availability.csv"))
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args(argv)

    root = Path(args.project_root)
    out = run_occupation_code_availability(root=root, cohorts=_cohorts_from_args(args), output_path=args.output_path, top_n=args.top_n)
    computed = int((out["status"] == "computed").sum()) if "status" in out.columns else 0
    print(f"[ok] occupation availability rows computed: {computed}")
    print(f"[ok] wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
