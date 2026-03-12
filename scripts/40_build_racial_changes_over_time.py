#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import project_root

COHORT_ORDER = {
    "nlsy79": 0,
    "nlsy97": 1,
    "cnlsy": 2,
}

DEFAULT_SOURCE_DETAIL = Path("outputs/tables/race_sex_group_estimates.csv")
OUTPUT_COLUMNS = [
    "race_group",
    "status",
    "reason",
    "n_cohorts",
    "cohorts",
    "fit_weighting",
    "slope_per_cohort_step",
    "SE_slope",
    "z_slope",
    "p_value_slope",
    "ci_low_slope",
    "ci_high_slope",
    "delta_first_to_last",
    "ci_low_delta",
    "ci_high_delta",
    "source_detail",
]


def _empty_row(race_group: str, reason: str, source_detail: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "race_group": race_group,
        "status": "not_feasible",
        "reason": reason,
        "n_cohorts": 0,
        "cohorts": "",
        "fit_weighting": pd.NA,
        "source_detail": source_detail,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _fit_one_group(race_group: str, rows: pd.DataFrame, source_detail: str) -> dict[str, Any]:
    if rows.shape[0] < 2:
        return _empty_row(race_group, "insufficient_cohorts", source_detail) | {
            "n_cohorts": int(rows.shape[0]),
            "cohorts": ",".join(rows["cohort"].astype(str).tolist()),
        }

    x = rows["cohort_index"].astype(float).to_numpy()
    y = rows["d_g"].astype(float).to_numpy()
    se = rows["SE_d_g"].astype(float).to_numpy()

    if bool(((~pd.Series(se).apply(math.isfinite)) | (pd.Series(se) <= 0.0)).any()):
        w = pd.Series([1.0] * len(rows), dtype="float64").to_numpy()
        fit_weighting = "unweighted_equal"
    else:
        w = (1.0 / (pd.Series(se, dtype="float64") ** 2)).to_numpy()
        fit_weighting = "inverse_variance"

    w_sum = float(w.sum())
    if not math.isfinite(w_sum) or w_sum <= 0.0:
        return _empty_row(race_group, "invalid_weights", source_detail) | {
            "n_cohorts": int(rows.shape[0]),
            "cohorts": ",".join(rows["cohort"].astype(str).tolist()),
            "fit_weighting": fit_weighting,
        }

    x_bar = float((w * x).sum() / w_sum)
    y_bar = float((w * y).sum() / w_sum)
    sxx = float((w * (x - x_bar) ** 2).sum())
    if not math.isfinite(sxx) or sxx <= 0.0:
        return _empty_row(race_group, "degenerate_cohort_index", source_detail) | {
            "n_cohorts": int(rows.shape[0]),
            "cohorts": ",".join(rows["cohort"].astype(str).tolist()),
            "fit_weighting": fit_weighting,
        }

    slope = float((w * (x - x_bar) * (y - y_bar)).sum() / sxx)
    se_slope = float(math.sqrt(1.0 / sxx))
    if not math.isfinite(se_slope) or se_slope <= 0.0:
        return _empty_row(race_group, "nonpositive_slope_se", source_detail) | {
            "n_cohorts": int(rows.shape[0]),
            "cohorts": ",".join(rows["cohort"].astype(str).tolist()),
            "fit_weighting": fit_weighting,
        }

    z_slope = slope / se_slope
    p_value = float(2.0 * norm.sf(abs(z_slope)))
    ci_low = float(slope - 1.96 * se_slope)
    ci_high = float(slope + 1.96 * se_slope)
    x_delta = float(x.max() - x.min())
    delta = float(slope * x_delta)
    se_delta = float(abs(x_delta) * se_slope)
    ci_low_delta = float(delta - 1.96 * se_delta)
    ci_high_delta = float(delta + 1.96 * se_delta)

    row = {
        "race_group": race_group,
        "status": "computed",
        "reason": pd.NA,
        "n_cohorts": int(rows.shape[0]),
        "cohorts": ",".join(rows["cohort"].astype(str).tolist()),
        "fit_weighting": fit_weighting,
        "slope_per_cohort_step": slope,
        "SE_slope": se_slope,
        "z_slope": z_slope,
        "p_value_slope": p_value,
        "ci_low_slope": ci_low,
        "ci_high_slope": ci_high,
        "delta_first_to_last": delta,
        "ci_low_delta": ci_low_delta,
        "ci_high_delta": ci_high_delta,
        "source_detail": source_detail,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_racial_changes_over_time(
    *,
    root: Path,
    source_detail_path: Path = DEFAULT_SOURCE_DETAIL,
    output_path: Path = Path("outputs/tables/racial_changes_over_time.csv"),
) -> pd.DataFrame:
    detail_path = source_detail_path if source_detail_path.is_absolute() else root / source_detail_path
    source_detail = str(detail_path.relative_to(root)) if detail_path.exists() else str(source_detail_path)

    if not detail_path.exists():
        out = pd.DataFrame([_empty_row("", "missing_source_detail", source_detail)])
        out = out[OUTPUT_COLUMNS].copy()
        output_file = output_path if output_path.is_absolute() else root / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_file, index=False)
        return out

    detail = pd.read_csv(detail_path, low_memory=False)
    needed = {"cohort", "race_group", "status", "d_g", "SE_d_g"}
    if not needed.issubset(set(detail.columns)):
        out = pd.DataFrame([_empty_row("", "invalid_source_detail_columns", source_detail)])
        out = out[OUTPUT_COLUMNS].copy()
        output_file = output_path if output_path.is_absolute() else root / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_file, index=False)
        return out

    detail = detail[detail["status"].astype(str) == "computed"].copy()
    detail = detail[detail["cohort"].isin(COHORT_ORDER)].copy()
    detail["cohort_index"] = detail["cohort"].map(COHORT_ORDER).astype(float)
    detail["d_g"] = pd.to_numeric(detail["d_g"], errors="coerce")
    detail["SE_d_g"] = pd.to_numeric(detail["SE_d_g"], errors="coerce")
    detail = detail.dropna(subset=["race_group", "d_g", "SE_d_g", "cohort_index"]).copy()

    rows: list[dict[str, Any]] = []
    if detail.empty:
        rows.append(_empty_row("", "no_computed_rows_in_source_detail", source_detail))
    else:
        for race_group, grp in detail.groupby("race_group", sort=True):
            g = grp.sort_values(["cohort_index", "cohort"]).reset_index(drop=True)
            rows.append(_fit_one_group(str(race_group), g, source_detail))

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
        description="Build racial changes-over-time trend diagnostics from race-group cohort estimates."
    )
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--source-detail-path",
        type=Path,
        default=DEFAULT_SOURCE_DETAIL,
        help="Source detail CSV path from script 38 (relative to project-root if not absolute).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables/racial_changes_over_time.csv"),
        help="Output CSV path (relative to project-root if not absolute).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        out = run_racial_changes_over_time(
            root=root,
            source_detail_path=args.source_detail_path,
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
