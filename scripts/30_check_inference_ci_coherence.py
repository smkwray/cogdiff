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

from nls_pipeline.io import project_root

OUTPUT_COLUMNS: tuple[str, ...] = (
    "cohort",
    "inference_method",
    "estimate_type",
    "status",
    "estimate",
    "ci_low",
    "ci_high",
    "ci_contains_estimate",
    "issue",
)
DEFAULT_INPUT_PATH = Path("outputs/tables/robustness_inference.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/tables/inference_ci_coherence.csv")


def _safe_float(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    number = float(parsed)
    if not math.isfinite(number):
        return None
    return number


def _resolve_path(root: Path, path: Path | str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


def _read_inference_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _evaluate_row(row: pd.Series) -> tuple[bool | Any, str]:
    status = str(row.get("status", "")).strip().lower()
    if status != "computed":
        return pd.NA, f"non_computed_status_{status or 'missing'}"

    estimate = _safe_float(row.get("estimate"))
    ci_low = _safe_float(row.get("ci_low"))
    ci_high = _safe_float(row.get("ci_high"))
    if estimate is None or ci_low is None or ci_high is None:
        return pd.NA, "computed_missing_ci_or_estimate"
    if ci_low > ci_high:
        return pd.NA, "computed_invalid_ci_order"

    contains = ci_low <= estimate <= ci_high
    if contains:
        return True, "contains_estimate"
    return False, "computed_outside_ci"


def run_inference_ci_coherence(
    *,
    project_root_path: Path,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    root = Path(project_root_path).resolve()
    source_path = _resolve_path(root, input_path)
    diagnostics_path = _resolve_path(root, output_path)

    frame = _read_inference_table(source_path)
    diagnostics_rows: list[dict[str, Any]] = []

    for _, source_row in frame.iterrows():
        contains_estimate, issue = _evaluate_row(source_row)
        diagnostics_rows.append(
            {
                "cohort": source_row.get("cohort", pd.NA),
                "inference_method": source_row.get("inference_method", pd.NA),
                "estimate_type": source_row.get("estimate_type", pd.NA),
                "status": source_row.get("status", pd.NA),
                "estimate": source_row.get("estimate", pd.NA),
                "ci_low": source_row.get("ci_low", pd.NA),
                "ci_high": source_row.get("ci_high", pd.NA),
                "ci_contains_estimate": contains_estimate,
                "issue": issue,
            }
        )

    diagnostics = pd.DataFrame(diagnostics_rows)
    if diagnostics.empty:
        diagnostics = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        diagnostics["cohort"] = diagnostics["cohort"].astype(str).str.strip().replace({"nan": pd.NA})
        diagnostics["inference_method"] = diagnostics["inference_method"].astype(str).str.strip().replace({"nan": pd.NA})
        diagnostics["estimate_type"] = diagnostics["estimate_type"].astype(str).str.strip().replace({"nan": pd.NA})
        diagnostics = diagnostics.sort_values(
            by=["cohort", "inference_method", "estimate_type"],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
        diagnostics = diagnostics.reindex(columns=OUTPUT_COLUMNS)

    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(diagnostics_path, index=False)
    return diagnostics


def _violation_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    status_mask = frame["status"].astype(str).str.strip().str.lower() == "computed"
    return int((status_mask & (frame["issue"] != "contains_estimate")).sum())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check CI coherence for robustness_inference rows.",
    )
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input robustness inference CSV (relative to project root if not absolute).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output diagnostics CSV (relative to project root if not absolute).",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Return non-zero if any computed row violates CI containment.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    diagnostics = run_inference_ci_coherence(
        project_root_path=root,
        input_path=args.input_path,
        output_path=args.output_path,
    )
    violations = _violation_count(diagnostics)
    print(f"[ok] wrote {args.output_path}")
    print(f"[info] computed rows with CI violations: {violations}")
    if args.fail_on_violation and violations > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
