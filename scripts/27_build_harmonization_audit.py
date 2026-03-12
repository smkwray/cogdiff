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

DEFAULT_BASELINE_TOKEN = "signed_merge"
DEFAULT_ALT_TOKEN = "zscore_by_branch"
DEFAULT_D_G_ABS_THRESHOLD = 0.10
DEFAULT_LOG_VR_ABS_THRESHOLD = 0.10
DEFAULT_COHORT = "nlsy97"


def _safe_float(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def _pick_first(series: pd.Series, candidates: tuple[str, ...]) -> float | None:
    for name in candidates:
        if name in series.index:
            return _safe_float(series[name])
    return None


def _cohort_row(path: Path, cohort: str) -> tuple[pd.Series | None, bool]:
    if not path.exists():
        return None, False

    frame = pd.read_csv(path)
    if frame.empty:
        return None, False

    if "cohort" in frame.columns:
        filtered = frame[frame["cohort"].astype(str).str.lower() == str(cohort).lower()]
        if filtered.empty:
            return None, False
        return filtered.iloc[0], True

    return frame.iloc[0], True


def _load_estimates(
    *,
    tables_dir: Path,
    cohort: str,
    token: str,
) -> tuple[float | None, float | None]:
    mean_path = tables_dir / f"g_mean_diff_{token}.csv"
    vr_path = tables_dir / f"g_variance_ratio_{token}.csv"
    mean_row, _ = _cohort_row(mean_path, cohort)
    vr_row, _ = _cohort_row(vr_path, cohort)
    d_g = _pick_first(mean_row, ("d_g",)) if mean_row is not None else None
    vr_g = _pick_first(vr_row, ("VR_g", "vr_g")) if vr_row is not None else None
    return d_g if d_g is not None else None, vr_g if vr_g is not None else None


def _delta_log_ratio(vr_signed: float | None, vr_alt: float | None) -> tuple[float | None, str | None]:
    if vr_signed is None or vr_alt is None:
        return None, "missing"
    if vr_signed <= 0 or vr_alt <= 0:
        return None, "nonpositive_vr"
    return abs(math.log(vr_signed) - math.log(vr_alt)), None


def run_harmonization_audit(
    *,
    project_root_path: Path,
    cohort: str = DEFAULT_COHORT,
    baseline_token: str = DEFAULT_BASELINE_TOKEN,
    alternate_token: str = DEFAULT_ALT_TOKEN,
    d_g_abs_threshold: float = DEFAULT_D_G_ABS_THRESHOLD,
    log_vr_abs_threshold: float = DEFAULT_LOG_VR_ABS_THRESHOLD,
    output_path: Path = Path("outputs/tables/harmonization_audit_summary.csv"),
) -> pd.DataFrame:
    root = Path(project_root_path)
    tables_dir = root / "outputs" / "tables"
    d_g_signed, vr_signed = _load_estimates(tables_dir=tables_dir, cohort=cohort, token=baseline_token)
    d_g_alt, vr_alt = _load_estimates(tables_dir=tables_dir, cohort=cohort, token=alternate_token)

    delta_d_g_abs = None
    if d_g_signed is not None and d_g_alt is not None:
        delta_d_g_abs = abs(d_g_signed - d_g_alt)

    delta_log_vr_abs, log_vr_reason = _delta_log_ratio(vr_signed, vr_alt)

    d_g_abs_pass = False
    if delta_d_g_abs is not None:
        d_g_abs_pass = delta_d_g_abs <= d_g_abs_threshold

    log_vr_abs_pass = False
    if delta_log_vr_abs is not None:
        log_vr_abs_pass = delta_log_vr_abs <= log_vr_abs_threshold

    if delta_d_g_abs is None and delta_log_vr_abs is None:
        audit_status = "missing"
    elif delta_d_g_abs is None or delta_log_vr_abs is None:
        audit_status = "partial"
    else:
        audit_status = "complete"

    row = {
        "cohort": cohort,
        "baseline_token": baseline_token,
        "alternate_token": alternate_token,
        "d_g_signed": d_g_signed,
        "d_g_alt": d_g_alt,
        "delta_d_g_abs": delta_d_g_abs,
        "d_g_abs_threshold": d_g_abs_threshold,
        "d_g_abs_pass": d_g_abs_pass,
        "vr_signed": vr_signed,
        "vr_alt": vr_alt,
        "log_vr_reason": log_vr_reason,
        "delta_log_vr_abs": delta_log_vr_abs,
        "log_vr_abs_threshold": log_vr_abs_threshold,
        "log_vr_abs_pass": log_vr_abs_pass,
        "audit_status": audit_status,
    }

    summary = pd.DataFrame([row])
    output_file = Path(output_path)
    if not output_file.is_absolute():
        output_file = root / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_file, index=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build harmonization audit summary for NLSY97 variants.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", default=DEFAULT_COHORT, help="Cohort to audit.")
    parser.add_argument("--baseline-token", default=DEFAULT_BASELINE_TOKEN, help="Baseline harmonization token.")
    parser.add_argument("--alternate-token", default=DEFAULT_ALT_TOKEN, help="Alternate harmonization token.")
    parser.add_argument(
        "--d-g-abs-threshold",
        type=float,
        default=DEFAULT_D_G_ABS_THRESHOLD,
        help="Pass threshold for |Δd_g|.",
    )
    parser.add_argument(
        "--log-vr-abs-threshold",
        type=float,
        default=DEFAULT_LOG_VR_ABS_THRESHOLD,
        help="Pass threshold for |Δlog(VR_g)|.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables/harmonization_audit_summary.csv"),
        help="Output CSV path (relative to project-root if not absolute).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    run_harmonization_audit(
        project_root_path=root,
        cohort=args.cohort,
        baseline_token=args.baseline_token,
        alternate_token=args.alternate_token,
        d_g_abs_threshold=args.d_g_abs_threshold,
        log_vr_abs_threshold=args.log_vr_abs_threshold,
        output_path=args.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
