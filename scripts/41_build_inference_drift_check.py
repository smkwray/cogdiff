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

MEAN_BASELINE_CANDIDATES = (
    Path("outputs/tables/g_mean_diff.csv"),
    Path("outputs/tables/g_mean_diff_full_cohort.csv"),
)
VR_BASELINE_PATH = Path("outputs/tables/g_variance_ratio.csv")
MEAN_BOOTSTRAP_PATH = Path("outputs/tables/g_mean_diff_family_bootstrap.csv")
VR_BOOTSTRAP_PATH = Path("outputs/tables/g_variance_ratio_family_bootstrap.csv")

OUTPUT_COLUMNS = [
    "cohort",
    "estimand",
    "status",
    "reason",
    "baseline_estimate",
    "bootstrap_estimate",
    "delta_abs",
    "threshold",
    "pass",
    "source_baseline",
    "source_bootstrap",
]


def _resolve_first_existing(root: Path, candidates: tuple[Path, ...]) -> Path | None:
    for rel_path in candidates:
        path = root / rel_path
        if path.exists():
            return path
    return None


def _empty_row(cohort: str, estimand: str, reason: str, source_baseline: str, source_bootstrap: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "estimand": estimand,
        "status": "not_feasible",
        "reason": reason,
        "source_baseline": source_baseline,
        "source_bootstrap": source_bootstrap,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _read_table(path: Path, required: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def run_inference_drift_check(
    *,
    root: Path,
    max_abs_delta_d: float = 0.2,
    max_abs_delta_log_vr: float = 0.05,
    output_path: Path = Path("outputs/tables/inference_drift_check.csv"),
) -> pd.DataFrame:
    if max_abs_delta_d <= 0.0:
        raise ValueError("max_abs_delta_d must be > 0")
    if max_abs_delta_log_vr <= 0.0:
        raise ValueError("max_abs_delta_log_vr must be > 0")

    mean_baseline_path = _resolve_first_existing(root, MEAN_BASELINE_CANDIDATES)
    mean_bootstrap_path = root / MEAN_BOOTSTRAP_PATH
    vr_baseline_path = root / VR_BASELINE_PATH
    vr_bootstrap_path = root / VR_BOOTSTRAP_PATH

    rows: list[dict[str, Any]] = []

    # d_g drift
    if mean_baseline_path is None:
        rows.append(_empty_row("", "d_g", "missing_mean_baseline_table", "", str(MEAN_BOOTSTRAP_PATH)))
    elif not mean_bootstrap_path.exists():
        rows.append(
            _empty_row(
                "",
                "d_g",
                "missing_mean_bootstrap_table",
                str(mean_baseline_path.relative_to(root)),
                str(MEAN_BOOTSTRAP_PATH),
            )
        )
    else:
        base = _read_table(mean_baseline_path, {"cohort", "d_g"}).copy()
        boot = _read_table(mean_bootstrap_path, {"cohort", "status", "d_g"}).copy()
        base["cohort"] = base["cohort"].astype(str)
        boot["cohort"] = boot["cohort"].astype(str)
        merged = base.merge(
            boot[["cohort", "status", "reason", "d_g"]].rename(columns={"d_g": "d_boot"}),
            how="outer",
            on="cohort",
            suffixes=("_base", "_boot"),
        )
        for _, r in merged.iterrows():
            cohort = str(r.get("cohort", ""))
            base_d = pd.to_numeric(pd.Series([r.get("d_g")]), errors="coerce").iloc[0]
            boot_status = str(r.get("status")) if pd.notna(r.get("status")) else ""
            boot_reason = str(r.get("reason")) if pd.notna(r.get("reason")) else ""
            boot_d = pd.to_numeric(pd.Series([r.get("d_boot")]), errors="coerce").iloc[0]
            src_base = str(mean_baseline_path.relative_to(root))
            src_boot = str(mean_bootstrap_path.relative_to(root))

            if pd.isna(base_d):
                rows.append(_empty_row(cohort, "d_g", "missing_baseline_d", src_base, src_boot))
                continue
            if boot_status and boot_status != "computed":
                reason = boot_reason if boot_reason else f"bootstrap_status:{boot_status}"
                rows.append(_empty_row(cohort, "d_g", reason, src_base, src_boot) | {"baseline_estimate": float(base_d)})
                continue
            if pd.isna(boot_d):
                rows.append(_empty_row(cohort, "d_g", "missing_bootstrap_d", src_base, src_boot) | {"baseline_estimate": float(base_d)})
                continue

            delta = abs(float(base_d) - float(boot_d))
            out = {
                "cohort": cohort,
                "estimand": "d_g",
                "status": "computed",
                "reason": pd.NA,
                "baseline_estimate": float(base_d),
                "bootstrap_estimate": float(boot_d),
                "delta_abs": float(delta),
                "threshold": float(max_abs_delta_d),
                "pass": bool(delta <= max_abs_delta_d),
                "source_baseline": src_base,
                "source_bootstrap": src_boot,
            }
            for col in OUTPUT_COLUMNS:
                out.setdefault(col, pd.NA)
            rows.append(out)

    # log(VR_g) drift
    if not vr_baseline_path.exists():
        rows.append(_empty_row("", "log_vr_g", "missing_vr_baseline_table", str(VR_BASELINE_PATH), str(VR_BOOTSTRAP_PATH)))
    elif not vr_bootstrap_path.exists():
        rows.append(_empty_row("", "log_vr_g", "missing_vr_bootstrap_table", str(VR_BASELINE_PATH), str(VR_BOOTSTRAP_PATH)))
    else:
        base = _read_table(vr_baseline_path, {"cohort", "VR_g"}).copy()
        boot = _read_table(vr_bootstrap_path, {"cohort", "status", "VR_g"}).copy()
        base["cohort"] = base["cohort"].astype(str)
        boot["cohort"] = boot["cohort"].astype(str)
        merged = base.merge(
            boot[["cohort", "status", "reason", "VR_g"]].rename(columns={"VR_g": "vr_boot"}),
            how="outer",
            on="cohort",
        )
        for _, r in merged.iterrows():
            cohort = str(r.get("cohort", ""))
            base_vr = pd.to_numeric(pd.Series([r.get("VR_g")]), errors="coerce").iloc[0]
            boot_status = str(r.get("status")) if pd.notna(r.get("status")) else ""
            boot_reason = str(r.get("reason")) if pd.notna(r.get("reason")) else ""
            boot_vr = pd.to_numeric(pd.Series([r.get("vr_boot")]), errors="coerce").iloc[0]
            src_base = str(vr_baseline_path.relative_to(root))
            src_boot = str(vr_bootstrap_path.relative_to(root))

            if pd.isna(base_vr) or float(base_vr) <= 0.0:
                rows.append(_empty_row(cohort, "log_vr_g", "invalid_baseline_vr", src_base, src_boot))
                continue
            if boot_status and boot_status != "computed":
                reason = boot_reason if boot_reason else f"bootstrap_status:{boot_status}"
                rows.append(
                    _empty_row(cohort, "log_vr_g", reason, src_base, src_boot)
                    | {"baseline_estimate": float(math.log(float(base_vr)))}
                )
                continue
            if pd.isna(boot_vr) or float(boot_vr) <= 0.0:
                rows.append(
                    _empty_row(cohort, "log_vr_g", "invalid_bootstrap_vr", src_base, src_boot)
                    | {"baseline_estimate": float(math.log(float(base_vr)))}
                )
                continue

            base_log = float(math.log(float(base_vr)))
            boot_log = float(math.log(float(boot_vr)))
            delta = abs(base_log - boot_log)
            out = {
                "cohort": cohort,
                "estimand": "log_vr_g",
                "status": "computed",
                "reason": pd.NA,
                "baseline_estimate": base_log,
                "bootstrap_estimate": boot_log,
                "delta_abs": float(delta),
                "threshold": float(max_abs_delta_log_vr),
                "pass": bool(delta <= max_abs_delta_log_vr),
                "source_baseline": src_base,
                "source_bootstrap": src_boot,
            }
            for col in OUTPUT_COLUMNS:
                out.setdefault(col, pd.NA)
            rows.append(out)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        out_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    for col in OUTPUT_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = pd.NA
    out_df = out_df[OUTPUT_COLUMNS].copy()

    output_file = output_path if output_path.is_absolute() else root / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_file, index=False)
    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Build baseline-vs-bootstrap drift checks for inference outputs.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--max-abs-delta-d",
        type=float,
        default=0.2,
        help="Maximum allowed absolute drift for d_g.",
    )
    parser.add_argument(
        "--max-abs-delta-log-vr",
        type=float,
        default=0.05,
        help="Maximum allowed absolute drift for log(VR_g).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables/inference_drift_check.csv"),
        help="Output CSV path (relative to project-root if not absolute).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        out = run_inference_drift_check(
            root=root,
            max_abs_delta_d=float(args.max_abs_delta_d),
            max_abs_delta_log_vr=float(args.max_abs_delta_log_vr),
            output_path=args.output_path,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    output_file = args.output_path if args.output_path.is_absolute() else root / args.output_path
    computed = int((out["status"] == "computed").sum()) if "status" in out.columns else 0
    passed = int(((out["status"] == "computed") & (out["pass"] == True)).sum()) if "pass" in out.columns else 0
    print(f"[ok] wrote {output_file}")
    print(f"[ok] computed rows: {computed}")
    print(f"[ok] pass rows: {passed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
