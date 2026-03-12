#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.cnlsy import build_cnlsy_agebin_summary
from nls_pipeline.io import load_yaml, project_root

COHORT_CONFIG = "config/cnlsy.yml"

OUTPUT_COLUMNS = [
    "metric",
    "status",
    "reason",
    "n_bins",
    "n_obs_total",
    "linear_r2",
    "quadratic_r2",
    "quadratic_delta_r2",
    "linear_age_coef",
    "quadratic_age_coef",
    "quadratic_age2_coef",
    "turning_point_age",
    "source_data",
]


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _as_label(value: Any) -> str:
    if isinstance(value, bool):
        return "NO" if value is False else "YES"
    return str(value)


def _score_series(df: pd.DataFrame, sample_cfg: dict[str, Any], models_cfg: dict[str, Any]) -> pd.Series:
    subtests = [_as_label(x) for x in sample_cfg.get("subtests", models_cfg.get("cnlsy_single_factor", []))]
    available = [col for col in subtests if col in df.columns]
    if not available:
        raise ValueError("No CNLSY subtests available to build score.")
    return pd.DataFrame({col: pd.to_numeric(df[col], errors="coerce") for col in available}, index=df.index).mean(axis=1, skipna=True)


def _fit_poly(metric: str, data: pd.DataFrame, source_data: str) -> dict[str, object]:
    if data.shape[0] < 3:
        row = {"metric": metric, "status": "not_feasible", "reason": "insufficient_age_bins", "source_data": source_data}
        for col in OUTPUT_COLUMNS:
            row.setdefault(col, pd.NA)
        row["n_bins"] = int(data.shape[0])
        row["n_obs_total"] = int(pd.to_numeric(data.get("n_obs"), errors="coerce").fillna(0).sum())
        return row

    x = pd.to_numeric(data["age_mid"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(data["value"], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(data["n_obs"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    x = x[mask]
    y = y[mask]
    w = w[mask]
    if x.size < 3:
        row = {"metric": metric, "status": "not_feasible", "reason": "insufficient_valid_rows", "source_data": source_data}
        for col in OUTPUT_COLUMNS:
            row.setdefault(col, pd.NA)
        row["n_bins"] = int(x.size)
        row["n_obs_total"] = int(w.sum())
        return row

    x_center = x - float(x.mean())
    sst = float(np.sum(w * ((y - float(np.average(y, weights=w))) ** 2)))
    if sst <= 0.0:
        row = {"metric": metric, "status": "not_feasible", "reason": "nonpositive_total_variance", "source_data": source_data}
        for col in OUTPUT_COLUMNS:
            row.setdefault(col, pd.NA)
        row["n_bins"] = int(x.size)
        row["n_obs_total"] = int(w.sum())
        return row

    x_linear = np.column_stack([np.ones_like(x_center), x_center])
    beta_lin, _, _, _ = np.linalg.lstsq(x_linear * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)
    resid_lin = y - (x_linear @ beta_lin)
    linear_r2 = float(1.0 - (np.sum(w * resid_lin**2) / sst))

    x_quad = np.column_stack([np.ones_like(x_center), x_center, x_center**2])
    beta_quad, _, _, _ = np.linalg.lstsq(x_quad * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)
    resid_quad = y - (x_quad @ beta_quad)
    quadratic_r2 = float(1.0 - (np.sum(w * resid_quad**2) / sst))

    turning_point = pd.NA
    if abs(float(beta_quad[2])) > 1e-12:
        turning_point = float((-beta_quad[1] / (2.0 * beta_quad[2])) + x.mean())

    row = {
        "metric": metric,
        "status": "computed",
        "reason": pd.NA,
        "n_bins": int(x.size),
        "n_obs_total": int(w.sum()),
        "linear_r2": linear_r2,
        "quadratic_r2": quadratic_r2,
        "quadratic_delta_r2": float(quadratic_r2 - linear_r2),
        "linear_age_coef": float(beta_lin[1]),
        "quadratic_age_coef": float(beta_quad[1]),
        "quadratic_age2_coef": float(beta_quad[2]),
        "turning_point_age": turning_point,
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_cnlsy_nonlinear_age_patterns(
    *,
    root: Path,
    output_path: Path = Path("outputs/tables/cnlsy_nonlinear_age_patterns.csv"),
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    cohort_cfg = load_yaml(root / COHORT_CONFIG)
    sample_cfg = cohort_cfg.get("sample_construct", {}) if isinstance(cohort_cfg.get("sample_construct", {}), dict) else {}

    processed_dir = _resolve_path(paths_cfg.get("processed_dir", "data/processed"), root)
    source_path = processed_dir / "cnlsy_long.csv"
    if not source_path.exists():
        source_path = processed_dir / "cnlsy_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / "cnlsy_cfa.csv"
    if not source_path.exists():
        out = pd.DataFrame([_fit_poly("mean_diff", pd.DataFrame(), "cnlsy_cfa_resid_or_cfa.csv")])[OUTPUT_COLUMNS]
        target = output_path if output_path.is_absolute() else root / output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)
        return out

    df = pd.read_csv(source_path, low_memory=False)
    df = df.copy()
    df["g"] = _score_series(df, sample_cfg, models_cfg)
    age_col = str(sample_cfg.get("age_col", "csage"))
    sex_col = str(sample_cfg.get("sex_col", "sex"))
    id_col = str(sample_cfg.get("id_col", "person_id"))
    expected_age_range = cohort_cfg.get("expected_age_range", {})
    age_summary = build_cnlsy_agebin_summary(
        df,
        id_col=id_col,
        age_col=age_col,
        sex_col=sex_col,
        score_col="g",
        min_age=int(expected_age_range.get("min", 5)),
        max_age=int(expected_age_range.get("max", 18)),
        bin_width=2,
    )
    age_summary = age_summary.copy()
    age_summary["age_mid"] = (pd.to_numeric(age_summary["age_min"], errors="coerce") + pd.to_numeric(age_summary["age_max"], errors="coerce")) / 2.0
    age_summary["log_variance_ratio"] = pd.to_numeric(age_summary["variance_ratio"], errors="coerce").apply(
        lambda v: math.log(float(v)) if pd.notna(v) and float(v) > 0.0 else pd.NA
    )

    metric_frames = {
        "male_mean": pd.DataFrame(
            {"age_mid": age_summary["age_mid"], "value": age_summary["male_mean"], "n_obs": age_summary["n_male"]}
        ),
        "female_mean": pd.DataFrame(
            {"age_mid": age_summary["age_mid"], "value": age_summary["female_mean"], "n_obs": age_summary["n_female"]}
        ),
        "mean_diff": pd.DataFrame(
            {"age_mid": age_summary["age_mid"], "value": age_summary["mean_diff"], "n_obs": age_summary["n_obs"]}
        ),
        "log_variance_ratio": pd.DataFrame(
            {"age_mid": age_summary["age_mid"], "value": age_summary["log_variance_ratio"], "n_obs": age_summary["n_obs"]}
        ),
    }
    rows = [
        _fit_poly(metric, frame.dropna(subset=["age_mid", "value", "n_obs"]), str(source_path.relative_to(root)))
        for metric, frame in metric_frames.items()
    ]
    out = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS].copy()
    target = output_path if output_path.is_absolute() else root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize nonlinear age patterns in CNLSY age-bin metrics.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/cnlsy_nonlinear_age_patterns.csv"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        out = run_cnlsy_nonlinear_age_patterns(root=root, output_path=args.output_path)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(f"[ok] wrote {args.output_path if args.output_path.is_absolute() else root / args.output_path}")
    print(f"[ok] computed rows: {int((out['status'] == 'computed').sum()) if 'status' in out.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
