#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.cnlsy import (
    AGEBIN_SUMMARY_COLUMNS,
    LONGITUDINAL_SUMMARY_COLUMNS,
    build_cnlsy_agebin_summary,
    build_cnlsy_longitudinal_summary,
)
from nls_pipeline.io import load_yaml, project_root

COHORT_CONFIG = "config/cnlsy.yml"


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _as_label(value: Any) -> str:
    if isinstance(value, bool):
        return "NO" if value is False else "YES"
    return str(value)


def _build_score_column(
    df: pd.DataFrame,
    requested_score_col: str,
    sample_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    if requested_score_col in df.columns:
        return df, requested_score_col

    subtests = [
        _as_label(x)
        for x in sample_cfg.get("subtests", models_cfg.get("cnlsy_single_factor", []))
    ]
    available = [col for col in subtests if col in df.columns]
    if not available:
        raise ValueError(
            "CNLSY score column not found and no configured subtests were available "
            "to construct a temporary score."
        )

    out = df.copy()
    out["g"] = out[available].mean(axis=1)
    return out, "g"


def _write_trend_plot(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    output: Path,
    title: str,
    ylabel: str,
) -> None:
    data = frame.dropna(subset=[x_col, y_col]).copy()
    data = data.sort_values(x_col)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    if data.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
    elif len(data) == 1:
        ax.plot(data[x_col], data[y_col], marker="o", linestyle="none")
    else:
        ax.plot(data[x_col], data[y_col], marker="o")

    if not data.empty:
        ax.set_xticks(data[x_col].tolist())
        ax.set_xticklabels([str(v) for v in data[x_col].tolist()])
    else:
        ax.set_xticks([])

    ax.set_title(title)
    ax.set_xlabel("Age bin (start age)")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _load_inputs(root: Path, source_override: Path | None) -> tuple[Path, dict[str, Any], dict[str, Any], Path]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    cohort_cfg = load_yaml(root / COHORT_CONFIG)
    sample_cfg = cohort_cfg.get("sample_construct", {})
    if not isinstance(sample_cfg, dict):
        raise ValueError("CNLSY sample_construct config must be a mapping.")

    processed_dir = _resolve_path(paths_cfg["processed_dir"], root)
    long_source = processed_dir / "cnlsy_long.csv"
    if long_source.exists():
        default_source = long_source
    else:
        default_source = processed_dir / "cnlsy_cfa_resid.csv"
    if not default_source.exists():
        fallback_source = processed_dir / "cnlsy_cfa.csv"
        if fallback_source.exists():
            default_source = fallback_source
    if not default_source.exists():
        raise FileNotFoundError(
            f"Missing CNLSY processed input at {processed_dir / 'cnlsy_long.csv'} "
            f"or {processed_dir / 'cnlsy_cfa_resid.csv'} "
            f"or {processed_dir / 'cnlsy_cfa.csv'}."
        )

    source = source_override.resolve() if source_override is not None else default_source
    if not source.exists():
        raise FileNotFoundError(f"CNLSY input not found: {source}")

    outputs_dir = _resolve_path(paths_cfg["outputs_dir"], root)
    return source, models_cfg, sample_cfg, outputs_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CNLSY age-bin and longitudinal developmental outputs.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--source-path", type=Path, help="Optional explicit CNLSY input CSV path.")
    parser.add_argument(
        "--score-col",
        default="g",
        help="Score column for developmental summaries (default: g). If missing, built from subtests.",
    )
    parser.add_argument("--min-age", type=int, help="Override minimum age bin boundary.")
    parser.add_argument("--max-age", type=int, help="Override maximum age bin boundary.")
    parser.add_argument("--bin-width", type=int, default=2, help="Age bin width.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    source_path, models_cfg, sample_cfg, outputs_dir = _load_inputs(root, args.source_path)

    df = pd.read_csv(source_path, low_memory=False)
    cohort_cfg = load_yaml(root / COHORT_CONFIG)
    expected_age_range = cohort_cfg.get("expected_age_range", {})
    min_age = int(args.min_age if args.min_age is not None else expected_age_range.get("min", 5))
    max_age = int(args.max_age if args.max_age is not None else expected_age_range.get("max", 18))
    bin_width = int(args.bin_width)

    id_col = str(sample_cfg.get("id_col", "person_id"))
    age_col = str(sample_cfg.get("age_col", "age"))
    sex_col = str(sample_cfg.get("sex_col", "sex"))

    df, score_col = _build_score_column(
        df=df,
        requested_score_col=args.score_col,
        sample_cfg=sample_cfg,
        models_cfg=models_cfg,
    )
    if score_col not in df.columns:
        raise ValueError(f"Score column not found: {score_col}")

    age_summary = build_cnlsy_agebin_summary(
        df,
        id_col=id_col,
        age_col=age_col,
        sex_col=sex_col,
        score_col=score_col,
        min_age=min_age,
        max_age=max_age,
        bin_width=bin_width,
    )
    long_summary = build_cnlsy_longitudinal_summary(
        df,
        id_col=id_col,
        age_col=age_col,
        sex_col=sex_col,
        score_col=score_col,
    )

    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    age_path = tables_dir / "cnlsy_agebin_summary.csv"
    long_path = tables_dir / "cnlsy_longitudinal_model.csv"
    age_summary[AGEBIN_SUMMARY_COLUMNS].to_csv(age_path, index=False)
    long_summary[LONGITUDINAL_SUMMARY_COLUMNS].to_csv(long_path, index=False)

    mean_plot = figures_dir / "cnlsy_age_trends_mean.png"
    vr_plot = figures_dir / "cnlsy_age_trends_vr.png"
    _write_trend_plot(
        age_summary,
        x_col="age_min",
        y_col="mean_diff",
        output=mean_plot,
        title="CNLSY mean differences by age bin",
        ylabel="Female minus male mean",
    )
    _write_trend_plot(
        age_summary,
        x_col="age_min",
        y_col="variance_ratio",
        output=vr_plot,
        title="CNLSY variance ratios by age bin",
        ylabel="Female/male variance ratio",
    )

    print(f"[ok] wrote {age_path}")
    print(f"[ok] wrote {long_path}")
    print(f"[ok] wrote {mean_plot}")
    print(f"[ok] wrote {vr_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
