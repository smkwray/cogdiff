#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

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

OUTPUT_COLUMNS = (
    "cohort",
    "sex_group",
    "subtest",
    "n_total",
    "n_missing",
    "missing_rate",
    "n_unique_ids",
    "age_mean",
)
SELECTION_COLUMNS = (
    "cohort",
    "sex_group",
    "n_panel",
    "n_included",
    "n_excluded",
    "inclusion_rate",
)


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _normalize_sex(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    token = str(value).strip().lower()
    if token in {"f", "female", "2", "w", "woman", "girl"}:
        return "female"
    if token in {"m", "male", "1", "man", "boy"}:
        return "male"
    return token


def _normalize_missing_codes(df: pd.DataFrame, missing_codes: Iterable[Any]) -> pd.DataFrame:
    out = df.copy()
    for code in missing_codes:
        out = out.replace(code, np.nan)
        out = out.replace(str(code), np.nan)
    return out


def _diagnostic_rows(
    df: pd.DataFrame,
    *,
    cohort: str,
    id_col: str,
    age_col: str,
    sex_col: str,
    subtests: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = df.copy()
    work["_sex_group"] = work[sex_col].map(_normalize_sex)

    groups = ("all", "female", "male")
    for group in groups:
        if group == "all":
            group_df = work
        else:
            group_df = work[work["_sex_group"] == group]

        for subtest in subtests:
            if subtest not in group_df.columns:
                continue
            values = pd.to_numeric(group_df[subtest], errors="coerce")
            total = int(len(group_df))
            missing = int(values.isna().sum())
            missing_rate = float(missing / total) if total > 0 else float("nan")
            age_mean = pd.to_numeric(group_df[age_col], errors="coerce").mean()
            rows.append(
                {
                    "cohort": cohort,
                    "sex_group": group,
                    "subtest": subtest,
                    "n_total": total,
                    "n_missing": missing,
                    "missing_rate": missing_rate,
                    "n_unique_ids": int(group_df[id_col].nunique(dropna=True)),
                    "age_mean": float(age_mean) if pd.notna(age_mean) else np.nan,
                }
            )
    return rows


def _load_input_for_cohort(
    *,
    root: Path,
    paths_cfg: dict[str, Any],
    cohort: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cohort_cfg_path = root / COHORT_CONFIGS[cohort]
    cohort_cfg = load_yaml(cohort_cfg_path)
    sample_cfg = cohort_cfg.get("sample_construct", {})
    if not isinstance(sample_cfg, dict):
        raise ValueError(f"{cohort_cfg_path} sample_construct must be a mapping.")

    processed_dir = _resolve_path(paths_cfg["processed_dir"], root)
    preferred = processed_dir / f"{cohort}_cfa_resid.csv"
    fallback = processed_dir / f"{cohort}_cfa.csv"
    input_path = preferred if preferred.exists() else fallback
    if not input_path.exists():
        return pd.DataFrame(), sample_cfg
    return pd.read_csv(input_path, low_memory=False), sample_cfg


def _selection_rows(
    *,
    panel_df: pd.DataFrame,
    included_df: pd.DataFrame,
    cohort: str,
    id_col: str,
    sex_col: str,
) -> list[dict[str, Any]]:
    if panel_df.empty or id_col not in panel_df.columns:
        return []

    panel = panel_df[[id_col] + ([sex_col] if sex_col in panel_df.columns else [])].copy()
    panel = panel.dropna(subset=[id_col]).drop_duplicates(subset=[id_col], keep="first")
    panel["_sex_group"] = panel[sex_col].map(_normalize_sex) if sex_col in panel.columns else "unknown"

    included_ids: set[str] = set()
    if not included_df.empty and id_col in included_df.columns:
        included_ids = set(included_df[id_col].dropna().astype(str).tolist())

    rows: list[dict[str, Any]] = []
    for group in ("all", "female", "male"):
        if group == "all":
            group_panel = panel
        else:
            group_panel = panel[panel["_sex_group"] == group]

        n_panel = int(len(group_panel))
        panel_ids = set(group_panel[id_col].astype(str).tolist())
        n_included = int(len(panel_ids & included_ids))
        n_excluded = int(n_panel - n_included)
        inclusion_rate = float(n_included / n_panel) if n_panel > 0 else float("nan")
        rows.append(
            {
                "cohort": cohort,
                "sex_group": group,
                "n_panel": n_panel,
                "n_included": n_included,
                "n_excluded": n_excluded,
                "inclusion_rate": inclusion_rate,
            }
        )
    return rows


def _write_heatmap(rows: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    if rows.empty:
        ax.text(0.5, 0.5, "No missingness data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Missingness heatmap")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    table = rows.pivot_table(
        index=["cohort", "subtest"],
        columns="sex_group",
        values="missing_rate",
        aggfunc="mean",
    )
    for group in ("female", "male", "all"):
        if group not in table.columns:
            table[group] = np.nan
    table = table[["female", "male", "all"]]
    matrix = table.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)

    image = ax.imshow(masked, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels([str(c) for c in table.columns])
    ylabels = [f"{c}:{s}" for c, s in table.index]
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title("Subtest missingness by cohort and sex")
    ax.set_xlabel("Sex group")
    ax.set_ylabel("Cohort:Subtest")
    fig.colorbar(image, ax=ax, label="Missing rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out[list(OUTPUT_COLUMNS)]


def _ensure_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in SELECTION_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out[list(SELECTION_COLUMNS)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate missingness diagnostics and heatmap outputs.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    paths_cfg = load_yaml(root / "config/paths.yml")
    outputs_dir = _resolve_path(paths_cfg["outputs_dir"], root)
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    all_selection_rows: list[dict[str, Any]] = []
    for cohort in _cohorts_from_args(args):
        frame, sample_cfg = _load_input_for_cohort(root=root, paths_cfg=paths_cfg, cohort=cohort)
        if frame.empty:
            continue

        id_col = str(sample_cfg.get("id_col", "person_id"))
        age_col = str(sample_cfg.get("age_col", "age"))
        sex_col = str(sample_cfg.get("sex_col", "sex"))
        subtests = [str(x) for x in sample_cfg.get("subtests", [])]
        missing_codes = sample_cfg.get("missing_codes", [])
        if not isinstance(missing_codes, list):
            missing_codes = []

        required = [id_col, age_col, sex_col]
        missing_required = [col for col in required if col not in frame.columns]
        if missing_required:
            print(f"[warn] skipping {cohort}: missing required columns {missing_required}")
            continue

        clean = _normalize_missing_codes(frame, missing_codes)
        rows = _diagnostic_rows(
            clean,
            cohort=cohort,
            id_col=id_col,
            age_col=age_col,
            sex_col=sex_col,
            subtests=subtests,
        )
        all_rows.extend(rows)

        interim_dir = _resolve_path(paths_cfg.get("interim_dir", "data/interim"), root)
        panel_input_name = str(sample_cfg.get("input_file", "panel_extract.csv"))
        panel_path = interim_dir / cohort / panel_input_name
        if panel_path.exists():
            panel_df = pd.read_csv(panel_path, low_memory=False)
            all_selection_rows.extend(
                _selection_rows(
                    panel_df=panel_df,
                    included_df=clean,
                    cohort=cohort,
                    id_col=id_col,
                    sex_col=sex_col,
                )
            )

    output_df = _ensure_columns(pd.DataFrame(all_rows))
    table_path = tables_dir / "missingness_diagnostics.csv"
    output_df.to_csv(table_path, index=False)
    selection_df = _ensure_selection_columns(pd.DataFrame(all_selection_rows))
    selection_path = tables_dir / "inclusion_exclusion_diagnostics.csv"
    selection_df.to_csv(selection_path, index=False)

    heatmap_path = figures_dir / "missingness_heatmap.png"
    _write_heatmap(output_df, heatmap_path)

    print(f"[ok] wrote {table_path}")
    print(f"[ok] wrote {selection_path}")
    print(f"[ok] wrote {heatmap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
