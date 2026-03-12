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

COHORTS: tuple[str, ...] = ("nlsy79", "nlsy97", "cnlsy")
DEFAULT_SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)
DEFAULT_G_MEAN_TEMPLATE = "outputs/tables/g_mean_diff_one_pair_per_family_seed_{seed}.csv"
DEFAULT_G_VR_TEMPLATE = "outputs/tables/g_variance_ratio_one_pair_per_family_seed_{seed}.csv"
DEFAULT_RUN_OUTPUT = Path("outputs/tables/dedup_seed_sensitivity_runs.csv")
DEFAULT_SUMMARY_OUTPUT = Path("outputs/tables/dedup_seed_sensitivity_summary.csv")


def _safe_float(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    number = float(number)
    if not math.isfinite(number):
        return None
    return number


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _cohort_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _read_metric(
    path: Path,
    cohort: str,
    candidate_columns: tuple[str, ...],
) -> tuple[float | None, str | None]:
    df = _safe_read_csv(path)
    if df.empty or "cohort" not in df.columns:
        return None, None

    cohort_rows = df[df["cohort"].astype(str).str.strip() == cohort]
    if cohort_rows.empty:
        return None, None
    row = cohort_rows.iloc[0]
    for column in candidate_columns:
        if column not in df.columns:
            continue
        value = _safe_float(row.get(column))
        if value is not None:
            return value, column
    return None, None


def _read_d_g(path: Path, cohort: str) -> tuple[float | None, str | None]:
    return _read_metric(path, cohort, ("d_g", "mean_diff"))


def _read_log_vr_g(path: Path, cohort: str) -> tuple[float | None, str | None, bool]:
    df = _safe_read_csv(path)
    if df.empty or "cohort" not in df.columns:
        return None, None, False
    cohort_rows = df[df["cohort"].astype(str).str.strip() == cohort]
    if cohort_rows.empty:
        return None, None, False
    row = cohort_rows.iloc[0]
    for column in ("log_vr_g", "log_vr", "logVR", "log_vr_ratio"):
        if column in df.columns:
            value = _safe_float(row.get(column))
            if value is not None:
                return value, column, False
    for column in ("VR_g", "VR", "variance_ratio"):
        if column in df.columns:
            value = _safe_float(row.get(column))
            if value is None or value <= 0.0:
                return None, column, False
            return math.log(value), column, True
    return None, None, False


def _format_path(template: str, seed: int) -> str:
    return template.format(seed=seed)


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _collect_run_rows(
    root: Path,
    seeds: list[int],
    cohorts: list[str],
    g_mean_template: str,
    g_vr_template: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        mean_path = _resolve_path(root, _format_path(g_mean_template, seed))
        vr_path = _resolve_path(root, _format_path(g_vr_template, seed))
        for cohort in cohorts:
            missing: list[str] = []
            if not mean_path.exists():
                missing.append(f"missing_d_g_file:{mean_path.name}")
            if not vr_path.exists():
                missing.append(f"missing_log_vr_file:{vr_path.name}")

            d_g, d_g_source = _read_d_g(mean_path, cohort)
            if d_g is None and d_g_source is None:
                if not mean_path.exists():
                    pass
                elif not mean_path.exists() or not mean_path.exists():
                    pass
                else:
                    missing.append(f"missing_d_g_metric:{cohort}")
            log_vr_g, log_vr_source, _ = _read_log_vr_g(vr_path, cohort)
            if log_vr_g is None and log_vr_source is None and vr_path.exists():
                missing.append(f"missing_log_vr_metric:{cohort}")

            row: dict[str, Any] = {
                "seed": int(seed),
                "cohort": cohort,
                "d_g": d_g,
                "log_vr_g": log_vr_g,
                "d_g_source": d_g_source,
                "log_vr_source": log_vr_source,
                "missing_reason": ";".join(missing) if missing else "",
            }
            rows.append(row)
    return rows


def _safe_sd(values: pd.Series) -> float | None:
    if values.empty:
        return None
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1))


def _build_summary_rows(
    run_rows: list[dict[str, Any]],
    cohorts: list[str],
) -> list[dict[str, Any]]:
    if not run_rows:
        return [
            {
                "cohort": cohort,
                "n_runs": 0,
                "d_g_sd": pd.NA,
                "d_g_mean": pd.NA,
                "d_g_min": pd.NA,
                "d_g_max": pd.NA,
                "log_vr_g_sd": pd.NA,
                "log_vr_g_mean": pd.NA,
                "log_vr_g_min": pd.NA,
                "log_vr_g_max": pd.NA,
            }
            for cohort in sorted(set(COHORTS).intersection(set(cohorts)))
        ]

    df = pd.DataFrame(run_rows)
    summary: list[dict[str, Any]] = []
    for cohort in sorted(set(cohorts)):
        subset = df[df["cohort"].astype(str) == str(cohort)]
        d_g = pd.to_numeric(subset["d_g"], errors="coerce").dropna()
        log_vr_g = pd.to_numeric(subset["log_vr_g"], errors="coerce").dropna()

        n_runs = int(len(subset))
        d_g_finite = d_g.loc[d_g.notna()]
        log_vr_finite = log_vr_g.loc[log_vr_g.notna()]

        row = {
            "cohort": cohort,
            "n_runs": int(len(subset)),
            "d_g_sd": _safe_sd(d_g_finite),
            "d_g_mean": float(d_g_finite.mean()) if not d_g_finite.empty else pd.NA,
            "d_g_min": float(d_g_finite.min()) if not d_g_finite.empty else pd.NA,
            "d_g_max": float(d_g_finite.max()) if not d_g_finite.empty else pd.NA,
            "log_vr_g_sd": _safe_sd(log_vr_finite),
            "log_vr_g_mean": float(log_vr_finite.mean()) if not log_vr_finite.empty else pd.NA,
            "log_vr_g_min": float(log_vr_finite.min()) if not log_vr_finite.empty else pd.NA,
            "log_vr_g_max": float(log_vr_finite.max()) if not log_vr_finite.empty else pd.NA,
        }
        summary.append(row)
    return summary


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_dedup_seed_sensitivity(
    *,
    root: Path,
    seeds: list[int],
    cohorts: list[str],
    g_mean_template: str,
    g_vr_template: str,
    run_output: Path,
    summary_output: Path,
) -> dict[str, str]:
    unique_seeds = sorted({int(seed) for seed in seeds})
    unique_cohorts = [str(c).strip() for c in cohorts if str(c).strip()]

    if not unique_cohorts:
        unique_cohorts = list(COHORTS)
    unique_cohorts = sorted({c for c in unique_cohorts if c})

    run_rows = _collect_run_rows(
        root=root,
        seeds=unique_seeds,
        cohorts=unique_cohorts,
        g_mean_template=g_mean_template,
        g_vr_template=g_vr_template,
    )
    run_df = pd.DataFrame(run_rows)
    if run_df.empty:
        run_df = pd.DataFrame(
            columns=["seed", "cohort", "d_g", "log_vr_g", "d_g_source", "log_vr_source", "missing_reason"]
        )
    run_df = run_df.sort_values(["seed", "cohort"], ignore_index=True)
    _write_csv(run_output, run_df)

    summary_rows = _build_summary_rows(run_rows, unique_cohorts)
    summary_df = pd.DataFrame(summary_rows)
    _write_csv(summary_output, summary_df)

    return {
        "run_csv": str(run_output),
        "summary_csv": str(summary_output),
        "n_rows": str(len(run_rows)),
        "n_cohorts": str(len(unique_cohorts)),
        "n_seeds": str(len(unique_seeds)),
    }


def _cohort_arg(values: list[str] | None) -> list[str]:
    if values:
        return values
    return list(COHORTS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic dedup-seed sensitivity check for d_g and log_vr_g.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Seed value. Repeat to pass multiple seeds.",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        help="Cohort name. Repeat to pass multiple cohorts.",
    )
    parser.add_argument(
        "--g-mean-template",
        default=DEFAULT_G_MEAN_TEMPLATE,
        help="Template for mean file with {seed} placeholder, default: outputs/tables/g_mean_diff_one_pair_per_family_seed_{seed}.csv",
    )
    parser.add_argument(
        "--g-variance-ratio-template",
        default=DEFAULT_G_VR_TEMPLATE,
        help="Template for variance-ratio file with {seed} placeholder, default: outputs/tables/g_variance_ratio_one_pair_per_family_seed_{seed}.csv",
    )
    parser.add_argument(
        "--run-output",
        type=Path,
        default=DEFAULT_RUN_OUTPUT,
        help="Output path for per-run seed/cohort rows.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Output path for dispersion summary rows.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    run_output = args.run_output if args.run_output.is_absolute() else root / args.run_output
    summary_output = args.summary_output if args.summary_output.is_absolute() else root / args.summary_output

    try:
        outputs = run_dedup_seed_sensitivity(
            root=root,
            seeds=list(args.seed) if args.seed else list(DEFAULT_SEEDS),
            cohorts=_cohort_arg(args.cohort),
            g_mean_template=str(args.g_mean_template),
            g_vr_template=str(args.g_variance_ratio_template),
            run_output=run_output,
            summary_output=summary_output,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {outputs['run_csv']}")
    print(f"[ok] wrote {outputs['summary_csv']}")
    print(f"[ok] rows={outputs['n_rows']} cohorts={outputs['n_cohorts']} seeds={outputs['n_seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
