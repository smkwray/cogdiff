#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sampling import ResidualDiagnostics, residualize_quadratic, standardize_series

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}


def _as_label(value: Any) -> str:
    if isinstance(value, bool):
        return "NO" if value is False else "YES"
    return str(value)


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _default_subtests(cohort: str, models_cfg: dict[str, Any]) -> list[str]:
    if cohort == "cnlsy":
        return [_as_label(x) for x in models_cfg.get("cnlsy_single_factor", [])]
    groups = models_cfg.get("hierarchical_factors", {})
    ordered: list[str] = []
    for key in ("speed", "math", "verbal", "technical"):
        for val in groups.get(key, []):
            sval = _as_label(val)
            if sval not in ordered:
                ordered.append(sval)
    return ordered


def _sample_cfg(cohort_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = cohort_cfg.get("sample_construct", {})
    return cfg if isinstance(cfg, dict) else {}


def _apply_column_mapping(df: pd.DataFrame, sample_cfg: dict[str, Any]) -> pd.DataFrame:
    mapping = sample_cfg.get("column_map", {})
    if not isinstance(mapping, dict) or not mapping:
        return df
    rename_map: dict[str, str] = {}
    for raw, mapped in mapping.items():
        raw_col = str(raw)
        mapped_col = str(mapped)
        if raw_col in df.columns and mapped_col not in df.columns:
            rename_map[raw_col] = mapped_col
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def _should_standardize_output(sample_cfg: dict[str, Any], cli_flag: bool) -> bool:
    return bool(cli_flag or sample_cfg.get("standardize_output", False))


def _diagnostic_row(cohort: str, subtest: str, predictor_col: str, diag: ResidualDiagnostics) -> dict[str, Any]:
    return {
        "cohort": cohort,
        "subtest": subtest,
        "predictor_col": predictor_col,
        "n_used": diag.n_used,
        "r2": diag.r2,
        "beta0": diag.beta0,
        "beta1": diag.beta1,
        "beta2": diag.beta2,
        "resid_sd": diag.resid_sd,
        "outliers_3sd": diag.outliers_3sd,
    }


def _process_cohort(
    root: Path,
    cohort: str,
    paths_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    source_override: Path | None,
    standardize_output: bool,
) -> list[dict[str, Any]]:
    cohort_cfg = load_yaml(root / COHORT_CONFIGS[cohort])
    sample_cfg = _sample_cfg(cohort_cfg)
    standardize_output = _should_standardize_output(sample_cfg, standardize_output)
    processed_dir = _resolve_path(paths_cfg["processed_dir"], root)
    outputs_dir = _resolve_path(paths_cfg["outputs_dir"], root)

    input_path = source_override if source_override is not None else processed_dir / f"{cohort}_cfa.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing cohort sample file: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    df = _apply_column_mapping(df, sample_cfg)
    predictor_col = str(sample_cfg.get("age_resid_col", sample_cfg.get("age_col", "age")))
    if predictor_col not in df.columns:
        raise ValueError(f"{cohort}: age residualization predictor column not found: {predictor_col}")

    subtests = [_as_label(x) for x in sample_cfg.get("subtests", _default_subtests(cohort, models_cfg))]
    missing = [c for c in subtests if c not in df.columns]
    if missing:
        raise ValueError(f"{cohort}: missing subtests for residualization: {missing}")

    diagnostics: list[dict[str, Any]] = []
    out = df.copy()
    for subtest in subtests:
        resid, diag = residualize_quadratic(out[subtest], out[predictor_col])
        if standardize_output:
            resid = standardize_series(resid)
        out[subtest] = resid
        diagnostics.append(_diagnostic_row(cohort, subtest, predictor_col, diag))

    out_path = processed_dir / f"{cohort}_cfa_resid.csv"
    out.to_csv(out_path, index=False)

    diag_path = outputs_dir / "tables" / f"residualization_diagnostics_{cohort}.csv"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(diag_path, index=False)

    print(f"[ok] {cohort}: wrote residualized data -> {out_path}")
    print(f"[ok] {cohort}: wrote diagnostics -> {diag_path}")
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Residualize cohort test scores on age/birth-year quadratic terms.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--source-path", type=Path, help="Optional explicit cohort input CSV (single-cohort mode).")
    parser.add_argument(
        "--standardize-output",
        action="store_true",
        help="Standardize each residualized subtest to mean 0 and sample SD 1.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    cohorts = _cohorts_from_args(args)

    if args.source_path is not None and len(cohorts) != 1:
        raise ValueError("--source-path is only supported when exactly one --cohort is provided.")
    source_override = args.source_path.resolve() if args.source_path else None

    all_rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        all_rows.extend(
            _process_cohort(
                root=root,
                cohort=cohort,
                paths_cfg=paths_cfg,
                models_cfg=models_cfg,
                source_override=source_override,
                standardize_output=args.standardize_output,
            )
        )

    summary_path = _resolve_path(paths_cfg["outputs_dir"], root) / "tables" / "residualization_diagnostics_all.csv"
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"[ok] wrote combined diagnostics -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
