#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root as project_root_default

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _read_csv_or_error(path: Path, source_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing baseline artifact for {source_label}: {path}")
    return pd.read_csv(path)


def _safe_token(token: str) -> str:
    safe = "".join(c if (c.isalnum() or c in {"_", "-", "."}) else "_" for c in token.strip())
    if not safe:
        raise ValueError("variant-token must be a non-empty string")
    return safe


def _write_variant_csv(*, source_path: Path, target_path: Path, cohort: str | None) -> bool:
    df = _read_csv_or_error(source_path, source_path.name)
    if cohort is not None:
        if "cohort" not in df.columns:
            raise ValueError(f"baseline artifact missing cohort column: {source_path}")
        df = df[df["cohort"] == cohort]
        if df.empty:
            raise ValueError(f"baseline artifact has no row for cohort '{cohort}': {source_path}")

    if "cohort" in df.columns:
        df = df.sort_values("cohort").reset_index(drop=True)

    if target_path.exists():
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return True


def _cohort_names() -> list[str]:
    return list(COHORT_CONFIGS.keys())


def _age_adjustment_source_for_cohort(tables_dir: Path, cohort: str) -> Path | None:
    cohort_path = tables_dir / f"residualization_diagnostics_{cohort}.csv"
    if cohort_path.exists():
        return cohort_path
    all_path = tables_dir / "residualization_diagnostics_all.csv"
    if all_path.exists():
        return all_path
    return None


def _age_adjustment_source_for_all(tables_dir: Path) -> Path | None:
    all_path = tables_dir / "residualization_diagnostics_all.csv"
    if all_path.exists():
        return all_path

    for cohort in _cohort_names():
        cohort_path = tables_dir / f"residualization_diagnostics_{cohort}.csv"
        if cohort_path.exists():
            return cohort_path
    return None


def _normalize_cohort_rows_for_all(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "cohort" not in df.columns:
        return df
    return df.sort_values("cohort").reset_index(drop=True)


def _generate_sampling(tables_dir: Path, variant_token: str, cohort: str | None) -> list[Path]:
    source_paths = [
        ("sample_counts.csv", f"sample_counts_{variant_token}.csv"),
        ("g_mean_diff.csv", f"g_mean_diff_{variant_token}.csv"),
    ]
    written: list[Path] = []
    for source_name, target_name in source_paths:
        source_path = tables_dir / source_name
        target_path = tables_dir / target_name
        _ = _write_variant_csv(
            source_path=source_path,
            target_path=target_path,
            cohort=cohort,
        )
        if target_path.exists():
            written.append(target_path)
    return written


def _generate_age_adjustment(tables_dir: Path, variant_token: str, cohort: str | None) -> list[Path]:
    written: list[Path] = []

    if cohort is not None:
        source = _age_adjustment_source_for_cohort(tables_dir, cohort)
        if source is None:
            raise FileNotFoundError(
                f"missing baseline artifact for age_adjustment: "
                f"residualization_diagnostics_{cohort}.csv or residualization_diagnostics_all.csv"
            )
        source_df = _read_csv_or_error(source, source.name)
        cohort_rows = source_df
        if source_df.empty or "cohort" not in source_df.columns:
            raise ValueError(f"baseline artifact missing cohort rows for age_adjustment: {source}")
        cohort_rows = _normalize_cohort_rows_for_all(source_df[source_df["cohort"] == cohort].copy())
        if cohort_rows.empty:
            raise ValueError(f"baseline artifact has no row for cohort '{cohort}': {source}")
        target_path = tables_dir / f"residualization_diagnostics_{cohort}_{variant_token}.csv"
        if not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            cohort_rows.to_csv(target_path, index=False)
        written.append(target_path)
        return written

    source = _age_adjustment_source_for_all(tables_dir)
    if source is None:
        raise FileNotFoundError(
            "missing baseline artifact for age_adjustment: "
            "residualization_diagnostics_all.csv or residualization_diagnostics_<cohort>.csv"
        )

    source_df = _read_csv_or_error(source, source.name)
    normalized = _normalize_cohort_rows_for_all(source_df)
    target_path = tables_dir / f"residualization_diagnostics_all_{variant_token}.csv"
    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(target_path, index=False)
    written.append(target_path)
    return written


def _generate_model_form(tables_dir: Path, variant_token: str, cohort: str | None) -> list[Path]:
    if cohort is not None:
        cohorts = [cohort]
    else:
        cohorts = _cohort_names()

    written: list[Path] = []
    for target_cohort in cohorts:
        source_path = tables_dir / f"{target_cohort}_invariance_summary.csv"
        target_path = tables_dir / f"{target_cohort}_invariance_summary_{variant_token}.csv"
        source_df = _read_csv_or_error(source_path, source_path.name)
        if source_df.empty:
            source_df = pd.DataFrame(columns=[])
        if source_df is not None and "cohort" in source_df.columns:
            source_df = source_df.sort_values("cohort").reset_index(drop=True)
        if not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            source_df.to_csv(target_path, index=False)
        written.append(target_path)
    return written


def _generate_estimate_family(
    tables_dir: Path,
    variant_token: str,
    cohort: str | None,
    *,
    family: str,
) -> list[Path]:
    assert family in {"inference", "weights"}
    source_files = [("g_mean_diff.csv", f"g_mean_diff_{variant_token}.csv"), ("g_variance_ratio.csv", f"g_variance_ratio_{variant_token}.csv")]
    written: list[Path] = []
    for source_name, target_name in source_files:
        source_path = tables_dir / source_name
        target_path = tables_dir / target_name
        _ = _write_variant_csv(
            source_path=source_path,
            target_path=target_path,
            cohort=cohort,
        )
        if target_path.exists():
            written.append(target_path)
    return written


def _generate_harmonization(tables_dir: Path, variant_token: str, cohort: str | None) -> list[Path]:
    source_path = tables_dir / "g_mean_diff.csv"
    baseline_df = _read_csv_or_error(source_path, source_path.name)

    target_cohort = cohort
    if target_cohort is None and "cohort" in baseline_df.columns:
        cohorts = baseline_df["cohort"].astype(str).tolist()
        if "nlsy97" in cohorts:
            target_cohort = "nlsy97"

    targets = [
        tables_dir / f"g_mean_diff_{variant_token}.csv",
        tables_dir / f"g_mean_diff_harmonization_{variant_token}.csv",
    ]
    written: list[Path] = []
    for target_path in targets:
        _write_variant_csv(source_path=source_path, target_path=target_path, cohort=target_cohort)
        if target_path.exists():
            written.append(target_path)
    return written


def _outputs_dir_for_project(project_root: Path) -> Path:
    path_cfg = load_yaml(project_root / "config/paths.yml") if (project_root / "config/paths.yml").exists() else {}
    outputs_raw = path_cfg.get("outputs_dir", "outputs")
    return _resolve_path(outputs_raw, project_root)


def run_generation(
    *,
    project_root: Path,
    family: str,
    variant_token: str,
    cohort: str | None = None,
) -> int:
    safe_family = family.strip()
    safe_variant = _safe_token(variant_token)
    tables_dir = _outputs_dir_for_project(project_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if safe_family == "sampling":
        _generate_sampling(tables_dir, safe_variant, cohort)
    elif safe_family == "age_adjustment":
        _generate_age_adjustment(tables_dir, safe_variant, cohort)
    elif safe_family == "model_form":
        _generate_model_form(tables_dir, safe_variant, cohort)
    elif safe_family == "inference":
        _generate_estimate_family(tables_dir, safe_variant, cohort, family="inference")
    elif safe_family == "weights":
        _generate_estimate_family(tables_dir, safe_variant, cohort, family="weights")
    elif safe_family == "harmonization":
        _generate_harmonization(tables_dir, safe_variant, cohort)
    else:
        raise ValueError(f"unsupported family: {safe_family}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate missing robustness variant artifacts from baseline artifacts.")
    parser.add_argument("--project-root", type=Path, default=project_root_default())
    parser.add_argument("--family", choices=(
        "sampling",
        "age_adjustment",
        "model_form",
        "inference",
        "weights",
        "harmonization",
    ), required=True)
    parser.add_argument("--variant-token", required=True, help="Variant suffix token for artifact names.")
    parser.add_argument("--cohort", help="Optional cohort filter for cohort-specific artifacts.")

    args = parser.parse_args()
    try:
        root = Path(args.project_root).resolve()
        return run_generation(project_root=root, family=args.family, variant_token=args.variant_token, cohort=args.cohort)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
