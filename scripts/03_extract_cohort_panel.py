#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root, relative_path


COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _find_source_file(cohort_dir: Path) -> Path:
    panel_parquet = cohort_dir / "panel.parquet"
    if panel_parquet.exists():
        return panel_parquet

    raw_files = sorted((cohort_dir / "raw_files").rglob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No panel source file found under {cohort_dir / 'raw_files'}")
    return raw_files[0]


def _resolve_columns(available: list[str], cohort_cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    panel_cfg = cohort_cfg.get("panel_extract", {}) if isinstance(cohort_cfg.get("panel_extract", {}), dict) else {}
    required_columns = [str(v) for v in panel_cfg.get("required_columns", [])]
    optional_columns = [str(v) for v in panel_cfg.get("optional_columns", [])]
    patterns = [str(v) for v in panel_cfg.get("column_patterns", [])]

    configured = bool(required_columns or optional_columns or patterns)
    if not configured:
        sample_cfg = cohort_cfg.get("sample_construct", {})
        if not isinstance(sample_cfg, dict):
            sample_cfg = {}
        column_map = sample_cfg.get("column_map", {})
        if not isinstance(column_map, dict):
            column_map = {}
        normalized_map = {str(raw): str(mapped) for raw, mapped in column_map.items()}
        inverse_map = {mapped: raw for raw, mapped in normalized_map.items()}

        logical_vars: list[str] = []
        for col in (
            sample_cfg.get("id_col"),
            sample_cfg.get("sex_col"),
            sample_cfg.get("age_col"),
            sample_cfg.get("age_resid_col"),
        ):
            if col is not None:
                logical_vars.append(str(col))

        for subtest in sample_cfg.get("subtests", []):
            logical_vars.append(str(subtest))

        candidates: list[str] = []
        for logical in logical_vars:
            candidates.append(inverse_map.get(logical, logical))

        for raw_col in normalized_map.keys():
            candidates.append(raw_col)

        deduped_candidates: list[str] = []
        seen: set[str] = set()
        for col in candidates:
            if col not in seen:
                deduped_candidates.append(col)
                seen.add(col)

        fallback_selected = [col for col in deduped_candidates if col in available]
        if fallback_selected:
            return fallback_selected, [col for col in deduped_candidates if col not in fallback_selected]

        raise ValueError(
            "No panel_extract selector was configured and no sample_construct-derived columns were found. "
            "Configure panel_extract.required_columns/optional_columns/column_patterns in cohort config."
        )

    requested = list(dict.fromkeys(required_columns + optional_columns))
    selected: list[str] = []
    missing_optional: list[str] = []

    for col in requested:
        if col in available and col not in selected:
            selected.append(col)
        elif col in required_columns:
            raise ValueError(f"Required column not found in source: {col}")
        else:
            missing_optional.append(col)

    for pattern in patterns:
        regex = re.compile(pattern)
        for col in available:
            if col not in selected and regex.search(col):
                selected.append(col)

    if not selected:
        raise ValueError("No configured columns matched the source schema.")

    return selected, missing_optional


def _csv_header_columns(source_path: Path) -> list[str]:
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if not header:
        raise ValueError(f"CSV source has no header row: {source_path}")
    return [str(col).strip() for col in header]


def _selection_signature(cohort: str, source_path: Path, source_path_token: str, columns: list[str]) -> str:
    payload = {
        "cohort": cohort,
        "source_path": source_path_token,
        "source_size_bytes": source_path.stat().st_size,
        "source_mtime_ns": source_path.stat().st_mtime_ns,
        "columns": columns,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_cohort_cfg(project_root_path: Path, cohort: str) -> dict[str, Any]:
    cohort_cfg_path = project_root_path / COHORT_CONFIGS[cohort]
    if not cohort_cfg_path.exists():
        raise FileNotFoundError(f"Missing cohort config: {cohort_cfg_path}")
    return load_yaml(cohort_cfg_path)


def _extract_panel_for_cohort(
    project_root_path: Path,
    cohort: str,
    source_override: Path | None,
    force: bool,
    compute_row_count: bool,
) -> Path:
    paths_cfg = load_yaml(project_root_path / "config/paths.yml")
    interim_dir = _resolve_path(paths_cfg["interim_dir"], project_root_path)
    cohort_dir = interim_dir / cohort

    cohort_cfg = _load_cohort_cfg(project_root_path, cohort)
    source_path = source_override if source_override is not None else _find_source_file(cohort_dir)

    is_parquet = source_path.suffix.lower() == ".parquet"
    if is_parquet:
        scanner = pl.scan_parquet(str(source_path))
        available_columns = list(scanner.collect_schema().names())
    else:
        scanner = None
        available_columns = _csv_header_columns(source_path)

    selected_columns, missing_optional = _resolve_columns(available_columns, cohort_cfg)
    out_path = cohort_dir / "panel_extract.csv"
    manifest_path = cohort_dir / "panel_extract.manifest.json"

    source_path_token = relative_path(project_root_path, source_path)
    signature = _selection_signature(cohort, source_path, source_path_token, selected_columns)
    legacy_signature = _selection_signature(cohort, source_path, str(source_path), selected_columns)
    if out_path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("selection_signature") in {signature, legacy_signature} and not force:
            print(f"[skip] {out_path} already up-to-date")
            return out_path
        if existing.get("selection_signature") not in {signature, legacy_signature} and not force:
            raise FileExistsError(
                f"Output exists but selection changed; rerun with --force to overwrite: {out_path}"
            )

    if out_path.exists() and not force:
        raise FileExistsError(f"Output exists; use --force: {out_path}")

    if is_parquet:
        assert scanner is not None
        projected = scanner.select(selected_columns)
        projected.sink_csv(str(out_path))
        row_count = int(projected.select(pl.len()).collect().item()) if compute_row_count else None
    else:
        projected_df = pl.read_csv(
            str(source_path),
            columns=selected_columns,
            infer_schema_length=1000,
            ignore_errors=True,
        )
        projected_df.write_csv(str(out_path))
        row_count = int(projected_df.height) if compute_row_count else None

    manifest = {
        "cohort": cohort,
        "source_path": source_path_token,
        "output_path": relative_path(project_root_path, out_path),
        "selection_signature": signature,
        "selected_columns": selected_columns,
        "missing_optional_columns": missing_optional,
        "n_rows": row_count,
        "n_columns": len(selected_columns),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[ok] wrote {out_path} with {manifest['n_rows']} rows and {manifest['n_columns']} columns"
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract cohort panel artifact from extracted cohort files.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort selector.")
    parser.add_argument("--all", action="store_true", help="Process all configured cohorts.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root(),
        help="Project root for config and interim outputs.",
    )
    parser.add_argument("--source-path", type=Path, help="Explicit source CSV/Parquet path.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--compute-row-count",
        action="store_true",
        help="Compute output row count in manifest (requires an additional full-file scan).",
    )
    args = parser.parse_args()

    args.project_root = Path(args.project_root).resolve()
    source_path = args.source_path.resolve() if args.source_path else None

    for cohort in _cohorts_from_args(args):
        _extract_panel_for_cohort(
            project_root_path=args.project_root,
            cohort=cohort,
            source_override=source_path,
            force=args.force,
            compute_row_count=args.compute_row_count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
