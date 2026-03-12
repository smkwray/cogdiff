#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, resolve_from_project

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}


def _safe_member_name(name: str) -> Path:
    normalized = name.replace("\\", "/")
    member_path = Path(normalized)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"Unsafe zip member path: {name}")
    return member_path


def _extract_zip(zip_path: Path, out_dir: Path, force: bool) -> list[dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        entries = []
        for info in archive.infolist():
            safe_member = _safe_member_name(info.filename)
            target = out_dir / safe_member

            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                entries.append(
                    {
                        "filename": info.filename,
                        "file_size": info.file_size,
                        "compress_size": info.compress_size,
                    }
                )
                continue

            entries.append(
                {
                    "filename": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                }
            )

            if target.exists() and not force:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(chunk)

    return entries


def _convert_first_csv_to_parquet(raw_dir: Path, parquet_path: Path) -> str:
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if not csv_files:
        return "no-csv-found"

    csv_path = csv_files[0]
    try:
        import polars as pl
    except ImportError:
        return "polars-missing"

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    scan = pl.scan_csv(str(csv_path), infer_schema_length=1000)
    scan.sink_parquet(str(parquet_path))
    return f"ok:{csv_path.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Unpack cohort zips and index contents.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process")
    parser.add_argument("--all", action="store_true", help="Process all cohorts")
    parser.add_argument("--force", action="store_true", help="Re-extract already present files")
    parser.add_argument("--to-parquet", action="store_true", help="Convert first discovered CSV to parquet")
    args = parser.parse_args()

    cohorts = list(COHORT_CONFIGS.keys()) if args.all or not args.cohort else args.cohort
    paths_cfg = load_yaml(resolve_from_project("config/paths.yml"))
    raw_dir = resolve_from_project(paths_cfg["raw_dir"])
    interim_dir = resolve_from_project(paths_cfg["interim_dir"])

    for cohort in cohorts:
        cfg = load_yaml(resolve_from_project(COHORT_CONFIGS[cohort]))
        zip_path = raw_dir / cfg["raw_zip_name"]
        if not zip_path.exists():
            raise FileNotFoundError(f"Raw zip missing for {cohort}: {zip_path}")

        cohort_dir = interim_dir / cohort
        raw_extract_dir = cohort_dir / "raw_files"

        print(f"[extract] {cohort}: {zip_path} -> {raw_extract_dir}")
        entries = _extract_zip(zip_path, raw_extract_dir, force=args.force)

        index_path = cohort_dir / "zip_contents.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2)
        print(f"[ok] wrote index: {index_path}")

        if args.to_parquet:
            parquet_path = cohort_dir / "panel.parquet"
            status = _convert_first_csv_to_parquet(raw_extract_dir, parquet_path)
            print(f"[parquet] {cohort}: {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
