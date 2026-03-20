#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import project_root

DEFAULT_SNAPSHOT_PATHS: tuple[str, ...] = (
    "outputs/tables/analysis_tiers.csv",
    "outputs/tables/specification_stability_summary.csv",
    "outputs/tables/confirmatory_exclusions.csv",
)
MANIFEST_COLUMNS = ("snapshot_utc", "path", "sha256", "size_bytes", "mtime_utc")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_row(path: Path, *, snapshot_utc: str, root: Path) -> dict[str, str]:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    try:
        rel = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = str(path.resolve())
    return {
        "snapshot_utc": snapshot_utc,
        "path": rel,
        "sha256": _sha256(path),
        "size_bytes": str(int(path.stat().st_size)),
        "mtime_utc": mtime,
    }


def _write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _coerce_path_list(manifest_path: Path, explicit_paths: list[str], root: Path) -> list[Path]:
    if explicit_paths:
        return [root / p for p in explicit_paths]
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if "path" in manifest.columns and not manifest.empty:
            return [root / str(p) for p in manifest["path"].astype(str).tolist()]
    return [root / p for p in DEFAULT_SNAPSHOT_PATHS]


def _update_manifest(manifest_path: Path, explicit_paths: list[str], root: Path) -> int:
    paths = _coerce_path_list(manifest_path, explicit_paths, root)
    snapshot_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    rows: list[dict[str, str]] = []
    missing = [p for p in paths if not p.exists()]
    if missing:
        for path in missing:
            print(f"[error] missing snapshot path: {path}", file=sys.stderr)
        return 1
    for path in paths:
        rows.append(_snapshot_row(path, snapshot_utc=snapshot_utc, root=root))
    _write_manifest(manifest_path, rows)
    print(f"[ok] wrote {manifest_path}")
    return 0


def _verify_manifest(manifest_path: Path, root: Path) -> int:
    if not manifest_path.exists():
        print(f"[error] missing manifest: {manifest_path}", file=sys.stderr)
        return 1
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        print(f"[error] manifest is empty: {manifest_path}", file=sys.stderr)
        return 1
    for required in MANIFEST_COLUMNS:
        if required not in manifest.columns:
            print(f"[error] manifest missing column: {required}", file=sys.stderr)
            return 1

    # Fail if SEM fallback sentinel files or status artifacts indicate fallback was used
    fallback_cohorts: list[str] = []
    model_fits_dir = root / "outputs" / "model_fits"
    if model_fits_dir.exists():
        for flag in model_fits_dir.rglob("SEM_FALLBACK_USED.flag"):
            fallback_cohorts.append(flag.parent.name)
        for rs in model_fits_dir.rglob("run_status.json"):
            try:
                import json as _json
                status_data = _json.loads(rs.read_text(encoding="utf-8"))
                if status_data.get("python_fallback"):
                    cohort_name = rs.parent.name
                    if cohort_name not in fallback_cohorts:
                        fallback_cohorts.append(cohort_name)
            except Exception:
                pass
    if fallback_cohorts:
        print(
            f"[error] SEM fallback was used for cohort(s) {fallback_cohorts}. "
            f"Snapshot contains approximate results, not real SEM estimates.",
            file=sys.stderr,
        )
        return 1

    failures: list[str] = []
    for _, row in manifest.iterrows():
        rel_path = str(row["path"])
        file_path = root / rel_path
        if not file_path.exists():
            failures.append(f"{rel_path}: missing")
            continue
        expected_hash = str(row["sha256"]).strip()
        expected_size = int(pd.to_numeric(pd.Series([row["size_bytes"]]), errors="coerce").fillna(-1).iloc[0])
        actual_hash = _sha256(file_path)
        actual_size = int(file_path.stat().st_size)
        if actual_hash != expected_hash:
            failures.append(f"{rel_path}: sha256 mismatch expected={expected_hash} actual={actual_hash}")
        if actual_size != expected_size:
            failures.append(f"{rel_path}: size mismatch expected={expected_size} actual={actual_size}")

    if failures:
        print("[error] publication snapshot verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        cmd = (
            f"{sys.executable} scripts/25_verify_publication_snapshot.py "
            f"--project-root {root} --update-manifest"
        )
        print(f"[hint] Run to refresh after intentional changes: {cmd}", file=sys.stderr)
        return 1

    print(f"[ok] verified {len(manifest)} snapshot entries: {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify or update publication snapshot manifest hashes.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("outputs/tables/publication_snapshot_manifest.csv"),
        help="Manifest path (relative to project root unless absolute).",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Optional relative path(s) to include when writing manifest in --update-manifest mode.",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Rewrite the snapshot manifest with current file hashes.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    manifest_path = args.manifest_path if args.manifest_path.is_absolute() else root / args.manifest_path

    if args.update_manifest:
        return _update_manifest(manifest_path, list(args.path), root)
    return _verify_manifest(manifest_path, root)


if __name__ == "__main__":
    raise SystemExit(main())

