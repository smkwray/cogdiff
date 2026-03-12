#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS: tuple[str, ...] = (
    "outputs/tables/results_snapshot.md",
    "outputs/tables/publication_snapshot_manifest.csv",
    "outputs/tables/publication_results_lock",
    "outputs/tables/publication_results_lock/publication_results_lock_manifest.csv",
    "outputs/tables/publication_results_lock/manuscript_results_lock.md",
    "outputs/tables/publication_results_lock.zip",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_required_paths(root: Path) -> list[str]:
    failures: list[str] = []
    for token in REQUIRED_PATHS:
        path = root / token
        if not path.exists():
            failures.append(f"missing required path: {token}")
    return failures


def _run_subprocess_check(argv: list[str], *, label: str) -> list[str]:
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        return []
    detail = (proc.stderr or proc.stdout or "").strip()
    return [f"{label} failed: {detail or f'exit {proc.returncode}'}"]


def _read_bundle_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _check_bundle_integrity(root: Path) -> list[str]:
    failures: list[str] = []
    bundle_dir = root / "outputs/tables/publication_results_lock"
    manifest_path = bundle_dir / "publication_results_lock_manifest.csv"
    methods_path = bundle_dir / "manuscript_results_lock.md"
    zip_path = root / "outputs/tables/publication_results_lock.zip"

    if not bundle_dir.exists() or not manifest_path.exists() or not methods_path.exists() or not zip_path.exists():
        return failures

    rows = _read_bundle_manifest(manifest_path)
    expected_copied = {Path(str(row["bundle_path"])).name for row in rows}
    expected_all = expected_copied | {"publication_results_lock_manifest.csv", "manuscript_results_lock.md"}
    actual_all = {path.name for path in bundle_dir.iterdir() if path.is_file()}

    stale = sorted(actual_all - expected_all)
    missing = sorted(expected_all - actual_all)
    if stale:
        failures.append(f"bundle has stale files: {', '.join(stale)}")
    if missing:
        failures.append(f"bundle missing expected files: {', '.join(missing)}")

    for row in rows:
        bundle_name = Path(str(row["bundle_path"])).name
        bundle_file = bundle_dir / bundle_name
        if not bundle_file.exists():
            continue
        expected_hash = str(row["sha256"]).strip()
        actual_hash = _sha256(bundle_file)
        if actual_hash != expected_hash:
            failures.append(f"bundle hash mismatch for {bundle_name}: expected {expected_hash} actual {actual_hash}")

    with zipfile.ZipFile(zip_path) as archive:
        zipped_names = {Path(name).name for name in archive.namelist() if not name.endswith("/")}
    extra_in_zip = sorted(zipped_names - actual_all)
    missing_in_zip = sorted(actual_all - zipped_names)
    if extra_in_zip:
        failures.append(f"zip has files not present in bundle dir: {', '.join(extra_in_zip)}")
    if missing_in_zip:
        failures.append(f"zip missing bundle files: {', '.join(missing_in_zip)}")

    return failures


def run_doctor(
    *,
    root: Path,
    skip_snapshot_verify: bool = False,
    skip_portability: bool = False,
) -> tuple[bool, list[str]]:
    failures = _check_required_paths(root)

    if not skip_snapshot_verify:
        failures.extend(
            _run_subprocess_check(
                [sys.executable, str(root / "scripts/25_verify_publication_snapshot.py"), "--project-root", str(root)],
                label="publication snapshot verify",
            )
        )
    if not skip_portability:
        failures.extend(
            _run_subprocess_check(
                [sys.executable, str(root / "scripts/99_portability_smoke_check.py"), "--project-root", str(root)],
                label="portability smoke check",
            )
        )

    failures.extend(_check_bundle_integrity(root))
    return (len(failures) == 0), failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Doctor check for publication artifacts, lock bundle consistency, and portability.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root path.")
    parser.add_argument("--skip-snapshot-verify", action="store_true", help="Skip `scripts/25_verify_publication_snapshot.py`.")
    parser.add_argument("--skip-portability", action="store_true", help="Skip `scripts/99_portability_smoke_check.py`.")
    args = parser.parse_args(argv)

    ok, failures = run_doctor(
        root=Path(args.project_root).resolve(),
        skip_snapshot_verify=bool(args.skip_snapshot_verify),
        skip_portability=bool(args.skip_portability),
    )
    if ok:
        print("[ok] publication artifact doctor passed")
        return 0

    print("[fail] publication artifact doctor found issues:")
    for failure in failures:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
