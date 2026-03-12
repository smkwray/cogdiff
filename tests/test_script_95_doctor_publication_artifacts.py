from __future__ import annotations

import importlib.util
from pathlib import Path
import zipfile

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "95_doctor_publication_artifacts.py"
    spec = importlib.util.spec_from_file_location("script95_doctor_publication_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _sha256(module: object, path: Path) -> str:
    return module._sha256(path)


def test_run_doctor_passes_for_exact_bundle(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    tables = root / "outputs/tables"
    bundle = tables / "publication_results_lock"
    bundle.mkdir(parents=True, exist_ok=True)

    results_snapshot = tables / "results_snapshot.md"
    results_snapshot.write_text("# snapshot\n", encoding="utf-8")
    manifest = tables / "publication_snapshot_manifest.csv"
    pd.DataFrame([{"path": "outputs/tables/results_snapshot.md", "sha256": _sha256(module, results_snapshot), "size_bytes": results_snapshot.stat().st_size, "snapshot_utc": "2026-03-11T00:00:00Z", "mtime_utc": "2026-03-11T00:00:00Z"}]).to_csv(manifest, index=False)

    copied = bundle / "foo.csv"
    copied.write_text("a,b\n1,2\n", encoding="utf-8")
    methods = bundle / "manuscript_results_lock.md"
    methods.write_text("# lock\n", encoding="utf-8")
    bundle_manifest = bundle / "publication_results_lock_manifest.csv"
    pd.DataFrame(
        [
            {
                "generated_utc": "2026-03-11T00:00:00Z",
                "source_path": "outputs/tables/foo.csv",
                "bundle_path": "outputs/tables/publication_results_lock/foo.csv",
                "sha256": _sha256(module, copied),
                "size_bytes": copied.stat().st_size,
                "mtime_utc": "2026-03-11T00:00:00Z",
            }
        ]
    ).to_csv(bundle_manifest, index=False)

    zip_path = tables / "publication_results_lock.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(copied, arcname="foo.csv")
        archive.write(methods, arcname="manuscript_results_lock.md")
        archive.write(bundle_manifest, arcname="publication_results_lock_manifest.csv")

    ok, failures = module.run_doctor(root=root, skip_snapshot_verify=True, skip_portability=True)
    assert ok
    assert failures == []


def test_run_doctor_fails_on_stale_bundle_file(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    tables = root / "outputs/tables"
    bundle = tables / "publication_results_lock"
    bundle.mkdir(parents=True, exist_ok=True)

    (tables / "results_snapshot.md").write_text("# snapshot\n", encoding="utf-8")
    pd.DataFrame([{"path": "outputs/tables/results_snapshot.md", "sha256": "x", "size_bytes": 1, "snapshot_utc": "2026-03-11T00:00:00Z", "mtime_utc": "2026-03-11T00:00:00Z"}]).to_csv(tables / "publication_snapshot_manifest.csv", index=False)
    copied = bundle / "foo.csv"
    copied.write_text("a,b\n1,2\n", encoding="utf-8")
    (bundle / "stale.csv").write_text("stale\n", encoding="utf-8")
    methods = bundle / "manuscript_results_lock.md"
    methods.write_text("# lock\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "generated_utc": "2026-03-11T00:00:00Z",
                "source_path": "outputs/tables/foo.csv",
                "bundle_path": "outputs/tables/publication_results_lock/foo.csv",
                "sha256": _sha256(module, copied),
                "size_bytes": copied.stat().st_size,
                "mtime_utc": "2026-03-11T00:00:00Z",
            }
        ]
    ).to_csv(bundle / "publication_results_lock_manifest.csv", index=False)
    with zipfile.ZipFile(tables / "publication_results_lock.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(copied, arcname="foo.csv")
        archive.write(methods, arcname="manuscript_results_lock.md")
        archive.write(bundle / "publication_results_lock_manifest.csv", arcname="publication_results_lock_manifest.csv")
        archive.write(bundle / "stale.csv", arcname="stale.csv")

    ok, failures = module.run_doctor(root=root, skip_snapshot_verify=True, skip_portability=True)
    assert not ok
    assert any("stale files" in failure for failure in failures)
