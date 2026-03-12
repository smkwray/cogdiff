from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_script_25_update_then_verify_passes(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "outputs/tables/analysis_tiers.csv", "cohort,estimand\nnlsy79,d_g\n")
    _write(root / "outputs/tables/specification_stability_summary.csv", "cohort,estimand\nnlsy79,d_g\n")
    _write(root / "outputs/tables/confirmatory_exclusions.csv", "cohort\nnlsy79\n")

    script = _repo_root() / "scripts" / "25_verify_publication_snapshot.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--update-manifest"],
        cwd=_repo_root(),
        check=True,
    )
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        check=True,
    )

    manifest = root / "outputs/tables/publication_snapshot_manifest.csv"
    assert manifest.exists()


def test_script_25_fails_on_hash_mismatch_and_suggests_update(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "outputs/tables/analysis_tiers.csv", "cohort,estimand\nnlsy79,d_g\n")
    _write(root / "outputs/tables/specification_stability_summary.csv", "cohort,estimand\nnlsy79,d_g\n")
    _write(root / "outputs/tables/confirmatory_exclusions.csv", "cohort\nnlsy79\n")

    script = _repo_root() / "scripts" / "25_verify_publication_snapshot.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--update-manifest"],
        cwd=_repo_root(),
        check=True,
    )

    _write(root / "outputs/tables/analysis_tiers.csv", "cohort,estimand\nnlsy79,vr_g\n")

    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "--update-manifest" in result.stderr

