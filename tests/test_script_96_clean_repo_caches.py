from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_script_96_supports_dry_run(tmp_path: Path) -> None:
    script = _repo_root() / "scripts" / "96_clean_repo_caches.sh"
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"\0\0")
    (tmp_path / ".pytest_cache").mkdir()

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        ["bash", str(script), "--path", str(tmp_path), "--dry-run"],
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "__pycache__" in result.stdout
    assert ".pytest_cache" in result.stdout


def test_script_96_deletes_cache_dirs(tmp_path: Path) -> None:
    script = _repo_root() / "scripts" / "96_clean_repo_caches.sh"
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"\0\0")
    (tmp_path / ".pytest_cache").mkdir()

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        ["bash", str(script), "--path", str(tmp_path)],
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert not (tmp_path / "__pycache__").exists()
    assert not (tmp_path / ".pytest_cache").exists()

