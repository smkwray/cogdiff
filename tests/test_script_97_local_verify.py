from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_script_97_self_test_prints_commands() -> None:
    script = _repo_root() / "scripts" / "97_local_verify.sh"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["SEXG_VENV"] = "/path/to/venv"
    result = subprocess.run(
        ["bash", str(script), "--fast", "--self-test"],
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "scripts/96_clean_repo_caches.sh" in result.stdout
    assert "tests/test_script_99_portability_smoke_check.py" in result.stdout
    assert "scripts/99_portability_smoke_check.py" in result.stdout


def test_script_97_self_test_includes_artifact_doctor_when_requested() -> None:
    script = _repo_root() / "scripts" / "97_local_verify.sh"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["SEXG_VENV"] = "/path/to/venv"
    result = subprocess.run(
        ["bash", str(script), "--fast", "--artifacts", "--self-test"],
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "artifacts=1" in result.stdout
    assert "scripts/95_doctor_publication_artifacts.py" in result.stdout
