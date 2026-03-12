from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_script(root: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    script = _repo_root() / "scripts" / "99_portability_smoke_check.py"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), *extra_args],
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_script_99_passes_when_no_forbidden_matches(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "outputs/tables/ok.csv", "cohort,value\nnlsy79,1\n")
    _write(root / "data/interim/links/link_exports.csv", "cohort,path\nnlsy79,data/interim/links/links79_pair_expanded.csv\n")
    result = _run_script(root)
    assert result.returncode == 0
    assert "[ok]" in result.stdout


def test_script_99_fails_when_forbidden_match_found(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "outputs/tables/bad.csv", "path\n/Users/example/project/file.csv\n")
    result = _run_script(root)
    assert result.returncode == 1
    assert "[fail]" in result.stdout
    assert "bad.csv" in result.stdout


def test_script_99_excludes_outputs_logs_by_default(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "outputs/logs/pipeline/run.log", "ran with /Users/example/project\n")
    result = _run_script(root, "--scan-path", ".")
    assert result.returncode == 0
    assert "[ok]" in result.stdout
