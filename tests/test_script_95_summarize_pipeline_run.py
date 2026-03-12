from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "script",
        "cohort_scope",
        "status",
        "returncode",
        "command",
        "log_file",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    script = _repo_root() / "scripts" / "95_summarize_pipeline_run.py"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_script_95_uses_latest_summary_by_default(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    log_dir = root / "outputs/logs/pipeline"
    old_summary = log_dir / "20260101_010101_pipeline_run_summary.csv"
    new_summary = log_dir / "20260101_020202_pipeline_run_summary.csv"
    _write_summary(
        old_summary,
        [
            {
                "stage": "7",
                "script": "07_fit_sem_models.py",
                "cohort_scope": "nlsy79",
                "status": "ok",
                "returncode": "0",
                "command": "python scripts/07_fit_sem_models.py",
                "log_file": "outputs/logs/pipeline/stage07_old.log",
                "note": "",
            }
        ],
    )
    _write_summary(
        new_summary,
        [
            {
                "stage": "8",
                "script": "08_invariance_and_partial.py",
                "cohort_scope": "nlsy79",
                "status": "failed",
                "returncode": "1",
                "command": "python scripts/08_invariance_and_partial.py",
                "log_file": "outputs/logs/pipeline/stage08_new.log",
                "note": "see log",
            }
        ],
    )

    result = _run_script("--project-root", str(root))
    assert result.returncode == 0
    assert f"run_summary: {new_summary}" in result.stdout
    assert "status_counts: failed=1" in result.stdout
    assert "stage_status_table:" in result.stdout
    assert "08_invariance_and_partial.py" in result.stdout
    assert str((new_summary.parent / "outputs/logs/pipeline/stage08_new.log").resolve()) in result.stdout
    assert "stage20_artifacts:" not in result.stdout


def test_script_95_path_override_selects_requested_summary(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    log_dir = root / "outputs/logs/pipeline"
    selected = log_dir / "20260102_111111_pipeline_run_summary.csv"
    _write_summary(
        selected,
        [
            {
                "stage": "5",
                "script": "05_construct_samples.py",
                "cohort_scope": "nlsy79,nlsy97,cnlsy",
                "status": "ok",
                "returncode": "0",
                "command": "python scripts/05_construct_samples.py",
                "log_file": "/tmp/stage05.log",
                "note": "",
            }
        ],
    )

    result = _run_script("--path", str(selected))
    assert result.returncode == 0
    assert f"run_summary: {selected}" in result.stdout
    assert "status_counts: ok=1" in result.stdout
    assert "05_construct_samples.py" in result.stdout
    assert "  - /tmp/stage05.log" in result.stdout


def test_script_95_show_stage20_outputs_reports_presence_and_mtime(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    summary = root / "outputs/logs/pipeline/20260103_111111_pipeline_run_summary.csv"
    _write_summary(
        summary,
        [
            {
                "stage": "20",
                "script": "20_run_inference_bootstrap.py",
                "cohort_scope": "nlsy79,nlsy97",
                "status": "ok",
                "returncode": "0",
                "command": "python scripts/20_run_inference_bootstrap.py",
                "log_file": "outputs/logs/pipeline/stage20.log",
                "note": "",
            }
        ],
    )
    present = root / "outputs/tables/g_mean_diff_family_bootstrap.csv"
    present.parent.mkdir(parents=True, exist_ok=True)
    present.write_text("cohort,d_g\nnlsy79,0.3\n", encoding="utf-8")

    result = _run_script(
        "--project-root",
        str(root),
        "--show-stage20-outputs",
    )
    assert result.returncode == 0
    assert "stage20_artifacts:" in result.stdout
    assert "outputs/tables/g_mean_diff_family_bootstrap.csv: present mtime_utc=" in result.stdout
    assert "outputs/tables/g_variance_ratio_family_bootstrap.csv: missing" in result.stdout
