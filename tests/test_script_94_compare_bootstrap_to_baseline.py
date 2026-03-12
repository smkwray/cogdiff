from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["cohort"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    script = _repo_root() / "scripts" / "94_compare_bootstrap_to_baseline.py"
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


def test_script_94_prints_cohort_comparison_when_files_exist(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    tables = root / "outputs/tables"
    _write_csv(
        tables / "g_mean_diff.csv",
        [
            {
                "cohort": "nlsy79",
                "d_g": "0.3",
                "SE_d_g": "0.03",
                "ci_low_d_g": "0.24",
                "ci_high_d_g": "0.36",
            }
        ],
    )
    _write_csv(
        tables / "g_mean_diff_family_bootstrap.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "",
                "d_g": "0.28",
                "SE_d_g": "0.19",
                "ci_low_d_g": "-0.10",
                "ci_high_d_g": "0.35",
            }
        ],
    )
    _write_csv(
        tables / "g_variance_ratio.csv",
        [
            {
                "cohort": "nlsy79",
                "VR_g": "1.31",
                "SE_logVR": "0.03",
                "ci_low": "1.16",
                "ci_high": "1.33",
            }
        ],
    )
    _write_csv(
        tables / "g_variance_ratio_family_bootstrap.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "",
                "VR_g": "1.29",
                "SE_logVR": "0.16",
                "ci_low": "0.79",
                "ci_high": "1.31",
            }
        ],
    )

    result = _run_script("--project-root", str(root))
    assert result.returncode == 0
    assert "comparison: g_mean_diff" in result.stdout
    assert "comparison: g_variance_ratio" in result.stdout
    assert "nlsy79" in result.stdout
    assert "-0.02" in result.stdout
    assert "computed" in result.stdout


def test_script_94_fails_with_clear_error_when_bootstrap_file_missing(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    tables = root / "outputs/tables"
    _write_csv(
        tables / "g_mean_diff.csv",
        [{"cohort": "nlsy79", "d_g": "0.3", "SE_d_g": "0.03", "ci_low_d_g": "0.24", "ci_high_d_g": "0.36"}],
    )
    _write_csv(
        tables / "g_variance_ratio.csv",
        [{"cohort": "nlsy79", "VR_g": "1.31", "SE_logVR": "0.03", "ci_low": "1.16", "ci_high": "1.33"}],
    )

    result = _run_script("--project-root", str(root))
    assert result.returncode == 1
    assert "missing bootstrap comparison file(s)" in result.stderr
    assert "g_mean_diff_family_bootstrap.csv" in result.stderr
