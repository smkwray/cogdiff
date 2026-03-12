from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_script_26_generates_discrepancy_matrix_with_weights(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    script = _repo_root() / "scripts" / "26_build_discrepancy_attribution.py"

    _write_csv(
        root / "outputs/tables/g_mean_diff.csv",
        pd.DataFrame([{"cohort": "nlsy79", "d_g": 0.12}]),
    )
    _write_csv(
        root / "outputs/tables/g_mean_diff_weighted.csv",
        pd.DataFrame([{"cohort": "nlsy79", "d_g": 0.08}]),
    )
    _write_csv(
        root / "outputs/tables/g_variance_ratio.csv",
        pd.DataFrame([{"cohort": "nlsy79", "VR_g": 1.30}]),
    )
    _write_csv(
        root / "outputs/tables/g_variance_ratio_weighted.csv",
        pd.DataFrame([{"cohort": "nlsy79", "VR_g": 1.05}]),
    )

    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        check=True,
    )

    output = root / "outputs/tables/discrepancy_attribution_matrix.csv"
    assert output.exists()
    matrix = pd.read_csv(output)
    assert set(matrix.columns) >= {
        "cohort",
        "claim_id",
        "metric",
        "baseline_estimate",
        "comparison_estimate",
        "delta",
        "likely_cause_bucket",
        "diagnostic_required",
        "verdict_hint",
    }
    assert set(matrix["cohort"]) == {"nlsy79"}
    assert set(matrix["metric"]) == {"d_g", "log_vr_g"}

    d_g_row = matrix[matrix["metric"] == "d_g"].iloc[0]
    assert float(d_g_row["baseline_estimate"]) == 0.12
    assert float(d_g_row["comparison_estimate"]) == 0.08
    assert math.isclose(float(d_g_row["delta"]), -0.04, rel_tol=0, abs_tol=1e-12)


def test_script_26_generates_partial_rows_when_comparison_input_is_missing(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    script = _repo_root() / "scripts" / "26_build_discrepancy_attribution.py"

    _write_csv(
        root / "outputs/tables/g_mean_diff.csv",
        pd.DataFrame([{"cohort": "nlsy97", "d_g": -0.04}]),
    )
    _write_csv(root / "outputs/tables/g_variance_ratio.csv", pd.DataFrame([{"cohort": "nlsy97", "VR_g": 1.20}]))

    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        check=True,
    )

    output = root / "outputs/tables/discrepancy_attribution_matrix.csv"
    matrix = pd.read_csv(output)
    assert set(matrix["cohort"]) == {"nlsy97"}
    assert set(matrix["metric"]) == {"d_g", "log_vr_g"}
    for metric in {"d_g", "log_vr_g"}:
        row = matrix[matrix["metric"] == metric].iloc[0]
        assert pd.isna(row["comparison_estimate"])
        assert pd.isna(row["delta"])


def test_script_26_is_robust_to_missing_inputs(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    script = _repo_root() / "scripts" / "26_build_discrepancy_attribution.py"

    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        check=True,
    )

    output = root / "outputs/tables/discrepancy_attribution_matrix.csv"
    assert output.exists()
    matrix = pd.read_csv(output)
    assert matrix.empty
    assert list(matrix.columns) == [
        "cohort",
        "claim_id",
        "metric",
        "baseline_estimate",
        "comparison_estimate",
        "delta",
        "likely_cause_bucket",
        "diagnostic_required",
        "verdict_hint",
    ]
