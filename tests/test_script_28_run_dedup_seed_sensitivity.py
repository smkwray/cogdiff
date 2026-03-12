from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _run_script(root: Path, *argv: str) -> None:
    script = _repo_root() / "scripts" / "28_run_dedup_seed_sensitivity.py"
    subprocess.run([sys.executable, str(script), "--project-root", str(root), *map(str, argv)], check=True)


def test_script_28_writes_seed_runs_and_summary_for_present_inputs(tmp_path: Path) -> None:
    root = tmp_path.resolve()

    _write_csv(
        root / "outputs/tables/g_mean_diff_one_pair_per_family_seed_11.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "d_g": 0.10},
                {"cohort": "nlsy97", "d_g": -0.04},
                {"cohort": "cnlsy", "d_g": 0.06},
            ]
        ),
    )
    _write_csv(
        root / "outputs/tables/g_variance_ratio_one_pair_per_family_seed_11.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "VR_g": 1.20},
                {"cohort": "nlsy97", "VR_g": 1.10},
                {"cohort": "cnlsy", "VR_g": 1.05},
            ]
        ),
    )
    _write_csv(
        root / "outputs/tables/g_mean_diff_one_pair_per_family_seed_22.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "d_g": 0.14},
                {"cohort": "nlsy97", "d_g": -0.02},
                {"cohort": "cnlsy", "d_g": 0.03},
            ]
        ),
    )
    _write_csv(
        root / "outputs/tables/g_variance_ratio_one_pair_per_family_seed_22.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "VR_g": 1.30},
                {"cohort": "nlsy97", "VR_g": 1.15},
                {"cohort": "cnlsy", "VR_g": 1.10},
            ]
        ),
    )

    _run_script(root, "--seed", "11", "--seed", "22")

    run_path = root / "outputs/tables/dedup_seed_sensitivity_runs.csv"
    summary_path = root / "outputs/tables/dedup_seed_sensitivity_summary.csv"
    assert run_path.exists()
    assert summary_path.exists()

    runs = pd.read_csv(run_path)
    assert {"seed", "cohort", "d_g", "log_vr_g", "missing_reason"}.issubset(set(runs.columns))
    assert len(runs) == 6

    nlsy79_11 = runs[(runs["seed"] == 11) & (runs["cohort"] == "nlsy79")].iloc[0]
    nlsy79_22 = runs[(runs["seed"] == 22) & (runs["cohort"] == "nlsy79")].iloc[0]
    assert float(nlsy79_11["d_g"]) == 0.10
    assert float(nlsy79_22["d_g"]) == 0.14
    assert math.isclose(
        float(nlsy79_11["log_vr_g"]),
        math.log(1.20),
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        float(nlsy79_22["log_vr_g"]),
        math.log(1.30),
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert (runs["missing_reason"].fillna("") == "").all()

    summary = pd.read_csv(summary_path)
    assert set(summary["cohort"]) == {"nlsy79", "nlsy97", "cnlsy"}
    nlsy79_summary = summary[summary["cohort"] == "nlsy79"].iloc[0]
    assert int(nlsy79_summary["n_runs"]) == 2
    assert float(nlsy79_summary["d_g_sd"]) > 0.0
    assert float(nlsy79_summary["log_vr_g_sd"]) > 0.0


def test_script_28_handles_missing_seed_inputs_gracefully(tmp_path: Path) -> None:
    root = tmp_path.resolve()

    _write_csv(
        root / "outputs/tables/g_mean_diff_one_pair_per_family_seed_11.csv",
        pd.DataFrame([{"cohort": "nlsy79", "d_g": 0.10}]),
    )
    _write_csv(
        root / "outputs/tables/g_variance_ratio_one_pair_per_family_seed_11.csv",
        pd.DataFrame([{"cohort": "nlsy79", "VR_g": 1.20}]),
    )
    _write_csv(
        root / "outputs/tables/g_variance_ratio_one_pair_per_family_seed_22.csv",
        pd.DataFrame([{"cohort": "nlsy97", "VR_g": 1.10}]),
    )

    _run_script(
        root,
        "--seed",
        "11",
        "--seed",
        "22",
        "--cohort",
        "nlsy79",
        "--cohort",
        "nlsy97",
    )

    runs = pd.read_csv(root / "outputs/tables/dedup_seed_sensitivity_runs.csv")
    assert len(runs) == 4
    row = runs[(runs["seed"] == 22) & (runs["cohort"] == "nlsy79")].iloc[0]
    assert pd.isna(row["d_g"])
    assert pd.isna(row["log_vr_g"])
    assert "missing_d_g_file" in row["missing_reason"]
    row_nlsy97_seed22 = runs[(runs["seed"] == 22) & (runs["cohort"] == "nlsy97")].iloc[0]
    assert pd.isna(row_nlsy97_seed22["d_g"])
    assert float(row_nlsy97_seed22["log_vr_g"]) == pytest.approx(math.log(1.10), rel=0, abs=1e-12)

    summary = pd.read_csv(root / "outputs/tables/dedup_seed_sensitivity_summary.csv")
    assert int(summary[summary["cohort"] == "nlsy79"]["n_runs"].iloc[0]) == 2


def test_script_28_writes_empty_outputs_without_inputs(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _run_script(root, "--seed", "7", "--cohort", "nlsy79")

    runs = pd.read_csv(root / "outputs/tables/dedup_seed_sensitivity_runs.csv")
    summary = pd.read_csv(root / "outputs/tables/dedup_seed_sensitivity_summary.csv")

    assert len(runs) == 1
    assert pd.isna(runs.iloc[0]["d_g"])
    assert pd.isna(runs.iloc[0]["log_vr_g"])
    assert runs.iloc[0]["cohort"] == "nlsy79"
    assert int(summary["n_runs"].iloc[0]) == 1
    assert pd.isna(summary["d_g_sd"].iloc[0])
    assert pd.isna(summary["log_vr_g_sd"].iloc[0])
