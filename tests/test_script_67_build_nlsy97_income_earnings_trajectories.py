from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_minimal_project(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "config/paths.yml").write_text("processed_dir: data/processed\n", encoding="utf-8")
    (root / "config/models.yml").write_text(
        "\n".join(
            [
                "hierarchical_factors:",
                "  speed: [GS, CS, MC]",
                "  math: [AR, MK]",
                "  verbal: [WK, PC, 'NO']",
                "  technical: [AS, EI]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_script_67_builds_trajectory_table(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)

    rows: list[dict[str, float]] = []
    for i in range(40):
        g = float(i)
        rows.append(
            {
                "person_id": i + 1,
                "GS": g + 1,
                "AR": g + 2,
                "WK": g + 3,
                "PC": g + 4,
                "NO": g + 5,
                "CS": g + 6,
                "AS": g + 7,
                "MK": g + 8,
                "MC": g + 9,
                "EI": g + 10,
                "household_income_2019": 40000.0 + (g * 1000.0),
                "household_income_2021": 47000.0 + (g * 1700.0),
                "annual_earnings_2019": 25000.0 + (g * 700.0),
                "annual_earnings_2021": 31000.0 + (g * 1200.0),
                "age_2019": 35.0 + (i % 3),
                "age_2021": 37.0 + (i % 3),
            }
        )
    pd.DataFrame(rows).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    script = _repo_root() / "scripts" / "67_build_nlsy97_income_earnings_trajectories.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--min-n", "10"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/nlsy97_income_earnings_trajectories.csv")
    assert set(out["outcome"]) == {"annual_earnings", "household_income"}
    assert set(out["model"]) == {"annualized_log_change", "followup_conditional_on_baseline"}
    computed = out.loc[out["status"] == "computed"].copy()
    assert len(computed) == 4
    assert (computed["n_used"] == 40).all()
    change_rows = computed.loc[computed["model"] == "annualized_log_change"].copy()
    assert (change_rows["beta_g"] > 0).all()
    conditional_rows = computed.loc[computed["model"] == "followup_conditional_on_baseline"].copy()
    assert conditional_rows["beta_g"].notna().all()


def test_script_67_marks_insufficient_rows(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)
    rows: list[dict[str, float]] = []
    for i in range(5):
        g = float(i + 1)
        rows.append(
            {
                "person_id": i + 1,
                "GS": g + 1,
                "AR": g + 2,
                "WK": g + 3,
                "PC": g + 4,
                "NO": g + 5,
                "CS": g + 6,
                "AS": g + 7,
                "MK": g + 8,
                "MC": g + 9,
                "EI": g + 10,
                "household_income_2019": 100.0 + (g * 5.0),
                "household_income_2021": 110.0 + (g * 7.0),
                "annual_earnings_2019": 50.0 + (g * 4.0),
                "annual_earnings_2021": 55.0 + (g * 6.0),
                "age_2019": 35.0,
                "age_2021": 37.0,
            }
        )
    pd.DataFrame(rows).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    script = _repo_root() / "scripts" / "67_build_nlsy97_income_earnings_trajectories.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--min-n", "10"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/nlsy97_income_earnings_trajectories.csv")
    assert (out["status"] == "not_feasible").all()
    assert set(out["reason"]) == {"insufficient_two_wave_rows"}
