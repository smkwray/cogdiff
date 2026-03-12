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


def _row(i: int) -> dict[str, float]:
    g = (i - 50.0) / 18.0
    if i % 3 == 0:
        occ_2013, occ_2021 = 4700.0, 1000.0
    elif i % 3 == 1:
        occ_2013, occ_2021 = 1000.0, 5000.0
    else:
        occ_2013, occ_2021 = 5000.0, 5000.0
    return {
        "person_id": i + 1,
        "GS": g + 1.0,
        "AR": g + 2.0,
        "WK": g + 3.0,
        "PC": g + 4.0,
        "NO": g + 5.0,
        "CS": g + 6.0,
        "AS": g + 7.0,
        "MK": g + 8.0,
        "MC": g + 9.0,
        "EI": g + 10.0,
        "occupation_code_2011": float("nan"),
        "occupation_code_2013": occ_2013,
        "occupation_code_2015": float("nan"),
        "occupation_code_2017": float("nan"),
        "occupation_code_2019": float("nan"),
        "occupation_code_2021": occ_2021,
        "age_2011": float("nan"),
        "age_2013": 27.0 + (i % 4),
        "age_2015": float("nan"),
        "age_2017": float("nan"),
        "age_2019": float("nan"),
        "age_2021": 35.0 + (i % 4),
    }


def test_script_80_builds_mobility_outputs(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)
    pd.DataFrame([_row(i) for i in range(90)]).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    script = _repo_root() / "scripts" / "80_build_nlsy97_occupational_mobility.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--min-class-n", "5"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    summary = pd.read_csv(root / "outputs/tables/nlsy97_occupational_mobility_summary.csv")
    models = pd.read_csv(root / "outputs/tables/nlsy97_occupational_mobility_models.csv")
    assert summary.loc[0, "status"] == "computed"
    assert int(summary.loc[0, "n_with_2plus_occupation_waves"]) == 90
    assert int(summary.loc[0, "n_changed_major_group"]) > 0
    assert "any_major_group_change" in set(models["model"])
    assert "computed" in set(models["status"])


def test_script_80_handles_missing_source(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)

    script = _repo_root() / "scripts" / "80_build_nlsy97_occupational_mobility.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    summary = pd.read_csv(root / "outputs/tables/nlsy97_occupational_mobility_summary.csv")
    models = pd.read_csv(root / "outputs/tables/nlsy97_occupational_mobility_models.csv")
    assert set(summary["status"]) == {"not_feasible"}
    assert set(models["status"]) == {"not_feasible"}
