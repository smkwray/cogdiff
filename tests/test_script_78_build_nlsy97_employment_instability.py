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


def test_script_78_builds_instability_table(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)

    rows: list[dict[str, float]] = []
    for i in range(120):
        g = (i - 60.0) / 18.0
        e2011 = 1.0 if g > -0.8 else 0.0
        e2019 = 1.0 if g > 0.2 else 0.0
        if i % 4 == 0:
            e2021 = e2019
        elif i % 4 == 1:
            e2021 = 0.0
        elif i % 4 == 2:
            e2021 = 1.0
        else:
            e2021 = 0.0 if e2019 == 1.0 else 1.0
        rows.append(
            {
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
                "employment_2011": e2011,
                "employment_2019": e2019,
                "employment_2021": e2021,
                "age_2019": 35.0 + (i % 4),
            }
        )
    pd.DataFrame(rows).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    script = _repo_root() / "scripts" / "78_build_nlsy97_employment_instability.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--min-class-n", "10"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/nlsy97_employment_instability.csv")
    assert set(out["model"]) == {
        "any_transition_2011_2021",
        "mixed_attachment_2011_2021",
        "double_transition_2011_2021",
    }
    computed = out.loc[out["status"] == "computed"].copy()
    assert len(computed) == 3
    assert (computed["n_used"] == 120).all()
    assert computed["odds_ratio_g"].notna().all()


def test_script_78_marks_insufficient_rows(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)
    rows: list[dict[str, float]] = []
    for i in range(8):
        g = float(i + 1)
        rows.append(
            {
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
                "employment_2011": 1.0,
                "employment_2019": 1.0,
                "employment_2021": 1.0,
                "age_2019": 35.0,
            }
        )
    pd.DataFrame(rows).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    script = _repo_root() / "scripts" / "78_build_nlsy97_employment_instability.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--min-class-n", "10"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/nlsy97_employment_instability.csv")
    assert (out["status"] == "not_feasible").all()
    assert set(out["reason"]) == {"employment_2011_not_binary_zero_one"}
