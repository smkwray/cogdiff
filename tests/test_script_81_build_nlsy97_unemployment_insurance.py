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


def test_script_81_builds_ui_table(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)
    rows: list[dict[str, float]] = []
    for i in range(160):
        g = (i - 80.0) / 20.0
        spells_2019 = 0.0 if i % 4 == 0 else float(1 + (i % 3))
        spells_2021 = 0.0 if i % 5 == 0 else float(1 + (i % 2))
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
                "age_2019": 38.0,
                "age_2021": 40.0,
                "ui_spells_2019": spells_2019,
                "ui_spells_2021": spells_2021,
                "ui_amount_2019": spells_2019 * (900.0 + 10.0 * i),
                "ui_amount_2021": spells_2021 * (1100.0 + 8.0 * i),
            }
        )
    pd.DataFrame(rows).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    script = _repo_root() / "scripts" / "81_build_nlsy97_unemployment_insurance.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--min-class-n", "10"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/nlsy97_unemployment_insurance.csv")
    assert set(out["model"]) == {"any_ui_receipt", "log1p_ui_spells", "log1p_ui_amount"}
    computed = out.loc[out["status"] == "computed"].copy()
    assert len(computed) == 6
    assert computed["n_used"].ge(100).all()
    assert computed["beta_g"].notna().all()


def test_script_81_handles_missing_source(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_minimal_project(root)

    script = _repo_root() / "scripts" / "81_build_nlsy97_unemployment_insurance.py"
    result = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    out = pd.read_csv(root / "outputs/tables/nlsy97_unemployment_insurance.csv")
    assert set(out["status"]) == {"not_feasible"}
    assert set(out["reason"]) == {"missing_source_data"}
