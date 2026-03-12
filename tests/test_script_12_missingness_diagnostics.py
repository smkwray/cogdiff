from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_script_12_writes_missingness_outputs(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        project_root / "config/nlsy79.yml",
        """
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  subtests: [GS, AR]
  missing_codes: [-1, -2]
""",
    )
    (project_root / "data/processed").mkdir(parents=True, exist_ok=True)
    (project_root / "data/interim/nlsy79").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"person_id": 1, "sex": "F", "age": 20, "GS": 1.0, "AR": 2.0},
            {"person_id": 2, "sex": "M", "age": 21, "GS": 2.0, "AR": 2.0},
            {"person_id": 3, "sex": "F", "age": 20, "GS": 1.0, "AR": 3.0},
        ]
    ).to_csv(project_root / "data/interim/nlsy79/panel_extract.csv", index=False)

    pd.DataFrame(
        [
            {"person_id": 1, "sex": "F", "age": 20, "GS": 1.0, "AR": -1},
            {"person_id": 2, "sex": "M", "age": 21, "GS": None, "AR": 2.0},
            {"person_id": 3, "sex": "female", "age": 19, "GS": 3.0, "AR": 4.0},
            {"person_id": 4, "sex": "male", "age": 20, "GS": -2, "AR": None},
        ]
    ).to_csv(project_root / "data/processed/nlsy79_cfa.csv", index=False)

    script = _repo_root() / "scripts" / "12_missingness_diagnostics.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy79"],
        cwd=_repo_root(),
        check=True,
    )

    output = pd.read_csv(project_root / "outputs/tables/missingness_diagnostics.csv")
    selection = pd.read_csv(project_root / "outputs/tables/inclusion_exclusion_diagnostics.csv")
    assert set(["cohort", "sex_group", "subtest", "n_total", "n_missing", "missing_rate"]).issubset(output.columns)
    gs_all = output[(output["cohort"] == "nlsy79") & (output["sex_group"] == "all") & (output["subtest"] == "GS")]
    assert len(gs_all) == 1
    assert int(gs_all.iloc[0]["n_total"]) == 4
    assert int(gs_all.iloc[0]["n_missing"]) == 2
    ar_female = output[
        (output["cohort"] == "nlsy79") & (output["sex_group"] == "female") & (output["subtest"] == "AR")
    ]
    assert len(ar_female) == 1
    assert int(ar_female.iloc[0]["n_missing"]) == 1
    all_selection = selection[(selection["cohort"] == "nlsy79") & (selection["sex_group"] == "all")]
    assert len(all_selection) == 1
    assert int(all_selection.iloc[0]["n_panel"]) == 3
    assert int(all_selection.iloc[0]["n_included"]) == 3
    assert int(all_selection.iloc[0]["n_excluded"]) == 0
    assert (project_root / "outputs/figures/missingness_heatmap.png").exists()


def test_script_12_handles_missing_inputs(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        project_root / "config/nlsy97.yml",
        """
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  subtests: [GS, AR]
  missing_codes: [-1, -2]
""",
    )

    script = _repo_root() / "scripts" / "12_missingness_diagnostics.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy97"],
        cwd=_repo_root(),
        check=True,
    )

    output = pd.read_csv(project_root / "outputs/tables/missingness_diagnostics.csv")
    selection = pd.read_csv(project_root / "outputs/tables/inclusion_exclusion_diagnostics.csv")
    assert output.empty
    assert selection.empty
    assert (project_root / "outputs/figures/missingness_heatmap.png").exists()


def test_script_12_multi_cohort_diagnostics(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        project_root / "config/nlsy79.yml",
        """
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  subtests: [GS, AR]
  missing_codes: [-1, -2]
""",
    )
    _write(
        project_root / "config/nlsy97.yml",
        """
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  subtests: [GS, AR]
  missing_codes: [-1, -2]
""",
    )
    (project_root / "data/processed").mkdir(parents=True, exist_ok=True)
    (project_root / "data/interim/nlsy79").mkdir(parents=True, exist_ok=True)
    (project_root / "data/interim/nlsy97").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"person_id": 1, "sex": "F", "age": 20, "GS": 1.0, "AR": 2.0},
            {"person_id": 2, "sex": "M", "age": 21, "GS": None, "AR": 2.0},
            {"person_id": 3, "sex": "F", "age": 19, "GS": 3.0, "AR": -2},
        ]
    ).to_csv(project_root / "data/processed/nlsy79_cfa.csv", index=False)
    pd.DataFrame(
        [
            {"person_id": 10, "sex": "F", "age": 21, "GS": 2.0, "AR": 3.0},
            {"person_id": 11, "sex": "M", "age": 22, "GS": -1, "AR": 1.0},
            {"person_id": 12, "sex": "M", "age": 23, "GS": 2.0, "AR": 2.0},
        ]
    ).to_csv(project_root / "data/processed/nlsy97_cfa.csv", index=False)

    pd.DataFrame(
        [
            {"person_id": 1, "sex": "F", "age": 20},
            {"person_id": 2, "sex": "M", "age": 21},
            {"person_id": 3, "sex": "F", "age": 19},
        ]
    ).to_csv(project_root / "data/interim/nlsy79/panel_extract.csv", index=False)
    pd.DataFrame(
        [
            {"person_id": 10, "sex": "F", "age": 21},
            {"person_id": 11, "sex": "M", "age": 22},
            {"person_id": 12, "sex": "M", "age": 23},
        ]
    ).to_csv(project_root / "data/interim/nlsy97/panel_extract.csv", index=False)

    script = _repo_root() / "scripts" / "12_missingness_diagnostics.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(project_root),
            "--cohort",
            "nlsy79",
            "--cohort",
            "nlsy97",
        ],
        cwd=_repo_root(),
        check=True,
    )

    missingness = pd.read_csv(project_root / "outputs/tables/missingness_diagnostics.csv")
    selection = pd.read_csv(project_root / "outputs/tables/inclusion_exclusion_diagnostics.csv")
    assert set(missingness["cohort"]) == {"nlsy79", "nlsy97"}

    nlsy79_all_gs = missingness[
        (missingness["cohort"] == "nlsy79") & (missingness["sex_group"] == "all") & (missingness["subtest"] == "GS")
    ]
    assert int(nlsy79_all_gs.iloc[0]["n_missing"]) == 1
    assert int(nlsy79_all_gs.iloc[0]["n_total"]) == 3

    nlsy97_all_ar = missingness[
        (missingness["cohort"] == "nlsy97") & (missingness["sex_group"] == "all") & (missingness["subtest"] == "AR")
    ]
    assert int(nlsy97_all_ar.iloc[0]["n_missing"]) == 0

    nlsy79_selection_all = selection[(selection["cohort"] == "nlsy79") & (selection["sex_group"] == "all")]
    assert int(nlsy79_selection_all.iloc[0]["n_panel"]) == 3
    assert int(nlsy79_selection_all.iloc[0]["n_included"]) == 3

    nlsy97_selection_all = selection[(selection["cohort"] == "nlsy97") & (selection["sex_group"] == "all")]
    assert int(nlsy97_selection_all.iloc[0]["n_panel"]) == 3
    assert int(nlsy97_selection_all.iloc[0]["n_included"]) == 3
    assert (project_root / "outputs/figures/missingness_heatmap.png").exists()
