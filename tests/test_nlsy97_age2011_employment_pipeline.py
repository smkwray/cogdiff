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


def test_nlsy97_interview_year_2011_and_employment_survive_construct_and_residualize(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "interim_dir: data/interim\nprocessed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write(
        root / "config/nlsy97.yml",
        """
cohort: nlsy97
sample_construct:
  input_file: panel_extract.csv
  id_col: person_id
  sex_col: sex
  age_col: null
  age_resid_col: birth_year
  subtests: [GS, AR]
  min_tests: 2
  missing_codes: [-1, -2, -3, -4, -5]
  column_map:
    R0000100: person_id
    R0536300: sex
    R0536402: birth_year
    R9705200: GS
    R9705300: AR
    T6680901: interview_month_2011
    T6680902: interview_year_2011
    T7295800: employment_2011
    T7311500: occupation_code_2011_slot01
    T7311600: occupation_code_2011_slot02
expected_age_range:
  min: 13
  max: 17
""",
    )

    input_csv = root / "data/interim/nlsy97/panel_extract.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"R0000100": 1, "R0536300": "F", "R0536402": 1981, "R9705200": 10, "R9705300": 11, "T6680901": 6, "T6680902": 2011, "T7295800": 1, "T7311500": 1020, "T7311600": -4},
            {"R0000100": 2, "R0536300": "M", "R0536402": 1982, "R9705200": 12, "R9705300": 13, "T6680901": 7, "T6680902": 2011, "T7295800": 0, "T7311500": -4, "T7311600": 3600},
            {"R0000100": 3, "R0536300": "F", "R0536402": 1983, "R9705200": 14, "R9705300": 15, "T6680901": 8, "T6680902": 2011, "T7295800": 1, "T7311500": 4210, "T7311600": -4},
        ]
    ).to_csv(input_csv, index=False)

    construct = _repo_root() / "scripts/05_construct_samples.py"
    residualize = _repo_root() / "scripts/06_age_residualize.py"
    subprocess.run([sys.executable, str(construct), "--project-root", str(root), "--cohort", "nlsy97"], cwd=_repo_root(), check=True)
    subprocess.run([sys.executable, str(residualize), "--project-root", str(root), "--cohort", "nlsy97"], cwd=_repo_root(), check=True)

    cfa = pd.read_csv(root / "data/processed/nlsy97_cfa.csv")
    resid = pd.read_csv(root / "data/processed/nlsy97_cfa_resid.csv")
    for frame in (cfa, resid):
        assert "interview_month_2011" in frame.columns
        assert "interview_year_2011" in frame.columns
        assert "age_2011" in frame.columns
        assert "employment_2011" in frame.columns
        assert "occupation_code_2011" in frame.columns
    assert set(pd.to_numeric(cfa["age_2011"], errors="coerce").dropna().astype(int)) == {30, 29, 28}
    assert set(pd.to_numeric(resid["age_2011"], errors="coerce").dropna().astype(int)) == {30, 29, 28}
    assert set(pd.to_numeric(cfa["employment_2011"], errors="coerce").dropna().astype(int)) == {0, 1}
    assert set(pd.to_numeric(resid["employment_2011"], errors="coerce").dropna().astype(int)) == {0, 1}
    assert set(pd.to_numeric(cfa["occupation_code_2011"], errors="coerce").dropna().astype(int)) == {1020, 3600, 4210}
    assert set(pd.to_numeric(resid["occupation_code_2011"], errors="coerce").dropna().astype(int)) == {1020, 3600, 4210}
