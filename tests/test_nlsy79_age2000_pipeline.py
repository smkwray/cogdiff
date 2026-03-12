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


def test_nlsy79_interview_year_2000_survives_construct_and_residualize(tmp_path: Path) -> None:
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
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  input_file: panel_extract.csv
  id_col: person_id
  sex_col: sex
  age_col: age
  age_resid_col: birth_year
  subtests: [GS, AR]
  min_tests: 2
  missing_codes: [-1, -2, -3, -4, -5]
  column_map:
    R0000100: person_id
    R0214800: sex
    R0216500: age
    R0000500: birth_year
    R0616000: GS
    R0616200: AR
    R7007501: employment_2000
    E6320100: occupation_code_2000_slot01
    E6320200: occupation_code_2000_slot02
    R6963301: interview_month_2000
    R6963302: interview_year_2000
expected_age_range:
  min: 16
  max: 23
""",
    )

    input_csv = root / "data/interim/nlsy79/panel_extract.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"R0000100": 1, "R0214800": "F", "R0216500": 18, "R0000500": 80, "R0616000": 10, "R0616200": 11, "R7007501": 1, "E6320100": 245, "E6320200": -4, "R6963301": 6, "R6963302": 2000},
            {"R0000100": 2, "R0214800": "M", "R0216500": 19, "R0000500": 81, "R0616000": 12, "R0616200": 13, "R7007501": 0, "E6320100": -4, "E6320200": 503, "R6963301": 7, "R6963302": 2000},
            {"R0000100": 3, "R0214800": "F", "R0216500": 20, "R0000500": 82, "R0616000": 14, "R0616200": 15, "R7007501": 1, "E6320100": 95, "E6320200": -4, "R6963301": 8, "R6963302": 2000},
        ]
    ).to_csv(input_csv, index=False)

    construct = _repo_root() / "scripts/05_construct_samples.py"
    residualize = _repo_root() / "scripts/06_age_residualize.py"
    subprocess.run([sys.executable, str(construct), "--project-root", str(root), "--cohort", "nlsy79"], cwd=_repo_root(), check=True)
    subprocess.run([sys.executable, str(residualize), "--project-root", str(root), "--cohort", "nlsy79"], cwd=_repo_root(), check=True)

    cfa = pd.read_csv(root / "data/processed/nlsy79_cfa.csv")
    resid = pd.read_csv(root / "data/processed/nlsy79_cfa_resid.csv")
    for frame in (cfa, resid):
        assert "interview_month_2000" in frame.columns
        assert "interview_year_2000" in frame.columns
        assert "age_2000" in frame.columns
        assert "employment_2000" in frame.columns
        assert "occupation_code_2000" in frame.columns
    assert set(pd.to_numeric(cfa["age_2000"], errors="coerce").dropna().astype(int)) == {20, 19, 18}
    assert set(pd.to_numeric(resid["age_2000"], errors="coerce").dropna().astype(int)) == {20, 19, 18}
    assert set(pd.to_numeric(cfa["employment_2000"], errors="coerce").dropna().astype(int)) == {0, 1}
    assert set(pd.to_numeric(resid["employment_2000"], errors="coerce").dropna().astype(int)) == {0, 1}
    assert set(pd.to_numeric(cfa["occupation_code_2000"], errors="coerce").dropna().astype(int)) == {95, 245, 503}
    assert set(pd.to_numeric(resid["occupation_code_2000"], errors="coerce").dropna().astype(int)) == {95, 245, 503}
