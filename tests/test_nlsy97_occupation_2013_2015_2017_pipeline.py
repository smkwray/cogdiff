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


def test_nlsy97_occupation_2013_2015_2017_and_ages_survive_construct_and_residualize(tmp_path: Path) -> None:
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
    T8154001: interview_month_2013
    T8154002: interview_year_2013
    T8821300: occupation_code_2013_slot01
    T8821400: occupation_code_2013_slot02
    U0036301: interview_month_2015
    U0036302: interview_year_2015
    U0741900: occupation_code_2015_slot01
    U0742000: occupation_code_2015_slot02
    U1876601: interview_month_2017
    U1876602: interview_year_2017
    U2679300: occupation_code_2017_slot01
    U2679400: occupation_code_2017_slot02
expected_age_range:
  min: 13
  max: 17
""",
    )

    input_csv = root / "data/interim/nlsy97/panel_extract.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "R0000100": 1,
                "R0536300": "F",
                "R0536402": 1981,
                "R9705200": 10,
                "R9705300": 11,
                "T8154001": 6,
                "T8154002": 2013,
                "T8821300": 1020,
                "T8821400": -4,
                "U0036301": 7,
                "U0036302": 2015,
                "U0741900": -4,
                "U0742000": 3600,
                "U1876601": 8,
                "U1876602": 2017,
                "U2679300": 4210,
                "U2679400": -4,
            },
            {
                "R0000100": 2,
                "R0536300": "M",
                "R0536402": 1982,
                "R9705200": 12,
                "R9705300": 13,
                "T8154001": 9,
                "T8154002": 2013,
                "T8821300": -4,
                "T8821400": 3600,
                "U0036301": 10,
                "U0036302": 2015,
                "U0741900": 4210,
                "U0742000": -4,
                "U1876601": 11,
                "U1876602": 2017,
                "U2679300": -4,
                "U2679400": 5240,
            },
        ]
    ).to_csv(input_csv, index=False)

    construct = _repo_root() / "scripts/05_construct_samples.py"
    residualize = _repo_root() / "scripts/06_age_residualize.py"
    subprocess.run([sys.executable, str(construct), "--project-root", str(root), "--cohort", "nlsy97"], cwd=_repo_root(), check=True)
    subprocess.run([sys.executable, str(residualize), "--project-root", str(root), "--cohort", "nlsy97"], cwd=_repo_root(), check=True)

    cfa = pd.read_csv(root / "data/processed/nlsy97_cfa.csv")
    resid = pd.read_csv(root / "data/processed/nlsy97_cfa_resid.csv")
    for frame in (cfa, resid):
        assert "age_2013" in frame.columns
        assert "age_2015" in frame.columns
        assert "age_2017" in frame.columns
        assert "occupation_code_2013" in frame.columns
        assert "occupation_code_2015" in frame.columns
        assert "occupation_code_2017" in frame.columns
    assert set(pd.to_numeric(cfa["age_2013"], errors="coerce").dropna().astype(int)) == {32, 31}
    assert set(pd.to_numeric(cfa["age_2015"], errors="coerce").dropna().astype(int)) == {34, 33}
    assert set(pd.to_numeric(resid["age_2017"], errors="coerce").dropna().astype(int)) == {36, 35}
    assert set(pd.to_numeric(cfa["occupation_code_2013"], errors="coerce").dropna().astype(int)) == {1020, 3600}
    assert set(pd.to_numeric(cfa["occupation_code_2015"], errors="coerce").dropna().astype(int)) == {3600, 4210}
    assert set(pd.to_numeric(resid["occupation_code_2017"], errors="coerce").dropna().astype(int)) == {4210, 5240}
