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


def test_nlsy97_degree_2021_survives_construct_and_residualize(tmp_path: Path) -> None:
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
    U4976701: interview_month_2021
    U4976702: interview_year_2021
    U5072600: degree_2021
expected_age_range:
  min: 13
  max: 17
""",
    )

    input_csv = root / "data/interim/nlsy97/panel_extract.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"R0000100": 1, "R0536300": "F", "R0536402": 1981, "R9705200": 10, "R9705300": 11, "U4976701": 6, "U4976702": 2021, "U5072600": 5},
            {"R0000100": 2, "R0536300": "M", "R0536402": 1982, "R9705200": 12, "R9705300": 13, "U4976701": 7, "U4976702": 2021, "U5072600": 8},
            {"R0000100": 3, "R0536300": "F", "R0536402": 1983, "R9705200": 14, "R9705300": 15, "U4976701": 8, "U4976702": 2021, "U5072600": -4},
        ]
    ).to_csv(input_csv, index=False)

    construct = _repo_root() / "scripts/05_construct_samples.py"
    residualize = _repo_root() / "scripts/06_age_residualize.py"
    subprocess.run([sys.executable, str(construct), "--project-root", str(root), "--cohort", "nlsy97"], cwd=_repo_root(), check=True)
    subprocess.run([sys.executable, str(residualize), "--project-root", str(root), "--cohort", "nlsy97"], cwd=_repo_root(), check=True)

    cfa = pd.read_csv(root / "data/processed/nlsy97_cfa.csv")
    resid = pd.read_csv(root / "data/processed/nlsy97_cfa_resid.csv")
    for frame in (cfa, resid):
        assert "degree_2021" in frame.columns
        assert "age_2021" in frame.columns
    assert set(pd.to_numeric(cfa["degree_2021"], errors="coerce").dropna().astype(int)) == {5, 8}
    assert set(pd.to_numeric(resid["degree_2021"], errors="coerce").dropna().astype(int)) == {5, 8}
