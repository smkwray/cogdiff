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


def test_cnlsy_mpubid_survives_construct_and_residualize(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "interim_dir: data/interim\nprocessed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        "cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n",
    )
    _write(
        root / "config/cnlsy.yml",
        """
cohort: cnlsy
expected_age_range:
  min: 5
  max: 18
sample_construct:
  input_file: panel_extract.csv
  id_col: person_id
  sex_col: sex
  age_col: csage
  age_resid_col: birth_year
  age_unit: years
  subtests: [PPVT, PIAT_RR, PIAT_RC]
  min_tests: 3
  missing_codes: [-1, -2, -3, -4, -5, -7]
  column_map:
    C0000100: person_id
    C0000200: MPUBID
    C0005400: sex
    C5801600: csage
    C0005700: birth_year
    C0053500: mother_education
    Y1211300: education_years
    C3616100: PPVT
    C3993900: PIAT_RR
    C3994200: PIAT_RC
panel_extract:
  required_columns:
    - C0000100
    - C0000200
    - C0005400
    - C0005700
    - C5801600
    - C3616100
    - C3993900
    - C3994200
    - Y1211300
""",
    )

    input_csv = root / "data/interim/cnlsy/panel_extract.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "C0000100": 1,
                "C0000200": 1001,
                "C0005400": 2,
                "C5801600": 10.0,
                "C0005700": 1990,
                "C0053500": 12,
                "Y1211300": 8,
                "C3616100": 90,
                "C3993900": 88,
                "C3994200": 86,
            },
            {
                "C0000100": 2,
                "C0000200": 1002,
                "C0005400": 1,
                "C5801600": 12.0,
                "C0005700": 1989,
                "C0053500": 11,
                "Y1211300": 9,
                "C3616100": 95,
                "C3993900": 90,
                "C3994200": 87,
            },
            {
                "C0000100": 3,
                "C0000200": 1001,
                "C0005400": 2,
                "C5801600": 14.0,
                "C0005700": 1988,
                "C0053500": 12,
                "Y1211300": 10,
                "C3616100": 97,
                "C3993900": 92,
                "C3994200": 89,
            },
        ]
    ).to_csv(input_csv, index=False)

    construct = _repo_root() / "scripts/05_construct_samples.py"
    residualize = _repo_root() / "scripts/06_age_residualize.py"
    subprocess.run([sys.executable, str(construct), "--project-root", str(root), "--cohort", "cnlsy"], cwd=_repo_root(), check=True)
    subprocess.run([sys.executable, str(residualize), "--project-root", str(root), "--cohort", "cnlsy"], cwd=_repo_root(), check=True)

    cfa = pd.read_csv(root / "data/processed/cnlsy_cfa.csv")
    resid = pd.read_csv(root / "data/processed/cnlsy_cfa_resid.csv")
    assert "MPUBID" in cfa.columns
    assert "MPUBID" in resid.columns
    assert set(pd.to_numeric(cfa["MPUBID"], errors="coerce").dropna().astype(int)) == {1001, 1002}
    assert set(pd.to_numeric(resid["MPUBID"], errors="coerce").dropna().astype(int)) == {1001, 1002}
