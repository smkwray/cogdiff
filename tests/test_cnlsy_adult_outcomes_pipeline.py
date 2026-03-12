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


def test_cnlsy_adult_outcomes_survive_construct_and_residualize(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "interim_dir: data/interim\nprocessed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", "cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n")
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
    Y1211300: education_years
    Y3291400: wage_income_2014_raw
    Y3291500: wage_income_2014_best_est
    Y3299800: family_income_2014_raw
    Y3299900: family_income_2014_best_est
    Y3066000: employment_2014
    Y3112400: annual_earnings
    Y3331900: age_2014
    Y3332100: education_years_2014
    Y3332200: degree_2014
    Y3333000: enrolled_2014
    Y3333300: num_current_jobs_2014
    Y3333400: total_hours_2014
    C3616100: PPVT
    C3993900: PIAT_RR
    C3994200: PIAT_RC
panel_extract:
  required_columns:
    - C0000100
    - C0000200
    - C0005400
    - C5801600
    - C0005700
    - C3616100
    - C3993900
    - C3994200
    - Y1211300
  optional_columns:
    - Y3291400
    - Y3291500
    - Y3299800
    - Y3299900
    - Y3066000
    - Y3112400
    - Y3331900
    - Y3332100
    - Y3332200
    - Y3333000
    - Y3333300
    - Y3333400
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
                "Y1211300": 8,
                "Y3291400": 31000,
                "Y3291500": 32000,
                "Y3299800": 45000,
                "Y3299900": 47000,
                "Y3066000": 1,
                "Y3112400": 42000,
                "Y3331900": 24,
                "Y3332100": 14,
                "Y3332200": 4,
                "Y3333000": 1,
                "Y3333300": 1,
                "Y3333400": 35,
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
                "Y1211300": 9,
                "Y3291400": -5,
                "Y3291500": 28000,
                "Y3299800": -5,
                "Y3299900": 40000,
                "Y3066000": 2,
                "Y3112400": 51000,
                "Y3331900": 25,
                "Y3332100": 15,
                "Y3332200": 5,
                "Y3333000": 0,
                "Y3333300": 2,
                "Y3333400": 40,
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
                "Y1211300": 10,
                "Y3291400": 0,
                "Y3291500": 0,
                "Y3299800": 52000,
                "Y3299900": 53000,
                "Y3066000": 3,
                "Y3112400": 0,
                "Y3331900": 26,
                "Y3332100": 16,
                "Y3332200": 6,
                "Y3333000": 0,
                "Y3333300": 0,
                "Y3333400": 0,
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
    for frame in (cfa, resid):
        assert "employment_2014" in frame.columns
        assert "annual_earnings" in frame.columns
        assert "age_2014" in frame.columns
        assert "education_years_2014" in frame.columns
        assert "degree_2014" in frame.columns
        assert "enrolled_2014" in frame.columns
        assert "num_current_jobs_2014" in frame.columns
        assert "total_hours_2014" in frame.columns
        assert "wage_income_2014" in frame.columns
        assert "family_income_2014" in frame.columns
    assert set(pd.to_numeric(cfa["employment_2014"], errors="coerce").dropna().astype(int)) == {0, 1}
    assert set(pd.to_numeric(resid["employment_2014"], errors="coerce").dropna().astype(int)) == {0, 1}
    assert set(pd.to_numeric(cfa["annual_earnings"], errors="coerce").dropna().astype(int)) == {0, 42000, 51000}
    assert set(pd.to_numeric(resid["annual_earnings"], errors="coerce").dropna().astype(int)) == {0, 42000, 51000}
    assert set(pd.to_numeric(cfa["wage_income_2014"], errors="coerce").dropna().astype(int)) == {0, 31000}
    assert set(pd.to_numeric(cfa["family_income_2014"], errors="coerce").dropna().astype(int)) == {45000, 52000}
