from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_construct_samples_script_nlsy97(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write_file(
        project_root / "config/paths.yml",
        "interim_dir: data/interim\nprocessed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write_file(
        project_root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write_file(
        project_root / "config/nlsy97.yml",
        """
cohort: nlsy97
expected_age_range:
  min: 13
  max: 17
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  age_resid_col: birth_year
  subtests: [GS, AR, WK, PC, 'NO', CS, AS, MK, MC, EI]
  auto_col: AUTO
  shop_col: SHOP
  min_tests: 10
  missing_codes: [-1, -2, -3, -4, -5]
""",
    )

    input_csv = project_root / "data/interim/nlsy97/panel_extract.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            # retained after complete-case + dedupe
            {"person_id": 1, "sex": "M", "age": 15, "birth_year": 1982, "GS": 1, "AR": 1, "WK": 1, "PC": 1, "NO": 1, "CS": 1, "AUTO": 10, "SHOP": 8, "MK": 1, "MC": 1, "EI": 1},
            {"person_id": 1, "sex": "M", "age": 15, "birth_year": 1982, "GS": 2, "AR": 2, "WK": 2, "PC": 2, "NO": 2, "CS": 2, "AUTO": 11, "SHOP": 9, "MK": 2, "MC": 2, "EI": 2},
            # removed by age filter
            {"person_id": 2, "sex": "F", "age": 12, "birth_year": 1985, "GS": 1, "AR": 1, "WK": 1, "PC": 1, "NO": 1, "CS": 1, "AUTO": 7, "SHOP": 6, "MK": 1, "MC": 1, "EI": 1},
            # removed by test missing code
            {"person_id": 3, "sex": "F", "age": 16, "birth_year": 1981, "GS": 1, "AR": 1, "WK": 1, "PC": 1, "NO": 1, "CS": 1, "AUTO": 8, "SHOP": 6, "MK": 1, "MC": -4, "EI": 1},
            # retained
            {"person_id": 4, "sex": "M", "age": 17, "birth_year": 1980, "GS": 1, "AR": 1, "WK": 1, "PC": 1, "NO": 1, "CS": 1, "AUTO": 9, "SHOP": 9, "MK": 1, "MC": 1, "EI": 1},
        ]
    ).to_csv(input_csv, index=False)

    script = _repo_root() / "scripts/05_construct_samples.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy97"],
        cwd=_repo_root(),
        check=True,
    )

    out = pd.read_csv(project_root / "data/processed/nlsy97_cfa.csv")
    assert len(out) == 2
    assert "AS" in out.columns

    counts = pd.read_csv(project_root / "outputs/tables/sample_counts.csv")
    assert int(counts.loc[0, "n_after_age"]) == 4
    assert int(counts.loc[0, "n_after_test_rule"]) == 3
    assert int(counts.loc[0, "n_after_dedupe"]) == 2
    assert counts.loc[0, "input_path"] == "data/interim/nlsy97/panel_extract.csv"
    assert counts.loc[0, "output_path"] == "data/processed/nlsy97_cfa.csv"


def test_construct_samples_script_nlsy97_pos_neg_harmonization(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write_file(
        project_root / "config/paths.yml",
        "interim_dir: data/interim\nprocessed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write_file(
        project_root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write_file(
        project_root / "config/nlsy97.yml",
        """
cohort: nlsy97
expected_age_range:
  min: 13
  max: 17
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  age_resid_col: birth_year
  subtests: [GS, AR, WK, PC, 'NO', CS, AS, MK, MC, EI]
  auto_col: AUTO
  shop_col: SHOP
  min_tests: 10
  missing_codes: [-1, -2, -3, -4, -5]
  branch_harmonization:
    enabled: true
    method: signed_merge
    emit_source_cols: true
    pairs:
      - output: GS
        pos_col: GS_POS
        neg_col: GS_NEG
      - output: AR
        pos_col: AR_POS
        neg_col: AR_NEG
      - output: WK
        pos_col: WK_POS
        neg_col: WK_NEG
      - output: PC
        pos_col: PC_POS
        neg_col: PC_NEG
      - output: 'NO'
        pos_col: 'NO_POS'
        neg_col: 'NO_NEG'
      - output: CS
        pos_col: CS_POS
        neg_col: CS_NEG
      - output: AUTO
        pos_col: AUTO_POS
        neg_col: AUTO_NEG
      - output: SHOP
        pos_col: SHOP_POS
        neg_col: SHOP_NEG
      - output: MK
        pos_col: MK_POS
        neg_col: MK_NEG
      - output: MC
        pos_col: MC_POS
        neg_col: MC_NEG
      - output: EI
        pos_col: EI_POS
        neg_col: EI_NEG
""",
    )

    input_csv = project_root / "data/interim/nlsy97/panel_extract.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "person_id": 1,
                "sex": "M",
                "age": 15,
                "birth_year": 1982,
                "GS_POS": 100,
                "GS_NEG": -4,
                "AR_POS": 100,
                "AR_NEG": -4,
                "WK_POS": 100,
                "WK_NEG": -4,
                "PC_POS": 100,
                "PC_NEG": -4,
                "NO_POS": 100,
                "NO_NEG": -4,
                "CS_POS": 100,
                "CS_NEG": -4,
                "AUTO_POS": 100,
                "AUTO_NEG": -4,
                "SHOP_POS": 100,
                "SHOP_NEG": -4,
                "MK_POS": 100,
                "MK_NEG": -4,
                "MC_POS": 100,
                "MC_NEG": -4,
                "EI_POS": 100,
                "EI_NEG": -4,
            },
            {
                "person_id": 2,
                "sex": "F",
                "age": 16,
                "birth_year": 1981,
                "GS_POS": -4,
                "GS_NEG": 900,
                "AR_POS": -4,
                "AR_NEG": 900,
                "WK_POS": -4,
                "WK_NEG": 900,
                "PC_POS": -4,
                "PC_NEG": 900,
                "NO_POS": -4,
                "NO_NEG": 900,
                "CS_POS": -4,
                "CS_NEG": 900,
                "AUTO_POS": -4,
                "AUTO_NEG": 900,
                "SHOP_POS": -4,
                "SHOP_NEG": 900,
                "MK_POS": -4,
                "MK_NEG": 900,
                "MC_POS": -4,
                "MC_NEG": 900,
                "EI_POS": -4,
                "EI_NEG": 900,
            },
            {
                "person_id": 3,
                "sex": "F",
                "age": 12,
                "birth_year": 1984,
                "GS_POS": 105,
                "GS_NEG": -4,
                "AR_POS": 105,
                "AR_NEG": -4,
                "WK_POS": 105,
                "WK_NEG": -4,
                "PC_POS": 105,
                "PC_NEG": -4,
                "NO_POS": 105,
                "NO_NEG": -4,
                "CS_POS": 105,
                "CS_NEG": -4,
                "AUTO_POS": 105,
                "AUTO_NEG": -4,
                "SHOP_POS": 105,
                "SHOP_NEG": -4,
                "MK_POS": 105,
                "MK_NEG": -4,
                "MC_POS": 105,
                "MC_NEG": -4,
                "EI_POS": 105,
                "EI_NEG": -4,
            },
            {
                "person_id": 4,
                "sex": "M",
                "age": 17,
                "birth_year": 1980,
                "GS_POS": -4,
                "GS_NEG": 1000,
                "AR_POS": -4,
                "AR_NEG": 1000,
                "WK_POS": -4,
                "WK_NEG": 1000,
                "PC_POS": -4,
                "PC_NEG": 1000,
                "NO_POS": -4,
                "NO_NEG": 1000,
                "CS_POS": -4,
                "CS_NEG": 1000,
                "AUTO_POS": -4,
                "AUTO_NEG": 1000,
                "SHOP_POS": -4,
                "SHOP_NEG": 1000,
                "MK_POS": -4,
                "MK_NEG": 1000,
                "MC_POS": -4,
                "MC_NEG": 1000,
                "EI_POS": -4,
                "EI_NEG": 1000,
            },
        ]
    ).to_csv(input_csv, index=False)

    script = _repo_root() / "scripts/05_construct_samples.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy97"],
        cwd=_repo_root(),
        check=True,
    )

    out = pd.read_csv(project_root / "data/processed/nlsy97_cfa.csv")
    assert len(out) == 3
    assert "AS" in out.columns
    assert out[["GS", "AR", "WK", "PC", "NO", "CS", "MK", "MC", "EI", "AS"]].notna().all().all()
    assert set(out["GS_source"]) == {"pos", "neg"}
    assert set(out["AUTO_source"]) == {"pos", "neg"}
    pos_rows = out[out["GS_source"] == "pos"]
    neg_rows = out[out["GS_source"] == "neg"]
    assert (pos_rows["GS"] > 0).all()
    assert (neg_rows["GS"] < 0).all()

    counts = pd.read_csv(project_root / "outputs/tables/sample_counts.csv")
    assert int(counts.loc[0, "n_after_age"]) == 3
    assert int(counts.loc[0, "n_after_test_rule"]) == 3
    assert int(counts.loc[0, "n_after_dedupe"]) == 3


def test_construct_samples_script_nlsy97_harmonization_implied_decimals(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write_file(
        project_root / "config/paths.yml",
        "interim_dir: data/interim\nprocessed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write_file(
        project_root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write_file(
        project_root / "config/nlsy97.yml",
        """
cohort: nlsy97
expected_age_range:
  min: 13
  max: 17
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  age_resid_col: birth_year
  subtests: [GS, AR]
  min_tests: 2
  missing_codes: [-1, -2, -3, -4, -5]
  branch_harmonization:
    enabled: true
    method: signed_merge
    implied_decimal_places: 3
    pairs:
      - output: GS
        pos_col: GS_POS
        neg_col: GS_NEG
      - output: AR
        pos_col: AR_POS
        neg_col: AR_NEG
""",
    )

    input_csv = project_root / "data/interim/nlsy97/panel_extract.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "person_id": 1,
                "sex": "M",
                "age": 15,
                "birth_year": 1982,
                "GS_POS": 1500.0,
                "GS_NEG": -4.0,
                "AR_POS": 1000.0,
                "AR_NEG": -4.0,
            },
            {
                "person_id": 2,
                "sex": "F",
                "age": 16,
                "birth_year": 1981,
                "GS_POS": -4.0,
                "GS_NEG": 875.0,
                "AR_POS": -4.0,
                "AR_NEG": 500.0,
            },
        ]
    ).to_csv(input_csv, index=False)

    script = _repo_root() / "scripts/05_construct_samples.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy97"],
        cwd=_repo_root(),
        check=True,
    )

    out = pd.read_csv(project_root / "data/processed/nlsy97_cfa.csv")
    gs = sorted(float(x) for x in out["GS"].tolist())
    ar = sorted(float(x) for x in out["AR"].tolist())
    assert gs == [-0.875, 1.5]
    assert ar == [-0.5, 1.0]


def test_construct_samples_script_nlsy97_pos_neg_harmonization_diagnostics(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write_file(
        project_root / "config/paths.yml",
        "interim_dir: data/interim\nprocessed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write_file(
        project_root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write_file(
        project_root / "config/nlsy97.yml",
        """
cohort: nlsy97
expected_age_range:
  min: 13
  max: 17
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  age_resid_col: birth_year
  subtests: [GS, AR]
  auto_col: AUTO
  shop_col: SHOP
  min_tests: 2
  missing_codes: [-1, -2, -3, -4, -5]
  branch_harmonization:
    enabled: true
    method: signed_merge
    emit_source_cols: true
    pairs:
      - output: GS
        pos_col: GS_POS
        neg_col: GS_NEG
      - output: AR
        pos_col: AR_POS
        neg_col: AR_NEG
""",
    )

    input_csv = project_root / "data/interim/nlsy97/panel_extract.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "person_id": 1,
                "sex": "M",
                "age": 15,
                "birth_year": 1982,
                "GS_POS": 100.0,
                "GS_NEG": -4.0,
                "AR_POS": 20.0,
                "AR_NEG": -4.0,
            },
            {
                "person_id": 2,
                "sex": "F",
                "age": 16,
                "birth_year": 1981,
                "GS_POS": -1.0,
                "GS_NEG": -110.0,
                "AR_POS": -1.0,
                "AR_NEG": -120.0,
            },
            {
                "person_id": 3,
                "sex": "F",
                "age": 16,
                "birth_year": 1981,
                "GS_POS": 60.0,
                "GS_NEG": -80.0,
                "AR_POS": 70.0,
                "AR_NEG": -90.0,
            },
            {
                "person_id": 4,
                "sex": "M",
                "age": 16,
                "birth_year": 1980,
                "GS_POS": -1.0,
                "GS_NEG": -140.0,
                "AR_POS": 55.0,
                "AR_NEG": -65.0,
            },
        ]
    ).to_csv(input_csv, index=False)

    script = _repo_root() / "scripts/05_construct_samples.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy97"],
        cwd=_repo_root(),
        check=True,
    )

    balance = pd.read_csv(project_root / "outputs/tables/nlsy97_harmonization_branch_balance_by_sex.csv")
    continuity = pd.read_csv(project_root / "outputs/tables/nlsy97_harmonization_continuity_near_zero.csv")
    assert set(balance["subtest"]) == {"GS", "AR"}
    assert set(balance["source"]) >= {"pos", "neg", "both"}

    gs_male_pos = balance.loc[(balance["subtest"] == "GS") & (balance["sex"] == "male") & (balance["source"] == "pos")]
    gs_female_neg = balance.loc[
        (balance["subtest"] == "GS") & (balance["sex"] == "female") & (balance["source"] == "neg")
    ]
    gs_female_both = balance.loc[
        (balance["subtest"] == "GS") & (balance["sex"] == "female") & (balance["source"] == "both")
    ]
    assert len(gs_male_pos) == 1
    assert int(gs_male_pos.iloc[0]["count"]) == 1
    assert float(gs_male_pos.iloc[0]["share_within_sex"]) == 0.5
    assert int(gs_female_neg.iloc[0]["count"]) == 1
    assert int(gs_female_both.iloc[0]["count"]) == 1

    ar_row = continuity.loc[continuity["subtest"] == "AR"].iloc[0]
    gs_row = continuity.loc[continuity["subtest"] == "GS"].iloc[0]
    assert int(gs_row["n_non_missing"]) == 4
    assert int(gs_row["source_neg_count"]) == 2
    assert int(gs_row["source_pos_count"]) == 1
    assert int(gs_row["source_both_count"]) == 1
    assert int(gs_row["n_within_window"]) == 0
    assert int(ar_row["n_within_window"]) == 0
    assert float(gs_row["continuity_gap"]) == 170.0
    assert float(ar_row["continuity_gap"]) == 140.0


def test_age_residualize_script(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write_file(
        project_root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write_file(
        project_root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write_file(
        project_root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  age_resid_col: birth_year
  subtests: [GS, AR]
""",
    )

    x = pd.Series([1979, 1980, 1981, 1982, 1983, 1984], dtype=float)
    frame = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "sex": ["F", "M", "F", "M", "F", "M"],
            "birth_year": x,
            "GS": 1.0 + 2.0 * x + 0.1 * (x**2),
            "AR": 3.0 + 1.0 * x + 0.2 * (x**2),
        }
    )
    in_path = project_root / "data/processed/nlsy79_cfa.csv"
    in_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(in_path, index=False)

    script = _repo_root() / "scripts/06_age_residualize.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy79", "--source-path", str(in_path)],
        cwd=_repo_root(),
        check=True,
    )

    out = pd.read_csv(project_root / "data/processed/nlsy79_cfa_resid.csv")
    assert abs(float(out["GS"].mean())) < 1e-8
    assert abs(float(out["AR"].mean())) < 1e-8
    assert float(out["GS"].std(ddof=1)) < 1e-8
    assert float(out["AR"].std(ddof=1)) < 1e-8

    diag = pd.read_csv(project_root / "outputs/tables/residualization_diagnostics_nlsy79.csv")
    assert set(diag["subtest"]) == {"GS", "AR"}


def test_age_residualize_script_standardize_output_from_config(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write_file(
        project_root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write_file(
        project_root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write_file(
        project_root / "config/nlsy97.yml",
        """
cohort: nlsy97
sample_construct:
  age_resid_col: birth_year
  subtests: [GS, AR]
  standardize_output: true
""",
    )

    in_path = project_root / "data/processed/nlsy97_cfa.csv"
    in_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "birth_year": [1, 2, 3, 4, 5],
            "GS": [1, 8, 27, 64, 125],
            "AR": [2, 16, 54, 128, 250],
        }
    ).to_csv(in_path, index=False)

    script = _repo_root() / "scripts/06_age_residualize.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy97"],
        cwd=_repo_root(),
        check=True,
    )

    out = pd.read_csv(project_root / "data/processed/nlsy97_cfa_resid.csv")
    assert len(out) == 5
    assert abs(float(out["GS"].mean())) < 1e-8
    assert abs(float(out["AR"].mean())) < 1e-8
    assert abs(float(out["GS"].std(ddof=1)) - 1.0) < 1e-8
    assert abs(float(out["AR"].std(ddof=1)) - 1.0) < 1e-8


def test_age_residualize_script_applies_column_map(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write_file(
        project_root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write_file(
        project_root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write_file(
        project_root / "config/nlsy97.yml",
        """
cohort: nlsy97
sample_construct:
  age_resid_col: birth_year
  subtests: [GS]
  column_map:
    RAW_RACE: race_ethnicity
    RAW_BIRTH: birth_year
""",
    )

    in_path = project_root / "data/processed/nlsy97_cfa.csv"
    in_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "sex": ["F", "M", "F", "M"],
            "RAW_RACE": [1, 2, 1, 3],
            "RAW_BIRTH": [1980, 1981, 1982, 1983],
            "GS": [10, 12, 14, 16],
        }
    ).to_csv(in_path, index=False)

    script = _repo_root() / "scripts/06_age_residualize.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy97"],
        cwd=_repo_root(),
        check=True,
    )

    out = pd.read_csv(project_root / "data/processed/nlsy97_cfa_resid.csv")
    assert "race_ethnicity" in out.columns
    assert "RAW_RACE" not in out.columns
    assert "birth_year" in out.columns
    assert "RAW_BIRTH" not in out.columns


def test_construct_samples_cnlsy_age_months_conversion(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write_file(
        project_root / "config/paths.yml",
        "interim_dir: data/interim\nprocessed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write_file(
        project_root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write_file(
        project_root / "config/cnlsy.yml",
        """
cohort: cnlsy
expected_age_range:
  min: 5
  max: 18
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: csage
  age_resid_col: birth_year
  age_unit: months
  subtests: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
  min_tests: 3
  adequacy:
    min_total_n: 10
    min_per_sex_n: 3
  missing_codes: [-1, -2, -3, -4, -5, -7]
""",
    )

    input_csv = project_root / "data/interim/cnlsy/panel_extract.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"person_id": 1, "sex": "F", "csage": 60, "birth_year": 1998, "PPVT": 90, "PIAT_RR": 85, "PIAT_RC": 80, "PIAT_MATH": 88, "DIGITSPAN": 92},
            {"person_id": 2, "sex": "M", "csage": 180, "birth_year": 1996, "PPVT": 95, "PIAT_RR": 86, "PIAT_RC": 82, "PIAT_MATH": 90, "DIGITSPAN": 93},
            {"person_id": 3, "sex": "F", "csage": 240, "birth_year": 1995, "PPVT": 88, "PIAT_RR": 84, "PIAT_RC": 79, "PIAT_MATH": 86, "DIGITSPAN": 90},
            {"person_id": 4, "sex": "M", "csage": 120, "birth_year": 1997, "PPVT": -7, "PIAT_RR": -7, "PIAT_RC": -7, "PIAT_MATH": 91, "DIGITSPAN": 94},
        ]
    ).to_csv(input_csv, index=False)

    script = _repo_root() / "scripts/05_construct_samples.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "cnlsy"],
        cwd=_repo_root(),
        check=True,
    )

    counts = pd.read_csv(project_root / "outputs/tables/sample_counts.csv")
    assert int(counts.loc[0, "n_after_age"]) == 3
    assert int(counts.loc[0, "n_after_test_rule"]) == 2
    assert counts.loc[0, "information_adequacy_status"] == "low_information"
    assert "n_total<10" in str(counts.loc[0, "information_adequacy_reason"])

    coverage = pd.read_csv(project_root / "outputs/tables/cnlsy_indicator_coverage.csv")
    scenarios = pd.read_csv(project_root / "outputs/tables/cnlsy_test_rule_scenarios.csv")
    assert set(coverage["subtest"]) == {"PPVT", "PIAT_RR", "PIAT_RC", "PIAT_MATH", "DIGITSPAN"}
    assert set(scenarios["min_tests"]) == {1, 2, 3, 4, 5}
