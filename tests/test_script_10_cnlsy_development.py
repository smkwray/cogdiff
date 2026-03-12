from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from nls_pipeline.cnlsy import AGEBIN_SUMMARY_COLUMNS, LONGITUDINAL_SUMMARY_COLUMNS


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_script_10_cnlsy_development_writes_outputs_and_schemas(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_file(
        root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write_file(
        root / "config/models.yml",
        """
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write_file(
        root / "config/cnlsy.yml",
        """
cohort: cnlsy
expected_age_range:
  min: 5
  max: 18
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  subtests: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )

    input_path = root / "data/processed/cnlsy_cfa_resid.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
            "age": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18],
            "g": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "PPVT": [1.0] * 12,
            "PIAT_RR": [2.0] * 12,
            "PIAT_RC": [3.0] * 12,
            "PIAT_MATH": [4.0] * 12,
            "DIGITSPAN": [5.0] * 12,
        }
    ).to_csv(input_path, index=False)

    script = _repo_root() / "scripts/10_cnlsy_development.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        check=True,
    )

    age_summary = pd.read_csv(root / "outputs/tables/cnlsy_agebin_summary.csv")
    long_summary = pd.read_csv(root / "outputs/tables/cnlsy_longitudinal_model.csv")

    assert list(age_summary.columns) == AGEBIN_SUMMARY_COLUMNS
    assert len(age_summary) == 7
    assert list(long_summary.columns) == LONGITUDINAL_SUMMARY_COLUMNS
    assert len(long_summary) == 1
    assert int(long_summary.loc[0, "n_obs"]) == 12

    mean_plot = root / "outputs/figures/cnlsy_age_trends_mean.png"
    vr_plot = root / "outputs/figures/cnlsy_age_trends_vr.png"
    assert mean_plot.exists() and mean_plot.stat().st_size > 0
    assert vr_plot.exists() and vr_plot.stat().st_size > 0


def test_script_10_cnlsy_development_derives_score_when_missing(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write_file(
        root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write_file(
        root / "config/models.yml",
        """
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write_file(
        root / "config/cnlsy.yml",
        """
cohort: cnlsy
expected_age_range:
  min: 5
  max: 18
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  subtests: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )

    input_path = root / "data/processed/cnlsy_cfa_resid.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2, 2, 2],
            "sex": ["M", "M", "M", "F", "F", "F"],
            "age": [5, 7, 9, 5, 7, 9],
            "PPVT": [10.0, 11.0, 12.0, 24.0, 25.0, 26.0],
            "PIAT_RR": [12.0, 13.0, 14.0, 26.0, 27.0, 28.0],
            "PIAT_RC": [14.0, 15.0, 16.0, 28.0, 29.0, 30.0],
            "PIAT_MATH": [16.0, 17.0, 18.0, 30.0, 31.0, 32.0],
            "DIGITSPAN": [18.0, 19.0, 20.0, 32.0, 33.0, 34.0],
        }
    ).to_csv(input_path, index=False)

    script = _repo_root() / "scripts/10_cnlsy_development.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        check=True,
    )

    age_summary = pd.read_csv(root / "outputs/tables/cnlsy_agebin_summary.csv")
    long_summary = pd.read_csv(root / "outputs/tables/cnlsy_longitudinal_model.csv")

    assert len(age_summary) == 7
    assert long_summary.loc[0, "model_type"] != "insufficient_data"
    assert "n_obs" in long_summary.columns
