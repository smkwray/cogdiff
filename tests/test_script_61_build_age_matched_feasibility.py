from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "61_build_age_matched_feasibility.py"
    spec = importlib.util.spec_from_file_location("script61_build_age_matched_feasibility", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_age_matched_feasibility_summarizes_overlap(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "age_2000": [36, 38, 40],
            "education_years": [12, 14, 16],
            "household_income": [30000, 40000, 50000],
            "net_worth": [1000, 5000, 9000],
            "employment_2000": [1, 1, 0],
        }
    ).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame(
        {
            "age_2010": [26, 28, 30],
            "age_2011": [27, 29, 31],
            "age_2019": [35, 37, 39],
            "age_2021": [37, 39, 41],
            "education_years": [13, 15, 17],
            "household_income": [32000, 42000, 52000],
            "household_income_2019": [40000, 50000, 60000],
            "household_income_2021": [42000, 52000, 62000],
            "net_worth": [1500, 5500, 9500],
            "employment_2011": [1, 0, 1],
            "employment_2019": [1, 1, 0],
            "employment_2021": [1, 1, 1],
            "annual_earnings_2019": [25000, 35000, 45000],
            "annual_earnings_2021": [27000, 37000, 47000],
        }
    ).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    windows, overlaps = module.run_age_matched_feasibility(root=root, cohorts=["nlsy79", "nlsy97"])

    assert (root / "outputs" / "tables" / "age_matched_feasibility.csv").exists()
    assert (root / "outputs" / "tables" / "age_matched_overlap_summary.csv").exists()
    assert set(windows["status"]) == {"computed", "not_feasible"}
    education_79 = windows.loc[(windows["cohort"] == "nlsy79") & (windows["outcome"] == "education")].iloc[0]
    assert float(education_79["age_min"]) == 36.0
    assert int(education_79["n_used"]) == 3

    overlap_row = overlaps.loc[
        (overlaps["outcome"] == "education")
        & (overlaps["age_col_a"] == "age_2000")
        & (overlaps["age_col_b"] == "age_2010")
    ].iloc[0]
    assert overlap_row["status"] == "not_feasible"
    assert overlap_row["reason"] == "no_age_overlap"


def test_run_age_matched_feasibility_detects_real_overlap(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "age_2000": [29, 30, 31],
            "education_years": [12, 13, 14],
            "household_income": [38000, 48000, 58000],
        }
    ).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame(
        {
            "age_2010": [30, 31, 32],
            "education_years": [13, 14, 15],
            "age_2011": [31, 32, 33],
            "age_2019": [30, 31, 32],
            "household_income_2019": [40000, 50000, 60000],
            "employment_2019": [1, 1, 0],
            "annual_earnings_2019": [30000, 40000, 50000],
        }
    ).to_csv(
        processed / "nlsy97_cfa_resid.csv",
        index=False,
    )

    _, overlaps = module.run_age_matched_feasibility(root=root, cohorts=["nlsy79", "nlsy97"])
    overlap_row = overlaps.loc[
        (overlaps["outcome"] == "education")
        & (overlaps["age_col_a"] == "age_2000")
        & (overlaps["age_col_b"] == "age_2010")
    ].iloc[0]
    assert overlap_row["status"] == "computed"
    assert float(overlap_row["overlap_min"]) == 30.0
    assert float(overlap_row["overlap_max"]) == 31.0

    later_overlap = overlaps.loc[
        (overlaps["outcome"] == "household_income")
        & (overlaps["age_col_a"] == "age_2000")
        & (overlaps["age_col_b"] == "age_2019")
    ].iloc[0]
    assert later_overlap["status"] == "computed"
    assert float(later_overlap["overlap_min"]) == 30.0
    assert float(later_overlap["overlap_max"]) == 31.0
