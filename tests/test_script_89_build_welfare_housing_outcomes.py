from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "89_build_welfare_housing_outcomes.py"
    spec = importlib.util.spec_from_file_location("script89_build_welfare_housing_outcomes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


MODELS_YML = """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
"""


def _subtests(i: int, g: float) -> dict:
    return {
        "GS": g + 0.1, "AR": g - 0.1, "WK": g + 0.2, "PC": g - 0.2,
        "NO": g + 0.15, "CS": g - 0.15, "AS": g + 0.05, "MK": g - 0.05,
        "MC": g + 0.03, "EI": g - 0.03,
    }


def test_nlsy79_welfare_housing(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy79.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    rows = []
    for i in range(120):
        g = (i - 60) / 15.0
        noise = (((i * 7) % 11) - 5) / 10.0
        # Lower g → more likely to receive food stamps
        if (g + noise) < -0.3:
            food_stamps = max(0, min(5000, int(500 - g * 200 + noise * 100)))
        else:
            food_stamps = 0
        # Lower g → more likely to receive AFDC
        if (g + noise) < -0.8:
            afdc = max(0, min(5000, int(500 - g * 200 + noise * 100)))
        else:
            afdc = 0
        # Higher g → more likely to be homeowner
        homeowner = 1 if (g + noise) > -0.5 else 0
        row = _subtests(i, g)
        row.update({
            "food_stamps_amount_2000": food_stamps,
            "afdc_amount_2000": afdc,
            "homeowner_2000": homeowner,
            "birth_year": 1960 + (i % 4),
            "age": 20 + (i % 8),
        })
        rows.append(row)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    out = module.run_welfare_housing_outcomes(root=root, cohorts=["nlsy79"])
    computed = out[out["status"] == "computed"]
    computed_names = set(computed["outcome"])
    assert computed_names == {"received_food_stamps", "received_afdc", "any_welfare", "homeowner"}, (
        f"Expected all 4 welfare/housing outcomes computed, got {computed_names}"
    )
    # Higher g → less welfare
    fs_row = computed[computed["outcome"] == "received_food_stamps"].iloc[0]
    assert fs_row["odds_ratio_g"] < 1.0, (
        f"received_food_stamps: OR={fs_row['odds_ratio_g']} should be < 1"
    )
    # Higher g → more likely homeowner
    ho_row = computed[computed["outcome"] == "homeowner"].iloc[0]
    assert ho_row["odds_ratio_g"] > 1.0, (
        f"homeowner: OR={ho_row['odds_ratio_g']} should be > 1"
    )


def test_nlsy97_welfare_housing(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy97.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    rows = []
    for i in range(100):
        g = (i - 50) / 12.0
        noise = (((i * 3) % 7) - 3) / 6.0
        # Lower g → more likely to receive govt program income
        govt_program = 1 if (g + noise) < -0.5 else 0
        # Higher g → more likely to own house (type 1 = owns, 6 = does not own)
        house_type = 1 if (g + noise) > -0.3 else 6
        row = _subtests(i, g)
        row.update({
            "govt_program_income_2019": govt_program,
            "house_type_40": house_type,
            "birth_year": 1981 + (i % 4),
        })
        rows.append(row)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    out = module.run_welfare_housing_outcomes(root=root, cohorts=["nlsy97"])
    computed = out[out["status"] == "computed"]
    computed_names = set(computed["outcome"])
    assert computed_names == {"received_govt_program_income", "homeowner"}, (
        f"Expected both nlsy97 outcomes computed, got {computed_names}"
    )
    # Higher g → less govt program income
    gpi_row = computed[computed["outcome"] == "received_govt_program_income"].iloc[0]
    assert gpi_row["odds_ratio_g"] < 1.0, (
        f"received_govt_program_income: OR={gpi_row['odds_ratio_g']} should be < 1"
    )
    # Higher g → more likely homeowner
    ho_row = computed[computed["outcome"] == "homeowner"].iloc[0]
    assert ho_row["odds_ratio_g"] > 1.0, (
        f"homeowner: OR={ho_row['odds_ratio_g']} should be > 1"
    )


def test_missing_columns_handled(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy97.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")
    _write(root / "config/nlsy79.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    # CSV with subtests only, no welfare/housing columns
    rows = [_subtests(i, (i - 5) / 5.0) | {"age": 20 + i, "birth_year": 1982} for i in range(10)]
    pd.DataFrame(rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    out = module.run_welfare_housing_outcomes(root=root, cohorts=["nlsy79", "nlsy97"])
    assert (out["status"] == "not_feasible").all()
