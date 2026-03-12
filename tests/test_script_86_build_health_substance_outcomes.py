from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "86_build_health_substance_outcomes.py"
    spec = importlib.util.spec_from_file_location("script86_build_health_substance_outcomes", path)
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


def test_nlsy79_health_substance(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy79.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    rows = []
    for i in range(120):
        g = (i - 60) / 30.0  # compressed range for class overlap
        noise = (((i * 7) % 11) - 5) / 5.0  # amplified noise
        # Lower g → worse health (with overlap near threshold)
        health = 4 if (g + noise) < -0.3 else 1
        if i % 3 == 0 and (g + noise) < -0.3:
            health = 5
        elif i % 3 == 0 and (g + noise) >= -0.3:
            health = 2
        # Smoking: daily if low g (with overlap)
        smoking = 1 if (g + noise) < -0.1 else 3
        # Alcohol: continuous
        alcohol = max(0, min(30, int(10 + g * 2 + noise)))
        # Marijuana: >0 if low g (with overlap)
        marijuana = (i % 5) + 1 if (g + noise) < 0.2 else 0
        # Height: varies around 66
        height = 66 + noise
        # Weight: centered near BMI=30 threshold at height~66 for balanced classes
        weight = 170 + int(noise * 25 - g * 15)
        row = _subtests(i, g)
        row.update({
            "birth_year": 1960 + (i % 4),
            "age": 20 + (i % 8),
            "health_status_40plus": health,
            "smoking_daily_2018": smoking,
            "alcohol_days_30_2018": alcohol,
            "marijuana_use_30_1984": marijuana,
            "height_inches_1985": height,
            "weight_pounds_2022": weight,
        })
        rows.append(row)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    out = module.run_health_substance_outcomes(root=root, cohorts=["nlsy79"])
    computed = out[out["status"] == "computed"]
    computed_names = set(computed["outcome"])
    # All expected binary outcomes should be computed
    expected_binary = {"poor_health", "current_smoker", "any_marijuana", "obese"}
    assert expected_binary.issubset(computed_names), (
        f"Expected binary outcomes {expected_binary} to be computed, got {computed_names}"
    )
    # Higher g should predict lower odds of poor health
    poor_health_row = computed[computed["outcome"] == "poor_health"].iloc[0]
    assert poor_health_row["odds_ratio_g"] < 1.0, (
        f"poor_health: OR={poor_health_row['odds_ratio_g']} should be < 1"
    )
    # Higher g should predict lower odds of current smoking
    smoker_row = computed[computed["outcome"] == "current_smoker"].iloc[0]
    assert smoker_row["odds_ratio_g"] < 1.0, (
        f"current_smoker: OR={smoker_row['odds_ratio_g']} should be < 1"
    )
    # Alcohol days should exist as a computed continuous outcome
    alcohol_rows = out[out["outcome"] == "alcohol_days_30"]
    assert len(alcohol_rows) > 0, "alcohol_days_30 outcome should exist"
    assert alcohol_rows.iloc[0]["status"] == "computed", "alcohol_days_30 should be computed"
    # Output CSV should exist
    assert (root / "outputs" / "tables" / "g_health_substance_outcomes.csv").exists()


def test_nlsy97_health_substance(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy97.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    rows = []
    for i in range(100):
        g = (i - 50) / 25.0  # compressed range for class overlap
        noise = (((i * 3) % 7) - 3) / 4.0  # amplified noise
        # Lower g → worse health (with overlap near threshold)
        health = 4 if (g + noise) < -0.3 else 1
        if i % 3 == 0 and (g + noise) < -0.3:
            health = 5
        elif i % 3 == 0 and (g + noise) >= -0.3:
            health = 2
        # Alcohol: continuous
        alcohol = max(0, min(30, int(8 + g * 2 + noise)))
        # Binge drinking: >0 if low g (with overlap)
        binge = (i % 4) + 1 if (g + noise) < 0.2 else 0
        # Marijuana: >0 if low g (with overlap)
        marijuana = (i % 5) + 1 if (g + noise) < 0.0 else 0
        # Height: 5 feet + inches (total 66-71 inches)
        height_feet = 5
        height_inches = 6 + (i % 6)
        # Weight: centered near BMI=30 at avg height ~68 for balanced classes
        weight = 195 + int(noise * 25 - g * 15)
        row = _subtests(i, g)
        row.update({
            "birth_year": 1981 + (i % 4),
            "health_status_2023": health,
            # smoking_days_30_2023 intentionally omitted (valid-skip issues)
            "alcohol_days_30_2023": alcohol,
            "binge_days_30_2023": binge,
            "marijuana_days_30_2015": marijuana,
            "height_feet_2011": height_feet,
            "height_inches_2011": height_inches,
            "weight_pounds_2011": weight,
        })
        rows.append(row)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    out = module.run_health_substance_outcomes(root=root, cohorts=["nlsy97"])
    computed = out[out["status"] == "computed"]
    computed_names = set(computed["outcome"])
    # All expected binary outcomes should be computed
    expected_binary = {"poor_health", "any_marijuana", "binge_drinker", "obese"}
    assert expected_binary.issubset(computed_names), (
        f"Expected binary outcomes {expected_binary} to be computed, got {computed_names}"
    )
    # Higher g should predict lower odds of poor health
    poor_health_row = computed[computed["outcome"] == "poor_health"].iloc[0]
    assert poor_health_row["odds_ratio_g"] < 1.0, (
        f"poor_health: OR={poor_health_row['odds_ratio_g']} should be < 1"
    )
    # Alcohol days should exist as a computed continuous outcome
    alcohol_rows = out[out["outcome"] == "alcohol_days_30"]
    assert len(alcohol_rows) > 0, "alcohol_days_30 outcome should exist"
    assert alcohol_rows.iloc[0]["status"] == "computed", "alcohol_days_30 should be computed"


def test_missing_columns_handled(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy97.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")
    _write(root / "config/nlsy79.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    # CSV with subtests only, no health columns
    rows = [_subtests(i, (i - 5) / 5.0) | {"age": 20 + i, "birth_year": 1982} for i in range(10)]
    pd.DataFrame(rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    out = module.run_health_substance_outcomes(root=root, cohorts=["nlsy79", "nlsy97"])
    assert (out["status"] == "not_feasible").all()
