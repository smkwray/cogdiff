from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "63_build_age_matched_cross_cohort_contrasts.py"
    spec = importlib.util.spec_from_file_location("script63_build_age_matched_cross_cohort_contrasts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_age_matched_cross_cohort_contrasts_computes_rows(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
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
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)

    nlsy79_rows = []
    for i in range(120):
        g = (i - 60) / 12.0
        age = 37 + (i % 4)
        nlsy79_rows.append(
            {
                "GS": g + 0.1,
                "AR": g - 0.1,
                "WK": g + 0.2,
                "PC": g - 0.2,
                "NO": g + 0.15,
                "CS": g - 0.15,
                "AS": g + 0.05,
                "MK": g - 0.05,
                "MC": g + 0.03,
                "EI": g - 0.03,
                "age_2000": age,
                "household_income": 50000 + (4000 * g) + (300 * age),
                "annual_earnings": 35000 + (3500 * g) + (250 * age),
                "employment_2000": 1 if (0.8 * g + 0.03 * age) > 0.8 else 0,
            }
        )
    pd.DataFrame(nlsy79_rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    nlsy97_rows = []
    for i in range(130):
        g = (i - 65) / 12.0
        age_2019 = 37 + (i % 4)
        age_2021 = 38 + (i % 4)
        nlsy97_rows.append(
            {
                "GS": g + 0.1,
                "AR": g - 0.1,
                "WK": g + 0.2,
                "PC": g - 0.2,
                "NO": g + 0.15,
                "CS": g - 0.15,
                "AS": g + 0.05,
                "MK": g - 0.05,
                "MC": g + 0.03,
                "EI": g - 0.03,
                "age_2019": age_2019,
                "age_2021": age_2021,
                "household_income_2019": 52000 + (4200 * g) + (280 * age_2019),
                "household_income_2021": 54000 + (4300 * g) + (260 * age_2021),
                "annual_earnings_2019": 36000 + (3300 * g) + (220 * age_2019),
                "annual_earnings_2021": 38000 + (3400 * g) + (210 * age_2021),
                "employment_2019": 1 if (0.7 * g + 0.03 * age_2019) > 0.7 else 0,
                "employment_2021": 1 if (0.75 * g + 0.03 * age_2021) > 1.2 else 0,
            }
        )
    pd.DataFrame(nlsy97_rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    estimates, contrasts = module.run_age_matched_cross_cohort_contrasts(root=root)

    assert (root / "outputs" / "tables" / "age_matched_outcome_validity.csv").exists()
    assert (root / "outputs" / "tables" / "age_matched_cross_cohort_contrasts.csv").exists()
    assert int((estimates["status"] == "computed").sum()) >= 6
    assert int((contrasts["status"] == "computed").sum()) >= 3
    hh_2019 = estimates.loc[(estimates["cohort"] == "nlsy97") & (estimates["outcome"] == "household_income") & (estimates["age_col"] == "age_2019")].iloc[0]
    assert hh_2019["model_type"] == "ols"


def test_run_age_matched_cross_cohort_contrasts_handles_missing_windows(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
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
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"GS": 1.0, "AR": 1.0}, {"GS": 2.0, "AR": 2.0}]).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame([{"GS": 1.0, "AR": 1.0}, {"GS": 2.0, "AR": 2.0}]).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    estimates, contrasts = module.run_age_matched_cross_cohort_contrasts(root=root)
    assert (estimates["status"] == "not_feasible").all()
    assert contrasts.empty or (contrasts["status"] == "not_feasible").all()
