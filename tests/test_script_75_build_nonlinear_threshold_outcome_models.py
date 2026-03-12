from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "75_build_nonlinear_threshold_outcome_models.py"
    spec = importlib.util.spec_from_file_location("script75_build_nonlinear_threshold_outcome_models", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_row(g: float, age: float, cohort: str) -> dict[str, float]:
    common = {
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
    }
    if cohort == "nlsy79":
        common.update(
            {
                "age_2000": age,
                "annual_earnings": float(np.expm1(8.5 + 0.4 * g + 0.08 * (g ** 2) + 0.01 * age)),
                "household_income": float(np.expm1(9.0 + 0.35 * g + 0.06 * (g ** 2) + 0.01 * age)),
                "employment_2000": float(1 if g + 0.03 * (age - 40) > -0.3 else 0),
                "highest_degree_ever": float(3 if g > 0.1 else 1),
            }
        )
    else:
        common.update(
            {
                "age_2021": age,
                "annual_earnings_2021": float(np.expm1(8.8 + 0.45 * g + 0.10 * (g ** 2) + 0.01 * age)),
                "household_income_2021": float(np.expm1(9.2 + 0.38 * g + 0.07 * (g ** 2) + 0.01 * age)),
                "employment_2021": float(1 if g + 0.02 * (age - 39) > -0.35 else 0),
                "degree_2021": float(5 if g > 0.15 else 3),
            }
        )
    return common


def test_run_nonlinear_threshold_outcome_models_computes_outputs(tmp_path: Path) -> None:
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
    nlsy79 = pd.DataFrame([_make_row((i - 150) / 50.0, 36 + (i % 9), "nlsy79") for i in range(300)])
    nlsy97 = pd.DataFrame([_make_row((i - 150) / 50.0, 37 + (i % 6), "nlsy97") for i in range(300)])
    nlsy79.to_csv(processed / "nlsy79_cfa_resid.csv", index=False)
    nlsy97.to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    detail, summary = module.run_nonlinear_threshold_outcome_models(
        root=root,
        min_n_continuous=80,
        min_class_n_binary=20,
        threshold_quantile=0.8,
    )

    assert (root / "outputs" / "tables" / "nonlinear_threshold_outcome_models.csv").exists()
    assert (root / "outputs" / "tables" / "nonlinear_threshold_outcome_summary.csv").exists()
    assert "computed" in set(detail["status"])
    assert "computed" in set(summary["status"])
    assert {"nlsy79", "nlsy97"} <= set(summary["cohort"])


def test_run_nonlinear_threshold_outcome_models_handles_missing_sources(tmp_path: Path) -> None:
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
    detail, summary = module.run_nonlinear_threshold_outcome_models(root=root)
    assert set(detail["status"]) == {"not_feasible"}
    assert set(summary["status"]) == {"not_feasible"}
