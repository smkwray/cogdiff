from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "59_build_employment_outcomes.py"
    spec = importlib.util.spec_from_file_location("script59_build_employment_outcomes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_employment_outcomes_computes_rows(tmp_path: Path) -> None:
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
    for i in range(80):
        g = (i - 40) / 10.0
        noise = (((i * 7) % 11) - 5) / 8.0
        employed = 1 if (0.7 * g + 0.05 * (36 + (i % 8)) + noise) > 1.8 else 0
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
                "employment_2000": employed,
                "age_2000": 36 + (i % 8),
            }
        )
    pd.DataFrame(nlsy79_rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    nlsy97_rows = []
    for i in range(90):
        g = (i - 45) / 12.0
        noise = (((i * 5) % 13) - 6) / 8.0
        employed = 1 if (0.6 * g + 0.06 * (27 + (i % 6)) + noise) > 1.9 else 0
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
                "employment_2011": employed,
                "age_2011": 27 + (i % 6),
            }
        )
    pd.DataFrame(nlsy97_rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    out = module.run_employment_outcomes(root=root, cohorts=["nlsy79", "nlsy97"])

    assert set(out["cohort"]) == {"nlsy79", "nlsy97"}
    assert set(out["status"]) == {"computed"}
    assert (out["odds_ratio_g"] > 1.0).all()
    assert (root / "outputs" / "tables" / "g_employment_outcomes.csv").exists()


def test_run_employment_outcomes_handles_missing_columns(tmp_path: Path) -> None:
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
    pd.DataFrame([{"GS": 1.0, "AR": 2.0}]).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    out = module.run_employment_outcomes(root=root, cohorts=["nlsy79"])
    assert out.loc[0, "status"] == "not_feasible"
    assert out.loc[0, "reason"] == "missing_employment_column"
