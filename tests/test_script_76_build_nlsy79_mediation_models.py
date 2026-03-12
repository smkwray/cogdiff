from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "76_build_nlsy79_mediation_models.py"
    spec = importlib.util.spec_from_file_location("script76_build_nlsy79_mediation_models", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _row(i: int) -> dict[str, float]:
    g = (i - 180.0) / 60.0
    age = 36.0 + float(i % 9)
    education = 12.0 + 1.5 * g + float(i % 3) * 0.3
    employment = 1.0 if g + 0.04 * (age - 40.0) > -0.7 else 0.0
    occ_code = 1000.0 if g > 0.1 else 2000.0
    job_zone = 5.0 if occ_code == 1000.0 else 2.0
    earnings = float(np.expm1(9.0 + 0.45 * g + 0.18 * education + 0.08 * job_zone + 0.20 * employment + 0.01 * age))
    income = float(np.expm1(9.3 + 0.40 * g + 0.16 * education + 0.10 * job_zone + 0.15 * employment + 0.01 * age))
    return {
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
        "education_years": education,
        "employment_2000": employment,
        "occupation_code_2000": occ_code,
        "annual_earnings": earnings,
        "household_income": income,
    }


def test_run_nlsy79_mediation_models_computes_outputs(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_row(i) for i in range(360)]).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve().parents[1] / "scripts" / "66_build_nlsy79_job_zone_complexity.py", scripts_dir / "66_build_nlsy79_job_zone_complexity.py")

    census_text = """Management Occupations 1000 11-1011
Sales Occupations 2000 41-1011
"""
    _write(root / "fixtures/census_codes.txt", census_text)
    _write(
        root / "fixtures/job_zones.txt",
        "O*NET-SOC Code\tJob Zone\n11-1011.00\t5\n41-1011.00\t2\n",
    )

    detail, summary = module.run_nlsy79_mediation_models(
        root=root,
        min_n=80,
        census_text_path=root / "fixtures/census_codes.txt",
        job_zones_source=str(root / "fixtures/job_zones.txt"),
    )

    assert (root / "outputs/tables/nlsy79_mediation_models.csv").exists()
    assert (root / "outputs/tables/nlsy79_mediation_summary.csv").exists()
    assert set(detail["status"]) == {"computed"}
    assert set(summary["status"]) == {"computed"}
    assert {"baseline", "plus_all_mediators"} <= set(summary["model"])
    full_rows = summary.loc[summary["model"].eq("plus_all_mediators")].copy()
    assert not full_rows.empty
    assert full_rows["pct_attenuation_g"].notna().all()


def test_run_nlsy79_mediation_models_handles_missing_sources(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve().parents[1] / "scripts" / "66_build_nlsy79_job_zone_complexity.py", scripts_dir / "66_build_nlsy79_job_zone_complexity.py")

    detail, summary = module.run_nlsy79_mediation_models(root=root, min_n=80)
    assert set(detail["status"]) == {"not_feasible"}
    assert set(summary["status"]) == {"not_feasible"}
