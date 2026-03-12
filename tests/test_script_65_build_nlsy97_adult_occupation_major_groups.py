from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "65_build_nlsy97_adult_occupation_major_groups.py"
    spec = importlib.util.spec_from_file_location("script65_build_nlsy97_adult_occupation_major_groups", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_nlsy97_adult_occupation_major_groups_computes_outputs(tmp_path: Path) -> None:
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
    rows = []
    for i in range(180):
        g = (i - 90) / 18.0
        high_skill = 1 if (g > 0.35 or i % 5 in {0, 1}) else 0
        occ_2021 = 245 if high_skill else pd.NA
        occ_2019 = pd.NA if high_skill else 715
        if i % 3 == 0:
            occ_2021 = pd.NA
            occ_2019 = 245 if high_skill else 715
        rows.append(
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
                "education_years": 12 + (2.5 * high_skill) + (0.15 * g),
                "occupation_code_2011": pd.NA,
                "occupation_code_2013": pd.NA,
                "occupation_code_2015": pd.NA,
                "occupation_code_2017": pd.NA,
                "occupation_code_2019": occ_2019,
                "occupation_code_2021": occ_2021,
                "age_2011": pd.NA,
                "age_2013": pd.NA,
                "age_2015": pd.NA,
                "age_2017": pd.NA,
                "age_2019": 38 + (i % 4),
                "age_2021": 40 + (i % 4),
            }
        )
    pd.DataFrame(rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    summary, model = module.run_nlsy97_adult_occupation_major_groups(root=root, min_class_n=20)

    assert (root / "outputs" / "tables" / "nlsy97_adult_occupation_major_group_summary.csv").exists()
    assert (root / "outputs" / "tables" / "nlsy97_high_skill_occupation_outcome.csv").exists()
    assert "management_professional_related" in set(summary["occupation_group"])
    assert int(summary.loc[summary["occupation_group"] == "management_professional_related", "n_used"].iloc[0]) > 0
    assert model.loc[0, "status"] == "computed"
    assert float(model.loc[0, "odds_ratio_g"]) > 0.0
    assert int(model.loc[0, "n_with_any_occupation"]) == 180


def test_run_nlsy97_adult_occupation_major_groups_handles_missing_columns(tmp_path: Path) -> None:
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
    pd.DataFrame([{"GS": 1.0, "AR": 2.0}]).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    summary, model = module.run_nlsy97_adult_occupation_major_groups(root=root)
    assert set(summary["status"]) == {"not_feasible"}
    assert model.loc[0, "status"] == "not_feasible"
