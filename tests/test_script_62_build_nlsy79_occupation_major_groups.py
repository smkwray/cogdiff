from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "62_build_nlsy79_occupation_major_groups.py"
    spec = importlib.util.spec_from_file_location("script62_build_nlsy79_occupation_major_groups", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_nlsy79_occupation_major_groups_computes_outputs(tmp_path: Path) -> None:
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
    codes = [245, 280, 372, 441, 520, 715, 984]
    for i in range(140):
        g = (i - 70) / 15.0
        code = codes[i % len(codes)]
        high_skill = 1 if code in {245, 280} else 0
        if i < 60:
            code = 245 if i % 2 == 0 else 280
            high_skill = 1
        elif i < 120:
            code = 715 if i % 2 == 0 else 984
            high_skill = 0
        else:
            high_skill = 1 if g > 0 else 0
            code = 245 if high_skill == 1 else 715
        employed_income = 30000 + (9000 * high_skill) + (2500 * g)
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
                "occupation_code_2000": code,
                "age_2000": 36 + (i % 8),
                "education_years": 12 + (3 * high_skill) + (0.2 * g),
                "household_income": employed_income,
                "net_worth": 5000 + (4000 * high_skill) + (600 * g),
            }
        )
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    summary, model = module.run_nlsy79_occupation_major_groups(root=root, min_class_n=20)

    assert (root / "outputs" / "tables" / "nlsy79_occupation_major_group_summary.csv").exists()
    assert (root / "outputs" / "tables" / "nlsy79_high_skill_occupation_outcome.csv").exists()
    assert "management_professional_related" in set(summary["occupation_group"])
    assert int(summary.loc[summary["occupation_group"] == "management_professional_related", "n_used"].iloc[0]) > 0
    assert model.loc[0, "status"] == "computed"
    assert float(model.loc[0, "odds_ratio_g"]) > 0.0
    assert 0.0 < float(model.loc[0, "prevalence"]) < 1.0


def test_run_nlsy79_occupation_major_groups_handles_missing_columns(tmp_path: Path) -> None:
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

    summary, model = module.run_nlsy79_occupation_major_groups(root=root)
    assert set(summary["status"]) == {"not_feasible"}
    assert model.loc[0, "status"] == "not_feasible"
