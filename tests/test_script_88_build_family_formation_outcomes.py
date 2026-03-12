from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "88_build_family_formation_outcomes.py"
    spec = importlib.util.spec_from_file_location("script88_build_family_formation_outcomes", path)
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


def test_nlsy79_family(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy79.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    rows = []
    for i in range(150):
        g = (i - 75) / 35.0  # compressed range for class overlap
        noise = (((i * 7) % 11) - 5) / 5.0  # amplified noise

        # Higher g → more likely married (with overlap near threshold)
        if (g + noise) > 0.0:
            marital_status = 1  # married
        else:
            marital_status = 0  # never married

        # Divorced: among married, every 3rd (orthogonal to g for convergence)
        if marital_status == 1 and i % 3 == 0:
            marital_status = 3  # divorced

        # age_first_marriage_2000: 20 + int(g * 2 + 5) for married, -999 for never married
        if marital_status in (1, 3):
            afm = 20 + int(g * 2 + 5)
            afm = max(15, min(40, afm))
        else:
            afm = -999

        # num_children_2000: shift formula so ~30% have 0 children
        num_children = max(0, int(2 - g * 2 + noise))

        # age_first_birth_2000: 18 + int(g * 2 + 3) for those with children, -998 for no children
        if num_children > 0:
            afb = 18 + int(g * 2 + 3)
            afb = max(14, min(40, afb))
        else:
            afb = -998

        row = _subtests(i, g)
        row.update({
            "birth_year": 1960 + (i % 4),
            "age": 20 + (i % 8),
            "marital_status_2000": marital_status,
            "age_first_marriage_2000": afm,
            "num_children_2000": num_children,
            "age_first_birth_2000": afb,
        })
        rows.append(row)

    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    out = module.run_family_formation_outcomes(root=root, cohorts=["nlsy79"])
    computed = out[out["status"] == "computed"]
    computed_names = set(computed["outcome"])

    # Binary outcomes
    binary_expected = {"ever_married", "ever_divorced", "has_children"}
    assert binary_expected.issubset(computed_names), (
        f"Expected binary outcomes {binary_expected} computed, got {computed_names}"
    )

    # Continuous outcomes
    continuous_expected = {"num_children", "age_first_marriage", "age_first_birth"}
    assert continuous_expected.issubset(computed_names), (
        f"Expected continuous outcomes {continuous_expected} computed, got {computed_names}"
    )

    # Total computed == 6
    assert len(computed_names) == 6, (
        f"Expected 6 computed outcomes, got {len(computed_names)}: {computed_names}"
    )


def test_nlsy97_family(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy97.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    rows = []
    for i in range(130):
        g = (i - 65) / 30.0  # compressed range for class overlap
        noise = (((i * 3) % 7) - 3) / 4.0  # amplified noise

        # marital_status_cumulative: married if g+noise > 0, else never married (with overlap)
        if (g + noise) > 0.0:
            marital_status = 1
        else:
            marital_status = 0

        # Divorced: among married, every 3rd (orthogonal to g for convergence)
        if marital_status == 1 and i % 3 == 0:
            marital_status = 3

        # num_marriages: 1 for married, 0 for never married, 2 for some divorced
        if marital_status == 1:
            num_marriages = 1
        elif marital_status == 3:
            num_marriages = 2
        else:
            num_marriages = 0

        # first_marriage_year: 2005 + int(g * 2) for married, NaN for never married
        if marital_status in (1, 3):
            first_marriage_year = 2005 + int(g * 2)
        else:
            first_marriage_year = float("nan")

        # num_bio_children: shift formula so ~30% have 0 children
        if i % 13 == 0:
            num_bio_children = float("nan")
        else:
            num_bio_children = max(0, int(2 - g * 2 + noise))

        # first_child_birth_year_2019: 2005 + int(g) for those with kids, NaN for none
        if num_bio_children is not None and not (isinstance(num_bio_children, float) and num_bio_children != num_bio_children) and num_bio_children > 0:
            first_child_birth_year = 2005 + int(g)
        else:
            first_child_birth_year = float("nan")

        # first_marriage_end_reason: 1 (divorced) for status 3, 2 (widowed) for married, NaN for never-married
        if marital_status == 3:
            first_marriage_end_reason = 1  # divorced
        elif marital_status == 1:
            first_marriage_end_reason = 2  # widowed — provides 0 class for ever_divorced
        else:
            first_marriage_end_reason = float("nan")

        row = _subtests(i, g)
        row.update({
            "birth_year": 1981 + (i % 4),
            "marital_status_cumulative": marital_status,
            "num_marriages": num_marriages,
            "first_marriage_year": first_marriage_year,
            "num_bio_children": num_bio_children,
            "first_child_birth_year_2019": first_child_birth_year,
            "first_marriage_end_reason": first_marriage_end_reason,
        })
        rows.append(row)

    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    out = module.run_family_formation_outcomes(root=root, cohorts=["nlsy97"])
    computed = out[out["status"] == "computed"]
    computed_names = set(computed["outcome"])

    # Binary outcomes
    binary_expected = {"ever_married", "ever_divorced", "has_children"}
    assert binary_expected.issubset(computed_names), (
        f"Expected binary outcomes {binary_expected} computed, got {computed_names}"
    )

    # Continuous outcomes
    continuous_expected = {"num_children", "age_first_marriage", "age_first_birth"}
    assert continuous_expected.issubset(computed_names), (
        f"Expected continuous outcomes {continuous_expected} computed, got {computed_names}"
    )


def test_missing_columns_handled(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy79.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")
    _write(root / "config/nlsy97.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    # CSV with subtests + age + birth_year only, no family formation columns
    rows = [_subtests(i, (i - 5) / 5.0) | {"age": 20 + i, "birth_year": 1982} for i in range(10)]
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame(rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    out = module.run_family_formation_outcomes(root=root, cohorts=["nlsy79", "nlsy97"])
    assert (out["status"] == "not_feasible").all()
