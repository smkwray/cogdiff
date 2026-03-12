from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "87_build_mental_health_outcomes.py"
    spec = importlib.util.spec_from_file_location("script87_build_mental_health_outcomes", path)
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


def test_nlsy79_mental_health(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy79.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    rows = []
    for i in range(120):
        g = (i - 60) / 30.0
        noise = (((i * 7) % 11) - 5) / 5.0
        cesd = max(0, min(21, int(8 - g * 2 + noise)))
        rosenberg = min(30, max(6, int(15 + g * 2 + noise)))
        rotter = min(16, max(4, int(11 - g * 1.5 + noise)))
        row = _subtests(i, g)
        row.update({
            "birth_year": 1960 + (i % 4),
            "age": 20 + (i % 8),
            "cesd_score_2022": cesd,
            "rosenberg_self_esteem_1980": rosenberg,
            "rotter_locus_control_1979": rotter,
        })
        rows.append(row)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    out = module.run_mental_health_outcomes(root=root, cohorts=["nlsy79"])

    # Binary outcomes computed
    binary_computed = out[(out["status"] == "computed") & out["outcome"].isin(["depressed", "low_self_esteem", "external_locus"])]
    binary_names = set(binary_computed["outcome"])
    assert binary_names == {"depressed", "low_self_esteem", "external_locus"}, (
        f"Expected all 3 binary outcomes computed, got {binary_names}"
    )

    # Continuous outcomes computed
    continuous_computed = out[(out["status"] == "computed") & out["outcome"].isin(["cesd_score", "rosenberg_score", "rotter_score"])]
    continuous_names = set(continuous_computed["outcome"])
    assert continuous_names == {"cesd_score", "rosenberg_score", "rotter_score"}, (
        f"Expected all 3 continuous outcomes computed, got {continuous_names}"
    )

    # Higher g should predict lower odds of depression
    dep_row = binary_computed[binary_computed["outcome"] == "depressed"].iloc[0]
    assert dep_row["odds_ratio_g"] < 1.0, f"depressed: OR={dep_row['odds_ratio_g']} should be < 1"

    # Higher g should predict lower odds of low self-esteem
    lse_row = binary_computed[binary_computed["outcome"] == "low_self_esteem"].iloc[0]
    assert lse_row["odds_ratio_g"] < 1.0, f"low_self_esteem: OR={lse_row['odds_ratio_g']} should be < 1"

    # Higher g should predict lower odds of external locus of control
    el_row = binary_computed[binary_computed["outcome"] == "external_locus"].iloc[0]
    assert el_row["odds_ratio_g"] < 1.0, f"external_locus: OR={el_row['odds_ratio_g']} should be < 1"


def test_nlsy97_mental_health(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy97.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    rows = []
    for i in range(100):
        g = (i - 50) / 25.0
        noise = (((i * 3) % 7) - 3) / 4.0
        cesd = max(0, min(21, int(8 - g * 2 + noise)))
        row = _subtests(i, g)
        row.update({
            "birth_year": 1981 + (i % 4),
            "cesd_score_2023": cesd,
        })
        rows.append(row)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    out = module.run_mental_health_outcomes(root=root, cohorts=["nlsy97"])

    # Binary outcome computed
    binary_computed = out[(out["status"] == "computed") & (out["outcome"] == "depressed")]
    assert set(binary_computed["outcome"]) == {"depressed"}, (
        f"Expected depressed binary outcome computed, got {set(binary_computed['outcome'])}"
    )

    # Continuous outcome computed
    continuous_computed = out[(out["status"] == "computed") & (out["outcome"] == "cesd_score")]
    assert set(continuous_computed["outcome"]) == {"cesd_score"}, (
        f"Expected cesd_score continuous outcome computed, got {set(continuous_computed['outcome'])}"
    )

    # Higher g should predict lower odds of depression
    dep_row = binary_computed.iloc[0]
    assert dep_row["odds_ratio_g"] < 1.0, f"depressed: OR={dep_row['odds_ratio_g']} should be < 1"


def test_missing_columns_handled(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy97.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")
    _write(root / "config/nlsy79.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    # CSV with subtests only, no mental health columns
    rows = [_subtests(i, (i - 5) / 5.0) | {"age": 20 + i, "birth_year": 1982} for i in range(10)]
    pd.DataFrame(rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    out = module.run_mental_health_outcomes(root=root, cohorts=["nlsy79", "nlsy97"])
    assert (out["status"] == "not_feasible").all()
    assert (root / "outputs" / "tables" / "g_mental_health_outcomes.csv").exists()
