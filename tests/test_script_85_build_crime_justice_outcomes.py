from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "85_build_crime_justice_outcomes.py"
    spec = importlib.util.spec_from_file_location("script85_build_crime_justice_outcomes", path)
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


def test_nlsy97_adult_outcomes(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy97.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    rows = []
    for i in range(120):
        g = (i - 60) / 15.0
        noise = (((i * 7) % 11) - 5) / 10.0
        # Lower g → more likely arrested/incarcerated
        # Mix event-history codes: use 1-98 for some rows, 99 for others,
        # so both branches of _binarize_event_history are exercised.
        if (g + noise) < -0.3:
            arrested = (i % 98) + 1  # cycles through 1-98
        else:
            arrested = 0
        if (g + noise) < -1.0:
            incarcerated = 99 if i % 2 == 0 else (i % 50) + 1
        else:
            incarcerated = 0
        row = _subtests(i, g)
        row.update({
            "arrest_status_2019_12": arrested,
            "incarc_status_2019_12": incarcerated,
            "birth_year": 1981 + (i % 4),
        })
        rows.append(row)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    out = module.run_crime_justice_outcomes(root=root, cohorts=["nlsy97"])
    adult_rows = out[out["outcome"].isin(["ever_arrested", "ever_incarcerated"])]
    computed = adult_rows[adult_rows["status"] == "computed"]
    computed_names = set(computed["outcome"])
    assert computed_names == {"ever_arrested", "ever_incarcerated"}, (
        f"Expected both adult outcomes computed, got {computed_names}"
    )
    # Higher g should predict lower odds of arrest/incarceration
    for _, row in computed.iterrows():
        assert row["odds_ratio_g"] < 1.0, f"{row['outcome']}: OR={row['odds_ratio_g']} should be < 1"


def test_nlsy97_selfreport_outcomes(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy97.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    rows = []
    for i in range(100):
        g = (i - 50) / 12.0
        noise = (((i * 3) % 7) - 3) / 6.0
        destroyed = 1 if (g + noise) < 0.0 else 0
        theft_u50 = 1 if (g + noise) < -0.5 else 0
        theft_o50 = 0
        attacked = 1 if (g + noise) < -0.2 else 0
        sold_drugs = 1 if (g + noise) < -0.8 else 0
        row = _subtests(i, g)
        row.update({
            "ever_destroyed_property": destroyed,
            "ever_theft_under50": theft_u50,
            "ever_theft_over50": theft_o50,
            "ever_attacked": attacked,
            "ever_sold_drugs": sold_drugs,
            "birth_year": 1981 + (i % 4),
        })
        rows.append(row)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    out = module.run_crime_justice_outcomes(root=root, cohorts=["nlsy97"])
    sr_rows = out[out["outcome"].str.endswith("_sr")]
    computed = sr_rows[sr_rows["status"] == "computed"]
    expected_sr = {"ever_destroyed_property_sr", "ever_theft_sr", "ever_attacked_sr", "ever_sold_drugs_sr"}
    computed_names = set(computed["outcome"])
    assert computed_names == expected_sr, (
        f"Expected all 4 self-report outcomes computed, got {computed_names}"
    )


def test_nlsy79_delinquency_composites(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy79.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    rows = []
    for i in range(100):
        g = (i - 50) / 12.0
        noise = (((i * 5) % 9) - 4) / 6.0
        # Lower g → more delinquency counts
        prop_dmg = max(0, int(3 - g - noise)) if (g + noise) < 1.5 else 0
        fighting = max(0, int(2 - g - noise)) if (g + noise) < 1.0 else 0
        shoplifting = max(0, int(2.5 - g - noise)) if (g + noise) < 1.2 else 0
        theft_u = 0
        theft_o = 0
        force = max(0, int(1 - g - noise)) if (g + noise) < 0.5 else 0
        threatened = max(0, int(2 - g - noise)) if (g + noise) < 1.0 else 0
        attacked = 0
        sold_mj = max(0, int(1.5 - g - noise)) if (g + noise) < 0.8 else 0
        sold_hard = 0
        auto_theft = 0
        burglary = 0
        fencing = 0
        row = _subtests(i, g)
        row.update({
            "delin_property_damage": prop_dmg,
            "delin_fighting": fighting,
            "delin_shoplifting": shoplifting,
            "delin_theft_under50": theft_u,
            "delin_theft_over50": theft_o,
            "delin_used_force": force,
            "delin_threatened": threatened,
            "delin_attacked": attacked,
            "delin_sold_marijuana": sold_mj,
            "delin_sold_hard_drugs": sold_hard,
            "delin_auto_theft": auto_theft,
            "delin_burglary": burglary,
            "delin_fencing": fencing,
            "age": 16 + (i % 8),
        })
        rows.append(row)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    out = module.run_crime_justice_outcomes(root=root, cohorts=["nlsy79"])
    computed = out[out["status"] == "computed"]
    expected_composites = {"any_property_crime", "any_violent_crime", "any_drug_offense"}
    computed_names = set(computed["outcome"])
    assert computed_names == expected_composites, (
        f"Expected all 3 delinquency composites computed, got {computed_names}"
    )
    # Higher g should predict lower odds of delinquency
    for _, row in computed.iterrows():
        assert row["odds_ratio_g"] < 1.0, f"{row['outcome']}: OR={row['odds_ratio_g']} should be < 1"


def test_missing_columns_handled(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", MODELS_YML)
    _write(root / "config/nlsy97.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")
    _write(root / "config/nlsy79.yml", "sample_construct:\n  missing_codes: [-1, -2, -3, -4, -5]\n")

    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    # CSV with subtests only, no crime columns
    rows = [_subtests(i, (i - 5) / 5.0) | {"age": 20 + i, "birth_year": 1982} for i in range(10)]
    pd.DataFrame(rows).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    out = module.run_crime_justice_outcomes(root=root, cohorts=["nlsy79", "nlsy97"])
    assert (out["status"] == "not_feasible").all()
    assert (root / "outputs" / "tables" / "g_crime_justice_outcomes.csv").exists()
