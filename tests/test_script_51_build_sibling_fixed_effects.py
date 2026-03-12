from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "51_build_sibling_fixed_effects.py"
    spec = importlib.util.spec_from_file_location("script51_build_sibling_fixed_effects", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_script_51_computes_within_family_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "hierarchical_factors:\n  speed: [GS, AR]\n  math: [MK]\n  verbal: [WK, PC]\n  technical: [NO]\ncnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n",
    )
    _write(root / "config" / "nlsy79.yml", "pair_rules:\n  relatedness_r: 0.5\n  relationship_path: Gen1Housemates\n")

    processed_rows: list[dict[str, object]] = []
    for family, base in [("fam1", 0.0), ("fam2", 3.0), ("fam3", 6.0)]:
        for offset, person_id in [(0.0, len(processed_rows) + 1), (1.0, len(processed_rows) + 2)]:
            g = base + offset
            processed_rows.append(
                {
                    "person_id": person_id,
                    "GS": g,
                    "AR": g + 0.2,
                    "WK": g + 0.4,
                    "PC": g + 0.6,
                    "NO": g + 0.8,
                    "MK": g + 1.0,
                    "education_years": 12 + (2 * g),
                    "household_income": 40 + (5 * g),
                    "net_worth": 100 + (8 * g),
                    "annual_earnings": 25 + (4 * g),
                }
            )
    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", processed_rows)

    link_rows = [
        {"SubjectTag": 1, "PartnerTag": 2, "R": 0.5, "RelationshipPath": "Gen1Housemates", "ExtendedID": 101, "pair_id": "1|2", "family_id": "fam1"},
        {"SubjectTag": 3, "PartnerTag": 4, "R": 0.5, "RelationshipPath": "Gen1Housemates", "ExtendedID": 102, "pair_id": "3|4", "family_id": "fam2"},
        {"SubjectTag": 5, "PartnerTag": 6, "R": 0.5, "RelationshipPath": "Gen1Housemates", "ExtendedID": 103, "pair_id": "5|6", "family_id": "fam3"},
    ]
    _write_csv(root / "data" / "interim" / "links" / "links79_links.csv", link_rows)

    out = module.run_sibling_fixed_effects(root=root, cohorts=["nlsy79"])
    assert len(out) == 4
    assert set(out["status"]) == {"computed"}
    assert set(out["outcome"]) == {"education", "household_income", "net_worth", "earnings"}
    assert (pd.to_numeric(out["n_families"], errors="coerce") == 3).all()
    assert (pd.to_numeric(out["beta_g_within"], errors="coerce") > 0).all()


def test_script_51_computes_cnlsy_rows_when_adult_outcomes_are_available(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config" / "models.yml", "cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n")
    _write(root / "config" / "cnlsy.yml", "pair_rules:\n  relatedness_r: 0.5\n  relationship_path: Gen2Siblings\n")

    processed_rows = [
        {"person_id": 101, "PPVT": 90, "PIAT_RR": 88, "PIAT_RC": 87, "education_years": 12, "annual_earnings": 20000},
        {"person_id": 102, "PPVT": 92, "PIAT_RR": 90, "PIAT_RC": 89, "education_years": 13, "annual_earnings": 24000},
        {"person_id": 201, "PPVT": 96, "PIAT_RR": 95, "PIAT_RC": 94, "education_years": 14, "annual_earnings": 32000},
        {"person_id": 202, "PPVT": 98, "PIAT_RR": 97, "PIAT_RC": 96, "education_years": 15, "annual_earnings": 36000},
    ]
    _write_csv(root / "data" / "processed" / "cnlsy_cfa_resid.csv", processed_rows)

    link_rows = [
        {"SubjectTag": 101, "PartnerTag": 102, "R": 0.5, "RelationshipPath": "Gen2Siblings", "ExtendedID": "10", "pair_id": "101|102", "family_id": "10"},
        {"SubjectTag": 201, "PartnerTag": 202, "R": 0.5, "RelationshipPath": "Gen2Siblings", "ExtendedID": "20", "pair_id": "201|202", "family_id": "20"},
    ]
    _write_csv(root / "data" / "interim" / "links" / "links_cnlsy_links.csv", link_rows)

    out = module.run_sibling_fixed_effects(root=root, cohorts=["cnlsy"])
    computed = out.loc[out["status"] == "computed"].copy()
    assert set(computed["outcome"]) == {"education", "earnings"}
    assert (pd.to_numeric(computed["n_families"], errors="coerce") == 2).all()
