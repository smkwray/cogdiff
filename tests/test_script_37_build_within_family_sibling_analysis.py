from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "37_build_within_family_sibling_analysis.py"
    spec = importlib.util.spec_from_file_location("script37_build_within_family_sibling_analysis", path)
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


def test_run_within_family_sibling_analysis_computes_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(
        root / "config" / "nlsy79.yml",
        "pair_rules:\n  relatedness_r: 0.5\n  relationship_path: Gen1Housemates\nsample_construct:\n  id_col: person_id\n  sex_col: sex\n  subtests: [AR, WK]\n",
    )

    _write_csv(
        root / "data" / "processed" / "nlsy79_cfa_resid.csv",
        [
            {"person_id": 1, "sex": 1, "AR": 3.0, "WK": 3.0},
            {"person_id": 2, "sex": 2, "AR": 2.0, "WK": 2.0},
            {"person_id": 3, "sex": 1, "AR": 4.0, "WK": 4.0},
            {"person_id": 4, "sex": 2, "AR": 1.0, "WK": 1.0},
            {"person_id": 5, "sex": 1, "AR": 5.0, "WK": 5.0},
            {"person_id": 6, "sex": 2, "AR": 0.0, "WK": 0.0},
        ],
    )
    _write_csv(
        root / "data" / "interim" / "links" / "links79_links.csv",
        [
            {
                "SubjectTag": 1,
                "PartnerTag": 2,
                "R": 0.5,
                "RelationshipPath": "Gen1Housemates",
                "pair_id": "1|2",
                "family_id": "f1",
            },
            {
                "SubjectTag": 3,
                "PartnerTag": 4,
                "R": 0.5,
                "RelationshipPath": "Gen1Housemates",
                "pair_id": "3|4",
                "family_id": "f2",
            },
            {
                "SubjectTag": 5,
                "PartnerTag": 6,
                "R": 0.5,
                "RelationshipPath": "Gen1Housemates",
                "pair_id": "5|6",
                "family_id": "f3",
            },
        ],
    )

    summary, pairs = module.run_within_family_sibling_analysis(
        root=root,
        cohorts=["nlsy79"],
        summary_output_path=Path("outputs/tables/within_family_sibling_analysis.csv"),
        pairs_output_path=Path("outputs/tables/within_family_sibling_pairs.csv"),
    )

    assert summary.shape[0] == 1
    row = summary.iloc[0]
    assert row["status"] == "computed"
    assert int(row["n_pairs_opposite_sex"]) == 3
    assert float(row["within_pair_mean_diff"]) > 0.0
    assert float(row["within_pair_d_pooled"]) > 0.0
    assert float(row["between_d_g"]) > 0.0
    assert pairs.shape[0] == 3
    assert set(pairs["cohort"]) == {"nlsy79"}

    assert (root / "outputs" / "tables" / "within_family_sibling_analysis.csv").exists()
    assert (root / "outputs" / "tables" / "within_family_sibling_pairs.csv").exists()


def test_run_within_family_sibling_analysis_missing_links(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: []\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(
        root / "config" / "nlsy79.yml",
        "pair_rules:\n  relatedness_r: 0.5\n  relationship_path: Gen1Housemates\nsample_construct:\n  id_col: person_id\n  sex_col: sex\n  subtests: [AR]\n",
    )
    _write_csv(
        root / "data" / "processed" / "nlsy79_cfa_resid.csv",
        [
            {"person_id": 1, "sex": 1, "AR": 1.0},
            {"person_id": 2, "sex": 2, "AR": 0.0},
        ],
    )

    summary, pairs = module.run_within_family_sibling_analysis(
        root=root,
        cohorts=["nlsy79"],
        summary_output_path=Path("outputs/tables/within_family_sibling_analysis.csv"),
        pairs_output_path=Path("outputs/tables/within_family_sibling_pairs.csv"),
    )

    assert summary.shape[0] == 1
    row = summary.iloc[0]
    assert row["status"] == "not_feasible"
    assert row["reason"] == "missing_links_file"
    assert pairs.empty
