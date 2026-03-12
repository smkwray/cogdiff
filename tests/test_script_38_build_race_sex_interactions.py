from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "38_build_race_sex_interactions.py"
    spec = importlib.util.spec_from_file_location("script38_build_race_sex_interactions", path)
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


def test_run_race_sex_interactions_computes_heterogeneity(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(root / "config" / "nlsy79.yml", "sample_construct:\n  sex_col: sex\n  subtests: [AR, WK]\n")

    rows: list[dict[str, object]] = []
    # Group A: larger male-female gap
    for i in range(8):
        rows.append({"sex": 1, "race_ethnicity": "A", "AR": float(10 + i), "WK": float(10 + i)})
    for i in range(8):
        rows.append({"sex": 2, "race_ethnicity": "A", "AR": float(3 + i), "WK": float(3 + i)})
    # Group B: near-zero gap
    for i in range(8):
        rows.append({"sex": 1, "race_ethnicity": "B", "AR": float(6 + i), "WK": float(6 + i)})
    for i in range(8):
        rows.append({"sex": 2, "race_ethnicity": "B", "AR": float(5.5 + i), "WK": float(5.5 + i)})
    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", rows)

    summary, detail = module.run_race_sex_interactions(
        root=root,
        cohorts=["nlsy79"],
        summary_output_path=Path("outputs/tables/race_sex_interaction_summary.csv"),
        detail_output_path=Path("outputs/tables/race_sex_group_estimates.csv"),
    )

    assert summary.shape[0] == 1
    s = summary.iloc[0]
    assert s["status"] == "computed"
    assert s["race_col"] == "race_ethnicity"
    assert int(s["n_groups_used"]) == 2
    assert float(s["heterogeneity_Q"]) >= 0.0
    assert 0.0 <= float(s["heterogeneity_p_value"]) <= 1.0

    d = detail[detail["status"] == "computed"].copy()
    assert d.shape[0] == 2
    assert set(d["race_group"]) == {"A", "B"}
    assert set(d["g_construct"]) == {"g_proxy"}
    assert (d["d_g_proxy"] == d["d_g"]).all()
    assert "mean_all" in d.columns
    assert "var_all" in d.columns
    assert d["mean_all"].notna().all()
    assert d["var_all"].notna().all()
    assert (d["var_all"].astype(float) > 0.0).all()
    a_d = float(d.loc[d["race_group"] == "A", "d_g"].iloc[0])
    b_d = float(d.loc[d["race_group"] == "B", "d_g"].iloc[0])
    assert a_d > b_d

    assert (root / "outputs" / "tables" / "race_sex_interaction_summary.csv").exists()
    assert (root / "outputs" / "tables" / "race_sex_group_estimates.csv").exists()


def test_run_race_sex_interactions_handles_missing_race_column(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: []\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(root / "config" / "nlsy79.yml", "sample_construct:\n  sex_col: sex\n  subtests: [AR]\n")
    _write_csv(
        root / "data" / "processed" / "nlsy79_cfa_resid.csv",
        [
            {"sex": 1, "AR": 1.0},
            {"sex": 2, "AR": 0.0},
            {"sex": 1, "AR": 2.0},
            {"sex": 2, "AR": 1.0},
        ],
    )

    summary, detail = module.run_race_sex_interactions(
        root=root,
        cohorts=["nlsy79"],
        summary_output_path=Path("outputs/tables/race_sex_interaction_summary.csv"),
        detail_output_path=Path("outputs/tables/race_sex_group_estimates.csv"),
    )

    assert summary.shape[0] == 1
    s = summary.iloc[0]
    assert s["status"] == "not_feasible"
    assert s["reason"] == "missing_race_column"
    assert detail.shape[0] == 1
    assert detail.iloc[0]["status"] == "not_feasible"
    assert detail.iloc[0]["reason"] == "missing_race_column"
