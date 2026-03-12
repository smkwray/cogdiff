from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "50_build_cross_cohort_validity_and_stability.py"
    spec = importlib.util.spec_from_file_location("script50_build_cross_cohort_validity_and_stability", path)
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


def test_script_50_computes_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config" / "models.yml", "hierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n")
    _write(root / "config" / "nlsy79.yml", "sample_construct:\n  sex_col: sex\n")
    _write(root / "config" / "nlsy97.yml", "sample_construct:\n  sex_col: sex\n")
    _write(root / "config" / "cnlsy.yml", "sample_construct:\n  sex_col: sex\n")

    common_rows = [{"AR": float(i), "WK": float(i + 1), "household_income": float(20000 + 100 * i), "net_worth": float(5000 + 20 * i), "education_years": float(10 + (i % 6))} for i in range(60)]
    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", common_rows)
    _write_csv(root / "data" / "processed" / "nlsy97_cfa_resid.csv", common_rows)
    _write_csv(root / "data" / "processed" / "cnlsy_cfa_resid.csv", [{"PPVT": float(i), "education_years": float(8 + (i % 5))} for i in range(60)])

    _write_csv(root / "outputs" / "tables" / "g_mean_diff.csv", [{"cohort": "nlsy79", "estimate": 0.3}, {"cohort": "nlsy97", "estimate": 0.2}, {"cohort": "cnlsy", "estimate": 0.1}])
    _write_csv(root / "outputs" / "tables" / "g_variance_ratio.csv", [{"cohort": "nlsy79", "estimate": 1.3}, {"cohort": "nlsy97", "estimate": 1.2}, {"cohort": "cnlsy", "estimate": 1.1}])

    assoc_df, contrast_df, stability_df = module.run_cross_cohort_suite(root=root, cohorts=["nlsy79", "nlsy97", "cnlsy"])
    assert (assoc_df["status"] == "computed").any()
    assert (contrast_df["status"] == "computed").any()
    assert (stability_df["status"] == "computed").all()
