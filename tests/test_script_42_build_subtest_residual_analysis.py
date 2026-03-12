from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "42_build_subtest_residual_analysis.py"
    spec = importlib.util.spec_from_file_location("script42_build_subtest_residual_analysis", path)
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


def test_run_subtest_residual_analysis_reduces_common_factor_gap(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: [PC]\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(root / "config" / "nlsy79.yml", "sample_construct:\n  sex_col: sex\n  subtests: [AR, WK, PC]\n")

    rows: list[dict[str, object]] = []
    # Common latent-like shift by sex inflates raw subtest d; residualization should attenuate.
    for i in range(30):
        g = float(i) / 10.0
        rows.append({"sex": 1, "AR": 2.0 + g, "WK": 2.5 + g, "PC": 3.0 + g})
    for i in range(30):
        g = float(i) / 10.0
        rows.append({"sex": 2, "AR": 1.0 + g, "WK": 1.5 + g, "PC": 2.0 + g})
    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", rows)

    out = module.run_subtest_residual_analysis(
        root=root,
        cohorts=["nlsy79"],
        output_path=Path("outputs/tables/subtest_residual_analysis.csv"),
    )

    assert out.shape[0] == 3
    assert set(out["status"]) == {"computed"}
    ar = out[out["subtest"] == "AR"].iloc[0]
    assert float(ar["raw_d_subtest"]) > 0.0
    assert abs(float(ar["resid_d_subtest"])) < abs(float(ar["raw_d_subtest"]))
    assert "g_other" in str(ar["predictors_used"])
    assert (root / "outputs" / "tables" / "subtest_residual_analysis.csv").exists()


def test_run_subtest_residual_analysis_handles_missing_source(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: []\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(root / "config" / "nlsy79.yml", "sample_construct:\n  sex_col: sex\n  subtests: [AR]\n")

    out = module.run_subtest_residual_analysis(
        root=root,
        cohorts=["nlsy79"],
        output_path=Path("outputs/tables/subtest_residual_analysis.csv"),
    )

    assert out.shape[0] == 1
    row = out.iloc[0]
    assert row["status"] == "not_feasible"
    assert row["reason"] == "missing_source_data"
