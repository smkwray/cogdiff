from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "35_build_tail_ratio_profiles.py"
    spec = importlib.util.spec_from_file_location("script35_build_tail_ratio_profiles", path)
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


def test_run_tail_ratio_profiles_computes_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(
        root / "config" / "nlsy79.yml",
        "sample_construct:\n  sex_col: sex\n  subtests: [AR, WK]\n",
    )

    rows: list[dict[str, object]] = []
    for i in range(20):
        rows.append({"sex": 1, "AR": float(i + 5), "WK": float(i + 5)})
    for i in range(20):
        rows.append({"sex": 2, "AR": float(i), "WK": float(i)})
    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", rows)

    out = module.run_tail_ratio_profiles(
        root=root,
        cohorts=["nlsy79"],
        quantiles=(0.95, 0.99),
        n_bootstrap=20,
        seed=17,
        output_path=Path("outputs/tables/tail_ratio_profiles.csv"),
    )

    assert out.shape[0] == 4
    assert int((out["status"] == "computed").sum()) >= 2
    computed = out[out["status"] == "computed"].iloc[0]
    assert float(computed["male_female_tail_rate_ratio"]) >= 0.0
    assert float(computed["ci_high"]) >= float(computed["ci_low"])
    assert int(computed["n_bootstrap_success"]) > 0

    output_file = root / "outputs" / "tables" / "tail_ratio_profiles.csv"
    assert output_file.exists()


def test_run_tail_ratio_profiles_handles_missing_source(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: []\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(
        root / "config" / "nlsy79.yml",
        "sample_construct:\n  sex_col: sex\n  subtests: [AR]\n",
    )

    out = module.run_tail_ratio_profiles(
        root=root,
        cohorts=["nlsy79"],
        quantiles=(0.95,),
        n_bootstrap=5,
        seed=5,
        output_path=Path("outputs/tables/tail_ratio_profiles.csv"),
    )

    assert out.shape[0] == 1
    row = out.iloc[0]
    assert row["status"] == "not_feasible"
    assert row["reason"] == "missing_source_data"
