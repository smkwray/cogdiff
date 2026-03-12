from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "48_build_subtest_predictive_validity.py"
    spec = importlib.util.spec_from_file_location("script48_build_subtest_predictive_validity", path)
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


def test_script_48_computes_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config" / "models.yml", "hierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: [PC]\n  technical: []\ncnlsy_single_factor: [PPVT]\n")
    _write(root / "config" / "nlsy97.yml", "sample_construct:\n  sex_col: sex\n")

    rows: list[dict[str, object]] = []
    for i in range(80):
        rows.append(
            {
                "AR": float(i),
                "WK": float(i + 1),
                "PC": float(i + 2),
                "sat_math_2007_bin": 1 + (i % 6),
                "household_income": float(20000 + 100 * i),
                "education_years": float(10 + (i % 6)),
            }
        )
    _write_csv(root / "data" / "processed" / "nlsy97_cfa_resid.csv", rows)

    out = module.run_subtest_predictive_validity(root=root, cohorts=["nlsy97"], min_n=20)
    assert (out["status"] == "computed").any()
    assert {"subtest", "factor"} <= set(out["predictor_type"])

