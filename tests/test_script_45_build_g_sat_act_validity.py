from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "45_build_g_sat_act_validity.py"
    spec = importlib.util.spec_from_file_location("script45_build_g_sat_act_validity", path)
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


def test_script_45_computes_validity_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "hierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: []\n  technical: []\n",
    )
    _write(root / "config" / "nlsy97.yml", "sample_construct:\n  sex_col: sex\n")

    rows: list[dict[str, object]] = []
    for i in range(30):
        rows.append(
            {
                "sex": 1 if i % 2 == 0 else 2,
                "AR": float(i),
                "WK": float(i + 1),
                "sat_math_2007_bin": 1 + (i % 6),
                "sat_verbal_2007_bin": 1 + (i % 6),
                "act_2007_bin": 1 + (i % 6),
            }
        )
    _write_csv(root / "data" / "processed" / "nlsy97_cfa_resid.csv", rows)

    out = module.run_g_sat_act_validity(root=root, cohorts=["nlsy97"], output_path=Path("outputs/tables/g_sat_act_validity.csv"))
    assert out.shape[0] == 3
    assert set(out["status"]) == {"computed"}
    assert (root / "outputs" / "tables" / "g_sat_act_validity.csv").exists()


def test_script_45_handles_missing_outcomes(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "hierarchical_factors:\n  speed: [AR]\n  math: []\n  verbal: []\n  technical: []\n",
    )
    _write(root / "config" / "nlsy97.yml", "sample_construct:\n  sex_col: sex\n")
    _write_csv(
        root / "data" / "processed" / "nlsy97_cfa_resid.csv",
        [
            {"sex": 1, "AR": 1.0},
            {"sex": 2, "AR": 2.0},
            {"sex": 1, "AR": 3.0},
            {"sex": 2, "AR": 4.0},
        ],
    )

    out = module.run_g_sat_act_validity(root=root, cohorts=["nlsy97"], output_path=Path("outputs/tables/g_sat_act_validity.csv"))
    assert out.shape[0] == 3
    assert set(out["status"]) == {"not_feasible"}
    assert set(out["reason"]) == {"missing_outcome_column"}

