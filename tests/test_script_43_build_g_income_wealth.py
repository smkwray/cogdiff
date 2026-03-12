from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "43_build_g_income_wealth.py"
    spec = importlib.util.spec_from_file_location("script43_build_g_income_wealth", path)
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


def test_run_g_income_wealth_computes_effects(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(root / "config" / "nlsy79.yml", "sample_construct:\n  sex_col: sex\n  subtests: [AR, WK]\n")

    rows: list[dict[str, object]] = []
    for i in range(40):
        g_base = float(i + 1)
        rows.append(
            {
                "sex": 1,
                "AR": g_base + 1.0,
                "WK": g_base + 1.5,
                "annual_earnings": 20000 + 1200 * g_base,
                "household_income": 30000 + 1400 * g_base,
                "net_worth": 10000 + 1600 * g_base,
                "education_years": 10 + 0.1 * g_base,
            }
        )
        rows.append(
            {
                "sex": 2,
                "AR": g_base,
                "WK": g_base + 0.5,
                "annual_earnings": 18000 + 1000 * g_base,
                "household_income": 28000 + 1200 * g_base,
                "net_worth": 8000 + 1300 * g_base,
                "education_years": 10 + 0.08 * g_base,
            }
        )
    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", rows)

    out = module.run_g_income_wealth(
        root=root,
        cohorts=["nlsy79"],
        output_path=Path("outputs/tables/g_income_wealth_associations.csv"),
    )

    assert out.shape[0] == 3
    assert set(out["outcome"]) == {"earnings", "household_income", "net_worth"}
    assert set(out["status"]) == {"computed"}
    assert (root / "outputs" / "tables" / "g_income_wealth_associations.csv").exists()


def test_run_g_income_wealth_handles_missing_outcomes(tmp_path: Path) -> None:
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
            {"sex": 2, "AR": 0.5},
            {"sex": 1, "AR": 1.2},
            {"sex": 2, "AR": 0.7},
        ],
    )

    out = module.run_g_income_wealth(
        root=root,
        cohorts=["nlsy79"],
        output_path=Path("outputs/tables/g_income_wealth_associations.csv"),
    )

    assert out.shape[0] == 3
    assert set(out["status"]) == {"not_feasible"}
    assert set(out["reason"]) == {"missing_outcome_column"}
