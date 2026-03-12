from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "44_build_asvab_life_outcomes.py"
    spec = importlib.util.spec_from_file_location("script44_build_asvab_life_outcomes", path)
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


def test_run_asvab_life_outcomes_computes_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(root / "config" / "nlsy79.yml", "sample_construct:\n  sex_col: sex\n  subtests: [AR, WK]\n")

    rows: list[dict[str, object]] = []
    for i in range(35):
        g_base = float(i + 1)
        rows.append(
            {
                "sex": 1,
                "AR": g_base + 1.0,
                "WK": g_base + 1.0,
                "employment_status": 0.50 + 0.02 * g_base,
                "hourly_wage": 8.0 + 0.30 * g_base,
                "education_years": 10.0 + 0.05 * g_base,
                "health_index": 40.0 + 0.20 * g_base,
            }
        )
        rows.append(
            {
                "sex": 2,
                "AR": g_base,
                "WK": g_base + 0.5,
                "employment_status": 0.45 + 0.015 * g_base,
                "hourly_wage": 7.5 + 0.24 * g_base,
                "education_years": 10.0 + 0.04 * g_base,
                "health_index": 39.0 + 0.15 * g_base,
            }
        )
    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", rows)

    out = module.run_asvab_life_outcomes(
        root=root,
        cohorts=["nlsy79"],
        output_path=Path("outputs/tables/asvab_life_outcomes_by_sex.csv"),
    )

    assert out.shape[0] == 4
    assert set(out["status"]) == {"computed"}
    assert set(out["outcome"]) == {"employment", "wages", "education", "health"}
    assert (root / "outputs" / "tables" / "asvab_life_outcomes_by_sex.csv").exists()


def test_run_asvab_life_outcomes_handles_missing_outcomes(tmp_path: Path) -> None:
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

    out = module.run_asvab_life_outcomes(
        root=root,
        cohorts=["nlsy79"],
        output_path=Path("outputs/tables/asvab_life_outcomes_by_sex.csv"),
    )

    assert out.shape[0] == 4
    assert set(out["status"]) == {"not_feasible"}
    assert set(out["reason"]) == {"missing_outcome_column"}
