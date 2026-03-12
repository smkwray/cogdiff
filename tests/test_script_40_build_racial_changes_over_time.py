from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "40_build_racial_changes_over_time.py"
    spec = importlib.util.spec_from_file_location("script40_build_racial_changes_over_time", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_racial_changes_over_time_computes_group_slopes(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write_csv(
        root / "outputs" / "tables" / "race_sex_group_estimates.csv",
        [
            {"cohort": "nlsy79", "race_group": "A", "status": "computed", "d_g": 0.30, "SE_d_g": 0.05},
            {"cohort": "nlsy97", "race_group": "A", "status": "computed", "d_g": 0.20, "SE_d_g": 0.05},
            {"cohort": "cnlsy", "race_group": "A", "status": "computed", "d_g": 0.10, "SE_d_g": 0.05},
            {"cohort": "nlsy79", "race_group": "B", "status": "computed", "d_g": 0.10, "SE_d_g": 0.05},
            {"cohort": "nlsy97", "race_group": "B", "status": "computed", "d_g": 0.20, "SE_d_g": 0.05},
            {"cohort": "cnlsy", "race_group": "B", "status": "computed", "d_g": 0.30, "SE_d_g": 0.05},
        ],
    )

    out = module.run_racial_changes_over_time(
        root=root,
        source_detail_path=Path("outputs/tables/race_sex_group_estimates.csv"),
        output_path=Path("outputs/tables/racial_changes_over_time.csv"),
    )

    assert out.shape[0] == 2
    assert set(out["status"]) == {"computed"}
    a = out[out["race_group"] == "A"].iloc[0]
    b = out[out["race_group"] == "B"].iloc[0]
    assert float(a["slope_per_cohort_step"]) == pytest.approx(-0.1)
    assert float(b["slope_per_cohort_step"]) == pytest.approx(0.1)
    assert int(a["n_cohorts"]) == 3
    assert int(b["n_cohorts"]) == 3
    assert (root / "outputs" / "tables" / "racial_changes_over_time.csv").exists()


def test_run_racial_changes_over_time_handles_no_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write_csv(
        root / "outputs" / "tables" / "race_sex_group_estimates.csv",
        [
            {"cohort": "nlsy79", "race_group": "", "status": "not_feasible", "d_g": "", "SE_d_g": ""},
        ],
    )

    out = module.run_racial_changes_over_time(
        root=root,
        source_detail_path=Path("outputs/tables/race_sex_group_estimates.csv"),
        output_path=Path("outputs/tables/racial_changes_over_time.csv"),
    )

    assert out.shape[0] == 1
    row = out.iloc[0]
    assert row["status"] == "not_feasible"
    assert row["reason"] == "no_computed_rows_in_source_detail"
