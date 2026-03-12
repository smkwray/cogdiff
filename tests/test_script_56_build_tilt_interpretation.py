from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "56_build_tilt_interpretation.py"
    spec = importlib.util.spec_from_file_location("script56_build_tilt_interpretation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_script_56_summarizes_tilt_as_small_add_on_when_incremental_r2_is_tiny(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write_csv(
        root / "outputs" / "tables" / "subtest_profile_tilt.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "d_tilt": -0.4,
                "se_d_tilt": 0.02,
                "tilt_g_corr": -0.01,
                "tilt_incremental_r2_education": 0.0024,
                "p_tilt_incremental": 0.0001,
            }
        ],
    )
    _write_csv(
        root / "outputs" / "tables" / "g_proxy_mean_diff_family_bootstrap.csv",
        [{"cohort": "nlsy79", "status": "computed", "d_g": 0.2}],
    )

    out = module.run_tilt_interpretation(root=root)
    assert list(out["cohort"]) == ["nlsy79"]
    assert out.loc[0, "incremental_r2_band"] == "very_small"
    assert out.loc[0, "tilt_vs_g_band"] == "larger_than_g"
    assert out.loc[0, "interpretation"] == "small_add_on"


def test_script_56_marks_missing_g_proxy_rows_as_not_feasible(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write_csv(
        root / "outputs" / "tables" / "subtest_profile_tilt.csv",
        [{"cohort": "nlsy97", "status": "computed", "d_tilt": -0.1, "tilt_incremental_r2_education": 0.0}],
    )
    _write_csv(
        root / "outputs" / "tables" / "g_proxy_mean_diff_family_bootstrap.csv",
        [{"cohort": "nlsy79", "status": "computed", "d_g": 0.2}],
    )

    out = module.run_tilt_interpretation(root=root)
    assert out.loc[0, "status"] == "not_feasible"
    assert out.loc[0, "reason"] == "missing_g_proxy_row"
