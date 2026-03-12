from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "36_build_cohort_trends.py"
    spec = importlib.util.spec_from_file_location("script36_build_cohort_trends", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_cohort_trends_computes_expected_positive_slope(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write_csv(
        root / "outputs" / "tables" / "analysis_tiers.csv",
        [
            {"cohort": "nlsy79", "estimand": "d_g", "analysis_tier": "confirmatory"},
            {"cohort": "nlsy97", "estimand": "d_g", "analysis_tier": "exploratory_sensitivity"},
            {"cohort": "cnlsy", "estimand": "d_g", "analysis_tier": "exploratory_sensitivity"},
            {"cohort": "nlsy79", "estimand": "vr_g", "analysis_tier": "confirmatory"},
            {"cohort": "nlsy97", "estimand": "vr_g", "analysis_tier": "confirmatory"},
            {"cohort": "cnlsy", "estimand": "vr_g", "analysis_tier": "confirmatory"},
        ],
    )
    _write_csv(
        root / "outputs" / "tables" / "g_mean_diff_full_cohort.csv",
        [
            {"cohort": "nlsy79", "d_g": 0.10, "SE_d_g": 0.05},
            {"cohort": "nlsy97", "d_g": 0.20, "SE_d_g": 0.05},
            {"cohort": "cnlsy", "d_g": 0.30, "SE_d_g": 0.05},
        ],
    )
    _write_csv(
        root / "outputs" / "tables" / "g_variance_ratio.csv",
        [
            {"cohort": "nlsy79", "VR_g": 1.10, "SE_logVR": 0.05},
            {"cohort": "nlsy97", "VR_g": 1.20, "SE_logVR": 0.05},
            {"cohort": "cnlsy", "VR_g": 1.30, "SE_logVR": 0.05},
        ],
    )

    trends, pairwise = module.run_cohort_trends(
        root=root,
        cohorts=["nlsy79", "nlsy97", "cnlsy"],
        trend_output_path=Path("outputs/tables/cohort_trends_sex_differences.csv"),
        pairwise_output_path=Path("outputs/tables/cohort_pairwise_diffs_sex_differences.csv"),
    )

    d_all = trends[(trends["estimand"] == "d_g") & (trends["scope"] == "all_cohorts")].iloc[0]
    assert d_all["status"] == "computed"
    assert float(d_all["slope_per_cohort_step"]) == pytest.approx(0.1)
    assert float(d_all["delta_first_to_last"]) == pytest.approx(0.2)
    assert float(d_all["delta_iq_points"]) == pytest.approx(3.0)

    d_confirm = trends[(trends["estimand"] == "d_g") & (trends["scope"] == "confirmatory_only")].iloc[0]
    assert d_confirm["status"] == "not_feasible"
    assert d_confirm["reason"] == "insufficient_cohorts"
    assert int(d_confirm["n_cohorts"]) == 1

    vr_confirm = trends[(trends["estimand"] == "vr_g") & (trends["scope"] == "confirmatory_only")].iloc[0]
    assert vr_confirm["status"] == "computed"
    assert int(vr_confirm["n_cohorts"]) == 3

    assert pairwise.shape[0] == 9
    assert (root / "outputs" / "tables" / "cohort_trends_sex_differences.csv").exists()
    assert (root / "outputs" / "tables" / "cohort_pairwise_diffs_sex_differences.csv").exists()


def test_run_cohort_trends_handles_missing_sources(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    trends, pairwise = module.run_cohort_trends(
        root=root,
        cohorts=["nlsy79"],
        trend_output_path=Path("outputs/tables/cohort_trends_sex_differences.csv"),
        pairwise_output_path=Path("outputs/tables/cohort_pairwise_diffs_sex_differences.csv"),
    )

    assert trends.shape[0] == 4
    assert set(trends["status"]) == {"not_feasible"}
    d_all = trends[(trends["estimand"] == "d_g") & (trends["scope"] == "all_cohorts")].iloc[0]
    assert "missing_g_mean_diff_source" in str(d_all["reason"])
    vr_all = trends[(trends["estimand"] == "vr_g") & (trends["scope"] == "all_cohorts")].iloc[0]
    assert "missing_g_variance_ratio_source" in str(vr_all["reason"])
    assert pairwise.empty
