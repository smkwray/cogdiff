from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "41_build_inference_drift_check.py"
    spec = importlib.util.spec_from_file_location("script41_build_inference_drift_check", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_inference_drift_check_computes_expected_deltas(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write_csv(
        root / "outputs" / "tables" / "g_mean_diff_full_cohort.csv",
        [
            {"cohort": "nlsy79", "d_g": 0.30},
            {"cohort": "nlsy97", "d_g": 0.05},
        ],
    )
    _write_csv(
        root / "outputs" / "tables" / "g_mean_diff_family_bootstrap.csv",
        [
            {"cohort": "nlsy79", "status": "computed", "reason": "", "d_g": 0.28},
            {"cohort": "nlsy97", "status": "computed", "reason": "", "d_g": 0.01},
        ],
    )
    _write_csv(
        root / "outputs" / "tables" / "g_variance_ratio.csv",
        [
            {"cohort": "nlsy79", "VR_g": 1.30},
            {"cohort": "nlsy97", "VR_g": 1.25},
        ],
    )
    _write_csv(
        root / "outputs" / "tables" / "g_variance_ratio_family_bootstrap.csv",
        [
            {"cohort": "nlsy79", "status": "computed", "reason": "", "VR_g": 1.28},
            {"cohort": "nlsy97", "status": "computed", "reason": "", "VR_g": 1.10},
        ],
    )

    out = module.run_inference_drift_check(
        root=root,
        max_abs_delta_d=0.05,
        max_abs_delta_log_vr=0.05,
        output_path=Path("outputs/tables/inference_drift_check.csv"),
    )

    assert out.shape[0] == 4
    assert set(out["status"]) == {"computed"}

    d79 = out[(out["cohort"] == "nlsy79") & (out["estimand"] == "d_g")].iloc[0]
    assert float(d79["delta_abs"]) == pytest.approx(0.02)
    assert bool(d79["pass"]) is True

    vr97 = out[(out["cohort"] == "nlsy97") & (out["estimand"] == "log_vr_g")].iloc[0]
    assert float(vr97["delta_abs"]) > 0.05
    assert bool(vr97["pass"]) is False

    assert (root / "outputs" / "tables" / "inference_drift_check.csv").exists()


def test_run_inference_drift_check_handles_bootstrap_not_feasible(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write_csv(
        root / "outputs" / "tables" / "g_mean_diff.csv",
        [
            {"cohort": "nlsy79", "d_g": 0.30},
        ],
    )
    _write_csv(
        root / "outputs" / "tables" / "g_mean_diff_family_bootstrap.csv",
        [
            {"cohort": "nlsy79", "status": "not_feasible", "reason": "timeout", "d_g": ""},
        ],
    )
    _write_csv(
        root / "outputs" / "tables" / "g_variance_ratio.csv",
        [
            {"cohort": "nlsy79", "VR_g": 1.30},
        ],
    )
    _write_csv(
        root / "outputs" / "tables" / "g_variance_ratio_family_bootstrap.csv",
        [
            {"cohort": "nlsy79", "status": "not_feasible", "reason": "timeout", "VR_g": ""},
        ],
    )

    out = module.run_inference_drift_check(
        root=root,
        output_path=Path("outputs/tables/inference_drift_check.csv"),
    )

    assert out.shape[0] == 2
    assert set(out["status"]) == {"not_feasible"}
    assert set(out["reason"]) == {"timeout"}
