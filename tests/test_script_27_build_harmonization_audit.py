from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "27_build_harmonization_audit.py"
    spec = importlib.util.spec_from_file_location("stage27_harmonization_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_harmonization_audit_computes_deltas_and_flags(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path
    tables_dir = root / "outputs" / "tables"

    _write_csv(
        tables_dir / "g_mean_diff_signed_merge.csv",
        [
            {"cohort": "nlsy97", "d_g": 0.20},
            {"cohort": "nlsy79", "d_g": 0.03},
        ],
    )
    _write_csv(
        tables_dir / "g_variance_ratio_signed_merge.csv",
        [
            {"cohort": "nlsy97", "VR_g": 1.20},
            {"cohort": "nlsy79", "VR_g": 1.10},
        ],
    )
    _write_csv(
        tables_dir / "g_mean_diff_zscore_by_branch.csv",
        [
            {"cohort": "nlsy97", "d_g": 0.25},
        ],
    )
    _write_csv(
        tables_dir / "g_variance_ratio_zscore_by_branch.csv",
        [
            {"cohort": "nlsy97", "VR_g": 1.40},
        ],
    )

    summary = module.run_harmonization_audit(
        project_root_path=root,
        cohort="nlsy97",
        baseline_token="signed_merge",
        alternate_token="zscore_by_branch",
        d_g_abs_threshold=0.1,
        log_vr_abs_threshold=0.1,
        output_path=tables_dir / "harmonization_audit_summary.csv",
    )

    assert summary.shape == (1, 15)
    row = summary.iloc[0]
    assert row["cohort"] == "nlsy97"
    assert row["audit_status"] == "complete"
    assert float(row["delta_d_g_abs"]) == pytest.approx(0.05)
    assert bool(row["d_g_abs_pass"]) is True
    assert float(row["delta_log_vr_abs"]) > 0.15
    assert bool(row["log_vr_abs_pass"]) is False


def test_run_harmonization_audit_handles_missing_inputs(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path
    tables_dir = root / "outputs" / "tables"

    _write_csv(
        tables_dir / "g_mean_diff_signed_merge.csv",
        [{"cohort": "nlsy97", "d_g": 0.20}],
    )
    _write_csv(
        tables_dir / "g_variance_ratio_signed_merge.csv",
        [{"cohort": "nlsy97", "VR_g": 1.20}],
    )
    _write_csv(
        tables_dir / "g_mean_diff_zscore_by_branch.csv",
        [{"cohort": "nlsy97", "d_g": 0.23}],
    )

    summary = module.run_harmonization_audit(
        project_root_path=root,
        cohort="nlsy97",
        baseline_token="signed_merge",
        alternate_token="zscore_by_branch",
        d_g_abs_threshold=0.1,
        log_vr_abs_threshold=0.1,
        output_path=tables_dir / "harmonization_audit_summary.csv",
    )

    row = summary.iloc[0]
    assert summary.shape[0] == 1
    assert row["audit_status"] == "partial"
    assert float(row["delta_d_g_abs"]) == pytest.approx(0.03)
    assert bool(row["d_g_abs_pass"]) is True
    assert pd.isna(row["delta_log_vr_abs"])
    assert bool(row["log_vr_abs_pass"]) is False
