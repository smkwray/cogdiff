from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "30_check_inference_ci_coherence.py"
    spec = importlib.util.spec_from_file_location("script30_check_inference_ci_coherence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_inference_ci_coherence_checks_contains_and_skips_non_computed(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    input_path = root / "outputs/tables/robustness_inference.csv"
    output_path = root / "outputs/tables/inference_ci_coherence.csv"

    _write_csv(
        input_path,
        [
            {
                "cohort": "nlsy97",
                "inference_method": "robust_cluster",
                "estimate_type": "d_g",
                "status": "computed",
                "estimate": 0.20,
                "ci_low": 0.10,
                "ci_high": 0.30,
            },
            {
                "cohort": "nlsy79",
                "inference_method": "family_bootstrap",
                "estimate_type": "vr_g",
                "status": "computed",
                "estimate": 1.10,
                "ci_low": 0.90,
                "ci_high": 1.20,
            },
            {
                "cohort": "cnlsy",
                "inference_method": "robust_cluster",
                "estimate_type": "d_g",
                "status": "baseline_missing",
                "estimate": 0.40,
                "ci_low": 0.30,
                "ci_high": 0.50,
            },
        ],
    )

    diagnostics = module.run_inference_ci_coherence(
        project_root_path=root,
        input_path=input_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert list(diagnostics.columns) == list(module.OUTPUT_COLUMNS)
    assert diagnostics.shape[0] == 3
    row = diagnostics[diagnostics["cohort"] == "nlsy97"].iloc[0]
    assert str(row["cohort"]) == "nlsy97"
    assert bool(row["ci_contains_estimate"]) is True
    assert row["issue"] == "contains_estimate"
    skipped = diagnostics[diagnostics["cohort"] == "cnlsy"].iloc[0]
    assert str(skipped["issue"]).startswith("non_computed_status")
    assert pd.isna(skipped["ci_contains_estimate"])


def test_main_respects_fail_on_violation(tmp_path: Path, monkeypatch: object) -> None:
    module = _module()
    root = tmp_path.resolve()
    input_path = root / "outputs/tables/robustness_inference.csv"
    output_path = root / "outputs/tables/inference_ci_coherence.csv"
    _write_csv(
        input_path,
        [
            {
                "cohort": "nlsy79",
                "inference_method": "robust_cluster",
                "estimate_type": "d_g",
                "status": "computed",
                "estimate": 0.80,
                "ci_low": 0.10,
                "ci_high": 0.30,
            },
            {
                "cohort": "nlsy97",
                "inference_method": "robust_cluster",
                "estimate_type": "d_g",
                "status": "computed",
                "estimate": 0.20,
                "ci_low": 0.10,
                "ci_high": 0.30,
            },
        ],
    )

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "30_check_inference_ci_coherence.py",
            "--project-root",
            str(root),
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--fail-on-violation",
        ],
    )
    code = module.main()
    assert code == 1
    assert output_path.exists()
    diagnostics = pd.read_csv(output_path)
    assert diagnostics.loc[diagnostics["cohort"] == "nlsy79", "issue"].iloc[0] == "computed_outside_ci"
    assert str(diagnostics.loc[diagnostics["cohort"] == "nlsy79", "ci_contains_estimate"].iloc[0]).lower() == "false"


def test_main_allows_violations_without_fail_flag(tmp_path: Path, monkeypatch: object) -> None:
    module = _module()
    root = tmp_path.resolve()
    input_path = root / "outputs/tables/robustness_inference.csv"
    output_path = root / "outputs/tables/inference_ci_coherence.csv"
    _write_csv(
        input_path,
        [
            {
                "cohort": "nlsy79",
                "inference_method": "robust_cluster",
                "estimate_type": "d_g",
                "status": "computed",
                "estimate": 0.80,
                "ci_low": 0.10,
                "ci_high": 0.30,
            },
        ],
    )

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "30_check_inference_ci_coherence.py",
            "--project-root",
            str(root),
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
        ],
    )
    code = module.main()
    assert code == 0
    diagnostics = pd.read_csv(output_path)
    assert diagnostics.shape[0] == 1
    assert diagnostics["issue"].iloc[0] == "computed_outside_ci"
