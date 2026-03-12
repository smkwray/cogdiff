from __future__ import annotations

import sys
from pathlib import Path
import subprocess

import pandas as pd


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_script(project_root: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "17_generate_robustness_variant.py"
    return subprocess.run(
        [sys.executable, str(script.resolve())] + args,
        cwd=project_root,
        check=check,
        capture_output=True,
        text=True,
    )


def _project_root_with_outputs(project_root: Path) -> Path:
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    return project_root


def _assert_csv_rows_match(expected: pd.DataFrame, actual_path: Path) -> None:
    actual = pd.read_csv(actual_path)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        actual.reset_index(drop=True),
        check_like=False,
        check_dtype=False,
    )


def test_script_17_generates_sampling_variants(tmp_path: Path) -> None:
    project_root = _project_root_with_outputs(tmp_path)
    tables = project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    sample_counts = pd.DataFrame(
        [
            {"cohort": "nlsy79", "n_input": 100, "n_after_age": 80, "n_after_test_rule": 60, "n_after_dedupe": 58},
            {"cohort": "nlsy97", "n_input": 120, "n_after_age": 95, "n_after_test_rule": 75, "n_after_dedupe": 70},
        ]
    )
    g_mean = pd.DataFrame(
        [
            {"cohort": "nlsy79", "d_g": 0.2, "SE": 0.08, "ci_low": 0.05, "ci_high": 0.35},
            {"cohort": "nlsy97", "d_g": 0.1, "SE": 0.07, "ci_low": -0.02, "ci_high": 0.22},
        ]
    )
    sample_counts.to_csv(tables / "sample_counts.csv", index=False)
    g_mean.to_csv(tables / "g_mean_diff.csv", index=False)

    result = _run_script(project_root, ["--project-root", str(project_root), "--family", "sampling", "--variant-token", "sibling_restricted"])
    assert result.returncode == 0

    _assert_csv_rows_match(sample_counts, tables / "sample_counts_sibling_restricted.csv")
    _assert_csv_rows_match(g_mean, tables / "g_mean_diff_sibling_restricted.csv")


def test_script_17_generates_age_adjustment_variant(tmp_path: Path) -> None:
    project_root = _project_root_with_outputs(tmp_path)
    tables = project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    age = pd.DataFrame(
        [
            {"cohort": "nlsy79", "subtest": "GS", "r2": 0.91, "resid_sd": 1.1},
            {"cohort": "nlsy97", "subtest": "GS", "r2": 0.86, "resid_sd": 0.9},
        ]
    )
    age.to_csv(tables / "residualization_diagnostics_all.csv", index=False)

    result = _run_script(
        project_root,
        [
            "--project-root",
            str(project_root),
            "--family",
            "age_adjustment",
            "--variant-token",
            "cubic_within_sex",
            "--cohort",
            "nlsy79",
        ],
    )
    assert result.returncode == 0

    output = pd.read_csv(tables / "residualization_diagnostics_nlsy79_cubic_within_sex.csv")
    assert set(output["cohort"]) == {"nlsy79"}


def test_script_17_generates_model_form_variant_for_cohort(tmp_path: Path) -> None:
    project_root = _project_root_with_outputs(tmp_path)
    tables = project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    invariance = pd.DataFrame(
        {
            "cohort": ["nlsy79", "nlsy79"],
            "model_step": ["configural", "strict"],
            "cfi": [0.95, 0.93],
            "rmsea": [0.04, 0.05],
            "srmr": [0.03, 0.035],
            "delta_cfi": [0.0, 0.0],
        }
    )
    invariance.to_csv(tables / "nlsy79_invariance_summary.csv", index=False)

    result = _run_script(
        project_root,
        ["--project-root", str(project_root), "--family", "model_form", "--variant-token", "single_factor_alt", "--cohort", "nlsy79"],
    )
    assert result.returncode == 0
    _assert_csv_rows_match(invariance, tables / "nlsy79_invariance_summary_single_factor_alt.csv")


def test_script_17_generates_inference_variants(tmp_path: Path) -> None:
    project_root = _project_root_with_outputs(tmp_path)
    tables = project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    g_mean = pd.DataFrame(
        [
            {"cohort": "nlsy79", "d_g": 0.2, "SE_d_g": 0.07, "ci_low_d_g": 0.04, "ci_high_d_g": 0.36},
        ]
    )
    g_vr = pd.DataFrame(
        [
            {"cohort": "nlsy79", "VR_g": 1.18, "SE_logVR": 0.09, "ci_low": 0.95, "ci_high": 1.42},
        ]
    )
    g_mean.to_csv(tables / "g_mean_diff.csv", index=False)
    g_vr.to_csv(tables / "g_variance_ratio.csv", index=False)

    result = _run_script(
        project_root,
        [
            "--project-root",
            str(project_root),
            "--family",
            "inference",
            "--variant-token",
            "robust_cluster",
        ],
    )
    assert result.returncode == 0
    assert (tables / "g_mean_diff_robust_cluster.csv").exists()
    assert (tables / "g_variance_ratio_robust_cluster.csv").exists()


def test_script_17_generates_weights_variants(tmp_path: Path) -> None:
    project_root = _project_root_with_outputs(tmp_path)
    tables = project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    g_mean = pd.DataFrame([{"cohort": "nlsy79", "d_g": 0.2, "SE_d_g": 0.07, "ci_low_d_g": 0.04, "ci_high_d_g": 0.36}])
    g_vr = pd.DataFrame([{"cohort": "nlsy79", "VR_g": 1.18, "SE_logVR": 0.09, "ci_low": 0.95, "ci_high": 1.42}])
    g_mean.to_csv(tables / "g_mean_diff.csv", index=False)
    g_vr.to_csv(tables / "g_variance_ratio.csv", index=False)

    result = _run_script(
        project_root,
        [
            "--project-root",
            str(project_root),
            "--family",
            "weights",
            "--variant-token",
            "weighted",
        ],
    )
    assert result.returncode == 0
    assert (tables / "g_mean_diff_weighted.csv").exists()
    assert (tables / "g_variance_ratio_weighted.csv").exists()


def test_script_17_generates_harmonization_variants(tmp_path: Path) -> None:
    project_root = _project_root_with_outputs(tmp_path)
    tables = project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    g_mean = pd.DataFrame(
        [
            {"cohort": "nlsy79", "d_g": 0.2, "SE_d_g": 0.07, "ci_low_d_g": 0.04, "ci_high_d_g": 0.36},
            {"cohort": "nlsy97", "d_g": 0.1, "SE_d_g": 0.06, "ci_low_d_g": -0.01, "ci_high_d_g": 0.21},
        ]
    )
    g_mean.to_csv(tables / "g_mean_diff.csv", index=False)

    result = _run_script(
        project_root,
        [
            "--project-root",
            str(project_root),
            "--family",
            "harmonization",
            "--variant-token",
            "coalesce_raw",
        ],
    )
    assert result.returncode == 0
    assert (tables / "g_mean_diff_coalesce_raw.csv").exists()
    assert (tables / "g_mean_diff_harmonization_coalesce_raw.csv").exists()
    out = pd.read_csv(tables / "g_mean_diff_coalesce_raw.csv")
    assert set(out["cohort"]) == {"nlsy97"}


def test_script_17_fails_when_required_artifact_missing(tmp_path: Path) -> None:
    project_root = _project_root_with_outputs(tmp_path)
    tables = project_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"cohort": "nlsy79", "n_input": 100, "n_after_age": 80, "n_after_test_rule": 60, "n_after_dedupe": 58}]
    ).to_csv(
        tables / "sample_counts.csv", index=False
    )

    result = _run_script(
        project_root,
        [
            "--project-root",
            str(project_root),
            "--family",
            "sampling",
            "--variant-token",
            "sibling_restricted",
        ],
        check=False,
    )
    assert result.returncode != 0
    assert "missing baseline artifact" in result.stderr
