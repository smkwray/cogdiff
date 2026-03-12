from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _script_11_module():
    path = _repo_root() / "scripts" / "11_robustness_suite.py"
    spec = importlib.util.spec_from_file_location("stage11_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_robustness_suite(project_root: Path, args: list[str]) -> None:
    script = _repo_root() / "scripts" / "11_robustness_suite.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy79"] + args,
        cwd=_repo_root(),
        check=True,
    )


def _write_sampling_rerun_script(project_root: Path) -> Path:
    script = project_root / "scripts" / "write_sampling_rerun.py"
    _write(
        script,
        """
import pandas as pd
from pathlib import Path
import sys

project_root = Path(sys.argv[1])
outputs_dir = Path(sys.argv[2])
variant = sys.argv[3]
tables = outputs_dir / "tables"
tables.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    [
        {
            "cohort": "nlsy79",
            "n_input": 200,
            "n_after_age": 170,
            "n_after_test_rule": 140,
            "n_after_dedupe": 120,
        }
    ]
).to_csv(tables / f"sample_counts_{variant}.csv", index=False)
pd.DataFrame(
    [
        {
            "cohort": "nlsy79",
            "d_g": 0.32,
            "SE": 0.091,
            "ci_low": 0.14,
            "ci_high": 0.51,
        }
    ]
).to_csv(tables / f"g_mean_diff_{variant}.csv", index=False)
"""
    )
    return script


def _write_sleep_rerun_script(project_root: Path) -> Path:
    script = project_root / "scripts" / "write_sleep_rerun.py"
    _write(
        script,
        """
import time
time.sleep(1)
""",
    )
    return script


def _write_harmonization_rerun_script(project_root: Path) -> Path:
    script = project_root / "scripts" / "write_harmonization_rerun.py"
    _write(
        script,
        """
import pandas as pd
from pathlib import Path
import sys

project_root = Path(sys.argv[1])
outputs_dir = Path(sys.argv[2])
variant = sys.argv[3]
tables = outputs_dir / "tables"
tables.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    [
        {
            "cohort": "nlsy97",
            "d_g": 0.17,
            "SE": 0.081,
            "ci_low": 0.01,
            "ci_high": 0.33,
        }
    ]
).to_csv(tables / f"g_mean_diff_{variant}.csv", index=False)
"""
    )
    return script


def _write_minimal_robustness_fixture(
    project_root: Path,
    has_rerun_command: bool,
    rerun_command_override: str | None = None,
    rerun_timeout_seconds: float | None = None,
) -> Path:
    command_script = _write_sampling_rerun_script(project_root)
    rerun_command = (
        "{python_executable} {command_script} {{project_root}} {{outputs_dir}} {{variant_token}} {{cohort}}"
    ).format(python_executable=sys.executable, command_script=command_script)
    command_value = rerun_command if has_rerun_command else ""
    if rerun_command_override is not None:
        command_value = rerun_command_override
    timeout_line = (
        f"rerun_timeout_seconds: {rerun_timeout_seconds}\n"
        if rerun_timeout_seconds is not None
        else ""
    )
    _write(
        project_root / "config/robustness.yml",
        f"""
sampling_schemes:
  - sibling_restricted
  - one_pair_per_family
age_adjustment: []
residualization_mode: []
inference: []
weights: []
{timeout_line}rerun_commands:
  sampling:
    one_pair_per_family: \"{command_value}\"
""",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "n_input": 100,
                "n_after_age": 80,
                "n_after_test_rule": 60,
                "n_after_dedupe": 55,
            }
        ]
    ).to_csv(tables_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "d_g": 0.18,
                "SE": 0.078,
                "ci_low": 0.01,
                "ci_high": 0.35,
            }
        ]
    ).to_csv(tables_dir / "g_mean_diff.csv", index=False)

    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "model_step": "configural",
                "cfi": 0.95,
                "tli": 0.94,
                "rmsea": 0.04,
                "srmr": 0.03,
                "chisq_scaled": 1,
                "df": 10,
                "aic": 1,
                "bic": 1,
            },
            {
                "cohort": "nlsy79",
                "model_step": "strict",
                "cfi": 0.93,
                "tli": 0.92,
                "rmsea": 0.05,
                "srmr": 0.034,
                "chisq_scaled": 1,
                "df": 10,
                "aic": 1,
                "bic": 1,
            },
        ]
    ).to_csv(tables_dir / "nlsy79_invariance_summary.csv", index=False)
    return command_script


def test_default_rerun_templates_cover_default_contexts() -> None:
    module = _script_11_module()
    root = _repo_root()
    outputs_dir = root / "outputs"
    tables_dir = outputs_dir / "tables"
    cfg = module._load_robustness_config(root)
    contexts = module._build_rerun_contexts(
        tables_dir,
        ["nlsy79", "nlsy97", "cnlsy"],
        cfg["sampling_schemes"],
        cfg["age_adjustment"],
        cfg["residualization_mode"],
        cfg["inference"],
        cfg["weights"],
        cfg["harmonization_methods"],
        cfg["harmonization_baseline_method"],
    )

    missing_templates: list[tuple[str, str, str | None]] = []
    for family, variant, cohort, _, _ in contexts:
        template = cfg["rerun_commands"].get(family, {}).get(variant)
        if template is None:
            missing_templates.append((family, variant, cohort))
            continue
        assert module._unsupported_placeholder_tokens(template) == []
        rendered = module._format_rerun_command(
            template,
            project_root=root,
            outputs_dir=outputs_dir,
            tables_dir=tables_dir,
            cohort=cohort,
            variant_token=variant,
            robustness_family=family,
        )
        assert "{" not in rendered
        assert "}" not in rendered

    assert not missing_templates


def test_align_interval_to_point_recenters_when_point_falls_outside_ci() -> None:
    module = _script_11_module()
    ci_low, ci_high = module._align_interval_to_point(
        point_estimate=-0.30,
        source_estimate=0.20,
        ci_low=0.10,
        ci_high=0.30,
    )
    assert ci_low == pytest.approx(-0.40)
    assert ci_high == pytest.approx(-0.20)


def test_script_11_writes_required_robustness_tables(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(
        project_root / "config/paths.yml",
        "outputs_dir: outputs\n",
    )
    _write(
        project_root / "config/robustness.yml",
        """
sampling_schemes: [sibling_restricted, full_cohort, one_pair_per_family]
age_adjustment: [quadratic, cubic, spline]
residualization_mode: [pooled, within_sex]
""",
    )

    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "n_input": 100,
                "n_after_age": 80,
                "n_after_test_rule": 60,
                "n_after_dedupe": 55,
            },
            {
                "cohort": "nlsy97",
                "n_input": 120,
                "n_after_age": 95,
                "n_after_test_rule": 70,
                "n_after_dedupe": 67,
            },
        ]
    ).to_csv(tables_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "subtest": "GS",
                "predictor_col": "birth_year",
                "n_used": 55,
                "r2": 0.91,
                "beta0": 0.0,
                "beta1": 1.0,
                "beta2": 0.0,
                "resid_sd": 1.2,
                "outliers_3sd": 0,
            },
            {
                "cohort": "nlsy97",
                "subtest": "GS",
                "n_used": 67,
                "r2": 0.88,
                "beta0": 0.1,
                "beta1": 0.9,
                "beta2": 0.0,
                "resid_sd": 1.1,
                "outliers_3sd": 0,
            },
        ]
    ).to_csv(tables_dir / "residualization_diagnostics_all.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "subtest": "GS",
                "predictor_col": "birth_year",
                "n_used": 62,
                "r2": 0.93,
                "beta0": 0.0,
                "beta1": 0.95,
                "beta2": 0.0,
                "resid_sd": 1.0,
                "outliers_3sd": 1,
            }
        ]
    ).to_csv(tables_dir / "residualization_diagnostics_all_cubic_within_sex.csv", index=False)

    pd.DataFrame(
        {
            "cohort": ["nlsy79"] * 2,
            "model_step": ["configural", "strict"],
            "cfi": [0.95, 0.93],
            "tli": [0.92, 0.90],
            "rmsea": [0.04, 0.05],
            "srmr": [0.03, 0.034],
            "chisq_scaled": [1, 1],
            "df": [10, 10],
            "aic": [1, 1],
            "bic": [1, 1],
        }
    ).to_csv(tables_dir / "nlsy79_invariance_summary.csv", index=False)

    pd.DataFrame(
        {
            "cohort": ["nlsy97"] * 2,
            "model_step": ["configural", "strict"],
            "cfi": [0.94, 0.92],
            "tli": [0.93, 0.91],
            "rmsea": [0.042, 0.052],
            "srmr": [0.033, 0.035],
            "chisq_scaled": [1, 1],
            "df": [10, 10],
            "aic": [1, 1],
            "bic": [1, 1],
        }
    ).to_csv(tables_dir / "nlsy97_invariance_summary.csv", index=False)
    pd.DataFrame(
        {
            "cohort": ["nlsy79"] * 2,
            "model_step": ["configural", "strict"],
            "cfi": [0.95, 0.925],
            "tli": [0.92, 0.9],
            "rmsea": [0.04, 0.049],
            "srmr": [0.031, 0.033],
            "chisq_scaled": [1, 1],
            "df": [10, 10],
            "aic": [1, 1],
            "bic": [1, 1],
        }
    ).to_csv(tables_dir / "nlsy79_invariance_summary_single_factor_alt.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "d_g": 0.21,
                "SE": 0.071,
                "ci_low": 0.07,
                "ci_high": 0.35,
            }
        ]
    ).to_csv(tables_dir / "g_mean_diff_full_cohort.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "d_g": 0.23,
                "SE": 0.074,
                "ci_low": 0.09,
                "ci_high": 0.37,
            },
            {
                "cohort": "nlsy97",
                "d_g": 0.16,
                "SE": 0.082,
                "ci_low": -0.01,
                "ci_high": 0.33,
            }
        ]
    ).to_csv(tables_dir / "g_mean_diff_family_bootstrap.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "d_g": 0.22,
                "SE": 0.072,
                "ci_low": 0.08,
                "ci_high": 0.36,
            },
            {
                "cohort": "nlsy97",
                "d_g": 0.17,
                "SE": 0.083,
                "ci_low": -0.02,
                "ci_high": 0.34,
            },
        ]
    ).to_csv(tables_dir / "sample_counts_full_cohort.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "VR_g": 1.31,
                "SE_logVR": 0.12,
                "ci_low": 1.05,
                "ci_high": 1.75,
            },
            {
                "cohort": "nlsy97",
                "VR_g": 1.21,
                "SE_logVR": 0.14,
                "ci_low": 0.95,
                "ci_high": 1.56,
            },
        ]
    ).to_csv(tables_dir / "g_variance_ratio_family_bootstrap.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "d_g": 0.22,
                "SE": 0.070,
                "ci_low": 0.07,
                "ci_high": 0.35,
            },
            {
                "cohort": "nlsy97",
                "d_g": 0.17,
                "SE": 0.080,
                "ci_low": -0.01,
                "ci_high": 0.34,
            },
        ]
    ).to_csv(tables_dir / "g_mean_diff_weighted.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "VR_g": 1.31,
                "SE_logVR": 0.12,
                "ci_low": 1.05,
                "ci_high": 1.75,
            },
            {
                "cohort": "nlsy97",
                "VR_g": 1.21,
                "SE_logVR": 0.14,
                "ci_low": 0.95,
                "ci_high": 1.56,
            },
        ]
    ).to_csv(tables_dir / "g_variance_ratio_weighted.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "d_g": 0.2,
                "IQ_diff": 3.0,
                "SE": 0.07,
                "ci_low": 0.06,
                "ci_high": 0.34,
            },
            {
                "cohort": "nlsy97",
                "d_g": 0.15,
                "IQ_diff": 2.25,
                "SE": 0.08,
                "ci_low": -0.01,
                "ci_high": 0.31,
            },
        ]
    ).to_csv(tables_dir / "g_mean_diff.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "VR_g": 1.3,
                "SE_logVR": 0.11,
                "ci_low": 1.02,
                "ci_high": 1.66,
            },
            {
                "cohort": "nlsy97",
                "VR_g": 1.2,
                "SE_logVR": 0.13,
                "ci_low": 0.93,
                "ci_high": 1.55,
            },
        ]
    ).to_csv(tables_dir / "g_variance_ratio.csv", index=False)

    script = _repo_root() / "scripts" / "11_robustness_suite.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--all"],
        cwd=_repo_root(),
        check=True,
    )

    sampling = pd.read_csv(project_root / "outputs" / "tables" / "robustness_sampling.csv")
    age = pd.read_csv(project_root / "outputs" / "tables" / "robustness_age_adjustment.csv")
    forms = pd.read_csv(project_root / "outputs" / "tables" / "robustness_model_forms.csv")
    inference = pd.read_csv(project_root / "outputs" / "tables" / "robustness_inference.csv")
    weights = pd.read_csv(project_root / "outputs" / "tables" / "robustness_weights.csv")

    assert set(["cohort", "sampling_scheme", "status", "n_after_dedupe"]).issubset(set(sampling.columns))
    assert set(["cohort", "age_adjustment", "residualization_mode", "status"]).issubset(set(age.columns))
    assert set(["cohort", "model_form", "status", "cfi"]).issubset(set(forms.columns))
    assert set(["cohort", "inference_method", "estimate_type", "status", "estimate"]).issubset(set(inference.columns))
    assert set(["cohort", "weight_mode", "estimate_type", "status", "estimate"]).issubset(set(weights.columns))
    assert set(sampling["sampling_scheme"]) == {"sibling_restricted", "full_cohort", "one_pair_per_family"}
    assert set(inference["inference_method"]) == {"robust_cluster", "family_bootstrap"}
    assert set(weights["weight_mode"]) == {"unweighted", "weighted"}
    assert (sampling["status"] == "computed").any()
    assert (inference["status"] == "computed").any()
    assert (weights["status"] == "computed").any()
    assert sampling.loc[(sampling["cohort"] == "nlsy79") & (sampling["sampling_scheme"] == "full_cohort"), "status"].iloc[0] == "computed"
    assert sampling.loc[(sampling["cohort"] == "nlsy79") & (sampling["sampling_scheme"] == "full_cohort"), "d_g"].iloc[0] == 0.21
    assert (
        forms.loc[
            (forms["cohort"] == "nlsy79") & (forms["model_form"] == "single_factor_alt"),
            "status",
        ].iloc[0]
        == "computed"
    )
    assert (
        inference.loc[
            (inference["cohort"] == "nlsy79")
            & (inference["inference_method"] == "family_bootstrap")
            & (inference["estimate_type"] == "d_g"),
            "status",
        ].iloc[0]
        == "computed"
    )
    bootstrap_inference_dg = inference[
        (inference["cohort"] == "nlsy79")
        & (inference["inference_method"] == "family_bootstrap")
        & (inference["estimate_type"] == "d_g")
    ].iloc[0]
    baseline_inference_dg = inference[
        (inference["cohort"] == "nlsy79")
        & (inference["inference_method"] == "robust_cluster")
        & (inference["estimate_type"] == "d_g")
    ].iloc[0]
    assert bootstrap_inference_dg["estimate"] == baseline_inference_dg["estimate"]
    assert bootstrap_inference_dg["estimate"] != 0.23
    assert bootstrap_inference_dg["se"] == 0.074
    assert bootstrap_inference_dg["ci_low"] == 0.09
    assert bootstrap_inference_dg["ci_high"] == 0.37
    bootstrap_inference_vr = inference[
        (inference["cohort"] == "nlsy79")
        & (inference["inference_method"] == "family_bootstrap")
        & (inference["estimate_type"] == "vr_g")
    ].iloc[0]
    baseline_inference_vr = inference[
        (inference["cohort"] == "nlsy79")
        & (inference["inference_method"] == "robust_cluster")
        & (inference["estimate_type"] == "vr_g")
    ].iloc[0]
    assert bootstrap_inference_vr["estimate"] == baseline_inference_vr["estimate"]
    assert bootstrap_inference_vr["estimate"] != 1.31
    assert bootstrap_inference_vr["se"] == 0.12
    assert bootstrap_inference_vr["ci_low"] == 1.05
    assert bootstrap_inference_vr["ci_high"] == 1.75
    assert (
        weights.loc[
            (weights["cohort"] == "nlsy79")
            & (weights["weight_mode"] == "weighted")
            & (weights["estimate_type"] == "d_g"),
            "status",
        ].iloc[0]
        == "computed"
    )
    assert (
        age.loc[
            (age["cohort"] == "nlsy79")
            & (age["age_adjustment"] == "cubic")
            & (age["residualization_mode"] == "within_sex"),
            "status",
        ].iloc[0]
        == "computed"
    )
    assert age.loc[
        (age["cohort"] == "nlsy79")
        & (age["age_adjustment"] == "cubic")
        & (age["residualization_mode"] == "within_sex"),
        "n_subtests",
    ].iloc[0] > 0
    manifest = pd.read_csv(project_root / "outputs" / "tables" / "robustness_run_manifest.csv")
    assert set(["cohort", "robustness_family", "variant_token", "status", "source_paths"]).issubset(
        set(manifest.columns)
    )
    assert (
        manifest.loc[
            (manifest["cohort"] == "nlsy79")
            & (manifest["robustness_family"] == "sampling")
            & (manifest["variant_token"] == "sibling_restricted")
            & (manifest["status"] == "computed")
        ]
        .shape[0]
        >= 1
    )
    assert (
        manifest.loc[
            (manifest["cohort"] == "nlsy79")
            & (manifest["robustness_family"] == "sampling")
            & (manifest["variant_token"] == "full_cohort")
            & (manifest["status"] == "computed")
        ]
        .shape[0]
        >= 1
    )
    weight_sources = manifest.loc[
        (manifest["cohort"] == "nlsy79")
        & (manifest["robustness_family"] == "weights")
        & (manifest["variant_token"] == "weighted"),
        "source_paths",
    ].astype(str)
    assert not weight_sources.empty
    assert weight_sources.str.contains("weights_quality_diagnostics.csv").all()
    assert (manifest["source_paths"].astype(str).str.len() > 0).any()
    assert (project_root / "outputs" / "figures" / "robustness_forestplot.png").exists()


def test_script_11_executes_configured_rerun_to_generate_variant_artifact(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    command_script = _write_minimal_robustness_fixture(project_root, has_rerun_command=True)

    _run_robustness_suite(
        project_root,
        ["--rerun-sampling-variants"],
    )

    tables_dir = project_root / "outputs" / "tables"
    sampling = pd.read_csv(tables_dir / "robustness_sampling.csv")
    manifest = pd.read_csv(tables_dir / "robustness_run_manifest.csv")
    rerun_log = pd.read_csv(tables_dir / "robustness_rerun_log.csv")

    assert sampling.loc[
        (sampling["cohort"] == "nlsy79") & (sampling["sampling_scheme"] == "one_pair_per_family"),
        "status",
    ].iloc[0] == "computed"
    assert (tables_dir / "sample_counts_one_pair_per_family.csv").exists()
    assert (tables_dir / "g_mean_diff_one_pair_per_family.csv").exists()
    assert (
        manifest.loc[
            (manifest["robustness_family"] == "sampling")
            & (manifest["variant_token"] == "one_pair_per_family"),
            "rerun_status",
        ].iloc[0]
        == "success"
    )
    assert (
        rerun_log.loc[
            (rerun_log["robustness_family"] == "sampling")
            & (rerun_log["variant_token"] == "one_pair_per_family"),
            "status",
        ].iloc[0]
        == "success"
    )


def test_script_11_supports_python_executable_placeholder(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    command_script = _write_minimal_robustness_fixture(project_root, has_rerun_command=False)
    _write(
        project_root / "config/robustness.yml",
        f"""
sampling_schemes:
  - sibling_restricted
  - one_pair_per_family
age_adjustment: []
residualization_mode: []
inference: []
weights: []
rerun_commands:
  sampling:
    one_pair_per_family: "{{python_executable}} {command_script} {{project_root}} {{outputs_dir}} {{variant_token}}"
""",
    )

    _run_robustness_suite(
        project_root,
        ["--rerun-sampling-variants"],
    )

    tables_dir = project_root / "outputs" / "tables"
    rerun_log = pd.read_csv(tables_dir / "robustness_rerun_log.csv")
    assert (
        rerun_log.loc[
            (rerun_log["robustness_family"] == "sampling")
            & (rerun_log["variant_token"] == "one_pair_per_family"),
            "status",
        ].iloc[0]
        == "success"
    )
    assert (tables_dir / "sample_counts_one_pair_per_family.csv").exists()


def test_script_11_legacy_rerun_alias_still_runs_variants(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    _write_minimal_robustness_fixture(project_root, has_rerun_command=True)

    _run_robustness_suite(
        project_root,
        ["--rerun-missing-variants"],
    )

    tables_dir = project_root / "outputs" / "tables"
    rerun_log = pd.read_csv(tables_dir / "robustness_rerun_log.csv")
    assert (
        rerun_log.loc[
            (rerun_log["robustness_family"] == "sampling")
            & (rerun_log["variant_token"] == "one_pair_per_family"),
            "status",
        ].iloc[0]
        == "success"
    )


def test_script_11_marks_format_error_for_unsupported_placeholder(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    sleep_script = _write_sleep_rerun_script(project_root)
    _write_minimal_robustness_fixture(
        project_root,
        has_rerun_command=True,
        rerun_command_override=f"{sys.executable} {sleep_script} {{unknown_placeholder}}",
    )

    _run_robustness_suite(
        project_root,
        ["--rerun-sampling-variants"],
    )

    tables_dir = project_root / "outputs" / "tables"
    rerun_log = pd.read_csv(tables_dir / "robustness_rerun_log.csv")
    row = rerun_log.loc[
        (rerun_log["robustness_family"] == "sampling")
        & (rerun_log["variant_token"] == "one_pair_per_family")
    ].iloc[0]
    assert row["status"] == "format_error"
    assert "unsupported placeholder" in str(row["error"])


def test_script_11_marks_timeout_when_rerun_exceeds_timeout(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    sleep_script = _write_sleep_rerun_script(project_root)
    _write_minimal_robustness_fixture(
        project_root,
        has_rerun_command=True,
        rerun_command_override=f"{sys.executable} {sleep_script}",
        rerun_timeout_seconds=0.01,
    )

    _run_robustness_suite(
        project_root,
        ["--rerun-sampling-variants"],
    )

    tables_dir = project_root / "outputs" / "tables"
    rerun_log = pd.read_csv(tables_dir / "robustness_rerun_log.csv")
    row = rerun_log.loc[
        (rerun_log["robustness_family"] == "sampling")
        & (rerun_log["variant_token"] == "one_pair_per_family")
    ].iloc[0]
    assert row["status"] == "timeout"


def test_script_11_skips_sampling_plot_when_ci_interval_is_invalid(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        project_root / "config/robustness.yml",
        """
sampling_schemes: [sibling_restricted]
age_adjustment: []
residualization_mode: []
inference: []
weights: []
""",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "n_input": 100,
                "n_after_age": 80,
                "n_after_test_rule": 60,
                "n_after_dedupe": 55,
            }
        ]
    ).to_csv(tables_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "d_g": 0.2,
                "SE": 0.25,
                "ci_low": 2.6,
                "ci_high": 3.6,
            }
        ]
    ).to_csv(tables_dir / "g_mean_diff.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "model_step": "strict",
                "cfi": 0.93,
                "tli": 0.92,
                "rmsea": 0.05,
                "srmr": 0.034,
                "chisq_scaled": 1,
                "df": 10,
                "aic": 1,
                "bic": 1,
            }
        ]
    ).to_csv(tables_dir / "nlsy79_invariance_summary.csv", index=False)

    _run_robustness_suite(project_root, [])
    assert (project_root / "outputs/tables/robustness_sampling.csv").exists()
    assert not (project_root / "outputs/figures/robustness_forestplot.png").exists()


def test_script_11_prefers_d_g_scale_ci_columns_when_available(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        project_root / "config/robustness.yml",
        """
sampling_schemes: [sibling_restricted]
age_adjustment: []
residualization_mode: []
inference: []
weights: []
""",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "n_input": 100,
                "n_after_age": 80,
                "n_after_test_rule": 60,
                "n_after_dedupe": 55,
            }
        ]
    ).to_csv(tables_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "d_g": 0.2,
                "SE": 0.25,
                "ci_low": 2.6,
                "ci_high": 3.6,
                "SE_d_g": 0.05,
                "ci_low_d_g": 0.1,
                "ci_high_d_g": 0.3,
            }
        ]
    ).to_csv(tables_dir / "g_mean_diff.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "model_step": "strict",
                "cfi": 0.93,
                "tli": 0.92,
                "rmsea": 0.05,
                "srmr": 0.034,
                "chisq_scaled": 1,
                "df": 10,
                "aic": 1,
                "bic": 1,
            }
        ]
    ).to_csv(tables_dir / "nlsy79_invariance_summary.csv", index=False)

    _run_robustness_suite(project_root, [])

    sampling = pd.read_csv(project_root / "outputs/tables/robustness_sampling.csv")
    row = sampling.loc[sampling["sampling_scheme"] == "sibling_restricted"].iloc[0]
    assert row["se_d_g"] == 0.05
    assert row["ci_low"] == 0.1
    assert row["ci_high"] == 0.3
    assert (project_root / "outputs/figures/robustness_forestplot.png").exists()


def test_script_11_builds_harmonization_table_with_baseline_and_alternative(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        project_root / "config/robustness.yml",
        """
sampling_schemes: [sibling_restricted]
age_adjustment: []
residualization_mode: []
inference: []
weights: []
harmonization_baseline_method: zscore_by_branch
harmonization_methods:
  - coalesce_raw
""",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"cohort": "nlsy97", "n_input": 100, "n_after_age": 80, "n_after_test_rule": 60, "n_after_dedupe": 55}
        ]
    ).to_csv(tables_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy97",
                "d_g": 0.2,
                "SE": 0.08,
                "ci_low": 0.04,
                "ci_high": 0.36,
            }
        ]
    ).to_csv(tables_dir / "g_mean_diff.csv", index=False)
    pd.DataFrame(
        [
            {"cohort": "nlsy97", "d_g": 0.17, "SE": 0.09, "ci_low": 0.03, "ci_high": 0.31}
        ]
    ).to_csv(tables_dir / "g_mean_diff_coalesce_raw.csv", index=False)

    script = _repo_root() / "scripts" / "11_robustness_suite.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy97"],
        cwd=_repo_root(),
        check=True,
    )

    harmonization = pd.read_csv(project_root / "outputs/tables/robustness_harmonization.csv")
    row = harmonization.loc[harmonization["alternative_method"] == "coalesce_raw"].iloc[0]
    assert row["cohort"] == "nlsy97"
    assert row["status"] == "computed"
    assert row["baseline_estimate"] == 0.2
    assert row["alternative_estimate"] == 0.17
    assert row["delta_estimate"] == -0.03
    manifest = pd.read_csv(project_root / "outputs/tables/robustness_run_manifest.csv")
    assert (
        manifest.loc[
            (manifest["robustness_family"] == "harmonization")
            & (manifest["variant_token"] == "coalesce_raw"),
            "rerun_status",
        ].iloc[0]
        == "available"
    )


def test_script_11_executes_harmonization_rerun_from_config(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    rerun_script = _write_harmonization_rerun_script(project_root)
    _write(
        project_root / "config/robustness.yml",
        f'''
sampling_schemes: [sibling_restricted]
age_adjustment: []
residualization_mode: []
inference: []
weights: []
harmonization_baseline_method: zscore_by_branch
harmonization_methods:
  - coalesce_raw
rerun_commands:
  harmonization:
    coalesce_raw: "{sys.executable} {rerun_script} {{project_root}} {{outputs_dir}} {{variant_token}} {{cohort}}"
''',
    )

    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"cohort": "nlsy97", "n_input": 100, "n_after_age": 80, "n_after_test_rule": 60, "n_after_dedupe": 55}]
    ).to_csv(tables_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy97",
                "d_g": 0.2,
                "SE": 0.08,
                "ci_low": 0.04,
                "ci_high": 0.36,
            }
        ]
    ).to_csv(tables_dir / "g_mean_diff.csv", index=False)

    script = _repo_root() / "scripts" / "11_robustness_suite.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy97", "--rerun-harmonization-variants"],
        cwd=_repo_root(),
        check=True,
    )

    rerun_log = pd.read_csv(project_root / "outputs/tables/robustness_rerun_log.csv")
    row = rerun_log.loc[
        (rerun_log["robustness_family"] == "harmonization")
        & (rerun_log["variant_token"] == "coalesce_raw")
    ].iloc[0]
    assert row["status"] == "success"
    assert (tables_dir / "g_mean_diff_coalesce_raw.csv").exists()
    harmonization = pd.read_csv(project_root / "outputs/tables/robustness_harmonization.csv")
    row = harmonization.loc[harmonization["alternative_method"] == "coalesce_raw"].iloc[0]
    assert row["status"] == "computed"
    assert row["alternative_estimate"] == 0.17


def test_script_11_marks_harmonization_baseline_missing_explicitly(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        project_root / "config/robustness.yml",
        """
sampling_schemes: []
age_adjustment: []
residualization_mode: []
inference: []
weights: []
harmonization_baseline_method: signed_merge
harmonization_methods:
  - zscore_by_branch
""",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"cohort": "nlsy97", "n_input": 100, "n_after_age": 80, "n_after_test_rule": 60, "n_after_dedupe": 55}]
    ).to_csv(tables_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "d_g": 0.2, "SE": 0.08, "ci_low": 0.04, "ci_high": 0.36}]
    ).to_csv(tables_dir / "g_mean_diff.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy97", "d_g": 0.17, "SE": 0.09, "ci_low": 0.03, "ci_high": 0.31}]
    ).to_csv(tables_dir / "g_mean_diff_zscore_by_branch.csv", index=False)

    script = _repo_root() / "scripts" / "11_robustness_suite.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy97"],
        cwd=_repo_root(),
        check=True,
    )

    harmonization = pd.read_csv(project_root / "outputs/tables/robustness_harmonization.csv")
    row = harmonization.loc[harmonization["alternative_method"] == "zscore_by_branch"].iloc[0]
    assert row["cohort"] == "nlsy97"
    assert row["status"] == "baseline_missing"
    assert pd.isna(row["baseline_estimate"])
    assert row["alternative_estimate"] == 0.17
    assert pd.isna(row["delta_estimate"])


def test_script_11_uses_baseline_method_variant_file_when_main_baseline_missing(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        project_root / "config/robustness.yml",
        """
sampling_schemes: []
age_adjustment: []
residualization_mode: []
inference: []
weights: []
harmonization_baseline_method: signed_merge
harmonization_methods:
  - zscore_by_branch
""",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"cohort": "nlsy97", "n_input": 100, "n_after_age": 80, "n_after_test_rule": 60, "n_after_dedupe": 55}]
    ).to_csv(tables_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "d_g": 0.2, "SE": 0.08, "ci_low": 0.04, "ci_high": 0.36}]
    ).to_csv(tables_dir / "g_mean_diff.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy97", "d_g": 0.19, "SE": 0.09, "ci_low": 0.03, "ci_high": 0.35}]
    ).to_csv(tables_dir / "g_mean_diff_signed_merge.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy97", "d_g": 0.17, "SE": 0.09, "ci_low": 0.03, "ci_high": 0.31}]
    ).to_csv(tables_dir / "g_mean_diff_zscore_by_branch.csv", index=False)

    script = _repo_root() / "scripts" / "11_robustness_suite.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy97"],
        cwd=_repo_root(),
        check=True,
    )

    harmonization = pd.read_csv(project_root / "outputs/tables/robustness_harmonization.csv")
    row = harmonization.loc[harmonization["alternative_method"] == "zscore_by_branch"].iloc[0]
    assert row["status"] == "computed"
    assert row["baseline_estimate"] == 0.19
    assert row["alternative_estimate"] == 0.17
    assert float(row["delta_estimate"]) == pytest.approx(-0.02)


def test_script_11_marks_estimate_baseline_missing_explicitly(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        project_root / "config/robustness.yml",
        """
sampling_schemes: []
age_adjustment: []
residualization_mode: []
inference: [robust_cluster]
weights: [unweighted]
""",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"cohort": "nlsy79", "n_input": 100, "n_after_age": 80, "n_after_test_rule": 60, "n_after_dedupe": 55}]
    ).to_csv(tables_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy97", "d_g": 0.2, "SE": 0.08, "ci_low": 0.04, "ci_high": 0.36}]
    ).to_csv(tables_dir / "g_mean_diff.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy97", "VR_g": 1.2, "SE_logVR": 0.1, "ci_low": 1.0, "ci_high": 1.5}]
    ).to_csv(tables_dir / "g_variance_ratio.csv", index=False)

    _run_robustness_suite(project_root, [])

    inference = pd.read_csv(project_root / "outputs/tables/robustness_inference.csv")
    weights = pd.read_csv(project_root / "outputs/tables/robustness_weights.csv")
    inf_dg = inference[(inference["cohort"] == "nlsy79") & (inference["estimate_type"] == "d_g")].iloc[0]
    inf_vr = inference[(inference["cohort"] == "nlsy79") & (inference["estimate_type"] == "vr_g")].iloc[0]
    w_dg = weights[(weights["cohort"] == "nlsy79") & (weights["estimate_type"] == "d_g")].iloc[0]
    w_vr = weights[(weights["cohort"] == "nlsy79") & (weights["estimate_type"] == "vr_g")].iloc[0]
    assert inf_dg["status"] == "baseline_missing"
    assert inf_vr["status"] == "baseline_missing"
    assert w_dg["status"] == "baseline_missing"
    assert w_vr["status"] == "baseline_missing"


def test_script_11_does_not_rerun_without_flag(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    _write_minimal_robustness_fixture(project_root, has_rerun_command=True)

    _run_robustness_suite(
        project_root,
        [],
    )

    tables_dir = project_root / "outputs" / "tables"
    sampling = pd.read_csv(tables_dir / "robustness_sampling.csv")
    manifest = pd.read_csv(tables_dir / "robustness_run_manifest.csv")
    rerun_log = pd.read_csv(tables_dir / "robustness_rerun_log.csv")

    assert sampling.loc[
        (sampling["cohort"] == "nlsy79") & (sampling["sampling_scheme"] == "one_pair_per_family"),
        "status",
    ].iloc[0] == "not_run_placeholder"
    assert not (tables_dir / "sample_counts_one_pair_per_family.csv").exists()
    assert not (tables_dir / "g_mean_diff_one_pair_per_family.csv").exists()
    assert (
        manifest.loc[
            (manifest["robustness_family"] == "sampling")
            & (manifest["variant_token"] == "one_pair_per_family"),
            "rerun_status",
        ].iloc[0]
        == "not_run"
    )
    assert (
        rerun_log.loc[
            (rerun_log["robustness_family"] == "sampling")
            & (rerun_log["variant_token"] == "one_pair_per_family"),
            "status",
        ].iloc[0]
        == "not_run"
    )


def test_script_11_marks_placeholder_artifact_variants_not_feasible(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        project_root / "config/robustness.yml",
        """
sampling_schemes: [sibling_restricted, one_pair_per_family]
age_adjustment: []
residualization_mode: []
inference: []
weights: []
rerun_commands:
  sampling:
    one_pair_per_family: "{python_executable} {project_root}/scripts/17_generate_robustness_variant.py --project-root {project_root} --family sampling --variant-token one_pair_per_family"
""",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"cohort": "nlsy79", "n_input": 100, "n_after_age": 80, "n_after_test_rule": 60, "n_after_dedupe": 55}]
    ).to_csv(tables_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "d_g": 0.2, "SE": 0.08, "ci_low": 0.04, "ci_high": 0.36}]
    ).to_csv(tables_dir / "g_mean_diff.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "n_input": 90, "n_after_age": 70, "n_after_test_rule": 50, "n_after_dedupe": 45}]
    ).to_csv(tables_dir / "sample_counts_one_pair_per_family.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "d_g": 0.19, "SE": 0.08, "ci_low": 0.03, "ci_high": 0.35}]
    ).to_csv(tables_dir / "g_mean_diff_one_pair_per_family.csv", index=False)

    _run_robustness_suite(project_root, [])

    sampling = pd.read_csv(tables_dir / "robustness_sampling.csv")
    row = sampling.loc[
        (sampling["cohort"] == "nlsy79") & (sampling["sampling_scheme"] == "one_pair_per_family")
    ].iloc[0]
    assert row["status"] == "not_feasible"


def test_script_11_marks_weights_not_feasible_from_variant_status_column(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        project_root / "config/robustness.yml",
        """
sampling_schemes: []
age_adjustment: []
residualization_mode: []
inference: []
weights: [unweighted, weighted]
""",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"cohort": "nlsy79", "d_g": 0.2, "SE": 0.08, "ci_low": 0.04, "ci_high": 0.36}]
    ).to_csv(tables_dir / "g_mean_diff.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "VR_g": 1.2, "SE_logVR": 0.1, "ci_low": 1.0, "ci_high": 1.5}]
    ).to_csv(tables_dir / "g_variance_ratio.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "status": "not_feasible", "reason": "weight_column_unavailable"}]
    ).to_csv(tables_dir / "g_mean_diff_weighted.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "status": "not_feasible", "reason": "weight_column_unavailable"}]
    ).to_csv(tables_dir / "g_variance_ratio_weighted.csv", index=False)

    _run_robustness_suite(project_root, [])

    weights = pd.read_csv(project_root / "outputs/tables/robustness_weights.csv")
    dg_row = weights[
        (weights["cohort"] == "nlsy79")
        & (weights["weight_mode"] == "weighted")
        & (weights["estimate_type"] == "d_g")
    ].iloc[0]
    vr_row = weights[
        (weights["cohort"] == "nlsy79")
        & (weights["weight_mode"] == "weighted")
        & (weights["estimate_type"] == "vr_g")
    ].iloc[0]
    assert dg_row["status"] == "not_feasible"
    assert vr_row["status"] == "not_feasible"


def test_script_11_family_bootstrap_not_feasible_is_deterministic(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        project_root / "config/robustness.yml",
        """
sampling_schemes: []
age_adjustment: []
residualization_mode: []
inference: [robust_cluster, family_bootstrap]
weights: []
""",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"cohort": "nlsy79", "n_input": 100, "n_after_age": 80, "n_after_test_rule": 60, "n_after_dedupe": 55}]
    ).to_csv(tables_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "d_g": 0.2, "SE": 0.08, "ci_low": 0.04, "ci_high": 0.36}]
    ).to_csv(tables_dir / "g_mean_diff.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "VR_g": 1.2, "SE_logVR": 0.1, "ci_low": 1.0, "ci_high": 1.5}]
    ).to_csv(tables_dir / "g_variance_ratio.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "status": "not_feasible", "reason": "bootstrap_failed"}]
    ).to_csv(tables_dir / "g_mean_diff_family_bootstrap.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "status": "not_feasible", "reason": "bootstrap_failed", "VR_g": 1.45, "SE_logVR": 0.01, "ci_low": 1.1, "ci_high": 1.8}]
    ).to_csv(tables_dir / "g_variance_ratio_family_bootstrap.csv", index=False)

    _run_robustness_suite(project_root, [])

    inference = pd.read_csv(project_root / "outputs/tables/robustness_inference.csv")
    bootstrap_dg_row = inference[
        (inference["cohort"] == "nlsy79")
        & (inference["inference_method"] == "family_bootstrap")
        & (inference["estimate_type"] == "d_g")
    ].iloc[0]
    bootstrap_vr_row = inference[
        (inference["cohort"] == "nlsy79")
        & (inference["inference_method"] == "family_bootstrap")
        & (inference["estimate_type"] == "vr_g")
    ].iloc[0]
    baseline_dg_row = inference[
        (inference["cohort"] == "nlsy79")
        & (inference["inference_method"] == "robust_cluster")
        & (inference["estimate_type"] == "d_g")
    ].iloc[0]
    baseline_vr_row = inference[
        (inference["cohort"] == "nlsy79")
        & (inference["inference_method"] == "robust_cluster")
        & (inference["estimate_type"] == "vr_g")
    ].iloc[0]

    assert baseline_dg_row["status"] == "computed"
    assert baseline_vr_row["status"] == "computed"
    assert bootstrap_dg_row["status"] == "not_feasible"
    assert bootstrap_vr_row["status"] == "not_feasible"
    assert pd.isna(bootstrap_dg_row["estimate"])
    assert pd.isna(bootstrap_vr_row["estimate"])
