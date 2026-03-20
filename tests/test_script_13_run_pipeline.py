from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _latest_summary(log_dir: Path) -> Path:
    summaries = sorted(log_dir.glob("*_pipeline_run_summary.csv"))
    assert summaries, f"No pipeline summary found in {log_dir}"
    return summaries[-1]


def _pipeline_script_module():
    path = _repo_root() / "scripts" / "13_run_pipeline.py"
    spec = importlib.util.spec_from_file_location("stage13_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_13_dry_run_writes_plan_summary(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--stage",
            "0",
            "--stage",
            "2",
            "--dry-run",
        ],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    assert set(summary["status"]) == {"planned"}
    assert set(summary["stage"]) == {0, 2}
    assert summary["command"].str.contains("00_download_raw.py|02_build_variable_map.py").any()


def test_script_13_dry_run_includes_stages_14_and_15(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--stage",
            "14",
            "--stage",
            "15",
            "--dry-run",
        ],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    assert set(summary["stage"]) == {14, 15}
    assert set(summary["status"]) == {"planned"}
    stage14 = summary[summary["stage"] == 14].iloc[0]
    stage15 = summary[summary["stage"] == 15].iloc[0]
    assert "14_reproducibility_report.py" in stage14["command"]
    assert "15_specification_curve_summary.py" in stage15["command"]
    assert "--cohort" not in stage14["command"]
    assert "--cohort nlsy79" in stage15["command"]


def test_script_13_dry_run_includes_preflight_when_requested(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--stage",
            "14",
            "--preflight",
            "--preflight-strict",
            "--dry-run",
        ],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    preflight_rows = summary[summary["script"] == "16_preflight_dependencies.py"]
    assert len(preflight_rows) == 1
    preflight_row = preflight_rows.iloc[0]
    assert preflight_row["stage"] == -1
    assert preflight_row["status"] == "planned"
    assert "--project-root" in preflight_row["command"]
    assert "--cohort nlsy79" in preflight_row["command"]
    assert "--strict" in preflight_row["command"]
    assert summary.iloc[0]["script"] == "16_preflight_dependencies.py"


def test_script_13_all_runs_preflight_by_default(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--all",
            "--stage",
            "14",
            "--dry-run",
        ],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    preflight_rows = summary[summary["script"] == "16_preflight_dependencies.py"]
    assert len(preflight_rows) == 1, "preflight should run by default with --all"
    assert "--strict" in preflight_rows.iloc[0]["command"], "preflight should be strict by default with --all"


def test_script_13_skip_preflight_omits_preflight(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--all",
            "--stage",
            "14",
            "--skip-preflight",
            "--dry-run",
        ],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    preflight_rows = summary[summary["script"] == "16_preflight_dependencies.py"]
    assert len(preflight_rows) == 0, "preflight should be skipped with --skip-preflight"


def test_script_13_executes_preflight_before_stages(tmp_path: Path, monkeypatch: object) -> None:
    module = _pipeline_script_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    execution_order: list[str] = []

    def fake_build_stage_commands(
        *, stage: int, repo_root: Path, target_root: Path, cohorts: list[str], args: object
    ) -> list[module.CommandPlan]:
        return [
            module.CommandPlan(
                stage=stage,
                script=f"stage_{stage}.py",
                cohort_scope="nlsy79",
                argv=[sys.executable, "-c", "print('ok')"],
            )
        ]

    def fake_run_command(
        *, plan: module.CommandPlan, repo_root: Path, target_root: Path, log_file: Path, timeout_seconds: float | None
    ) -> tuple[int | None, str, str]:
        execution_order.append(plan.script)
        return 0, "ok", ""

    monkeypatch.setattr(module, "_build_stage_commands", fake_build_stage_commands)
    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "13_run_pipeline.py",
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--stage",
            "14",
            "--preflight",
        ],
    )

    code = module.main()
    assert code == 0
    assert execution_order == ["16_preflight_dependencies.py", "stage_14.py"]


def test_script_13_preflight_strict_fails_pipeline_on_preflight_error(tmp_path: Path, monkeypatch: object) -> None:
    module = _pipeline_script_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    execution_order: list[str] = []

    def fake_run_command(
        *, plan: module.CommandPlan, repo_root: Path, target_root: Path, log_file: Path, timeout_seconds: float | None
    ) -> tuple[int | None, str, str]:
        execution_order.append(plan.script)
        if plan.script == "16_preflight_dependencies.py":
            assert "--strict" in plan.argv
            return 1, "failed", "preflight failed"
        raise AssertionError("Stage execution should be skipped after preflight failure with --preflight-strict.")

    def fake_build_stage_commands(
        *, stage: int, repo_root: Path, target_root: Path, cohorts: list[str], args: object
    ) -> list[module.CommandPlan]:
        raise AssertionError("Stage commands should not be built when preflight strict fails.")

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module, "_build_stage_commands", fake_build_stage_commands)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "13_run_pipeline.py",
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--stage",
            "14",
            "--preflight",
            "--preflight-strict",
        ],
    )

    code = module.main()
    assert code == 1
    assert execution_order == ["16_preflight_dependencies.py"]


def test_script_13_preflight_non_strict_continues_on_preflight_error(tmp_path: Path, monkeypatch: object) -> None:
    module = _pipeline_script_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    execution_order: list[str] = []

    def fake_run_command(
        *, plan: module.CommandPlan, repo_root: Path, target_root: Path, log_file: Path, timeout_seconds: float | None
    ) -> tuple[int | None, str, str]:
        execution_order.append(plan.script)
        if plan.script == "16_preflight_dependencies.py":
            return 1, "failed", "preflight failed"
        return 0, "ok", ""

    def fake_build_stage_commands(
        *, stage: int, repo_root: Path, target_root: Path, cohorts: list[str], args: object
    ) -> list[module.CommandPlan]:
        return [
            module.CommandPlan(
                stage=stage,
                script=f"stage_{stage}.py",
                cohort_scope="nlsy79",
                argv=[sys.executable, "-c", "print('ok')"],
            )
        ]

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module, "_build_stage_commands", fake_build_stage_commands)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "13_run_pipeline.py",
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--stage",
            "14",
            "--preflight",
        ],
    )

    code = module.main()
    assert code == 0
    assert execution_order == ["16_preflight_dependencies.py", "stage_14.py"]


def test_script_13_executes_stages_14_and_15(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--stage",
            "14",
            "--stage",
            "15",
        ],
        cwd=_repo_root(),
        check=True,
    )

    tables = root / "outputs/tables"
    assert (tables / "reproducibility_hashes.csv").exists()
    assert (tables / "reproducibility_command_provenance.csv").exists()
    assert (tables / "specification_stability_summary.csv").exists()
    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    assert set(summary[summary["stage"].isin([14, 15])]["status"]) == {"ok"}
    stage14 = summary[summary["stage"] == 14].iloc[0]
    stage15 = summary[summary["stage"] == 15].iloc[0]
    assert "--cohort" not in stage14["command"]
    assert "--cohort nlsy79" in stage15["command"]


def test_script_13_range_run_skips_reserved_stage_13(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--from-stage",
            "12",
            "--to-stage",
            "14",
            "--dry-run",
        ],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    stage13 = summary[summary["stage"] == 13].iloc[0]
    assert stage13["status"] == "skipped"
    assert "No applicable command" in stage13["note"]
    assert set(summary["stage"]) == {12, 13, 14}


def test_script_13_executes_stage_12(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/nlsy79.yml",
        """
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  subtests: [GS, AR]
  missing_codes: [-1, -2]
""",
    )
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"person_id": 1, "sex": "F", "age": 20, "GS": 1.0, "AR": -1},
            {"person_id": 2, "sex": "M", "age": 21, "GS": None, "AR": 2.0},
        ]
    ).to_csv(root / "data/processed/nlsy79_cfa.csv", index=False)

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--stage",
            "12",
        ],
        cwd=_repo_root(),
        check=True,
    )

    assert (root / "outputs/tables/missingness_diagnostics.csv").exists()
    assert (root / "outputs/figures/missingness_heatmap.png").exists()
    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    stage12_rows = summary[summary["stage"] == 12]
    assert not stage12_rows.empty
    assert set(stage12_rows["status"]) == {"ok"}


def test_script_13_executes_multi_stage_subset(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/nlsy79.yml",
        """
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  subtests: [GS, AR]
  missing_codes: [-1, -2]
""",
    )
    _write(
        root / "config/robustness.yml",
        """
sampling_schemes: [sibling_restricted, full_cohort]
age_adjustment: [quadratic]
residualization_mode: [pooled]
inference: [robust_cluster, family_bootstrap]
weights: [unweighted, weighted]
""",
    )
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"person_id": 1, "sex": "F", "age": 20, "GS": 1.0, "AR": -1},
            {"person_id": 2, "sex": "M", "age": 21, "GS": None, "AR": 2.0},
        ]
    ).to_csv(root / "data/processed/nlsy79_cfa.csv", index=False)
    tables = root / "outputs/tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "n_input": 10,
                "n_after_age": 8,
                "n_after_test_rule": 7,
                "n_after_dedupe": 6,
            }
        ]
    ).to_csv(tables / "sample_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "subtest": "GS",
                "predictor_col": "age",
                "n_used": 6,
                "r2": 0.8,
                "beta0": 0.0,
                "beta1": 0.1,
                "beta2": 0.0,
                "resid_sd": 1.1,
                "outliers_3sd": 0,
            }
        ]
    ).to_csv(tables / "residualization_diagnostics_all.csv", index=False)
    pd.DataFrame(
        [
            {"cohort": "nlsy79", "model_step": "configural", "cfi": 0.95, "rmsea": 0.04, "srmr": 0.03},
            {"cohort": "nlsy79", "model_step": "strict", "cfi": 0.93, "rmsea": 0.05, "srmr": 0.035},
        ]
    ).to_csv(tables / "nlsy79_invariance_summary.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "d_g": 0.2, "SE": 0.08, "ci_low": 0.04, "ci_high": 0.36}]
    ).to_csv(tables / "g_mean_diff.csv", index=False)
    pd.DataFrame(
        [{"cohort": "nlsy79", "VR_g": 1.2, "SE_logVR": 0.1, "ci_low": 1.0, "ci_high": 1.5}]
    ).to_csv(tables / "g_variance_ratio.csv", index=False)

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--stage",
            "11",
            "--stage",
            "12",
            "--robustness-reruns",
            "--robustness-sampling",
            "--robustness-model-form",
            "--robustness-age-adjustment",
            "--robustness-inference",
            "--robustness-weights",
        ],
        cwd=_repo_root(),
        check=True,
    )

    assert (root / "outputs/tables/robustness_sampling.csv").exists()
    assert (root / "outputs/tables/robustness_inference.csv").exists()
    assert (root / "outputs/tables/missingness_diagnostics.csv").exists()
    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    rows = summary[summary["stage"].isin([11, 12])]
    assert set(rows["status"]) == {"ok"}
    stage11_row = rows[rows["stage"] == 11].iloc[0]
    assert "--rerun-robustness" in stage11_row["command"]
    assert "--rerun-sampling-variants" in stage11_row["command"]


def test_script_13_skip_successful_reuses_latest_summary(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/nlsy79.yml",
        """
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  subtests: [GS, AR]
  missing_codes: [-1, -2]
""",
    )
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"person_id": 1, "sex": "F", "age": 20, "GS": 1.0, "AR": -1},
            {"person_id": 2, "sex": "M", "age": 21, "GS": None, "AR": 2.0},
        ]
    ).to_csv(root / "data/processed/nlsy79_cfa.csv", index=False)

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    base_args = [
        sys.executable,
        str(script),
        "--project-root",
        str(root),
        "--cohort",
        "nlsy79",
        "--stage",
        "12",
    ]
    subprocess.run(base_args, cwd=_repo_root(), check=True)
    subprocess.run(base_args + ["--skip-successful"], cwd=_repo_root(), check=True)

    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    stage12_rows = summary[summary["stage"] == 12]
    assert len(stage12_rows) == 1
    assert stage12_rows.iloc[0]["status"] == "skipped_successful"


def test_script_13_multi_cohort_stage_12_orchestration(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/nlsy79.yml",
        """
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  subtests: [GS, AR]
  missing_codes: [-1, -2]
""",
    )
    _write(
        root / "config/nlsy97.yml",
        """
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  subtests: [GS, AR]
  missing_codes: [-1, -2]
""",
    )
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/nlsy79").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/nlsy97").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"person_id": 1, "sex": "F", "age": 20, "GS": 1.0, "AR": 2.0},
            {"person_id": 2, "sex": "M", "age": 21, "GS": None, "AR": 2.0},
            {"person_id": 3, "sex": "F", "age": 19, "GS": 3.0, "AR": -2},
        ]
    ).to_csv(root / "data/processed/nlsy79_cfa.csv", index=False)
    pd.DataFrame(
        [
            {"person_id": 10, "sex": "F", "age": 21, "GS": 2.0, "AR": 3.0},
            {"person_id": 11, "sex": "M", "age": 22, "GS": -1, "AR": 1.0},
            {"person_id": 12, "sex": "M", "age": 23, "GS": 2.0, "AR": 2.0},
        ]
    ).to_csv(root / "data/processed/nlsy97_cfa.csv", index=False)

    pd.DataFrame(
        [
            {"person_id": 1, "sex": "F", "age": 20},
            {"person_id": 2, "sex": "M", "age": 21},
            {"person_id": 3, "sex": "F", "age": 19},
        ]
    ).to_csv(root / "data/interim/nlsy79/panel_extract.csv", index=False)
    pd.DataFrame(
        [
            {"person_id": 10, "sex": "F", "age": 21},
            {"person_id": 11, "sex": "M", "age": 22},
            {"person_id": 12, "sex": "M", "age": 23},
        ]
    ).to_csv(root / "data/interim/nlsy97/panel_extract.csv", index=False)

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--cohort",
            "nlsy97",
            "--stage",
            "12",
        ],
        cwd=_repo_root(),
        check=True,
    )

    missingness = pd.read_csv(root / "outputs/tables/missingness_diagnostics.csv")
    assert set(missingness["cohort"]) == {"nlsy79", "nlsy97"}
    selection = pd.read_csv(root / "outputs/tables/inclusion_exclusion_diagnostics.csv")
    assert len(selection[selection["cohort"] == "nlsy79"]) > 0
    assert len(selection[selection["cohort"] == "nlsy97"]) > 0
    assert (root / "outputs/figures/missingness_heatmap.png").exists()

    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    rows = summary[summary["stage"] == 12]
    assert len(rows) == 1
    assert rows.iloc[0]["status"] == "ok"
    assert rows.iloc[0]["cohort_scope"] == "nlsy79,nlsy97"
    assert rows.iloc[0]["command"].count("--cohort") == 2
    assert "--cohort nlsy79" in rows.iloc[0]["command"]
    assert "--cohort nlsy97" in rows.iloc[0]["command"]


def test_script_13_stage_11_rerun_flags_pass_to_stage_command_and_summary(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--stage",
            "11",
            "--dry-run",
            "--robustness-reruns",
            "--robustness-sampling",
            "--robustness-model-form",
            "--robustness-age-adjustment",
            "--robustness-inference",
            "--robustness-weights",
        ],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    stage_rows = summary[summary["stage"] == 11]
    assert list(stage_rows["status"]) == ["planned"]
    command = stage_rows.iloc[0]["command"]
    assert "--rerun-robustness" in command
    assert "--rerun-sampling-variants" in command
    assert "--rerun-model-forms" in command
    assert "--rerun-age-adjustment-variants" in command
    assert "--rerun-inference-variants" in command
    assert "--rerun-weight-variants" in command
    assert "--cohort nlsy79" in command


def test_script_13_stage_07_warning_policy_flags_pass_to_stage_command(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--stage",
            "7",
            "--dry-run",
            "--enforce-warning-policy",
            "--warning-policy-threshold",
            "fail",
        ],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    stage_rows = summary[summary["stage"] == 7]
    assert list(stage_rows["status"]) == ["planned"]
    command = stage_rows.iloc[0]["command"]
    assert "--enforce-warning-policy" in command
    assert "--warning-policy-threshold fail" in command
    assert "--cohort nlsy79" in command


def test_script_13_build_post_stage_15_commands(tmp_path: Path) -> None:
    module = _pipeline_script_module()
    plan_root = _repo_root()

    args = SimpleNamespace(
        post_claim_verdicts=True,
        post_inference_ci_check=True,
        post_report_sections=True,
    )
    plans = module._build_post_stage_15_commands(
        repo_root=plan_root,
        target_root=tmp_path,
        args=args,
    )

    assert [plan.stage for plan in plans] == [29, 30, 31]
    assert plans[0].script == "29_build_claim_verdicts.py"
    assert plans[0].cohort_scope == "post-stage15"
    assert plans[0].argv == [
        sys.executable,
        str(plan_root / "scripts" / "29_build_claim_verdicts.py"),
        "--project-root",
        str(tmp_path),
    ]
    assert plans[1].script == "30_check_inference_ci_coherence.py"
    assert plans[1].cohort_scope == "post-stage15"
    assert plans[1].argv == [
        sys.executable,
        str(plan_root / "scripts" / "30_check_inference_ci_coherence.py"),
        "--project-root",
        str(tmp_path),
    ]
    assert plans[2].script == "31_export_report_sections.py"
    assert plans[2].cohort_scope == "post-stage15"
    assert plans[2].argv == [
        sys.executable,
        str(plan_root / "scripts" / "31_export_report_sections.py"),
        "--project-root",
        str(tmp_path),
    ]


def test_script_13_dry_run_includes_post_stage_15_hooks(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")

    script = _repo_root() / "scripts" / "13_run_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--stage",
            "15",
            "--dry-run",
            "--post-claim-verdicts",
            "--post-inference-ci-check",
            "--post-report-sections",
        ],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    stage15_rows = summary[summary["stage"].isin([15, 29, 30, 31])]
    assert list(stage15_rows["stage"]) == [15, 29, 30, 31]
    assert set(stage15_rows["status"]) == {"planned"}
    stage15 = stage15_rows[stage15_rows["stage"] == 15].iloc[0]
    stage29 = stage15_rows[stage15_rows["stage"] == 29].iloc[0]
    stage30 = stage15_rows[stage15_rows["stage"] == 30].iloc[0]
    stage31 = stage15_rows[stage15_rows["stage"] == 31].iloc[0]
    assert "15_specification_curve_summary.py" in stage15["command"]
    assert stage29["script"] == "29_build_claim_verdicts.py"
    assert stage30["script"] == "30_check_inference_ci_coherence.py"
    assert stage31["script"] == "31_export_report_sections.py"
    assert str(root) in stage29["command"]
    assert str(root) in stage30["command"]
    assert str(root) in stage31["command"]
    assert "project-root" in stage29["command"]
    assert "project-root" in stage30["command"]
    assert "project-root" in stage31["command"]


def test_script_13_run_command_records_timeout_path(tmp_path: Path) -> None:
    module = _pipeline_script_module()
    log_file = tmp_path / "timeout_stage.log"
    plan = module.CommandPlan(
        stage=99,
        script="timeout-test",
        cohort_scope="nlsy79",
        argv=[sys.executable, "-c", "import time; time.sleep(0.2)"],
    )
    returncode, status, note = module._run_command(
        plan=plan,
        repo_root=_repo_root(),
        target_root=tmp_path,
        log_file=log_file,
        timeout_seconds=0.05,
    )
    assert status == "timeout"
    assert returncode is None
    assert note == "timed out after 0.05 seconds"
    assert log_file.exists()
    assert "timed out after 0.05 seconds" in log_file.read_text()


def test_script_13_run_command_records_success_path(tmp_path: Path) -> None:
    module = _pipeline_script_module()
    log_file = tmp_path / "success_stage.log"
    plan = module.CommandPlan(
        stage=99,
        script="success-test",
        cohort_scope="nlsy79",
        argv=[sys.executable, "-c", "print('ok')"],
    )
    returncode, status, note = module._run_command(
        plan=plan,
        repo_root=_repo_root(),
        target_root=tmp_path,
        log_file=log_file,
        timeout_seconds=1.0,
    )
    assert status == "ok"
    assert returncode == 0
    assert note == ""
    assert log_file.exists()
    assert "ok" in log_file.read_text()


def test_script_13_timeout_written_in_pipeline_summary(tmp_path: Path, monkeypatch: object) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    module = _pipeline_script_module()
    long_plan = module.CommandPlan(
        stage=2,
        script="timeout-command.py",
        cohort_scope="nlsy79",
        argv=[sys.executable, "-c", "import time; time.sleep(0.2)"],
    )

    def fake_build_stage_commands(
        *, stage: int, repo_root: Path, target_root: Path, cohorts: list[str], args: object
    ) -> list[module.CommandPlan]:
        return [long_plan]

    monkeypatch.setattr(module, "_build_stage_commands", fake_build_stage_commands)
    monkeypatch.setattr(module.sys, "argv", [
        "13_run_pipeline.py",
        "--project-root",
        str(root),
        "--cohort",
        "nlsy79",
        "--stage",
        "2",
        "--stage-timeout-seconds",
        "0.05",
    ])
    code = module.main()
    assert code == 1

    summary = pd.read_csv(_latest_summary(root / "outputs/logs/pipeline"))
    assert list(summary["status"]) == ["timeout"]
    assert summary["note"].iloc[0] == "timed out after 0.05 seconds"


def test_script_13_parse_stage_timeout_overrides() -> None:
    module = _pipeline_script_module()
    parsed = module._parse_stage_timeout_overrides(["3=1200", "7 = 5.5"])
    assert parsed == {3: 1200.0, 7: 5.5}

    for raw in ["3-1200", "abc=foo", "16=10"]:
        with pytest.raises(ValueError):
            module._parse_stage_timeout_overrides([raw])
