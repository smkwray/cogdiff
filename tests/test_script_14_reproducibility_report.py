from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_script_14_writes_hash_and_provenance_outputs(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(
        root / "config/paths.yml",
        "raw_dir: data/raw\ninterim_dir: data/interim\nprocessed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write(root / "data/raw/manifest.json", '{"cohort":"nlsy79"}\n')
    _write(root / "data/processed/nlsy79_cfa.csv", "person_id,sex,GS\n1,F,1.0\n")
    _write(root / "outputs/tables/g_mean_diff.csv", "cohort,d_g,SE,ci_low,ci_high\nnlsy79,0.2,0.1,0.0,0.4\n")
    _write(root / "outputs/figures/robustness_forestplot.png", "fake_png_bytes")
    _write(
        root / "outputs/logs/pipeline/20260220_010203_pipeline_run_summary.csv",
        "stage,script,cohort_scope,status,returncode,command,log_file,note\n"
        f"11,11_robustness_suite.py,nlsy79,ok,0,{sys.executable} {root / 'scripts/11_robustness_suite.py'} --project-root {root},,\n",
    )
    _write(
        root / "outputs/tables/robustness_rerun_log.csv",
        "cohort,robustness_family,variant_token,status,command,return_code,elapsed_seconds,error,source_paths\n"
        "__all__,sampling,one_pair_per_family,success,python scripts/fake.py,0,0.1,,outputs/tables/sample_counts_one_pair_per_family.csv\n",
    )

    script = _repo_root() / "scripts" / "14_reproducibility_report.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        check=True,
    )

    hashes_path = root / "outputs/tables/reproducibility_hashes.csv"
    determinism_path = root / "outputs/tables/reproducibility_determinism_audit.csv"
    provenance_path = root / "outputs/tables/reproducibility_command_provenance.csv"
    environment_path = root / "outputs/tables/reproducibility_environment.json"
    summary_path = root / "outputs/tables/reproducibility_report_summary.json"

    assert hashes_path.exists()
    assert determinism_path.exists()
    assert provenance_path.exists()
    assert environment_path.exists()
    assert summary_path.exists()

    hashes = pd.read_csv(hashes_path)
    assert set(["artifact_group", "relative_path", "size_bytes", "modified_utc", "sha256"]).issubset(hashes.columns)
    assert len(hashes[hashes["relative_path"] == "config/paths.yml"]) == 1
    assert len(hashes[hashes["relative_path"] == "outputs/tables/g_mean_diff.csv"]) == 1
    assert (hashes["sha256"].astype(str).str.len() == 64).all()

    provenance = pd.read_csv(provenance_path)
    assert set(["source", "command", "status", "script"]).issubset(provenance.columns)
    assert (provenance["source"] == "pipeline_summary").any()
    assert (provenance["source"] == "robustness_rerun").any()
    pipeline_cmd = str(provenance.loc[provenance["source"] == "pipeline_summary", "command"].iloc[0])
    assert "<PROJECT_ROOT>" in pipeline_cmd
    assert str(root) not in pipeline_cmd

    determinism = pd.read_csv(determinism_path)
    assert set(["relative_path", "status", "current_sha256", "previous_sha256"]).issubset(determinism.columns)
    assert set(determinism["status"]) == {"no_previous", "missing_current"}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_hashed_files"] >= 1
    assert summary["n_provenance_rows"] >= 2
    assert summary["project_root"] == "."
    assert summary["environment_json"] == "outputs/tables/reproducibility_environment.json"
    assert summary["determinism_audit_csv"] == "outputs/tables/reproducibility_determinism_audit.csv"
    assert summary["determinism_no_previous_count"] >= 1

    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    assert "environment" in environment
    assert "git" in environment
    assert isinstance(environment["environment"].get("python_version"), str)
    assert isinstance(environment["environment"].get("pip_freeze"), list)
    assert "rscript_executable" in environment["environment"]
    assert "r_session_info" in environment["environment"]


def test_script_14_handles_missing_optional_sources(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(
        root / "config/paths.yml",
        "raw_dir: data/raw\ninterim_dir: data/interim\nprocessed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write(root / "data/processed/nlsy79_cfa.csv", "person_id,sex,GS\n1,F,1.0\n")

    script = _repo_root() / "scripts" / "14_reproducibility_report.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        check=True,
    )

    hashes = pd.read_csv(root / "outputs/tables/reproducibility_hashes.csv")
    determinism = pd.read_csv(root / "outputs/tables/reproducibility_determinism_audit.csv")
    provenance = pd.read_csv(root / "outputs/tables/reproducibility_command_provenance.csv")
    environment = json.loads((root / "outputs/tables/reproducibility_environment.json").read_text(encoding="utf-8"))
    assert len(hashes) >= 1
    assert len(determinism) == 4
    assert provenance.empty
    assert "environment" in environment


def test_script_14_determinism_audit_marks_changes_against_previous_hashes(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(
        root / "config/paths.yml",
        "raw_dir: data/raw\ninterim_dir: data/interim\nprocessed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write(root / "data/processed/nlsy79_cfa.csv", "person_id,sex,GS\n1,F,1.0\n")
    _write(root / "outputs/tables/sample_counts.csv", "cohort,n_input\nnlsy79,10\n")
    _write(root / "outputs/tables/g_mean_diff.csv", "cohort,d_g,SE,ci_low,ci_high\nnlsy79,0.2,0.1,0.0,0.4\n")
    _write(root / "outputs/tables/g_variance_ratio.csv", "cohort,VR_g,SE_logVR,ci_low,ci_high\nnlsy79,1.1,0.1,0.9,1.4\n")

    script = _repo_root() / "scripts" / "14_reproducibility_report.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        check=True,
    )

    _write(root / "outputs/tables/g_mean_diff.csv", "cohort,d_g,SE,ci_low,ci_high\nnlsy79,0.25,0.1,0.05,0.45\n")
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        check=True,
    )

    audit = pd.read_csv(root / "outputs/tables/reproducibility_determinism_audit.csv")
    g_mean_row = audit[audit["relative_path"] == "outputs/tables/g_mean_diff.csv"].iloc[0]
    assert g_mean_row["status"] == "changed"
    sample_counts_row = audit[audit["relative_path"] == "outputs/tables/sample_counts.csv"].iloc[0]
    assert sample_counts_row["status"] == "unchanged"
