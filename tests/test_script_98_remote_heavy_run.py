from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_wrapper(*args: str) -> subprocess.CompletedProcess[str]:
    script = _repo_root() / "scripts" / "98_remote_heavy_run.sh"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _new_log_for_tag(tag: str, before: set[Path]) -> Path:
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    after = set(log_dir.glob(f"*_{tag}.log"))
    created = sorted(after - before)
    assert len(created) == 1
    return created[0]


def test_script_98_pipeline_dry_run_target_writes_expected_command() -> None:
    tag = f"ut98_stage_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remote proj/sexg",
        "--target",
        "pipeline_dry_run_07_15",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    assert "[summary] mode=dry-run" in result.stdout
    assert "target=pipeline_dry_run_07_15" in result.stdout

    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "scripts/13_run_pipeline.py" in log_text
        assert "--all --from-stage 07 --to-stage 15 --sem-dry-run" in log_text
        assert "--all" in log_text
        assert "OMP_NUM_THREADS=1" in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_bootstrap_target_dry_run_writes_expected_command() -> None:
    tag = f"ut98_boot_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remoteproj/sexg",
        "--target",
        "bootstrap_inference",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    assert "[summary] mode=dry-run" in result.stdout
    assert "target=bootstrap_inference" in result.stdout

    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "scripts/20_run_inference_bootstrap.py" in log_text
        assert "--variant-token family_bootstrap" in log_text
        assert "--n-bootstrap 100" in log_text
        assert "--sem-jobs 16 --sem-threads-per-job 1" in log_text
        assert "--cohort nlsy79" in log_text
        assert "--cohort nlsy97" in log_text
        assert "--skip-successful" in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_bootstrap_target_can_disable_skip_successful_flag() -> None:
    tag = f"ut98_bootnoskip_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remoteproj/sexg",
        "--target",
        "bootstrap_inference",
        "--no-skip-successful",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "--n-bootstrap 100" in log_text
        assert "--skip-successful" not in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_sync_code_flag_is_logged_in_dry_run() -> None:
    tag = f"ut98_sync_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remoteproj/sexg",
        "--target",
        "bootstrap_inference",
        "--sync-code",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "[local] sync_code=1" in log_text
        assert "planned rsync commands" in log_text
        assert "rsync -az" in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_sync_back_flag_is_logged_in_dry_run() -> None:
    tag = f"ut98_syncback_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remote proj/sexg",
        "--target",
        "bootstrap_inference",
        "--sync-back",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "[local] sync_back=1" in log_text
        assert "[local] sync_back_bootstrap_dirs=0" in log_text
        assert "planned sync-back commands" in log_text
        assert "g_mean_diff_family_bootstrap.csv" in log_text
        assert "publication_results_lock.zip" in log_text
        assert "results_snapshot.md" in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_sync_back_bootstrap_dirs_includes_selected_cohort() -> None:
    tag = f"ut98_syncbackdir_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remote proj/sexg",
        "--target",
        "bootstrap_inference",
        "--cohort",
        "nlsy79",
        "--sync-back",
        "--sync-back-bootstrap-dirs",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "[local] sync_back=1" in log_text
        assert "[local] sync_back_bootstrap_dirs=1" in log_text
        assert "outputs/model_fits/bootstrap_inference/nlsy79/" in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_rejects_sync_back_bootstrap_dirs_without_sync_back() -> None:
    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remoteproj/sexg",
        "--target",
        "bootstrap_inference",
        "--sync-back-bootstrap-dirs",
    )
    assert result.returncode == 2
    assert "--sync-back-bootstrap-dirs requires --sync-back" in result.stderr


def test_script_98_bootstrap_target_remote_python_override_is_applied() -> None:
    tag = f"ut98_bootrpy_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remoteproj/sexg",
        "--target",
        "bootstrap_inference",
        "--remote-python",
        "/opt/venvs/sexg/bin/python",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "/opt/venvs/sexg/bin/python scripts/20_run_inference_bootstrap.py" in log_text
        assert "$REMOTE_PYTHON scripts/20_run_inference_bootstrap.py" not in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_publication_lock_target_writes_expected_command() -> None:
    tag = f"ut98_pub_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remoteproj/sexg",
        "--target",
        "publication_lock",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    assert "[summary] mode=dry-run" in result.stdout
    assert "target=publication_lock" in result.stdout

    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "scripts/09_results_and_figures.py --all" in log_text
        assert "scripts/93_build_results_snapshot.py" in log_text
        assert "scripts/24_build_publication_results_lock.py" in log_text
        assert "scripts/25_verify_publication_snapshot.py" in log_text
        assert "&&" in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_spec_curve_summary_target_writes_expected_command() -> None:
    tag = f"ut98_spec_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remoteproj/sexg",
        "--target",
        "spec_curve_summary",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    assert "[summary] mode=dry-run" in result.stdout
    assert "target=spec_curve_summary" in result.stdout

    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "scripts/15_specification_curve_summary.py --all" in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_repro_report_target_writes_expected_command() -> None:
    tag = f"ut98_repro_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remoteproj/sexg",
        "--target",
        "repro_report",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    assert "[summary] mode=dry-run" in result.stdout
    assert "target=repro_report" in result.stdout

    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "scripts/14_reproducibility_report.py" in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_legacy_stage_target_alias_still_works() -> None:
    tag = f"ut98_alias_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remote proj/sexg",
        "--target",
        "stage07_15_dry_run",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    assert "target=stage07_15_dry_run" in result.stdout
    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "--from-stage 07 --to-stage 15 --sem-dry-run" in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_rejects_target_and_cmd_together() -> None:
    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remoteproj/sexg",
        "--target",
        "stage07_15_dry_run",
        "--cmd",
        "echo hello",
    )
    assert result.returncode == 2
    assert "either --target or --cmd" in result.stderr


def test_script_98_remote_python_override_rewrites_python_prefix_in_cmd() -> None:
    tag = f"ut98_rpy_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remoteproj/sexg",
        "--cmd",
        "python scripts/11_robustness_suite.py --project-root /tmp/remoteproj/sexg",
        "--remote-python",
        "/opt/venvs/sexg/bin/python",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "/opt/venvs/sexg/bin/python scripts/11_robustness_suite.py" in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_env_exports_are_written_to_remote_script() -> None:
    tag = f"ut98_env_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob(f"*_{tag}.log"))

    result = _run_wrapper(
        "--remote-dir",
        "/tmp/remoteproj/sexg",
        "--cmd",
        "echo hello",
        "--env",
        "NLS_REMOTE_MODE=heavy",
        "--env",
        "RUN_LABEL=nightly rerun",
        "--dry-run",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    assert "[summary] mode=dry-run" in result.stdout
    log_path = _new_log_for_tag(tag, before)
    try:
        log_text = log_path.read_text(encoding="utf-8")
        assert "export NLS_REMOTE_MODE=heavy" in log_text
        assert "export RUN_LABEL=nightly\\ rerun" in log_text
    finally:
        log_path.unlink(missing_ok=True)


def test_script_98_self_test_prints_remote_script_without_writing_log() -> None:
    tag = f"ut98_self_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob("*.log"))

    result = _run_wrapper(
        "--self-test",
        "--remote-dir",
        "/tmp/remote proj/sexg",
        "--cmd",
        "echo ok",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    assert "cd '/tmp/remote proj/sexg'" in result.stdout
    assert "export OMP_NUM_THREADS=1" in result.stdout
    assert "echo ok" in result.stdout
    assert "[summary] mode=self-test" in result.stdout
    assert "log=<none>" in result.stdout

    after = set(log_dir.glob("*.log"))
    assert after == before


def test_script_98_list_targets_prints_expected_names_without_writing_log() -> None:
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob("*.log")) if log_dir.exists() else set()
    result = _run_wrapper("--list-targets")
    assert result.returncode == 0
    assert "pipeline_dry_run_07_15" in result.stdout
    assert "bootstrap_inference" in result.stdout
    assert "publication_lock" in result.stdout
    assert "spec_curve_summary" in result.stdout
    assert "repro_report" in result.stdout
    assert "stage07_15_dry_run" in result.stdout
    after = set(log_dir.glob("*.log")) if log_dir.exists() else set()
    assert after == before


def test_script_98_print_log_path_exits_without_writing_log(tmp_path: Path) -> None:
    tag = f"ut98_logpath_{uuid.uuid4().hex[:10]}"
    log_dir = _repo_root() / "outputs" / "logs" / "remote_runs"
    before = set(log_dir.glob("*.log")) if log_dir.exists() else set()

    result = _run_wrapper(
        "--print-log-path",
        "--remote-dir",
        str(tmp_path / "remote proj/sexg"),
        "--cmd",
        "echo ok",
        "--tag",
        tag,
    )
    assert result.returncode == 0
    assert result.stdout.strip().endswith(f"_{tag}.log")

    after = set(log_dir.glob("*.log")) if log_dir.exists() else set()
    assert after == before
