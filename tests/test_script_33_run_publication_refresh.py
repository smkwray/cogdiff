from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "33_run_publication_refresh.py"
    spec = importlib.util.spec_from_file_location("script33_run_publication_refresh", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_script_33_build_chain_plan_order():
    module = _module()
    repo_root = _repo_root()
    plans = module._build_plan(
        repo_root=repo_root,
        project_root_path=repo_root / "project",
        python_executable="/usr/bin/python3",
        n_bootstrap=499,
        engine="sem_refit",
        sem_timeout_seconds=60.0,
        min_success_share=0.90,
    )

    assert [plan.script for plan in plans] == [
        "20_run_inference_bootstrap.py",
        "11_robustness_suite.py",
        "15_specification_curve_summary.py",
        "29_build_claim_verdicts.py",
        "30_check_inference_ci_coherence.py",
        "24_build_publication_results_lock.py",
        "31_export_report_sections.py",
    ]
    assert plans[0].argv == [
        "/usr/bin/python3",
        str(repo_root / "scripts" / "20_run_inference_bootstrap.py"),
        "--project-root",
        str(repo_root / "project"),
        "--variant-token",
        "family_bootstrap",
        "--engine",
        "sem_refit",
        "--n-bootstrap",
        "499",
        "--min-success-share",
        "0.9",
        "--sem-timeout-seconds",
        "60.0",
    ]
    assert "--all" in plans[2].argv


def test_script_33_dry_run_outputs_ordered_plan(tmp_path: Path, monkeypatch: object, capsys: object) -> None:
    module = _module()
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")

    def fail_if_executed(*args, **kwargs):
        raise AssertionError("dry-run should never execute any publication step")

    monkeypatch.setattr(module, "_run_command", fail_if_executed)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "33_run_publication_refresh.py",
            "--project-root",
            str(project_root),
            "--python-executable",
            "/usr/bin/python3",
            "--dry-run",
        ],
    )

    code = module.main()
    captured = capsys.readouterr().out.strip().splitlines()
    assert code == 0
    dry_run_lines = [line for line in captured if line.startswith("[dry-run]")]
    assert len(dry_run_lines) == 7
    assert "20_run_inference_bootstrap.py" in dry_run_lines[0]
    assert "11_robustness_suite.py" in dry_run_lines[1]
    assert "15_specification_curve_summary.py" in dry_run_lines[2]
    assert "29_build_claim_verdicts.py" in dry_run_lines[3]
    assert "30_check_inference_ci_coherence.py" in dry_run_lines[4]
    assert "24_build_publication_results_lock.py" in dry_run_lines[5]
    assert "31_export_report_sections.py" in dry_run_lines[6]


def test_script_33_dry_run_reflects_flag_overrides(tmp_path: Path, monkeypatch: object, capsys: object) -> None:
    module = _module()
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "33_run_publication_refresh.py",
            "--project-root",
            str(project_root),
            "--python-executable",
            "/custom/venv/bin/python",
            "--engine",
            "proxy",
            "--n-bootstrap",
            "512",
            "--sem-timeout-seconds",
            "42.5",
            "--min-success-share",
            "0.8",
            "--dry-run",
        ],
    )
    code = module.main()
    captured = capsys.readouterr().out.strip()
    assert code == 0
    assert "/custom/venv/bin/python" in captured
    assert "--engine proxy" in captured
    assert "--n-bootstrap 512" in captured
    assert "--sem-timeout-seconds 42.5" in captured
    assert "--min-success-share 0.8" in captured
