from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from nls_pipeline import preflight


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_script_module():
    path = _repo_root() / "scripts" / "16_preflight_dependencies.py"
    spec = importlib.util.spec_from_file_location("script16_preflight_dependencies", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _command_runner(
    *,
    package_state: dict[str, bool] | None = None,
    missing_r: bool = False,
) -> tuple[callable[[list[str]], tuple[int, str]], dict[str, str]]:
    state = {
        "lavaan": True,
        "NlsyLinks": True,
    }
    if package_state:
        state.update(package_state)

    log: dict[str, str] = {}

    def _runner(command: list[str]) -> tuple[int, str]:
        if missing_r and command[:2] == ["Rscript", "--version"]:
            raise FileNotFoundError("Rscript not found")
        if command[0] != "Rscript":
            raise AssertionError(f"Unexpected command: {command}")

        if command[:2] == ["Rscript", "--version"]:
            log["rscript"] = command[1]
            return 0, "Rscript (R) version 4.3.2 (2025-01-01)"
        if command[1] == "-e":
            expression = command[2]
            if "lavaan" in expression:
                log["lavaan"] = expression
                if state["lavaan"]:
                    return 0, "lavaan available"
                return 1, "Error: package lavaan missing"
            if "NlsyLinks" in expression:
                log["NlsyLinks"] = expression
                if state["NlsyLinks"]:
                    return 0, "NlsyLinks available"
                return 1, "Error: package NlsyLinks missing"
        return 1, f"Unexpected command: {' '.join(command)}"

    return _runner, log


def test_script_16_preflight_strict_passes_with_required_inputs(tmp_path: Path, monkeypatch: object) -> None:
    root = tmp_path.resolve()
    _write(root / "data/interim/links/links79_pair_expanded.csv", "a,b\n1,2\n")
    module = _load_script_module()
    runner, command_log = _command_runner()
    monkeypatch.setattr(module.preflight, "_run_command", runner)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "16_preflight_dependencies.py",
            "--project-root",
            str(root),
            "--strict",
        ],
    )

    code = module.main()
    assert code == 0
    status_path = root / "outputs/tables/preflight_status.csv"
    summary_path = root / "outputs/tables/preflight_summary.json"
    assert status_path.exists()
    assert summary_path.exists()

    status = pd.read_csv(status_path)
    assert status.shape[0] >= 4
    assert set(["python.runtime", "rscript.availability"]).issubset(set(status["check"]))
    assert status.loc[status["check"] == "rscript.availability", "status"].iloc[0] == "pass"
    assert (
        status.loc[
            status["check"] == "file.data/interim/links/links79_pair_expanded.csv",
            "status",
        ].iloc[0]
        == "pass"
    )
    assert (
        status.loc[
            status["check"] == "file.data/interim/links/links79_pair_expanded.csv",
            "value",
        ].iloc[0]
        == "data/interim/links/links79_pair_expanded.csv"
    )
    assert "rpackage.lavaan" in set(status["check"])
    assert "rpackage.NlsyLinks" in set(status["check"])
    assert command_log == {
        "rscript": "--version",
        "lavaan": "if (!requireNamespace('lavaan', quietly = TRUE)) { quit(status = 1) }",
        "NlsyLinks": "if (!requireNamespace('NlsyLinks', quietly = TRUE)) { quit(status = 1) }",
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["project_root"] == "."
    assert summary["overall"] == "pass"
    assert summary["strict"] is True
    assert summary["critical_failures"] == []
    assert summary["critical_failure_count"] == 0


def test_script_16_preflight_strict_fails_when_r_missing(tmp_path: Path, monkeypatch: object) -> None:
    root = tmp_path.resolve()
    _write(root / "data/interim/links/links79_pair_expanded.csv", "a,b\n1,2\n")
    module = _load_script_module()
    runner, _ = _command_runner(missing_r=True)
    monkeypatch.setattr(module.preflight, "_run_command", runner)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "16_preflight_dependencies.py",
            "--project-root",
            str(root),
            "--strict",
        ],
    )

    code = module.main()
    assert code == 1

    summary = json.loads((root / "outputs/tables/preflight_summary.json").read_text(encoding="utf-8"))
    assert summary["overall"] == "fail"
    assert "rscript.availability" in summary["critical_failures"]

    status = pd.read_csv(root / "outputs/tables/preflight_status.csv")
    assert (
        status.loc[status["check"] == "rscript.availability", "status"].iloc[0]
        == "fail"
    )
    assert (
        status.loc[status["check"] == "rpackage.lavaan", "status"].iloc[0]
        == "skipped"
    )


def test_script_16_preflight_nonstrict_allows_critical_failures(tmp_path: Path, monkeypatch: object) -> None:
    root = tmp_path.resolve()
    _write(root / "data/interim/links/links79_pair_expanded.csv", "a,b\n1,2\n")
    module = _load_script_module()
    runner, _ = _command_runner(missing_r=True)
    monkeypatch.setattr(module.preflight, "_run_command", runner)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "16_preflight_dependencies.py",
            "--project-root",
            str(root),
        ],
    )

    code = module.main()
    assert code == 0
    summary = json.loads((root / "outputs/tables/preflight_summary.json").read_text(encoding="utf-8"))
    assert summary["overall"] == "fail"
    assert summary["strict"] is False


def test_script_16_missing_r_package_checks_are_critical(tmp_path: Path, monkeypatch: object) -> None:
    root = tmp_path.resolve()
    _write(root / "data/interim/links/links79_pair_expanded.csv", "a,b\n1,2\n")
    module = _load_script_module()
    runner, _ = _command_runner(package_state={"NlsyLinks": False})
    monkeypatch.setattr(module.preflight, "_run_command", runner)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "16_preflight_dependencies.py",
            "--project-root",
            str(root),
            "--strict",
            "--all",
        ],
    )

    code = module.main()
    assert code == 1

    status = pd.read_csv(root / "outputs/tables/preflight_status.csv")
    assert status.loc[status["check"] == "rpackage.NlsyLinks", "status"].iloc[0] == "fail"
    assert (
        bool(status.loc[status["check"] == "rpackage.NlsyLinks", "critical"].iloc[0]) is True
    )
    summary = json.loads((root / "outputs/tables/preflight_summary.json").read_text(encoding="utf-8"))
    assert summary["overall"] == "fail"
    assert "rpackage.NlsyLinks" in summary["critical_failures"]


def test_preflight_module_collects_required_rows(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "data/interim/links/links79_pair_expanded.csv", "a,b\n1,2\n")

    runner, _ = _command_runner()
    checks = preflight.collect_preflight_checks(
        root,
        cohorts=None,
        all_cohorts=False,
        command_runner=runner,
    )
    assert any(item.check == "file.data/interim/links/links79_pair_expanded.csv" and item.status == "pass" for item in checks)
    assert any(item.check == "rpackage.lavaan" and item.status == "pass" for item in checks)
    assert any(item.check == "rpackage.NlsyLinks" and item.status == "pass" for item in checks)

    _, _, summary = preflight.write_preflight_outputs(
        root,
        checks,
        strict=False,
    )
    assert summary["strict"] is False
    assert (root / "outputs/tables/preflight_summary.json").exists()
    assert (root / "outputs/tables/preflight_status.csv").exists()


def test_preflight_module_reads_required_paths_and_packages_from_config(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(
        root / "config/preflight.yml",
        """
required_paths:
  __all__:
    - data/interim/links/custom_links.csv
required_r_packages:
  - lavaan
""",
    )
    _write(root / "data/interim/links/custom_links.csv", "a,b\n1,2\n")

    runner, _ = _command_runner()
    checks = preflight.collect_preflight_checks(
        root,
        cohorts=["nlsy79"],
        all_cohorts=False,
        command_runner=runner,
    )
    names = {item.check for item in checks}
    assert "file.data/interim/links/custom_links.csv" in names
    assert "file.data/interim/links/links79_pair_expanded.csv" not in names
    assert "rpackage.lavaan" in names
    assert "rpackage.NlsyLinks" not in names
