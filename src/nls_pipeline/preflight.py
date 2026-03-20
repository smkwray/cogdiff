from __future__ import annotations

import csv
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .io import dump_json, load_yaml, project_root, relative_path

CheckRunner = Callable[[list[str]], tuple[int, str]]


@dataclass(frozen=True)
class PreflightCheck:
    check: str
    status: str
    critical: bool
    message: str
    command: str | None = None
    value: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "status": self.status,
            "critical": self.critical,
            "message": self.message,
            "command": self.command or "",
            "value": self.value or "",
        }


REQUIRED_LINK_FILES = {
    "nlsy79": "data/interim/links/links79_pair_expanded.csv",
}

REQUIRED_R_PACKAGES = ("lavaan", "NlsyLinks")


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _resolve_outputs_dir(root: Path) -> Path:
    paths_file = root / "config" / "paths.yml"
    if paths_file.exists():
        paths_cfg = load_yaml(paths_file)
        return _resolve_path(paths_cfg.get("outputs_dir", "outputs"), root)
    return root / "outputs"


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _load_preflight_config(root: Path) -> tuple[dict[str, list[str]], tuple[str, ...]]:
    required_paths: dict[str, list[str]] = {
        cohort: [path] for cohort, path in REQUIRED_LINK_FILES.items()
    }
    r_packages: tuple[str, ...] = REQUIRED_R_PACKAGES

    cfg_path = root / "config" / "preflight.yml"
    if not cfg_path.exists():
        return required_paths, r_packages

    cfg = load_yaml(cfg_path)
    if not isinstance(cfg, dict):
        return required_paths, r_packages

    raw_paths = cfg.get("required_paths")
    if isinstance(raw_paths, dict):
        parsed: dict[str, list[str]] = {}
        for cohort, values in raw_paths.items():
            entries = _as_list(values)
            if entries:
                parsed[str(cohort)] = entries
        if parsed:
            required_paths = parsed
    elif isinstance(raw_paths, list):
        entries = _as_list(raw_paths)
        if entries:
            required_paths = {"__all__": entries}

    raw_r_packages = cfg.get("required_r_packages")
    parsed_r_packages = _as_list(raw_r_packages)
    if parsed_r_packages:
        r_packages = tuple(parsed_r_packages)

    return required_paths, r_packages


def _cohorts_to_check(*, all_cohorts: bool, cohorts: list[str] | None) -> list[str]:
    if all_cohorts or not cohorts:
        return ["nlsy79", "nlsy97", "cnlsy"]
    return sorted(set(cohorts))


def _run_command(command: list[str]) -> tuple[int, str]:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    output = ((proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")).strip()
    return proc.returncode, output


def _safe_version(raw_output: str) -> str:
    match = re.search(r"\d+\.\d+\.\d+", raw_output)
    return match.group(0) if match else raw_output


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _sanitize_text_root_paths(root: Path, text: Any) -> str:
    token = str(root.resolve())
    return str(text or "").replace(token, ".")


def check_python_runtime() -> PreflightCheck:
    import sys

    executable_name = Path(sys.executable).name or "python"
    return PreflightCheck(
        check="python.runtime",
        status="pass",
        critical=False,
        message="Python executable and version captured.",
        command=f"{shlex.quote(executable_name)} --version",
        value=sys.version.splitlines()[0],
    )


def check_rscript_availability(
    *,
    rscript: str = "Rscript",
    command_runner: CheckRunner | None = None,
) -> PreflightCheck:
    runner = command_runner or _run_command
    command = [rscript, "--version"]
    command_text = _command_text(command)
    try:
        rc, output = runner(command)
    except FileNotFoundError:
        return PreflightCheck(
            check="rscript.availability",
            status="fail",
            critical=True,
            message="Rscript not found in PATH.",
            command=command_text,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return PreflightCheck(
            check="rscript.availability",
            status="error",
            critical=True,
            message=f"Unexpected error running Rscript --version: {exc}",
            command=command_text,
        )

    if rc == 0:
        return PreflightCheck(
            check="rscript.availability",
            status="pass",
            critical=True,
            message=f"Rscript available: {_safe_version(output)}",
            command=command_text,
            value=_safe_version(output),
        )

    return PreflightCheck(
        check="rscript.availability",
        status="fail",
        critical=True,
        message=f"Rscript command returned non-zero status ({rc}).",
        command=command_text,
        value=output,
    )


def check_r_package(
    *,
    package: str,
    rscript: str = "Rscript",
    command_runner: CheckRunner | None = None,
) -> PreflightCheck:
    runner = command_runner or _run_command
    command = [rscript, "-e", f"if (!requireNamespace('{package}', quietly = TRUE)) {{ quit(status = 1) }}"]
    command_text = _command_text(command)
    try:
        rc, output = runner(command)
    except FileNotFoundError:
        return PreflightCheck(
            check=f"rpackage.{package}",
            status="skipped",
            critical=False,
            message="Skipped because Rscript is unavailable.",
            command=command_text,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return PreflightCheck(
            check=f"rpackage.{package}",
            status="error",
            critical=True,
            message=f"Unexpected error checking R package {package}: {exc}",
            command=command_text,
        )

    if rc == 0:
        return PreflightCheck(
            check=f"rpackage.{package}",
            status="pass",
            critical=False,
            message=f"R package available: {package}.",
            command=command_text,
        )

    return PreflightCheck(
        check=f"rpackage.{package}",
        status="fail",
        critical=True,
        message=f"R package missing: {package}.",
        command=command_text,
        value=output,
    )


def check_link_file(root: Path, rel_path: str) -> PreflightCheck:
    path = _resolve_path(rel_path, root)
    if path.exists() and path.is_file():
        return PreflightCheck(
            check=f"file.{rel_path}",
            status="pass",
            critical=True,
            message="Required input file exists.",
            command="",
            value=relative_path(root, path),
        )

    return PreflightCheck(
        check=f"file.{rel_path}",
        status="fail",
        critical=True,
        message=f"Required file is missing: {rel_path}.",
        command="",
        value=relative_path(root, path),
    )


def collect_preflight_checks(
    root: Path,
    *,
    cohorts: list[str] | None,
    all_cohorts: bool,
    command_runner: CheckRunner | None = None,
) -> list[PreflightCheck]:
    runner = command_runner or _run_command
    checks: list[PreflightCheck] = []
    target_cohorts = _cohorts_to_check(all_cohorts=all_cohorts, cohorts=cohorts)
    required_paths, r_packages = _load_preflight_config(root)

    checks.append(check_python_runtime())
    r_check = check_rscript_availability(command_runner=runner)
    checks.append(r_check)

    if r_check.status == "pass":
        for package in r_packages:
            checks.append(check_r_package(package=package, command_runner=runner))
    else:
        for package in r_packages:
            checks.append(
                PreflightCheck(
                    check=f"rpackage.{package}",
                    status="skipped",
                    critical=False,
                    message="Skipped because Rscript is unavailable.",
                    command=f"Rscript -e if (!requireNamespace('{package}', quietly = TRUE)) {{ quit(status = 1) }}",
                )
            )

    seen_paths: set[str] = set()
    for rel_path in required_paths.get("__all__", []):
        if rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)
        checks.append(check_link_file(root, rel_path=rel_path))

    for cohort in target_cohorts:
        for rel_path in required_paths.get(cohort, []):
            if rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            checks.append(check_link_file(root, rel_path=rel_path))

    return checks


def build_preflight_summary(
    *,
    root: Path,
    checks: list[PreflightCheck],
    strict: bool,
) -> dict[str, Any]:
    critical_failures = [check.check for check in checks if check.critical and check.status != "pass"]
    check_rows = []
    for check in checks:
        row = check.as_row()
        row["command"] = _sanitize_text_root_paths(root, row.get("command", ""))
        row["value"] = _sanitize_text_root_paths(root, row.get("value", ""))
        check_rows.append(row)
    return {
        "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "project_root": relative_path(root, root),
        "strict": strict,
        "overall": "pass" if not critical_failures else "fail",
        "critical_check_count": sum(1 for check in checks if check.critical),
        "critical_failure_count": len(critical_failures),
        "critical_failures": critical_failures,
        "checks": check_rows,
    }


def write_preflight_outputs(
    root: Path,
    checks: list[PreflightCheck],
    *,
    strict: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    outputs_dir = _resolve_outputs_dir(root)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = build_preflight_summary(root=root, checks=checks, strict=strict)
    status_path = tables_dir / "preflight_status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["check", "status", "critical", "message", "command", "value"],
        )
        writer.writeheader()
        for row in summary["checks"]:
            writer.writerow(row)

    summary_path = tables_dir / "preflight_summary.json"
    dump_json(summary_path, summary)
    return status_path, summary_path, summary


def run_preflight(
    root: Path,
    *,
    cohorts: list[str] | None,
    all_cohorts: bool,
    strict: bool,
    command_runner: CheckRunner | None = None,
) -> tuple[int, Path, Path, dict[str, Any]]:
    runner = command_runner or _run_command
    resolved_root = root.resolve()
    checks = collect_preflight_checks(
        resolved_root,
        cohorts=cohorts,
        all_cohorts=all_cohorts,
        command_runner=runner,
    )
    status_path, summary_path, summary = write_preflight_outputs(
        resolved_root,
        checks,
        strict=strict,
    )
    exit_code = 1 if strict and summary["critical_failures"] else 0
    return exit_code, status_path, summary_path, summary
