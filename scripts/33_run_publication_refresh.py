#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LOCAL_SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root

DEFAULT_ENGINE = "sem_refit"
DEFAULT_N_BOOTSTRAP = 499
DEFAULT_MIN_SUCCESS_SHARE = 0.90
DEFAULT_SEM_TIMEOUT_SECONDS = 60.0
LOG_DIR_NAME = "publication_refresh"


@dataclass(frozen=True)
class CommandPlan:
    step: int
    script: str
    argv: list[str]


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _logs_dir_for_root(root: Path) -> Path:
    paths_file = root / "config" / "paths.yml"
    if paths_file.exists():
        cfg = load_yaml(paths_file)
        outputs_dir = _resolve_path(cfg.get("outputs_dir", "outputs"), root)
    else:
        outputs_dir = root / "outputs"
    log_dir = outputs_dir / "logs" / LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _latest_successful_commands(log_dir: Path) -> set[str]:
    summaries = sorted(log_dir.glob("*_publication_refresh_summary.csv"))
    if not summaries:
        return set()
    latest = summaries[-1]
    try:
        with latest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            commands: set[str] = set()
            for row in reader:
                status = str(row.get("status", "")).strip().lower()
                if status in {"ok", "skipped_successful"}:
                    command = str(row.get("command", "")).strip()
                    if command:
                        commands.add(command)
            return commands
    except OSError:
        return set()


def _build_plan(
    *,
    repo_root: Path,
    project_root_path: Path,
    python_executable: str,
    n_bootstrap: int,
    engine: str,
    sem_timeout_seconds: float,
    min_success_share: float,
) -> list[CommandPlan]:
    scripts_dir = repo_root / "scripts"
    return [
        CommandPlan(
            step=1,
            script="20_run_inference_bootstrap.py",
            argv=[
                python_executable,
                str(scripts_dir / "20_run_inference_bootstrap.py"),
                "--project-root",
                str(project_root_path),
                "--variant-token",
                "family_bootstrap",
                "--engine",
                str(engine),
                "--n-bootstrap",
                str(n_bootstrap),
                "--min-success-share",
                str(min_success_share),
                "--sem-timeout-seconds",
                str(sem_timeout_seconds),
            ],
        ),
        CommandPlan(
            step=2,
            script="11_robustness_suite.py",
            argv=[python_executable, str(scripts_dir / "11_robustness_suite.py"), "--project-root", str(project_root_path)],
        ),
        CommandPlan(
            step=3,
            script="15_specification_curve_summary.py",
            argv=[
                python_executable,
                str(scripts_dir / "15_specification_curve_summary.py"),
                "--project-root",
                str(project_root_path),
                "--all",
            ],
        ),
        CommandPlan(
            step=4,
            script="29_build_claim_verdicts.py",
            argv=[python_executable, str(scripts_dir / "29_build_claim_verdicts.py"), "--project-root", str(project_root_path)],
        ),
        CommandPlan(
            step=5,
            script="30_check_inference_ci_coherence.py",
            argv=[
                python_executable,
                str(scripts_dir / "30_check_inference_ci_coherence.py"),
                "--project-root",
                str(project_root_path),
            ],
        ),
        CommandPlan(
            step=6,
            script="24_build_publication_results_lock.py",
            argv=[
                python_executable,
                str(scripts_dir / "24_build_publication_results_lock.py"),
                "--project-root",
                str(project_root_path),
            ],
        ),
        CommandPlan(
            step=7,
            script="31_export_report_sections.py",
            argv=[python_executable, str(scripts_dir / "31_export_report_sections.py"), "--project-root", str(project_root_path)],
        ),
    ]


def _run_command(
    *,
    plan: CommandPlan,
    target_root: Path,
    log_file: Path,
) -> tuple[int | None, str, str]:
    env = os.environ.copy()
    env["NLS_PROJECT_ROOT"] = str(target_root)
    try:
        cp = subprocess.run(
            plan.argv,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        note = f"Executable not found: {exc}"
        log_file.write_text(note, encoding="utf-8")
        return 1, "failed", note
    except OSError as exc:
        note = f"Execution failed: {exc}"
        log_file.write_text(note, encoding="utf-8")
        return 1, "failed", note

    output = cp.stdout
    if cp.stderr:
        output = output + ("\n" if output else "") + cp.stderr
    log_file.write_text(output, encoding="utf-8")
    status = "ok" if cp.returncode == 0 else "failed"
    return cp.returncode, status, ""


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["step", "script", "status", "returncode", "command", "log_file", "note"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sanitized = {k: "" if v is None else str(v) for k, v in row.items()}
            writer.writerow({field: sanitized.get(field, "") for field in fieldnames})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the publication refresh chain in deterministic order.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to run stage scripts.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help="Bootstrap repetitions for stage 20.",
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        help="Inference bootstrap engine for stage 20.",
    )
    parser.add_argument(
        "--skip-successful",
        action="store_true",
        help="Reuse commands already successful in latest publication refresh summary.",
    )
    parser.add_argument(
        "--sem-timeout-seconds",
        type=float,
        default=DEFAULT_SEM_TIMEOUT_SECONDS,
        help="Per-refit timeout in seconds for sem_refit stage.",
    )
    parser.add_argument(
        "--min-success-share",
        type=float,
        default=DEFAULT_MIN_SUCCESS_SHARE,
        help="Minimum required converged bootstrap share.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command plan only; do not execute.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_bootstrap < 1:
        print("[error] --n-bootstrap must be positive.", file=sys.stderr)
        return 1
    if args.sem_timeout_seconds <= 0:
        print("[error] --sem-timeout-seconds must be positive.", file=sys.stderr)
        return 1
    if not 0 < args.min_success_share <= 1:
        print("[error] --min-success-share must be in (0,1].", file=sys.stderr)
        return 1

    repo_root = PROJECT_ROOT.resolve()
    target_root = Path(args.project_root).resolve()
    plans = _build_plan(
        repo_root=repo_root,
        project_root_path=target_root,
        python_executable=str(args.python_executable),
        n_bootstrap=int(args.n_bootstrap),
        engine=str(args.engine),
        sem_timeout_seconds=float(args.sem_timeout_seconds),
        min_success_share=float(args.min_success_share),
    )

    log_dir = _logs_dir_for_root(target_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summary_path = log_dir / f"{timestamp}_publication_refresh_summary.csv"
    successful_commands = _latest_successful_commands(log_dir) if args.skip_successful else set()

    summary_rows: list[dict[str, object]] = []
    encountered_failure = False

    for index, plan in enumerate(plans, start=1):
        command = shlex.join(plan.argv)
        log_file = log_dir / f"{timestamp}_step{index:02d}_{plan.script}.log"
        if args.dry_run:
            status = "planned"
            returncode = 0
            note = "dry-run"
            print(f"[dry-run] {index:02d}) {command}")
        elif args.skip_successful and command in successful_commands:
            status = "skipped_successful"
            returncode = 0
            note = "matched successful command in latest publication refresh summary"
            print(f"[skip-successful] {plan.script}")
        else:
            print(f"[run] {index:02d}) {command}")
            returncode, status, note = _run_command(
                plan=plan,
                target_root=target_root,
                log_file=log_file,
            )
            if status != "ok":
                encountered_failure = True

        summary_rows.append(
            {
                "step": index,
                "script": plan.script,
                "status": status,
                "returncode": returncode,
                "command": command,
                "log_file": "" if args.dry_run or (status == "skipped_successful") else str(log_file),
                "note": note,
            }
        )

        if encountered_failure:
            break

    if encountered_failure:
        for plan in plans[index:]:
            command = shlex.join(plan.argv)
            summary_rows.append(
                {
                    "step": index + 1,
                    "script": plan.script,
                    "status": "not_run",
                    "returncode": "",
                    "command": command,
                    "log_file": "",
                    "note": "aborted due earlier step failure",
                }
            )
            index += 1

    _write_summary(summary_path, summary_rows)

    if encountered_failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
