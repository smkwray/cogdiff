#!/usr/bin/env python3
# Machine-specific pipeline runner variant. See 13_run_pipeline.py for the canonical version.
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root

COHORTS = ["nlsy79", "nlsy97", "cnlsy"]
STAGE_MIN = 0
STAGE_MAX = 15
PRE_FLIGHT_SCRIPT = "16_preflight_dependencies.py"

STAGE_SCRIPTS: dict[int, str] = {
    0: "00_download_raw.py",
    1: "01_unpack_and_index.py",
    2: "02_build_variable_map.py",
    3: "03_extract_cohort_panel.py",
    4: "04_export_links.py",
    5: "05_construct_samples.py",
    6: "06_age_residualize.py",
    7: "07_fit_sem_models.py",
    8: "08_invariance_and_partial.py",
    9: "09_results_and_figures.py",
    10: "10_cnlsy_development.py",
    11: "11_robustness_suite.py",
    12: "12_missingness_diagnostics.py",
    13: "13_run_pipeline.py",
    14: "14_reproducibility_report.py",
    15: "15_specification_curve_summary.py",
}


class CommandPlan:
    def __init__(self, stage: int, script: str, cohort_scope: str, argv: list[str]):
        self.stage = stage
        self.script = script
        self.cohort_scope = cohort_scope
        self.argv = argv


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return COHORTS
    return args.cohort


def _selected_stages(args: argparse.Namespace) -> list[int]:
    if args.stage:
        stages = sorted(set(args.stage))
    else:
        if args.from_stage > args.to_stage:
            raise ValueError("--from-stage must be <= --to-stage.")
        stages = list(range(args.from_stage, args.to_stage + 1))
    invalid = [s for s in stages if s < STAGE_MIN or s > STAGE_MAX]
    if invalid:
        raise ValueError(f"Invalid stage(s): {invalid}. Valid range is {STAGE_MIN}-{STAGE_MAX}.")
    return stages


def _cohort_flags(cohorts: list[str]) -> list[str]:
    flags: list[str] = []
    for cohort in cohorts:
        flags.extend(["--cohort", cohort])
    return flags


def _parse_stage_timeout_overrides(raw_overrides: list[str]) -> dict[int, float]:
    timeouts: dict[int, float] = {}
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError(
                f"Invalid --stage-timeout-override '{raw}'. "
                "Expected STAGE=SECONDS format (for example, 3=1200)."
            )
        stage_text, timeout_text = raw.split("=", 1)
        try:
            stage = int(stage_text.strip())
            timeout_seconds = float(timeout_text.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid --stage-timeout-override '{raw}'. Stage and seconds must be numeric."
            ) from exc
        if stage < STAGE_MIN or stage > STAGE_MAX:
            raise ValueError(f"Invalid stage in --stage-timeout-override '{raw}'.")
        if timeout_seconds <= 0:
            raise ValueError(f"Timeout in --stage-timeout-override '{raw}' must be positive.")
        timeouts[stage] = timeout_seconds
    return timeouts


def _build_stage_commands(
    *,
    stage: int,
    repo_root: Path,
    target_root: Path,
    cohorts: list[str],
    args: argparse.Namespace,
) -> list[CommandPlan]:
    script_name = STAGE_SCRIPTS[stage]
    script_path = repo_root / "scripts" / script_name
    py = sys.executable

    def base_cmd() -> list[str]:
        return [py, str(script_path)]

    plans: list[CommandPlan] = []

    if stage == 0:
        cmd = base_cmd() + _cohort_flags(cohorts)
        if args.force:
            cmd.append("--force")
        plans.append(CommandPlan(stage, script_name, ",".join(cohorts), cmd))
        return plans

    if stage == 1:
        cmd = base_cmd() + _cohort_flags(cohorts)
        if args.force:
            cmd.append("--force")
        if args.to_parquet:
            cmd.append("--to-parquet")
        plans.append(CommandPlan(stage, script_name, ",".join(cohorts), cmd))
        return plans

    if stage == 2:
        for cohort in cohorts:
            cmd = base_cmd() + ["--cohort", cohort]
            plans.append(CommandPlan(stage, script_name, cohort, cmd))
        return plans

    if stage == 10:
        if "cnlsy" not in cohorts:
            return []
        cmd = base_cmd() + ["--project-root", str(target_root)]
        if args.cnlsy_score_col:
            cmd.extend(["--score-col", args.cnlsy_score_col])
        plans.append(CommandPlan(stage, script_name, "cnlsy", cmd))
        return plans

    if stage == 13:
        # Stage 13 is this orchestrator script and is reserved/no-op in stage ranges.
        return []

    if stage == 11:
        cmd = base_cmd() + ["--project-root", str(target_root)] + _cohort_flags(cohorts)
        if args.robustness_reruns:
            cmd.append("--rerun-robustness")
        if args.robustness_sampling:
            cmd.append("--rerun-sampling-variants")
        if args.robustness_model_form:
            cmd.append("--rerun-model-forms")
        if args.robustness_age_adjustment:
            cmd.append("--rerun-age-adjustment-variants")
        if args.robustness_inference:
            cmd.append("--rerun-inference-variants")
        if args.robustness_weights:
            cmd.append("--rerun-weight-variants")
        plans.append(CommandPlan(stage, script_name, ",".join(cohorts), cmd))
        return plans

    if stage == 14:
        cmd = base_cmd() + ["--project-root", str(target_root)]
        plans.append(CommandPlan(stage, script_name, ",".join(cohorts), cmd))
        return plans

    cmd = base_cmd() + ["--project-root", str(target_root)] + _cohort_flags(cohorts)
    if stage == 3 and args.force:
        cmd.append("--force")
    if stage == 6 and args.standardize_output:
        cmd.append("--standardize-output")
    if stage == 7:
        if args.skip_r:
            cmd.append("--skip-r")
        if args.sem_dry_run:
            cmd.append("--dry-run")
        if args.enforce_warning_policy:
            cmd.append("--enforce-warning-policy")
        if args.warning_policy_threshold:
            cmd.extend(["--warning-policy-threshold", args.warning_policy_threshold])
    plans.append(CommandPlan(stage, script_name, ",".join(cohorts), cmd))
    return plans


def _build_post_stage_15_commands(
    *,
    repo_root: Path,
    target_root: Path,
    args: argparse.Namespace,
) -> list[CommandPlan]:
    plans: list[CommandPlan] = []

    if args.post_claim_verdicts:
        script_name = "29_build_claim_verdicts.py"
        cmd = [sys.executable, str(repo_root / "scripts" / script_name), "--project-root", str(target_root)]
        plans.append(CommandPlan(29, script_name, "post-stage15", cmd))

    if args.post_inference_ci_check:
        script_name = "30_check_inference_ci_coherence.py"
        cmd = [sys.executable, str(repo_root / "scripts" / script_name), "--project-root", str(target_root)]
        plans.append(CommandPlan(30, script_name, "post-stage15", cmd))

    if args.post_report_sections:
        script_name = "31_export_report_sections.py"
        cmd = [sys.executable, str(repo_root / "scripts" / script_name), "--project-root", str(target_root)]
        plans.append(CommandPlan(31, script_name, "post-stage15", cmd))

    return plans


def _build_preflight_command(
    *,
    repo_root: Path,
    target_root: Path,
    cohorts: list[str],
    args: argparse.Namespace,
) -> CommandPlan:
    script_path = repo_root / "scripts" / PRE_FLIGHT_SCRIPT
    cmd = [sys.executable, str(script_path), "--project-root", str(target_root)] + _cohort_flags(
        cohorts
    )
    if args.preflight_strict:
        cmd.append("--strict")
    return CommandPlan(-1, PRE_FLIGHT_SCRIPT, ",".join(cohorts), cmd)


def _logs_dir_for_root(root: Path) -> Path:
    paths_file = root / "config" / "paths.yml"
    if paths_file.exists():
        cfg = load_yaml(paths_file)
        outputs_dir = _resolve_path(cfg.get("outputs_dir", "outputs"), root)
    else:
        outputs_dir = root / "outputs"
    log_dir = outputs_dir / "logs" / "pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _run_command(
    *,
    plan: CommandPlan,
    repo_root: Path,
    target_root: Path,
    log_file: Path,
    timeout_seconds: float | None,
) -> tuple[int | None, str, str]:
    env = os.environ.copy()
    env["NLS_PROJECT_ROOT"] = str(target_root)
    command_display = shlex.join(plan.argv)
    start = time.perf_counter()
    try:
        cp = subprocess.run(
            plan.argv,
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        timeout_note = f"timed out after {timeout_seconds} seconds"
        output = exc.stdout or ""
        if exc.stderr:
            output = output + ("\n" if output else "") + exc.stderr
        output = output + ("\n" if output else "") + timeout_note
        log_file.write_text(output, encoding="utf-8")
        print(f"[timeout] {command_display} after {timeout_note} (elapsed {round(elapsed, 2)}s)")
        return None, "timeout", timeout_note

    output = cp.stdout
    if cp.stderr:
        output = output + ("\n" if output else "") + cp.stderr
    log_file.write_text(output, encoding="utf-8")
    status = "ok" if cp.returncode == 0 else "failed"
    return cp.returncode, status, ""


def _latest_successful_commands(log_dir: Path) -> set[str]:
    summaries = sorted(log_dir.glob("*_pipeline_run_summary.csv"))
    if not summaries:
        return set()
    latest = summaries[-1]
    df = pd.read_csv(latest)
    if df.empty or "status" not in df.columns or "command" not in df.columns:
        return set()
    ok_rows = df[df["status"] == "ok"]
    return set(ok_rows["command"].dropna().astype(str).tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pipeline stages 00-15 with consistent orchestration and logging.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Target project root.")
    parser.add_argument("--cohort", action="append", choices=COHORTS, help="Cohort(s) to run.")
    parser.add_argument("--all", action="store_true", help="Run all cohorts.")
    parser.add_argument("--from-stage", type=int, default=0, help="Starting stage (inclusive).")
    parser.add_argument("--to-stage", type=int, default=15, help="Ending stage (inclusive).")
    parser.add_argument("--stage", type=int, action="append", help="Explicit stage(s) to run. Overrides from/to range.")
    parser.add_argument("--dry-run", action="store_true", help="Write run plan only; do not execute commands.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run preflight dependency checks before requested stages.",
    )
    parser.add_argument(
        "--preflight-strict",
        action="store_true",
        help="Pass --strict to preflight check and fail pipeline on preflight failure.",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Continue remaining commands after failures.")
    parser.add_argument("--force", action="store_true", help="Pass --force to compatible stages.")
    parser.add_argument("--to-parquet", action="store_true", help="Pass --to-parquet to stage 01.")
    parser.add_argument(
        "--standardize-output",
        action="store_true",
        help="Pass --standardize-output to stage 06.",
    )
    parser.add_argument("--skip-r", action="store_true", help="Pass --skip-r to stage 07.")
    parser.add_argument("--sem-dry-run", action="store_true", help="Pass --dry-run to stage 07.")
    parser.add_argument(
        "--enforce-warning-policy",
        action="store_true",
        help="Pass --enforce-warning-policy to stage 07.",
    )
    parser.add_argument(
        "--warning-policy-threshold",
        choices=["clean", "info", "caution", "fail"],
        help="Pass --warning-policy-threshold to stage 07.",
    )
    parser.add_argument("--cnlsy-score-col", default="g", help="Score column for stage 10.")
    parser.add_argument(
        "--robustness-reruns",
        action="store_true",
        help="Pass robustness rerun flag set to stage 11 (all families).",
    )
    parser.add_argument(
        "--robustness-sampling",
        action="store_true",
        help="Pass sampling-variant rerun flag to stage 11.",
    )
    parser.add_argument(
        "--robustness-model-form",
        action="store_true",
        help="Pass model-form variant rerun flag to stage 11.",
    )
    parser.add_argument(
        "--robustness-age-adjustment",
        action="store_true",
        help="Pass age-adjustment variant rerun flag to stage 11.",
    )
    parser.add_argument(
        "--robustness-inference",
        action="store_true",
        help="Pass inference variant rerun flag to stage 11.",
    )
    parser.add_argument(
        "--robustness-weights",
        action="store_true",
        help="Pass weight variant rerun flag to stage 11.",
    )
    parser.add_argument(
        "--post-claim-verdicts",
        action="store_true",
        help="Run claim-verdict script (29) after requested stage-15 completes.",
    )
    parser.add_argument(
        "--post-inference-ci-check",
        action="store_true",
        help="Run inference CI-coherence script (30) after requested stage-15 completes.",
    )
    parser.add_argument(
        "--post-report-sections",
        action="store_true",
        help="Run report sections export script (31) after requested stage-15 completes.",
    )
    parser.add_argument(
        "--skip-successful",
        action="store_true",
        help="Skip commands that were successful in the latest pipeline summary for this project root.",
    )
    parser.add_argument(
        "--stage-timeout-seconds",
        type=float,
        default=None,
        help="Default timeout in seconds for each stage command.",
    )
    parser.add_argument(
        "--stage-timeout-override",
        action="append",
        default=[],
        metavar="STAGE=SECONDS",
        help="Per-stage timeout override in STAGE=SECONDS format, e.g. --stage-timeout-override 3=1200.",
    )
    args = parser.parse_args()
    if args.stage_timeout_seconds is not None and args.stage_timeout_seconds <= 0:
        raise ValueError("--stage-timeout-seconds must be positive.")

    repo_root = PROJECT_ROOT.resolve()
    target_root = Path(args.project_root).resolve()
    cohorts = _cohorts_from_args(args)
    stages = _selected_stages(args)
    timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    stage_timeout_overrides = _parse_stage_timeout_overrides(args.stage_timeout_override)

    log_dir = _logs_dir_for_root(target_root)
    successful_commands = _latest_successful_commands(log_dir) if args.skip_successful else set()
    plan_rows: list[dict[str, Any]] = []
    failure_count = 0
    preflight_failed = False

    if args.preflight:
        preflight_plan = _build_preflight_command(
            repo_root=repo_root,
            target_root=target_root,
            cohorts=cohorts,
            args=args,
        )
        preflight_log = log_dir / f"{timestamp}_preflight_01.log"
        preflight_command_display = shlex.join(preflight_plan.argv)
        print(f"[preflight] {preflight_command_display}")

        if args.dry_run:
            plan_rows.append(
                {
                    "stage": preflight_plan.stage,
                    "script": preflight_plan.script,
                    "cohort_scope": preflight_plan.cohort_scope,
                    "status": "planned",
                    "returncode": 0,
                    "command": preflight_command_display,
                    "log_file": str(preflight_log),
                    "note": "dry-run",
                }
            )
        else:
            if args.skip_successful and preflight_command_display in successful_commands:
                print(f"[skip] previously successful command: {preflight_command_display}")
                plan_rows.append(
                    {
                        "stage": preflight_plan.stage,
                        "script": preflight_plan.script,
                        "cohort_scope": preflight_plan.cohort_scope,
                        "status": "skipped_successful",
                        "returncode": 0,
                        "command": preflight_command_display,
                        "log_file": "",
                        "note": "matched successful command in latest summary",
                    }
                )
            else:
                returncode, status, note = _run_command(
                    plan=preflight_plan,
                    repo_root=repo_root,
                    target_root=target_root,
                    log_file=preflight_log,
                    timeout_seconds=stage_timeout_overrides.get(
                        preflight_plan.stage, args.stage_timeout_seconds
                    ),
                )
                if status in {"failed", "timeout"}:
                    preflight_failed = True
                    if args.preflight_strict:
                        failure_count += 1
                plan_rows.append(
                    {
                        "stage": preflight_plan.stage,
                        "script": preflight_plan.script,
                        "cohort_scope": preflight_plan.cohort_scope,
                        "status": status,
                        "returncode": returncode,
                        "command": preflight_command_display,
                        "log_file": str(preflight_log),
                        "note": note,
                    }
                )

    if args.preflight_strict and preflight_failed:
        print("[error] preflight checks failed (strict mode enabled).")

    for stage in stages:
        if preflight_failed and args.preflight_strict:
            break
        plans = _build_stage_commands(
            stage=stage,
            repo_root=repo_root,
            target_root=target_root,
            cohorts=cohorts,
            args=args,
        )
        if not plans:
            plan_rows.append(
                {
                    "stage": stage,
                    "script": STAGE_SCRIPTS[stage],
                    "cohort_scope": "n/a",
                    "status": "skipped",
                    "returncode": 0,
                    "command": "",
                    "log_file": "",
                    "note": "No applicable command for selected cohorts.",
                }
            )
            continue

        for idx, plan in enumerate(plans, start=1):
            command_display = shlex.join(plan.argv)
            log_file = log_dir / f"{timestamp}_stage{stage:02d}_{idx:02d}.log"
            print(f"[stage {stage:02d}] {command_display}")
            stage_timeout_seconds = stage_timeout_overrides.get(stage, args.stage_timeout_seconds)

            if args.dry_run:
                plan_rows.append(
                    {
                        "stage": stage,
                        "script": plan.script,
                        "cohort_scope": plan.cohort_scope,
                        "status": "planned",
                        "returncode": 0,
                        "command": command_display,
                        "log_file": str(log_file),
                        "note": "dry-run",
                    }
                )
                continue

            if args.skip_successful and command_display in successful_commands:
                print(f"[skip] previously successful command: {command_display}")
                plan_rows.append(
                    {
                        "stage": stage,
                        "script": plan.script,
                        "cohort_scope": plan.cohort_scope,
                        "status": "skipped_successful",
                        "returncode": 0,
                        "command": command_display,
                        "log_file": "",
                        "note": "matched successful command in latest summary",
                    }
                )
                continue

            returncode, status, note = _run_command(
                plan=plan,
                repo_root=repo_root,
                target_root=target_root,
                log_file=log_file,
                timeout_seconds=stage_timeout_seconds,
            )
            if status in {"failed", "timeout"}:
                failure_count += 1

            plan_rows.append(
                {
                    "stage": stage,
                    "script": plan.script,
                    "cohort_scope": plan.cohort_scope,
                    "status": status,
                    "returncode": returncode,
                    "command": command_display,
                    "log_file": str(log_file),
                    "note": note,
                }
            )

            if status != "ok":
                print(f"[error] stage {stage:02d} failed (see {log_file})")
                if not args.continue_on_error:
                    break
                print("[warn] continuing due to --continue-on-error")

            if failure_count > 0 and not args.continue_on_error and not args.dry_run:
                break

    if 15 in stages and (
        args.post_claim_verdicts or args.post_inference_ci_check or args.post_report_sections
    ):
        post_stage_plans = _build_post_stage_15_commands(
            repo_root=repo_root,
            target_root=target_root,
            args=args,
        )
        for post_plan in post_stage_plans:
            command_display = shlex.join(post_plan.argv)
            log_file = log_dir / f"{timestamp}_stage{post_plan.stage:02d}_post.log"
            print(f"[stage {post_plan.stage:02d}] {command_display}")

            if failure_count > 0 and not args.continue_on_error:
                print(f"[skip] previous stage failure prevents post-stage hook execution.")
                plan_rows.append(
                    {
                        "stage": post_plan.stage,
                        "script": post_plan.script,
                        "cohort_scope": post_plan.cohort_scope,
                        "status": "skipped",
                        "returncode": 0,
                        "command": command_display,
                        "log_file": "",
                        "note": "skipped due earlier stage failure",
                    }
                )
                continue

            if args.dry_run:
                plan_rows.append(
                    {
                        "stage": post_plan.stage,
                        "script": post_plan.script,
                        "cohort_scope": post_plan.cohort_scope,
                        "status": "planned",
                        "returncode": 0,
                        "command": command_display,
                        "log_file": str(log_file),
                        "note": "dry-run",
                    }
                )
                continue

            if args.skip_successful and command_display in successful_commands:
                print(f"[skip] previously successful command: {command_display}")
                plan_rows.append(
                    {
                        "stage": post_plan.stage,
                        "script": post_plan.script,
                        "cohort_scope": post_plan.cohort_scope,
                        "status": "skipped_successful",
                        "returncode": 0,
                        "command": command_display,
                        "log_file": "",
                        "note": "matched successful command in latest summary",
                    }
                )
                continue

            returncode, status, note = _run_command(
                plan=post_plan,
                repo_root=repo_root,
                target_root=target_root,
                log_file=log_file,
                timeout_seconds=None,
            )
            if status in {"failed", "timeout"}:
                failure_count += 1

            plan_rows.append(
                {
                    "stage": post_plan.stage,
                    "script": post_plan.script,
                    "cohort_scope": post_plan.cohort_scope,
                    "status": status,
                    "returncode": returncode,
                    "command": command_display,
                    "log_file": str(log_file),
                    "note": note,
                }
            )

            if status != "ok":
                print(f"[error] post-stage script failed (see {log_file})")
                if not args.continue_on_error:
                    break

    summary_df = pd.DataFrame(plan_rows)
    summary_path = log_dir / f"{timestamp}_pipeline_run_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[ok] wrote pipeline summary: {summary_path}")

    if args.dry_run:
        return 0
    return 1 if failure_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
