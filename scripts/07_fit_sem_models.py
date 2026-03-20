#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root, relative_path
from nls_pipeline.sem import (
    cnlsy_model_syntax,
    hierarchical_model_syntax,
    hierarchical_subtests,
    rscript_path,
    run_python_sem_fallback,
    run_sem_r_script,
    write_sem_inputs,
)

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

WARNING_CLASS_CONFIG: dict[str, dict[str, Any]] = {
    "se_not_computed": {
        "severity": "fail",
        "patterns": [
            r"could not compute standard errors",
            r"information matrix could not be inverted",
        ],
    },
    "non_convergence": {
        "severity": "fail",
        "patterns": [
            r"non-?convergence",
            r"did not converge",
            r"failed to converge",
            r"failed convergence",
        ],
    },
    "robust_fit_failed": {
        "severity": "caution",
        "patterns": [
            r"could not invert information .*robust test statistic",
            r"computation of robust cfi failed",
            r"computation of robust rmsea failed",
        ],
    },
    "vcov_not_posdef_or_identification": {
        "severity": "fail",
        "patterns": [
            r"latent (?:covariance|variance) matrix .*does not .*positive definite",
            r"latent (?:covariance|variance) matrix .*not positive definite",
            r"covariance matrix of latent variables .*not positive definite",
            r"underidentified",
            r"under-identified",
            r"model .*underidentified",
        ],
    },
    "negative_variance_heywood": {
        "severity": "fail",
        "patterns": [
            r"negative variance",
            r"negative (?:error|residual) variance",
            r"negative residual variance",
            r"variances are negative",
            r"estimated .* variances are negative",
            r"heywood",
        ],
    },
    "gradient_warning": {
        "severity": "caution",
        "patterns": [
            r"gradient (?:is|was) not",
            r"gradient.*warning",
            r"gradient.*problem",
            r"check\.gradient",
        ],
    },
    "modindices_unavailable": {
        "severity": "caution",
        "patterns": [
            r"modification indices unavailable",
            r"could not compute modification indices",
        ],
    },
    "vcov_not_posdef": {
        "severity": "caution",
        "patterns": [
            r"variance-covariance matrix .*does not .*positive definite",
        ],
    },
    "modindices_constraints_notice": {
        "severity": "info",
        "patterns": [
            r"modindices\(\) function ignores equality constraints",
        ],
    },
    "runtime_sem_failure": {
        "severity": "fail",
        "patterns": [],
    },
    "unclassified_warning": {
        "severity": "caution",
        "patterns": [],
    },
}

WARNING_SEVERITY_ORDER = {"fail": 3, "caution": 2, "info": 1, "clean": 0}
RUNTIME_FAIL_STATUSES = {"r-failed", "r-script-missing"}


@dataclass(frozen=True)
class WarningPolicyConfig:
    enforced: bool
    threshold: str


def _parse_warning_policy(args: argparse.Namespace, models_cfg: dict[str, Any]) -> WarningPolicyConfig:
    reporting_cfg = models_cfg.get("reporting", {})
    if not isinstance(reporting_cfg, dict):
        reporting_cfg = {}
    policy_cfg = reporting_cfg.get("warning_policy", {})
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}

    policy_enabled = bool(policy_cfg.get("enabled", False))
    policy_threshold = str(policy_cfg.get("threshold", "fail")).strip().lower()

    if args.enforce_warning_policy:
        policy_enabled = True
    if args.warning_policy_threshold is not None:
        policy_threshold = args.warning_policy_threshold.strip().lower()

    if policy_threshold not in WARNING_SEVERITY_ORDER:
        raise ValueError(f"Invalid warning policy threshold: {policy_threshold}")
    return WarningPolicyConfig(enforced=policy_enabled, threshold=policy_threshold)


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _input_data_path(root: Path, paths_cfg: dict[str, Any], cohort: str, source_override: Path | None) -> Path:
    if source_override is not None:
        return source_override
    processed_dir = _resolve_path(paths_cfg["processed_dir"], root)
    preferred = processed_dir / f"{cohort}_cfa_resid.csv"
    fallback = processed_dir / f"{cohort}_cfa.csv"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing cohort input for SEM: expected {preferred} or {fallback}")


def _request_payload(cohort: str, data_csv: Path, models_cfg: dict[str, Any], cohort_cfg: dict[str, Any]) -> dict[str, Any]:
    sample_cfg = cohort_cfg.get("sample_construct", {}) if isinstance(cohort_cfg.get("sample_construct", {}), dict) else {}
    sem_fit_cfg = cohort_cfg.get("sem_fit", {}) if isinstance(cohort_cfg.get("sem_fit", {}), dict) else {}
    group_col = str(sample_cfg.get("sex_col", "sex"))
    std_lv = bool(sem_fit_cfg.get("std_lv", True))
    invariance_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    steps = [str(x) for x in invariance_cfg.get("steps", ["configural", "metric", "scalar", "strict"])]
    partial_cfg = invariance_cfg.get("partial_intercepts", {})
    if not isinstance(partial_cfg, dict):
        partial_cfg = {}
    partial_intercepts = [str(x) for x in partial_cfg.get(cohort, [])]

    if cohort == "cnlsy":
        observed = [str(x) for x in models_cfg.get("cnlsy_single_factor", [])]
    else:
        observed = hierarchical_subtests(models_cfg)

    return {
        "cohort": cohort,
        "data_csv": str(data_csv),
        "group_col": group_col,
        "reference_group": str(models_cfg.get("reference_group", "female")),
        "estimator": "MLR",
        "missing": "fiml",
        "std_lv": std_lv,
        "invariance_steps": steps,
        "partial_intercepts": partial_intercepts,
        "observed_tests": observed,
    }


def _model_syntax_for_cohort(cohort: str, models_cfg: dict[str, Any]) -> str:
    return cnlsy_model_syntax(models_cfg) if cohort == "cnlsy" else hierarchical_model_syntax(models_cfg)


def _warning_count(stderr_text: str) -> int:
    if not stderr_text:
        return 0
    explicit = re.search(r"There were\s+(\d+)\s+warnings", stderr_text, flags=re.IGNORECASE)
    if explicit:
        return int(explicit.group(1))
    enumerated = re.findall(r"^\s*\d+:\s", stderr_text, flags=re.MULTILINE)
    if enumerated:
        return len(enumerated)
    prefixed = re.findall(r"^\s*Warning(?: in)?\b", stderr_text, flags=re.MULTILINE | re.IGNORECASE)
    if prefixed:
        return len(prefixed)
    if "warning" in stderr_text.lower():
        return 1
    return 0


def _classify_sem_warnings(stderr_text: str, status: str | None = None) -> dict[str, Any]:
    status_key = str(status or "").strip().lower()
    class_counts = {name: 0 for name in WARNING_CLASS_CONFIG}
    if stderr_text:
        for name, cfg in WARNING_CLASS_CONFIG.items():
            if name in {"runtime_sem_failure", "unclassified_warning"}:
                continue
            count = 0
            for pattern in cfg["patterns"]:
                count += len(re.findall(pattern, stderr_text, flags=re.IGNORECASE | re.DOTALL))
            class_counts[name] = int(count)
    if status_key in RUNTIME_FAIL_STATUSES:
        class_counts["runtime_sem_failure"] = max(class_counts.get("runtime_sem_failure", 0), 1)

    warning_count = _warning_count(stderr_text)
    classified_count = sum(
        count for name, count in class_counts.items() if name not in {"unclassified_warning", "runtime_sem_failure"}
    )
    if status_key in RUNTIME_FAIL_STATUSES:
        classified_count += class_counts.get("runtime_sem_failure", 0)
    if warning_count > 0 and classified_count == 0:
        class_counts["unclassified_warning"] = warning_count

    present_classes = [name for name, count in class_counts.items() if count > 0]
    highest = "clean"
    for name in present_classes:
        severity = str(WARNING_CLASS_CONFIG[name]["severity"])
        if WARNING_SEVERITY_ORDER[severity] > WARNING_SEVERITY_ORDER[highest]:
            highest = severity
    if warning_count > 0 and highest == "clean":
        highest = "info"

    return {
        "warning_count": warning_count,
        "warning_policy_status": highest,
        "warning_classes": ";".join(sorted(present_classes)),
        "warning_class_counts": class_counts,
    }


def _apply_warning_policy(classification: dict[str, Any], policy: WarningPolicyConfig) -> dict[str, Any]:
    status = str(classification.get("warning_policy_status", "clean"))
    warning_count = int(classification.get("warning_count", 0))
    warning_classes = str(classification.get("warning_classes", ""))

    violated = False
    reason = ""
    if policy.enforced and WARNING_SEVERITY_ORDER[status] >= WARNING_SEVERITY_ORDER[policy.threshold]:
        violated = True
        reason = (
            f"severity={status};count={warning_count};"
            f"classes={warning_classes};threshold={policy.threshold}"
        )

    return {
        **classification,
        "warning_policy_enforced": policy.enforced,
        "warning_policy_violated": violated,
        "warning_policy_violation_reason": reason,
    }


def _warning_text_for_classification(status_payload: dict[str, Any]) -> str:
    parts = [
        str(status_payload.get("r_stderr", "") or "").strip(),
        str(status_payload.get("r_stdout", "") or "").strip(),
        str(status_payload.get("error", "") or "").strip(),
    ]
    return "\n".join([p for p in parts if p])


def _sanitize_text_root_paths(root: Path, text: Any) -> str:
    token = str(root.resolve())
    return str(text or "").replace(token, ".")


def _write_run_status(outdir: Path, payload: dict[str, Any]) -> None:
    status_path = outdir / "run_status.json"
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_for_cohort(
    root: Path,
    cohort: str,
    paths_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    source_override: Path | None,
    run_r: bool,
    dry_run: bool,
    se_mode: str = "standard",
    cluster_col: str | None = None,
    weight_col: str | None = None,
    python_fallback: bool = False,
    warning_policy: WarningPolicyConfig = WarningPolicyConfig(enforced=False, threshold="fail"),
) -> dict[str, Any]:
    cohort_cfg = load_yaml(root / COHORT_CONFIGS[cohort])
    sem_interim_dir = _resolve_path(paths_cfg["sem_interim_dir"], root)
    model_fits_root = _resolve_path(paths_cfg["outputs_dir"], root) / "model_fits"
    model_fits_root.mkdir(parents=True, exist_ok=True)
    cohort_fit_dir = model_fits_root / cohort
    cohort_fit_dir.mkdir(parents=True, exist_ok=True)

    data_path = _input_data_path(root, paths_cfg, cohort, source_override)
    request = _request_payload(cohort, data_path, models_cfg=models_cfg, cohort_cfg=cohort_cfg)
    request.update(
        {
            "se_mode": str(se_mode).strip().lower() or "standard",
            "cluster_col": (str(cluster_col).strip() if cluster_col else None),
            "weight_col": (str(weight_col).strip() if weight_col else None),
        }
    )
    model_syntax = _model_syntax_for_cohort(cohort, models_cfg)
    sem_ctx = write_sem_inputs(
        cohort=cohort,
        df_path=data_path,
        sem_interim_dir=sem_interim_dir,
        model_syntax=model_syntax,
        request_payload=request,
    )

    status: dict[str, Any] = {
        "cohort": cohort,
        "input_data": relative_path(root, data_path),
        "request_file": relative_path(root, sem_ctx.request_file),
        "model_syntax_file": relative_path(root, sem_ctx.model_syntax_file),
        "r_executed": False,
        "r_success": False,
        "r_stdout": "",
        "r_stderr": "",
    }

    if dry_run:
        status["status"] = "dry-run"
        status.update(_apply_warning_policy(_classify_sem_warnings("", status="dry-run"), warning_policy))
        _write_run_status(cohort_fit_dir, status)
        return status

    if run_r:
        r_script = root / "scripts/sem_fit.R"
        if not r_script.exists():
            status["status"] = "r-script-missing"
            status.update(
                _apply_warning_policy(_classify_sem_warnings("", status="r-script-missing"), warning_policy)
            )
            _write_run_status(cohort_fit_dir, status)
            return status
        try:
            cp = run_sem_r_script(r_script=r_script, request_file=sem_ctx.request_file, outdir=cohort_fit_dir)
            status["r_executed"] = True
            status["r_success"] = True
            status["r_stdout"] = _sanitize_text_root_paths(root, cp.stdout)
            status["r_stderr"] = _sanitize_text_root_paths(root, cp.stderr)
            status["status"] = "ok"
            status.update(
                _apply_warning_policy(
                    _classify_sem_warnings(_warning_text_for_classification(status), status="ok"),
                    warning_policy,
                )
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on local R setup
            status["r_executed"] = True
            status["r_success"] = False
            status["status"] = "r-failed"
            status["r_stdout"] = _sanitize_text_root_paths(root, exc.stdout)
            status["r_stderr"] = _sanitize_text_root_paths(root, exc.stderr)
            status["error"] = _sanitize_text_root_paths(root, exc)
            status.update(
                _apply_warning_policy(
                    _classify_sem_warnings(
                        _warning_text_for_classification(status),
                        status="r-failed",
                    ),
                    warning_policy,
                )
            )
        except Exception as exc:  # pragma: no cover - depends on local R setup
            status["r_executed"] = True
            status["r_success"] = False
            status["status"] = "r-failed"
            status["error"] = _sanitize_text_root_paths(root, exc)
            status.update(
                _apply_warning_policy(
                    _classify_sem_warnings(_warning_text_for_classification(status), status="r-failed"),
                    warning_policy,
                )
            )
    else:
        if python_fallback:
            fallback = run_python_sem_fallback(
                cohort=cohort,
                data_csv=data_path,
                outdir=cohort_fit_dir,
                group_col=request["group_col"],
                models_cfg=models_cfg,
                invariance_steps=[str(x) for x in request.get("invariance_steps", [])],
                observed_tests=[str(x) for x in request.get("observed_tests", [])],
            )
            status["status"] = str(fallback.get("status", "python-fallback"))
            status["python_fallback"] = True
            status["python_fallback_groups"] = int(fallback.get("n_groups", 0))
            status["python_fallback_factors"] = int(fallback.get("n_factors", 0))
            status.update(_apply_warning_policy(_classify_sem_warnings("", status=status["status"]), warning_policy))
        else:
            status["status"] = "skipped-r"
            status.update(_apply_warning_policy(_classify_sem_warnings("", status="skipped-r"), warning_policy))

    _write_run_status(cohort_fit_dir, status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and run SEM model fits (Python->R orchestration).")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to run.")
    parser.add_argument("--all", action="store_true", help="Run all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--source-path", type=Path, help="Optional explicit source CSV (single-cohort mode).")
    parser.add_argument("--skip-r", action="store_true", help="Only prepare SEM inputs; do not call R.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare inputs and status files without execution.")
    parser.add_argument("--se-mode", default="standard", help="SEM SE mode passed to R (standard/robust/robust.cluster/weighted).")
    parser.add_argument("--cluster-col", help="Cluster column for cluster-robust SEM.")
    parser.add_argument("--weight-col", help="Weight column for weighted SEM.")
    parser.add_argument(
        "--python-fallback",
        action="store_true",
        help="When R is not run, write approximate SEM outputs from Python composites for downstream stages.",
    )
    parser.add_argument(
        "--enforce-warning-policy",
        action="store_true",
        help="Exit non-zero when warning-policy threshold is violated.",
    )
    parser.add_argument(
        "--warning-policy-threshold",
        choices=["clean", "info", "caution", "fail"],
        help="Severity threshold for warning-policy violations when enforced.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    warning_policy = _parse_warning_policy(args, models_cfg)
    cohorts = _cohorts_from_args(args)

    if args.source_path is not None and len(cohorts) != 1:
        raise ValueError("--source-path is only supported when exactly one --cohort is provided.")
    source_override = args.source_path.resolve() if args.source_path else None

    r_available = rscript_path() is not None
    run_r = (not args.skip_r) and r_available
    use_python_fallback = args.python_fallback
    if (not args.skip_r) and (not r_available) and (not args.python_fallback):
        print(
            "[error] R (Rscript) is not available and --python-fallback was not passed. "
            "Install R with lavaan, pass --skip-r to skip SEM fitting, or pass "
            "--python-fallback to use approximate Python composites.",
            file=sys.stderr,
        )
        sys.exit(1)
    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        result = _run_for_cohort(
            root=root,
            cohort=cohort,
            paths_cfg=paths_cfg,
            models_cfg=models_cfg,
            source_override=source_override,
            run_r=run_r,
            dry_run=args.dry_run,
            se_mode=args.se_mode,
            cluster_col=args.cluster_col,
            weight_col=args.weight_col,
            python_fallback=use_python_fallback,
            warning_policy=warning_policy,
        )
        rows.append(result)
        print(f"[ok] {cohort}: {result['status']}")

    summary_path = _resolve_path(paths_cfg["outputs_dir"], root) / "tables" / "sem_run_status.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_path, index=False)
    triage_rows: list[dict[str, Any]] = []
    for row in rows:
        class_counts = row.get("warning_class_counts", {})
        if not isinstance(class_counts, dict):
            class_counts = {}
        cohort = str(row.get("cohort", ""))
        for class_name, cfg in WARNING_CLASS_CONFIG.items():
            triage_rows.append(
                {
                    "cohort": cohort,
                    "warning_class": class_name,
                    "severity": cfg["severity"],
                    "count": int(class_counts.get(class_name, 0)),
                    "warning_policy_status": row.get("warning_policy_status", "clean"),
                }
            )
    triage_path = summary_path.parent / "sem_warning_triage.csv"
    pd.DataFrame(triage_rows).to_csv(triage_path, index=False)
    print(f"[ok] wrote status summary: {summary_path}")
    print(f"[ok] wrote warning triage: {triage_path}")
    if any(str(row.get("status", "")).strip().lower() in RUNTIME_FAIL_STATUSES for row in rows):
        return 1
    if any(bool(row.get("warning_policy_violated", False)) for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
