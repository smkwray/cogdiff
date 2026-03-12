#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
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
    run_python_sem_fallback,
    run_sem_r_script,
    rscript_path,
    write_sem_inputs,
)

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "metric": {"delta_cfi_min": -0.01, "delta_rmsea_max": 0.015, "delta_srmr_max": 0.03},
    "scalar": {"delta_cfi_min": -0.01, "delta_rmsea_max": 0.015, "delta_srmr_max": 0.015},
}
DEFAULT_TOLERANCE: dict[str, dict[str, float]] = {
    "metric": {"delta_cfi_excess_max": 0.002, "delta_rmsea_excess_max": 0.005, "delta_srmr_excess_max": 0.005},
    "scalar": {"delta_cfi_excess_max": 0.002, "delta_rmsea_excess_max": 0.005, "delta_srmr_excess_max": 0.005},
}

ELIGIBILITY_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "group_col",
    "reference_group",
    "n_total",
    "n_nonmissing_group",
    "n_groups",
    "group_levels",
    "group_counts",
    "smallest_group_n",
    "min_group_n",
    "metric_pass",
    "scalar_pass",
    "race_invariance_ok_for_vr",
    "race_invariance_ok_for_d",
    "reason_vr_g",
    "reason_d_g",
    "source_data",
    "fit_dir",
    "request_file",
    "model_syntax_file",
]

TRANSITION_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "transition",
    "prev_step",
    "next_step",
    "delta_cfi",
    "delta_rmsea",
    "delta_srmr",
    "delta_cfi_min",
    "delta_rmsea_max",
    "delta_srmr_max",
    "evaluated_criteria",
    "passed",
    "check_source",
    "fit_dir",
]


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _to_float(raw: Any, default: float | None = None) -> float | None:
    parsed = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return float(parsed)


def _as_float(raw: Any, default: float) -> float:
    value = _to_float(raw, default)
    assert value is not None
    return float(value)


def _empty_eligibility(
    cohort: str,
    reason: str,
    source_data: str,
    *,
    group_col: str = "race_ethnicity_3cat",
    reference_group: str = "NON-BLACK, NON-HISPANIC",
    n_total: int = 0,
    n_nonmissing_group: int = 0,
    n_groups: int = 0,
    group_levels: str = "",
    group_counts: str = "",
    smallest_group_n: int = 0,
    min_group_n: int = 0,
    fit_dir: str = "",
    request_file: str = "",
    model_syntax_file: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "status": "not_feasible",
        "reason": reason,
        "group_col": group_col,
        "reference_group": reference_group,
        "n_total": int(n_total),
        "n_nonmissing_group": int(n_nonmissing_group),
        "n_groups": int(n_groups),
        "group_levels": group_levels or pd.NA,
        "group_counts": group_counts or pd.NA,
        "smallest_group_n": int(smallest_group_n),
        "min_group_n": int(min_group_n),
        "source_data": source_data,
        "fit_dir": fit_dir or pd.NA,
        "request_file": request_file or pd.NA,
        "model_syntax_file": model_syntax_file or pd.NA,
    }
    for col in ELIGIBILITY_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _empty_transition(cohort: str, reason: str, fit_dir: str, transition: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "status": "not_feasible",
        "reason": reason,
        "transition": transition or pd.NA,
        "fit_dir": fit_dir or pd.NA,
    }
    for col in TRANSITION_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _expected_steps(models_cfg: dict[str, Any]) -> list[str]:
    invariance_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    return [str(x) for x in invariance_cfg.get("steps", ["configural", "metric", "scalar"])]


def _thresholds(models_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    invariance_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    gate_cfg = invariance_cfg.get("gatekeeping", {}) if isinstance(invariance_cfg.get("gatekeeping", {}), dict) else {}
    thresholds_cfg = gate_cfg.get("thresholds", {}) if isinstance(gate_cfg.get("thresholds", {}), dict) else {}
    out: dict[str, dict[str, float]] = {}
    for step, defaults in DEFAULT_THRESHOLDS.items():
        raw = thresholds_cfg.get(step, {}) if isinstance(thresholds_cfg.get(step, {}), dict) else {}
        out[step] = {
            "delta_cfi_min": _as_float(raw.get("delta_cfi_min"), defaults["delta_cfi_min"]),
            "delta_rmsea_max": _as_float(raw.get("delta_rmsea_max"), defaults["delta_rmsea_max"]),
            "delta_srmr_max": _as_float(raw.get("delta_srmr_max"), defaults["delta_srmr_max"]),
        }
    return out


def _tolerance(models_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    invariance_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    gate_cfg = invariance_cfg.get("gatekeeping", {}) if isinstance(invariance_cfg.get("gatekeeping", {}), dict) else {}
    tol_cfg = gate_cfg.get("near_pass_tolerance", {}) if isinstance(gate_cfg.get("near_pass_tolerance", {}), dict) else {}
    out: dict[str, dict[str, float]] = {}
    for step, defaults in DEFAULT_TOLERANCE.items():
        raw = tol_cfg.get(step, {}) if isinstance(tol_cfg.get(step, {}), dict) else {}
        out[step] = {
            "delta_cfi_excess_max": max(0.0, _as_float(raw.get("delta_cfi_excess_max"), defaults["delta_cfi_excess_max"])),
            "delta_rmsea_excess_max": max(0.0, _as_float(raw.get("delta_rmsea_excess_max"), defaults["delta_rmsea_excess_max"])),
            "delta_srmr_excess_max": max(0.0, _as_float(raw.get("delta_srmr_excess_max"), defaults["delta_srmr_excess_max"])),
        }
    return out


def _ensure_step_rows(df: pd.DataFrame, steps: list[str], cohort: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"cohort": cohort, "model_step": steps})
    work = df.copy()
    if "model_step" not in work.columns:
        raise ValueError("fit_indices.csv missing required column 'model_step'")
    if "cohort" not in work.columns:
        work["cohort"] = cohort
    return pd.DataFrame({"cohort": cohort, "model_step": steps}).merge(work, on=["cohort", "model_step"], how="left")


def _add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in ("cfi", "rmsea", "srmr"):
        if metric in out.columns:
            out[f"delta_{metric}"] = pd.to_numeric(out[metric], errors="coerce").diff()
        else:
            out[f"delta_{metric}"] = pd.NA
    return out


def _lookup_step_row(summary: pd.DataFrame, step: str) -> pd.Series | None:
    rows = summary.loc[summary["model_step"].astype(str) == step]
    if rows.empty:
        return None
    return rows.iloc[0]


def _transition_check(
    *,
    summary: pd.DataFrame,
    cohort: str,
    prev_step: str,
    next_step: str,
    thresholds: dict[str, float],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    prev_row = _lookup_step_row(summary, prev_step)
    next_row = _lookup_step_row(summary, next_step)
    out = {
        "cohort": cohort,
        "status": "computed",
        "reason": pd.NA,
        "transition": f"{prev_step}->{next_step}",
        "prev_step": prev_step,
        "next_step": next_step,
        "delta_cfi": pd.NA,
        "delta_rmsea": pd.NA,
        "delta_srmr": pd.NA,
        "delta_cfi_min": thresholds["delta_cfi_min"],
        "delta_rmsea_max": thresholds["delta_rmsea_max"],
        "delta_srmr_max": thresholds["delta_srmr_max"],
        "evaluated_criteria": 0,
        "passed": False,
        "check_source": "baseline",
        "fit_dir": pd.NA,
    }
    if prev_row is None or next_row is None:
        out["status"] = "not_feasible"
        out["reason"] = "step_missing"
        return out

    def _delta(metric: str) -> float | None:
        delta_col = f"delta_{metric}"
        val = _to_float(next_row.get(delta_col))
        if val is not None:
            return val
        prev_val = _to_float(prev_row.get(metric))
        next_val = _to_float(next_row.get(metric))
        if prev_val is None or next_val is None:
            return None
        return next_val - prev_val

    d_cfi = _delta("cfi")
    d_rmsea = _delta("rmsea")
    d_srmr = _delta("srmr")
    out["delta_cfi"] = d_cfi if d_cfi is not None else pd.NA
    out["delta_rmsea"] = d_rmsea if d_rmsea is not None else pd.NA
    out["delta_srmr"] = d_srmr if d_srmr is not None else pd.NA

    checks: list[bool] = []
    excess: dict[str, float] = {}
    if d_cfi is not None:
        passed = d_cfi >= thresholds["delta_cfi_min"]
        checks.append(passed)
        if not passed:
            excess["delta_cfi"] = thresholds["delta_cfi_min"] - d_cfi
    if d_rmsea is not None:
        passed = d_rmsea <= thresholds["delta_rmsea_max"]
        checks.append(passed)
        if not passed:
            excess["delta_rmsea"] = d_rmsea - thresholds["delta_rmsea_max"]
    if d_srmr is not None:
        passed = d_srmr <= thresholds["delta_srmr_max"]
        checks.append(passed)
        if not passed:
            excess["delta_srmr"] = d_srmr - thresholds["delta_srmr_max"]

    out["evaluated_criteria"] = len(checks)
    if not checks:
        out["status"] = "not_feasible"
        out["reason"] = "criteria_not_evaluable"
        return out
    if all(checks):
        out["passed"] = True
        return out

    if len(excess) == 1 and len(checks) >= 2:
        if "delta_cfi" in excess and excess["delta_cfi"] <= tolerance["delta_cfi_excess_max"]:
            out["passed"] = True
            out["reason"] = "near_pass_tolerance"
            return out
        if "delta_rmsea" in excess and excess["delta_rmsea"] <= tolerance["delta_rmsea_excess_max"]:
            out["passed"] = True
            out["reason"] = "near_pass_tolerance"
            return out
        if "delta_srmr" in excess and excess["delta_srmr"] <= tolerance["delta_srmr_excess_max"]:
            out["passed"] = True
            out["reason"] = "near_pass_tolerance"
            return out

    failed_parts: list[str] = []
    if d_cfi is not None and not (d_cfi >= thresholds["delta_cfi_min"]):
        failed_parts.append("delta_cfi")
    if d_rmsea is not None and not (d_rmsea <= thresholds["delta_rmsea_max"]):
        failed_parts.append("delta_rmsea")
    if d_srmr is not None and not (d_srmr <= thresholds["delta_srmr_max"]):
        failed_parts.append("delta_srmr")
    out["reason"] = "failed_" + ",".join(failed_parts) if failed_parts else "criteria_not_evaluable"
    return out


def _input_data_path(root: Path, paths_cfg: dict[str, Any], cohort: str) -> Path:
    processed_dir = _resolve_path(paths_cfg["processed_dir"], root)
    preferred = processed_dir / f"{cohort}_cfa_resid.csv"
    fallback = processed_dir / f"{cohort}_cfa.csv"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing cohort input for race invariance: expected {preferred} or {fallback}")


def _observed_tests(cohort: str, models_cfg: dict[str, Any]) -> list[str]:
    if cohort == "cnlsy":
        return [str(x) for x in models_cfg.get("cnlsy_single_factor", [])]
    return hierarchical_subtests(models_cfg)


def _model_syntax(cohort: str, models_cfg: dict[str, Any]) -> str:
    return cnlsy_model_syntax(models_cfg) if cohort == "cnlsy" else hierarchical_model_syntax(models_cfg)


def _group_metadata(df: pd.DataFrame, group_col: str) -> tuple[int, int, pd.Series]:
    total = int(len(df))
    counts = df[group_col].dropna().astype("string").str.strip()
    counts = counts.loc[counts.ne("")]
    value_counts = counts.value_counts(dropna=True)
    return total, int(len(counts)), value_counts


def run_race_ethnicity_invariance(
    *,
    root: Path,
    cohorts: list[str],
    min_group_n: int = 50,
    reference_group: str = "NON-BLACK, NON-HISPANIC",
    group_col: str = "race_ethnicity_3cat",
    python_fallback: bool = False,
    summary_output_path: Path = Path("outputs/tables/race_invariance_summary.csv"),
    transition_output_path: Path = Path("outputs/tables/race_invariance_transition_checks.csv"),
    eligibility_output_path: Path = Path("outputs/tables/race_invariance_eligibility.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    outputs_dir = _resolve_path(paths_cfg["outputs_dir"], root)
    sem_interim_root = root / "data" / "interim" / "sem_race_invariance"
    fit_root = outputs_dir / "model_fits" / "race_invariance"
    thresholds = _thresholds(models_cfg)
    tolerance = _tolerance(models_cfg)

    summary_rows: list[pd.DataFrame] = []
    transition_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []

    for cohort in cohorts:
        source_path = _input_data_path(root, paths_cfg, cohort)
        source_data = relative_path(root, source_path)
        df = pd.read_csv(source_path, low_memory=False)
        total_n = int(len(df))
        if group_col not in df.columns:
            eligibility_rows.append(_empty_eligibility(cohort, "missing_group_col", source_data, n_total=total_n, min_group_n=min_group_n))
            transition_rows.extend(
                [
                    _empty_transition(cohort, "missing_group_col", "", "configural->metric"),
                    _empty_transition(cohort, "missing_group_col", "", "metric->scalar"),
                ]
            )
            continue

        n_total, n_nonmissing_group, counts = _group_metadata(df, group_col)
        n_groups = int(len(counts))
        group_levels = ",".join(counts.index.astype(str).tolist())
        group_counts = ";".join([f"{idx}:{int(val)}" for idx, val in counts.items()])
        smallest_group_n = int(counts.min()) if not counts.empty else 0
        if n_groups < 2:
            eligibility_rows.append(
                _empty_eligibility(
                    cohort,
                    "fewer_than_two_groups",
                    source_data,
                    n_total=n_total,
                    n_nonmissing_group=n_nonmissing_group,
                    n_groups=n_groups,
                    group_levels=group_levels,
                    group_counts=group_counts,
                    smallest_group_n=smallest_group_n,
                    min_group_n=min_group_n,
                )
            )
            transition_rows.extend(
                [
                    _empty_transition(cohort, "fewer_than_two_groups", "", "configural->metric"),
                    _empty_transition(cohort, "fewer_than_two_groups", "", "metric->scalar"),
                ]
            )
            continue
        if smallest_group_n < min_group_n:
            eligibility_rows.append(
                _empty_eligibility(
                    cohort,
                    "insufficient_group_n",
                    source_data,
                    n_total=n_total,
                    n_nonmissing_group=n_nonmissing_group,
                    n_groups=n_groups,
                    group_levels=group_levels,
                    group_counts=group_counts,
                    smallest_group_n=smallest_group_n,
                    min_group_n=min_group_n,
                )
            )
            transition_rows.extend(
                [
                    _empty_transition(cohort, "insufficient_group_n", "", "configural->metric"),
                    _empty_transition(cohort, "insufficient_group_n", "", "metric->scalar"),
                ]
            )
            continue

        observed = _observed_tests(cohort, models_cfg)
        required = [group_col, *observed]
        work = df[[col for col in required if col in df.columns]].copy()
        missing_observed = [col for col in observed if col not in work.columns]
        if missing_observed:
            eligibility_rows.append(
                _empty_eligibility(
                    cohort,
                    "missing_observed_tests",
                    source_data,
                    n_total=n_total,
                    n_nonmissing_group=n_nonmissing_group,
                    n_groups=n_groups,
                    group_levels=group_levels,
                    group_counts=group_counts,
                    smallest_group_n=smallest_group_n,
                    min_group_n=min_group_n,
                )
            )
            transition_rows.extend(
                [
                    _empty_transition(cohort, "missing_observed_tests", "", "configural->metric"),
                    _empty_transition(cohort, "missing_observed_tests", "", "metric->scalar"),
                ]
            )
            continue

        temp_csv = fit_root / cohort / "race_sem_source.csv"
        temp_csv.parent.mkdir(parents=True, exist_ok=True)
        work.to_csv(temp_csv, index=False)

        request_payload = {
            "cohort": cohort,
            "data_csv": str(temp_csv),
            "group_col": group_col,
            "reference_group": reference_group,
            "estimator": "MLR",
            "missing": "fiml",
            "std_lv": bool((load_yaml(root / COHORT_CONFIGS[cohort]).get("sem_fit", {}) or {}).get("std_lv", True)),
            "invariance_steps": _expected_steps(models_cfg),
            "partial_intercepts": [],
            "observed_tests": observed,
            "se_mode": "standard",
            "cluster_col": None,
            "weight_col": None,
        }
        sem_ctx = write_sem_inputs(
            cohort=cohort,
            df_path=temp_csv,
            sem_interim_dir=sem_interim_root,
            model_syntax=_model_syntax(cohort, models_cfg),
            request_payload=request_payload,
        )
        fit_dir = fit_root / cohort
        fit_dir.mkdir(parents=True, exist_ok=True)

        run_reason = ""
        try:
            if python_fallback:
                run_python_sem_fallback(
                    cohort=cohort,
                    data_csv=sem_ctx.data_csv,
                    outdir=fit_dir,
                    group_col=group_col,
                    models_cfg=models_cfg,
                    invariance_steps=_expected_steps(models_cfg),
                    observed_tests=observed,
                )
            else:
                if rscript_path() is None:
                    raise FileNotFoundError("rscript_not_available")
                run_sem_r_script(root / "scripts" / "sem_fit.R", sem_ctx.request_file, fit_dir)
        except Exception as exc:
            run_reason = f"sem_failed:{str(exc)}"
            eligibility_rows.append(
                _empty_eligibility(
                    cohort,
                    run_reason,
                    source_data,
                    n_total=n_total,
                    n_nonmissing_group=n_nonmissing_group,
                    n_groups=n_groups,
                    group_levels=group_levels,
                    group_counts=group_counts,
                    smallest_group_n=smallest_group_n,
                    min_group_n=min_group_n,
                    fit_dir=relative_path(root, fit_dir),
                    request_file=relative_path(root, sem_ctx.request_file),
                    model_syntax_file=relative_path(root, sem_ctx.model_syntax_file),
                )
            )
            transition_rows.extend(
                [
                    _empty_transition(cohort, run_reason, relative_path(root, fit_dir), "configural->metric"),
                    _empty_transition(cohort, run_reason, relative_path(root, fit_dir), "metric->scalar"),
                ]
            )
            continue

        fit_indices_path = fit_dir / "fit_indices.csv"
        if not fit_indices_path.exists():
            run_reason = "missing_fit_indices"
            eligibility_rows.append(
                _empty_eligibility(
                    cohort,
                    run_reason,
                    source_data,
                    n_total=n_total,
                    n_nonmissing_group=n_nonmissing_group,
                    n_groups=n_groups,
                    group_levels=group_levels,
                    group_counts=group_counts,
                    smallest_group_n=smallest_group_n,
                    min_group_n=min_group_n,
                    fit_dir=relative_path(root, fit_dir),
                    request_file=relative_path(root, sem_ctx.request_file),
                    model_syntax_file=relative_path(root, sem_ctx.model_syntax_file),
                )
            )
            transition_rows.extend(
                [
                    _empty_transition(cohort, run_reason, relative_path(root, fit_dir), "configural->metric"),
                    _empty_transition(cohort, run_reason, relative_path(root, fit_dir), "metric->scalar"),
                ]
            )
            continue

        summary = _add_deltas(_ensure_step_rows(pd.read_csv(fit_indices_path), _expected_steps(models_cfg), cohort))
        summary["status"] = "computed"
        summary["reason"] = pd.NA
        summary["group_col"] = group_col
        summary["reference_group"] = reference_group
        summary["n_total"] = n_total
        summary["n_nonmissing_group"] = n_nonmissing_group
        summary["n_groups"] = n_groups
        summary["group_levels"] = group_levels
        summary["group_counts"] = group_counts
        summary["smallest_group_n"] = smallest_group_n
        summary["min_group_n"] = min_group_n
        summary["source_data"] = source_data
        summary["fit_dir"] = relative_path(root, fit_dir)
        summary_rows.append(summary)

        metric_check = _transition_check(
            summary=summary,
            cohort=cohort,
            prev_step="configural",
            next_step="metric",
            thresholds=thresholds["metric"],
            tolerance=tolerance["metric"],
        )
        metric_check["fit_dir"] = relative_path(root, fit_dir)
        scalar_check = _transition_check(
            summary=summary,
            cohort=cohort,
            prev_step="metric",
            next_step="scalar",
            thresholds=thresholds["scalar"],
            tolerance=tolerance["scalar"],
        )
        scalar_check["fit_dir"] = relative_path(root, fit_dir)
        transition_rows.extend([metric_check, scalar_check])

        metric_pass = bool(metric_check.get("passed", False)) and str(metric_check.get("status", "")) == "computed"
        scalar_pass = bool(scalar_check.get("passed", False)) and str(scalar_check.get("status", "")) == "computed"
        eligibility_rows.append(
            {
                "cohort": cohort,
                "status": "computed",
                "reason": pd.NA,
                "group_col": group_col,
                "reference_group": reference_group,
                "n_total": n_total,
                "n_nonmissing_group": n_nonmissing_group,
                "n_groups": n_groups,
                "group_levels": group_levels,
                "group_counts": group_counts,
                "smallest_group_n": smallest_group_n,
                "min_group_n": min_group_n,
                "metric_pass": metric_pass,
                "scalar_pass": scalar_pass,
                "race_invariance_ok_for_vr": metric_pass,
                "race_invariance_ok_for_d": metric_pass and scalar_pass,
                "reason_vr_g": pd.NA if metric_pass else f"metric_gate:{metric_check.get('reason')}",
                "reason_d_g": pd.NA if (metric_pass and scalar_pass) else (
                    f"metric_gate:{metric_check.get('reason')}" if not metric_pass else f"scalar_gate:{scalar_check.get('reason')}"
                ),
                "source_data": source_data,
                "fit_dir": relative_path(root, fit_dir),
                "request_file": relative_path(root, sem_ctx.request_file),
                "model_syntax_file": relative_path(root, sem_ctx.model_syntax_file),
            }
        )

    summary_out = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    transition_out = pd.DataFrame(transition_rows)
    eligibility_out = pd.DataFrame(eligibility_rows)

    if not summary_out.empty:
        summary_columns = list(summary_out.columns)
        for col in ("status", "reason", "cohort", "model_step", "cfi", "rmsea", "srmr", "delta_cfi", "delta_rmsea", "delta_srmr", "group_col", "reference_group", "n_total", "n_nonmissing_group", "n_groups", "group_levels", "group_counts", "smallest_group_n", "min_group_n", "source_data", "fit_dir"):
            if col not in summary_columns:
                summary_out[col] = pd.NA
        summary_out = summary_out[[col for col in ("cohort", "model_step", "status", "reason", "cfi", "tli", "rmsea", "srmr", "chisq_scaled", "df", "aic", "bic", "delta_cfi", "delta_rmsea", "delta_srmr", "group_col", "reference_group", "n_total", "n_nonmissing_group", "n_groups", "group_levels", "group_counts", "smallest_group_n", "min_group_n", "source_data", "fit_dir") if col in summary_out.columns]].copy()

    for col in TRANSITION_COLUMNS:
        if col not in transition_out.columns:
            transition_out[col] = pd.NA
    transition_out = transition_out[TRANSITION_COLUMNS].copy()

    for col in ELIGIBILITY_COLUMNS:
        if col not in eligibility_out.columns:
            eligibility_out[col] = pd.NA
    eligibility_out = eligibility_out[ELIGIBILITY_COLUMNS].copy()

    summary_target = summary_output_path if summary_output_path.is_absolute() else root / summary_output_path
    transition_target = transition_output_path if transition_output_path.is_absolute() else root / transition_output_path
    eligibility_target = eligibility_output_path if eligibility_output_path.is_absolute() else root / eligibility_output_path
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    transition_target.parent.mkdir(parents=True, exist_ok=True)
    eligibility_target.parent.mkdir(parents=True, exist_ok=True)
    summary_out.to_csv(summary_target, index=False)
    transition_out.to_csv(transition_target, index=False)
    eligibility_out.to_csv(eligibility_target, index=False)
    return summary_out, transition_out, eligibility_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exploratory race/ethnicity measurement invariance tables.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all supported cohorts.")
    parser.add_argument("--min-group-n", type=int, default=50, help="Minimum rows required in each race/ethnicity group.")
    parser.add_argument("--reference-group", default="NON-BLACK, NON-HISPANIC", help="Reference group label passed to lavaan.")
    parser.add_argument("--group-col", default="race_ethnicity_3cat", help="Grouping column in processed cohort data.")
    parser.add_argument("--python-fallback", action="store_true", help="Use the Python SEM fallback instead of R/lavaan.")
    parser.add_argument("--summary-output-path", type=Path, default=Path("outputs/tables/race_invariance_summary.csv"))
    parser.add_argument("--transition-output-path", type=Path, default=Path("outputs/tables/race_invariance_transition_checks.csv"))
    parser.add_argument("--eligibility-output-path", type=Path, default=Path("outputs/tables/race_invariance_eligibility.csv"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        summary, transitions, eligibility = run_race_ethnicity_invariance(
            root=root,
            cohorts=_cohorts_from_args(args),
            min_group_n=int(args.min_group_n),
            reference_group=str(args.reference_group),
            group_col=str(args.group_col),
            python_fallback=bool(args.python_fallback),
            summary_output_path=args.summary_output_path,
            transition_output_path=args.transition_output_path,
            eligibility_output_path=args.eligibility_output_path,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.summary_output_path if args.summary_output_path.is_absolute() else root / args.summary_output_path}")
    print(f"[ok] wrote {args.transition_output_path if args.transition_output_path.is_absolute() else root / args.transition_output_path}")
    print(f"[ok] wrote {args.eligibility_output_path if args.eligibility_output_path.is_absolute() else root / args.eligibility_output_path}")
    print(f"[ok] computed cohorts: {int((eligibility.get('status') == 'computed').sum()) if 'status' in eligibility.columns else 0}")
    print(f"[ok] transition rows: {int((transitions.get('status') == 'computed').sum()) if 'status' in transitions.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
