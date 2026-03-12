#!/usr/bin/env python3
from __future__ import annotations

import argparse
from string import Formatter
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root, relative_path
from nls_pipeline.plots import save_forest_plot

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

DEFAULT_ROBUSTNESS_CONFIG: dict[str, Any] = {
    "sampling_schemes": ["sibling_restricted", "full_cohort", "one_pair_per_family"],
    "age_adjustment": ["quadratic", "cubic", "spline"],
    "residualization_mode": ["pooled", "within_sex"],
    "inference": ["robust_cluster", "family_bootstrap"],
    "weights": ["unweighted", "weighted"],
    "harmonization_methods": ["zscore_by_branch"],
    "harmonization_baseline_method": "signed_merge",
}
DEFAULT_RERUN_TIMEOUT_SECONDS = 300.0

ROBUSTNESS_MODEL_FORMS: tuple[str, ...] = (
    "baseline",
    "single_factor_alt",
    "bifactor_alt",
)
RERUN_FAMILIES: tuple[str, ...] = ("sampling", "age_adjustment", "model_form", "inference", "weights")
HARMONIZATION_FAMILY = "harmonization"
HARMONIZATION_BASELINE_METHOD = "signed_merge"
RERUN_FAMILIES = RERUN_FAMILIES + (HARMONIZATION_FAMILY,)
GLOBAL_RERUN_COHORT = "__all__"
RERUN_LOG_PATH = "robustness_rerun_log.csv"

RERUN_STATUS_AVAILABLE = "available"
RERUN_STATUS_NOT_CONFIGURED = "not_configured"
RERUN_STATUS_NOT_RUN = "not_run"
RERUN_STATUS_SUCCESS = "success"
RERUN_STATUS_FAILURE = "failure"
RERUN_STATUS_TIMEOUT = "timeout"
RERUN_STATUS_FORMAT_ERROR = "format_error"
ALLOWED_RERUN_PLACEHOLDERS: tuple[str, ...] = (
    "python_executable",
    "project_root",
    "outputs_dir",
    "tables_dir",
    "cohort",
    "variant_token",
    "variant",
    "robustness_family",
)
NOT_FEASIBLE_STATUS = "not_feasible"

SAMPLING_REQUIRED = (
    "cohort",
    "sampling_scheme",
    "status",
    "n_input",
    "n_after_age",
    "n_after_test_rule",
    "n_after_dedupe",
    "d_g",
    "se_d_g",
    "ci_low",
    "ci_high",
)
AGE_REQUIRED = (
    "cohort",
    "age_adjustment",
    "residualization_mode",
    "status",
    "n_subtests",
    "n_used_mean",
    "r2_mean",
    "avg_resid_sd",
)
MODEL_FORM_REQUIRED = ("cohort", "model_form", "status", "cfi", "rmsea", "srmr", "delta_cfi")
INFERENCE_REQUIRED = (
    "cohort",
    "inference_method",
    "estimate_type",
    "status",
    "reason",
    "estimate",
    "se",
    "ci_low",
    "ci_high",
)
WEIGHTS_REQUIRED = (
    "cohort",
    "weight_mode",
    "estimate_type",
    "status",
    "reason",
    "estimate",
    "se",
    "ci_low",
    "ci_high",
)
HARMONIZATION_REQUIRED = (
    "cohort",
    "estimate_type",
    "baseline_method",
    "alternative_method",
    "status",
    "baseline_estimate",
    "baseline_se",
    "baseline_ci_low",
    "baseline_ci_high",
    "alternative_estimate",
    "alternative_se",
    "alternative_ci_low",
    "alternative_ci_high",
    "delta_estimate",
)
MANIFEST_REQUIRED = (
    "cohort",
    "robustness_family",
    "variant_token",
    "status",
    "source_paths",
    "estimate_type",
)


def _add_manifest_entry(
    manifest_rows: list[dict[str, Any]],
    *,
    project_root: Path,
    cohort: str,
    robustness_family: str,
    variant_token: str,
    status: str,
    source_paths: Iterable[Path],
    estimate_type: str | None = None,
    rerun_status: str | None = None,
    rerun_command: str | None = None,
    rerun_return_code: int | None = None,
) -> None:
    manifest_rows.append(
        {
            "cohort": cohort,
            "robustness_family": robustness_family,
            "variant_token": variant_token,
            "status": status,
            "source_paths": ";".join(
                [relative_path(project_root, p) for p in source_paths if p is not None and str(p).strip()]
            ),
            "estimate_type": estimate_type if estimate_type is not None else pd.NA,
            "rerun_status": rerun_status if rerun_status is not None else pd.NA,
            "rerun_command": rerun_command if rerun_command is not None else pd.NA,
            "rerun_return_code": rerun_return_code if rerun_return_code is not None else pd.NA,
        }
    )


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _variant_path(base: Path, token: str) -> Path:
    safe_token = str(token).strip().replace(" ", "_")
    return base.with_name(f"{base.stem}_{safe_token}{base.suffix}")


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _as_list(value: Any, fallback: list[str], *, field: str) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in list(value)]
    _ = field  # explicit local variable to satisfy lint if configured
    return fallback


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_float(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def _first_available_float(row: dict[str, Any], *columns: str) -> float | None:
    for column in columns:
        value = _safe_float(row.get(column))
        if value is not None:
            return value
    return None


def _cohort_first_row(df: pd.DataFrame, cohort: str) -> dict[str, Any]:
    if df.empty or "cohort" not in df.columns:
        return {}
    rows = df[df["cohort"] == cohort]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _row_declares_not_feasible(row: dict[str, Any]) -> bool:
    if not row:
        return False
    token = str(row.get("status", "")).strip().lower()
    return token in {NOT_FEASIBLE_STATUS, "infeasible", "not-feasible"}


def _cohort_base_row(sample_counts: pd.DataFrame, cohort: str) -> dict[str, Any]:
    if sample_counts.empty or "cohort" not in sample_counts.columns:
        return {}
    rows = sample_counts[sample_counts["cohort"] == cohort]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _cohort_diag_rows(path: Path, cohort: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = pd.read_csv(path)
    if rows.empty or "cohort" not in rows.columns:
        return []
    return rows[rows["cohort"] == cohort].to_dict(orient="records")


def _align_interval_to_point(
    *,
    point_estimate: float | None,
    source_estimate: float | None,
    ci_low: float | None,
    ci_high: float | None,
) -> tuple[float | None, float | None]:
    if point_estimate is None or ci_low is None or ci_high is None:
        return ci_low, ci_high
    if ci_low <= point_estimate <= ci_high:
        return ci_low, ci_high

    if source_estimate is not None:
        delta = float(point_estimate) - float(source_estimate)
    else:
        midpoint = (float(ci_low) + float(ci_high)) / 2.0
        delta = float(point_estimate) - midpoint

    shifted_low = float(ci_low) + delta
    shifted_high = float(ci_high) + delta
    if shifted_low <= shifted_high:
        return shifted_low, shifted_high
    return shifted_high, shifted_low


def _cohort_diag_rows_for_variant(
    tables_dir: Path,
    cohort: str,
    age_adjustment: str,
    residualization_mode: str,
) -> tuple[list[dict[str, Any]], Path | None]:
    base_cohort_path = tables_dir / f"residualization_diagnostics_{cohort}.csv"
    base_all_path = tables_dir / "residualization_diagnostics_all.csv"
    tokens = [
        f"{age_adjustment}_{residualization_mode}",
        age_adjustment,
        residualization_mode,
    ]
    seen_tokens: set[str] = set()
    for token in tokens:
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        for source_path in (_variant_path(base_cohort_path, token), _variant_path(base_all_path, token)):
            rows = _cohort_diag_rows(source_path, cohort)
            if rows:
                return rows, source_path
    return [], None


def _harmonization_variant_paths(tables_dir: Path, method: str) -> list[Path]:
    token = str(method).strip().replace(" ", "_")
    return [
        tables_dir / f"g_mean_diff_harmonization_{token}.csv",
        tables_dir / f"g_mean_diff_{token}.csv",
    ]


def _read_harmonization_variant_row(
    tables_dir: Path,
    cohort: str,
    method: str,
) -> tuple[dict[str, Any], Path | None]:
    for variant_path in _harmonization_variant_paths(tables_dir, method):
        variant_rows = _cohort_diag_rows(variant_path, cohort)
        if not variant_rows:
            continue
        return variant_rows[0], variant_path
    return {}, None


def _build_harmonization_rows(
    tables_dir: Path,
    cohorts: Iterable[str],
    baseline_method: str,
    harmonization_methods: list[str],
    manifest_rows: list[dict[str, Any]],
    project_root: Path,
    rerun_outcomes: dict[tuple[str, str, str], dict[str, str | int | float | None]] | None = None,
    placeholder_variants: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_candidates = [tables_dir / "g_mean_diff.csv", *_harmonization_variant_paths(tables_dir, baseline_method)]
    for cohort in cohorts:
        if cohort != "nlsy97":
            continue
        baseline_row: dict[str, Any] = {}
        baseline_source_path: Path | None = None
        for candidate in baseline_candidates:
            baseline_df = _safe_read_csv(candidate)
            candidate_row = _cohort_first_row(baseline_df, cohort)
            if candidate_row:
                baseline_row = candidate_row
                baseline_source_path = candidate
                break
        baseline = (
            {
                "estimate": _safe_float(baseline_row.get("d_g")),
                "se": _first_available_float(baseline_row, "SE_d_g", "SE"),
                "ci_low": _first_available_float(baseline_row, "ci_low_d_g", "ci_low"),
                "ci_high": _first_available_float(baseline_row, "ci_high_d_g", "ci_high"),
            }
            if baseline_row
            else {"estimate": None, "se": None, "ci_low": None, "ci_high": None}
        )
        for method in harmonization_methods:
            if not method or str(method) == baseline_method:
                continue
            alt_row, alt_source = _read_harmonization_variant_row(tables_dir, cohort, method)
            has_inputs = any(p.exists() for p in _harmonization_variant_paths(tables_dir, method))
            alternative = {
                "estimate": _safe_float(alt_row.get("d_g")) if alt_row else pd.NA,
                "se": _first_available_float(alt_row, "SE_d_g", "SE") if alt_row else pd.NA,
                "ci_low": _first_available_float(alt_row, "ci_low_d_g", "ci_low") if alt_row else pd.NA,
                "ci_high": _first_available_float(alt_row, "ci_high_d_g", "ci_high") if alt_row else pd.NA,
            }
            if baseline_row and alt_row and baseline["estimate"] is not None and alternative["estimate"] is not None:
                delta = float(alternative["estimate"]) - float(baseline["estimate"])
                status = "computed"
            else:
                if not baseline_row:
                    status = "baseline_missing"
                else:
                    status = "missing_source" if has_inputs else "not_run_placeholder"
                delta = pd.NA

            if status == "computed" and _is_placeholder_variant(placeholder_variants, HARMONIZATION_FAMILY, method):
                status = NOT_FEASIBLE_STATUS

            row: dict[str, Any] = {
                "cohort": cohort,
                "estimate_type": "d_g",
                "baseline_method": baseline_method,
                "alternative_method": method,
                "status": status,
                "baseline_estimate": baseline["estimate"] if baseline["estimate"] is not None else pd.NA,
                "baseline_se": baseline["se"] if baseline["se"] is not None else pd.NA,
                "baseline_ci_low": baseline["ci_low"] if baseline["ci_low"] is not None else pd.NA,
                "baseline_ci_high": baseline["ci_high"] if baseline["ci_high"] is not None else pd.NA,
                "alternative_estimate": alternative["estimate"],
                "alternative_se": alternative["se"],
                "alternative_ci_low": alternative["ci_low"],
                "alternative_ci_high": alternative["ci_high"],
                "delta_estimate": delta,
            }
            source_paths = [baseline_source_path] if baseline_source_path is not None else [baseline_candidates[0]]
            if alt_source is not None:
                source_paths.append(alt_source)
            _add_manifest_entry(
                manifest_rows,
                project_root=project_root,
                cohort=cohort,
                robustness_family=HARMONIZATION_FAMILY,
                variant_token=method,
                status=status,
                source_paths=source_paths,
                estimate_type="d_g",
                rerun_status=_rerun_outcome_fields(rerun_outcomes, HARMONIZATION_FAMILY, method, cohort)[0],
                rerun_command=_rerun_outcome_fields(rerun_outcomes, HARMONIZATION_FAMILY, method, cohort)[1],
                rerun_return_code=_rerun_outcome_fields(rerun_outcomes, HARMONIZATION_FAMILY, method, cohort)[2],
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _build_sampling_rows(
    tables_dir: Path,
    sample_counts: pd.DataFrame,
    g_mean: pd.DataFrame,
    cohorts: Iterable[str],
    sampling_schemes: list[str],
    manifest_rows: list[dict[str, Any]],
    project_root: Path,
    rerun_outcomes: dict[tuple[str, str, str], dict[str, str | int | float | None]] | None = None,
    placeholder_variants: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sample_counts_path = tables_dir / "sample_counts.csv"
    g_mean_path = tables_dir / "g_mean_diff.csv"
    for cohort in cohorts:
        base = _cohort_base_row(sample_counts, cohort)
        for idx, scheme in enumerate(sampling_schemes):
            row: dict[str, Any] = {
                "cohort": cohort,
                "sampling_scheme": scheme,
            }
            if idx == 0:
                g_row = _cohort_first_row(g_mean, cohort)
                baseline_d_g = _safe_float(g_row.get("d_g")) if g_row else None
                if base:
                    row.update(
                        {
                            "status": "computed" if baseline_d_g is not None else "baseline_missing",
                            "n_input": base.get("n_input", pd.NA),
                            "n_after_age": base.get("n_after_age", pd.NA),
                            "n_after_test_rule": base.get("n_after_test_rule", pd.NA),
                            "n_after_dedupe": base.get("n_after_dedupe", pd.NA),
                            "d_g": baseline_d_g if baseline_d_g is not None else pd.NA,
                            "se_d_g": _first_available_float(g_row, "SE_d_g", "SE") if g_row else pd.NA,
                            "ci_low": _first_available_float(g_row, "ci_low_d_g", "ci_low") if g_row else pd.NA,
                            "ci_high": _first_available_float(g_row, "ci_high_d_g", "ci_high") if g_row else pd.NA,
                        }
                    )
                else:
                    row.update(
                        {
                            "status": "not_run_placeholder",
                            "n_input": pd.NA,
                            "n_after_age": pd.NA,
                            "n_after_test_rule": pd.NA,
                            "n_after_dedupe": pd.NA,
                            "d_g": pd.NA,
                            "se_d_g": pd.NA,
                            "ci_low": pd.NA,
                            "ci_high": pd.NA,
                        }
                    )
            else:
                variant_sample_counts_path = _variant_path(sample_counts_path, scheme)
                variant_g_mean_path = _variant_path(g_mean_path, scheme)
                variant_counts = _safe_read_csv(variant_sample_counts_path)
                variant_g_mean = _safe_read_csv(variant_g_mean_path)
                variant_base = _cohort_base_row(variant_counts, cohort)
                variant_mean = _cohort_first_row(variant_g_mean, cohort)
                variant_d_g = _safe_float(variant_mean.get("d_g")) if variant_mean else None
                has_inputs = variant_sample_counts_path.exists() and variant_g_mean_path.exists()
                if has_inputs and variant_base and variant_mean and variant_d_g is not None:
                    row.update(
                        {
                            "status": "computed",
                            "n_input": variant_base.get("n_input", pd.NA),
                            "n_after_age": variant_base.get("n_after_age", pd.NA),
                            "n_after_test_rule": variant_base.get("n_after_test_rule", pd.NA),
                            "n_after_dedupe": variant_base.get("n_after_dedupe", pd.NA),
                            "d_g": variant_d_g,
                            "se_d_g": _first_available_float(variant_mean, "SE_d_g", "SE"),
                            "ci_low": _first_available_float(variant_mean, "ci_low_d_g", "ci_low"),
                            "ci_high": _first_available_float(variant_mean, "ci_high_d_g", "ci_high"),
                        }
                    )
                else:
                    row.update(
                        {
                            "status": "missing_source" if has_inputs else "not_run_placeholder",
                            "n_input": pd.NA,
                            "n_after_age": pd.NA,
                            "n_after_test_rule": pd.NA,
                            "n_after_dedupe": pd.NA,
                            "d_g": pd.NA,
                            "se_d_g": pd.NA,
                            "ci_low": pd.NA,
                            "ci_high": pd.NA,
                        }
                    )
                if row.get("status") == "computed" and _is_placeholder_variant(placeholder_variants, "sampling", scheme):
                    row["status"] = NOT_FEASIBLE_STATUS
            source_paths = [sample_counts_path, g_mean_path] if idx == 0 else [
                variant_sample_counts_path,
                variant_g_mean_path,
            ]
            rerun_status, rerun_command, rerun_return_code = _rerun_outcome_fields(
                rerun_outcomes, "sampling", scheme, None
            )
            _add_manifest_entry(
                manifest_rows,
                project_root=project_root,
                cohort=cohort,
                robustness_family="sampling",
                variant_token=scheme,
                status=row["status"],
                source_paths=source_paths,
                rerun_status=rerun_status,
                rerun_command=rerun_command,
                rerun_return_code=rerun_return_code,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_diagnostics(diag_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not diag_rows:
        return {}
    df = pd.DataFrame(diag_rows)
    numeric_cols = [c for c in ("n_used", "r2", "resid_sd") if c in df.columns]
    if not numeric_cols:
        return {}
    values = df[numeric_cols].astype(float)
    return {
        "n_subtests": len(df),
        "n_used_mean": float(values["n_used"].mean()) if "n_used" in values else pd.NA,
        "r2_mean": float(values["r2"].mean()) if "r2" in values else pd.NA,
        "avg_resid_sd": float(values["resid_sd"].mean()) if "resid_sd" in values else pd.NA,
    }


def _build_age_rows(
    outputs_dir: Path,
    cohorts: Iterable[str],
    age_adjustment: list[str],
    residualization_mode: list[str],
    manifest_rows: list[dict[str, Any]],
    project_root: Path,
    rerun_outcomes: dict[tuple[str, str, str], dict[str, str | int | float | None]] | None = None,
    placeholder_variants: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    tables_dir = outputs_dir / "tables"
    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        base_rows = _cohort_diag_rows(tables_dir / f"residualization_diagnostics_{cohort}.csv", cohort)
        if not base_rows:
            base_rows = _cohort_diag_rows(tables_dir / "residualization_diagnostics_all.csv", cohort)

        base = _aggregate_diagnostics(base_rows)

        for aa in age_adjustment:
            for mode in residualization_mode:
                is_baseline = aa == age_adjustment[0] and mode == residualization_mode[0]
                variant_rows, variant_source = _cohort_diag_rows_for_variant(
                    tables_dir,
                    cohort,
                    aa,
                    mode,
                )
                has_variant_input = False
                if variant_rows:
                    source_variant = _aggregate_diagnostics(variant_rows)
                else:
                    source_variant = {}
                if not is_baseline:
                    variant_cohort_path = tables_dir / f"residualization_diagnostics_{cohort}_{aa}_{mode}.csv"
                    variant_all_path = tables_dir / f"residualization_diagnostics_all_{aa}_{mode}.csv"
                    has_variant_input = variant_cohort_path.exists() or variant_all_path.exists()
                row = {
                    "cohort": cohort,
                    "age_adjustment": aa,
                    "residualization_mode": mode,
                }
                if is_baseline and base:
                    row.update(
                        {
                            "status": "computed",
                            "n_subtests": base["n_subtests"],
                            "n_used_mean": base["n_used_mean"],
                            "r2_mean": base["r2_mean"],
                            "avg_resid_sd": base["avg_resid_sd"],
                        }
                    )
                elif (not is_baseline) and source_variant:
                    row.update(
                        {
                            "status": "computed",
                            "n_subtests": source_variant["n_subtests"],
                            "n_used_mean": source_variant["n_used_mean"],
                            "r2_mean": source_variant["r2_mean"],
                            "avg_resid_sd": source_variant["avg_resid_sd"],
                        }
                    )
                    if _is_placeholder_variant(placeholder_variants, "age_adjustment", f"{aa}_{mode}"):
                        row["status"] = NOT_FEASIBLE_STATUS
                else:
                    row.update(
                        {
                            "status": "missing_source" if not is_baseline and has_variant_input else "not_run_placeholder",
                            "n_subtests": pd.NA,
                            "n_used_mean": pd.NA,
                            "r2_mean": pd.NA,
                            "avg_resid_sd": pd.NA,
                        }
                    )
                variant_token = f"{aa}_{mode}"
                if is_baseline:
                    source_paths = []
                    cohort_path = tables_dir / f"residualization_diagnostics_{cohort}.csv"
                    all_path = tables_dir / "residualization_diagnostics_all.csv"
                    if cohort_path.exists():
                        source_paths.append(cohort_path)
                    elif all_path.exists():
                        source_paths.append(all_path)
                else:
                    source_paths = []
                    if variant_source is not None:
                        source_paths.append(variant_source)
                    else:
                        source_paths.extend(
                            [
                                tables_dir / f"residualization_diagnostics_{cohort}_{aa}_{mode}.csv",
                                tables_dir / f"residualization_diagnostics_all_{aa}_{mode}.csv",
                            ]
                        )

                if not is_baseline:
                    variant_source_list = [
                        p
                        for p in source_paths
                        if p.exists() or p == variant_source
                    ]
                    source_paths = variant_source_list

                rerun_status, rerun_command, rerun_return_code = _rerun_outcome_fields(
                    rerun_outcomes, "age_adjustment", f"{aa}_{mode}", cohort
                )
                _add_manifest_entry(
                    manifest_rows,
                    project_root=project_root,
                    cohort=cohort,
                    robustness_family="age_adjustment",
                    variant_token=variant_token,
                    status=row["status"],
                    source_paths=source_paths,
                    estimate_type=pd.NA,
                    rerun_status=rerun_status,
                    rerun_command=rerun_command,
                    rerun_return_code=rerun_return_code,
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _latest_fit_metrics(summary_path: Path) -> tuple[float | None, float | None, float | None]:
    if not summary_path.exists():
        return None, None, None
    df = pd.read_csv(summary_path)
    if df.empty:
        return None, None, None
    if "model_step" in df.columns:
        strict_rows = df[df["model_step"] == "strict"]
        row = strict_rows.iloc[0] if not strict_rows.empty else df.iloc[-1]
    else:
        row = df.iloc[-1]
    return (
        float(row["cfi"]) if "cfi" in row else None,
        float(row["rmsea"]) if "rmsea" in row else None,
        float(row["srmr"]) if "srmr" in row else None,
    )


def _build_model_form_rows(
    outputs_dir: Path,
    cohorts: Iterable[str],
    model_forms: list[str],
    manifest_rows: list[dict[str, Any]],
    project_root: Path,
    rerun_outcomes: dict[tuple[str, str, str], dict[str, str | int | float | None]] | None = None,
    placeholder_variants: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    tables_dir = outputs_dir / "tables"
    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        baseline_path = tables_dir / f"{cohort}_invariance_summary.csv"
        baseline_cfi, baseline_rmsea, baseline_srmr = _latest_fit_metrics(baseline_path)
        for idx, form in enumerate(model_forms):
            row = {"cohort": cohort, "model_form": form}
            source_path = baseline_path if idx == 0 else _variant_path(baseline_path, form)
            cfi, rmsea, srmr = _latest_fit_metrics(source_path)
            has_inputs = source_path.exists()
            if idx == 0 and None not in (cfi, rmsea, srmr):
                row.update(
                    {
                        "status": "computed",
                        "cfi": cfi,
                        "rmsea": rmsea,
                        "srmr": srmr,
                        "delta_cfi": 0.0,
                    }
                )
            elif idx > 0 and has_inputs and None not in (cfi, rmsea, srmr):
                row.update(
                    {
                        "status": "computed",
                        "cfi": cfi,
                        "rmsea": rmsea,
                        "srmr": srmr,
                        "delta_cfi": cfi - baseline_cfi if baseline_cfi is not None else pd.NA,
                    }
                )
                if _is_placeholder_variant(placeholder_variants, "model_form", form):
                    row["status"] = NOT_FEASIBLE_STATUS
            else:
                missing = idx > 0 and has_inputs
                row.update(
                    {
                        "status": "missing_source" if missing else "not_run_placeholder",
                        "cfi": cfi if idx == 0 else pd.NA,
                        "rmsea": rmsea if idx == 0 else pd.NA,
                        "srmr": srmr if idx == 0 else pd.NA,
                        "delta_cfi": 0.0 if idx == 0 and None not in (cfi, rmsea, srmr) else pd.NA,
                    }
                )
            _add_manifest_entry(
                manifest_rows,
                project_root=project_root,
                cohort=cohort,
                robustness_family="model_form",
                variant_token=form,
                status=row["status"],
                source_paths=[source_path],
                estimate_type=pd.NA,
                rerun_status=_rerun_outcome_fields(rerun_outcomes, "model_form", form, cohort)[0],
                rerun_command=_rerun_outcome_fields(rerun_outcomes, "model_form", form, cohort)[1],
                rerun_return_code=_rerun_outcome_fields(rerun_outcomes, "model_form", form, cohort)[2],
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _build_estimate_rows(
    *,
    cohorts: Iterable[str],
    methods: list[str],
    method_column: str,
    g_mean: pd.DataFrame,
    g_vr: pd.DataFrame,
    tables_dir: Path,
    manifest_rows: list[dict[str, Any]],
    project_root: Path,
    rerun_outcomes: dict[tuple[str, str, str], dict[str, str | int | float | None]] | None = None,
    placeholder_variants: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    g_mean_path = tables_dir / "g_mean_diff.csv"
    g_vr_path = tables_dir / "g_variance_ratio.csv"
    rerun_family = "inference" if method_column == "inference_method" else "weights"

    for cohort in cohorts:
        baseline_mean_row = _cohort_first_row(g_mean, cohort)
        baseline_vr_row = _cohort_first_row(g_vr, cohort)
        baseline_dg = _safe_float(baseline_mean_row.get("d_g")) if baseline_mean_row else None
        baseline_vr = _safe_float(baseline_vr_row.get("VR_g")) if baseline_vr_row else None
        for idx, method in enumerate(methods):
            is_baseline = idx == 0
            variant_mean_source_path = _variant_path(g_mean_path, method)
            variant_vr_source_path = _variant_path(g_vr_path, method)
            mean_source = g_mean if is_baseline else _safe_read_csv(variant_mean_source_path)
            vr_source = g_vr if is_baseline else _safe_read_csv(variant_vr_source_path)
            mean_row = _cohort_first_row(mean_source, cohort)
            vr_row = _cohort_first_row(vr_source, cohort)
            dg = _safe_float(mean_row.get("d_g")) if mean_row else None
            dg_se = _first_available_float(mean_row, "SE_d_g", "SE") if mean_row else None
            dg_low = _first_available_float(mean_row, "ci_low_d_g", "ci_low") if mean_row else None
            dg_high = _first_available_float(mean_row, "ci_high_d_g", "ci_high") if mean_row else None
            vr = _safe_float(vr_row.get("VR_g")) if vr_row else None
            vr_se = _safe_float(vr_row.get("SE_logVR")) if vr_row else None
            vr_low = _safe_float(vr_row.get("ci_low")) if vr_row else None
            vr_high = _safe_float(vr_row.get("ci_high")) if vr_row else None
            dg_point_estimate = (
                baseline_dg
                if (
                    not is_baseline
                    and method_column == "inference_method"
                    and method == "family_bootstrap"
                    and baseline_dg is not None
                    and dg is not None
                )
                else dg
            )
            vr_point_estimate = (
                baseline_vr
                if (
                    not is_baseline
                    and method_column == "inference_method"
                    and method == "family_bootstrap"
                    and baseline_vr is not None
                    and vr is not None
                )
                else vr
            )
            dg_reason = str(mean_row.get("reason", "")).strip() if mean_row else ""
            vr_reason = str(vr_row.get("reason", "")).strip() if vr_row else ""
            dg_low_out = dg_low
            dg_high_out = dg_high
            if (
                dg_point_estimate is not None
                and dg is not None
                and abs(float(dg_point_estimate) - float(dg)) > 1e-12
            ):
                dg_low_out, dg_high_out = _align_interval_to_point(
                    point_estimate=dg_point_estimate,
                    source_estimate=dg,
                    ci_low=dg_low,
                    ci_high=dg_high,
                )
            vr_low_out = vr_low
            vr_high_out = vr_high
            if (
                vr_point_estimate is not None
                and vr is not None
                and abs(float(vr_point_estimate) - float(vr)) > 1e-12
            ):
                vr_low_out, vr_high_out = _align_interval_to_point(
                    point_estimate=vr_point_estimate,
                    source_estimate=vr,
                    ci_low=vr_low,
                    ci_high=vr_high,
                )
            has_inputs = (
                (is_baseline)
                or (
                    variant_mean_source_path.exists()
                    and variant_vr_source_path.exists()
                )
            )
            source_paths = [g_mean_path, g_vr_path] if is_baseline else [variant_mean_source_path, variant_vr_source_path]
            if not is_baseline and rerun_family == "weights":
                source_paths.append(tables_dir / "weights_quality_diagnostics.csv")
            row_status, row_command, row_return_code = _rerun_outcome_fields(
                rerun_outcomes, rerun_family, method, None
            )
            force_not_feasible = False
            missing_artifact_not_feasible = False
            if not is_baseline:
                declared_not_feasible = _row_declares_not_feasible(mean_row) or _row_declares_not_feasible(vr_row)
                success_without_artifacts = (
                    rerun_family == "weights"
                    and str(row_status) == RERUN_STATUS_SUCCESS
                    and not has_inputs
                )
                force_not_feasible = declared_not_feasible or success_without_artifacts
                missing_artifact_not_feasible = success_without_artifacts

            d_row: dict[str, Any] = {
                "cohort": cohort,
                method_column: method,
                "estimate_type": "d_g",
            }
            if force_not_feasible:
                d_row.update(
                    {
                        "status": NOT_FEASIBLE_STATUS,
                        "reason": (
                            "missing_variant_artifacts"
                            if missing_artifact_not_feasible
                            else (dg_reason or vr_reason or pd.NA)
                        ),
                        "estimate": pd.NA,
                        "se": pd.NA,
                        "ci_low": pd.NA,
                        "ci_high": pd.NA,
                    }
                )
            elif dg is not None and has_inputs:
                d_row.update(
                    {
                        "status": "computed",
                        "reason": pd.NA,
                        "estimate": dg_point_estimate,
                        "se": dg_se if dg_se is not None else pd.NA,
                        "ci_low": dg_low_out if dg_low_out is not None else pd.NA,
                        "ci_high": dg_high_out if dg_high_out is not None else pd.NA,
                    }
                )
                if not is_baseline and _is_placeholder_variant(placeholder_variants, rerun_family, method):
                    d_row["status"] = NOT_FEASIBLE_STATUS
            else:
                d_row.update(
                    {
                        "status": "baseline_missing" if is_baseline else "missing_source" if has_inputs else "not_run_placeholder",
                        "reason": dg_reason if dg_reason else pd.NA,
                        "estimate": pd.NA,
                        "se": pd.NA,
                        "ci_low": pd.NA,
                        "ci_high": pd.NA,
                    }
                )
            _add_manifest_entry(
                manifest_rows,
                project_root=project_root,
                cohort=cohort,
                robustness_family="inference" if method_column == "inference_method" else "weights",
                variant_token=method,
                status=d_row["status"],
                source_paths=source_paths,
                estimate_type="d_g",
                rerun_status=row_status,
                rerun_command=row_command,
                rerun_return_code=row_return_code,
            )
            rows.append(d_row)

            vr_row_out: dict[str, Any] = {
                "cohort": cohort,
                method_column: method,
                "estimate_type": "vr_g",
            }
            if force_not_feasible:
                vr_row_out.update(
                    {
                        "status": NOT_FEASIBLE_STATUS,
                        "reason": (
                            "missing_variant_artifacts"
                            if missing_artifact_not_feasible
                            else (vr_reason or dg_reason or pd.NA)
                        ),
                        "estimate": pd.NA,
                        "se": pd.NA,
                        "ci_low": pd.NA,
                        "ci_high": pd.NA,
                    }
                )
            elif vr is not None and has_inputs:
                vr_row_out.update(
                    {
                        "status": "computed",
                        "reason": pd.NA,
                        "estimate": vr_point_estimate,
                        "se": vr_se if vr_se is not None else pd.NA,
                        "ci_low": vr_low_out if vr_low_out is not None else pd.NA,
                        "ci_high": vr_high_out if vr_high_out is not None else pd.NA,
                    }
                )
                if not is_baseline and _is_placeholder_variant(placeholder_variants, rerun_family, method):
                    vr_row_out["status"] = NOT_FEASIBLE_STATUS
            else:
                vr_row_out.update(
                    {
                        "status": "baseline_missing" if is_baseline else "missing_source" if has_inputs else "not_run_placeholder",
                        "reason": vr_reason if vr_reason else pd.NA,
                        "estimate": pd.NA,
                        "se": pd.NA,
                        "ci_low": pd.NA,
                        "ci_high": pd.NA,
                    }
                )
            _add_manifest_entry(
                manifest_rows,
                project_root=project_root,
                cohort=cohort,
                robustness_family="inference" if method_column == "inference_method" else "weights",
                variant_token=method,
                status=vr_row_out["status"],
                source_paths=source_paths,
                estimate_type="vr_g",
                rerun_status=row_status,
                rerun_command=row_command,
                rerun_return_code=row_return_code,
            )
            rows.append(vr_row_out)
    return pd.DataFrame(rows)


def _write_sampling_plot(sampling_df: pd.DataFrame, output_path: Path) -> bool:
    if sampling_df.empty:
        return False
    plot_df = sampling_df.dropna(subset=["d_g", "ci_low", "ci_high"]).copy()
    if not plot_df.empty:
        plot_df["d_g"] = pd.to_numeric(plot_df["d_g"], errors="coerce")
        plot_df["ci_low"] = pd.to_numeric(plot_df["ci_low"], errors="coerce")
        plot_df["ci_high"] = pd.to_numeric(plot_df["ci_high"], errors="coerce")
        plot_df = plot_df.dropna(subset=["d_g", "ci_low", "ci_high"])
        # Guard against mixed-scale or malformed intervals where estimate is outside CI.
        plot_df = plot_df[(plot_df["ci_low"] <= plot_df["d_g"]) & (plot_df["d_g"] <= plot_df["ci_high"])]
    if plot_df.empty:
        return False
    plot_df["label"] = plot_df["cohort"].astype(str) + ":" + plot_df["sampling_scheme"].astype(str)
    save_forest_plot(
        plot_df,
        output_path,
        label_col="label",
        estimate_col="d_g",
        lower_col="ci_low",
        upper_col="ci_high",
        title="Sampling robustness (d_g)",
        xlabel="Male minus female latent mean (d_g)",
    )
    return True


def _ensure_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    *,
    preserve_extra: bool = False,
) -> pd.DataFrame:
    for column in required:
        if column not in df.columns:
            df[column] = pd.NA
    if preserve_extra:
        return df
    return df[list(required)]


def _cohort_key(cohort: str | None) -> str:
    return cohort if cohort is not None else GLOBAL_RERUN_COHORT


def _selected_rerun_families(args: argparse.Namespace) -> set[str]:
    if args.rerun_robustness or args.rerun_missing_variants:
        return set(RERUN_FAMILIES)

    selected: set[str] = set()
    if args.rerun_sampling_variants:
        selected.add("sampling")
    if args.rerun_age_adjustment_variants:
        selected.add("age_adjustment")
    if args.rerun_model_forms:
        selected.add("model_form")
    if args.rerun_inference_variants:
        selected.add("inference")
    if args.rerun_weight_variants:
        selected.add("weights")
    if getattr(args, "rerun_harmonization_variants", False):
        selected.add(HARMONIZATION_FAMILY)
    return selected


def _coerce_rerun_commands(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    commands: dict[str, str] = {}
    for token, command in raw.items():
        if isinstance(command, str):
            commands[str(token)] = command
        elif isinstance(command, dict) and isinstance(command.get("command"), str):
            commands[str(token)] = str(command["command"])
    return commands


def _placeholder_variant_map(rerun_commands: dict[str, dict[str, str]]) -> set[tuple[str, str]]:
    placeholders: set[tuple[str, str]] = set()
    for family, variants in rerun_commands.items():
        if not isinstance(variants, dict):
            continue
        for variant, command in variants.items():
            if not isinstance(command, str):
                continue
            if "17_generate_robustness_variant.py" in command:
                placeholders.add((str(family), str(variant)))
    return placeholders


def _is_placeholder_variant(
    placeholder_variants: set[tuple[str, str]] | None,
    family: str,
    variant: str,
) -> bool:
    if not placeholder_variants:
        return False
    return (family, variant) in placeholder_variants


def _build_rerun_contexts(
    tables_dir: Path,
    cohorts: Iterable[str],
    sampling_schemes: list[str],
    age_adjustment: list[str],
    residualization_mode: list[str],
    inference: list[str],
    weights: list[str],
    harmonization_methods: list[str],
    harmonization_baseline_method: str,
) -> list[tuple[str, str, str | None, list[Path], bool]]:
    contexts: list[tuple[str, str, str | None, list[Path], bool]] = []
    cohort_list = [str(c) for c in cohorts]

    if sampling_schemes:
        sample_counts_path = tables_dir / "sample_counts.csv"
        g_mean_path = tables_dir / "g_mean_diff.csv"
        baseline_scheme = sampling_schemes[0]
        for scheme in sampling_schemes:
            if scheme == baseline_scheme:
                continue
            contexts.append(
                (
                    "sampling",
                    scheme,
                    None,
                    [
                        _variant_path(sample_counts_path, scheme),
                        _variant_path(g_mean_path, scheme),
                    ],
                    True,
                )
            )

    if age_adjustment and residualization_mode:
        baseline_pair = (age_adjustment[0], residualization_mode[0])
        for age_adj in age_adjustment:
            for mode in residualization_mode:
                if (age_adj, mode) == baseline_pair:
                    continue
                token = f"{age_adj}_{mode}"
                for cohort in cohort_list:
                    contexts.append(
                        (
                            "age_adjustment",
                            token,
                            cohort,
                            [
                                tables_dir
                                / f"residualization_diagnostics_{cohort}_{age_adj}_{mode}.csv",
                                tables_dir
                                / f"residualization_diagnostics_all_{age_adj}_{mode}.csv",
                            ],
                            False,
                        )
                    )

    model_form_contexts = [
        ("model_form", form, cohort)
        for cohort in cohort_list
        for form in ROBUSTNESS_MODEL_FORMS[1:]
    ]
    for family, form, cohort in model_form_contexts:
        contexts.append(
            (
                family,
                form,
                cohort,
                [
                    _variant_path(
                        tables_dir / f"{cohort}_invariance_summary.csv",
                        form,
                    )
                ],
                True,
            )
        )

    g_mean_path = tables_dir / "g_mean_diff.csv"
    g_vr_path = tables_dir / "g_variance_ratio.csv"
    if inference and len(inference) > 1:
        baseline = inference[0]
        for method in inference:
            if method == baseline:
                continue
            contexts.append(
                (
                    "inference",
                    method,
                    None,
                    [_variant_path(g_mean_path, method), _variant_path(g_vr_path, method)],
                    True,
                )
            )

    if weights and len(weights) > 1:
        baseline = weights[0]
        for method in weights:
            if method == baseline:
                continue
            weight_quality_path = tables_dir / "weights_quality_diagnostics.csv"
            contexts.append(
                (
                    "weights",
                    method,
                    None,
                    [
                        _variant_path(g_mean_path, method),
                        _variant_path(g_vr_path, method),
                        weight_quality_path,
                    ],
                    True,
                )
            )

    if "nlsy97" in cohort_list:
        for method in harmonization_methods:
            if not method or method == harmonization_baseline_method:
                continue
            contexts.append(
                (
                    HARMONIZATION_FAMILY,
                    str(method),
                    "nlsy97",
                    _harmonization_variant_paths(tables_dir, method),
                    False,
                )
            )

    return contexts


def _format_rerun_command(
    command_template: str,
    *,
    project_root: Path,
    outputs_dir: Path,
    tables_dir: Path,
    cohort: str | None,
    variant_token: str,
    robustness_family: str,
) -> str:
    return command_template.format(
        python_executable=sys.executable,
        project_root=str(project_root),
        outputs_dir=str(outputs_dir),
        tables_dir=str(tables_dir),
        cohort=_cohort_key(cohort),
        variant_token=variant_token,
        variant=variant_token,
        robustness_family=robustness_family,
    )


def _unsupported_placeholder_tokens(command_template: str) -> list[str]:
    formatter = Formatter()
    unsupported: list[str] = []
    for _, field_name, _, _ in formatter.parse(command_template):
        if field_name is None or field_name == "":
            continue
        token = str(field_name).split(".", 1)[0].split("[", 1)[0]
        if token not in ALLOWED_RERUN_PLACEHOLDERS and token not in unsupported:
            unsupported.append(token)
    return unsupported


def _path_list(path_values: Iterable[Path], project_root: Path) -> list[str]:
    return [relative_path(project_root, p) for p in path_values]


def _run_configured_reruns(
    *,
    project_root: Path,
    outputs_dir: Path,
    cohorts: Iterable[str],
    robustness_cfg: dict[str, Any],
    rerun_commands: dict[str, dict[str, str]],
    selected_families: set[str],
    rerun_timeout_seconds: float,
    harmonization_baseline_method: str,
) -> tuple[
    dict[tuple[str, str, str], dict[str, str | int | float | None]],
    pd.DataFrame,
]:
    contexts = _build_rerun_contexts(
        outputs_dir / "tables",
        cohorts,
        robustness_cfg["sampling_schemes"],
        robustness_cfg["age_adjustment"],
        robustness_cfg["residualization_mode"],
        robustness_cfg["inference"],
        robustness_cfg["weights"],
        robustness_cfg["harmonization_methods"],
        harmonization_baseline_method,
    )
    outcomes: dict[tuple[str, str, str], dict[str, str | int | float | None]] = {}
    log_rows: list[dict[str, Any]] = []

    tables_dir = outputs_dir / "tables"

    for family, variant_token, cohort, source_paths, all_required in contexts:
        cohort_key = _cohort_key(cohort)
        key = (family, variant_token, cohort_key)
        paths = [p.resolve() for p in source_paths]

        if all_required:
            needs_rerun = any(not p.exists() for p in paths)
        else:
            needs_rerun = all(not p.exists() for p in paths)
        if not needs_rerun:
            outcomes[key] = {
                "cohort": cohort_key,
                "robustness_family": family,
                "variant_token": variant_token,
                "status": RERUN_STATUS_AVAILABLE,
                "command": pd.NA,
                "return_code": pd.NA,
                "elapsed_seconds": pd.NA,
                "error": pd.NA,
                "source_paths": ";".join(_path_list(paths, project_root)),
            }
            log_rows.append(outcomes[key].copy())
            continue

        if family not in selected_families:
            status_row = {
                "cohort": cohort_key,
                "robustness_family": family,
                "variant_token": variant_token,
                "status": RERUN_STATUS_NOT_RUN,
                "command": pd.NA,
                "return_code": pd.NA,
                "elapsed_seconds": pd.NA,
                "error": "family rerun disabled",
                "source_paths": ";".join(_path_list(paths, project_root)),
            }
            outcomes[key] = status_row
            log_rows.append(status_row)
            continue

        family_commands = rerun_commands.get(family, {})
        command_template = family_commands.get(variant_token)
        if command_template is None:
            status_row = {
                "cohort": cohort_key,
                "robustness_family": family,
                "variant_token": variant_token,
                "status": RERUN_STATUS_NOT_CONFIGURED,
                "command": pd.NA,
                "return_code": pd.NA,
                "elapsed_seconds": pd.NA,
                "error": "missing command",
                "source_paths": ";".join(_path_list(paths, project_root)),
            }
            outcomes[key] = status_row
            log_rows.append(status_row)
            continue

        unsupported_tokens = _unsupported_placeholder_tokens(command_template)
        if unsupported_tokens:
            status_row = {
                "cohort": cohort_key,
                "robustness_family": family,
                "variant_token": variant_token,
                "status": RERUN_STATUS_FORMAT_ERROR,
                "command": command_template,
                "return_code": pd.NA,
                "elapsed_seconds": 0.0,
                "error": f"unsupported placeholder(s): {', '.join(sorted(unsupported_tokens))}",
                "source_paths": ";".join(_path_list(paths, project_root)),
            }
            outcomes[key] = status_row
            log_rows.append(status_row)
            continue

        try:
            command = _format_rerun_command(
                command_template,
                project_root=project_root,
                outputs_dir=outputs_dir,
                tables_dir=tables_dir,
                cohort=cohort,
                variant_token=variant_token,
                robustness_family=family,
            )
        except KeyError as exc:
            status_row = {
                "cohort": cohort_key,
                "robustness_family": family,
                "variant_token": variant_token,
                "status": RERUN_STATUS_FORMAT_ERROR,
                "command": command_template,
                "return_code": pd.NA,
                "elapsed_seconds": 0.0,
                "error": f"missing placeholder: {exc}",
                "source_paths": ";".join(_path_list(paths, project_root)),
            }
            outcomes[key] = status_row
            log_rows.append(status_row)
            continue

        start = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                cwd=project_root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=rerun_timeout_seconds,
                check=False,
            )
            elapsed = time.perf_counter() - start
            status = RERUN_STATUS_SUCCESS if result.returncode == 0 else RERUN_STATUS_FAILURE
            status_row = {
                "cohort": cohort_key,
                "robustness_family": family,
                "variant_token": variant_token,
                "status": status,
                "command": command,
                "return_code": result.returncode,
                "elapsed_seconds": round(elapsed, 4),
                "error": result.stderr.strip(),
                "source_paths": ";".join(_path_list(paths, project_root)),
            }
            outcomes[key] = status_row
            log_rows.append(status_row)
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            status_row = {
                "cohort": cohort_key,
                "robustness_family": family,
                "variant_token": variant_token,
                "status": RERUN_STATUS_TIMEOUT,
                "command": command,
                "return_code": pd.NA,
                "elapsed_seconds": round(elapsed, 4),
                "error": f"timed out after {rerun_timeout_seconds} seconds",
                "source_paths": ";".join(_path_list(paths, project_root)),
            }
            outcomes[key] = status_row
            log_rows.append(status_row)

    return outcomes, pd.DataFrame(log_rows)


def _rerun_outcome_fields(
    outcomes: dict[tuple[str, str, str], dict[str, str | int | float | None]] | None,
    family: str,
    variant: str,
    cohort: str | None,
) -> tuple[Any, Any, Any]:
    if outcomes is None:
        return pd.NA, pd.NA, pd.NA
    entry = outcomes.get((family, variant, _cohort_key(cohort)))
    if entry is None:
        return pd.NA, pd.NA, pd.NA
    return (
        entry.get("status", pd.NA),
        entry.get("command", pd.NA),
        entry.get("return_code", pd.NA),
    )


def _load_robustness_config(root: Path) -> dict[str, Any]:
    path = root / "config/robustness.yml"
    cfg = load_yaml(path) if path.exists() else {}
    if not isinstance(cfg, dict):
        cfg = {}
    rerun_cfg = cfg.get("rerun_commands", {})
    if not isinstance(rerun_cfg, dict):
        rerun_cfg = {}
    timeout_value = cfg.get("rerun_timeout_seconds", DEFAULT_RERUN_TIMEOUT_SECONDS)
    try:
        rerun_timeout_seconds = float(timeout_value)
        if rerun_timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
    except Exception:
        rerun_timeout_seconds = DEFAULT_RERUN_TIMEOUT_SECONDS

    return {
        "sampling_schemes": _as_list(
            cfg.get("sampling_schemes"),
            DEFAULT_ROBUSTNESS_CONFIG["sampling_schemes"],
            field="sampling_schemes",
        ),
        "age_adjustment": _as_list(
            cfg.get("age_adjustment"),
            DEFAULT_ROBUSTNESS_CONFIG["age_adjustment"],
            field="age_adjustment",
        ),
        "residualization_mode": _as_list(
            cfg.get("residualization_mode"),
            DEFAULT_ROBUSTNESS_CONFIG["residualization_mode"],
            field="residualization_mode",
        ),
        "inference": _as_list(
            cfg.get("inference"),
            DEFAULT_ROBUSTNESS_CONFIG["inference"],
            field="inference",
        ),
        "weights": _as_list(
            cfg.get("weights"),
            DEFAULT_ROBUSTNESS_CONFIG["weights"],
            field="weights",
        ),
        "harmonization_methods": _as_list(
            cfg.get("harmonization_methods"),
            DEFAULT_ROBUSTNESS_CONFIG["harmonization_methods"],
            field="harmonization_methods",
        ),
        "harmonization_baseline_method": str(
            cfg.get(
                "harmonization_baseline_method",
                DEFAULT_ROBUSTNESS_CONFIG["harmonization_baseline_method"],
            )
        ).strip()
        or DEFAULT_ROBUSTNESS_CONFIG["harmonization_baseline_method"],
        "rerun_commands": {
            "sampling": _coerce_rerun_commands(rerun_cfg.get("sampling")),
            "age_adjustment": _coerce_rerun_commands(rerun_cfg.get("age_adjustment")),
            "model_form": _coerce_rerun_commands(rerun_cfg.get("model_form")),
            "inference": _coerce_rerun_commands(rerun_cfg.get("inference")),
            "weights": _coerce_rerun_commands(rerun_cfg.get("weights")),
            "harmonization": _coerce_rerun_commands(rerun_cfg.get("harmonization")),
        },
        "rerun_timeout_seconds": rerun_timeout_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline robustness sensitivity tables.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument(
        "--rerun-robustness",
        action="store_true",
        help="Execute configured rerun commands for all missing non-baseline variant families.",
    )
    parser.add_argument(
        "--rerun-sampling-variants",
        action="store_true",
        help="Execute configured rerun commands for missing sampling variants.",
    )
    parser.add_argument(
        "--rerun-age-adjustment-variants",
        action="store_true",
        help="Execute configured rerun commands for missing age-adjustment variants.",
    )
    parser.add_argument(
        "--rerun-model-forms",
        action="store_true",
        help="Execute configured rerun commands for missing model-form variants.",
    )
    parser.add_argument(
        "--rerun-inference-variants",
        action="store_true",
        help="Execute configured rerun commands for missing inference variants.",
    )
    parser.add_argument(
        "--rerun-weight-variants",
        action="store_true",
        help="Execute configured rerun commands for missing weight variants.",
    )
    parser.add_argument(
        "--rerun-harmonization-variants",
        action="store_true",
        help="Execute configured rerun commands for missing harmonization variants.",
    )
    parser.add_argument(
        "--rerun-missing-variants",
        action="store_true",
        help="Legacy alias for --rerun-robustness.",
    )
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    paths_cfg = load_yaml(root / "config/paths.yml")
    outputs_dir = _resolve_path(paths_cfg["outputs_dir"], root)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    robustness_cfg = _load_robustness_config(root)
    placeholder_variants = _placeholder_variant_map(robustness_cfg["rerun_commands"])
    cohorts = _cohorts_from_args(args)
    selected_families = _selected_rerun_families(args)

    sample_counts = _safe_read_csv(tables_dir / "sample_counts.csv")
    g_mean = _safe_read_csv(tables_dir / "g_mean_diff.csv")
    g_vr = _safe_read_csv(tables_dir / "g_variance_ratio.csv")
    rerun_outcomes, rerun_log_df = _run_configured_reruns(
        project_root=root,
        outputs_dir=outputs_dir,
        cohorts=cohorts,
        robustness_cfg=robustness_cfg,
        rerun_commands=robustness_cfg["rerun_commands"],
        selected_families=selected_families,
        rerun_timeout_seconds=robustness_cfg["rerun_timeout_seconds"],
        harmonization_baseline_method=robustness_cfg["harmonization_baseline_method"],
    )
    manifest_rows: list[dict[str, Any]] = []

    sampling_df = _build_sampling_rows(
        tables_dir,
        sample_counts,
        g_mean,
        cohorts,
        robustness_cfg["sampling_schemes"],
        manifest_rows=manifest_rows,
        project_root=root,
        rerun_outcomes=rerun_outcomes,
        placeholder_variants=placeholder_variants,
    )
    sampling_df = _ensure_columns(sampling_df, SAMPLING_REQUIRED)
    sampling_path = tables_dir / "robustness_sampling.csv"
    sampling_df.to_csv(sampling_path, index=False)
    robustness_fig = outputs_dir / "figures" / "robustness_forestplot.png"
    wrote_sampling_plot = _write_sampling_plot(sampling_df, robustness_fig)

    age_df = _build_age_rows(
        outputs_dir,
        cohorts,
        robustness_cfg["age_adjustment"],
        robustness_cfg["residualization_mode"],
        manifest_rows=manifest_rows,
        project_root=root,
        rerun_outcomes=rerun_outcomes,
        placeholder_variants=placeholder_variants,
    )
    age_df = _ensure_columns(age_df, AGE_REQUIRED)
    age_path = tables_dir / "robustness_age_adjustment.csv"
    age_df.to_csv(age_path, index=False)

    model_form_df = _build_model_form_rows(
        outputs_dir,
        cohorts,
        model_forms=list(ROBUSTNESS_MODEL_FORMS),
        manifest_rows=manifest_rows,
        project_root=root,
        rerun_outcomes=rerun_outcomes,
        placeholder_variants=placeholder_variants,
    )
    model_form_df = _ensure_columns(model_form_df, MODEL_FORM_REQUIRED)
    model_form_path = tables_dir / "robustness_model_forms.csv"
    model_form_df.to_csv(model_form_path, index=False)

    inference_df = _build_estimate_rows(
        cohorts=cohorts,
        methods=robustness_cfg["inference"],
        method_column="inference_method",
        g_mean=g_mean,
        g_vr=g_vr,
        tables_dir=tables_dir,
        manifest_rows=manifest_rows,
        project_root=root,
        rerun_outcomes=rerun_outcomes,
        placeholder_variants=placeholder_variants,
    )
    inference_df = _ensure_columns(inference_df, INFERENCE_REQUIRED)
    inference_path = tables_dir / "robustness_inference.csv"
    inference_df.to_csv(inference_path, index=False)

    weights_df = _build_estimate_rows(
        cohorts=cohorts,
        methods=robustness_cfg["weights"],
        method_column="weight_mode",
        g_mean=g_mean,
        g_vr=g_vr,
        tables_dir=tables_dir,
        manifest_rows=manifest_rows,
        project_root=root,
        rerun_outcomes=rerun_outcomes,
        placeholder_variants=placeholder_variants,
    )
    weights_df = _ensure_columns(weights_df, WEIGHTS_REQUIRED)
    weights_path = tables_dir / "robustness_weights.csv"
    weights_df.to_csv(weights_path, index=False)

    harmonization_df = _build_harmonization_rows(
        tables_dir=tables_dir,
        cohorts=cohorts,
        baseline_method=robustness_cfg["harmonization_baseline_method"],
        harmonization_methods=robustness_cfg["harmonization_methods"],
        manifest_rows=manifest_rows,
        project_root=root,
        rerun_outcomes=rerun_outcomes,
        placeholder_variants=placeholder_variants,
    )
    harmonization_df = _ensure_columns(harmonization_df, HARMONIZATION_REQUIRED)
    harmonization_path = tables_dir / "robustness_harmonization.csv"
    harmonization_df.to_csv(harmonization_path, index=False)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df = _ensure_columns(manifest_df, MANIFEST_REQUIRED, preserve_extra=True)
    manifest_path = tables_dir / "robustness_run_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    rerun_log_path = tables_dir / RERUN_LOG_PATH
    rerun_log_path.parent.mkdir(parents=True, exist_ok=True)
    if rerun_log_df.empty:
        rerun_log_df = pd.DataFrame(
            columns=[
                "cohort",
                "robustness_family",
                "variant_token",
                "status",
                "command",
                "return_code",
                "elapsed_seconds",
                "error",
                "source_paths",
            ]
        )
    rerun_log_df.to_csv(rerun_log_path, index=False)

    print(f"[ok] wrote {sampling_path}")
    if wrote_sampling_plot:
        print(f"[ok] wrote {robustness_fig}")
    else:
        print(f"[skip] {robustness_fig} not written (no CI-ready sampling rows)")
    print(f"[ok] wrote {age_path}")
    print(f"[ok] wrote {model_form_path}")
    print(f"[ok] wrote {inference_path}")
    print(f"[ok] wrote {weights_path}")
    print(f"[ok] wrote {harmonization_path}")
    print(f"[ok] wrote {manifest_path}")
    print(f"[ok] wrote {rerun_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
