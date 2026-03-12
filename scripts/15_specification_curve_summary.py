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

from nls_pipeline.io import load_yaml, project_root

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

DEFAULT_ROBUSTNESS_DIMENSIONS = {
    "sampling_schemes": ["sibling_restricted", "full_cohort", "one_pair_per_family"],
    "inference": ["robust_cluster", "family_bootstrap"],
    "weights": ["unweighted", "weighted"],
}

DEFAULT_SIGN_STABILITY_BANDS = {"robust": 0.90, "mixed": 0.60}
DEFAULT_SESOI_BY_ESTIMAND = {"d_g": 0.2, "vr_g": 0.2}
DEFAULT_SESOI_CENTER_BY_ESTIMAND = {"d_g": 0.0, "vr_g": 1.0}
DEFAULT_PRIMARY_ESTIMATE_BY_ESTIMAND = {"d_g": 0.0, "vr_g": 1.0}
DEFAULT_PRIMARY_DEVIATION_BY_ESTIMAND = {"d_g": 0.2, "vr_g": 0.2}
DEFAULT_WEIGHT_CONCORDANCE_DG_ABS_MAX = 0.1
DEFAULT_WEIGHT_CONCORDANCE_VR_LOG_MAX = 0.1
WEIGHT_PAIR_POLICY = "replication_unweighted_primary_weighted_sensitivity"
OUTPUT_COLUMNS = [
    "cohort",
    "estimand",
    "spec_count",
    "expected_spec_count",
    "spec_coverage_share",
    "spec_coverage_label",
    "not_feasible_count",
    "not_feasible_share",
    "median",
    "p2_5",
    "p97_5",
    "sign_stability_count",
    "sign_stability_share",
    "sign_stability_label",
    "magnitude_stability_count",
    "magnitude_stability_share",
    "robust_claim_sign_eligible",
    "robust_claim_primary_eligible",
    "robust_claim_weight_eligible",
    "robust_claim_warning_eligible",
    "robust_claim_eligible",
    "weight_pair_policy",
    "weight_concordance_checked",
    "weight_concordance_reason",
    "weight_sign_match",
    "weight_abs_diff",
    "weight_log_diff",
    "weight_diff_threshold",
    "weight_unweighted_status",
    "weight_unweighted_reason",
    "weight_weighted_status",
    "weight_weighted_reason",
    "primary_estimate",
    "median_deviation_from_primary",
    "primary_deviation_threshold",
]
NOT_FEASIBLE_STATUSES = {"baseline_missing", "missing_source", "not_feasible"}
ESTIMATE_FILES = (
    ("robustness_inference.csv", "inference"),
    ("robustness_weights.csv", "weights"),
    ("robustness_sampling.csv", "sampling"),
)


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _safe_float(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _coerce_numeric_map(raw: Any, fallback: dict[str, float], *, allow_negative: bool = False) -> dict[str, float]:
    values = dict(fallback)
    if not isinstance(raw, dict):
        return values
    for estimand, raw_value in raw.items():
        if isinstance(raw_value, dict):
            parsed = _safe_float(raw_value.get("value"))
        else:
            parsed = _safe_float(raw_value)
        if parsed is None:
            continue
        if not allow_negative and parsed < 0:
            continue
        values[str(estimand)] = parsed
    return values


def _coerce_primary_estimate_map(raw: Any, fallback: dict[str, float]) -> dict[str, float]:
    values = dict(fallback)
    if not isinstance(raw, dict):
        return values
    for estimand, raw_value in raw.items():
        parsed: float | None
        if isinstance(raw_value, dict):
            parsed = _safe_float(raw_value.get("value", raw_value.get("estimate")))
        else:
            parsed = _safe_float(raw_value)
        if parsed is None:
            continue
        values[str(estimand)] = parsed
    return values


def _coerce_reporting_config(root: Path) -> dict[str, Any]:
    cfg = load_yaml(root / "config/models.yml") if (root / "config/models.yml").exists() else {}
    reporting = cfg.get("reporting", {})
    if not isinstance(reporting, dict):
        reporting = {}
    stability_cfg = reporting.get("specification_stability", {})
    if not isinstance(stability_cfg, dict):
        stability_cfg = {}

    threshold_raw = stability_cfg.get("sign_stability_threshold", DEFAULT_SIGN_STABILITY_BANDS)
    if isinstance(threshold_raw, dict):
        robust = _safe_float(threshold_raw.get("robust"))
        mixed = _safe_float(threshold_raw.get("mixed"))
    else:
        robust = _safe_float(threshold_raw)
        mixed = None

    bands = dict(DEFAULT_SIGN_STABILITY_BANDS)
    if robust is None:
        robust = bands["robust"]
    if mixed is None:
        mixed = bands["mixed"]

    if robust is None or mixed is None or not (0 < robust <= 1 and 0 < mixed < robust <= 1):
        bands = dict(DEFAULT_SIGN_STABILITY_BANDS)
    else:
        bands = {"robust": robust, "mixed": mixed}

    se_soi_raw = stability_cfg.get("se_soi", {})
    se_soi_center_raw = stability_cfg.get("se_soi_center", {})
    primary_estimate_raw = stability_cfg.get("primary_estimate", {})
    primary_deviation_raw = stability_cfg.get("primary_deviation", {})
    weight_concordance_raw = stability_cfg.get("weight_concordance", {})

    se_soi_by_estimand = _coerce_numeric_map(se_soi_raw, DEFAULT_SESOI_BY_ESTIMAND)
    se_soi_center_by_estimand = _coerce_numeric_map(
        se_soi_center_raw, DEFAULT_SESOI_CENTER_BY_ESTIMAND, allow_negative=True
    )
    primary_estimate_by_estimand = _coerce_primary_estimate_map(
        primary_estimate_raw,
        se_soi_center_by_estimand if se_soi_center_by_estimand else DEFAULT_PRIMARY_ESTIMATE_BY_ESTIMAND,
    )
    if not primary_estimate_by_estimand:
        primary_estimate_by_estimand = dict(DEFAULT_PRIMARY_ESTIMATE_BY_ESTIMAND)
    primary_deviation_by_estimand = _coerce_numeric_map(
        primary_deviation_raw,
        DEFAULT_PRIMARY_DEVIATION_BY_ESTIMAND,
    )
    if not isinstance(weight_concordance_raw, dict):
        weight_concordance_raw = {}
    dg_abs_max = _safe_float(weight_concordance_raw.get("d_g_abs_diff_max"))
    vr_log_max = _safe_float(weight_concordance_raw.get("vr_g_log_diff_max"))
    if dg_abs_max is None or dg_abs_max < 0:
        dg_abs_max = float(DEFAULT_WEIGHT_CONCORDANCE_DG_ABS_MAX)
    if vr_log_max is None or vr_log_max < 0:
        vr_log_max = float(DEFAULT_WEIGHT_CONCORDANCE_VR_LOG_MAX)

    return {
        "sign_stability_thresholds": bands,
        "se_soi_by_estimand": se_soi_by_estimand,
        "se_soi_center_by_estimand": se_soi_center_by_estimand,
        "primary_estimate_by_estimand": primary_estimate_by_estimand,
        "primary_deviation_by_estimand": primary_deviation_by_estimand,
        "weight_concordance": {
            "d_g_abs_diff_max": float(dg_abs_max),
            "vr_g_log_diff_max": float(vr_log_max),
        },
    }


def _coerce_robustness_dimensions(root: Path) -> dict[str, list[str]]:
    cfg = load_yaml(root / "config/robustness.yml") if (root / "config/robustness.yml").exists() else {}
    if not isinstance(cfg, dict):
        cfg = {}

    out: dict[str, list[str]] = {}
    for key, fallback in DEFAULT_ROBUSTNESS_DIMENSIONS.items():
        raw = cfg.get(key)
        if isinstance(raw, list):
            values = [str(v) for v in raw if str(v).strip()]
            out[key] = values if values else list(fallback)
        else:
            out[key] = list(fallback)
    return out


def _expected_spec_count_by_estimand(dimensions: dict[str, list[str]]) -> dict[str, int]:
    inference_n = len(dimensions.get("inference", []))
    weights_n = len(dimensions.get("weights", []))
    sampling_n = len(dimensions.get("sampling_schemes", []))
    return {
        "d_g": inference_n + weights_n + sampling_n,
        "vr_g": inference_n + weights_n,
    }


def _collect_rows_from_df(
    df: pd.DataFrame,
    *,
    source_label: str,
    cohort_col: str,
    estimate_col: str,
    estimand_col: str | None,
    variant_col: str | None,
) -> list[dict[str, Any]]:
    if df.empty:
        return []

    if "status" not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        status = str(row.get("status", "")).strip().lower()
        if status != "computed":
            continue
        cohort_raw = row.get(cohort_col)
        if cohort_raw is None or pd.isna(cohort_raw):
            continue
        cohort = str(cohort_raw).strip()
        if not cohort:
            continue

        estimate = _safe_float(row.get(estimate_col))
        if estimate is None:
            continue

        if estimand_col is not None:
            estimand = str(row.get(estimand_col, "")).strip()
        else:
            estimand = "d_g"
        if not estimand:
            continue

        rows.append(
            {
                "cohort": cohort,
                "estimand": estimand,
                "estimate": estimate,
                "source": source_label,
                "variant": str(row.get(variant_col, "")).strip() if variant_col else None,
            }
        )
    return rows


def _gather_inputs(tables_dir: Path, cohorts: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        _collect_rows_from_df(
            _safe_read_csv(tables_dir / "robustness_inference.csv"),
            source_label="inference",
            cohort_col="cohort",
            estimate_col="estimate",
            estimand_col="estimate_type",
            variant_col="inference_method",
        )
    )
    rows.extend(
        _collect_rows_from_df(
            _safe_read_csv(tables_dir / "robustness_weights.csv"),
            source_label="weights",
            cohort_col="cohort",
            estimate_col="estimate",
            estimand_col="estimate_type",
            variant_col="weight_mode",
        )
    )
    rows.extend(
        _collect_rows_from_df(
            _safe_read_csv(tables_dir / "robustness_sampling.csv"),
            source_label="sampling",
            cohort_col="cohort",
            estimate_col="d_g",
            estimand_col=None,
            variant_col="sampling_scheme",
        )
    )

    if not cohorts:
        return rows
    return [row for row in rows if row["cohort"] in cohorts]


def _collect_primary_estimates_from_df(
    df: pd.DataFrame,
    *,
    cohort_col: str,
    estimand: str,
    estimate_col: str,
    cohort_filter: set[str],
) -> dict[tuple[str, str], float]:
    if df.empty or cohort_col not in df.columns:
        return {}

    estimate_col_resolved = estimate_col if estimate_col in df.columns else None
    if estimate_col_resolved is None:
        target = estimate_col.strip().lower()
        for col in df.columns:
            if str(col).strip().lower() == target:
                estimate_col_resolved = str(col)
                break
    if estimate_col_resolved is None:
        return {}

    estimates: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        cohort_raw = row.get(cohort_col)
        if cohort_raw is None or pd.isna(cohort_raw):
            continue
        cohort = str(cohort_raw).strip()
        if not cohort:
            continue
        if cohort_filter and cohort not in cohort_filter:
            continue

        status = str(row.get("status", "")).strip().lower()
        if status in NOT_FEASIBLE_STATUSES:
            continue

        estimate = _safe_float(row.get(estimate_col_resolved))
        if estimate is None:
            continue
        estimates[(cohort, estimand)] = estimate
    return estimates


def _gather_primary_estimates(tables_dir: Path, cohort_filter: set[str]) -> dict[tuple[str, str], float]:
    primary: dict[tuple[str, str], float] = {}
    primary.update(
        _collect_primary_estimates_from_df(
            _safe_read_csv(tables_dir / "g_mean_diff.csv"),
            cohort_col="cohort",
            estimand="d_g",
            estimate_col="d_g",
            cohort_filter=cohort_filter,
        )
    )
    primary.update(
        _collect_primary_estimates_from_df(
            _safe_read_csv(tables_dir / "g_variance_ratio.csv"),
            cohort_col="cohort",
            estimand="vr_g",
            estimate_col="vr_g",
            cohort_filter=cohort_filter,
        )
    )
    return primary


def _gather_warning_fail_cohorts(tables_dir: Path, cohort_filter: set[str]) -> set[str]:
    status = _safe_read_csv(tables_dir / "sem_run_status.csv")
    if status.empty or "cohort" not in status.columns:
        return set()

    out: set[str] = set()
    for _, row in status.iterrows():
        cohort_raw = row.get("cohort")
        if cohort_raw is None or pd.isna(cohort_raw):
            continue
        cohort = str(cohort_raw).strip()
        if not cohort:
            continue
        if cohort_filter and cohort not in cohort_filter:
            continue

        warning_status = str(row.get("warning_policy_status", "")).strip().lower()
        run_status = str(row.get("status", "")).strip().lower()
        if warning_status == "fail" or run_status in {"r-failed", "failed", "error", "no-output", "no_output"}:
            out.add(cohort)
    return out


def _sign_with_center(value: float, center: float) -> int:
    delta = float(value) - float(center)
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def _gather_weight_concordance(
    tables_dir: Path,
    cohorts: set[str],
    reporting_cfg: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    weights = _safe_read_csv(tables_dir / "robustness_weights.csv")
    if weights.empty:
        return out
    required = {"cohort", "weight_mode", "estimate_type", "status", "estimate"}
    if not required.issubset(set(weights.columns)):
        return out

    dg_abs_max = float(reporting_cfg["weight_concordance"]["d_g_abs_diff_max"])
    vr_log_max = float(reporting_cfg["weight_concordance"]["vr_g_log_diff_max"])
    centers = reporting_cfg["se_soi_center_by_estimand"]

    filtered = weights.copy()
    filtered["cohort"] = filtered["cohort"].astype(str).str.strip()
    filtered["weight_mode"] = filtered["weight_mode"].astype(str).str.strip().str.lower()
    filtered["estimate_type"] = filtered["estimate_type"].astype(str).str.strip()
    filtered["status"] = filtered["status"].astype(str).str.strip().str.lower()
    filtered["reason"] = filtered.get("reason", pd.Series([pd.NA] * len(filtered))).astype(str).str.strip()
    filtered["estimate_num"] = pd.to_numeric(filtered["estimate"], errors="coerce")
    if cohorts:
        filtered = filtered.loc[filtered["cohort"].isin(cohorts)].copy()
    if filtered.empty:
        return out

    for (cohort, estimand), grp in filtered.groupby(["cohort", "estimate_type"], sort=False):
        key = (str(cohort), str(estimand))
        row = {
            "weight_concordance_checked": False,
            "weight_concordance_reason": "nonconfirmatory_missing_weight_pair",
            "weight_sign_match": pd.NA,
            "weight_abs_diff": pd.NA,
            "weight_log_diff": pd.NA,
            "weight_diff_threshold": pd.NA,
            "robust_claim_weight_eligible": False,
            "weight_pair_policy": WEIGHT_PAIR_POLICY,
            "weight_unweighted_status": pd.NA,
            "weight_unweighted_reason": pd.NA,
            "weight_weighted_status": pd.NA,
            "weight_weighted_reason": pd.NA,
        }

        unweighted_rows = grp.loc[grp["weight_mode"].eq("unweighted")].copy()
        weighted_rows = grp.loc[grp["weight_mode"].eq("weighted")].copy()
        if unweighted_rows.empty or weighted_rows.empty:
            out[key] = row
            continue
        unweighted_row = unweighted_rows.iloc[0]
        weighted_row = weighted_rows.iloc[0]
        unweighted_status = str(unweighted_row.get("status", "")).strip().lower()
        weighted_status = str(weighted_row.get("status", "")).strip().lower()
        unweighted_reason = str(unweighted_row.get("reason", "")).strip()
        weighted_reason = str(weighted_row.get("reason", "")).strip()
        row["weight_unweighted_status"] = unweighted_status if unweighted_status else pd.NA
        row["weight_unweighted_reason"] = unweighted_reason if unweighted_reason else pd.NA
        row["weight_weighted_status"] = weighted_status if weighted_status else pd.NA
        row["weight_weighted_reason"] = weighted_reason if weighted_reason else pd.NA

        if unweighted_status != "computed":
            if unweighted_status in {"baseline_missing", "missing_source", "not_feasible", "not_run_placeholder"}:
                row["weight_concordance_reason"] = "nonconfirmatory_missing_unweighted_baseline"
            elif unweighted_status:
                row["weight_concordance_reason"] = f"nonconfirmatory_unweighted_status_{unweighted_status}"
            else:
                row["weight_concordance_reason"] = "nonconfirmatory_missing_unweighted_baseline"
            out[key] = row
            continue
        if weighted_status != "computed":
            if weighted_status == "not_feasible" and weighted_reason:
                row["weight_concordance_reason"] = f"nonconfirmatory_weighted_not_feasible:{weighted_reason}"
            elif weighted_status:
                row["weight_concordance_reason"] = f"nonconfirmatory_weighted_{weighted_status}"
            else:
                row["weight_concordance_reason"] = "nonconfirmatory_missing_weighted_estimate"
            out[key] = row
            continue

        u_val = _safe_float(unweighted_row.get("estimate_num"))
        w_val = _safe_float(weighted_row.get("estimate_num"))
        if u_val is None or w_val is None:
            row["weight_concordance_reason"] = "nonconfirmatory_invalid_weight_pair_estimate"
            out[key] = row
            continue
        u = float(u_val)
        w = float(w_val)
        center = float(centers.get(str(estimand), 0.0))
        sign_match = _sign_with_center(u, center) == _sign_with_center(w, center)
        abs_diff = abs(w - u)

        log_diff = pd.NA
        if str(estimand) == "vr_g":
            if u <= 0 or w <= 0:
                row["weight_concordance_checked"] = True
                row["weight_concordance_reason"] = "nonpositive_vr_weight_pair"
                row["weight_sign_match"] = bool(sign_match)
                row["weight_abs_diff"] = abs_diff
                row["weight_diff_threshold"] = vr_log_max
                row["robust_claim_weight_eligible"] = False
                out[key] = row
                continue
            log_diff = abs(math.log(w / u))
            magnitude_ok = log_diff <= vr_log_max
            threshold = vr_log_max
        else:
            magnitude_ok = abs_diff <= dg_abs_max
            threshold = dg_abs_max

        eligible = bool(sign_match and magnitude_ok)
        row.update(
            {
                "weight_concordance_checked": True,
                "weight_concordance_reason": "ok" if eligible else "discordant_weight_estimates",
                "weight_sign_match": bool(sign_match),
                "weight_abs_diff": abs_diff,
                "weight_log_diff": log_diff,
                "weight_diff_threshold": threshold,
                "robust_claim_weight_eligible": eligible,
            }
        )
        out[key] = row
    return out


def _collect_not_feasible_from_df(
    df: pd.DataFrame,
    *,
    cohort_col: str,
    estimand_col: str | None,
) -> list[tuple[str, str]]:
    if df.empty or "status" not in df.columns:
        return []
    out: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        status = str(row.get("status", "")).strip().lower()
        if status not in NOT_FEASIBLE_STATUSES:
            continue
        cohort_raw = row.get(cohort_col)
        if cohort_raw is None or pd.isna(cohort_raw):
            continue
        cohort = str(cohort_raw).strip()
        if not cohort:
            continue
        estimand = str(row.get(estimand_col, "")).strip() if estimand_col else "d_g"
        if not estimand:
            continue
        out.append((cohort, estimand))
    return out


def _gather_not_feasible_counts(tables_dir: Path, cohorts: set[str]) -> dict[tuple[str, str], int]:
    pairs: list[tuple[str, str]] = []
    pairs.extend(
        _collect_not_feasible_from_df(
            _safe_read_csv(tables_dir / "robustness_inference.csv"),
            cohort_col="cohort",
            estimand_col="estimate_type",
        )
    )
    pairs.extend(
        _collect_not_feasible_from_df(
            _safe_read_csv(tables_dir / "robustness_weights.csv"),
            cohort_col="cohort",
            estimand_col="estimate_type",
        )
    )
    pairs.extend(
        _collect_not_feasible_from_df(
            _safe_read_csv(tables_dir / "robustness_sampling.csv"),
            cohort_col="cohort",
            estimand_col=None,
        )
    )

    counts: dict[tuple[str, str], int] = {}
    for cohort, estimand in pairs:
        if cohorts and cohort not in cohorts:
            continue
        key = (cohort, estimand)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _log_vr_tolerance(center: float, se_soi: float) -> float:
    if center <= 0 or se_soi <= 0:
        return 0.0
    return math.log((center + se_soi) / center)


def _stability_label(
    sign: float,
    sign_share: float | None,
    *,
    robust_threshold: float,
    mixed_threshold: float,
) -> str:
    if sign_share is None:
        return "insufficient"
    if sign_share >= robust_threshold:
        return f"robust_{'positive' if sign > 0 else 'negative' if sign < 0 else 'null'}"
    if sign_share >= mixed_threshold:
        return f"mixed_{'positive' if sign > 0 else 'negative' if sign < 0 else 'null'}"
    return "unstable"


def _summarize_by_spec(
    records: list[dict[str, Any]],
    reporting_cfg: dict[str, Any],
    cohorts: set[str],
    expected_spec_count_by_estimand: dict[str, int],
    not_feasible_counts: dict[tuple[str, str], int],
    primary_estimates: dict[tuple[str, str], float],
    primary_estimate_by_estimand: dict[str, float],
    primary_deviation_by_estimand: dict[str, float],
    warning_fail_cohorts: set[str],
    weight_concordance: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = df[df["cohort"].isin(cohorts)]
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    summary_rows: list[dict[str, Any]] = []
    thresholds = reporting_cfg["sign_stability_thresholds"]
    se_soi_by_estimand = reporting_cfg["se_soi_by_estimand"]
    se_soi_center_by_estimand = reporting_cfg["se_soi_center_by_estimand"]
    robust_sign_threshold = float(thresholds["robust"])

    for (cohort, estimand), group in df.groupby(["cohort", "estimand"], sort=False):
        raw_values = pd.to_numeric(group["estimate"], errors="coerce").dropna().astype(float)
        expected_spec_count = int(expected_spec_count_by_estimand.get(estimand, 0))
        not_feasible_count = int(not_feasible_counts.get((cohort, estimand), 0))
        not_feasible_share = (
            float(not_feasible_count / expected_spec_count) if expected_spec_count > 0 else pd.NA
        )
        if raw_values.empty:
            summary_rows.append(
                {
                    "cohort": cohort,
                    "estimand": estimand,
                    "spec_count": 0,
                    "expected_spec_count": expected_spec_count,
                    "spec_coverage_share": pd.NA,
                    "spec_coverage_label": "none",
                    "not_feasible_count": not_feasible_count,
                    "not_feasible_share": not_feasible_share,
                    "median": pd.NA,
                    "p2_5": pd.NA,
                    "p97_5": pd.NA,
                    "sign_stability_count": pd.NA,
                    "sign_stability_share": pd.NA,
                    "sign_stability_label": "insufficient",
                    "magnitude_stability_count": pd.NA,
                    "magnitude_stability_share": pd.NA,
                    "robust_claim_sign_eligible": pd.NA,
                    "robust_claim_primary_eligible": pd.NA,
                    "robust_claim_weight_eligible": pd.NA,
                    "robust_claim_warning_eligible": pd.NA,
                    "robust_claim_eligible": pd.NA,
                    "weight_pair_policy": WEIGHT_PAIR_POLICY,
                    "weight_concordance_checked": pd.NA,
                    "weight_concordance_reason": pd.NA,
                    "weight_sign_match": pd.NA,
                    "weight_abs_diff": pd.NA,
                    "weight_log_diff": pd.NA,
                    "weight_diff_threshold": pd.NA,
                    "weight_unweighted_status": pd.NA,
                    "weight_unweighted_reason": pd.NA,
                    "weight_weighted_status": pd.NA,
                    "weight_weighted_reason": pd.NA,
                    "primary_estimate": pd.NA,
                    "median_deviation_from_primary": pd.NA,
                    "primary_deviation_threshold": pd.NA,
                }
            )
            continue

        center = se_soi_center_by_estimand.get(estimand, 0.0)
        se_soi = se_soi_by_estimand.get(estimand, 0.0)

        if estimand == "vr_g":
            if center <= 0:
                sign_values = pd.Series(pd.NA, index=raw_values.index)
                magnitude_mask = pd.Series(False, index=raw_values.index)
            else:
                sign_values = raw_values.where(raw_values > 0).map(
                    lambda x: math.log(x / center) if pd.notna(x) else pd.NA
                )
                mag_threshold = _log_vr_tolerance(float(center), float(se_soi or 0.0))
                magnitude_mask = sign_values.abs() <= mag_threshold
        else:
            sign_values = raw_values - center
            magnitude_mask = (sign_values.abs() <= float(se_soi))

        magnitude_mask = magnitude_mask.reindex(raw_values.index, fill_value=False)
        valid_mask = sign_values.notna()
        if valid_mask.sum() == 0:
            summary_rows.append(
                {
                    "cohort": cohort,
                    "estimand": estimand,
                    "spec_count": 0,
                    "expected_spec_count": expected_spec_count,
                    "spec_coverage_share": pd.NA,
                    "spec_coverage_label": "none",
                    "not_feasible_count": not_feasible_count,
                    "not_feasible_share": not_feasible_share,
                    "median": pd.NA,
                    "p2_5": pd.NA,
                    "p97_5": pd.NA,
                    "sign_stability_count": pd.NA,
                    "sign_stability_share": pd.NA,
                    "sign_stability_label": "insufficient",
                    "magnitude_stability_count": pd.NA,
                    "magnitude_stability_share": pd.NA,
                    "robust_claim_sign_eligible": pd.NA,
                    "robust_claim_primary_eligible": pd.NA,
                    "robust_claim_weight_eligible": pd.NA,
                    "robust_claim_warning_eligible": pd.NA,
                    "robust_claim_eligible": pd.NA,
                    "weight_pair_policy": WEIGHT_PAIR_POLICY,
                    "weight_concordance_checked": pd.NA,
                    "weight_concordance_reason": pd.NA,
                    "weight_sign_match": pd.NA,
                    "weight_abs_diff": pd.NA,
                    "weight_log_diff": pd.NA,
                    "weight_diff_threshold": pd.NA,
                    "weight_unweighted_status": pd.NA,
                    "weight_unweighted_reason": pd.NA,
                    "weight_weighted_status": pd.NA,
                    "weight_weighted_reason": pd.NA,
                    "primary_estimate": pd.NA,
                    "median_deviation_from_primary": pd.NA,
                    "primary_deviation_threshold": pd.NA,
                }
            )
            continue

        values = raw_values[valid_mask]
        sign_values = sign_values[valid_mask]
        sign_magnitude = magnitude_mask[valid_mask]

        spec_count = int(len(values))
        if expected_spec_count > 0:
            coverage_share = float(spec_count / expected_spec_count)
            if coverage_share >= 1.0:
                coverage_label = "complete"
            elif coverage_share > 0.0:
                coverage_label = "partial"
            else:
                coverage_label = "none"
        else:
            coverage_share = pd.NA
            coverage_label = "not_defined"
        sign_median = float(sign_values.quantile(0.5))
        if sign_median > 0:
            sign_count = int((sign_values > 0).sum())
        elif sign_median < 0:
            sign_count = int((sign_values < 0).sum())
        else:
            sign_count = int((sign_values == 0).sum())
        sign_share = float(sign_count / spec_count) if spec_count else pd.NA

        se_soi_f = float(_safe_float(se_soi) or 0.0)
        if estimand == "vr_g":
            magnitude_count = int(sign_magnitude.sum())
            magnitude_share = float(magnitude_count / spec_count)
        else:
            magnitude_count = int(((values - center).abs() <= se_soi_f).sum())
            magnitude_share = float((values - center).abs().le(se_soi_f).mean())

        primary_has_observed = (cohort, estimand) in primary_estimates
        primary_estimate = primary_estimates.get((cohort, estimand))

        median = float(values.quantile(0.5))
        sign_stability_eligible = bool(sign_share >= robust_sign_threshold) if sign_share is not pd.NA else pd.NA

        if (not primary_has_observed) or primary_estimate is None or primary_estimate != primary_estimate:
            median_deviation_from_primary = pd.NA
            primary_deviation_threshold = pd.NA
            primary_stability_eligible = False
            primary_estimate = pd.NA
        elif estimand == "vr_g":
            if median <= 0 or primary_estimate <= 0:
                median_deviation_from_primary = pd.NA
                primary_deviation_threshold = pd.NA
                primary_stability_eligible = False
            else:
                primary_deviation_threshold = primary_deviation_by_estimand.get(estimand, 0.0)
                median_deviation_from_primary = abs(math.log(median / float(primary_estimate)))
                primary_stability_eligible = (
                    median_deviation_from_primary <= float(primary_deviation_threshold)
                )
        else:
            primary_deviation_threshold = primary_deviation_by_estimand.get(estimand, 0.0)
            median_deviation_from_primary = abs(median - float(primary_estimate))
            primary_stability_eligible = (
                median_deviation_from_primary <= float(primary_deviation_threshold)
            )

        warning_eligible = cohort not in warning_fail_cohorts
        weight_info = weight_concordance.get(
            (cohort, estimand),
            {
                "weight_concordance_checked": False,
                "weight_concordance_reason": "nonconfirmatory_missing_weight_pair",
                "weight_sign_match": pd.NA,
                "weight_abs_diff": pd.NA,
                "weight_log_diff": pd.NA,
                "weight_diff_threshold": pd.NA,
                "robust_claim_weight_eligible": False,
                "weight_pair_policy": WEIGHT_PAIR_POLICY,
                "weight_unweighted_status": pd.NA,
                "weight_unweighted_reason": pd.NA,
                "weight_weighted_status": pd.NA,
                "weight_weighted_reason": pd.NA,
            },
        )
        weight_eligible = bool(weight_info.get("robust_claim_weight_eligible", False))

        if (
            sign_stability_eligible is pd.NA
            or primary_stability_eligible is pd.NA
        ):
            robust_claim_eligible = pd.NA
        else:
            robust_claim_eligible = bool(
                sign_stability_eligible
                and primary_stability_eligible
                and warning_eligible
            )

        summary_rows.append(
            {
                "cohort": cohort,
                "estimand": estimand,
                "spec_count": spec_count,
                "expected_spec_count": expected_spec_count,
                "spec_coverage_share": coverage_share,
                "spec_coverage_label": coverage_label,
                "not_feasible_count": not_feasible_count,
                "not_feasible_share": not_feasible_share,
                "median": float(values.quantile(0.5)),
                "p2_5": float(values.quantile(0.025)),
                "p97_5": float(values.quantile(0.975)),
                "sign_stability_count": sign_count,
                "sign_stability_share": sign_share,
                "sign_stability_label": _stability_label(
                    sign_median,
                    sign_share,
                    robust_threshold=float(thresholds["robust"]),
                    mixed_threshold=float(thresholds["mixed"]),
                ),
                "magnitude_stability_count": magnitude_count,
                "magnitude_stability_share": magnitude_share,
                "robust_claim_sign_eligible": sign_stability_eligible,
                "robust_claim_primary_eligible": primary_stability_eligible,
                "robust_claim_weight_eligible": weight_eligible,
                "robust_claim_warning_eligible": warning_eligible,
                "robust_claim_eligible": robust_claim_eligible,
                "weight_pair_policy": weight_info.get("weight_pair_policy", WEIGHT_PAIR_POLICY),
                "weight_concordance_checked": weight_info.get("weight_concordance_checked", pd.NA),
                "weight_concordance_reason": weight_info.get("weight_concordance_reason", pd.NA),
                "weight_sign_match": weight_info.get("weight_sign_match", pd.NA),
                "weight_abs_diff": weight_info.get("weight_abs_diff", pd.NA),
                "weight_log_diff": weight_info.get("weight_log_diff", pd.NA),
                "weight_diff_threshold": weight_info.get("weight_diff_threshold", pd.NA),
                "weight_unweighted_status": weight_info.get("weight_unweighted_status", pd.NA),
                "weight_unweighted_reason": weight_info.get("weight_unweighted_reason", pd.NA),
                "weight_weighted_status": weight_info.get("weight_weighted_status", pd.NA),
                "weight_weighted_reason": weight_info.get("weight_weighted_reason", pd.NA),
                "primary_estimate": primary_estimate,
                "median_deviation_from_primary": median_deviation_from_primary,
                "primary_deviation_threshold": (
                    float(primary_deviation_threshold) if primary_deviation_threshold is not pd.NA else pd.NA
                ),
            }
        )
    return pd.DataFrame(summary_rows, columns=OUTPUT_COLUMNS).sort_values(["cohort", "estimand"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize specification robustness for estimand sign and magnitude stability."
    )
    parser.add_argument(
        "--cohort",
        action="append",
        choices=sorted(COHORT_CONFIGS),
        help="Cohort(s) to summarize.",
    )
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    paths_cfg = load_yaml(root / "config/paths.yml")
    outputs_dir = _resolve_path(paths_cfg["outputs_dir"], root)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cohorts = set(_cohorts_from_args(args))
    reporting_cfg = _coerce_reporting_config(root)
    expected_spec_count_by_estimand = _expected_spec_count_by_estimand(_coerce_robustness_dimensions(root))
    records = _gather_inputs(tables_dir, cohorts)
    not_feasible_counts = _gather_not_feasible_counts(tables_dir, cohorts)
    primary_estimates = _gather_primary_estimates(tables_dir, cohorts)
    warning_fail_cohorts = _gather_warning_fail_cohorts(tables_dir, cohorts)
    weight_concordance = _gather_weight_concordance(tables_dir, cohorts, reporting_cfg)
    summary = _summarize_by_spec(
        records,
        reporting_cfg,
        cohorts,
        expected_spec_count_by_estimand,
        not_feasible_counts,
        primary_estimates,
        reporting_cfg["primary_estimate_by_estimand"],
        reporting_cfg["primary_deviation_by_estimand"],
        warning_fail_cohorts,
        weight_concordance,
    )
    summary_path = tables_dir / "specification_stability_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[ok] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
