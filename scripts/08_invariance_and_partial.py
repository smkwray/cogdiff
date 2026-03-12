#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root, relative_path, resolve_token_path
from nls_pipeline.sem import rscript_path, run_sem_r_script

COHORTS = ["nlsy79", "nlsy97", "cnlsy"]
DEFAULT_INVARIANCE_THRESHOLDS: dict[str, dict[str, float]] = {
    "metric": {"delta_cfi_min": -0.01, "delta_rmsea_max": 0.015, "delta_srmr_max": 0.03},
    "scalar": {"delta_cfi_min": -0.01, "delta_rmsea_max": 0.015, "delta_srmr_max": 0.015},
}
DEFAULT_INVARIANCE_TOLERANCE: dict[str, dict[str, float]] = {
    "metric": {"delta_cfi_excess_max": 0.002, "delta_rmsea_excess_max": 0.005, "delta_srmr_excess_max": 0.005},
    "scalar": {"delta_cfi_excess_max": 0.002, "delta_rmsea_excess_max": 0.005, "delta_srmr_excess_max": 0.005},
}
DEFAULT_PARTIAL_POLICY = {
    "max_free_per_factor": 2,
    "max_free_share_per_factor": 0.2,
    "min_invariant_share_per_factor": 0.5,
    "min_free_allowance_per_factor": 2,
}
DEFAULT_PARTIAL_REPLICABILITY_POLICY = {
    "enabled": True,
    "mode": "cross_cohort_overlap",
    "apply_when_partial_refit": True,
    "min_overlap_share": 0.7,
    "comparison_cohorts": ["nlsy79", "nlsy97"],
    "bootstrap_replicates": 200,
    "bootstrap_seed": 1729,
    "min_bootstrap_indicator_share": 0.7,
    "min_bootstrap_success_reps": 100,
    "bootstrap_min_group_n": 50,
    "bootstrap_cluster_cols": {
        "nlsy79": [],
        "nlsy97": ["R9708601", "R9708602"],
        "cnlsy": [],
    },
}


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return COHORTS
    return args.cohort


def _expected_steps(models_cfg: dict[str, Any]) -> list[str]:
    invariance_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    return [str(x) for x in invariance_cfg.get("steps", ["configural", "metric", "scalar", "strict"])]


def _as_float(value: Any, default: float) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return float(default)
    return float(parsed)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    token = str(value).strip().lower()
    if token in {"true", "1", "yes", "y"}:
        return True
    if token in {"false", "0", "no", "n"}:
        return False
    return default


def _invariance_thresholds(models_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    invariance_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    gate_cfg = invariance_cfg.get("gatekeeping", {})
    if not isinstance(gate_cfg, dict):
        gate_cfg = {}
    thresholds_cfg = gate_cfg.get("thresholds", {})
    if not isinstance(thresholds_cfg, dict):
        thresholds_cfg = {}

    out: dict[str, dict[str, float]] = {}
    for step, defaults in DEFAULT_INVARIANCE_THRESHOLDS.items():
        raw = thresholds_cfg.get(step, {})
        if not isinstance(raw, dict):
            raw = {}
        out[step] = {
            "delta_cfi_min": _as_float(raw.get("delta_cfi_min"), defaults["delta_cfi_min"]),
            "delta_rmsea_max": _as_float(raw.get("delta_rmsea_max"), defaults["delta_rmsea_max"]),
            "delta_srmr_max": _as_float(raw.get("delta_srmr_max"), defaults["delta_srmr_max"]),
        }
    return out


def _gatekeeping_enabled(models_cfg: dict[str, Any]) -> bool:
    invariance_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    gate_cfg = invariance_cfg.get("gatekeeping", {})
    if not isinstance(gate_cfg, dict):
        return False
    raw = gate_cfg.get("enabled", False)
    if isinstance(raw, bool):
        return raw
    token = str(raw).strip().lower()
    return token in {"true", "1", "yes", "y"}


def _invariance_tolerance(models_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    invariance_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    gate_cfg = invariance_cfg.get("gatekeeping", {})
    if not isinstance(gate_cfg, dict):
        gate_cfg = {}
    tolerance_cfg = gate_cfg.get("near_pass_tolerance", {})
    if not isinstance(tolerance_cfg, dict):
        tolerance_cfg = {}

    out: dict[str, dict[str, float]] = {}
    for step, defaults in DEFAULT_INVARIANCE_TOLERANCE.items():
        raw = tolerance_cfg.get(step, {})
        if not isinstance(raw, dict):
            raw = {}
        out[step] = {
            "delta_cfi_excess_max": max(0.0, _as_float(raw.get("delta_cfi_excess_max"), defaults["delta_cfi_excess_max"])),
            "delta_rmsea_excess_max": max(
                0.0, _as_float(raw.get("delta_rmsea_excess_max"), defaults["delta_rmsea_excess_max"])
            ),
            "delta_srmr_excess_max": max(0.0, _as_float(raw.get("delta_srmr_excess_max"), defaults["delta_srmr_excess_max"])),
        }
    return out


def _partial_policy(models_cfg: dict[str, Any]) -> dict[str, float]:
    invariance_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    gate_cfg = invariance_cfg.get("gatekeeping", {})
    if not isinstance(gate_cfg, dict):
        gate_cfg = {}
    policy_cfg = gate_cfg.get("partial_policy", {})
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}

    return {
        "max_free_per_factor": max(1.0, _as_float(policy_cfg.get("max_free_per_factor"), 2.0)),
        "max_free_share_per_factor": max(
            0.0, min(1.0, _as_float(policy_cfg.get("max_free_share_per_factor"), 0.2))
        ),
        "min_invariant_share_per_factor": max(
            0.0, min(1.0, _as_float(policy_cfg.get("min_invariant_share_per_factor"), 0.5))
        ),
        "min_free_allowance_per_factor": max(0.0, _as_float(policy_cfg.get("min_free_allowance_per_factor"), 2.0)),
    }


def _partial_replicability_policy(models_cfg: dict[str, Any]) -> dict[str, Any]:
    invariance_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    gate_cfg = invariance_cfg.get("gatekeeping", {})
    if not isinstance(gate_cfg, dict):
        gate_cfg = {}
    policy_cfg = gate_cfg.get("partial_replicability", {})
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}

    raw_cohorts = policy_cfg.get("comparison_cohorts", DEFAULT_PARTIAL_REPLICABILITY_POLICY["comparison_cohorts"])
    if isinstance(raw_cohorts, (list, tuple)):
        comparison_cohorts = [str(x) for x in raw_cohorts if str(x).strip()]
    else:
        comparison_cohorts = list(DEFAULT_PARTIAL_REPLICABILITY_POLICY["comparison_cohorts"])
    if not comparison_cohorts:
        comparison_cohorts = list(DEFAULT_PARTIAL_REPLICABILITY_POLICY["comparison_cohorts"])

    mode = str(policy_cfg.get("mode", DEFAULT_PARTIAL_REPLICABILITY_POLICY["mode"])).strip() or str(
        DEFAULT_PARTIAL_REPLICABILITY_POLICY["mode"]
    )
    allowed_modes = {"cross_cohort_overlap", "bootstrap_stability", "combined"}
    if mode not in allowed_modes:
        mode = str(DEFAULT_PARTIAL_REPLICABILITY_POLICY["mode"])

    raw_cluster_cols = policy_cfg.get(
        "bootstrap_cluster_cols",
        DEFAULT_PARTIAL_REPLICABILITY_POLICY.get("bootstrap_cluster_cols", {}),
    )
    cluster_cols: dict[str, list[str]] = {}
    if isinstance(raw_cluster_cols, dict):
        for cohort_key, columns in raw_cluster_cols.items():
            if isinstance(columns, (list, tuple)):
                cluster_cols[str(cohort_key)] = [str(x) for x in columns if str(x).strip()]
            else:
                cluster_cols[str(cohort_key)] = []
    for cohort_name in COHORTS:
        cluster_cols.setdefault(cohort_name, [])

    bootstrap_reps_default = int(DEFAULT_PARTIAL_REPLICABILITY_POLICY["bootstrap_replicates"])
    bootstrap_replicates = int(max(20, round(_as_float(policy_cfg.get("bootstrap_replicates"), float(bootstrap_reps_default)))))
    min_bootstrap_success_default = int(DEFAULT_PARTIAL_REPLICABILITY_POLICY["min_bootstrap_success_reps"])
    min_bootstrap_success_reps = int(
        max(
            1,
            min(
                bootstrap_replicates,
                round(_as_float(policy_cfg.get("min_bootstrap_success_reps"), float(min_bootstrap_success_default))),
            ),
        )
    )

    return {
        "enabled": _as_bool(policy_cfg.get("enabled"), bool(DEFAULT_PARTIAL_REPLICABILITY_POLICY["enabled"])),
        "mode": mode,
        "apply_when_partial_refit": _as_bool(
            policy_cfg.get("apply_when_partial_refit"),
            bool(DEFAULT_PARTIAL_REPLICABILITY_POLICY["apply_when_partial_refit"]),
        ),
        "min_overlap_share": max(
            0.0,
            min(1.0, _as_float(policy_cfg.get("min_overlap_share"), float(DEFAULT_PARTIAL_REPLICABILITY_POLICY["min_overlap_share"]))),
        ),
        "comparison_cohorts": comparison_cohorts,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": int(
            round(_as_float(policy_cfg.get("bootstrap_seed"), float(DEFAULT_PARTIAL_REPLICABILITY_POLICY["bootstrap_seed"])))
        ),
        "min_bootstrap_indicator_share": max(
            0.0,
            min(
                1.0,
                _as_float(
                    policy_cfg.get(
                        "min_bootstrap_indicator_share",
                        float(DEFAULT_PARTIAL_REPLICABILITY_POLICY["min_bootstrap_indicator_share"]),
                    ),
                    float(DEFAULT_PARTIAL_REPLICABILITY_POLICY["min_bootstrap_indicator_share"]),
                ),
            ),
        ),
        "min_bootstrap_success_reps": min_bootstrap_success_reps,
        "bootstrap_min_group_n": int(
            max(2, round(_as_float(policy_cfg.get("bootstrap_min_group_n"), float(DEFAULT_PARTIAL_REPLICABILITY_POLICY["bootstrap_min_group_n"]))))
        ),
        "bootstrap_cluster_cols": cluster_cols,
    }


def _apply_partial_replicability_overrides(
    policy: dict[str, Any],
    *,
    enabled_override: bool | None = None,
    apply_when_partial_refit_override: bool | None = None,
    mode_override: str | None = None,
    min_overlap_share_override: float | None = None,
    min_bootstrap_indicator_share_override: float | None = None,
    min_bootstrap_success_reps_override: int | None = None,
    bootstrap_replicates_override: int | None = None,
) -> dict[str, Any]:
    out = dict(policy)
    if enabled_override is not None:
        out["enabled"] = bool(enabled_override)
    if apply_when_partial_refit_override is not None:
        out["apply_when_partial_refit"] = bool(apply_when_partial_refit_override)
    if mode_override is not None:
        out["mode"] = str(mode_override)
    if min_overlap_share_override is not None:
        out["min_overlap_share"] = max(0.0, min(1.0, float(min_overlap_share_override)))
    if min_bootstrap_indicator_share_override is not None:
        out["min_bootstrap_indicator_share"] = max(0.0, min(1.0, float(min_bootstrap_indicator_share_override)))
    if bootstrap_replicates_override is not None:
        out["bootstrap_replicates"] = max(20, int(bootstrap_replicates_override))
    if min_bootstrap_success_reps_override is not None:
        out["min_bootstrap_success_reps"] = max(1, int(min_bootstrap_success_reps_override))
    # Keep bootstrap success threshold bounded to the (possibly overridden) replicate count.
    out["min_bootstrap_success_reps"] = min(
        int(out.get("min_bootstrap_success_reps", 1)),
        int(out.get("bootstrap_replicates", 20)),
    )
    return out


def _partial_intercepts(models_cfg: dict[str, Any], cohort: str) -> list[str]:
    invariance_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    partial_cfg = invariance_cfg.get("partial_intercepts", {})
    if not isinstance(partial_cfg, dict):
        return []
    return [str(x) for x in partial_cfg.get(cohort, [])]


def _load_fit_indices(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_score_tests(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _ensure_step_rows(df: pd.DataFrame, steps: list[str], cohort: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"cohort": cohort, "model_step": steps})
    work = df.copy()
    if "model_step" not in work.columns:
        raise ValueError(f"fit_indices.csv missing required column 'model_step': {work}")
    if "cohort" not in work.columns:
        work["cohort"] = cohort
    merged = pd.DataFrame({"cohort": cohort, "model_step": steps}).merge(work, on=["cohort", "model_step"], how="left")
    return merged


def _add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in ("cfi", "rmsea", "srmr"):
        if metric in out.columns:
            out[f"delta_{metric}"] = out[metric].diff()
        else:
            out[f"delta_{metric}"] = pd.NA
    return out


def _indicator_map_for_cohort(models_cfg: dict[str, Any], cohort: str) -> dict[str, list[str]]:
    if cohort == "cnlsy":
        indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", []) if str(x).strip()]
        return {"g_cnlsy": indicators}

    factor_cfg = models_cfg.get("hierarchical_factors", {})
    if not isinstance(factor_cfg, dict):
        factor_cfg = {}
    return {
        "Speed": [str(x) for x in factor_cfg.get("speed", []) if str(x).strip()],
        "Math": [str(x) for x in factor_cfg.get("math", []) if str(x).strip()],
        "Verbal": [str(x) for x in factor_cfg.get("verbal", []) if str(x).strip()],
        "Tech": [str(x) for x in factor_cfg.get("technical", []) if str(x).strip()],
    }


def _parse_partial_indicator(parameter: str) -> str:
    token = str(parameter).strip()
    if "~1" not in token:
        return token
    return token.split("~1", 1)[0].strip()


def _normalize_group_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    token = str(value).strip().lower()
    if token in {"f", "female", "2", "woman", "w"}:
        return "female"
    if token in {"m", "male", "1", "man", "boy"}:
        return "male"
    return token


def _infer_group_pair(groups: list[str], reference_group: str) -> tuple[str | None, str | None]:
    if len(groups) != 2:
        return None, None
    norm_ref = _normalize_group_label(reference_group)
    if norm_ref:
        matches = [g for g in groups if _normalize_group_label(g) == norm_ref]
        if len(matches) == 1:
            female = matches[0]
            male = groups[0] if groups[1] == female else groups[1]
            return female, male
    female_matches = [g for g in groups if _normalize_group_label(g) == "female"]
    male_matches = [g for g in groups if _normalize_group_label(g) == "male"]
    if len(female_matches) == 1 and len(male_matches) == 1:
        return female_matches[0], male_matches[0]
    # Deterministic fallback for unknown labels.
    ordered = sorted(groups)
    return ordered[0], ordered[1]


def _factor_allowance_limits(
    *,
    indicator_map: dict[str, list[str]],
    partial_policy: dict[str, float],
) -> dict[str, int]:
    max_free = int(partial_policy["max_free_per_factor"])
    free_share = float(partial_policy["max_free_share_per_factor"])
    min_invariant_share = float(partial_policy["min_invariant_share_per_factor"])
    min_allowance = int(partial_policy["min_free_allowance_per_factor"])

    limits: dict[str, int] = {}
    for factor, indicators in indicator_map.items():
        n_indicators = len(indicators)
        if n_indicators <= 0:
            limits[factor] = 0
            continue
        cap_by_share = int(math.floor(free_share * n_indicators))
        cap = min(max_free, max(min_allowance, cap_by_share))
        max_by_invariant_share = int(math.floor((1.0 - min_invariant_share) * n_indicators + 1e-12))
        limits[factor] = max(0, min(cap, max_by_invariant_share))
    return limits


def _metric_intercept_gap_candidates(
    *,
    params: pd.DataFrame,
    indicator_map: dict[str, list[str]],
    reference_group: str,
) -> pd.DataFrame:
    if params.empty:
        return pd.DataFrame(columns=["indicator", "factor", "female_est", "male_est", "abs_intercept_gap"])
    metric = params.loc[(params.get("model_step") == "metric") & (params.get("op") == "~1")].copy()
    if metric.empty:
        return pd.DataFrame(columns=["indicator", "factor", "female_est", "male_est", "abs_intercept_gap"])

    indicators_to_factor: dict[str, str] = {}
    for factor, indicators in indicator_map.items():
        for indicator in indicators:
            indicators_to_factor[indicator] = factor

    metric = metric.loc[metric["lhs"].astype(str).isin(indicators_to_factor.keys())].copy()
    if metric.empty:
        return pd.DataFrame(columns=["indicator", "factor", "female_est", "male_est", "abs_intercept_gap"])
    metric["group"] = metric["group"].astype(str)
    metric["est"] = pd.to_numeric(metric["est"], errors="coerce")
    metric = metric.dropna(subset=["est"])
    if metric.empty:
        return pd.DataFrame(columns=["indicator", "factor", "female_est", "male_est", "abs_intercept_gap"])

    rows: list[dict[str, Any]] = []
    for indicator, subset in metric.groupby("lhs", sort=True):
        groups = sorted(subset["group"].dropna().astype(str).unique().tolist())
        female_group, male_group = _infer_group_pair(groups, reference_group=reference_group)
        if female_group is None or male_group is None:
            continue
        female_vals = subset.loc[subset["group"] == female_group, "est"]
        male_vals = subset.loc[subset["group"] == male_group, "est"]
        if female_vals.empty or male_vals.empty:
            continue
        female_est = float(female_vals.iloc[0])
        male_est = float(male_vals.iloc[0])
        rows.append(
            {
                "indicator": str(indicator),
                "factor": indicators_to_factor.get(str(indicator), ""),
                "female_est": female_est,
                "male_est": male_est,
                "abs_intercept_gap": abs(male_est - female_est),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["indicator", "factor", "female_est", "male_est", "abs_intercept_gap"])
    ranked = pd.DataFrame(rows).sort_values(["abs_intercept_gap", "indicator"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def _score_test_intercept_candidates(
    *,
    score_tests: pd.DataFrame,
    indicator_map: dict[str, list[str]],
) -> pd.DataFrame:
    if score_tests.empty:
        return pd.DataFrame(columns=["indicator", "factor", "score_test_x2", "score_test_pvalue", "rank"])
    rows = score_tests.copy()
    if "model_step" in rows.columns:
        scalar_rows = rows.loc[rows["model_step"].astype(str).str.lower() == "scalar"].copy()
        if not scalar_rows.empty:
            rows = scalar_rows

    if "constraint_type" in rows.columns:
        rows = rows.loc[rows["constraint_type"].astype(str).str.lower() == "intercept"].copy()
    elif "mapped_op" in rows.columns:
        rows = rows.loc[rows["mapped_op"].astype(str) == "~1"].copy()
    if rows.empty:
        return pd.DataFrame(columns=["indicator", "factor", "score_test_x2", "score_test_pvalue", "rank"])

    rows["indicator"] = rows.get("mapped_lhs", pd.Series(dtype=object)).astype(str).str.strip()
    rows["score_test_x2"] = pd.to_numeric(rows.get("x2"), errors="coerce")
    rows["score_test_pvalue"] = pd.to_numeric(rows.get("p_value"), errors="coerce")

    indicator_to_factor = {
        indicator: factor
        for factor, indicators in indicator_map.items()
        for indicator in indicators
    }
    rows = rows.loc[rows["indicator"].isin(indicator_to_factor.keys())].copy()
    if rows.empty:
        return pd.DataFrame(columns=["indicator", "factor", "score_test_x2", "score_test_pvalue", "rank"])
    rows["factor"] = rows["indicator"].map(indicator_to_factor)
    rows = rows.dropna(subset=["score_test_x2"])
    if rows.empty:
        return pd.DataFrame(columns=["indicator", "factor", "score_test_x2", "score_test_pvalue", "rank"])

    ranked = (
        rows.sort_values(
            ["score_test_x2", "score_test_pvalue", "indicator"],
            ascending=[False, True, True],
        )
        .drop_duplicates(subset=["indicator"], keep="first")
        .reset_index(drop=True)
    )
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked[["indicator", "factor", "score_test_x2", "score_test_pvalue", "rank"]]


def _select_partial_intercepts(
    *,
    indicator_map: dict[str, list[str]],
    partial_policy: dict[str, float],
    configured_partial: list[str],
    candidates: pd.DataFrame,
    candidate_source: str,
    selected_reason: str,
    exceeds_reason: str,
    candidate_score_col: str,
    candidate_pvalue_col: str | None = None,
) -> tuple[list[str], pd.DataFrame]:
    indicator_to_factor = {
        indicator: factor
        for factor, indicators in indicator_map.items()
        for indicator in indicators
    }
    limits = _factor_allowance_limits(indicator_map=indicator_map, partial_policy=partial_policy)
    selected_indicators: list[str] = []
    selected_counts: dict[str, int] = {factor: 0 for factor in indicator_map}
    proposal_rows: list[dict[str, Any]] = []

    for parameter in configured_partial:
        indicator = _parse_partial_indicator(parameter)
        factor = indicator_to_factor.get(indicator, "")
        selected = False
        reason = "config_unknown_indicator"
        if factor:
            if selected_counts.get(factor, 0) < limits.get(factor, 0) and indicator not in selected_indicators:
                selected_indicators.append(indicator)
                selected_counts[factor] = selected_counts.get(factor, 0) + 1
                selected = True
                reason = "config_selected"
            else:
                reason = "config_exceeds_factor_capacity"
        proposal_rows.append(
            {
                "parameter": f"{indicator}~1",
                "indicator": indicator,
                "factor": factor or pd.NA,
                "source": "config",
                "selected": selected,
                "selection_reason": reason,
                "rank": pd.NA,
                "abs_intercept_gap": pd.NA,
                "candidate_score_type": pd.NA,
                "candidate_pvalue": pd.NA,
            }
        )

    if not candidates.empty:
        for _, row in candidates.iterrows():
            indicator = str(row["indicator"])
            factor = str(row["factor"])
            if not factor:
                continue
            score_val = row.get(candidate_score_col, pd.NA)
            if pd.isna(pd.to_numeric(pd.Series([score_val]), errors="coerce").iloc[0]):
                continue
            p_val = row.get(candidate_pvalue_col, pd.NA) if candidate_pvalue_col else pd.NA
            if indicator in selected_indicators:
                continue
            if selected_counts.get(factor, 0) >= limits.get(factor, 0):
                proposal_rows.append(
                    {
                        "parameter": f"{indicator}~1",
                        "indicator": indicator,
                        "factor": factor,
                        "source": candidate_source,
                        "selected": False,
                        "selection_reason": exceeds_reason,
                        "rank": int(row.get("rank", pd.NA)) if pd.notna(row.get("rank", pd.NA)) else pd.NA,
                        "abs_intercept_gap": float(score_val)
                        if pd.notna(pd.to_numeric(pd.Series([score_val]), errors="coerce").iloc[0])
                        else pd.NA,
                        "candidate_score_type": candidate_score_col,
                        "candidate_pvalue": float(p_val)
                        if pd.notna(pd.to_numeric(pd.Series([p_val]), errors="coerce").iloc[0])
                        else pd.NA,
                    }
                )
                continue
            selected_indicators.append(indicator)
            selected_counts[factor] = selected_counts.get(factor, 0) + 1
            proposal_rows.append(
                {
                    "parameter": f"{indicator}~1",
                    "indicator": indicator,
                    "factor": factor,
                    "source": candidate_source,
                    "selected": True,
                    "selection_reason": selected_reason,
                    "rank": int(row.get("rank", pd.NA)) if pd.notna(row.get("rank", pd.NA)) else pd.NA,
                    "abs_intercept_gap": float(score_val)
                    if pd.notna(pd.to_numeric(pd.Series([score_val]), errors="coerce").iloc[0])
                    else pd.NA,
                    "candidate_score_type": candidate_score_col,
                    "candidate_pvalue": float(p_val)
                    if pd.notna(pd.to_numeric(pd.Series([p_val]), errors="coerce").iloc[0])
                    else pd.NA,
                }
            )

    selected_params = [f"{indicator}~1" for indicator in selected_indicators]
    proposal = pd.DataFrame(proposal_rows)
    if proposal.empty:
        proposal = pd.DataFrame(
            columns=[
                "parameter",
                "indicator",
                "factor",
                "source",
                "selected",
                "selection_reason",
                "rank",
                "abs_intercept_gap",
                "candidate_score_type",
                "candidate_pvalue",
            ]
        )
    return selected_params, proposal


def _attempt_partial_scalar_refit(
    *,
    root: Path,
    cohort: str,
    paths_cfg: dict[str, Any],
    partial_params: list[str],
) -> dict[str, Any]:
    sem_interim_dir = _resolve_path(paths_cfg["sem_interim_dir"], root) / cohort
    base_request = sem_interim_dir / "request.json"
    model_file = sem_interim_dir / "model.lavaan"
    r_script = root / "scripts" / "sem_fit.R"
    outdir = _resolve_path(paths_cfg["outputs_dir"], root) / "model_fits" / cohort / "partial_scalar_refit"

    if not base_request.exists():
        return {"attempted": False, "success": False, "reason": "missing_request_json", "fit_dir": pd.NA}
    if not model_file.exists():
        return {"attempted": False, "success": False, "reason": "missing_model_lavaan", "fit_dir": pd.NA}
    if rscript_path() is None:
        return {"attempted": False, "success": False, "reason": "rscript_not_available", "fit_dir": pd.NA}
    if not r_script.exists():
        return {"attempted": False, "success": False, "reason": "missing_sem_fit_script", "fit_dir": pd.NA}

    request_payload = json.loads(base_request.read_text(encoding="utf-8"))
    request_payload["partial_intercepts"] = list(partial_params)
    request_payload["invariance_steps"] = ["configural", "metric", "scalar"]

    request_file = sem_interim_dir / "request_partial_scalar.json"
    request_file.write_text(json.dumps(request_payload, indent=2, sort_keys=True), encoding="utf-8")
    outdir.mkdir(parents=True, exist_ok=True)

    def _sanitize_refit_error(message: str) -> str:
        token = str(message)
        root_str = str(root)
        if root_str and root_str in token:
            token = token.replace(root_str, "<PROJECT_ROOT>")
        return token

    try:
        cp = run_sem_r_script(r_script=r_script, request_file=request_file, outdir=outdir)
        return {
            "attempted": True,
            "success": True,
            "reason": "",
            "fit_dir": relative_path(root, outdir),
            "stdout": cp.stdout,
            "stderr": cp.stderr,
        }
    except Exception as exc:
        return {
            "attempted": True,
            "success": False,
            "reason": f"r_refit_failed:{_sanitize_refit_error(str(exc))}",
            "fit_dir": relative_path(root, outdir),
        }


def _eval_partial_constraints(
    *,
    partial_params: list[str],
    indicator_map: dict[str, list[str]],
    partial_policy: dict[str, float],
) -> dict[str, Any]:
    max_free = int(partial_policy["max_free_per_factor"])
    free_share = float(partial_policy["max_free_share_per_factor"])
    min_invariant_share = float(partial_policy["min_invariant_share_per_factor"])
    min_allowance = int(partial_policy["min_free_allowance_per_factor"])

    indicators_to_factor: dict[str, str] = {}
    for factor, indicators in indicator_map.items():
        for indicator in indicators:
            indicators_to_factor[indicator] = factor

    parsed = [_parse_partial_indicator(p) for p in partial_params if str(p).strip()]
    unknown = [p for p in parsed if p not in indicators_to_factor]
    per_factor: dict[str, int] = {}
    for item in parsed:
        factor = indicators_to_factor.get(item)
        if factor is None:
            continue
        per_factor[factor] = per_factor.get(factor, 0) + 1

    cap_violations: list[str] = []
    share_violations: list[str] = []
    per_factor_rows: list[dict[str, Any]] = []
    for factor, indicators in indicator_map.items():
        n_indicators = len(indicators)
        freed = int(per_factor.get(factor, 0))
        if n_indicators <= 0:
            continue
        cap_by_share = int(math.floor(free_share * n_indicators))
        cap = min(max_free, max(min_allowance, cap_by_share))
        invariant_share = (n_indicators - freed) / float(n_indicators)
        if freed > cap:
            cap_violations.append(factor)
        if invariant_share < min_invariant_share:
            share_violations.append(factor)
        per_factor_rows.append(
            {
                "factor": factor,
                "indicator_count": n_indicators,
                "freed_count": freed,
                "cap": cap,
                "invariant_share": invariant_share,
            }
        )

    constraints_ok = (not unknown) and (not cap_violations) and (not share_violations)
    return {
        "constraints_ok": constraints_ok,
        "total_freed_intercepts": len(parsed),
        "unknown_freed_intercepts": len(unknown),
        "cap_violation_factors": ";".join(cap_violations) if cap_violations else pd.NA,
        "invariant_share_violation_factors": ";".join(share_violations) if share_violations else pd.NA,
        "per_factor": per_factor_rows,
    }


def _lookup_step_row(summary: pd.DataFrame, step: str) -> pd.Series | None:
    rows = summary[summary["model_step"] == step]
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
    strict_step_requirement: bool = True,
) -> dict[str, Any]:
    prev_row = _lookup_step_row(summary, prev_step)
    next_row = _lookup_step_row(summary, next_step)
    out = {
        "cohort": cohort,
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
        "reason": "step_missing",
    }
    if prev_row is None or next_row is None:
        return out

    def _delta(metric: str) -> float | None:
        delta_col = f"delta_{metric}"
        val = pd.to_numeric(pd.Series([next_row.get(delta_col)]), errors="coerce").iloc[0]
        if not pd.isna(val):
            return float(val)
        prev_val = pd.to_numeric(pd.Series([prev_row.get(metric)]), errors="coerce").iloc[0]
        next_val = pd.to_numeric(pd.Series([next_row.get(metric)]), errors="coerce").iloc[0]
        if pd.isna(prev_val) or pd.isna(next_val):
            return None
        return float(next_val - prev_val)

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
            excess["delta_cfi"] = float(thresholds["delta_cfi_min"] - d_cfi)
    if d_rmsea is not None:
        passed = d_rmsea <= thresholds["delta_rmsea_max"]
        checks.append(passed)
        if not passed:
            excess["delta_rmsea"] = float(d_rmsea - thresholds["delta_rmsea_max"])
    if d_srmr is not None:
        passed = d_srmr <= thresholds["delta_srmr_max"]
        checks.append(passed)
        if not passed:
            excess["delta_srmr"] = float(d_srmr - thresholds["delta_srmr_max"])

    out["evaluated_criteria"] = len(checks)
    if strict_step_requirement and len(checks) == 0:
        out["passed"] = False
        out["reason"] = "criteria_not_evaluable"
        return out

    if checks and all(checks):
        out["passed"] = True
        out["reason"] = ""
        return out

    if checks and len(excess) == 1 and len(checks) >= 2:
        tolerable = True
        if "delta_cfi" in excess:
            tolerable = excess["delta_cfi"] <= tolerance["delta_cfi_excess_max"]
        elif "delta_rmsea" in excess:
            tolerable = excess["delta_rmsea"] <= tolerance["delta_rmsea_excess_max"]
        elif "delta_srmr" in excess:
            tolerable = excess["delta_srmr"] <= tolerance["delta_srmr_excess_max"]
        if tolerable:
            out["passed"] = True
            out["reason"] = "near_pass_tolerance"
            return out

    out["passed"] = False
    failed_parts: list[str] = []
    if d_cfi is not None and not (d_cfi >= thresholds["delta_cfi_min"]):
        failed_parts.append("delta_cfi")
    if d_rmsea is not None and not (d_rmsea <= thresholds["delta_rmsea_max"]):
        failed_parts.append("delta_rmsea")
    if d_srmr is not None and not (d_srmr <= thresholds["delta_srmr_max"]):
        failed_parts.append("delta_srmr")
    out["reason"] = "failed_" + ",".join(failed_parts) if failed_parts else "criteria_not_evaluable"
    return out


def _decision_rows(
    *,
    summary: pd.DataFrame,
    cohort: str,
    models_cfg: dict[str, Any],
    partial_params: list[str],
    partial_refit: dict[str, Any] | None = None,
    partial_refit_summary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gatekeeping_on = _gatekeeping_enabled(models_cfg)
    thresholds = _invariance_thresholds(models_cfg)
    tolerance = _invariance_tolerance(models_cfg)
    partial_policy = _partial_policy(models_cfg)
    indicator_map = _indicator_map_for_cohort(models_cfg, cohort)
    partial_eval = _eval_partial_constraints(
        partial_params=partial_params,
        indicator_map=indicator_map,
        partial_policy=partial_policy,
    )

    metric_check = _transition_check(
        summary=summary,
        cohort=cohort,
        prev_step="configural",
        next_step="metric",
        thresholds=thresholds["metric"],
        tolerance=tolerance["metric"],
    )
    scalar_check = _transition_check(
        summary=summary,
        cohort=cohort,
        prev_step="metric",
        next_step="scalar",
        thresholds=thresholds["scalar"],
        tolerance=tolerance["scalar"],
    )
    scalar_check_source = "baseline"

    if isinstance(partial_refit_summary, pd.DataFrame) and not partial_refit_summary.empty:
        refit_summary = _add_deltas(_ensure_step_rows(partial_refit_summary, ["configural", "metric", "scalar"], cohort))
        scalar_check_refit = _transition_check(
            summary=refit_summary,
            cohort=cohort,
            prev_step="metric",
            next_step="scalar",
            thresholds=thresholds["scalar"],
            tolerance=tolerance["scalar"],
        )
        if bool(scalar_check_refit.get("passed", False)):
            scalar_check = scalar_check_refit
            scalar_check_source = "partial_refit"

    metric_pass = bool(metric_check["passed"])
    scalar_pass = bool(scalar_check["passed"])
    partial_ok = bool(partial_eval["constraints_ok"])

    if gatekeeping_on:
        vr_ok = metric_pass
        dg_ok = metric_pass and scalar_pass and partial_ok
        reason_vr = "" if vr_ok else f"metric_gate:{metric_check['reason']}"
        if dg_ok:
            reason_dg = ""
        elif not metric_pass:
            reason_dg = f"metric_gate:{metric_check['reason']}"
        elif not scalar_pass:
            reason_dg = f"scalar_gate:{scalar_check['reason']}"
        else:
            reason_dg = "partial_policy_violation"
    else:
        vr_ok = True
        dg_ok = True
        reason_vr = ""
        reason_dg = ""

    decision = pd.DataFrame(
        [
            {
                "cohort": cohort,
                "gatekeeping_enabled": gatekeeping_on,
                "metric_pass": metric_pass,
                "scalar_pass": scalar_pass,
                "partial_constraints_ok": partial_ok,
                "confirmatory_vr_g_eligible": vr_ok,
                "confirmatory_d_g_eligible": dg_ok,
                "invariance_ok_for_vr": vr_ok,
                "invariance_ok_for_d": dg_ok,
                "reason_vr_g": reason_vr if reason_vr else pd.NA,
                "reason_d_g": reason_dg if reason_dg else pd.NA,
                "total_freed_intercepts": partial_eval["total_freed_intercepts"],
                "unknown_freed_intercepts": partial_eval["unknown_freed_intercepts"],
                "partial_cap_violations": partial_eval["cap_violation_factors"],
                "partial_invariant_share_violations": partial_eval["invariant_share_violation_factors"],
                "partial_freed_parameters": ";".join(partial_params) if partial_params else pd.NA,
                "scalar_check_source": scalar_check_source,
                "partial_refit_attempted": bool((partial_refit or {}).get("attempted", False)),
                "partial_refit_success": bool((partial_refit or {}).get("success", False)),
                "partial_refit_reason": str((partial_refit or {}).get("reason", "")).strip() or pd.NA,
                "partial_refit_dir": (partial_refit or {}).get("fit_dir", pd.NA),
                "partial_refit_used": scalar_check_source == "partial_refit",
            }
        ]
    )

    metric_row = dict(metric_check)
    metric_row["check_source"] = "baseline"
    scalar_row = dict(scalar_check)
    scalar_row["check_source"] = scalar_check_source
    transition_df = pd.DataFrame([metric_row, scalar_row])
    return decision, transition_df


def _selected_indicators_by_cohort(partial_rows: pd.DataFrame) -> dict[str, set[str]]:
    if partial_rows.empty or "cohort" not in partial_rows.columns:
        return {}

    selected = partial_rows.copy()
    if "selected" in selected.columns:
        selected["selected"] = selected["selected"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
        selected = selected.loc[selected["selected"]].copy()
    if selected.empty:
        return {}

    selected["indicator"] = selected.get("indicator", pd.Series(dtype=object)).astype(str).str.strip()
    selected = selected.loc[selected["indicator"].ne("") & selected["indicator"].ne("nan")].copy()
    if selected.empty:
        return {}

    out: dict[str, set[str]] = {}
    for cohort, rows in selected.groupby("cohort", sort=False):
        out[str(cohort)] = set(rows["indicator"].astype(str).tolist())
    return out


def _flatten_indicators(indicator_map: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for indicators in indicator_map.values():
        for indicator in indicators:
            token = str(indicator).strip()
            if token and token not in seen:
                seen.add(token)
                out.append(token)
    return out


def _bootstrap_cluster_ids(
    *,
    cohort: str,
    df: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.Series:
    configured = policy.get("bootstrap_cluster_cols", {})
    cluster_cols: list[str] = []
    if isinstance(configured, dict):
        raw = configured.get(cohort, [])
        if isinstance(raw, (list, tuple)):
            cluster_cols = [str(x) for x in raw if str(x).strip()]

    cluster = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    for col in cluster_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        cluster = cluster.where(cluster.notna(), vals)

    if cluster.isna().all() and "person_id" in df.columns:
        cluster = pd.to_numeric(df["person_id"], errors="coerce")
    cluster = cluster.map(lambda x: f"cluster_{int(x)}" if pd.notna(x) else pd.NA)
    fallback = pd.Series([f"singleton_{i}" for i in range(len(df))], index=df.index, dtype="object")
    return cluster.where(cluster.notna(), fallback).astype(str)


def _bootstrap_metric_gap_candidates(
    *,
    sample_df: pd.DataFrame,
    indicator_map: dict[str, list[str]],
    min_group_n: int,
) -> pd.DataFrame:
    if "sex" not in sample_df.columns:
        return pd.DataFrame(columns=["indicator", "factor", "abs_intercept_gap", "rank"])

    sex = sample_df["sex"].map(_normalize_group_label)
    male = sex.eq("male")
    female = sex.eq("female")
    if int(male.sum()) < min_group_n or int(female.sum()) < min_group_n:
        return pd.DataFrame(columns=["indicator", "factor", "abs_intercept_gap", "rank"])

    rows: list[dict[str, Any]] = []
    for factor, indicators in indicator_map.items():
        for indicator in indicators:
            if indicator not in sample_df.columns:
                continue
            vals = pd.to_numeric(sample_df[indicator], errors="coerce")
            male_vals = vals[male].dropna()
            female_vals = vals[female].dropna()
            if len(male_vals) < 2 or len(female_vals) < 2:
                continue
            gap = abs(float(male_vals.mean()) - float(female_vals.mean()))
            rows.append(
                {
                    "indicator": indicator,
                    "factor": factor,
                    "abs_intercept_gap": gap,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["indicator", "factor", "abs_intercept_gap", "rank"])
    ranked = pd.DataFrame(rows).sort_values(["abs_intercept_gap", "indicator"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def _bootstrap_partial_support_for_cohort(
    *,
    root: Path,
    cohort: str,
    paths_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    policy: dict[str, Any],
    selected_indicators: set[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    base_summary = {
        "cohort": cohort,
        "bootstrap_checked": False,
        "bootstrap_pass": pd.NA,
        "bootstrap_reason": "not_applicable",
        "bootstrap_replicates_requested": int(policy.get("bootstrap_replicates", 0)),
        "bootstrap_successful_reps": 0,
        "bootstrap_min_indicator_share": pd.NA,
        "bootstrap_threshold_share": float(policy.get("min_bootstrap_indicator_share", 0.7)),
        "bootstrap_threshold_success_reps": int(policy.get("min_bootstrap_success_reps", 1)),
    }
    if not selected_indicators:
        return base_summary, pd.DataFrame(columns=["cohort", "indicator", "selection_share", "selected_count", "successful_reps"])

    processed_dir = _resolve_path(paths_cfg["processed_dir"], root)
    source_path = processed_dir / f"{cohort}_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / f"{cohort}_cfa.csv"
    if not source_path.exists():
        out = dict(base_summary)
        out["bootstrap_checked"] = True
        out["bootstrap_pass"] = False
        out["bootstrap_reason"] = "missing_source_data"
        return out, pd.DataFrame(columns=["cohort", "indicator", "selection_share", "selected_count", "successful_reps"])

    indicator_map = _indicator_map_for_cohort(models_cfg, cohort)
    indicators = _flatten_indicators(indicator_map)
    required_cols = ["sex", "person_id", *indicators]
    for c in policy.get("bootstrap_cluster_cols", {}).get(cohort, []):
        if c not in required_cols:
            required_cols.append(c)
    # Read only required columns that exist to keep this lightweight.
    header = pd.read_csv(source_path, nrows=0)
    existing_cols = [col for col in required_cols if col in header.columns]
    if "sex" not in existing_cols:
        out = dict(base_summary)
        out["bootstrap_checked"] = True
        out["bootstrap_pass"] = False
        out["bootstrap_reason"] = "missing_sex_column"
        return out, pd.DataFrame(columns=["cohort", "indicator", "selection_share", "selected_count", "successful_reps"])

    df = pd.read_csv(source_path, usecols=existing_cols, low_memory=False)
    if df.empty:
        out = dict(base_summary)
        out["bootstrap_checked"] = True
        out["bootstrap_pass"] = False
        out["bootstrap_reason"] = "empty_source_data"
        return out, pd.DataFrame(columns=["cohort", "indicator", "selection_share", "selected_count", "successful_reps"])

    cluster_ids = _bootstrap_cluster_ids(cohort=cohort, df=df, policy=policy)
    cluster_to_rows: dict[str, np.ndarray] = {}
    for cluster, rows in pd.Series(np.arange(len(df)), index=cluster_ids).groupby(level=0, sort=False):
        idx = rows.to_numpy(dtype=int)
        if idx.size > 0:
            cluster_to_rows[str(cluster)] = idx
    clusters = list(cluster_to_rows.keys())
    if not clusters:
        out = dict(base_summary)
        out["bootstrap_checked"] = True
        out["bootstrap_pass"] = False
        out["bootstrap_reason"] = "no_clusters"
        return out, pd.DataFrame(columns=["cohort", "indicator", "selection_share", "selected_count", "successful_reps"])

    configured_partial = _partial_intercepts(models_cfg, cohort)
    partial_policy = _partial_policy(models_cfg)
    n_boot = int(policy.get("bootstrap_replicates", 200))
    min_group_n = int(policy.get("bootstrap_min_group_n", 50))
    seed = int(policy.get("bootstrap_seed", 1729))

    selected_counts = {indicator: 0 for indicator in sorted(selected_indicators)}
    successful_reps = 0
    cohort_seed_offset = sum(ord(ch) for ch in cohort) % 10_000
    rng = np.random.default_rng(seed + cohort_seed_offset)
    cluster_count = len(clusters)
    for _ in range(n_boot):
        picks = rng.integers(0, cluster_count, size=cluster_count)
        sampled_idx = np.concatenate([cluster_to_rows[clusters[i]] for i in picks])
        if sampled_idx.size == 0:
            continue
        sample = df.iloc[sampled_idx]
        candidates = _bootstrap_metric_gap_candidates(
            sample_df=sample,
            indicator_map=indicator_map,
            min_group_n=min_group_n,
        )
        if candidates.empty:
            continue
        selected_params, _ = _select_partial_intercepts(
            indicator_map=indicator_map,
            partial_policy=partial_policy,
            configured_partial=configured_partial,
            candidates=candidates,
            candidate_source="bootstrap_metric_gap",
            selected_reason="bootstrap_selected",
            exceeds_reason="bootstrap_exceeds_factor_capacity",
            candidate_score_col="abs_intercept_gap",
        )
        selected_now = {_parse_partial_indicator(item) for item in selected_params}
        successful_reps += 1
        for indicator in selected_counts:
            if indicator in selected_now:
                selected_counts[indicator] += 1

    summary = dict(base_summary)
    summary["bootstrap_checked"] = True
    summary["bootstrap_successful_reps"] = int(successful_reps)
    min_success = int(policy.get("min_bootstrap_success_reps", 1))
    if successful_reps < min_success:
        summary["bootstrap_pass"] = False
        summary["bootstrap_reason"] = "insufficient_successful_reps"
        return summary, pd.DataFrame(
            [
                {
                    "cohort": cohort,
                    "indicator": indicator,
                    "selected_count": int(selected_counts[indicator]),
                    "successful_reps": int(successful_reps),
                    "selection_share": pd.NA,
                }
                for indicator in sorted(selected_counts.keys())
            ]
        )

    shares = {
        indicator: (float(count) / float(successful_reps))
        for indicator, count in selected_counts.items()
    }
    min_share = min(shares.values()) if shares else 0.0
    threshold = float(policy.get("min_bootstrap_indicator_share", 0.7))
    passed = min_share >= threshold
    summary["bootstrap_min_indicator_share"] = min_share
    summary["bootstrap_pass"] = bool(passed)
    summary["bootstrap_reason"] = "ok" if passed else "below_bootstrap_threshold"

    detail = pd.DataFrame(
        [
            {
                "cohort": cohort,
                "indicator": indicator,
                "selected_count": int(selected_counts[indicator]),
                "successful_reps": int(successful_reps),
                "selection_share": float(shares[indicator]),
            }
            for indicator in sorted(shares.keys())
        ]
    )
    return summary, detail


def _build_bootstrap_partial_support(
    *,
    root: Path,
    paths_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    decisions: pd.DataFrame,
    partial_rows: pd.DataFrame,
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indicator_sets = _selected_indicators_by_cohort(partial_rows)
    enabled = bool(policy.get("enabled", False))
    mode = str(policy.get("mode", "cross_cohort_overlap"))
    apply_when_partial_refit = bool(policy.get("apply_when_partial_refit", True))
    use_bootstrap = mode in {"bootstrap_stability", "combined"}

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[pd.DataFrame] = []
    for _, row in decisions.iterrows():
        cohort = str(row.get("cohort", ""))
        gatekeeping_enabled = _as_bool(row.get("gatekeeping_enabled"), False)
        dg_eligible = _as_bool(row.get("confirmatory_d_g_eligible"), False)
        partial_refit_used = _as_bool(row.get("partial_refit_used"), False)
        total_freed = int(_as_float(row.get("total_freed_intercepts"), 0.0))
        should_check = enabled and use_bootstrap and gatekeeping_enabled and dg_eligible and total_freed > 0
        if apply_when_partial_refit:
            should_check = should_check and partial_refit_used
        if not should_check:
            summary_rows.append(
                {
                    "cohort": cohort,
                    "bootstrap_checked": False,
                    "bootstrap_pass": pd.NA,
                    "bootstrap_reason": "not_applicable",
                    "bootstrap_replicates_requested": int(policy.get("bootstrap_replicates", 0)),
                    "bootstrap_successful_reps": 0,
                    "bootstrap_min_indicator_share": pd.NA,
                    "bootstrap_threshold_share": float(policy.get("min_bootstrap_indicator_share", 0.7)),
                    "bootstrap_threshold_success_reps": int(policy.get("min_bootstrap_success_reps", 1)),
                }
            )
            continue

        summary, detail = _bootstrap_partial_support_for_cohort(
            root=root,
            cohort=cohort,
            paths_cfg=paths_cfg,
            models_cfg=models_cfg,
            policy=policy,
            selected_indicators=indicator_sets.get(cohort, set()),
        )
        summary_rows.append(summary)
        if not detail.empty:
            detail_rows.append(detail)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "cohort",
                "bootstrap_checked",
                "bootstrap_pass",
                "bootstrap_reason",
                "bootstrap_replicates_requested",
                "bootstrap_successful_reps",
                "bootstrap_min_indicator_share",
                "bootstrap_threshold_share",
                "bootstrap_threshold_success_reps",
            ]
        )
    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame(
        columns=["cohort", "indicator", "selected_count", "successful_reps", "selection_share"]
    )
    return summary_df, detail_df


def _append_reason(base_reason: Any, extra_reason: str) -> str:
    base = "" if pd.isna(base_reason) else str(base_reason).strip()
    if not base:
        return extra_reason
    parts = [x.strip() for x in base.split(";") if x.strip()]
    if extra_reason in parts:
        return base
    return ";".join(parts + [extra_reason])


def _apply_partial_replicability_guard(
    *,
    decisions: pd.DataFrame,
    partial_rows: pd.DataFrame,
    policy: dict[str, Any],
    bootstrap_support: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = decisions.copy()
    if "confirmatory_vr_g_eligible" not in out.columns:
        out["confirmatory_vr_g_eligible"] = out["confirmatory_d_g_eligible"]
    if "invariance_ok_for_vr" not in out.columns:
        out["invariance_ok_for_vr"] = out["confirmatory_vr_g_eligible"]
    if "invariance_ok_for_d" not in out.columns:
        out["invariance_ok_for_d"] = out["confirmatory_d_g_eligible"]

    indicator_sets = _selected_indicators_by_cohort(partial_rows)
    report_rows: list[dict[str, Any]] = []
    enabled = bool(policy.get("enabled", False))
    apply_when_partial_refit = bool(policy.get("apply_when_partial_refit", True))
    min_overlap_share = float(policy.get("min_overlap_share", 0.7))
    comparison_cohorts = [str(x) for x in policy.get("comparison_cohorts", [])]
    mode = str(policy.get("mode", "cross_cohort_overlap"))
    use_cross = mode in {"cross_cohort_overlap", "combined"}
    use_bootstrap = mode in {"bootstrap_stability", "combined"}
    bootstrap_lookup = {}
    if isinstance(bootstrap_support, pd.DataFrame) and not bootstrap_support.empty and "cohort" in bootstrap_support.columns:
        bootstrap_lookup = {
            str(cohort): row
            for cohort, row in bootstrap_support.groupby("cohort", sort=False).first().iterrows()
        }

    for idx, row in out.iterrows():
        cohort = str(row.get("cohort", ""))
        gatekeeping_enabled = _as_bool(row.get("gatekeeping_enabled"), False)
        dg_eligible = _as_bool(row.get("confirmatory_d_g_eligible"), False)
        partial_refit_used = _as_bool(row.get("partial_refit_used"), False)
        total_freed = int(_as_float(row.get("total_freed_intercepts"), 0.0))
        should_check = enabled and gatekeeping_enabled and dg_eligible and total_freed > 0
        if apply_when_partial_refit:
            should_check = should_check and partial_refit_used

        report = {
            "cohort": cohort,
            "partial_replicability_checked": bool(should_check),
            "partial_replicability_pass": pd.NA,
            "partial_replicability_mode": mode,
            "partial_replicability_min_overlap_share": min_overlap_share,
            "partial_replicability_overlap_share": pd.NA,
            "partial_replicability_reference_cohort": pd.NA,
            "partial_replicability_indicator_count": pd.NA,
            "partial_replicability_shared_count": pd.NA,
            "partial_replicability_bootstrap_checked": pd.NA,
            "partial_replicability_bootstrap_pass": pd.NA,
            "partial_replicability_bootstrap_reason": pd.NA,
            "partial_replicability_bootstrap_successful_reps": pd.NA,
            "partial_replicability_bootstrap_replicates_requested": pd.NA,
            "partial_replicability_bootstrap_min_indicator_share": pd.NA,
            "partial_replicability_bootstrap_threshold_share": pd.NA,
            "partial_replicability_bootstrap_threshold_success_reps": pd.NA,
            "partial_replicability_reason": "not_applicable",
        }

        if not should_check:
            report_rows.append(report)
            continue

        own = indicator_sets.get(cohort, set())
        report["partial_replicability_indicator_count"] = int(len(own))
        cross_pass = True
        cross_reason = "ok"
        if use_cross:
            if not own:
                cross_pass = False
                cross_reason = "no_selected_intercepts"
            else:
                candidate_refs: list[str] = []
                for other in comparison_cohorts:
                    if other == cohort:
                        continue
                    if other in indicator_sets and indicator_sets[other]:
                        candidate_refs.append(other)
                if not candidate_refs:
                    cross_pass = False
                    cross_reason = "no_reference_cohort"
                else:
                    best_ref = ""
                    best_share = -1.0
                    best_shared_count = 0
                    for other in candidate_refs:
                        other_set = indicator_sets.get(other, set())
                        shared = own.intersection(other_set)
                        share = float(len(shared)) / float(len(own))
                        if share > best_share:
                            best_share = share
                            best_ref = other
                            best_shared_count = len(shared)
                    cross_pass = best_share >= min_overlap_share
                    cross_reason = "ok" if cross_pass else "below_overlap_threshold"
                    report["partial_replicability_overlap_share"] = best_share
                    report["partial_replicability_reference_cohort"] = best_ref
                    report["partial_replicability_shared_count"] = int(best_shared_count)

        bootstrap_pass = True
        bootstrap_reason = "ok"
        if use_bootstrap:
            bs_row = bootstrap_lookup.get(cohort)
            if bs_row is None:
                bootstrap_pass = False
                bootstrap_reason = "missing_bootstrap_support"
                report["partial_replicability_bootstrap_checked"] = False
            else:
                bs_checked = _as_bool(bs_row.get("bootstrap_checked"), False)
                bs_pass_val = _as_bool(bs_row.get("bootstrap_pass"), False)
                bs_reason = str(bs_row.get("bootstrap_reason", "")).strip() or "unknown_bootstrap_reason"
                report["partial_replicability_bootstrap_checked"] = bool(bs_checked)
                report["partial_replicability_bootstrap_pass"] = bool(bs_pass_val) if bs_checked else pd.NA
                report["partial_replicability_bootstrap_reason"] = bs_reason
                report["partial_replicability_bootstrap_successful_reps"] = int(
                    _as_float(bs_row.get("bootstrap_successful_reps"), 0.0)
                )
                report["partial_replicability_bootstrap_replicates_requested"] = int(
                    _as_float(bs_row.get("bootstrap_replicates_requested"), 0.0)
                )
                report["partial_replicability_bootstrap_min_indicator_share"] = pd.to_numeric(
                    pd.Series([bs_row.get("bootstrap_min_indicator_share")]), errors="coerce"
                ).iloc[0]
                report["partial_replicability_bootstrap_threshold_share"] = float(
                    _as_float(bs_row.get("bootstrap_threshold_share"), 0.7)
                )
                report["partial_replicability_bootstrap_threshold_success_reps"] = int(
                    _as_float(bs_row.get("bootstrap_threshold_success_reps"), 1.0)
                )
                if not bs_checked:
                    bootstrap_pass = False
                    bootstrap_reason = "bootstrap_not_checked"
                else:
                    bootstrap_pass = bs_pass_val
                    bootstrap_reason = bs_reason

        passed = cross_pass and bootstrap_pass
        report["partial_replicability_pass"] = bool(passed)
        reason_parts: list[str] = []
        if use_cross and cross_reason != "ok":
            reason_parts.append(cross_reason)
        if use_bootstrap and bootstrap_reason != "ok":
            reason_parts.append(bootstrap_reason)
        report["partial_replicability_reason"] = "ok" if not reason_parts else ";".join(reason_parts)
        if not passed:
            out.loc[idx, "confirmatory_d_g_eligible"] = False
            out.loc[idx, "invariance_ok_for_d"] = False
            out.loc[idx, "reason_d_g"] = _append_reason(out.loc[idx, "reason_d_g"], "partial_replicability_guard")

        report_rows.append(report)

    report_df = pd.DataFrame(report_rows)
    if report_df.empty:
        report_df = pd.DataFrame(
            columns=[
                "cohort",
                "partial_replicability_checked",
                "partial_replicability_pass",
                "partial_replicability_mode",
                "partial_replicability_min_overlap_share",
                "partial_replicability_overlap_share",
                "partial_replicability_reference_cohort",
                "partial_replicability_indicator_count",
                "partial_replicability_shared_count",
                "partial_replicability_bootstrap_checked",
                "partial_replicability_bootstrap_pass",
                "partial_replicability_bootstrap_reason",
                "partial_replicability_bootstrap_successful_reps",
                "partial_replicability_bootstrap_replicates_requested",
                "partial_replicability_bootstrap_min_indicator_share",
                "partial_replicability_bootstrap_threshold_share",
                "partial_replicability_bootstrap_threshold_success_reps",
                "partial_replicability_reason",
            ]
        )
    out = out.merge(report_df, on="cohort", how="left")
    out["invariance_ok_for_vr"] = out["confirmatory_vr_g_eligible"]
    out["invariance_ok_for_d"] = out["confirmatory_d_g_eligible"]
    return out, report_df


def _cohort_outputs(
    root: Path,
    cohort: str,
    paths_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs_dir = _resolve_path(paths_cfg["outputs_dir"], root)
    fit_dir = outputs_dir / "model_fits" / cohort
    fit_path = fit_dir / "fit_indices.csv"
    steps = _expected_steps(models_cfg)
    summary = _ensure_step_rows(_load_fit_indices(fit_path), steps, cohort)
    summary = _add_deltas(summary)

    configured_partial = _partial_intercepts(models_cfg, cohort)

    def _config_partial_rows(partial_params: list[str]) -> pd.DataFrame:
        if partial_params:
            return pd.DataFrame(
                {
                    "cohort": cohort,
                    "parameter": partial_params,
                    "indicator": [_parse_partial_indicator(p) for p in partial_params],
                    "factor": pd.NA,
                    "source": "config",
                    "selected": True,
                    "selection_reason": "config_selected",
                    "rank": pd.NA,
                    "abs_intercept_gap": pd.NA,
                    "candidate_score_type": pd.NA,
                    "candidate_pvalue": pd.NA,
                }
            )
        return pd.DataFrame(
            {
                "cohort": [cohort],
                "parameter": [pd.NA],
                "indicator": [pd.NA],
                "factor": [pd.NA],
                "source": ["none"],
                "selected": [False],
                "selection_reason": ["none"],
                "rank": [pd.NA],
                "abs_intercept_gap": [pd.NA],
                "candidate_score_type": [pd.NA],
                "candidate_pvalue": [pd.NA],
            }
        )

    selected_partial = list(configured_partial)
    partial_rows = _config_partial_rows(selected_partial)
    initial_decisions, _ = _decision_rows(
        summary=summary,
        cohort=cohort,
        models_cfg=models_cfg,
        partial_params=configured_partial,
    )

    metric_pass = bool(initial_decisions.iloc[0].get("metric_pass", False))
    scalar_pass = bool(initial_decisions.iloc[0].get("scalar_pass", False))
    gatekeeping_enabled = bool(initial_decisions.iloc[0].get("gatekeeping_enabled", False))

    partial_refit: dict[str, Any] = {"attempted": False, "success": False, "reason": "", "fit_dir": pd.NA}
    partial_refit_summary = pd.DataFrame()
    if gatekeeping_enabled and metric_pass and (not scalar_pass):
        indicator_map = _indicator_map_for_cohort(models_cfg, cohort)
        partial_policy = _partial_policy(models_cfg)
        reference_group = str(models_cfg.get("reference_group", "female"))
        params_path = fit_dir / "params.csv"
        score_tests_path = fit_dir / "lavtestscore.csv"
        params = pd.read_csv(params_path) if params_path.exists() else pd.DataFrame()
        score_candidates = _score_test_intercept_candidates(
            score_tests=_load_score_tests(score_tests_path),
            indicator_map=indicator_map,
        )
        if not score_candidates.empty:
            selected_partial, partial_rows = _select_partial_intercepts(
                indicator_map=indicator_map,
                partial_policy=partial_policy,
                configured_partial=configured_partial,
                candidates=score_candidates,
                candidate_source="lavtestscore_intercept",
                selected_reason="score_selected",
                exceeds_reason="score_exceeds_factor_capacity",
                candidate_score_col="score_test_x2",
                candidate_pvalue_col="score_test_pvalue",
            )
        else:
            metric_candidates = _metric_intercept_gap_candidates(
                params=params,
                indicator_map=indicator_map,
                reference_group=reference_group,
            )
            selected_partial, partial_rows = _select_partial_intercepts(
                indicator_map=indicator_map,
                partial_policy=partial_policy,
                configured_partial=configured_partial,
                candidates=metric_candidates,
                candidate_source="metric_intercept_gap",
                selected_reason="metric_selected",
                exceeds_reason="metric_exceeds_factor_capacity",
                candidate_score_col="abs_intercept_gap",
            )
        partial_rows["cohort"] = cohort
        if partial_rows.empty:
            partial_rows = _config_partial_rows(selected_partial)

    if gatekeeping_enabled and metric_pass and (not scalar_pass) and selected_partial and selected_partial != configured_partial:
        partial_refit = _attempt_partial_scalar_refit(
            root=root,
            cohort=cohort,
            paths_cfg=paths_cfg,
            partial_params=selected_partial,
        )
        refit_dir = partial_refit.get("fit_dir")
        if bool(partial_refit.get("success", False)) and isinstance(refit_dir, str) and refit_dir.strip():
            partial_fit_path = resolve_token_path(root, refit_dir) / "fit_indices.csv"
            partial_refit_summary = _load_fit_indices(partial_fit_path)

    decisions, transitions = _decision_rows(
        summary=summary,
        cohort=cohort,
        models_cfg=models_cfg,
        partial_params=selected_partial,
        partial_refit=partial_refit,
        partial_refit_summary=partial_refit_summary,
    )
    return summary, partial_rows, decisions, transitions


def main() -> int:
    parser = argparse.ArgumentParser(description="Build invariance summaries and partial-intercept records.")
    parser.add_argument("--cohort", action="append", choices=COHORTS, help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--partial-replicability-mode",
        choices=("cross_cohort_overlap", "bootstrap_stability", "combined"),
        help="Override partial replicability guard mode for this run.",
    )
    parser.add_argument(
        "--partial-min-overlap-share",
        type=float,
        help="Override min overlap share threshold [0,1] for partial replicability guard.",
    )
    parser.add_argument(
        "--partial-min-bootstrap-indicator-share",
        type=float,
        help="Override bootstrap indicator share threshold [0,1] for partial replicability guard.",
    )
    parser.add_argument(
        "--partial-min-bootstrap-success-reps",
        type=int,
        help="Override minimum bootstrap successful replicates threshold for partial replicability guard.",
    )
    parser.add_argument(
        "--partial-bootstrap-replicates",
        type=int,
        help="Override bootstrap replicates count for partial replicability guard support checks.",
    )
    enabled_group = parser.add_mutually_exclusive_group()
    enabled_group.add_argument(
        "--partial-replicability-enabled",
        action="store_true",
        help="Force-enable partial replicability guard for this run.",
    )
    enabled_group.add_argument(
        "--partial-replicability-disabled",
        action="store_true",
        help="Force-disable partial replicability guard for this run.",
    )
    apply_group = parser.add_mutually_exclusive_group()
    apply_group.add_argument(
        "--partial-apply-when-partial-refit",
        action="store_true",
        help="Apply partial replicability guard only when partial refit is used.",
    )
    apply_group.add_argument(
        "--partial-apply-always",
        action="store_true",
        help="Apply partial replicability guard regardless of partial refit use.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    cohorts = _cohorts_from_args(args)

    outputs_dir = _resolve_path(paths_cfg["outputs_dir"], root) / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    all_summary: list[pd.DataFrame] = []
    all_partial: list[pd.DataFrame] = []
    all_decisions: list[pd.DataFrame] = []
    all_transitions: list[pd.DataFrame] = []

    for cohort in cohorts:
        summary, partial, decisions, transitions = _cohort_outputs(
            root=root,
            cohort=cohort,
            paths_cfg=paths_cfg,
            models_cfg=models_cfg,
        )
        summary_path = outputs_dir / f"{cohort}_invariance_summary.csv"
        freed_path = outputs_dir / f"{cohort}_freed_parameters.csv"
        decision_path = outputs_dir / f"{cohort}_invariance_decision.csv"
        transition_path = outputs_dir / f"{cohort}_invariance_transition_checks.csv"
        summary.to_csv(summary_path, index=False)
        partial.to_csv(freed_path, index=False)
        decisions.to_csv(decision_path, index=False)
        transitions.to_csv(transition_path, index=False)
        all_summary.append(summary)
        all_partial.append(partial)
        all_decisions.append(decisions)
        all_transitions.append(transitions)
        print(f"[ok] {cohort}: wrote {summary_path.name} and {freed_path.name}")

    summary_all = pd.concat(all_summary, ignore_index=True)
    partial_all = pd.concat(all_partial, ignore_index=True)
    decisions_all = pd.concat(all_decisions, ignore_index=True)
    transitions_all = pd.concat(all_transitions, ignore_index=True)

    replicability_policy = _partial_replicability_policy(models_cfg)
    enabled_override: bool | None = None
    if args.partial_replicability_enabled:
        enabled_override = True
    elif args.partial_replicability_disabled:
        enabled_override = False
    apply_when_partial_refit_override: bool | None = None
    if args.partial_apply_when_partial_refit:
        apply_when_partial_refit_override = True
    elif args.partial_apply_always:
        apply_when_partial_refit_override = False
    replicability_policy = _apply_partial_replicability_overrides(
        replicability_policy,
        enabled_override=enabled_override,
        apply_when_partial_refit_override=apply_when_partial_refit_override,
        mode_override=args.partial_replicability_mode,
        min_overlap_share_override=args.partial_min_overlap_share,
        min_bootstrap_indicator_share_override=args.partial_min_bootstrap_indicator_share,
        min_bootstrap_success_reps_override=args.partial_min_bootstrap_success_reps,
        bootstrap_replicates_override=args.partial_bootstrap_replicates,
    )
    bootstrap_support, bootstrap_support_details = _build_bootstrap_partial_support(
        root=root,
        paths_cfg=paths_cfg,
        models_cfg=models_cfg,
        decisions=decisions_all,
        partial_rows=partial_all,
        policy=replicability_policy,
    )
    decisions_all, replicability_report = _apply_partial_replicability_guard(
        decisions=decisions_all,
        partial_rows=partial_all,
        policy=replicability_policy,
        bootstrap_support=bootstrap_support,
    )

    summary_all.to_csv(outputs_dir / "invariance_summary_all.csv", index=False)
    partial_all.to_csv(outputs_dir / "freed_parameters_all.csv", index=False)
    decisions_all.to_csv(outputs_dir / "invariance_confirmatory_eligibility.csv", index=False)
    transitions_all.to_csv(outputs_dir / "invariance_transition_checks.csv", index=False)
    replicability_report.to_csv(outputs_dir / "partial_replicability_guard.csv", index=False)
    bootstrap_support.to_csv(outputs_dir / "partial_replicability_bootstrap_support.csv", index=False)
    bootstrap_support_details.to_csv(outputs_dir / "partial_replicability_bootstrap_support_details.csv", index=False)

    for cohort in cohorts:
        cohort_decisions = decisions_all.loc[decisions_all["cohort"].astype(str) == str(cohort)].copy()
        if not cohort_decisions.empty:
            cohort_decisions.to_csv(outputs_dir / f"{cohort}_invariance_decision.csv", index=False)
        cohort_replicability = replicability_report.loc[
            replicability_report["cohort"].astype(str) == str(cohort)
        ].copy()
        if not cohort_replicability.empty:
            cohort_replicability.to_csv(outputs_dir / f"{cohort}_partial_replicability.csv", index=False)
        cohort_bootstrap = bootstrap_support.loc[bootstrap_support["cohort"].astype(str) == str(cohort)].copy()
        if not cohort_bootstrap.empty:
            cohort_bootstrap.to_csv(outputs_dir / f"{cohort}_partial_replicability_bootstrap.csv", index=False)
    print(f"[ok] wrote combined invariance summaries in {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
