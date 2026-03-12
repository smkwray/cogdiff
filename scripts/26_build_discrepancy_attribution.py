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

from nls_pipeline.io import project_root

OUTPUT_PATH = Path("outputs/tables/discrepancy_attribution_matrix.csv")
OUTPUT_COLUMNS: tuple[str, ...] = (
    "cohort",
    "claim_id",
    "metric",
    "baseline_estimate",
    "comparison_estimate",
    "delta",
    "likely_cause_bucket",
    "diagnostic_required",
    "verdict_hint",
)


def _safe_float(value: object) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    number = float(number)
    if not math.isfinite(number):
        return None
    return number


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _cohort_value(value: object) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return value


def _cohort_row(df: pd.DataFrame, cohort: str, metric_column: str | None = None, metric_value: str | None = None) -> pd.Series | None:
    if df.empty or "cohort" not in df.columns:
        return None
    cohort_rows = df[df["cohort"].astype(str).str.strip() == cohort]
    if cohort_rows.empty:
        return None
    if metric_column is None or metric_value is None or metric_column not in cohort_rows.columns:
        return cohort_rows.iloc[0]
    matches = cohort_rows[cohort_rows[metric_column].astype(str).str.strip().str.lower() == metric_value]
    if not matches.empty:
        return matches.iloc[0]
    return cohort_rows.iloc[0]


def _metric_key_to_metric(metric: str) -> str:
    metric = str(metric).strip().lower()
    if metric in {"vr_g", "log_vr", "log_vr_g"}:
        return "log_vr_g"
    if metric in {"d_g", "mean_diff", "d"}:
        return "d_g"
    return ""


def _read_metric_map(
    root: Path,
    table: str,
    primary_columns: tuple[str, ...],
    fallback_columns: tuple[str, ...] = (),
    transform_log: bool = False,
) -> dict[str, float]:
    df = _safe_read_csv(root / table)
    if df.empty or "cohort" not in df.columns:
        return {}

    source_col: str | None = None
    direct = True
    for candidate in primary_columns:
        if candidate in df.columns:
            source_col = candidate
            direct = True
            break
    if source_col is None:
        for candidate in fallback_columns:
            if candidate in df.columns:
                source_col = candidate
                direct = False
                break
    if source_col is None:
        return {}

    out: dict[str, float] = {}
    for _, row in df.iterrows():
        cohort = _cohort_value(row.get("cohort"))
        if not cohort:
            continue
        value = _safe_float(row.get(source_col))
        if value is None:
            continue
        if not direct and transform_log:
            if value <= 0.0:
                continue
            value = math.log(value)
        if value is None or not math.isfinite(value):
            continue
        out[cohort] = value
    return out


def _spec_guide_value(row: pd.Series | None, metric: str) -> str:
    if row is None:
        return ""
    for col in ("weight_concordance_reason", "weight_weighted_status", "weight_unweighted_status", "reason", "blocked_reason", "note"):
        if col in row and not pd.isna(row[col]):
            text = str(row[col]).lower()
            if text.strip():
                return text
    block_col = f"blocked_confirmatory_{metric if metric != 'log_vr_g' else 'vr_g'}"
    if block_col in row and pd.notna(row[block_col]):
        return str(row[block_col]).lower()
    return ""


def _tier_value(row: pd.Series | None) -> str:
    if row is None:
        return ""
    for col in ("analysis_tier", "tier", "status"):
        if col in row and pd.notna(row[col]):
            return str(row[col]).strip().lower()
    return ""


def _exclusion_rows_for_cohort(exclusions: pd.DataFrame, cohort: str) -> pd.Series | None:
    if exclusions.empty or "cohort" not in exclusions.columns:
        return None
    rows = exclusions[exclusions["cohort"].astype(str).str.strip() == cohort]
    if rows.empty:
        return None
    return rows.iloc[0]


def _exclusion_reason(row: pd.Series | None, metric: str) -> str:
    if row is None:
        return ""
    metric_key = "vr_g" if metric == "log_vr_g" else "d_g"
    for col in (
        f"reason_{metric_key}",
        "reason",
        f"blocked_reason_{metric_key}",
        "blocked_reason",
    ):
        if col in row and not pd.isna(row[col]):
            text = str(row[col]).strip().lower()
            if text:
                return text
    blocked = row.get(f"blocked_confirmatory_{metric_key}") if f"blocked_confirmatory_{metric_key}" in row else pd.NA
    if pd.notna(blocked) and str(blocked).strip().lower() in {"true", "1"}:
        return "blocked_confirmatory"
    return ""


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    text = text.lower()
    return any(term in text for term in terms)


def _cause_bucket(metric: str, spec_row: pd.Series | None, tiers_row: pd.Series | None, exclusion_row: pd.Series | None) -> str:
    guide = _spec_guide_value(spec_row, metric)
    exclusion_reason = _exclusion_reason(exclusion_row, metric)
    tier_value = _tier_value(tiers_row)

    if _contains_any(guide + " " + exclusion_reason, ("weight", "weighted", "re-weight", "cluster")):
        return "clustering/weighting/inference method"
    if _contains_any(guide + " " + exclusion_reason, ("invariance", "identification", "model_step", "scalar")):
        return "invariance/identification constraints"
    if _contains_any(guide + " " + exclusion_reason + " " + tier_value, ("warning", "quality", "blocked", "guard", "eligible", "exclude", "information")):
        return "warning/quality gating and exclusion rules"
    if metric == "d_g":
        return "sample definition"
    return "data construction/harmonization"


def _diagnostic_for_cause(cause: str, metric: str) -> str:
    if cause == "clustering/weighting/inference method":
        return (
            f"Recompute baseline vs weighted {metric} with full manifest, and inspect effective-N/weight positivity diagnostics."
        )
    if cause == "invariance/identification constraints":
        return "Re-run invariance and identification checks, then compare partial-refit and metric-specific constraints."
    if cause == "warning/quality gating and exclusion rules":
        return (
            "Review confirmatory and robustness gating logs (analysis_tiers, confirmatory_exclusions, specification_stability_summary)."
        )
    if cause == "sample definition":
        return "Audit harmonization, recoding, and subgroup sample inclusion logic for the affected cohorts."
    return "Compare preprocessing and harmonization diagnostics for subgroup consistency and missingness."


def _verdict_hint(delta: float | None, cause: str, cohort: str, metric: str) -> str:
    if delta is None or not math.isfinite(delta):
        return f"Inconclusive for {cohort} / {metric}: comparison estimate unavailable."
    if abs(delta) <= 0.05:
        return "Small discrepancy; likely non-material."
    if cause == "clustering/weighting/inference method":
        return "Material discrepancy likely tied to weighting or inference implementation."
    if cause == "invariance/identification constraints":
        return "Material discrepancy likely tied to invariance or identification constraints."
    if cause == "warning/quality gating and exclusion rules":
        return "Material discrepancy likely tied to quality filters; treat as provisional."
    return "Material discrepancy requiring focused diagnostics."


def _build_context_rows(
    root: Path,
    metric_map: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, set[str]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = _safe_read_csv(root / "outputs/tables/specification_stability_summary.csv")
    tiers = _safe_read_csv(root / "outputs/tables/analysis_tiers.csv")
    exclusions = _safe_read_csv(root / "outputs/tables/confirmatory_exclusions.csv")

    cohorts: dict[str, set[str]] = {}
    for metric, cohort_values in metric_map.items():
        for cohort in cohort_values:
            cohorts.setdefault(cohort, set()).add(metric)

    for _, row in spec.iterrows():
        cohort = _cohort_value(row.get("cohort"))
        if not cohort:
            continue
        metric = _metric_key_to_metric(row.get("estimand")) or _metric_key_to_metric(row.get("metric"))
        if metric:
            cohorts.setdefault(cohort, set()).add(metric)

    return metric_map, cohorts, spec, tiers, exclusions


def _cohort_metric_from_context(row: pd.Series, cohort: str) -> set[str]:
    metrics: set[str] = set()
    for key, metric in (("d_g", "d_g"), ("vr_g", "log_vr_g"), ("log_vr_g", "log_vr_g"), ("mean_diff", "d_g")):
        blocked_col = f"blocked_confirmatory_{key}"
        reason_col = f"reason_{key}"
        if blocked_col in row or reason_col in row:
            metrics.add(metric)
    return metrics


def _build_discrepancy_rows(root: Path) -> list[dict[str, Any]]:
    baseline_map = {
        "d_g": _read_metric_map(
            root,
            table="outputs/tables/g_mean_diff.csv",
            primary_columns=("d_g",),
        ),
        "log_vr_g": _read_metric_map(
            root,
            table="outputs/tables/g_variance_ratio.csv",
            primary_columns=("log_vr_g",),
            fallback_columns=("VR_g", "VR", "vr",),
            transform_log=True,
        ),
    }
    comparison_map = {
        "d_g": _read_metric_map(
            root,
            table="outputs/tables/g_mean_diff_weighted.csv",
            primary_columns=("d_g",),
        ),
        "log_vr_g": _read_metric_map(
            root,
            table="outputs/tables/g_variance_ratio_weighted.csv",
            primary_columns=("log_vr_g",),
            fallback_columns=("VR_g", "VR", "vr"),
            transform_log=True,
        ),
    }

    _, cohorts_by_metric, spec, tiers, exclusions = _build_context_rows(root, baseline_map)
    _, temp_map, _, _, _ = _build_context_rows(root, comparison_map)
    cohorts_by_metric.update(temp_map)

    for _, row in tiers.iterrows():
        cohort = _cohort_value(row.get("cohort"))
        if not cohort:
            continue
        metric = _metric_key_to_metric(row.get("estimand")) or _metric_key_to_metric(row.get("metric"))
        if metric:
            cohorts_by_metric.setdefault(cohort, set()).add(metric)
        for extra in _cohort_metric_from_context(row, cohort):
            cohorts_by_metric.setdefault(cohort, set()).add(extra)

    for _, row in spec.iterrows():
        cohort = _cohort_value(row.get("cohort"))
        if not cohort:
            continue
        metric = _metric_key_to_metric(row.get("estimand")) or _metric_key_to_metric(row.get("metric"))
        if metric:
            cohorts_by_metric.setdefault(cohort, set()).add(metric)
        reason = _spec_guide_value(row, "d_g")
        if "vr" in reason and "vr_g" in reason:
            cohorts_by_metric.setdefault(cohort, set()).add("log_vr_g")
        if "d_g" in reason:
            cohorts_by_metric.setdefault(cohort, set()).add("d_g")

    for _, row in exclusions.iterrows():
        cohort = _cohort_value(row.get("cohort"))
        if not cohort:
            continue
        for metric in _cohort_metric_from_context(row, cohort):
            cohorts_by_metric.setdefault(cohort, set()).add(metric)

    rows: list[dict[str, Any]] = []
    for cohort in sorted(cohorts_by_metric):
        for metric in sorted(cohorts_by_metric[cohort]):
            baseline_estimate = baseline_map.get(metric, {}).get(cohort)
            comparison_estimate = comparison_map.get(metric, {}).get(cohort)
            delta: float | None
            if baseline_estimate is None or comparison_estimate is None:
                delta = None
            else:
                delta = comparison_estimate - baseline_estimate

            spec_row = _cohort_row(
                spec,
                cohort,
                metric_column="estimand",
                metric_value=metric if metric != "log_vr_g" else "vr_g",
            )
            if spec_row is None and metric == "log_vr_g":
                spec_row = _cohort_row(spec, cohort, metric_column="estimand", metric_value="vr_g")
                if spec_row is None:
                    spec_row = _cohort_row(spec, cohort, metric_column="estimand", metric_value="log_vr_g")

            tier_row = _cohort_row(
                tiers,
                cohort,
                metric_column="estimand" if "estimand" in tiers.columns else None,
                metric_value=metric if metric != "log_vr_g" else "vr_g",
            )
            exclusion_row = _exclusion_rows_for_cohort(exclusions, cohort)

            cause = _cause_bucket(metric, spec_row, tier_row, exclusion_row)
            diagnostic = _diagnostic_for_cause(cause, metric)
            verdict = _verdict_hint(delta, cause, cohort, metric)

            rows.append(
                {
                    "cohort": cohort,
                    "claim_id": f"{cohort}:{metric}:unweighted_vs_weighted",
                    "metric": metric,
                    "baseline_estimate": baseline_estimate,
                    "comparison_estimate": comparison_estimate,
                    "delta": delta,
                    "likely_cause_bucket": cause,
                    "diagnostic_required": diagnostic,
                    "verdict_hint": verdict,
                }
            )

    return rows


def build_discrepancy_attribution_matrix(root: Path) -> pd.DataFrame:
    rows = _build_discrepancy_rows(root)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build discrepancy attribution scaffold matrix.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Relative output path for matrix CSV.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    matrix = build_discrepancy_attribution_matrix(root)
    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
