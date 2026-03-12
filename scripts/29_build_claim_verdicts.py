#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

OUTPUT_PATH = Path("outputs/tables/claim_verdicts.csv")
DEFAULT_SPEC_PATH = Path("outputs/tables/specification_stability_summary.csv")
DEFAULT_TIERS_PATH = Path("outputs/tables/analysis_tiers.csv")
DEFAULT_DISCREPANCY_PATH = Path("outputs/tables/discrepancy_attribution_matrix.csv")
OUTPUT_COLUMNS: tuple[str, ...] = (
    "claim_id",
    "claim_label",
    "verdict",
    "reason",
    "cohorts_evaluated",
    "thresholds_used",
)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_float(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    number = float(parsed)
    if not math.isfinite(number):
        return None
    return number


def _safe_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return None


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _resolve_path(root: Path, value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _normalize_metric(value: Any) -> str:
    if pd.isna(value):
        return ""
    metric = str(value).strip().lower().replace(" ", "_")
    if metric in {"d_g", "mean_diff", "mean_diff_gender", "d"}:
        return "d_g"
    if metric in {"vr_g", "log_vr_g", "log_vr", "variance_ratio", "vr", "logvr_g", "logvr"}:
        return "log_vr_g"
    return metric


def _claim_id(cohort: str, metric: str) -> str:
    return f"{cohort}:{metric}:unweighted_vs_weighted"


def _build_reason(field_rows: dict[str, Any]) -> str:
    reasons = []
    spec_row = field_rows.get("spec")
    discrepancy_row = field_rows.get("discrepancy")

    if not _safe_bool(spec_row.get("robust_claim_eligible") if isinstance(spec_row, dict) else None):
        reasons.append("robust_claim_eligible=false")

    if discrepancy_row is not None:
        for key in ("likely_cause_bucket", "verdict_hint", "likely_cause"):
            value = _safe_text(discrepancy_row.get(key))
            if value:
                reasons.append(f"discrepancy:{value}")
                break
        delta = _safe_float(discrepancy_row.get("delta"))
        if delta is not None:
            reasons.append(f"delta={delta}")
    if spec_row is not None:
        for key in ("weight_concordance_reason", "blocked_reason", "reason", "weight_unweighted_reason", "weight_weighted_reason"):
            value = _safe_text(spec_row.get(key))
            if value:
                reasons.append(value)
                break
    if not reasons:
        reasons.append("no_failure_mode_recorded")
    return "; ".join(reasons)


def _thresholds_used(spec_row: pd.Series | None) -> dict[str, float | str | None]:
    if spec_row is None:
        return {}

    thresholds: dict[str, float | str | None] = {}
    for key in ("weight_diff_threshold", "primary_deviation_threshold"):
        value = _safe_float(spec_row.get(key))
        if value is not None:
            thresholds[key] = value

    if not thresholds:
        for fallback in ("d_g_abs_diff_max", "vr_g_log_diff_max"):
            value = _safe_float(spec_row.get(fallback))
            if value is not None:
                thresholds[fallback] = value

    return thresholds


def _row_for_cohort_metric(
    frame: pd.DataFrame,
    cohort: str,
    metric: str,
    metric_columns: tuple[str, ...],
) -> pd.Series | None:
    if frame.empty or "cohort" not in frame.columns:
        return None

    cohort_filter = frame["cohort"].astype(str).str.strip() == cohort
    subset = frame.loc[cohort_filter]
    if subset.empty:
        return None

    for column in metric_columns:
        if column not in subset.columns:
            continue
        normalized = subset[column].map(_normalize_metric)
        mask = normalized == metric
        if mask.any():
            return subset.loc[mask].iloc[0]

    return subset.iloc[0]


def _collect_claim_rows(
    spec: pd.DataFrame,
    tiers: pd.DataFrame,
    discrepancy: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    claims: dict[str, dict[str, Any]] = {}

    def _cohort_row(value: Any) -> str:
        return _safe_text(value)

    def _upsert(cohort: str, metric: str) -> str | None:
        metric = _normalize_metric(metric)
        if not cohort or not metric:
            return None
        claim = _claim_id(cohort, metric)
        claims.setdefault(claim, {"cohort": cohort, "metric": metric, "spec": None, "tiers": None, "discrepancy": None})
        return claim

    for _, row in spec.iterrows():
        cohort = _cohort_row(row.get("cohort"))
        metric = _normalize_metric(row.get("estimand") or row.get("metric"))
        claim = _upsert(cohort, metric)
        if claim is None:
            continue
        claims[claim]["spec"] = row

    for _, row in tiers.iterrows():
        cohort = _cohort_row(row.get("cohort"))
        metric = _normalize_metric(row.get("estimand") or row.get("metric"))
        claim = _upsert(cohort, metric)
        if claim is None:
            continue
        claims[claim]["tiers"] = row

    for _, row in discrepancy.iterrows():
        cohort = _cohort_row(row.get("cohort"))
        metric = _normalize_metric(row.get("metric"))
        claim = _upsert(cohort, metric)
        if claim is None:
            continue
        if claims[claim]["discrepancy"] is None:
            claims[claim]["discrepancy"] = row

    return claims


def _cohorts_to_evaluated(claim_rows: list[dict[str, Any]]) -> str:
    cohorts = sorted({_safe_text(row.get("cohort")) for row in claim_rows if _safe_text(row.get("cohort"))})
    return ";".join(cohorts)


def _evaluate_verdict(cohort: str, metric: str, rows: dict[str, Any]) -> tuple[str, str, dict[str, float | str | None]]:
    spec_row = rows.get("spec")
    tiers_row = rows.get("tiers")
    discrepancy_row = rows.get("discrepancy")

    missing = []
    if spec_row is None:
        missing.append("specification_stability_summary")
    if tiers_row is None:
        missing.append("analysis_tiers")
    if discrepancy_row is None:
        missing.append("discrepancy_attribution_matrix")

    threshold_map = _thresholds_used(spec_row if isinstance(spec_row, pd.Series) else None)
    thresholds_used = json.dumps(threshold_map, sort_keys=True)
    threshold_value = _safe_float(spec_row.get("weight_diff_threshold") if spec_row is not None else None)
    delta = _safe_float(discrepancy_row.get("delta") if discrepancy_row is not None else None)

    if missing:
        reason = f"missing_artifact: {', '.join(sorted(missing))}"
        return "inconclusive", reason, json.loads(thresholds_used)

    robust_claim_eligible = _safe_bool(spec_row.get("robust_claim_eligible"))
    if robust_claim_eligible is None:
        return "inconclusive", "robust_claim_eligible_unavailable", json.loads(thresholds_used)

    blocked = _safe_bool(tiers_row.get("blocked_confirmatory")) if tiers_row is not None else False
    if blocked:
        return "not_confirmed", f"blocked_confirmatory: {cohort}:{metric}", json.loads(thresholds_used)

    if robust_claim_eligible:
        if threshold_value is None or delta is None:
            if delta is None:
                return (
                    "inconclusive",
                    f"missing_discrepancy_delta_for_threshold_check: {cohort}:{metric}",
                    json.loads(thresholds_used),
                )
            return "confirmed", f"robust_claim_eligible=true: {cohort}:{metric}", json.loads(thresholds_used)

        if abs(delta) <= threshold_value:
            return "confirmed", f"robust_claim_eligible=true_and_delta_within_threshold: {cohort}:{metric}", json.loads(
                thresholds_used,
            )
        return "not_confirmed", f"delta_exceeds_threshold_{threshold_value}: {cohort}:{metric}", json.loads(thresholds_used)

    return "not_confirmed", f"robust_claim_eligible=false: {cohort}:{metric}", json.loads(thresholds_used)


def build_claim_verdicts(
    *,
    project_root_path: Path,
    specification_stability_path: Path = DEFAULT_SPEC_PATH,
    analysis_tiers_path: Path = DEFAULT_TIERS_PATH,
    discrepancy_matrix_path: Path = DEFAULT_DISCREPANCY_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    root = Path(project_root_path).resolve()
    spec_path = _resolve_path(root, specification_stability_path)
    tiers_path = _resolve_path(root, analysis_tiers_path)
    discrepancy_path = _resolve_path(root, discrepancy_matrix_path)
    output_file = _resolve_path(root, output_path)

    spec = _safe_read_csv(spec_path)
    tiers = _safe_read_csv(tiers_path)
    discrepancy = _safe_read_csv(discrepancy_path)

    claim_rows = _collect_claim_rows(spec, tiers, discrepancy)
    rows: list[dict[str, Any]] = []

    for claim_id in sorted(claim_rows):
        claim_group = claim_rows[claim_id]
        cohort = claim_group.get("cohort", "")
        metric = claim_group.get("metric", "")
        verdict, reason, thresholds_used = _evaluate_verdict(cohort, metric, claim_group)
        thresholds_text = json.dumps(thresholds_used, sort_keys=True)
        row = {
            "claim_id": claim_id,
            "claim_label": f"claim:{claim_id}",
            "verdict": verdict,
            "reason": reason,
            "cohorts_evaluated": _cohorts_to_evaluated([claim_group]),
            "thresholds_used": thresholds_text,
        }

        rows.append(row)

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output.sort_values("claim_id", kind="mergesort", inplace=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic claim verdict contract artifact.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--specification-stability-summary",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help="Path to specification_stability_summary.csv (relative to project root).",
    )
    parser.add_argument(
        "--analysis-tiers",
        type=Path,
        default=DEFAULT_TIERS_PATH,
        help="Path to analysis_tiers.csv (relative to project root).",
    )
    parser.add_argument(
        "--discrepancy-attribution-matrix",
        type=Path,
        default=DEFAULT_DISCREPANCY_PATH,
        help="Path to discrepancy_attribution_matrix.csv (relative to project root).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output path for claim_verdicts.csv (relative to project root).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    build_claim_verdicts(
        project_root_path=root,
        specification_stability_path=args.specification_stability_summary,
        analysis_tiers_path=args.analysis_tiers,
        discrepancy_matrix_path=args.discrepancy_attribution_matrix,
        output_path=args.output,
    )

    print(f"[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
