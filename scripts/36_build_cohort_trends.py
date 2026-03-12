#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import project_root

COHORT_ORDER = {
    "nlsy79": 0,
    "nlsy97": 1,
    "cnlsy": 2,
}

MEAN_DIFF_CANDIDATES = (
    Path("outputs/tables/g_mean_diff_full_cohort.csv"),
    Path("outputs/tables/g_mean_diff.csv"),
)
VR_TABLE_PATH = Path("outputs/tables/g_variance_ratio.csv")
ANALYSIS_TIERS_PATH = Path("outputs/tables/analysis_tiers.csv")

TREND_COLUMNS = [
    "estimand",
    "metric",
    "scope",
    "status",
    "reason",
    "n_cohorts",
    "cohorts",
    "fit_weighting",
    "slope_per_cohort_step",
    "SE_slope",
    "z_slope",
    "p_value_slope",
    "ci_low_slope",
    "ci_high_slope",
    "delta_first_to_last",
    "ci_low_delta",
    "ci_high_delta",
    "delta_iq_points",
    "source_estimates",
]

PAIRWISE_COLUMNS = [
    "estimand",
    "metric",
    "scope",
    "cohort_a",
    "cohort_b",
    "estimate_a",
    "estimate_b",
    "diff_b_minus_a",
    "SE_diff",
    "z_diff",
    "p_value_diff",
    "ci_low_diff",
    "ci_high_diff",
    "diff_iq_points",
    "source_estimates",
]


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_ORDER.keys())
    return args.cohort


def _resolve_first_existing(root: Path, candidates: tuple[Path, ...]) -> Path | None:
    for rel_path in candidates:
        path = root / rel_path
        if path.exists():
            return path
    return None


def _normalize_analysis_tiers(root: Path) -> pd.DataFrame:
    path = root / ANALYSIS_TIERS_PATH
    if not path.exists():
        return pd.DataFrame(columns=["cohort", "estimand", "analysis_tier"])
    tiers = pd.read_csv(path, low_memory=False)
    needed = {"cohort", "estimand", "analysis_tier"}
    if not needed.issubset(set(tiers.columns)):
        return pd.DataFrame(columns=["cohort", "estimand", "analysis_tier"])
    return tiers[list(needed)].copy()


def _valid_number(value: Any) -> bool:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(num)


def _join_source(series: pd.Series) -> str:
    vals = [str(v) for v in series.dropna().astype(str).unique() if str(v).strip()]
    return ",".join(sorted(vals))


def _trend_row_not_feasible(estimand: str, metric: str, scope: str, reason: str, source_estimates: str) -> dict[str, Any]:
    row = {
        "estimand": estimand,
        "metric": metric,
        "scope": scope,
        "status": "not_feasible",
        "reason": reason,
        "n_cohorts": 0,
        "cohorts": "",
        "fit_weighting": pd.NA,
        "source_estimates": source_estimates,
    }
    for col in TREND_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _fit_trend(estimand: str, metric: str, scope: str, rows: pd.DataFrame) -> dict[str, Any]:
    if rows.shape[0] < 2:
        return _trend_row_not_feasible(
            estimand=estimand,
            metric=metric,
            scope=scope,
            reason="insufficient_cohorts",
            source_estimates=_join_source(rows["source_estimate"]),
        ) | {
            "n_cohorts": int(rows.shape[0]),
            "cohorts": ",".join(rows["cohort"].astype(str).tolist()),
        }

    x = rows["cohort_index"].astype(float).to_numpy()
    y = rows["estimate"].astype(float).to_numpy()
    se = rows["SE"].astype(float).to_numpy()

    if bool(((~pd.Series(se).apply(math.isfinite)) | (pd.Series(se) <= 0.0)).any()):
        w = pd.Series([1.0] * len(rows), dtype="float64").to_numpy()
        fit_weighting = "unweighted_equal"
    else:
        w = (1.0 / (pd.Series(se, dtype="float64") ** 2)).to_numpy()
        fit_weighting = "inverse_variance"

    w_sum = float(w.sum())
    if not math.isfinite(w_sum) or w_sum <= 0.0:
        return _trend_row_not_feasible(
            estimand=estimand,
            metric=metric,
            scope=scope,
            reason="invalid_weights",
            source_estimates=_join_source(rows["source_estimate"]),
        ) | {
            "n_cohorts": int(rows.shape[0]),
            "cohorts": ",".join(rows["cohort"].astype(str).tolist()),
            "fit_weighting": fit_weighting,
        }

    x_bar = float((w * x).sum() / w_sum)
    y_bar = float((w * y).sum() / w_sum)
    sxx = float((w * (x - x_bar) ** 2).sum())
    if not math.isfinite(sxx) or sxx <= 0.0:
        return _trend_row_not_feasible(
            estimand=estimand,
            metric=metric,
            scope=scope,
            reason="degenerate_cohort_index",
            source_estimates=_join_source(rows["source_estimate"]),
        ) | {
            "n_cohorts": int(rows.shape[0]),
            "cohorts": ",".join(rows["cohort"].astype(str).tolist()),
            "fit_weighting": fit_weighting,
        }

    slope = float((w * (x - x_bar) * (y - y_bar)).sum() / sxx)
    se_slope = float(math.sqrt(1.0 / sxx))
    if not math.isfinite(se_slope) or se_slope <= 0.0:
        return _trend_row_not_feasible(
            estimand=estimand,
            metric=metric,
            scope=scope,
            reason="nonpositive_slope_se",
            source_estimates=_join_source(rows["source_estimate"]),
        ) | {
            "n_cohorts": int(rows.shape[0]),
            "cohorts": ",".join(rows["cohort"].astype(str).tolist()),
            "fit_weighting": fit_weighting,
        }

    z_slope = slope / se_slope
    p_value = float(2.0 * norm.sf(abs(z_slope)))
    ci_low = float(slope - 1.96 * se_slope)
    ci_high = float(slope + 1.96 * se_slope)
    x_delta = float(x.max() - x.min())
    delta = float(slope * x_delta)
    se_delta = float(abs(x_delta) * se_slope)
    ci_low_delta = float(delta - 1.96 * se_delta)
    ci_high_delta = float(delta + 1.96 * se_delta)

    row = {
        "estimand": estimand,
        "metric": metric,
        "scope": scope,
        "status": "computed",
        "reason": pd.NA,
        "n_cohorts": int(rows.shape[0]),
        "cohorts": ",".join(rows["cohort"].astype(str).tolist()),
        "fit_weighting": fit_weighting,
        "slope_per_cohort_step": slope,
        "SE_slope": se_slope,
        "z_slope": z_slope,
        "p_value_slope": p_value,
        "ci_low_slope": ci_low,
        "ci_high_slope": ci_high,
        "delta_first_to_last": delta,
        "ci_low_delta": ci_low_delta,
        "ci_high_delta": ci_high_delta,
        "delta_iq_points": float(delta * 15.0) if estimand == "d_g" else pd.NA,
        "source_estimates": _join_source(rows["source_estimate"]),
    }
    for col in TREND_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _build_pairwise_rows(estimand: str, metric: str, scope: str, rows: pd.DataFrame) -> list[dict[str, Any]]:
    pairwise: list[dict[str, Any]] = []
    for idx_a, idx_b in combinations(range(rows.shape[0]), 2):
        row_a = rows.iloc[idx_a]
        row_b = rows.iloc[idx_b]
        est_a = float(row_a["estimate"])
        est_b = float(row_b["estimate"])
        se_a = float(row_a["SE"])
        se_b = float(row_b["SE"])
        se_diff = math.sqrt(max(0.0, se_a * se_a + se_b * se_b))
        if se_diff <= 0.0 or not math.isfinite(se_diff):
            z_val = pd.NA
            p_val = pd.NA
            ci_low = pd.NA
            ci_high = pd.NA
        else:
            diff = est_b - est_a
            z_val = float(diff / se_diff)
            p_val = float(2.0 * norm.sf(abs(z_val)))
            ci_low = float(diff - 1.96 * se_diff)
            ci_high = float(diff + 1.96 * se_diff)

        diff_val = float(est_b - est_a)
        out = {
            "estimand": estimand,
            "metric": metric,
            "scope": scope,
            "cohort_a": str(row_a["cohort"]),
            "cohort_b": str(row_b["cohort"]),
            "estimate_a": est_a,
            "estimate_b": est_b,
            "diff_b_minus_a": diff_val,
            "SE_diff": se_diff,
            "z_diff": z_val,
            "p_value_diff": p_val,
            "ci_low_diff": ci_low,
            "ci_high_diff": ci_high,
            "diff_iq_points": float(diff_val * 15.0) if estimand == "d_g" else pd.NA,
            "source_estimates": _join_source(pd.Series([row_a["source_estimate"], row_b["source_estimate"]])),
        }
        for col in PAIRWISE_COLUMNS:
            out.setdefault(col, pd.NA)
        pairwise.append(out)
    return pairwise


def run_cohort_trends(
    *,
    root: Path,
    cohorts: list[str],
    trend_output_path: Path = Path("outputs/tables/cohort_trends_sex_differences.csv"),
    pairwise_output_path: Path = Path("outputs/tables/cohort_pairwise_diffs_sex_differences.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_tiers = _normalize_analysis_tiers(root)
    requested = [c for c in cohorts if c in COHORT_ORDER]
    requested_set = set(requested)

    estimate_rows: list[dict[str, Any]] = []
    reasons_by_estimand: dict[str, list[str]] = {"d_g": [], "vr_g": []}

    mean_diff_path = _resolve_first_existing(root, MEAN_DIFF_CANDIDATES)
    if mean_diff_path is None:
        reasons_by_estimand["d_g"].append("missing_g_mean_diff_source")
    else:
        mean_df = pd.read_csv(mean_diff_path, low_memory=False)
        needed = {"cohort", "d_g", "SE_d_g"}
        if not needed.issubset(set(mean_df.columns)):
            reasons_by_estimand["d_g"].append("invalid_g_mean_diff_columns")
        else:
            for _, row in mean_df.iterrows():
                cohort = str(row["cohort"])
                if cohort not in requested_set:
                    continue
                if not (_valid_number(row["d_g"]) and _valid_number(row["SE_d_g"])):
                    continue
                estimate_rows.append(
                    {
                        "cohort": cohort,
                        "cohort_index": float(COHORT_ORDER[cohort]),
                        "estimand": "d_g",
                        "metric": "d_g",
                        "estimate": float(row["d_g"]),
                        "SE": float(row["SE_d_g"]),
                        "source_estimate": str(mean_diff_path.relative_to(root)),
                    }
                )
    if not any(r["estimand"] == "d_g" for r in estimate_rows):
        reasons_by_estimand["d_g"].append("no_valid_d_g_rows")

    vr_path = root / VR_TABLE_PATH
    if not vr_path.exists():
        reasons_by_estimand["vr_g"].append("missing_g_variance_ratio_source")
    else:
        vr_df = pd.read_csv(vr_path, low_memory=False)
        needed = {"cohort", "VR_g", "SE_logVR"}
        if not needed.issubset(set(vr_df.columns)):
            reasons_by_estimand["vr_g"].append("invalid_g_variance_ratio_columns")
        else:
            for _, row in vr_df.iterrows():
                cohort = str(row["cohort"])
                if cohort not in requested_set:
                    continue
                if not (_valid_number(row["VR_g"]) and _valid_number(row["SE_logVR"])):
                    continue
                vr = float(row["VR_g"])
                if vr <= 0.0:
                    continue
                estimate_rows.append(
                    {
                        "cohort": cohort,
                        "cohort_index": float(COHORT_ORDER[cohort]),
                        "estimand": "vr_g",
                        "metric": "log_vr_g",
                        "estimate": float(math.log(vr)),
                        "SE": float(row["SE_logVR"]),
                        "source_estimate": str(vr_path.relative_to(root)),
                    }
                )
    if not any(r["estimand"] == "vr_g" for r in estimate_rows):
        reasons_by_estimand["vr_g"].append("no_valid_vr_g_rows")

    estimates = pd.DataFrame(estimate_rows)
    if estimates.empty:
        estimates = pd.DataFrame(
            columns=[
                "cohort",
                "cohort_index",
                "estimand",
                "metric",
                "estimate",
                "SE",
                "source_estimate",
            ]
        )

    if not analysis_tiers.empty and not estimates.empty:
        estimates = estimates.merge(
            analysis_tiers,
            how="left",
            on=["cohort", "estimand"],
        )
    else:
        estimates["analysis_tier"] = pd.NA
    estimates["analysis_tier"] = estimates["analysis_tier"].fillna("unknown")
    if not estimates.empty:
        estimates = estimates.sort_values(["estimand", "cohort_index", "cohort"]).reset_index(drop=True)

    trend_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for estimand, metric in (("d_g", "d_g"), ("vr_g", "log_vr_g")):
        base = estimates[estimates["estimand"] == estimand].copy()
        scope_map = {
            "all_cohorts": base,
            "confirmatory_only": base[base["analysis_tier"] == "confirmatory"].copy(),
        }
        for scope, scoped in scope_map.items():
            if scoped.empty:
                reason = ";".join(sorted(set(reasons_by_estimand.get(estimand, []) or ["no_rows_for_scope"])))
                trend_rows.append(
                    _trend_row_not_feasible(
                        estimand=estimand,
                        metric=metric,
                        scope=scope,
                        reason=reason,
                        source_estimates="",
                    )
                )
                continue
            scoped = scoped.sort_values(["cohort_index", "cohort"]).reset_index(drop=True)
            trend_row = _fit_trend(estimand=estimand, metric=metric, scope=scope, rows=scoped)
            trend_rows.append(trend_row)
            if str(trend_row.get("status")) == "computed":
                pairwise_rows.extend(_build_pairwise_rows(estimand=estimand, metric=metric, scope=scope, rows=scoped))

    trend_df = pd.DataFrame(trend_rows)
    if trend_df.empty:
        trend_df = pd.DataFrame(columns=TREND_COLUMNS)
    for col in TREND_COLUMNS:
        if col not in trend_df.columns:
            trend_df[col] = pd.NA
    trend_df = trend_df[TREND_COLUMNS].copy()

    pairwise_df = pd.DataFrame(pairwise_rows)
    if pairwise_df.empty:
        pairwise_df = pd.DataFrame(columns=PAIRWISE_COLUMNS)
    for col in PAIRWISE_COLUMNS:
        if col not in pairwise_df.columns:
            pairwise_df[col] = pd.NA
    pairwise_df = pairwise_df[PAIRWISE_COLUMNS].copy()

    trend_output = trend_output_path if trend_output_path.is_absolute() else root / trend_output_path
    pairwise_output = pairwise_output_path if pairwise_output_path.is_absolute() else root / pairwise_output_path
    trend_output.parent.mkdir(parents=True, exist_ok=True)
    pairwise_output.parent.mkdir(parents=True, exist_ok=True)
    trend_df.to_csv(trend_output, index=False)
    pairwise_df.to_csv(pairwise_output, index=False)
    return trend_df, pairwise_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build cohort trend tests for latent sex-difference estimands (d_g and log(VR_g))."
    )
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_ORDER), help="Cohort(s) to include.")
    parser.add_argument("--all", action="store_true", help="Include all known cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--trend-output-path",
        type=Path,
        default=Path("outputs/tables/cohort_trends_sex_differences.csv"),
        help="Trend output CSV path (relative to project-root if not absolute).",
    )
    parser.add_argument(
        "--pairwise-output-path",
        type=Path,
        default=Path("outputs/tables/cohort_pairwise_diffs_sex_differences.csv"),
        help="Pairwise output CSV path (relative to project-root if not absolute).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    cohorts = _cohorts_from_args(args)
    try:
        trends, pairwise = run_cohort_trends(
            root=root,
            cohorts=cohorts,
            trend_output_path=args.trend_output_path,
            pairwise_output_path=args.pairwise_output_path,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    trend_output = args.trend_output_path if args.trend_output_path.is_absolute() else root / args.trend_output_path
    pairwise_output = args.pairwise_output_path if args.pairwise_output_path.is_absolute() else root / args.pairwise_output_path
    computed = int((trends["status"] == "computed").sum()) if "status" in trends.columns else 0
    print(f"[ok] wrote {trend_output}")
    print(f"[ok] wrote {pairwise_output}")
    print(f"[ok] computed trend rows: {computed}")
    print(f"[ok] pairwise rows: {len(pairwise)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
