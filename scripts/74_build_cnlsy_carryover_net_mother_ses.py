#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.exploratory import g_proxy, ols_fit
from nls_pipeline.io import load_yaml, project_root

DETAIL_COLUMNS = [
    "cohort",
    "outcome",
    "model_type",
    "model",
    "status",
    "reason",
    "outcome_col",
    "age_col",
    "ses_col",
    "n_total",
    "n_used",
    "n_positive",
    "prevalence",
    "mean_outcome",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "odds_ratio_g",
    "beta_age",
    "SE_beta_age",
    "p_value_beta_age",
    "beta_mother_ses",
    "SE_beta_mother_ses",
    "p_value_beta_mother_ses",
    "r2_or_pseudo_r2",
    "source_data",
]
SUMMARY_COLUMNS = [
    "cohort",
    "outcome",
    "model_type",
    "status",
    "reason",
    "n_baseline",
    "n_mother_ses",
    "beta_g_baseline",
    "beta_g_mother_ses",
    "odds_ratio_g_baseline",
    "odds_ratio_g_mother_ses",
    "attenuation_abs",
    "attenuation_pct",
    "delta_r2_or_pseudo_r2",
    "source_data",
]

OUTCOME_SPECS: tuple[dict[str, str], ...] = (
    {"outcome": "log_wage_income_2014", "outcome_col": "wage_income_2014", "model_type": "ols_log_income"},
    {"outcome": "log_family_income_2014", "outcome_col": "family_income_2014", "model_type": "ols_log_income"},
    {"outcome": "num_current_jobs_2014", "outcome_col": "num_current_jobs_2014", "model_type": "ols_count_like"},
    {"outcome": "degree_any_2014", "outcome_col": "degree_2014", "model_type": "logit_binary"},
)


def _logistic_fit(y: pd.Series, x: pd.DataFrame, *, max_iter: int = 100, tol: float = 1e-8) -> tuple[dict[str, Any] | None, str | None]:
    y_num = pd.to_numeric(y, errors="coerce")
    x_num = x.apply(pd.to_numeric, errors="coerce")
    mask = y_num.notna() & x_num.notna().all(axis=1)
    if int(mask.sum()) == 0:
        return None, "empty_after_numeric_cast"

    yv = y_num[mask].to_numpy(dtype=float)
    xv = x_num[mask].to_numpy(dtype=float)
    n, p = xv.shape
    if n <= p:
        return None, "insufficient_rows_for_model"
    if np.unique(yv).size < 2:
        return None, "single_outcome_class"

    beta = np.zeros(p, dtype=float)
    converged = False
    for _ in range(max_iter):
        eta = np.clip(xv @ beta, -30.0, 30.0)
        p_hat = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p_hat * (1.0 - p_hat), 1e-8, None)
        xtwx = xv.T @ (w[:, None] * xv)
        grad = xv.T @ (yv - p_hat)
        try:
            step = np.linalg.pinv(xtwx) @ grad
        except np.linalg.LinAlgError:
            return None, "logit_xtwx_pinv_failed"
        beta_next = beta + step
        if float(np.max(np.abs(step))) < tol:
            beta = beta_next
            converged = True
            break
        beta = beta_next

    if not converged:
        return None, "logit_failed_to_converge"

    eta = np.clip(xv @ beta, -30.0, 30.0)
    p_hat = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-8, 1.0 - 1.0e-8)
    w = np.clip(p_hat * (1.0 - p_hat), 1e-8, None)
    xtwx = xv.T @ (w[:, None] * xv)
    cov = np.linalg.pinv(xtwx)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_stats = beta / se
    p_vals = np.full(shape=(p,), fill_value=np.nan, dtype=float)
    for i in range(p):
        if math.isfinite(float(z_stats[i])) and math.isfinite(float(se[i])) and float(se[i]) > 0.0:
            p_vals[i] = float(2.0 * norm.sf(abs(float(z_stats[i]))))

    ll_full = float(np.sum(yv * np.log(p_hat) + (1.0 - yv) * np.log(1.0 - p_hat)))
    y_bar = float(np.mean(yv))
    if y_bar <= 0.0 or y_bar >= 1.0:
        pseudo_r2 = float("nan")
    else:
        ll_null = float(np.sum(yv * math.log(y_bar) + (1.0 - yv) * math.log(1.0 - y_bar)))
        pseudo_r2 = float(1.0 - (ll_full / ll_null)) if ll_null != 0.0 else float("nan")

    return {
        "beta": beta,
        "se": se,
        "p": p_vals,
        "pseudo_r2": pseudo_r2,
        "n_used": int(n),
    }, None


def _empty_detail(*, outcome: str, model_type: str, model: str, outcome_col: str, reason: str, source_data: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": "cnlsy",
        "outcome": outcome,
        "model_type": model_type,
        "model": model,
        "status": "not_feasible",
        "reason": reason,
        "outcome_col": outcome_col,
        "age_col": "age_2014",
        "ses_col": "mother_education",
        "source_data": source_data,
    }
    for col in DETAIL_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _empty_summary(*, outcome: str, model_type: str, reason: str, source_data: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": "cnlsy",
        "outcome": outcome,
        "model_type": model_type,
        "status": "not_feasible",
        "reason": reason,
        "source_data": source_data,
    }
    for col in SUMMARY_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _prepare_outcome(df: pd.DataFrame, spec: dict[str, str]) -> pd.Series:
    raw = pd.to_numeric(df[spec["outcome_col"]], errors="coerce")
    if spec["outcome"] == "degree_any_2014":
        return (raw > 0).astype(float)
    if spec["outcome"].startswith("log_"):
        return np.log1p(raw)
    return raw


def run_cnlsy_carryover_net_mother_ses(
    *,
    root: Path,
    detail_output_path: Path = Path("outputs/tables/cnlsy_carryover_net_mother_ses.csv"),
    summary_output_path: Path = Path("outputs/tables/cnlsy_carryover_net_mother_ses_summary.csv"),
    min_n_continuous: int = 60,
    min_class_n_binary: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir
    source_path = processed_dir / "cnlsy_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / "cnlsy_cfa.csv"
    source_data = str(source_path.relative_to(root)) if source_path.exists() else "data/processed/cnlsy_cfa_resid_or_cfa.csv"

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    if not source_path.exists():
        for spec in OUTCOME_SPECS:
            detail_rows.append(_empty_detail(outcome=spec["outcome"], model_type=spec["model_type"], model="baseline", outcome_col=spec["outcome_col"], reason="missing_source_data", source_data=source_data))
            detail_rows.append(_empty_detail(outcome=spec["outcome"], model_type=spec["model_type"], model="mother_ses_controlled", outcome_col=spec["outcome_col"], reason="missing_source_data", source_data=source_data))
            summary_rows.append(_empty_summary(outcome=spec["outcome"], model_type=spec["model_type"], reason="missing_source_data", source_data=source_data))
        detail = pd.DataFrame(detail_rows)[DETAIL_COLUMNS]
        summary = pd.DataFrame(summary_rows)[SUMMARY_COLUMNS]
    else:
        df = pd.read_csv(source_path, low_memory=False)
        required = {"age_2014", "mother_education"}
        missing_base = sorted(col for col in required if col not in df.columns)
        indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", [])]
        df = df.copy()
        try:
            df["__g_proxy"] = g_proxy(df, indicators)
        except Exception:
            reason = "g_proxy_failed"
            for spec in OUTCOME_SPECS:
                detail_rows.append(_empty_detail(outcome=spec["outcome"], model_type=spec["model_type"], model="baseline", outcome_col=spec["outcome_col"], reason=reason, source_data=source_data))
                detail_rows.append(_empty_detail(outcome=spec["outcome"], model_type=spec["model_type"], model="mother_ses_controlled", outcome_col=spec["outcome_col"], reason=reason, source_data=source_data))
                summary_rows.append(_empty_summary(outcome=spec["outcome"], model_type=spec["model_type"], reason=reason, source_data=source_data))
            detail = pd.DataFrame(detail_rows)[DETAIL_COLUMNS]
            summary = pd.DataFrame(summary_rows)[SUMMARY_COLUMNS]
            detail_target = detail_output_path if detail_output_path.is_absolute() else root / detail_output_path
            summary_target = summary_output_path if summary_output_path.is_absolute() else root / summary_output_path
            detail_target.parent.mkdir(parents=True, exist_ok=True)
            summary_target.parent.mkdir(parents=True, exist_ok=True)
            detail.to_csv(detail_target, index=False)
            summary.to_csv(summary_target, index=False)
            return detail, summary
        n_total = int(len(df))

        for spec in OUTCOME_SPECS:
            outcome = spec["outcome"]
            outcome_col = spec["outcome_col"]
            model_type = spec["model_type"]
            if missing_base:
                reason = f"missing_required_columns:{','.join(missing_base)}"
                detail_rows.append(_empty_detail(outcome=outcome, model_type=model_type, model="baseline", outcome_col=outcome_col, reason=reason, source_data=source_data))
                detail_rows.append(_empty_detail(outcome=outcome, model_type=model_type, model="mother_ses_controlled", outcome_col=outcome_col, reason=reason, source_data=source_data))
                summary_rows.append(_empty_summary(outcome=outcome, model_type=model_type, reason=reason, source_data=source_data))
                continue
            if outcome_col not in df.columns:
                reason = "missing_outcome_column"
                detail_rows.append(_empty_detail(outcome=outcome, model_type=model_type, model="baseline", outcome_col=outcome_col, reason=reason, source_data=source_data))
                detail_rows.append(_empty_detail(outcome=outcome, model_type=model_type, model="mother_ses_controlled", outcome_col=outcome_col, reason=reason, source_data=source_data))
                summary_rows.append(_empty_summary(outcome=outcome, model_type=model_type, reason=reason, source_data=source_data))
                continue

            work = pd.DataFrame(
                {
                    "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
                    "age": pd.to_numeric(df["age_2014"], errors="coerce"),
                    "mother_ses": pd.to_numeric(df["mother_education"], errors="coerce"),
                    "outcome": _prepare_outcome(df, spec),
                }
            )
            baseline = work.dropna(subset=["g", "age", "outcome"]).copy()
            controlled = work.dropna(subset=["g", "age", "mother_ses", "outcome"]).copy()

            baseline_row = _empty_detail(outcome=outcome, model_type=model_type, model="baseline", outcome_col=outcome_col, reason="insufficient_rows", source_data=source_data)
            controlled_row = _empty_detail(outcome=outcome, model_type=model_type, model="mother_ses_controlled", outcome_col=outcome_col, reason="insufficient_rows", source_data=source_data)

            if model_type == "logit_binary":
                for row, frame, label in ((baseline_row, baseline, "baseline"), (controlled_row, controlled, "mother_ses_controlled")):
                    values = sorted(set(float(v) for v in frame["outcome"].dropna().tolist()))
                    row["n_total"] = n_total
                    row["n_used"] = int(len(frame))
                    if values != [0.0, 1.0]:
                        row["reason"] = "outcome_not_binary_zero_one"
                        continue
                    n_positive = int(frame["outcome"].eq(1.0).sum())
                    n_negative = int(frame["outcome"].eq(0.0).sum())
                    row["n_positive"] = n_positive
                    row["prevalence"] = float(n_positive / len(frame)) if len(frame) > 0 else pd.NA
                    row["mean_outcome"] = float(frame["outcome"].mean()) if len(frame) > 0 else pd.NA
                    if n_positive < min_class_n_binary or n_negative < min_class_n_binary:
                        row["reason"] = "insufficient_class_counts"
                        continue
                    x_data = {"intercept": 1.0, "g": frame["g"], "age": frame["age"]}
                    if label == "mother_ses_controlled":
                        x_data["mother_ses"] = frame["mother_ses"]
                    x = pd.DataFrame(x_data, index=frame.index)
                    fit, reason = _logistic_fit(frame["outcome"], x)
                    if fit is None:
                        row["reason"] = f"logit_failed:{reason or 'unknown'}"
                        continue
                    row.update(
                        {
                            "status": "computed",
                            "reason": pd.NA,
                            "n_used": int(fit["n_used"]),
                            "beta_g": float(fit["beta"][1]),
                            "SE_beta_g": float(fit["se"][1]),
                            "p_value_beta_g": float(fit["p"][1]) if math.isfinite(float(fit["p"][1])) else pd.NA,
                            "odds_ratio_g": float(math.exp(float(fit["beta"][1]))),
                            "beta_age": float(fit["beta"][2]),
                            "SE_beta_age": float(fit["se"][2]),
                            "p_value_beta_age": float(fit["p"][2]) if math.isfinite(float(fit["p"][2])) else pd.NA,
                            "r2_or_pseudo_r2": float(fit["pseudo_r2"]),
                        }
                    )
                    if label == "mother_ses_controlled":
                        row["beta_mother_ses"] = float(fit["beta"][3])
                        row["SE_beta_mother_ses"] = float(fit["se"][3])
                        row["p_value_beta_mother_ses"] = float(fit["p"][3]) if math.isfinite(float(fit["p"][3])) else pd.NA
            else:
                for row, frame, label in ((baseline_row, baseline, "baseline"), (controlled_row, controlled, "mother_ses_controlled")):
                    row["n_total"] = n_total
                    row["n_used"] = int(len(frame))
                    row["mean_outcome"] = float(frame["outcome"].mean()) if len(frame) > 0 else pd.NA
                    if len(frame) < min_n_continuous:
                        row["reason"] = "insufficient_rows"
                        continue
                    x_data = {"intercept": 1.0, "g": frame["g"], "age": frame["age"]}
                    if label == "mother_ses_controlled":
                        x_data["mother_ses"] = frame["mother_ses"]
                    x = pd.DataFrame(x_data, index=frame.index)
                    fit, reason = ols_fit(frame["outcome"], x)
                    if fit is None:
                        row["reason"] = f"ols_failed:{reason or 'unknown'}"
                        continue
                    row.update(
                        {
                            "status": "computed",
                            "reason": pd.NA,
                            "n_used": int(fit["n_used"]),
                            "beta_g": float(fit["beta"][1]),
                            "SE_beta_g": float(fit["se"][1]),
                            "p_value_beta_g": float(fit["p"][1]) if math.isfinite(float(fit["p"][1])) else pd.NA,
                            "beta_age": float(fit["beta"][2]),
                            "SE_beta_age": float(fit["se"][2]),
                            "p_value_beta_age": float(fit["p"][2]) if math.isfinite(float(fit["p"][2])) else pd.NA,
                            "r2_or_pseudo_r2": float(fit["r2"]),
                        }
                    )
                    if label == "mother_ses_controlled":
                        row["beta_mother_ses"] = float(fit["beta"][3])
                        row["SE_beta_mother_ses"] = float(fit["se"][3])
                        row["p_value_beta_mother_ses"] = float(fit["p"][3]) if math.isfinite(float(fit["p"][3])) else pd.NA

            for col in DETAIL_COLUMNS:
                baseline_row.setdefault(col, pd.NA)
                controlled_row.setdefault(col, pd.NA)
            detail_rows.extend([baseline_row, controlled_row])

            summary = _empty_summary(outcome=outcome, model_type=model_type, reason="model_pair_not_computed", source_data=source_data)
            if baseline_row.get("status") == "computed" and controlled_row.get("status") == "computed":
                beta_base = float(baseline_row["beta_g"])
                beta_ctrl = float(controlled_row["beta_g"])
                attenuation_abs = beta_base - beta_ctrl
                attenuation_pct = (attenuation_abs / beta_base * 100.0) if beta_base != 0 else pd.NA
                summary.update(
                    {
                        "status": "computed",
                        "reason": pd.NA,
                        "n_baseline": int(baseline_row["n_used"]),
                        "n_mother_ses": int(controlled_row["n_used"]),
                        "beta_g_baseline": beta_base,
                        "beta_g_mother_ses": beta_ctrl,
                        "odds_ratio_g_baseline": baseline_row.get("odds_ratio_g", pd.NA),
                        "odds_ratio_g_mother_ses": controlled_row.get("odds_ratio_g", pd.NA),
                        "attenuation_abs": attenuation_abs,
                        "attenuation_pct": attenuation_pct,
                        "delta_r2_or_pseudo_r2": float(controlled_row["r2_or_pseudo_r2"]) - float(baseline_row["r2_or_pseudo_r2"]),
                    }
                )
            else:
                summary["reason"] = f"baseline={baseline_row.get('reason', 'na')};mother_ses={controlled_row.get('reason', 'na')}"
            for col in SUMMARY_COLUMNS:
                summary.setdefault(col, pd.NA)
            summary_rows.append(summary)

        detail = pd.DataFrame(detail_rows)
        summary = pd.DataFrame(summary_rows)
        for col in DETAIL_COLUMNS:
            if col not in detail.columns:
                detail[col] = pd.NA
        for col in SUMMARY_COLUMNS:
            if col not in summary.columns:
                summary[col] = pd.NA
        detail = detail[DETAIL_COLUMNS].copy()
        summary = summary[SUMMARY_COLUMNS].copy()

    detail_target = detail_output_path if detail_output_path.is_absolute() else root / detail_output_path
    summary_target = summary_output_path if summary_output_path.is_absolute() else root / summary_output_path
    detail_target.parent.mkdir(parents=True, exist_ok=True)
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_target, index=False)
    summary.to_csv(summary_target, index=False)
    return detail, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CNLSY carryover models net of mother SES.")
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--detail-output-path", type=Path, default=Path("outputs/tables/cnlsy_carryover_net_mother_ses.csv"))
    parser.add_argument("--summary-output-path", type=Path, default=Path("outputs/tables/cnlsy_carryover_net_mother_ses_summary.csv"))
    parser.add_argument("--min-n-continuous", type=int, default=60)
    parser.add_argument("--min-class-n-binary", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        detail, summary = run_cnlsy_carryover_net_mother_ses(
            root=root,
            detail_output_path=args.detail_output_path,
            summary_output_path=args.summary_output_path,
            min_n_continuous=int(args.min_n_continuous),
            min_class_n_binary=int(args.min_class_n_binary),
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.detail_output_path if args.detail_output_path.is_absolute() else root / args.detail_output_path}")
    print(f"[ok] wrote {args.summary_output_path if args.summary_output_path.is_absolute() else root / args.summary_output_path}")
    print(f"[ok] computed detail rows: {int((detail['status'] == 'computed').sum()) if 'status' in detail.columns else 0}")
    print(f"[ok] computed summary rows: {int((summary['status'] == 'computed').sum()) if 'status' in summary.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
