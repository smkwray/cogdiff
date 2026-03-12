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

from nls_pipeline.exploratory import g_proxy
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT_SPECS: dict[str, dict[str, Any]] = {
    "nlsy79": {
        "degree_col": "highest_degree_ever",
        "age_col": None,
        "ba_codes": {3, 4, 5, 6, 7},
        "graduate_codes": {5, 6, 7},
        "label_map": {
            1: "hs_or_equivalent",
            2: "associate",
            3: "ba",
            4: "bs",
            5: "masters",
            6: "doctorate",
            7: "professional",
            8: "other",
        },
    },
    "nlsy97": {
        "degree_col": "degree_2021",
        "age_col": "age_2021",
        "ba_codes": {5, 6, 7, 8},
        "graduate_codes": {6, 7, 8},
        "label_map": {
            1: "none",
            2: "ged",
            3: "hs_diploma",
            4: "associate",
            5: "ba_bs",
            6: "masters",
            7: "phd",
            8: "professional",
        },
    },
}

THRESHOLDS = ("ba_or_more_explicit", "graduate_or_more_explicit")

OUTPUT_COLUMNS = [
    "cohort",
    "degree_col",
    "age_col",
    "threshold",
    "status",
    "reason",
    "n_total",
    "n_used",
    "n_positive",
    "prevalence",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "odds_ratio_g",
    "beta_age",
    "SE_beta_age",
    "p_value_beta_age",
    "pseudo_r2",
    "top_degree_code",
    "top_degree_label",
    "source_data",
]


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_SPECS.keys())
    return args.cohort


def _empty_row(
    cohort: str,
    threshold: str,
    *,
    degree_col: str,
    age_col: str | None,
    reason: str,
    source_data: str,
    n_total: int = 0,
    n_used: int = 0,
    n_positive: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "degree_col": degree_col,
        "age_col": age_col or pd.NA,
        "threshold": threshold,
        "status": "not_feasible",
        "reason": reason,
        "n_total": int(n_total),
        "n_used": int(n_used),
        "n_positive": int(n_positive),
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


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

    return {"beta": beta, "se": se, "p": p_vals, "pseudo_r2": pseudo_r2, "n_used": int(n)}, None


def run_explicit_degree_outcomes(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path = Path("outputs/tables/explicit_degree_outcomes.csv"),
    min_class_n: int = 20,
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        spec = COHORT_SPECS[cohort]
        degree_col = str(spec["degree_col"])
        age_col = str(spec["age_col"]) if spec["age_col"] is not None else ""
        label_map = dict(spec["label_map"])
        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{cohort}_cfa_resid_or_cfa.csv"
        if not source_path.exists():
            for threshold in THRESHOLDS:
                rows.append(_empty_row(cohort, threshold, degree_col=degree_col, age_col=age_col, reason="missing_source_data", source_data=source_data))
            continue

        df = pd.read_csv(source_path, low_memory=False)
        required_cols = [degree_col] + ([age_col] if age_col else [])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            reason = f"missing_required_columns:{','.join(missing_cols)}"
            for threshold in THRESHOLDS:
                rows.append(_empty_row(cohort, threshold, degree_col=degree_col, age_col=age_col, reason=reason, source_data=source_data, n_total=len(df)))
            continue

        indicators = hierarchical_subtests(models_cfg)
        df = df.copy()
        df["__g_proxy"] = g_proxy(df, indicators)
        work = pd.DataFrame(
            {
                "degree": pd.to_numeric(df[degree_col], errors="coerce"),
                "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
            }
        )
        if age_col:
            work["age"] = pd.to_numeric(df[age_col], errors="coerce")
        work = work.dropna()
        if work.empty:
            for threshold in THRESHOLDS:
                rows.append(_empty_row(cohort, threshold, degree_col=degree_col, age_col=age_col, reason="no_valid_rows_after_cleaning", source_data=source_data, n_total=len(df)))
            continue

        degree_counts = work["degree"].astype(int).value_counts()
        top_degree_code = int(degree_counts.index[0]) if not degree_counts.empty else pd.NA
        top_degree_label = label_map.get(top_degree_code, "unknown") if top_degree_code is not pd.NA else pd.NA

        for threshold in THRESHOLDS:
            qualifying_codes = spec["ba_codes"] if threshold == "ba_or_more_explicit" else spec["graduate_codes"]
            outcome = work["degree"].isin(set(int(x) for x in qualifying_codes)).astype(float)
            n_used = int(len(work))
            n_positive = int(outcome.sum())
            n_negative = int(n_used - n_positive)
            if n_positive < min_class_n or n_negative < min_class_n:
                rows.append(
                    _empty_row(
                        cohort,
                        threshold,
                        degree_col=degree_col,
                        age_col=age_col,
                        reason="insufficient_outcome_class_counts",
                        source_data=source_data,
                        n_total=len(df),
                        n_used=n_used,
                        n_positive=n_positive,
                    )
                )
                continue

            x_data = {"intercept": 1.0, "g": work["g"]}
            if age_col:
                x_data["age"] = work["age"]
            x = pd.DataFrame(x_data, index=work.index)
            fit, reason = _logistic_fit(outcome, x)
            if fit is None:
                rows.append(
                    _empty_row(
                        cohort,
                        threshold,
                        degree_col=degree_col,
                        age_col=age_col,
                        reason=f"logit_failed:{reason or 'unknown'}",
                        source_data=source_data,
                        n_total=len(df),
                        n_used=n_used,
                        n_positive=n_positive,
                    )
                )
                continue

            beta = fit["beta"]
            se = fit["se"]
            p_vals = fit["p"]
            rows.append(
                {
                    "cohort": cohort,
                    "degree_col": degree_col,
                    "age_col": age_col,
                    "threshold": threshold,
                    "status": "computed",
                    "reason": pd.NA,
                    "n_total": int(len(df)),
                    "n_used": int(fit["n_used"]),
                    "n_positive": n_positive,
                    "prevalence": float(n_positive / n_used) if n_used else float("nan"),
                    "beta_g": float(beta[1]),
                    "SE_beta_g": float(se[1]),
                    "p_value_beta_g": float(p_vals[1]),
                    "odds_ratio_g": float(np.exp(beta[1])),
                    "beta_age": float(beta[2]) if age_col else pd.NA,
                    "SE_beta_age": float(se[2]) if age_col else pd.NA,
                    "p_value_beta_age": float(p_vals[2]) if age_col else pd.NA,
                    "pseudo_r2": float(fit["pseudo_r2"]),
                    "top_degree_code": top_degree_code,
                    "top_degree_label": top_degree_label,
                    "source_data": source_data,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=OUTPUT_COLUMNS)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS].copy()
    dest = output_path if output_path.is_absolute() else root / output_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build explicit degree-outcome models from cohort-coded highest-degree variables.")
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/explicit_degree_outcomes.csv"))
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_SPECS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--min-class-n", type=int, default=20)
    args = parser.parse_args()

    out = run_explicit_degree_outcomes(
        root=args.project_root.resolve(),
        cohorts=_cohorts_from_args(args),
        output_path=args.output_path,
        min_class_n=args.min_class_n,
    )
    print(f"[ok] explicit degree rows computed: {int((out['status'] == 'computed').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
