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
from nls_pipeline.sem import hierarchical_subtests

COHORT = "nlsy97"
OUTPUT_COLUMNS = [
    "cohort",
    "year",
    "model",
    "status",
    "reason",
    "sample_definition",
    "outcome_col",
    "age_col",
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
    "p_value_age",
    "r2_or_pseudo_r2",
    "source_data",
]


def _logistic_fit(
    y: pd.Series,
    x: pd.DataFrame,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[dict[str, Any] | None, str | None]:
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
    cov = np.linalg.pinv(xv.T @ (w[:, None] * xv))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_stats = beta / se
    p_vals = np.full(shape=(p,), fill_value=np.nan, dtype=float)
    for i in range(p):
        if math.isfinite(float(z_stats[i])) and math.isfinite(float(se[i])) and float(se[i]) > 0.0:
            p_vals[i] = float(2.0 * norm.sf(abs(float(z_stats[i]))))

    ll_full = float(np.sum(yv * np.log(p_hat) + (1.0 - yv) * np.log(1.0 - p_hat)))
    y_bar = float(np.mean(yv))
    pseudo_r2 = float("nan")
    if 0.0 < y_bar < 1.0:
        ll_null = float(np.sum(yv * math.log(y_bar) + (1.0 - yv) * math.log(1.0 - y_bar)))
        if ll_null != 0.0:
            pseudo_r2 = float(1.0 - (ll_full / ll_null))
    return {"beta": beta, "se": se, "p": p_vals, "pseudo_r2": pseudo_r2, "n_used": int(n)}, None


def _empty_row(
    *,
    year: str,
    model: str,
    sample_definition: str,
    outcome_col: str,
    age_col: str,
    reason: str,
    source_data: str,
    n_total: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": COHORT,
        "year": year,
        "model": model,
        "status": "not_feasible",
        "reason": reason,
        "sample_definition": sample_definition,
        "outcome_col": outcome_col,
        "age_col": age_col,
        "n_total": int(n_total),
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_nlsy97_unemployment_insurance(
    *,
    root: Path,
    output_path: Path = Path("outputs/tables/nlsy97_unemployment_insurance.csv"),
    min_class_n: int = 25,
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir
    source_path = processed_dir / f"{COHORT}_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / f"{COHORT}_cfa.csv"
    source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{COHORT}_cfa_resid_or_cfa.csv"

    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "year": "2019",
            "model": "any_ui_receipt",
            "kind": "logit",
            "outcome_col": "ui_spells_2019",
            "age_col": "age_2019",
            "sample_definition": "all respondents with non-missing ui_spells_2019, age_2019, and g_proxy; outcome is any unemployment-insurance spells in 2019",
        },
        {
            "year": "2021",
            "model": "any_ui_receipt",
            "kind": "logit",
            "outcome_col": "ui_spells_2021",
            "age_col": "age_2021",
            "sample_definition": "all respondents with non-missing ui_spells_2021, age_2021, and g_proxy; outcome is any unemployment-insurance spells in 2021",
        },
        {
            "year": "2019",
            "model": "log1p_ui_spells",
            "kind": "ols",
            "outcome_col": "ui_spells_2019",
            "age_col": "age_2019",
            "sample_definition": "all respondents with non-missing ui_spells_2019, age_2019, and g_proxy; outcome is log1p UI spell count in 2019",
        },
        {
            "year": "2021",
            "model": "log1p_ui_spells",
            "kind": "ols",
            "outcome_col": "ui_spells_2021",
            "age_col": "age_2021",
            "sample_definition": "all respondents with non-missing ui_spells_2021, age_2021, and g_proxy; outcome is log1p UI spell count in 2021",
        },
        {
            "year": "2019",
            "model": "log1p_ui_amount",
            "kind": "ols",
            "outcome_col": "ui_amount_2019",
            "age_col": "age_2019",
            "sample_definition": "all respondents with non-missing ui_amount_2019, age_2019, and g_proxy; outcome is log1p UI amount in 2019",
        },
        {
            "year": "2021",
            "model": "log1p_ui_amount",
            "kind": "ols",
            "outcome_col": "ui_amount_2021",
            "age_col": "age_2021",
            "sample_definition": "all respondents with non-missing ui_amount_2021, age_2021, and g_proxy; outcome is log1p UI amount in 2021",
        },
    ]

    if not source_path.exists():
        out = pd.DataFrame(
            [
                _empty_row(
                    year=spec["year"],
                    model=spec["model"],
                    sample_definition=spec["sample_definition"],
                    outcome_col=spec["outcome_col"],
                    age_col=spec["age_col"],
                    reason="missing_source_data",
                    source_data=source_data,
                )
                for spec in specs
            ],
            columns=OUTPUT_COLUMNS,
        )
        out.to_csv(target, index=False)
        return out

    df = pd.read_csv(source_path, low_memory=False)
    needed = {"age_2019", "age_2021", "ui_spells_2019", "ui_spells_2021", "ui_amount_2019", "ui_amount_2021"}
    if not needed.issubset(df.columns):
        out = pd.DataFrame(
            [
                _empty_row(
                    year=spec["year"],
                    model=spec["model"],
                    sample_definition=spec["sample_definition"],
                    outcome_col=spec["outcome_col"],
                    age_col=spec["age_col"],
                    reason="missing_required_columns",
                    source_data=source_data,
                    n_total=len(df),
                )
                for spec in specs
            ],
            columns=OUTPUT_COLUMNS,
        )
        out.to_csv(target, index=False)
        return out

    indicators = hierarchical_subtests(models_cfg)
    df = df.copy()
    df["__g_proxy"] = g_proxy(df, indicators)
    rows: list[dict[str, Any]] = []

    for spec in specs:
        year = spec["year"]
        outcome_col = spec["outcome_col"]
        age_col = spec["age_col"]
        model = spec["model"]
        sample_definition = spec["sample_definition"]

        work = pd.DataFrame(
            {
                "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
                "age": pd.to_numeric(df[age_col], errors="coerce"),
                "outcome_raw": pd.to_numeric(df[outcome_col], errors="coerce"),
            }
        ).dropna()
        work = work.loc[work["outcome_raw"].ge(0.0)].copy()

        if work.empty:
            rows.append(
                _empty_row(
                    year=year,
                    model=model,
                    sample_definition=sample_definition,
                    outcome_col=outcome_col,
                    age_col=age_col,
                    reason="no_valid_rows_after_cleaning",
                    source_data=source_data,
                    n_total=len(df),
                )
            )
            continue

        if spec["kind"] == "logit":
            y = work["outcome_raw"].gt(0.0).astype(float)
            n_positive = int(y.sum())
            n_negative = int(len(y) - n_positive)
            if min(n_positive, n_negative) < min_class_n:
                rows.append(
                    {
                        **_empty_row(
                            year=year,
                            model=model,
                            sample_definition=sample_definition,
                            outcome_col=outcome_col,
                            age_col=age_col,
                            reason="insufficient_class_rows",
                            source_data=source_data,
                            n_total=len(df),
                        ),
                        "n_used": int(len(work)),
                        "n_positive": n_positive,
                        "prevalence": float(y.mean()),
                        "mean_outcome": float(work["outcome_raw"].mean()),
                    }
                )
                continue
            x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age": work["age"]}, index=work.index)
            fit, reason = _logistic_fit(y, x)
            if fit is None:
                rows.append(
                    {
                        **_empty_row(
                            year=year,
                            model=model,
                            sample_definition=sample_definition,
                            outcome_col=outcome_col,
                            age_col=age_col,
                            reason=reason or "model_failed",
                            source_data=source_data,
                            n_total=len(df),
                        ),
                        "n_used": int(len(work)),
                        "n_positive": n_positive,
                        "prevalence": float(y.mean()),
                        "mean_outcome": float(work["outcome_raw"].mean()),
                    }
                )
                continue
            beta = fit["beta"]
            se = fit["se"]
            p_vals = fit["p"]
            rows.append(
                {
                    "cohort": COHORT,
                    "year": year,
                    "model": model,
                    "status": "computed",
                    "reason": "",
                    "sample_definition": sample_definition,
                    "outcome_col": outcome_col,
                    "age_col": age_col,
                    "n_total": int(len(df)),
                    "n_used": int(fit["n_used"]),
                    "n_positive": n_positive,
                    "prevalence": float(y.mean()),
                    "mean_outcome": float(work["outcome_raw"].mean()),
                    "beta_g": float(beta[1]),
                    "SE_beta_g": float(se[1]),
                    "p_value_beta_g": float(p_vals[1]) if math.isfinite(float(p_vals[1])) else pd.NA,
                    "odds_ratio_g": float(math.exp(float(beta[1]))),
                    "beta_age": float(beta[2]),
                    "SE_beta_age": float(se[2]),
                    "p_value_age": float(p_vals[2]) if math.isfinite(float(p_vals[2])) else pd.NA,
                    "r2_or_pseudo_r2": float(fit["pseudo_r2"]),
                    "source_data": source_data,
                }
            )
            continue

        work["outcome"] = np.log1p(work["outcome_raw"])
        x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age": work["age"]}, index=work.index)
        fit, reason = ols_fit(work["outcome"], x)
        if fit is None:
            rows.append(
                {
                    **_empty_row(
                        year=year,
                        model=model,
                        sample_definition=sample_definition,
                        outcome_col=outcome_col,
                        age_col=age_col,
                        reason=reason or "model_failed",
                        source_data=source_data,
                        n_total=len(df),
                    ),
                    "n_used": int(len(work)),
                    "mean_outcome": float(work["outcome"].mean()),
                }
            )
            continue
        beta = fit["beta"]
        se = fit["se"]
        p_vals = fit["p"]
        rows.append(
            {
                "cohort": COHORT,
                "year": year,
                "model": model,
                "status": "computed",
                "reason": "",
                "sample_definition": sample_definition,
                "outcome_col": outcome_col,
                "age_col": age_col,
                "n_total": int(len(df)),
                "n_used": int(fit["n_used"]),
                "n_positive": int(work["outcome_raw"].gt(0.0).sum()),
                "prevalence": float(work["outcome_raw"].gt(0.0).mean()),
                "mean_outcome": float(work["outcome"].mean()),
                "beta_g": float(beta[1]),
                "SE_beta_g": float(se[1]),
                "p_value_beta_g": float(p_vals[1]) if math.isfinite(float(p_vals[1])) else pd.NA,
                "odds_ratio_g": pd.NA,
                "beta_age": float(beta[2]),
                "SE_beta_age": float(se[2]),
                "p_value_age": float(p_vals[2]) if math.isfinite(float(p_vals[2])) else pd.NA,
                "r2_or_pseudo_r2": float(fit["r2"]),
                "source_data": source_data,
            }
        )

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out.to_csv(target, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bounded NLSY97 unemployment-insurance receipt and intensity tables.")
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/nlsy97_unemployment_insurance.csv"))
    parser.add_argument("--min-class-n", type=int, default=25)
    args = parser.parse_args()

    run_nlsy97_unemployment_insurance(
        root=args.project_root.resolve(),
        output_path=args.output_path,
        min_class_n=args.min_class_n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

