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

COHORT = "nlsy97"
OUTPUT_COLUMNS = [
    "cohort",
    "model",
    "status",
    "reason",
    "sample_definition",
    "outcome_col",
    "conditioning_col",
    "n_total",
    "n_complete_three_wave",
    "n_used",
    "n_positive",
    "prevalence",
    "n_employed_2011",
    "n_employed_2019",
    "n_employed_2021",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "odds_ratio_g",
    "beta_prior_2011",
    "SE_beta_prior_2011",
    "p_value_beta_prior_2011",
    "pseudo_r2",
    "source_data",
]


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
    p_hat = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-8, 1.0 - 1e-8)
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


def _empty_row(
    *,
    model: str,
    sample_definition: str,
    outcome_col: str,
    conditioning_col: str,
    reason: str,
    source_data: str,
    n_total: int = 0,
    n_complete_three_wave: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": COHORT,
        "model": model,
        "status": "not_feasible",
        "reason": reason,
        "sample_definition": sample_definition,
        "outcome_col": outcome_col,
        "conditioning_col": conditioning_col,
        "n_total": int(n_total),
        "n_complete_three_wave": int(n_complete_three_wave),
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_nlsy97_employment_persistence(
    *,
    root: Path,
    output_path: Path = Path("outputs/tables/nlsy97_employment_persistence.csv"),
    min_class_n: int = 50,
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
            "model": "persistent_employment_2019_2021",
            "sample_definition": "all respondents with non-missing employment_2011, employment_2019, employment_2021, age_2019, and g_proxy",
            "outcome_col": "persistent_employment_2019_2021",
            "conditioning_col": "",
        },
        {
            "model": "retention_2021_given_employed_2019",
            "sample_definition": "subset with employment_2019 == 1 and non-missing employment_2011, employment_2021, age_2019, and g_proxy",
            "outcome_col": "employment_2021",
            "conditioning_col": "employment_2019=1",
        },
        {
            "model": "reentry_2021_given_not_employed_2019",
            "sample_definition": "subset with employment_2019 == 0 and non-missing employment_2011, employment_2021, age_2019, and g_proxy",
            "outcome_col": "employment_2021",
            "conditioning_col": "employment_2019=0",
        },
    ]

    if not source_path.exists():
        out = pd.DataFrame(
            [
                _empty_row(
                    model=spec["model"],
                    sample_definition=spec["sample_definition"],
                    outcome_col=spec["outcome_col"],
                    conditioning_col=spec["conditioning_col"],
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
    needed = {"employment_2011", "employment_2019", "employment_2021", "age_2019"}
    if not needed.issubset(df.columns):
        out = pd.DataFrame(
            [
                _empty_row(
                    model=spec["model"],
                    sample_definition=spec["sample_definition"],
                    outcome_col=spec["outcome_col"],
                    conditioning_col=spec["conditioning_col"],
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
    work = pd.DataFrame(
        {
            "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
            "age_2019": pd.to_numeric(df["age_2019"], errors="coerce"),
            "employment_2011": pd.to_numeric(df["employment_2011"], errors="coerce"),
            "employment_2019": pd.to_numeric(df["employment_2019"], errors="coerce"),
            "employment_2021": pd.to_numeric(df["employment_2021"], errors="coerce"),
        }
    ).dropna()

    for col in ("employment_2011", "employment_2019", "employment_2021"):
        work = work.loc[work[col].isin([0.0, 1.0])].copy()

    work["persistent_employment_2019_2021"] = (
        work["employment_2019"].eq(1.0) & work["employment_2021"].eq(1.0)
    ).astype(float)

    rows: list[dict[str, Any]] = []
    n_complete = int(len(work))
    for spec in specs:
        model = spec["model"]
        sample_definition = spec["sample_definition"]
        outcome_col = spec["outcome_col"]
        conditioning_col = spec["conditioning_col"]
        subset = work.copy()
        if conditioning_col == "employment_2019=1":
            subset = subset.loc[subset["employment_2019"].eq(1.0)].copy()
        elif conditioning_col == "employment_2019=0":
            subset = subset.loc[subset["employment_2019"].eq(0.0)].copy()

        if subset.empty:
            rows.append(
                _empty_row(
                    model=model,
                    sample_definition=sample_definition,
                    outcome_col=outcome_col,
                    conditioning_col=conditioning_col,
                    reason="empty_subset",
                    source_data=source_data,
                    n_total=len(df),
                    n_complete_three_wave=n_complete,
                )
            )
            continue

        outcome = pd.to_numeric(subset[outcome_col], errors="coerce")
        n_positive = int(outcome.eq(1.0).sum())
        n_negative = int(outcome.eq(0.0).sum())
        if n_positive < min_class_n or n_negative < min_class_n:
            rows.append(
                _empty_row(
                    model=model,
                    sample_definition=sample_definition,
                    outcome_col=outcome_col,
                    conditioning_col=conditioning_col,
                    reason="insufficient_class_counts",
                    source_data=source_data,
                    n_total=len(df),
                    n_complete_three_wave=n_complete,
                )
            )
            continue

        x = pd.DataFrame(
            {
                "intercept": 1.0,
                "g": subset["g"],
                "age_2019": subset["age_2019"],
                "employment_2011": subset["employment_2011"],
            },
            index=subset.index,
        )
        fit, reason = _logistic_fit(outcome, x)
        if fit is None:
            rows.append(
                _empty_row(
                    model=model,
                    sample_definition=sample_definition,
                    outcome_col=outcome_col,
                    conditioning_col=conditioning_col,
                    reason=f"logit_failed:{reason or 'unknown'}",
                    source_data=source_data,
                    n_total=len(df),
                    n_complete_three_wave=n_complete,
                )
            )
            continue

        rows.append(
            {
                "cohort": COHORT,
                "model": model,
                "status": "computed",
                "reason": pd.NA,
                "sample_definition": sample_definition,
                "outcome_col": outcome_col,
                "conditioning_col": conditioning_col,
                "n_total": int(len(df)),
                "n_complete_three_wave": n_complete,
                "n_used": int(fit["n_used"]),
                "n_positive": n_positive,
                "prevalence": float(n_positive / len(subset)) if len(subset) else pd.NA,
                "n_employed_2011": int(subset["employment_2011"].eq(1.0).sum()),
                "n_employed_2019": int(subset["employment_2019"].eq(1.0).sum()),
                "n_employed_2021": int(subset["employment_2021"].eq(1.0).sum()),
                "beta_g": float(fit["beta"][1]),
                "SE_beta_g": float(fit["se"][1]),
                "p_value_beta_g": float(fit["p"][1]),
                "odds_ratio_g": float(np.exp(fit["beta"][1])),
                "beta_prior_2011": float(fit["beta"][3]),
                "SE_beta_prior_2011": float(fit["se"][3]),
                "p_value_beta_prior_2011": float(fit["p"][3]),
                "pseudo_r2": float(fit["pseudo_r2"]),
                "source_data": source_data,
            }
        )

    out = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS]
    out.to_csv(target, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bounded NLSY97 labor-force persistence and employment-transition models.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables/nlsy97_employment_persistence.csv"),
        help="Output CSV path, relative to project root unless absolute.",
    )
    parser.add_argument("--min-class-n", type=int, default=50, help="Minimum positives and negatives per model.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output_path = args.output_path if args.output_path.is_absolute() else Path(args.output_path)
    out = run_nlsy97_employment_persistence(root=root, output_path=output_path, min_class_n=int(args.min_class_n))
    computed = int((out["status"] == "computed").sum()) if not out.empty else 0
    print(f"[ok] employment persistence rows computed: {computed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
