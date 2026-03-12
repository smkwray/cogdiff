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
    "beta_age_2019",
    "SE_beta_age_2019",
    "p_value_age_2019",
    "pseudo_r2",
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
    model: str,
    sample_definition: str,
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
        "n_total": int(n_total),
        "n_complete_three_wave": int(n_complete_three_wave),
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def run_nlsy97_employment_instability(
    *,
    root: Path,
    output_path: Path = Path("outputs/tables/nlsy97_employment_instability.csv"),
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
            "model": "any_transition_2011_2021",
            "sample_definition": "status changes in either 2011->2019 or 2019->2021 among respondents with all three employment waves and age_2019",
        },
        {
            "model": "mixed_attachment_2011_2021",
            "sample_definition": "at least one employed and one non-employed wave across 2011, 2019, 2021",
        },
        {
            "model": "double_transition_2011_2021",
            "sample_definition": "status changes in both 2011->2019 and 2019->2021",
        },
    ]

    if not source_path.exists():
        out = pd.DataFrame(
            [_empty_row(model=s["model"], sample_definition=s["sample_definition"], reason="missing_source_data", source_data=source_data) for s in specs],
            columns=OUTPUT_COLUMNS,
        )
        out.to_csv(target, index=False)
        return out

    df = pd.read_csv(source_path, low_memory=False)
    needed = {"employment_2011", "employment_2019", "employment_2021", "age_2019"}
    if not needed.issubset(df.columns):
        out = pd.DataFrame(
            [_empty_row(model=s["model"], sample_definition=s["sample_definition"], reason="missing_required_columns", source_data=source_data, n_total=len(df)) for s in specs],
            columns=OUTPUT_COLUMNS,
        )
        out.to_csv(target, index=False)
        return out

    indicators = hierarchical_subtests(models_cfg)
    df = df.copy()
    df["__g_proxy"] = g_proxy(df, indicators)

    work = pd.DataFrame(
        {
            "employment_2011": pd.to_numeric(df["employment_2011"], errors="coerce"),
            "employment_2019": pd.to_numeric(df["employment_2019"], errors="coerce"),
            "employment_2021": pd.to_numeric(df["employment_2021"], errors="coerce"),
            "age_2019": pd.to_numeric(df["age_2019"], errors="coerce"),
            "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
        }
    ).dropna()

    if work.empty:
        out = pd.DataFrame(
            [_empty_row(model=s["model"], sample_definition=s["sample_definition"], reason="no_valid_rows_after_cleaning", source_data=source_data, n_total=len(df)) for s in specs],
            columns=OUTPUT_COLUMNS,
        )
        out.to_csv(target, index=False)
        return out

    for col in ("employment_2011", "employment_2019", "employment_2021"):
        uniq = sorted(set(float(v) for v in work[col].unique().tolist()))
        if uniq != [0.0, 1.0]:
            out = pd.DataFrame(
                [_empty_row(model=s["model"], sample_definition=s["sample_definition"], reason=f"{col}_not_binary_zero_one", source_data=source_data, n_total=len(df), n_complete_three_wave=len(work)) for s in specs],
                columns=OUTPUT_COLUMNS,
            )
            out.to_csv(target, index=False)
            return out

    work = work.copy()
    work["transition_11_19"] = (work["employment_2011"] != work["employment_2019"]).astype(float)
    work["transition_19_21"] = (work["employment_2019"] != work["employment_2021"]).astype(float)
    work["any_transition_2011_2021"] = ((work["transition_11_19"] + work["transition_19_21"]) >= 1.0).astype(float)
    work["mixed_attachment_2011_2021"] = (
        (work[["employment_2011", "employment_2019", "employment_2021"]].max(axis=1) == 1.0)
        & (work[["employment_2011", "employment_2019", "employment_2021"]].min(axis=1) == 0.0)
    ).astype(float)
    work["double_transition_2011_2021"] = ((work["transition_11_19"] + work["transition_19_21"]) == 2.0).astype(float)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        outcome = spec["model"]
        y = work[outcome]
        n_positive = int(y.sum())
        n_negative = int(len(y) - n_positive)
        if min(n_positive, n_negative) < min_class_n:
            rows.append(
                {
                    **_empty_row(
                        model=spec["model"],
                        sample_definition=spec["sample_definition"],
                        reason="insufficient_class_rows",
                        source_data=source_data,
                        n_total=len(df),
                        n_complete_three_wave=len(work),
                    ),
                    "n_positive": n_positive,
                    "prevalence": float(y.mean()),
                    "n_employed_2011": int(work["employment_2011"].sum()),
                    "n_employed_2019": int(work["employment_2019"].sum()),
                    "n_employed_2021": int(work["employment_2021"].sum()),
                }
            )
            continue

        x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age_2019": work["age_2019"]}, index=work.index)
        fit, reason = _logistic_fit(y, x)
        if fit is None:
            rows.append(
                {
                    **_empty_row(
                        model=spec["model"],
                        sample_definition=spec["sample_definition"],
                        reason=f"logit_failed:{reason or 'unknown'}",
                        source_data=source_data,
                        n_total=len(df),
                        n_complete_three_wave=len(work),
                    ),
                    "n_positive": n_positive,
                    "prevalence": float(y.mean()),
                    "n_employed_2011": int(work["employment_2011"].sum()),
                    "n_employed_2019": int(work["employment_2019"].sum()),
                    "n_employed_2021": int(work["employment_2021"].sum()),
                }
            )
            continue

        row = {
            "cohort": COHORT,
            "model": spec["model"],
            "status": "computed",
            "reason": pd.NA,
            "sample_definition": spec["sample_definition"],
            "n_total": int(len(df)),
            "n_complete_three_wave": int(len(work)),
            "n_used": int(fit["n_used"]),
            "n_positive": n_positive,
            "prevalence": float(y.mean()),
            "n_employed_2011": int(work["employment_2011"].sum()),
            "n_employed_2019": int(work["employment_2019"].sum()),
            "n_employed_2021": int(work["employment_2021"].sum()),
            "beta_g": float(fit["beta"][1]),
            "SE_beta_g": float(fit["se"][1]),
            "p_value_beta_g": float(fit["p"][1]),
            "odds_ratio_g": float(math.exp(float(fit["beta"][1]))),
            "beta_age_2019": float(fit["beta"][2]),
            "SE_beta_age_2019": float(fit["se"][2]),
            "p_value_age_2019": float(fit["p"][2]),
            "pseudo_r2": float(fit["pseudo_r2"]),
            "source_data": source_data,
        }
        for col in OUTPUT_COLUMNS:
            row.setdefault(col, pd.NA)
        rows.append(row)

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out.to_csv(target, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build bounded NLSY97 multi-wave employment instability models.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/nlsy97_employment_instability.csv"))
    parser.add_argument("--min-class-n", type=int, default=50)
    args = parser.parse_args(argv)

    out = run_nlsy97_employment_instability(
        root=Path(args.project_root),
        output_path=args.output_path,
        min_class_n=int(args.min_class_n),
    )
    print(f"[ok] employment instability rows computed: {int((out['status'] == 'computed').sum())}")
    print(f"[ok] wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
