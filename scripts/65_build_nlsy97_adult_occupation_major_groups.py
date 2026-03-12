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
WAVE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("2021", "occupation_code_2021", "age_2021"),
    ("2019", "occupation_code_2019", "age_2019"),
    ("2017", "occupation_code_2017", "age_2017"),
    ("2015", "occupation_code_2015", "age_2015"),
    ("2013", "occupation_code_2013", "age_2013"),
    ("2011", "occupation_code_2011", "age_2011"),
)

MAJOR_GROUPS: tuple[tuple[str, str, int, int], ...] = (
    ("management_professional_related", "Management/professional/related", 10, 3540),
    ("service", "Service", 3600, 4650),
    ("sales_office", "Sales/office", 4700, 4965),
    ("farming_fishing_forestry", "Farming/fishing/forestry", 5000, 5940),
    ("construction_maintenance", "Construction/extraction/maintenance", 6000, 7630),
    ("production_transport", "Production/transport/material moving", 7700, 9750),
)

SUMMARY_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "occupation_group",
    "occupation_group_label",
    "n_total",
    "n_with_any_occupation",
    "n_used",
    "share_used",
    "mean_g_proxy",
    "mean_education_years",
    "mean_age_at_occupation",
    "top_source_wave",
    "source_data",
]

MODEL_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "outcome",
    "n_total",
    "n_with_any_occupation",
    "n_used",
    "n_positive",
    "prevalence",
    "age_col",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "odds_ratio_g",
    "beta_age",
    "SE_beta_age",
    "p_value_beta_age",
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
    try:
        cov = np.linalg.pinv(xtwx)
    except np.linalg.LinAlgError:
        return None, "logit_covariance_failed"
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


def _scaled_occupation_code(series: pd.Series) -> pd.Series:
    codes = pd.to_numeric(series, errors="coerce")
    return codes.where(codes >= 1000, codes * 10.0)


def _assign_major_group(series: pd.Series) -> pd.Series:
    scaled = _scaled_occupation_code(series)
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    for key, _label, low, high in MAJOR_GROUPS:
        mask = scaled.ge(float(low)) & scaled.le(float(high))
        out = out.mask(mask, key)
    return out


def _select_latest_occupation(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["occupation_code_latest"] = pd.Series(np.nan, index=df.index, dtype=float)
    out["age_latest"] = pd.Series(np.nan, index=df.index, dtype=float)
    out["source_wave"] = pd.Series(pd.NA, index=df.index, dtype="string")

    for wave, occ_col, age_col in WAVE_SPECS:
        occ = pd.to_numeric(df.get(occ_col), errors="coerce")
        age = pd.to_numeric(df.get(age_col), errors="coerce")
        mask = out["occupation_code_latest"].isna() & occ.notna()
        out.loc[mask, "occupation_code_latest"] = occ.loc[mask]
        out.loc[mask, "age_latest"] = age.loc[mask]
        out.loc[mask, "source_wave"] = wave
    return out


def _empty_summary(reason: str, source_data: str) -> pd.DataFrame:
    rows = []
    for group_key, group_label, _low, _high in MAJOR_GROUPS:
        row: dict[str, Any] = {
            "cohort": COHORT,
            "status": "not_feasible",
            "reason": reason,
            "occupation_group": group_key,
            "occupation_group_label": group_label,
            "n_total": 0,
            "n_with_any_occupation": 0,
            "n_used": 0,
            "source_data": source_data,
        }
        for col in SUMMARY_COLUMNS:
            row.setdefault(col, pd.NA)
        rows.append(row)
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def _empty_model(reason: str, source_data: str) -> pd.DataFrame:
    row: dict[str, Any] = {
        "cohort": COHORT,
        "status": "not_feasible",
        "reason": reason,
        "outcome": "management_professional_related",
        "n_total": 0,
        "n_with_any_occupation": 0,
        "n_used": 0,
        "n_positive": 0,
        "age_col": "age_latest",
        "source_data": source_data,
    }
    for col in MODEL_COLUMNS:
        row.setdefault(col, pd.NA)
    return pd.DataFrame([row])[MODEL_COLUMNS]


def run_nlsy97_adult_occupation_major_groups(
    *,
    root: Path,
    summary_output_path: Path = Path("outputs/tables/nlsy97_adult_occupation_major_group_summary.csv"),
    model_output_path: Path = Path("outputs/tables/nlsy97_high_skill_occupation_outcome.csv"),
    min_class_n: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir
    source_path = processed_dir / f"{COHORT}_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / f"{COHORT}_cfa.csv"
    source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{COHORT}_cfa_resid_or_cfa.csv"

    if not source_path.exists():
        summary = _empty_summary("missing_source_data", source_data)
        model = _empty_model("missing_source_data", source_data)
    else:
        df = pd.read_csv(source_path, low_memory=False)
        needed = {age_col for _wave, _occ_col, age_col in WAVE_SPECS}
        needed |= {occ_col for _wave, occ_col, _age_col in WAVE_SPECS}
        missing = sorted(col for col in needed if col not in df.columns)
        if missing:
            summary = _empty_summary(f"missing_required_columns:{','.join(missing)}", source_data)
            model = _empty_model(f"missing_required_columns:{','.join(missing)}", source_data)
        else:
            indicators = hierarchical_subtests(models_cfg)
            df = df.copy()
            df["__g_proxy"] = g_proxy(df, indicators)
            latest = _select_latest_occupation(df)
            df["__occupation_latest"] = latest["occupation_code_latest"]
            df["__age_latest"] = latest["age_latest"]
            df["__source_wave"] = latest["source_wave"]
            df["__occupation_group"] = _assign_major_group(df["__occupation_latest"])

            used = df.loc[df["__occupation_group"].notna()].copy()
            n_with_any = int(df["__occupation_latest"].notna().sum())
            rows: list[dict[str, Any]] = []
            for group_key, group_label, _low, _high in MAJOR_GROUPS:
                grp = used.loc[used["__occupation_group"] == group_key].copy()
                top_source_wave = pd.NA
                if not grp.empty:
                    wave_counts = grp["__source_wave"].astype("string").value_counts(dropna=True)
                    if not wave_counts.empty:
                        top_source_wave = str(wave_counts.index[0])
                row: dict[str, Any] = {
                    "cohort": COHORT,
                    "status": "computed" if not grp.empty else "not_feasible",
                    "reason": pd.NA if not grp.empty else "empty_group",
                    "occupation_group": group_key,
                    "occupation_group_label": group_label,
                    "n_total": int(len(df)),
                    "n_with_any_occupation": n_with_any,
                    "n_used": int(len(grp)),
                    "share_used": float(len(grp) / len(used)) if len(used) > 0 else pd.NA,
                    "mean_g_proxy": float(pd.to_numeric(grp["__g_proxy"], errors="coerce").mean()) if not grp.empty else pd.NA,
                    "mean_education_years": float(pd.to_numeric(grp.get("education_years"), errors="coerce").mean()) if not grp.empty and "education_years" in grp.columns else pd.NA,
                    "mean_age_at_occupation": float(pd.to_numeric(grp["__age_latest"], errors="coerce").mean()) if not grp.empty else pd.NA,
                    "top_source_wave": top_source_wave,
                    "source_data": source_data,
                }
                for col in SUMMARY_COLUMNS:
                    row.setdefault(col, pd.NA)
                rows.append(row)
            summary = pd.DataFrame(rows)[SUMMARY_COLUMNS]

            high_skill = used["__occupation_group"].eq("management_professional_related").astype(float)
            work = pd.DataFrame(
                {
                    "high_skill": high_skill,
                    "g": pd.to_numeric(used["__g_proxy"], errors="coerce"),
                    "age": pd.to_numeric(used["__age_latest"], errors="coerce"),
                }
            ).dropna()
            if work.empty:
                model = _empty_model("no_valid_rows_after_cleaning", source_data)
                model.loc[0, "n_total"] = int(len(df))
                model.loc[0, "n_with_any_occupation"] = n_with_any
            else:
                n_positive = int(work["high_skill"].sum())
                n_negative = int(len(work) - n_positive)
                if n_positive < min_class_n or n_negative < min_class_n:
                    model = _empty_model("insufficient_outcome_class_counts", source_data)
                    model.loc[0, "n_total"] = int(len(df))
                    model.loc[0, "n_with_any_occupation"] = n_with_any
                    model.loc[0, "n_used"] = int(len(work))
                    model.loc[0, "n_positive"] = int(n_positive)
                else:
                    x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age": work["age"]}, index=work.index)
                    fit, reason = _logistic_fit(work["high_skill"], x)
                    if fit is None:
                        model = _empty_model(f"logit_failed:{reason or 'unknown'}", source_data)
                        model.loc[0, "n_total"] = int(len(df))
                        model.loc[0, "n_with_any_occupation"] = n_with_any
                        model.loc[0, "n_used"] = int(len(work))
                        model.loc[0, "n_positive"] = int(n_positive)
                    else:
                        row = {
                            "cohort": COHORT,
                            "status": "computed",
                            "reason": pd.NA,
                            "outcome": "management_professional_related",
                            "n_total": int(len(df)),
                            "n_with_any_occupation": n_with_any,
                            "n_used": int(fit["n_used"]),
                            "n_positive": int(n_positive),
                            "prevalence": float(n_positive / len(work)),
                            "age_col": "age_latest",
                            "beta_g": float(fit["beta"][1]),
                            "SE_beta_g": float(fit["se"][1]),
                            "p_value_beta_g": float(fit["p"][1]),
                            "odds_ratio_g": float(math.exp(float(fit["beta"][1]))),
                            "beta_age": float(fit["beta"][2]),
                            "SE_beta_age": float(fit["se"][2]),
                            "p_value_beta_age": float(fit["p"][2]),
                            "pseudo_r2": float(fit["pseudo_r2"]),
                            "source_data": source_data,
                        }
                        for col in MODEL_COLUMNS:
                            row.setdefault(col, pd.NA)
                        model = pd.DataFrame([row])[MODEL_COLUMNS]

    summary_full = root / summary_output_path
    summary_full.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_full, index=False)

    model_full = root / model_output_path
    model_full.parent.mkdir(parents=True, exist_ok=True)
    model.to_csv(model_full, index=False)
    return summary, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build NLSY97 latest-adult occupation major-group summaries and a high-skill occupation proxy model.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--summary-output-path", type=Path, default=Path("outputs/tables/nlsy97_adult_occupation_major_group_summary.csv"))
    parser.add_argument("--model-output-path", type=Path, default=Path("outputs/tables/nlsy97_high_skill_occupation_outcome.csv"))
    parser.add_argument("--min-class-n", type=int, default=50)
    args = parser.parse_args(argv)

    root = Path(args.project_root)
    summary, model = run_nlsy97_adult_occupation_major_groups(
        root=root,
        summary_output_path=args.summary_output_path,
        model_output_path=args.model_output_path,
        min_class_n=args.min_class_n,
    )
    print(f"[ok] occupation summary rows computed: {int((summary['status'] == 'computed').sum())}")
    print(f"[ok] high-skill model rows computed: {int((model['status'] == 'computed').sum())}")
    print(f"[ok] wrote {args.summary_output_path}")
    print(f"[ok] wrote {args.model_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
