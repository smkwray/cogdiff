#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
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

COHORT = "nlsy79"
SUMMARY_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "group",
    "mismatch_threshold_years",
    "n_total",
    "n_occ_nonmissing",
    "n_matched_any",
    "n_used",
    "share_used",
    "mean_mismatch_years",
    "mean_abs_mismatch_years",
    "mean_g_proxy",
    "mean_education_years",
    "mean_required_education_years",
    "mean_annual_earnings",
    "mean_household_income",
    "source_data",
    "census_source",
    "ete_source",
    "ete_categories_source",
]
MODEL_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "outcome",
    "model_family",
    "mismatch_threshold_years",
    "n_total",
    "n_occ_nonmissing",
    "n_matched_any",
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
    "r2_or_pseudo_r2",
    "mean_outcome",
    "source_data",
    "census_source",
    "ete_source",
    "ete_categories_source",
]


def _load_occ_module():
    path = PROJECT_ROOT / "scripts" / "70_build_nlsy79_occupation_education_requirements.py"
    spec = importlib.util.spec_from_file_location("script70_build_nlsy79_occupation_education_requirements", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def _empty_summary(
    *,
    reason: str,
    source_data: str,
    threshold: float,
    census_source: str,
    ete_source: str,
    ete_categories_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in ("overall", "undereducated", "matched_band", "overeducated"):
        row: dict[str, Any] = {
            "cohort": COHORT,
            "status": "not_feasible",
            "reason": reason,
            "group": group,
            "mismatch_threshold_years": threshold,
            "source_data": source_data,
            "census_source": census_source,
            "ete_source": ete_source,
            "ete_categories_source": ete_categories_source,
        }
        for col in SUMMARY_COLUMNS:
            row.setdefault(col, pd.NA)
        rows.append(row)
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def _empty_models(
    *,
    reason: str,
    source_data: str,
    threshold: float,
    census_source: str,
    ete_source: str,
    ete_categories_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome, family in (
        ("mismatch_years", "ols"),
        ("abs_mismatch_years", "ols"),
        ("overeducated", "logit"),
        ("undereducated", "logit"),
    ):
        row: dict[str, Any] = {
            "cohort": COHORT,
            "status": "not_feasible",
            "reason": reason,
            "outcome": outcome,
            "model_family": family,
            "mismatch_threshold_years": threshold,
            "source_data": source_data,
            "census_source": census_source,
            "ete_source": ete_source,
            "ete_categories_source": ete_categories_source,
        }
        for col in MODEL_COLUMNS:
            row.setdefault(col, pd.NA)
        rows.append(row)
    return pd.DataFrame(rows)[MODEL_COLUMNS]


def _mapped_occupation_data(
    *,
    root: Path,
    census_pdf_url: str,
    census_text_path: Path | None,
    ete_source: str,
    ete_categories_source: str,
) -> tuple[pd.DataFrame, str, str]:
    occ_module = _load_occ_module()
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir
    source_path = processed_dir / f"{COHORT}_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / f"{COHORT}_cfa.csv"
    source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{COHORT}_cfa_resid_or_cfa.csv"
    if not source_path.exists():
        raise FileNotFoundError(source_data)

    df = pd.read_csv(source_path, low_memory=False)
    needed = {"occupation_code_2000", "age_2000", "education_years"}
    missing = sorted(col for col in needed if col not in df.columns)
    if missing:
        raise KeyError(",".join(missing))

    indicators = hierarchical_subtests(models_cfg)
    df = df.copy()
    df["__g_proxy"] = g_proxy(df, indicators)

    crosswalk = occ_module._load_census_crosswalk(census_pdf_url=census_pdf_url, census_text_path=census_text_path)
    ete_exact, ete_prefix = occ_module._load_education_requirements(
        ete_source=ete_source,
        ete_categories_source=ete_categories_source,
    )

    df["__census_code"] = occ_module._scaled_occupation_code(df["occupation_code_2000"])
    merged = df.merge(crosswalk, left_on="__census_code", right_on="census_code", how="left")
    merged = merged.merge(ete_exact, left_on="soc_code", right_on="soc_exact", how="left")
    exact_mask = merged["required_education_years"].notna()

    unmatched = merged.loc[~exact_mask].copy()
    unmatched["soc_prefix5"] = unmatched["soc_code"].astype(str).str[:6]
    unmatched = unmatched.drop(
        columns=[
            col
            for col in (
                "soc_exact",
                "required_education_years",
                "bachelor_plus_share",
                "modal_required_education_category",
                "modal_required_education_label",
                "pct_total",
            )
            if col in unmatched.columns
        ]
    )
    unmatched = unmatched.merge(ete_prefix, on="soc_prefix5", how="left")
    if "soc_prefix5_x" in unmatched.columns:
        unmatched = unmatched.rename(columns={"soc_prefix5_x": "soc_prefix5"})
    if "soc_prefix5_y" in unmatched.columns:
        unmatched = unmatched.drop(columns=["soc_prefix5_y"])

    exact_rows = merged.loc[exact_mask].copy()
    exact_rows["__match_type"] = "exact"
    unmatched["__match_type"] = np.where(unmatched["required_education_years"].notna(), "prefix_only", pd.NA)
    merged = pd.concat([exact_rows, unmatched], ignore_index=True, sort=False)

    census_source = str(census_text_path) if census_text_path is not None else str(census_pdf_url)
    return merged, source_data, census_source


def run_nlsy79_education_job_mismatch(
    *,
    root: Path,
    summary_output_path: Path = Path("outputs/tables/nlsy79_education_job_mismatch_summary.csv"),
    model_output_path: Path = Path("outputs/tables/nlsy79_education_job_mismatch_models.csv"),
    census_pdf_url: str | None = None,
    census_text_path: Path | None = None,
    ete_source: str | None = None,
    ete_categories_source: str | None = None,
    mismatch_threshold_years: float = 2.0,
    min_n: int = 500,
    min_class_n: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    occ_module = _load_occ_module()
    census_pdf_url = str(census_pdf_url or occ_module.CENSUS_2002_CODES_PDF)
    ete_source = str(ete_source or occ_module.ONET_ETE_TXT)
    ete_categories_source = str(ete_categories_source or occ_module.ONET_ETE_CATEGORIES_TXT)

    try:
        merged, source_data, census_source = _mapped_occupation_data(
            root=root,
            census_pdf_url=census_pdf_url,
            census_text_path=census_text_path,
            ete_source=ete_source,
            ete_categories_source=ete_categories_source,
        )
    except FileNotFoundError as exc:
        summary = _empty_summary(
            reason="missing_source_data",
            source_data=str(exc),
            threshold=mismatch_threshold_years,
            census_source=str(census_text_path) if census_text_path is not None else census_pdf_url,
            ete_source=ete_source,
            ete_categories_source=ete_categories_source,
        )
        model = _empty_models(
            reason="missing_source_data",
            source_data=str(exc),
            threshold=mismatch_threshold_years,
            census_source=str(census_text_path) if census_text_path is not None else census_pdf_url,
            ete_source=ete_source,
            ete_categories_source=ete_categories_source,
        )
    except KeyError as exc:
        reason = f"missing_required_columns:{exc.args[0]}"
        summary = _empty_summary(
            reason=reason,
            source_data=f"{COHORT}_cfa_resid_or_cfa.csv",
            threshold=mismatch_threshold_years,
            census_source=str(census_text_path) if census_text_path is not None else census_pdf_url,
            ete_source=ete_source,
            ete_categories_source=ete_categories_source,
        )
        model = _empty_models(
            reason=reason,
            source_data=f"{COHORT}_cfa_resid_or_cfa.csv",
            threshold=mismatch_threshold_years,
            census_source=str(census_text_path) if census_text_path is not None else census_pdf_url,
            ete_source=ete_source,
            ete_categories_source=ete_categories_source,
        )
    else:
        n_total = int(len(merged))
        n_occ_nonmissing = int(merged["__census_code"].notna().sum())
        n_matched_any = int(merged["required_education_years"].notna().sum())
        work = merged.loc[merged["required_education_years"].notna()].copy()
        work = work.assign(
            education_years_num=pd.to_numeric(work["education_years"], errors="coerce"),
            age_2000_num=pd.to_numeric(work["age_2000"], errors="coerce"),
            g_num=pd.to_numeric(work["__g_proxy"], errors="coerce"),
            annual_earnings_num=pd.to_numeric(work.get("annual_earnings"), errors="coerce"),
            household_income_num=pd.to_numeric(work.get("household_income"), errors="coerce"),
            required_education_years_num=pd.to_numeric(work["required_education_years"], errors="coerce"),
        )
        work = work.dropna(subset=["education_years_num", "age_2000_num", "g_num", "required_education_years_num"])
        work["mismatch_years"] = work["education_years_num"] - work["required_education_years_num"]
        work["abs_mismatch_years"] = work["mismatch_years"].abs()
        work["overeducated"] = (work["mismatch_years"] >= float(mismatch_threshold_years)).astype(float)
        work["undereducated"] = (work["mismatch_years"] <= -float(mismatch_threshold_years)).astype(float)
        work["matched_band"] = ((work["overeducated"] == 0.0) & (work["undereducated"] == 0.0)).astype(float)

        if len(work) < min_n:
            summary = _empty_summary(
                reason="insufficient_matched_rows",
                source_data=source_data,
                threshold=mismatch_threshold_years,
                census_source=census_source,
                ete_source=ete_source,
                ete_categories_source=ete_categories_source,
            )
            model = _empty_models(
                reason="insufficient_matched_rows",
                source_data=source_data,
                threshold=mismatch_threshold_years,
                census_source=census_source,
                ete_source=ete_source,
                ete_categories_source=ete_categories_source,
            )
        else:
            summary_rows: list[dict[str, Any]] = []
            group_masks = {
                "overall": pd.Series(True, index=work.index),
                "undereducated": work["undereducated"].eq(1.0),
                "matched_band": work["matched_band"].eq(1.0),
                "overeducated": work["overeducated"].eq(1.0),
            }
            for group, mask in group_masks.items():
                subset = work.loc[mask].copy()
                row: dict[str, Any] = {
                    "cohort": COHORT,
                    "status": "computed",
                    "reason": pd.NA,
                    "group": group,
                    "mismatch_threshold_years": float(mismatch_threshold_years),
                    "n_total": n_total,
                    "n_occ_nonmissing": n_occ_nonmissing,
                    "n_matched_any": n_matched_any,
                    "n_used": int(len(subset)),
                    "share_used": float(len(subset) / len(work)) if len(work) > 0 else pd.NA,
                    "mean_mismatch_years": float(subset["mismatch_years"].mean()) if not subset.empty else pd.NA,
                    "mean_abs_mismatch_years": float(subset["abs_mismatch_years"].mean()) if not subset.empty else pd.NA,
                    "mean_g_proxy": float(subset["g_num"].mean()) if not subset.empty else pd.NA,
                    "mean_education_years": float(subset["education_years_num"].mean()) if not subset.empty else pd.NA,
                    "mean_required_education_years": float(subset["required_education_years_num"].mean()) if not subset.empty else pd.NA,
                    "mean_annual_earnings": float(subset["annual_earnings_num"].mean()) if not subset["annual_earnings_num"].dropna().empty else pd.NA,
                    "mean_household_income": float(subset["household_income_num"].mean()) if not subset["household_income_num"].dropna().empty else pd.NA,
                    "source_data": source_data,
                    "census_source": census_source,
                    "ete_source": ete_source,
                    "ete_categories_source": ete_categories_source,
                }
                for col in SUMMARY_COLUMNS:
                    row.setdefault(col, pd.NA)
                summary_rows.append(row)
            summary = pd.DataFrame(summary_rows)[SUMMARY_COLUMNS]

            model_rows: list[dict[str, Any]] = []
            x = pd.DataFrame({"intercept": 1.0, "g_proxy": work["g_num"], "age_2000": work["age_2000_num"]}, index=work.index)
            for outcome, family in (
                ("mismatch_years", "ols"),
                ("abs_mismatch_years", "ols"),
                ("overeducated", "logit"),
                ("undereducated", "logit"),
            ):
                row: dict[str, Any] = {
                    "cohort": COHORT,
                    "status": "not_feasible",
                    "reason": pd.NA,
                    "outcome": outcome,
                    "model_family": family,
                    "mismatch_threshold_years": float(mismatch_threshold_years),
                    "n_total": n_total,
                    "n_occ_nonmissing": n_occ_nonmissing,
                    "n_matched_any": n_matched_any,
                    "n_used": int(len(work)),
                    "source_data": source_data,
                    "census_source": census_source,
                    "ete_source": ete_source,
                    "ete_categories_source": ete_categories_source,
                }
                if family == "ols":
                    fit, reason = occ_module._ols_fit(work[outcome], x)
                    if fit is None:
                        row["reason"] = f"ols_failed:{reason or 'unknown'}"
                    else:
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
                                "mean_outcome": float(pd.to_numeric(work[outcome], errors="coerce").mean()),
                            }
                        )
                else:
                    n_positive = int(work[outcome].sum())
                    n_negative = int(len(work) - n_positive)
                    row["n_positive"] = n_positive
                    row["prevalence"] = float(n_positive / len(work)) if len(work) > 0 else pd.NA
                    row["mean_outcome"] = float(work[outcome].mean()) if len(work) > 0 else pd.NA
                    if n_positive < min_class_n or n_negative < min_class_n:
                        row["reason"] = "insufficient_outcome_class_counts"
                    else:
                        fit, reason = _logistic_fit(work[outcome], x)
                        if fit is None:
                            row["reason"] = f"logit_failed:{reason or 'unknown'}"
                        else:
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
                for col in MODEL_COLUMNS:
                    row.setdefault(col, pd.NA)
                model_rows.append(row)
            model = pd.DataFrame(model_rows)[MODEL_COLUMNS]

    summary_path = summary_output_path if summary_output_path.is_absolute() else root / summary_output_path
    model_path = model_output_path if model_output_path.is_absolute() else root / model_output_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    model.to_csv(model_path, index=False)
    return summary, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NLSY79 education-job mismatch summaries and models.")
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--mismatch-threshold-years", type=float, default=2.0)
    parser.add_argument("--min-n", type=int, default=500)
    parser.add_argument("--min-class-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    run_nlsy79_education_job_mismatch(
        root=root,
        mismatch_threshold_years=args.mismatch_threshold_years,
        min_n=args.min_n,
        min_class_n=args.min_class_n,
    )


if __name__ == "__main__":
    main()
