#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.exploratory import g_proxy, ols_fit
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT = "nlsy79"
DETAIL_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "outcome",
    "model",
    "n_total",
    "n_used",
    "n_job_zone_matched",
    "age_col",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "beta_age",
    "SE_beta_age",
    "p_value_beta_age",
    "beta_education_years",
    "SE_beta_education_years",
    "p_value_beta_education_years",
    "beta_employment_2000",
    "SE_beta_employment_2000",
    "p_value_beta_employment_2000",
    "beta_job_zone",
    "SE_beta_job_zone",
    "p_value_beta_job_zone",
    "r2",
    "mean_outcome",
    "source_data",
    "census_source",
    "job_zones_source",
]
SUMMARY_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "outcome",
    "model",
    "mediators_in_model",
    "n_used",
    "beta_g_baseline",
    "beta_g_model",
    "pct_attenuation_g",
    "delta_r2",
    "source_data",
]
OUTCOME_SPECS: tuple[dict[str, str], ...] = (
    {"outcome": "log_annual_earnings", "source_col": "annual_earnings"},
    {"outcome": "log_household_income", "source_col": "household_income"},
)
MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {"model": "baseline", "mediators": []},
    {"model": "plus_education_years", "mediators": ["education_years"]},
    {"model": "plus_employment_2000", "mediators": ["employment_2000"]},
    {"model": "plus_job_zone", "mediators": ["__job_zone"]},
    {"model": "plus_all_mediators", "mediators": ["education_years", "employment_2000", "__job_zone"]},
)


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable_to_load_module:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _empty_detail(
    *,
    outcome: str,
    model: str,
    reason: str,
    source_data: str,
    census_source: str,
    job_zones_source: str,
    n_total: int = 0,
    n_job_zone_matched: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": COHORT,
        "status": "not_feasible",
        "reason": reason,
        "outcome": outcome,
        "model": model,
        "n_total": int(n_total),
        "n_job_zone_matched": int(n_job_zone_matched),
        "age_col": "age_2000",
        "source_data": source_data,
        "census_source": census_source,
        "job_zones_source": job_zones_source,
    }
    for col in DETAIL_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _empty_summary(
    *,
    outcome: str,
    model: str,
    reason: str,
    mediators_in_model: str,
    source_data: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": COHORT,
        "status": "not_feasible",
        "reason": reason,
        "outcome": outcome,
        "model": model,
        "mediators_in_model": mediators_in_model,
        "source_data": source_data,
    }
    for col in SUMMARY_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _derive_logged_outcome(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.where(values >= 0.0)
    return np.log1p(values)


def _fit_on_sample(
    work: pd.DataFrame,
    *,
    outcome_col: str,
    mediator_cols: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age": work["age"]}, index=work.index)
    if "education_years" in mediator_cols:
        x["education_years"] = work["education_years"]
    if "employment_2000" in mediator_cols:
        x["employment_2000"] = work["employment_2000"]
    if "__job_zone" in mediator_cols:
        x["job_zone"] = work["__job_zone"]
    return ols_fit(work[outcome_col], x)


def _fit_row(
    *,
    work: pd.DataFrame,
    outcome: str,
    model: str,
    mediator_cols: list[str],
    source_data: str,
    census_source: str,
    job_zones_source: str,
    n_total: int,
    n_job_zone_matched: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required_cols = ["g", "age", outcome, *mediator_cols]
    model_work = work[required_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if model_work.empty:
        reason = "empty_after_dropna"
        return (
            _empty_detail(
                outcome=outcome,
                model=model,
                reason=reason,
                source_data=source_data,
                census_source=census_source,
                job_zones_source=job_zones_source,
                n_total=n_total,
                n_job_zone_matched=n_job_zone_matched,
            ),
            _empty_summary(
                outcome=outcome,
                model=model,
                reason=reason,
                mediators_in_model=";".join(mediator_cols) if mediator_cols else "baseline",
                source_data=source_data,
            ),
        )

    baseline_fit, baseline_reason = _fit_on_sample(model_work, outcome_col=outcome, mediator_cols=[])
    if baseline_fit is None:
        reason = f"baseline_fit_failed:{baseline_reason or 'unknown'}"
        return (
            _empty_detail(
                outcome=outcome,
                model=model,
                reason=reason,
                source_data=source_data,
                census_source=census_source,
                job_zones_source=job_zones_source,
                n_total=n_total,
                n_job_zone_matched=n_job_zone_matched,
            ),
            _empty_summary(
                outcome=outcome,
                model=model,
                reason=reason,
                mediators_in_model=";".join(mediator_cols) if mediator_cols else "baseline",
                source_data=source_data,
            ),
        )

    fit, fit_reason = _fit_on_sample(model_work, outcome_col=outcome, mediator_cols=mediator_cols)
    if fit is None:
        reason = f"fit_failed:{fit_reason or 'unknown'}"
        return (
            _empty_detail(
                outcome=outcome,
                model=model,
                reason=reason,
                source_data=source_data,
                census_source=census_source,
                job_zones_source=job_zones_source,
                n_total=n_total,
                n_job_zone_matched=n_job_zone_matched,
            ),
            _empty_summary(
                outcome=outcome,
                model=model,
                reason=reason,
                mediators_in_model=";".join(mediator_cols) if mediator_cols else "baseline",
                source_data=source_data,
            ),
        )

    beta_names = ["intercept", "g", "age"]
    if "education_years" in mediator_cols:
        beta_names.append("education_years")
    if "employment_2000" in mediator_cols:
        beta_names.append("employment_2000")
    if "__job_zone" in mediator_cols:
        beta_names.append("job_zone")

    beta_idx = {name: idx for idx, name in enumerate(beta_names)}
    baseline_beta_g = float(baseline_fit["beta"][1])
    model_beta_g = float(fit["beta"][1])
    attenuation = pd.NA
    if abs(baseline_beta_g) > 1e-12:
        attenuation = float(100.0 * (baseline_beta_g - model_beta_g) / baseline_beta_g)

    detail_row: dict[str, Any] = {
        "cohort": COHORT,
        "status": "computed",
        "reason": pd.NA,
        "outcome": outcome,
        "model": model,
        "n_total": int(n_total),
        "n_used": int(fit["n_used"]),
        "n_job_zone_matched": int(n_job_zone_matched),
        "age_col": "age_2000",
        "beta_g": model_beta_g,
        "SE_beta_g": float(fit["se"][beta_idx["g"]]),
        "p_value_beta_g": float(fit["p"][beta_idx["g"]]),
        "beta_age": float(fit["beta"][beta_idx["age"]]),
        "SE_beta_age": float(fit["se"][beta_idx["age"]]),
        "p_value_beta_age": float(fit["p"][beta_idx["age"]]),
        "r2": float(fit["r2"]),
        "mean_outcome": float(model_work[outcome].mean()),
        "source_data": source_data,
        "census_source": census_source,
        "job_zones_source": job_zones_source,
    }
    if "education_years" in beta_idx:
        detail_row["beta_education_years"] = float(fit["beta"][beta_idx["education_years"]])
        detail_row["SE_beta_education_years"] = float(fit["se"][beta_idx["education_years"]])
        detail_row["p_value_beta_education_years"] = float(fit["p"][beta_idx["education_years"]])
    if "employment_2000" in beta_idx:
        detail_row["beta_employment_2000"] = float(fit["beta"][beta_idx["employment_2000"]])
        detail_row["SE_beta_employment_2000"] = float(fit["se"][beta_idx["employment_2000"]])
        detail_row["p_value_beta_employment_2000"] = float(fit["p"][beta_idx["employment_2000"]])
    if "job_zone" in beta_idx:
        detail_row["beta_job_zone"] = float(fit["beta"][beta_idx["job_zone"]])
        detail_row["SE_beta_job_zone"] = float(fit["se"][beta_idx["job_zone"]])
        detail_row["p_value_beta_job_zone"] = float(fit["p"][beta_idx["job_zone"]])
    for col in DETAIL_COLUMNS:
        detail_row.setdefault(col, pd.NA)

    summary_row: dict[str, Any] = {
        "cohort": COHORT,
        "status": "computed",
        "reason": pd.NA,
        "outcome": outcome,
        "model": model,
        "mediators_in_model": ";".join(mediator_cols) if mediator_cols else "baseline",
        "n_used": int(fit["n_used"]),
        "beta_g_baseline": baseline_beta_g,
        "beta_g_model": model_beta_g,
        "pct_attenuation_g": attenuation,
        "delta_r2": float(fit["r2"] - baseline_fit["r2"]),
        "source_data": source_data,
    }
    for col in SUMMARY_COLUMNS:
        summary_row.setdefault(col, pd.NA)
    return detail_row, summary_row


def run_nlsy79_mediation_models(
    *,
    root: Path,
    detail_output_path: Path = Path("outputs/tables/nlsy79_mediation_models.csv"),
    summary_output_path: Path = Path("outputs/tables/nlsy79_mediation_summary.csv"),
    min_n: int = 500,
    census_pdf_url: str | None = None,
    census_text_path: Path | None = None,
    job_zones_source: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir
    source_path = processed_dir / f"{COHORT}_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / f"{COHORT}_cfa.csv"
    source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{COHORT}_cfa_resid_or_cfa.csv"

    job_zone_module = _load_module(root / "scripts/66_build_nlsy79_job_zone_complexity.py", "script66_job_zone")
    census_source = str(census_text_path) if census_text_path is not None else str(
        census_pdf_url or job_zone_module.CENSUS_2002_CODES_PDF
    )
    resolved_job_zones_source = str(job_zones_source or job_zone_module.ONET_JOB_ZONES_TXT)

    if not source_path.exists():
        detail_rows = []
        summary_rows = []
        for outcome_spec in OUTCOME_SPECS:
            for model_spec in MODEL_SPECS:
                detail_rows.append(
                    _empty_detail(
                        outcome=outcome_spec["outcome"],
                        model=model_spec["model"],
                        reason="missing_source_data",
                        source_data=source_data,
                        census_source=census_source,
                        job_zones_source=resolved_job_zones_source,
                    )
                )
                summary_rows.append(
                    _empty_summary(
                        outcome=outcome_spec["outcome"],
                        model=model_spec["model"],
                        reason="missing_source_data",
                        mediators_in_model=";".join(model_spec["mediators"]) if model_spec["mediators"] else "baseline",
                        source_data=source_data,
                    )
                )
        detail = pd.DataFrame(detail_rows)[DETAIL_COLUMNS]
        summary = pd.DataFrame(summary_rows)[SUMMARY_COLUMNS]
    else:
        df = pd.read_csv(source_path, low_memory=False)
        needed = {
            "age_2000",
            "education_years",
            "employment_2000",
            "occupation_code_2000",
            "annual_earnings",
            "household_income",
        }
        missing = sorted(col for col in needed if col not in df.columns)
        if missing:
            reason = f"missing_required_columns:{','.join(missing)}"
            detail_rows = []
            summary_rows = []
            for outcome_spec in OUTCOME_SPECS:
                for model_spec in MODEL_SPECS:
                    detail_rows.append(
                        _empty_detail(
                            outcome=outcome_spec["outcome"],
                            model=model_spec["model"],
                            reason=reason,
                            source_data=source_data,
                            census_source=census_source,
                            job_zones_source=resolved_job_zones_source,
                            n_total=len(df),
                        )
                    )
                    summary_rows.append(
                        _empty_summary(
                            outcome=outcome_spec["outcome"],
                            model=model_spec["model"],
                            reason=reason,
                            mediators_in_model=";".join(model_spec["mediators"]) if model_spec["mediators"] else "baseline",
                            source_data=source_data,
                        )
                    )
            detail = pd.DataFrame(detail_rows)[DETAIL_COLUMNS]
            summary = pd.DataFrame(summary_rows)[SUMMARY_COLUMNS]
        else:
            indicators = hierarchical_subtests(models_cfg)
            df = df.copy()
            df["g"] = g_proxy(df, indicators)
            df["log_annual_earnings"] = _derive_logged_outcome(df["annual_earnings"])
            df["log_household_income"] = _derive_logged_outcome(df["household_income"])

            crosswalk = job_zone_module._load_census_crosswalk(
                census_pdf_url=census_pdf_url or job_zone_module.CENSUS_2002_CODES_PDF,
                census_text_path=census_text_path,
            )
            job = job_zone_module._load_job_zones(job_zones_source=resolved_job_zones_source)
            job_exact = job.groupby("soc_exact", as_index=True)["job_zone"].mean()
            job_prefix = job.groupby("soc_prefix5", as_index=True)["job_zone"].mean()
            df["__census_code"] = job_zone_module._scaled_occupation_code(df["occupation_code_2000"])
            occ = df.loc[df["__census_code"].notna(), ["__census_code", "occupation_code_2000"]].copy()
            occ["__row_id"] = occ.index
            occ["__census_code"] = occ["__census_code"].astype(int)
            occ = occ.merge(
                crosswalk[["census_code", "soc_code"]],
                left_on="__census_code",
                right_on="census_code",
                how="left",
            )
            occ["__job_zone_exact"] = occ["soc_code"].map(job_exact)
            occ["__job_zone_prefix"] = occ["soc_code"].astype("string").str[:6].map(job_prefix)
            occ["__job_zone"] = occ["__job_zone_exact"].fillna(occ["__job_zone_prefix"])
            df["__job_zone"] = pd.NA
            df.loc[occ["__row_id"], "__job_zone"] = occ["__job_zone"].to_numpy()
            n_job_zone_matched = int(pd.to_numeric(df["__job_zone"], errors="coerce").notna().sum())

            detail_rows = []
            summary_rows = []
            for outcome_spec in OUTCOME_SPECS:
                outcome = outcome_spec["outcome"]
                if int(df[outcome].notna().sum()) < min_n:
                    reason = "insufficient_outcome_rows"
                    for model_spec in MODEL_SPECS:
                        detail_rows.append(
                            _empty_detail(
                                outcome=outcome,
                                model=model_spec["model"],
                                reason=reason,
                                source_data=source_data,
                                census_source=census_source,
                                job_zones_source=resolved_job_zones_source,
                                n_total=len(df),
                                n_job_zone_matched=n_job_zone_matched,
                            )
                        )
                        summary_rows.append(
                            _empty_summary(
                                outcome=outcome,
                                model=model_spec["model"],
                                reason=reason,
                                mediators_in_model=";".join(model_spec["mediators"]) if model_spec["mediators"] else "baseline",
                                source_data=source_data,
                            )
                        )
                    continue

                work = df[
                    [
                        "g",
                        "age_2000",
                        "education_years",
                        "employment_2000",
                        "__job_zone",
                        outcome,
                    ]
                ].rename(columns={"age_2000": "age"})
                for model_spec in MODEL_SPECS:
                    mediator_cols = list(model_spec["mediators"])
                    comparable_required = ["g", "age", outcome, *mediator_cols]
                    comparable_n = int(work[comparable_required].apply(pd.to_numeric, errors="coerce").dropna().shape[0])
                    if comparable_n < min_n:
                        reason = "insufficient_complete_case_rows"
                        detail_rows.append(
                            _empty_detail(
                                outcome=outcome,
                                model=model_spec["model"],
                                reason=reason,
                                source_data=source_data,
                                census_source=census_source,
                                job_zones_source=resolved_job_zones_source,
                                n_total=len(df),
                                n_job_zone_matched=n_job_zone_matched,
                            )
                        )
                        summary_rows.append(
                            _empty_summary(
                                outcome=outcome,
                                model=model_spec["model"],
                                reason=reason,
                                mediators_in_model=";".join(mediator_cols) if mediator_cols else "baseline",
                                source_data=source_data,
                            )
                        )
                        continue

                    detail_row, summary_row = _fit_row(
                        work=work,
                        outcome=outcome,
                        model=model_spec["model"],
                        mediator_cols=mediator_cols,
                        source_data=source_data,
                        census_source=census_source,
                        job_zones_source=resolved_job_zones_source,
                        n_total=len(df),
                        n_job_zone_matched=n_job_zone_matched,
                    )
                    detail_rows.append(detail_row)
                    summary_rows.append(summary_row)

            detail = pd.DataFrame(detail_rows)[DETAIL_COLUMNS]
            summary = pd.DataFrame(summary_rows)[SUMMARY_COLUMNS]

    detail_full = root / detail_output_path
    detail_full.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_full, index=False)

    summary_full = root / summary_output_path
    summary_full.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_full, index=False)
    return detail, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build bounded NLSY79 mediation models for earnings and income outcomes.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--detail-output-path", type=Path, default=Path("outputs/tables/nlsy79_mediation_models.csv"))
    parser.add_argument("--summary-output-path", type=Path, default=Path("outputs/tables/nlsy79_mediation_summary.csv"))
    parser.add_argument("--min-n", type=int, default=500)
    parser.add_argument("--census-pdf-url", default=None)
    parser.add_argument("--census-text-path", type=Path, default=None)
    parser.add_argument("--job-zones-source", default=None)
    args = parser.parse_args(argv)

    detail, summary = run_nlsy79_mediation_models(
        root=Path(args.project_root),
        detail_output_path=args.detail_output_path,
        summary_output_path=args.summary_output_path,
        min_n=int(args.min_n),
        census_pdf_url=args.census_pdf_url,
        census_text_path=args.census_text_path,
        job_zones_source=args.job_zones_source,
    )
    print(f"[ok] mediation rows computed: {int((detail['status'] == 'computed').sum())}")
    print(f"[ok] mediation summary rows computed: {int((summary['status'] == 'computed').sum())}")
    print(f"[ok] wrote {args.detail_output_path}")
    print(f"[ok] wrote {args.summary_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
