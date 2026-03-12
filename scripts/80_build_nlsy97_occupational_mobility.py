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
    ("2011", "occupation_code_2011", "age_2011"),
    ("2013", "occupation_code_2013", "age_2013"),
    ("2015", "occupation_code_2015", "age_2015"),
    ("2017", "occupation_code_2017", "age_2017"),
    ("2019", "occupation_code_2019", "age_2019"),
    ("2021", "occupation_code_2021", "age_2021"),
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
    "n_total",
    "n_with_2plus_occupation_waves",
    "n_with_major_group_start_end",
    "n_changed_major_group",
    "pct_changed_major_group",
    "n_upward_to_management_professional",
    "n_downward_from_management_professional",
    "mean_year_gap",
    "mean_age_start",
    "top_wave_pair",
    "top_start_group",
    "top_end_group",
    "source_data",
]
MODEL_COLUMNS = [
    "cohort",
    "model",
    "status",
    "reason",
    "sample_definition",
    "n_total",
    "n_with_2plus_occupation_waves",
    "n_used",
    "n_positive",
    "prevalence",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "odds_ratio_g",
    "beta_age_start",
    "SE_beta_age_start",
    "p_value_age_start",
    "pseudo_r2",
    "source_data",
]


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


def _empty_summary(reason: str, source_data: str) -> pd.DataFrame:
    row: dict[str, Any] = {"cohort": COHORT, "status": "not_feasible", "reason": reason, "source_data": source_data}
    for col in SUMMARY_COLUMNS:
        row.setdefault(col, pd.NA)
    return pd.DataFrame([row])[SUMMARY_COLUMNS]


def _empty_model(model: str, sample_definition: str, reason: str, source_data: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": COHORT,
        "model": model,
        "status": "not_feasible",
        "reason": reason,
        "sample_definition": sample_definition,
        "source_data": source_data,
    }
    for col in MODEL_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _build_panel(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for wave, occ_col, age_col in WAVE_SPECS:
        occ = pd.to_numeric(df[occ_col], errors="coerce")
        age = pd.to_numeric(df[age_col], errors="coerce")
        chunk = pd.DataFrame(
            {
                "person_id": pd.to_numeric(df["person_id"], errors="coerce"),
                "wave": wave,
                "occupation_code": occ,
                "age": age,
                "g": pd.to_numeric(df["__g_proxy"], errors="coerce"),
            }
        )
        chunk = chunk.dropna(subset=["person_id", "occupation_code"]).copy()
        chunk["person_id"] = chunk["person_id"].astype(int)
        rows.append(chunk)
    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["person_id", "wave", "occupation_code", "age", "g"])
    if panel.empty:
        return panel
    panel["major_group"] = _assign_major_group(panel["occupation_code"])
    return panel


def run_nlsy97_occupational_mobility(
    *,
    root: Path,
    summary_output_path: Path = Path("outputs/tables/nlsy97_occupational_mobility_summary.csv"),
    model_output_path: Path = Path("outputs/tables/nlsy97_occupational_mobility_models.csv"),
    min_class_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir
    source_path = processed_dir / f"{COHORT}_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / f"{COHORT}_cfa.csv"
    source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{COHORT}_cfa_resid_or_cfa.csv"

    model_specs = [
        {
            "model": "any_major_group_change",
            "sample_definition": "respondents with earliest and latest observed adult occupation major groups",
            "outcome": "changed_major_group",
            "subset": None,
        },
        {
            "model": "downward_from_management_professional",
            "sample_definition": "subset with earliest observed occupation in management/professional/related",
            "outcome": "downward_from_management_professional",
            "subset": lambda x: x["major_group_start"].eq("management_professional_related"),
        },
        {
            "model": "upward_to_management_professional",
            "sample_definition": "subset with earliest observed occupation outside management/professional/related",
            "outcome": "upward_to_management_professional",
            "subset": lambda x: ~x["major_group_start"].eq("management_professional_related"),
        },
    ]

    if not source_path.exists():
        summary = _empty_summary("missing_source_data", source_data)
        models = pd.DataFrame([_empty_model(s["model"], s["sample_definition"], "missing_source_data", source_data) for s in model_specs])[MODEL_COLUMNS]
    else:
        df = pd.read_csv(source_path, low_memory=False)
        needed = {"person_id"} | {occ for _w, occ, _a in WAVE_SPECS} | {age for _w, _o, age in WAVE_SPECS}
        missing = sorted(col for col in needed if col not in df.columns)
        if missing:
            reason = f"missing_required_columns:{','.join(missing)}"
            summary = _empty_summary(reason, source_data)
            models = pd.DataFrame([_empty_model(s["model"], s["sample_definition"], reason, source_data) for s in model_specs])[MODEL_COLUMNS]
        else:
            indicators = hierarchical_subtests(models_cfg)
            df = df.copy()
            df["__g_proxy"] = g_proxy(df, indicators)
            panel = _build_panel(df)
            if panel.empty:
                summary = _empty_summary("no_occupation_rows", source_data)
                models = pd.DataFrame([_empty_model(s["model"], s["sample_definition"], "no_occupation_rows", source_data) for s in model_specs])[MODEL_COLUMNS]
            else:
                counts = panel.groupby("person_id").size()
                eligible_ids = counts.loc[counts >= 2].index
                panel2 = panel.loc[panel["person_id"].isin(eligible_ids)].copy()
                start = panel2.sort_values(["person_id", "wave"]).groupby("person_id", as_index=False).first()
                end = panel2.sort_values(["person_id", "wave"]).groupby("person_id", as_index=False).last()
                paired = start.merge(
                    end,
                    on="person_id",
                    suffixes=("_start", "_end"),
                    how="inner",
                )
                paired = paired.loc[paired["wave_start"] != paired["wave_end"]].copy()
                paired["year_gap"] = pd.to_numeric(paired["wave_end"], errors="coerce") - pd.to_numeric(paired["wave_start"], errors="coerce")
                paired["changed_major_group"] = (paired["major_group_start"] != paired["major_group_end"]).astype(float)
                paired["upward_to_management_professional"] = (
                    (~paired["major_group_start"].eq("management_professional_related"))
                    & (paired["major_group_end"].eq("management_professional_related"))
                ).astype(float)
                paired["downward_from_management_professional"] = (
                    (paired["major_group_start"].eq("management_professional_related"))
                    & (~paired["major_group_end"].eq("management_professional_related"))
                ).astype(float)

                paired_groups = paired.loc[paired["major_group_start"].notna() & paired["major_group_end"].notna()].copy()
                if paired_groups.empty:
                    summary = _empty_summary("no_paired_major_group_rows", source_data)
                    models = pd.DataFrame([_empty_model(s["model"], s["sample_definition"], "no_paired_major_group_rows", source_data) for s in model_specs])[MODEL_COLUMNS]
                else:
                    wave_pair_counts = (paired_groups["wave_start"].astype(str) + "->" + paired_groups["wave_end"].astype(str)).value_counts()
                    start_counts = paired_groups["major_group_start"].astype(str).value_counts()
                    end_counts = paired_groups["major_group_end"].astype(str).value_counts()
                    summary_row = {
                        "cohort": COHORT,
                        "status": "computed",
                        "reason": pd.NA,
                        "n_total": int(len(df)),
                        "n_with_2plus_occupation_waves": int(paired.shape[0]),
                        "n_with_major_group_start_end": int(paired_groups.shape[0]),
                        "n_changed_major_group": int(paired_groups["changed_major_group"].sum()),
                        "pct_changed_major_group": float(paired_groups["changed_major_group"].mean()),
                        "n_upward_to_management_professional": int(paired_groups["upward_to_management_professional"].sum()),
                        "n_downward_from_management_professional": int(paired_groups["downward_from_management_professional"].sum()),
                        "mean_year_gap": float(paired_groups["year_gap"].mean()),
                        "mean_age_start": float(pd.to_numeric(paired_groups["age_start"], errors="coerce").mean()),
                        "top_wave_pair": str(wave_pair_counts.index[0]) if not wave_pair_counts.empty else pd.NA,
                        "top_start_group": str(start_counts.index[0]) if not start_counts.empty else pd.NA,
                        "top_end_group": str(end_counts.index[0]) if not end_counts.empty else pd.NA,
                        "source_data": source_data,
                    }
                    for col in SUMMARY_COLUMNS:
                        summary_row.setdefault(col, pd.NA)
                    summary = pd.DataFrame([summary_row])[SUMMARY_COLUMNS]

                    model_rows: list[dict[str, Any]] = []
                    for spec in model_specs:
                        work = paired_groups.copy()
                        if spec["subset"] is not None:
                            work = work.loc[spec["subset"](work)].copy()
                        y = pd.to_numeric(work[spec["outcome"]], errors="coerce")
                        n_positive = int(y.sum()) if not y.empty else 0
                        n_negative = int(len(y) - n_positive)
                        if work.empty or min(n_positive, n_negative) < min_class_n:
                            model_rows.append(
                                _empty_model(
                                    spec["model"],
                                    spec["sample_definition"],
                                    "insufficient_class_rows",
                                    source_data,
                                )
                            )
                            model_rows[-1]["n_total"] = int(len(df))
                            model_rows[-1]["n_with_2plus_occupation_waves"] = int(paired.shape[0])
                            model_rows[-1]["n_used"] = int(len(work))
                            model_rows[-1]["n_positive"] = n_positive
                            model_rows[-1]["prevalence"] = float(y.mean()) if len(y) > 0 else pd.NA
                            continue

                        x = pd.DataFrame(
                            {
                                "intercept": 1.0,
                                "g": pd.to_numeric(work["g_start"], errors="coerce"),
                                "age_start": pd.to_numeric(work["age_start"], errors="coerce"),
                            },
                            index=work.index,
                        )
                        fit, reason = _logistic_fit(y, x)
                        if fit is None:
                            row = _empty_model(spec["model"], spec["sample_definition"], f"logit_failed:{reason or 'unknown'}", source_data)
                            row["n_total"] = int(len(df))
                            row["n_with_2plus_occupation_waves"] = int(paired.shape[0])
                            row["n_used"] = int(len(work))
                            row["n_positive"] = n_positive
                            row["prevalence"] = float(y.mean()) if len(y) > 0 else pd.NA
                            model_rows.append(row)
                            continue

                        row = {
                            "cohort": COHORT,
                            "model": spec["model"],
                            "status": "computed",
                            "reason": pd.NA,
                            "sample_definition": spec["sample_definition"],
                            "n_total": int(len(df)),
                            "n_with_2plus_occupation_waves": int(paired.shape[0]),
                            "n_used": int(fit["n_used"]),
                            "n_positive": n_positive,
                            "prevalence": float(y.mean()),
                            "beta_g": float(fit["beta"][1]),
                            "SE_beta_g": float(fit["se"][1]),
                            "p_value_beta_g": float(fit["p"][1]),
                            "odds_ratio_g": float(math.exp(float(fit["beta"][1]))),
                            "beta_age_start": float(fit["beta"][2]),
                            "SE_beta_age_start": float(fit["se"][2]),
                            "p_value_age_start": float(fit["p"][2]),
                            "pseudo_r2": float(fit["pseudo_r2"]),
                            "source_data": source_data,
                        }
                        for col in MODEL_COLUMNS:
                            row.setdefault(col, pd.NA)
                        model_rows.append(row)
                    models = pd.DataFrame(model_rows)[MODEL_COLUMNS]

    summary_target = root / summary_output_path
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_target, index=False)

    model_target = root / model_output_path
    model_target.parent.mkdir(parents=True, exist_ok=True)
    models.to_csv(model_target, index=False)
    return summary, models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build bounded NLSY97 occupational mobility outputs from earliest/latest adult occupation observations.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--summary-output-path", type=Path, default=Path("outputs/tables/nlsy97_occupational_mobility_summary.csv"))
    parser.add_argument("--model-output-path", type=Path, default=Path("outputs/tables/nlsy97_occupational_mobility_models.csv"))
    parser.add_argument("--min-class-n", type=int, default=5)
    args = parser.parse_args(argv)

    summary, models = run_nlsy97_occupational_mobility(
        root=Path(args.project_root),
        summary_output_path=args.summary_output_path,
        model_output_path=args.model_output_path,
        min_class_n=int(args.min_class_n),
    )
    print(f"[ok] occupational mobility summary rows computed: {int((summary['status'] == 'computed').sum())}")
    print(f"[ok] occupational mobility model rows computed: {int((models['status'] == 'computed').sum())}")
    print(f"[ok] wrote {args.summary_output_path}")
    print(f"[ok] wrote {args.model_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
