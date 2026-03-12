#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
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
CENSUS_2002_CODES_PDF = (
    "https://www.census.gov/content/dam/Census/about/about-the-bureau/adrm/"
    "data-linkage/HUDMetadata/HUD-MTO/IndustryOccupationCodes/2002%20Census%20Occupation%20Codes.pdf"
)
ONET_ETE_TXT = "https://www.onetcenter.org/dl_files/database/db_30_2_text/Education,%20Training,%20and%20Experience.txt"
ONET_ETE_CATEGORIES_TXT = (
    "https://www.onetcenter.org/dl_files/database/db_30_2_text/"
    "Education,%20Training,%20and%20Experience%20Categories.txt"
)

# Approximate year equivalents for O*NET's required-education categories.
EDUCATION_CATEGORY_YEARS: dict[int, float] = {
    1: 10.0,
    2: 12.0,
    3: 13.0,
    4: 13.0,
    5: 14.0,
    6: 16.0,
    7: 17.0,
    8: 18.0,
    9: 19.0,
    10: 19.0,
    11: 20.0,
    12: 21.0,
}

MAPPING_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "n_total",
    "n_occ_nonmissing",
    "n_matched_exact",
    "n_matched_prefix_only",
    "n_matched_any",
    "n_unmatched",
    "pct_matched_any",
    "n_unique_census_codes",
    "n_unique_census_codes_matched",
    "mean_required_education_years",
    "mean_bachelor_plus_share",
    "modal_required_education_category",
    "modal_required_education_label",
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
    "n_total",
    "n_occ_nonmissing",
    "n_used",
    "age_col",
    "beta_g",
    "SE_beta_g",
    "p_value_beta_g",
    "beta_age",
    "SE_beta_age",
    "p_value_beta_age",
    "r2",
    "mean_outcome",
    "source_data",
    "census_source",
    "ete_source",
    "ete_categories_source",
]


def _scaled_occupation_code(series: pd.Series) -> pd.Series:
    codes = pd.to_numeric(series, errors="coerce")
    return codes.where(codes >= 1000, codes * 10.0)


def _ols_fit(y: pd.Series, x: pd.DataFrame) -> tuple[dict[str, Any] | None, str | None]:
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

    try:
        beta, *_ = np.linalg.lstsq(xv, yv, rcond=None)
    except np.linalg.LinAlgError:
        return None, "lstsq_failed"

    fitted = xv @ beta
    resid = yv - fitted
    dof = n - p
    if dof <= 0:
        return None, "nonpositive_residual_dof"

    sse = float(np.sum(resid**2))
    sst = float(np.sum((yv - np.mean(yv)) ** 2))
    sigma2 = sse / dof
    try:
        cov = sigma2 * np.linalg.pinv(xv.T @ xv)
    except np.linalg.LinAlgError:
        return None, "covariance_failed"

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_stats = beta / se
    p_vals = np.full(shape=(p,), fill_value=np.nan, dtype=float)
    for i in range(p):
        if math.isfinite(float(z_stats[i])) and math.isfinite(float(se[i])) and float(se[i]) > 0.0:
            p_vals[i] = float(2.0 * norm.sf(abs(float(z_stats[i]))))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else float("nan")
    return {"beta": beta, "se": se, "p": p_vals, "r2": r2, "n_used": int(n)}, None


def _download_to_temp(url: str, dest: Path) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        dest.write_bytes(response.read())
    return dest


def _load_census_crosswalk(*, census_pdf_url: str, census_text_path: Path | None = None) -> pd.DataFrame:
    if census_text_path is not None:
        text = Path(census_text_path).read_text(encoding="utf-8", errors="ignore")
    else:
        if shutil.which("pdftotext") is None:
            raise RuntimeError("pdftotext_not_available")
        with tempfile.TemporaryDirectory(prefix="sexg_occ_") as tmpdir:
            pdf_path = Path(tmpdir) / "census_2002_occupation_codes.pdf"
            txt_path = Path(tmpdir) / "census_2002_occupation_codes.txt"
            _download_to_temp(census_pdf_url, pdf_path)
            subprocess.run(["pdftotext", "-layout", str(pdf_path), str(txt_path)], check=True)
            text = txt_path.read_text(encoding="utf-8", errors="ignore")

    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = re.search(r"^(.*?)\s+(\d{4})\s+(\d{2}-\d{4}|\d{2}-\d{3}X|none)\s*$", line)
        if not match:
            continue
        desc, census_code, soc_code = match.groups()
        if soc_code == "none":
            continue
        rows.append(
            {
                "census_code": int(census_code),
                "soc_code": soc_code,
                "occupation_label": desc.strip(),
            }
        )
    crosswalk = pd.DataFrame(rows)
    if crosswalk.empty:
        raise RuntimeError("census_crosswalk_parse_failed")
    return crosswalk.drop_duplicates(subset=["census_code"]).reset_index(drop=True)


def _load_ete_categories(source: str) -> pd.DataFrame:
    cats = pd.read_csv(source, sep="\t")
    needed = {"Element ID", "Scale ID", "Category", "Category Description"}
    if not needed.issubset(cats.columns):
        raise RuntimeError("ete_categories_missing_required_columns")
    cats = cats.loc[
        cats["Element ID"].astype(str).eq("2.D.1")
        & cats["Scale ID"].astype(str).eq("RL")
    ].copy()
    cats["category"] = pd.to_numeric(cats["Category"], errors="coerce")
    cats = cats.dropna(subset=["category"])
    cats["category"] = cats["category"].astype(int)
    cats["education_years"] = cats["category"].map(EDUCATION_CATEGORY_YEARS)
    cats = cats.dropna(subset=["education_years"])
    if cats.empty:
        raise RuntimeError("ete_categories_empty_after_filter")
    return cats[["category", "Category Description", "education_years"]].rename(
        columns={"Category Description": "education_label"}
    )


def _load_education_requirements(*, ete_source: str, ete_categories_source: str) -> pd.DataFrame:
    ete = pd.read_csv(ete_source, sep="\t")
    needed = {"O*NET-SOC Code", "Element ID", "Scale ID", "Category", "Data Value"}
    if not needed.issubset(ete.columns):
        raise RuntimeError("ete_missing_required_columns")

    categories = _load_ete_categories(ete_categories_source)
    ete = ete.loc[
        ete["Element ID"].astype(str).eq("2.D.1")
        & ete["Scale ID"].astype(str).eq("RL")
    ].copy()
    ete["category"] = pd.to_numeric(ete["Category"], errors="coerce")
    ete["data_value"] = pd.to_numeric(ete["Data Value"], errors="coerce")
    ete = ete.dropna(subset=["category", "data_value"])
    ete["category"] = ete["category"].astype(int)
    ete = ete.merge(categories, on="category", how="inner")
    ete["soc_exact"] = ete["O*NET-SOC Code"].astype(str).str.extract(r"^(\d{2}-\d{4})")[0]
    ete = ete.dropna(subset=["soc_exact"]).copy()
    if ete.empty:
        raise RuntimeError("ete_empty_after_cleaning")

    # O*NET sometimes has multiple detailed occupations under the same six-digit
    # SOC root (for example, .00 and specialty extensions). Average within
    # SOC/category first so we do not double-count the education distribution.
    ete = (
        ete.groupby(["soc_exact", "category"], as_index=False)
        .agg(
            data_value=("data_value", "mean"),
            education_label=("education_label", "first"),
            education_years=("education_years", "first"),
        )
        .reset_index(drop=True)
    )
    ete["soc_prefix5"] = ete["soc_exact"].str[:6]

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        pct = group["data_value"].sum()
        weighted_years = float((group["data_value"] * group["education_years"]).sum() / 100.0)
        bachelor_plus = float(group.loc[group["category"] >= 6, "data_value"].sum() / 100.0)
        modal_idx = group["data_value"].idxmax()
        modal_category = int(group.loc[modal_idx, "category"])
        modal_label = str(group.loc[modal_idx, "education_label"])
        return pd.Series(
            {
                "required_education_years": weighted_years,
                "bachelor_plus_share": bachelor_plus,
                "modal_required_education_category": modal_category,
                "modal_required_education_label": modal_label,
                "pct_total": float(pct),
            }
        )

    exact = ete.groupby("soc_exact", as_index=True).apply(_aggregate).reset_index()

    prefix_input = (
        ete.groupby(["soc_prefix5", "category"], as_index=False)
        .agg(
            data_value=("data_value", "mean"),
            education_label=("education_label", "first"),
            education_years=("education_years", "first"),
        )
        .reset_index(drop=True)
    )
    prefix = prefix_input.groupby("soc_prefix5", as_index=True).apply(_aggregate).reset_index()
    return exact, prefix


def _empty_mapping(reason: str, source_data: str, *, census_source: str, ete_source: str, ete_categories_source: str) -> pd.DataFrame:
    row: dict[str, Any] = {
        "cohort": COHORT,
        "status": "not_feasible",
        "reason": reason,
        "n_total": 0,
        "n_occ_nonmissing": 0,
        "n_matched_exact": 0,
        "n_matched_prefix_only": 0,
        "n_matched_any": 0,
        "n_unmatched": 0,
        "source_data": source_data,
        "census_source": census_source,
        "ete_source": ete_source,
        "ete_categories_source": ete_categories_source,
    }
    for col in MAPPING_COLUMNS:
        row.setdefault(col, pd.NA)
    return pd.DataFrame([row])[MAPPING_COLUMNS]


def _empty_model(reason: str, source_data: str, *, census_source: str, ete_source: str, ete_categories_source: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome in ("required_education_years", "bachelor_plus_share"):
        row: dict[str, Any] = {
            "cohort": COHORT,
            "status": "not_feasible",
            "reason": reason,
            "outcome": outcome,
            "n_total": 0,
            "n_occ_nonmissing": 0,
            "n_used": 0,
            "age_col": "age_2000",
            "source_data": source_data,
            "census_source": census_source,
            "ete_source": ete_source,
            "ete_categories_source": ete_categories_source,
        }
        for col in MODEL_COLUMNS:
            row.setdefault(col, pd.NA)
        rows.append(row)
    return pd.DataFrame(rows)[MODEL_COLUMNS]


def run_nlsy79_occupation_education_requirements(
    *,
    root: Path,
    mapping_output_path: Path = Path("outputs/tables/nlsy79_occupation_education_mapping_quality.csv"),
    model_output_path: Path = Path("outputs/tables/nlsy79_occupation_education_requirement_outcome.csv"),
    census_pdf_url: str = CENSUS_2002_CODES_PDF,
    census_text_path: Path | None = None,
    ete_source: str = ONET_ETE_TXT,
    ete_categories_source: str = ONET_ETE_CATEGORIES_TXT,
    min_n: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir
    source_path = processed_dir / f"{COHORT}_cfa_resid.csv"
    if not source_path.exists():
        source_path = processed_dir / f"{COHORT}_cfa.csv"
    source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{COHORT}_cfa_resid_or_cfa.csv"
    census_source = str(census_text_path) if census_text_path is not None else census_pdf_url

    if not source_path.exists():
        mapping = _empty_mapping(
            "missing_source_data",
            source_data,
            census_source=census_source,
            ete_source=ete_source,
            ete_categories_source=ete_categories_source,
        )
        model = _empty_model(
            "missing_source_data",
            source_data,
            census_source=census_source,
            ete_source=ete_source,
            ete_categories_source=ete_categories_source,
        )
    else:
        crosswalk = _load_census_crosswalk(census_pdf_url=census_pdf_url, census_text_path=census_text_path)
        ete_exact, ete_prefix = _load_education_requirements(
            ete_source=ete_source,
            ete_categories_source=ete_categories_source,
        )

        df = pd.read_csv(source_path, low_memory=False)
        needed = {"occupation_code_2000", "age_2000"}
        missing = sorted(col for col in needed if col not in df.columns)
        if missing:
            reason = f"missing_required_columns:{','.join(missing)}"
            mapping = _empty_mapping(
                reason,
                source_data,
                census_source=census_source,
                ete_source=ete_source,
                ete_categories_source=ete_categories_source,
            )
            model = _empty_model(
                reason,
                source_data,
                census_source=census_source,
                ete_source=ete_source,
                ete_categories_source=ete_categories_source,
            )
        else:
            indicators = hierarchical_subtests(models_cfg)
            df = df.copy()
            df["__g_proxy"] = g_proxy(df, indicators)
            df["__census_code"] = _scaled_occupation_code(df["occupation_code_2000"])
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
            unmatched["__match_type"] = np.where(
                unmatched["required_education_years"].notna(),
                "prefix_only",
                pd.NA,
            )
            merged = pd.concat([exact_rows, unmatched], ignore_index=True, sort=False)

            n_total = int(len(merged))
            occ_nonmissing_mask = merged["__census_code"].notna()
            n_occ_nonmissing = int(occ_nonmissing_mask.sum())
            matched_mask = merged["required_education_years"].notna()
            n_matched_exact = int(merged["__match_type"].eq("exact").sum())
            n_matched_prefix_only = int(merged["__match_type"].eq("prefix_only").sum())
            n_matched_any = int(matched_mask.sum())
            n_unmatched = int(n_occ_nonmissing - n_matched_any)
            pct_matched_any = float(n_matched_any / n_occ_nonmissing) if n_occ_nonmissing else float("nan")
            unique_occ = merged.loc[occ_nonmissing_mask, "__census_code"].dropna().nunique()
            unique_occ_matched = merged.loc[matched_mask, "__census_code"].dropna().nunique()

            matched = merged.loc[matched_mask].copy()
            modal_label = pd.NA
            modal_category = pd.NA
            if not matched.empty:
                top = (
                    matched["modal_required_education_label"]
                    .fillna("")
                    .value_counts()
                    .rename_axis("label")
                    .reset_index(name="n")
                )
                if not top.empty:
                    modal_label = top.iloc[0]["label"]
                    modal_lookup = matched.loc[
                        matched["modal_required_education_label"].astype(str).eq(str(modal_label)),
                        "modal_required_education_category",
                    ].dropna()
                    if not modal_lookup.empty:
                        modal_category = int(pd.to_numeric(modal_lookup, errors="coerce").mode().iloc[0])

            mapping = pd.DataFrame(
                [
                    {
                        "cohort": COHORT,
                        "status": "computed" if n_matched_any >= min_n else "not_feasible",
                        "reason": pd.NA if n_matched_any >= min_n else "insufficient_matched_rows",
                        "n_total": n_total,
                        "n_occ_nonmissing": n_occ_nonmissing,
                        "n_matched_exact": n_matched_exact,
                        "n_matched_prefix_only": n_matched_prefix_only,
                        "n_matched_any": n_matched_any,
                        "n_unmatched": n_unmatched,
                        "pct_matched_any": pct_matched_any,
                        "n_unique_census_codes": int(unique_occ),
                        "n_unique_census_codes_matched": int(unique_occ_matched),
                        "mean_required_education_years": float(pd.to_numeric(matched["required_education_years"], errors="coerce").mean()) if not matched.empty else pd.NA,
                        "mean_bachelor_plus_share": float(pd.to_numeric(matched["bachelor_plus_share"], errors="coerce").mean()) if not matched.empty else pd.NA,
                        "modal_required_education_category": modal_category,
                        "modal_required_education_label": modal_label,
                        "source_data": source_data,
                        "census_source": census_source,
                        "ete_source": ete_source,
                        "ete_categories_source": ete_categories_source,
                    }
                ]
            )[MAPPING_COLUMNS]

            model_rows: list[dict[str, Any]] = []
            for outcome in ("required_education_years", "bachelor_plus_share"):
                row: dict[str, Any] = {
                    "cohort": COHORT,
                    "status": "not_feasible",
                    "reason": "insufficient_matched_rows",
                    "outcome": outcome,
                    "n_total": n_total,
                    "n_occ_nonmissing": n_occ_nonmissing,
                    "n_used": n_matched_any,
                    "age_col": "age_2000",
                    "source_data": source_data,
                    "census_source": census_source,
                    "ete_source": ete_source,
                    "ete_categories_source": ete_categories_source,
                }
                if n_matched_any >= min_n:
                    fit, reason = _ols_fit(
                        matched[outcome],
                        pd.DataFrame(
                            {
                                "intercept": 1.0,
                                "g_proxy": matched["__g_proxy"],
                                "age_2000": matched["age_2000"],
                            }
                        ),
                    )
                    if fit is None:
                        row["reason"] = reason
                    else:
                        row.update(
                            {
                                "status": "computed",
                                "reason": pd.NA,
                                "n_used": fit["n_used"],
                                "beta_g": float(fit["beta"][1]),
                                "SE_beta_g": float(fit["se"][1]),
                                "p_value_beta_g": float(fit["p"][1]),
                                "beta_age": float(fit["beta"][2]),
                                "SE_beta_age": float(fit["se"][2]),
                                "p_value_beta_age": float(fit["p"][2]),
                                "r2": float(fit["r2"]),
                                "mean_outcome": float(pd.to_numeric(matched[outcome], errors="coerce").mean()),
                            }
                        )
                for col in MODEL_COLUMNS:
                    row.setdefault(col, pd.NA)
                model_rows.append(row)
            model = pd.DataFrame(model_rows)[MODEL_COLUMNS]

    mapping_full = mapping.reindex(columns=MAPPING_COLUMNS)
    model_full = model.reindex(columns=MODEL_COLUMNS)
    mapping_dest = mapping_output_path if mapping_output_path.is_absolute() else root / mapping_output_path
    model_dest = model_output_path if model_output_path.is_absolute() else root / model_output_path
    mapping_dest.parent.mkdir(parents=True, exist_ok=True)
    model_dest.parent.mkdir(parents=True, exist_ok=True)
    mapping_full.to_csv(mapping_dest, index=False)
    model_full.to_csv(model_dest, index=False)
    return mapping_full, model_full


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an NLSY79 occupation education-requirement analysis using official Census and O*NET sources."
    )
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument(
        "--mapping-output-path",
        type=Path,
        default=Path("outputs/tables/nlsy79_occupation_education_mapping_quality.csv"),
    )
    parser.add_argument(
        "--model-output-path",
        type=Path,
        default=Path("outputs/tables/nlsy79_occupation_education_requirement_outcome.csv"),
    )
    parser.add_argument("--census-pdf-url", default=CENSUS_2002_CODES_PDF)
    parser.add_argument("--census-text-path", type=Path, default=None)
    parser.add_argument("--ete-source", default=ONET_ETE_TXT)
    parser.add_argument("--ete-categories-source", default=ONET_ETE_CATEGORIES_TXT)
    parser.add_argument("--min-n", type=int, default=500)
    args = parser.parse_args()

    mapping, model = run_nlsy79_occupation_education_requirements(
        root=args.project_root.resolve(),
        mapping_output_path=args.mapping_output_path,
        model_output_path=args.model_output_path,
        census_pdf_url=args.census_pdf_url,
        census_text_path=args.census_text_path,
        ete_source=args.ete_source,
        ete_categories_source=args.ete_categories_source,
        min_n=args.min_n,
    )
    print(f"[ok] education-mapped occupation rows: {int(mapping['n_matched_any'].fillna(0).max() if not mapping.empty else 0)}")
    print(f"[ok] computed model rows: {int((model['status'] == 'computed').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
