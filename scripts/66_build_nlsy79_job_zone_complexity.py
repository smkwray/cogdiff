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
ONET_JOB_ZONES_TXT = "https://www.onetcenter.org/dl_files/database/db_30_2_text/Job%20Zones.txt"

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
    "mean_job_zone",
    "source_data",
    "census_source",
    "job_zones_source",
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
    "mean_job_zone",
    "source_data",
    "census_source",
    "job_zones_source",
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
    crosswalk = crosswalk.drop_duplicates(subset=["census_code"]).reset_index(drop=True)
    return crosswalk


def _load_job_zones(*, job_zones_source: str) -> pd.DataFrame:
    job = pd.read_csv(job_zones_source, sep="\t")
    if "O*NET-SOC Code" not in job.columns or "Job Zone" not in job.columns:
        raise RuntimeError("job_zones_missing_required_columns")
    job = job.copy()
    job["soc_exact"] = job["O*NET-SOC Code"].astype(str).str.extract(r"^(\d{2}-\d{4})")[0]
    job = job.dropna(subset=["soc_exact"]).copy()
    job["soc_prefix5"] = job["soc_exact"].str[:6]
    job["job_zone"] = pd.to_numeric(job["Job Zone"], errors="coerce")
    job = job.dropna(subset=["job_zone"])
    if job.empty:
        raise RuntimeError("job_zones_empty_after_cleaning")
    return job


def _empty_mapping(reason: str, source_data: str, *, census_source: str, job_zones_source: str) -> pd.DataFrame:
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
        "job_zones_source": job_zones_source,
    }
    for col in MAPPING_COLUMNS:
        row.setdefault(col, pd.NA)
    return pd.DataFrame([row])[MAPPING_COLUMNS]


def _empty_model(reason: str, source_data: str, *, census_source: str, job_zones_source: str) -> pd.DataFrame:
    row: dict[str, Any] = {
        "cohort": COHORT,
        "status": "not_feasible",
        "reason": reason,
        "outcome": "job_zone",
        "n_total": 0,
        "n_occ_nonmissing": 0,
        "n_used": 0,
        "age_col": "age_2000",
        "source_data": source_data,
        "census_source": census_source,
        "job_zones_source": job_zones_source,
    }
    for col in MODEL_COLUMNS:
        row.setdefault(col, pd.NA)
    return pd.DataFrame([row])[MODEL_COLUMNS]


def run_nlsy79_job_zone_complexity(
    *,
    root: Path,
    mapping_output_path: Path = Path("outputs/tables/nlsy79_job_zone_mapping_quality.csv"),
    model_output_path: Path = Path("outputs/tables/nlsy79_job_zone_complexity_outcome.csv"),
    census_pdf_url: str = CENSUS_2002_CODES_PDF,
    census_text_path: Path | None = None,
    job_zones_source: str = ONET_JOB_ZONES_TXT,
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
        mapping = _empty_mapping("missing_source_data", source_data, census_source=census_source, job_zones_source=job_zones_source)
        model = _empty_model("missing_source_data", source_data, census_source=census_source, job_zones_source=job_zones_source)
    else:
        crosswalk = _load_census_crosswalk(census_pdf_url=census_pdf_url, census_text_path=census_text_path)
        job = _load_job_zones(job_zones_source=job_zones_source)
        job_exact = job.groupby("soc_exact", as_index=True)["job_zone"].mean()
        job_prefix = job.groupby("soc_prefix5", as_index=True)["job_zone"].mean()

        df = pd.read_csv(source_path, low_memory=False)
        needed = {"occupation_code_2000", "age_2000"}
        missing = sorted(col for col in needed if col not in df.columns)
        if missing:
            reason = f"missing_required_columns:{','.join(missing)}"
            mapping = _empty_mapping(reason, source_data, census_source=census_source, job_zones_source=job_zones_source)
            model = _empty_model(reason, source_data, census_source=census_source, job_zones_source=job_zones_source)
        else:
            indicators = hierarchical_subtests(models_cfg)
            df = df.copy()
            df["__g_proxy"] = g_proxy(df, indicators)
            df["__census_code"] = _scaled_occupation_code(df["occupation_code_2000"])
            occ = df.loc[df["__census_code"].notna()].copy()
            occ["__census_code"] = occ["__census_code"].astype(int)
            occ = occ.merge(crosswalk[["census_code", "soc_code"]], left_on="__census_code", right_on="census_code", how="left")
            occ["__job_zone_exact"] = occ["soc_code"].map(job_exact)
            occ["__job_zone_prefix"] = occ["soc_code"].astype("string").str[:6].map(job_prefix)
            occ["__job_zone"] = occ["__job_zone_exact"].fillna(occ["__job_zone_prefix"])
            occ["__match_type"] = np.where(
                occ["__job_zone_exact"].notna(),
                "exact",
                np.where(occ["__job_zone"].notna(), "prefix", pd.NA),
            )

            n_occ_nonmissing = int(len(occ))
            n_matched_exact = int(occ["__job_zone_exact"].notna().sum())
            n_matched_any = int(occ["__job_zone"].notna().sum())
            n_matched_prefix_only = int(((occ["__job_zone_exact"].isna()) & occ["__job_zone"].notna()).sum())
            n_unmatched = int(n_occ_nonmissing - n_matched_any)
            mapping_row: dict[str, Any] = {
                "cohort": COHORT,
                "status": "computed" if n_matched_any >= min_n else "not_feasible",
                "reason": pd.NA if n_matched_any >= min_n else "insufficient_mapped_rows",
                "n_total": int(len(df)),
                "n_occ_nonmissing": n_occ_nonmissing,
                "n_matched_exact": n_matched_exact,
                "n_matched_prefix_only": n_matched_prefix_only,
                "n_matched_any": n_matched_any,
                "n_unmatched": n_unmatched,
                "pct_matched_any": float(n_matched_any / n_occ_nonmissing) if n_occ_nonmissing > 0 else pd.NA,
                "n_unique_census_codes": int(occ["__census_code"].nunique()),
                "n_unique_census_codes_matched": int(occ.loc[occ["__job_zone"].notna(), "__census_code"].nunique()),
                "mean_job_zone": float(pd.to_numeric(occ["__job_zone"], errors="coerce").mean()) if n_matched_any > 0 else pd.NA,
                "source_data": source_data,
                "census_source": census_source,
                "job_zones_source": job_zones_source,
            }
            for col in MAPPING_COLUMNS:
                mapping_row.setdefault(col, pd.NA)
            mapping = pd.DataFrame([mapping_row])[MAPPING_COLUMNS]

            work = occ.loc[occ["__job_zone"].notna(), ["__job_zone", "__g_proxy", "age_2000"]].copy()
            work = work.rename(columns={"__job_zone": "job_zone", "__g_proxy": "g", "age_2000": "age"})
            work = work.apply(pd.to_numeric, errors="coerce").dropna()
            if len(work) < min_n:
                model = _empty_model("insufficient_mapped_rows", source_data, census_source=census_source, job_zones_source=job_zones_source)
                model.loc[0, "n_total"] = int(len(df))
                model.loc[0, "n_occ_nonmissing"] = n_occ_nonmissing
                model.loc[0, "n_used"] = int(len(work))
                model.loc[0, "mean_job_zone"] = float(work["job_zone"].mean()) if not work.empty else pd.NA
            else:
                x = pd.DataFrame({"intercept": 1.0, "g": work["g"], "age": work["age"]}, index=work.index)
                fit, reason = _ols_fit(work["job_zone"], x)
                if fit is None:
                    model = _empty_model(f"ols_failed:{reason or 'unknown'}", source_data, census_source=census_source, job_zones_source=job_zones_source)
                    model.loc[0, "n_total"] = int(len(df))
                    model.loc[0, "n_occ_nonmissing"] = n_occ_nonmissing
                    model.loc[0, "n_used"] = int(len(work))
                    model.loc[0, "mean_job_zone"] = float(work["job_zone"].mean()) if not work.empty else pd.NA
                else:
                    row = {
                        "cohort": COHORT,
                        "status": "computed",
                        "reason": pd.NA,
                        "outcome": "job_zone",
                        "n_total": int(len(df)),
                        "n_occ_nonmissing": n_occ_nonmissing,
                        "n_used": int(fit["n_used"]),
                        "age_col": "age_2000",
                        "beta_g": float(fit["beta"][1]),
                        "SE_beta_g": float(fit["se"][1]),
                        "p_value_beta_g": float(fit["p"][1]),
                        "beta_age": float(fit["beta"][2]),
                        "SE_beta_age": float(fit["se"][2]),
                        "p_value_beta_age": float(fit["p"][2]),
                        "r2": float(fit["r2"]),
                        "mean_job_zone": float(work["job_zone"].mean()),
                        "source_data": source_data,
                        "census_source": census_source,
                        "job_zones_source": job_zones_source,
                    }
                    for col in MODEL_COLUMNS:
                        row.setdefault(col, pd.NA)
                    model = pd.DataFrame([row])[MODEL_COLUMNS]

    mapping_full = root / mapping_output_path
    mapping_full.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_full, index=False)

    model_full = root / model_output_path
    model_full.parent.mkdir(parents=True, exist_ok=True)
    model.to_csv(model_full, index=False)
    return mapping, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an NLSY79 Job Zone complexity prototype using official Census and O*NET crosswalk sources.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--mapping-output-path", type=Path, default=Path("outputs/tables/nlsy79_job_zone_mapping_quality.csv"))
    parser.add_argument("--model-output-path", type=Path, default=Path("outputs/tables/nlsy79_job_zone_complexity_outcome.csv"))
    parser.add_argument("--census-pdf-url", default=CENSUS_2002_CODES_PDF)
    parser.add_argument("--census-text-path", type=Path, default=None)
    parser.add_argument("--job-zones-source", default=ONET_JOB_ZONES_TXT)
    parser.add_argument("--min-n", type=int, default=500)
    args = parser.parse_args(argv)

    mapping, model = run_nlsy79_job_zone_complexity(
        root=Path(args.project_root),
        mapping_output_path=args.mapping_output_path,
        model_output_path=args.model_output_path,
        census_pdf_url=args.census_pdf_url,
        census_text_path=args.census_text_path,
        job_zones_source=args.job_zones_source,
        min_n=args.min_n,
    )
    print(f"[ok] job-zone mapping rows computed: {int((mapping['status'] == 'computed').sum())}")
    print(f"[ok] job-zone model rows computed: {int((model['status'] == 'computed').sum())}")
    print(f"[ok] wrote {args.mapping_output_path}")
    print(f"[ok] wrote {args.model_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
