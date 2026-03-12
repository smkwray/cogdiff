#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root, relative_path
from nls_pipeline.demographics import compute_parent_education, harmonize_race_ethnicity_3cat
from nls_pipeline.sampling import (
    build_auto_shop_composite,
    deduplicate_people,
    filter_age_range,
    harmonize_pos_neg_pairs,
    recode_missing_in_columns,
    require_complete_tests,
    require_min_tests_observed,
)

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

DEFAULT_MISSING_CODES = {-1, -2, -3, -4, -5}
HARMONIZATION_CONTINUITY_WINDOW = 1.0


def _as_label(value: Any) -> str:
    # YAML 1.1 can parse "NO" as boolean False; normalize back to expected test token.
    if isinstance(value, bool):
        return "NO" if value is False else "YES"
    return str(value)


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _default_subtests(cohort: str, models_cfg: dict[str, Any]) -> list[str]:
    if cohort == "cnlsy":
        return [_as_label(x) for x in models_cfg.get("cnlsy_single_factor", [])]
    groups = models_cfg.get("hierarchical_factors", {})
    ordered: list[str] = []
    for key in ("speed", "math", "verbal", "technical"):
        for val in groups.get(key, []):
            sval = _as_label(val)
            if sval not in ordered:
                ordered.append(sval)
    return ordered


def _normalize_sex_counts(series: pd.Series) -> tuple[int, int]:
    vals = series.astype(str).str.strip().str.lower()
    male = vals.isin({"m", "male", "1"})
    female = vals.isin({"f", "female", "2"})
    return int(male.sum()), int(female.sum())


def _information_adequacy(
    *,
    cohort: str,
    sample_cfg: dict[str, Any],
    n_total: int,
    n_male: int,
    n_female: int,
) -> dict[str, Any]:
    adequacy_cfg = sample_cfg.get("adequacy", {})
    if not isinstance(adequacy_cfg, dict):
        adequacy_cfg = {}
    min_total_n = int(pd.to_numeric(pd.Series([adequacy_cfg.get("min_total_n", 0)]), errors="coerce").fillna(0).iloc[0])
    min_per_sex_n = int(
        pd.to_numeric(pd.Series([adequacy_cfg.get("min_per_sex_n", 0)]), errors="coerce").fillna(0).iloc[0]
    )

    reasons: list[str] = []
    if min_total_n > 0 and n_total < min_total_n:
        reasons.append(f"n_total<{min_total_n}")
    if min_per_sex_n > 0 and n_male < min_per_sex_n:
        reasons.append(f"n_male<{min_per_sex_n}")
    if min_per_sex_n > 0 and n_female < min_per_sex_n:
        reasons.append(f"n_female<{min_per_sex_n}")

    status = "ok" if not reasons else "low_information"
    return {
        "information_adequacy_status": status,
        "information_adequacy_reason": ";".join(reasons) if reasons else pd.NA,
        "information_min_total_n": min_total_n,
        "information_min_per_sex_n": min_per_sex_n,
        "information_target_cohort": cohort,
    }


def _normalize_sex_label(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    normalized = str(value).strip().lower()
    if normalized in {"m", "male", "1"}:
        return "male"
    if normalized in {"f", "female", "2"}:
        return "female"
    return normalized or "missing"


def _branch_source_balance_by_sex(
    df: pd.DataFrame,
    cohort: str,
    outputs: list[str],
    sex_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for output_col in outputs:
        source_col = f"{output_col}_source"
        if source_col not in df.columns:
            continue
        source_df = df[[sex_col, source_col]].copy()
        source_df["sex"] = source_df[sex_col].apply(_normalize_sex_label)
        source_df["source"] = source_df[source_col].astype("string").str.lower().fillna("none")
        counts = (
            source_df.groupby(["sex", "source"], sort=True, dropna=False)
            .size()
            .reset_index(name="count")
        )
        sex_totals = source_df["sex"].value_counts(dropna=False)
        for _, row in counts.iterrows():
            sex = str(row["sex"])
            denom = float(sex_totals.get(sex, 0))
            share = float(row["count"]) / denom if denom > 0 else 0.0
            rows.append(
                {
                    "cohort": cohort,
                    "subtest": output_col,
                    "sex": sex,
                    "source": str(row["source"]),
                    "count": int(row["count"]),
                    "share_within_sex": share,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "cohort",
                "subtest",
                "sex",
                "source",
                "count",
                "share_within_sex",
            ]
        )
    return pd.DataFrame(rows).sort_values(["cohort", "subtest", "sex", "source"]).reset_index(drop=True)


def _continuity_near_zero_summary(
    df: pd.DataFrame,
    cohort: str,
    outputs: list[str],
    *,
    continuity_window: float = HARMONIZATION_CONTINUITY_WINDOW,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for output_col in outputs:
        if output_col not in df.columns:
            continue
        merged = pd.to_numeric(df[output_col], errors="coerce")
        n_non_missing = int(merged.notna().sum())
        pos_scores = merged[merged > 0]
        neg_scores = merged[merged < 0]
        source_col = f"{output_col}_source"
        source_series = (
            df[source_col].astype("string").str.lower().fillna("none")
            if source_col in df.columns
            else pd.Series(["none"] * len(df), index=df.index, dtype="string")
        )
        source_counts = source_series.value_counts(dropna=False)
        if n_non_missing > 0:
            n_within_window = int((merged.abs() <= continuity_window).sum())
            share_within_window = n_within_window / n_non_missing
            pos_min = float(pos_scores.min()) if not pos_scores.empty else float("nan")
            neg_max = float(neg_scores.max()) if not neg_scores.empty else float("nan")
            gap = float("nan") if pos_scores.empty or neg_scores.empty else float(pos_min - neg_max)
        else:
            n_within_window = 0
            share_within_window = 0.0
            pos_min = float("nan")
            neg_max = float("nan")
            gap = float("nan")

        rows.append(
            {
                "cohort": cohort,
                "subtest": output_col,
                "n_non_missing": n_non_missing,
                "n_pos": int((merged > 0).sum()),
                "n_neg": int((merged < 0).sum()),
                "n_zero": int((merged == 0).sum()),
                "continuity_window": float(continuity_window),
                "n_within_window": n_within_window,
                "share_within_window": share_within_window,
                "min_pos_score": pos_min,
                "max_neg_score": neg_max,
                "continuity_gap": gap,
                "source_pos_count": int(source_counts.get("pos", 0)),
                "source_neg_count": int(source_counts.get("neg", 0)),
                "source_both_count": int(source_counts.get("both", 0)),
                "source_none_count": int(source_counts.get("none", 0)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "cohort",
                "subtest",
                "n_non_missing",
                "n_pos",
                "n_neg",
                "n_zero",
                "continuity_window",
                "n_within_window",
                "share_within_window",
                "min_pos_score",
                "max_neg_score",
                "continuity_gap",
                "source_pos_count",
                "source_neg_count",
                "source_both_count",
                "source_none_count",
            ]
        )
    return pd.DataFrame(rows).sort_values(["cohort", "subtest"]).reset_index(drop=True)


def _cnlsy_indicator_coverage(df: pd.DataFrame, cohort: str, subtests: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_rows = len(df)
    for subtest in subtests:
        if subtest not in df.columns:
            rows.append(
                {
                    "cohort": cohort,
                    "subtest": subtest,
                    "n_age_filtered": n_rows,
                    "n_non_missing": 0,
                    "share_non_missing": 0.0,
                }
            )
            continue
        vals = pd.to_numeric(df[subtest], errors="coerce")
        n_non_missing = int(vals.notna().sum())
        share = (n_non_missing / n_rows) if n_rows > 0 else 0.0
        rows.append(
            {
                "cohort": cohort,
                "subtest": subtest,
                "n_age_filtered": n_rows,
                "n_non_missing": n_non_missing,
                "share_non_missing": share,
            }
        )
    return pd.DataFrame(rows)


def _cnlsy_test_rule_scenarios(df: pd.DataFrame, cohort: str, subtests: list[str]) -> pd.DataFrame:
    if not subtests:
        return pd.DataFrame(columns=["cohort", "min_tests", "n_passing"])
    observed = df[subtests].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
    rows = []
    for k in range(1, len(subtests) + 1):
        rows.append({"cohort": cohort, "min_tests": k, "n_passing": int((observed >= k).sum())})
    return pd.DataFrame(rows)


def _resolve_sample_cfg(cohort_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = cohort_cfg.get("sample_construct", {})
    return cfg if isinstance(cfg, dict) else {}


def _input_path_for_cohort(
    root: Path,
    paths_cfg: dict[str, Any],
    cohort: str,
    cohort_cfg: dict[str, Any],
    source_override: Path | None,
) -> Path:
    if source_override is not None:
        return source_override
    interim_dir = _resolve_path(paths_cfg["interim_dir"], root)
    sample_cfg = _resolve_sample_cfg(cohort_cfg)
    input_file = str(sample_cfg.get("input_file", "panel_extract.csv"))
    return interim_dir / cohort / input_file


def _apply_column_mapping(df: pd.DataFrame, sample_cfg: dict[str, Any]) -> pd.DataFrame:
    mapping = sample_cfg.get("column_map", {})
    if not isinstance(mapping, dict) or not mapping:
        return df
    clean_map = {str(k): str(v) for k, v in mapping.items()}
    return df.rename(columns=clean_map)


def _normalize_birth_year_for_age(birth_year: pd.Series, cohort: str) -> pd.Series:
    if cohort != "nlsy79":
        return birth_year
    # NLSY79 stores year of birth as two-digit years (57-64) in the processed sample.
    return birth_year.where(~birth_year.between(0, 99), birth_year + 1900)


def _coalesce_prefixed_columns(df: pd.DataFrame, *, target: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    slot_cols = sorted(col for col in out.columns if col.startswith(prefix))
    if not slot_cols:
        return out
    candidate_cols = ([target] if target in out.columns else []) + slot_cols
    numeric = out[candidate_cols].apply(pd.to_numeric, errors="coerce")
    out[target] = numeric.bfill(axis=1).iloc[:, 0]
    return out


def _coalesce_named_columns(df: pd.DataFrame, *, target: str, candidates: list[str]) -> pd.DataFrame:
    out = df.copy()
    usable = [col for col in candidates if col in out.columns]
    if not usable:
        return out
    numeric = out[usable].apply(pd.to_numeric, errors="coerce")
    out[target] = numeric.bfill(axis=1).iloc[:, 0]
    return out


def _normalize_employment_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("employment_2019", "employment_2021"):
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        # NLSY97 later DOI fields use 2 for military service without a civilian job.
        out[col] = series.mask(series == 2, 0)
    if "employment_2014" in out.columns:
        series = pd.to_numeric(out["employment_2014"], errors="coerce")
        # CNLSY YA 2014 uses 1=one job, 2=>1 job, 3=not employed.
        out["employment_2014"] = series.mask(series.isin([1, 2]), 1).mask(series == 3, 0)
    return out


def _derive_outcome_age_fields(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    out = df.copy()
    if cohort == "nlsy79" and "birth_year" in out.columns and "interview_year_2000" in out.columns and "age_2000" not in out.columns:
        birth_year = _normalize_birth_year_for_age(pd.to_numeric(out["birth_year"], errors="coerce"), cohort)
        interview_year = pd.to_numeric(out["interview_year_2000"], errors="coerce")
        valid_year = interview_year.where(interview_year.between(1979, 2022))
        if "interview_month_2000" in out.columns:
            interview_month = pd.to_numeric(out["interview_month_2000"], errors="coerce")
            out["interview_month_2000"] = interview_month.where(interview_month.between(1, 12))
        out["interview_year_2000"] = valid_year
        out["age_2000"] = valid_year - birth_year
    if cohort == "nlsy97" and "birth_year" in out.columns and "interview_year_2010" in out.columns and "age_2010" not in out.columns:
        birth_year = pd.to_numeric(out["birth_year"], errors="coerce")
        interview_year = pd.to_numeric(out["interview_year_2010"], errors="coerce")
        valid_year = interview_year.where(interview_year.between(1997, 2023))
        if "interview_month_2010" in out.columns:
            interview_month = pd.to_numeric(out["interview_month_2010"], errors="coerce")
            out["interview_month_2010"] = interview_month.where(interview_month.between(1, 12))
        out["interview_year_2010"] = valid_year
        out["age_2010"] = valid_year - birth_year
    if cohort == "nlsy97" and "birth_year" in out.columns and "interview_year_2011" in out.columns and "age_2011" not in out.columns:
        birth_year = pd.to_numeric(out["birth_year"], errors="coerce")
        interview_year = pd.to_numeric(out["interview_year_2011"], errors="coerce")
        valid_year = interview_year.where(interview_year.between(1997, 2023))
        if "interview_month_2011" in out.columns:
            interview_month = pd.to_numeric(out["interview_month_2011"], errors="coerce")
            out["interview_month_2011"] = interview_month.where(interview_month.between(1, 12))
        out["interview_year_2011"] = valid_year
        out["age_2011"] = valid_year - birth_year
    if cohort == "nlsy97" and "birth_year" in out.columns and "interview_year_2013" in out.columns and "age_2013" not in out.columns:
        birth_year = pd.to_numeric(out["birth_year"], errors="coerce")
        interview_year = pd.to_numeric(out["interview_year_2013"], errors="coerce")
        valid_year = interview_year.where(interview_year.between(1997, 2023))
        if "interview_month_2013" in out.columns:
            interview_month = pd.to_numeric(out["interview_month_2013"], errors="coerce")
            out["interview_month_2013"] = interview_month.where(interview_month.between(1, 12))
        out["interview_year_2013"] = valid_year
        out["age_2013"] = valid_year - birth_year
    if cohort == "nlsy97" and "birth_year" in out.columns and "interview_year_2015" in out.columns and "age_2015" not in out.columns:
        birth_year = pd.to_numeric(out["birth_year"], errors="coerce")
        interview_year = pd.to_numeric(out["interview_year_2015"], errors="coerce")
        valid_year = interview_year.where(interview_year.between(1997, 2023))
        if "interview_month_2015" in out.columns:
            interview_month = pd.to_numeric(out["interview_month_2015"], errors="coerce")
            out["interview_month_2015"] = interview_month.where(interview_month.between(1, 12))
        out["interview_year_2015"] = valid_year
        out["age_2015"] = valid_year - birth_year
    if cohort == "nlsy97" and "birth_year" in out.columns and "interview_year_2017" in out.columns and "age_2017" not in out.columns:
        birth_year = pd.to_numeric(out["birth_year"], errors="coerce")
        interview_year = pd.to_numeric(out["interview_year_2017"], errors="coerce")
        valid_year = interview_year.where(interview_year.between(1997, 2023))
        if "interview_month_2017" in out.columns:
            interview_month = pd.to_numeric(out["interview_month_2017"], errors="coerce")
            out["interview_month_2017"] = interview_month.where(interview_month.between(1, 12))
        out["interview_year_2017"] = valid_year
        out["age_2017"] = valid_year - birth_year
    if cohort == "nlsy97" and "birth_year" in out.columns and "interview_year_2019" in out.columns and "age_2019" not in out.columns:
        birth_year = pd.to_numeric(out["birth_year"], errors="coerce")
        interview_year = pd.to_numeric(out["interview_year_2019"], errors="coerce")
        valid_year = interview_year.where(interview_year.between(1997, 2023))
        if "interview_month_2019" in out.columns:
            interview_month = pd.to_numeric(out["interview_month_2019"], errors="coerce")
            out["interview_month_2019"] = interview_month.where(interview_month.between(1, 12))
        out["interview_year_2019"] = valid_year
        out["age_2019"] = valid_year - birth_year
    if cohort == "nlsy97" and "birth_year" in out.columns and "interview_year_2021" in out.columns and "age_2021" not in out.columns:
        birth_year = pd.to_numeric(out["birth_year"], errors="coerce")
        interview_year = pd.to_numeric(out["interview_year_2021"], errors="coerce")
        valid_year = interview_year.where(interview_year.between(1997, 2023))
        if "interview_month_2021" in out.columns:
            interview_month = pd.to_numeric(out["interview_month_2021"], errors="coerce")
            out["interview_month_2021"] = interview_month.where(interview_month.between(1, 12))
        out["interview_year_2021"] = valid_year
        out["age_2021"] = valid_year - birth_year
    return out


def _construct_for_cohort(
    root: Path,
    cohort: str,
    paths_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    source_override: Path | None,
) -> dict[str, Any]:
    cohort_cfg = load_yaml(root / COHORT_CONFIGS[cohort])
    sample_cfg = _resolve_sample_cfg(cohort_cfg)
    processed_dir = _resolve_path(paths_cfg["processed_dir"], root)
    outputs_dir = _resolve_path(paths_cfg["outputs_dir"], root)
    input_path = _input_path_for_cohort(root, paths_cfg, cohort, cohort_cfg, source_override)
    if not input_path.exists():
        raise FileNotFoundError(f"Input panel not found for {cohort}: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    n_input = len(df)
    df = _apply_column_mapping(df, sample_cfg)
    df = _normalize_employment_indicators(df)
    df = _derive_outcome_age_fields(df, cohort)

    id_col = str(sample_cfg.get("id_col", "person_id"))
    sex_col = str(sample_cfg.get("sex_col", "sex"))
    age_col = sample_cfg.get("age_col")
    age_col = str(age_col) if age_col is not None else None

    for required in (id_col, sex_col):
        if required not in df.columns:
            raise ValueError(f"{cohort}: required column not found: {required}")

    subtests = [_as_label(x) for x in sample_cfg.get("subtests", _default_subtests(cohort, models_cfg))]
    if not subtests:
        raise ValueError(f"{cohort}: no subtests configured")

    missing_codes = set(sample_cfg.get("missing_codes", sorted(DEFAULT_MISSING_CODES)))
    harmonize_cfg = sample_cfg.get("branch_harmonization", {})
    harmonization_enabled = False
    harmonized_outputs: list[str] = []
    if isinstance(harmonize_cfg, dict) and bool(harmonize_cfg.get("enabled", False)):
        harmonization_enabled = True
        pairs = harmonize_cfg.get("pairs", [])
        if not isinstance(pairs, list):
            raise ValueError(f"{cohort}: sample_construct.branch_harmonization.pairs must be a list")
        method = str(harmonize_cfg.get("method", "signed_merge"))
        emit_source_cols = bool(harmonize_cfg.get("emit_source_cols", False))
        implied_decimal_places = harmonize_cfg.get("implied_decimal_places")
        scale_factor = harmonize_cfg.get("scale_factor")
        pair_specs = [p for p in pairs if isinstance(p, dict)]
        if len(pair_specs) != len(pairs):
            raise ValueError(f"{cohort}: sample_construct.branch_harmonization.pairs must contain mapping objects")
        harmonized_outputs = [
            str(spec.get("output", "")).strip() for spec in pair_specs if str(spec.get("output", "")).strip()
        ]
        df = harmonize_pos_neg_pairs(
            df,
            pair_specs,
            method=method,
            missing_codes=missing_codes,
            emit_source_cols=emit_source_cols,
            implied_decimal_places=implied_decimal_places,
            scale_factor=scale_factor,
        )

    auto_col = str(sample_cfg.get("auto_col", "AUTO"))
    shop_col = str(sample_cfg.get("shop_col", "SHOP"))
    if cohort == "nlsy97" and "AS" in subtests and "AS" not in df.columns:
        df = build_auto_shop_composite(df, auto_col=auto_col, shop_col=shop_col, out_col="AS")

    recode_cols = list(dict.fromkeys(subtests + ([age_col] if age_col else [])))
    df = recode_missing_in_columns(df, recode_cols, missing_codes=missing_codes)

    # Recode common missing codes in non-test variables we may want to carry into
    # downstream exploratory summaries (race/SES/outcomes).
    extra_recode_candidates = [
        "race_ethnicity_raw",
        "mother_education",
        "father_education",
        "education_years",
        "highest_degree_ever",
        "parent_education",
        "household_income",
        "net_worth",
        "annual_earnings",
        "employment_2000",
        "degree_2000",
        "occupation_code_2000",
        "interview_month_2000",
        "interview_year_2000",
        "age_2000",
        "sat_math_2007_bin",
        "sat_verbal_2007_bin",
        "act_2007_bin",
        "employment_2011",
        "employment_2019",
        "employment_2021",
        "degree_2021",
        "employment_2014",
        "age_2014",
        "education_years_2014",
        "degree_2014",
        "enrolled_2014",
        "num_current_jobs_2014",
        "total_hours_2014",
        "wage_income_2014_raw",
        "wage_income_2014_best_est",
        "family_income_2014_raw",
        "family_income_2014_best_est",
        "wage_income_2014",
        "family_income_2014",
        "occupation_code_2011",
        "occupation_code_2013",
        "occupation_code_2015",
        "occupation_code_2017",
        "occupation_code_2019",
        "occupation_code_2021",
        "household_income_2019",
        "household_income_2021",
        "annual_earnings_2019",
        "annual_earnings_2021",
        "ui_spells_2019",
        "ui_spells_2021",
        "ui_amount_2019",
        "ui_amount_2021",
        "annual_earnings",
        "interview_month_2010",
        "interview_year_2010",
        "age_2010",
        "interview_month_2011",
        "interview_year_2011",
        "age_2011",
        "interview_month_2013",
        "interview_year_2013",
        "age_2013",
        "interview_month_2015",
        "interview_year_2015",
        "age_2015",
        "interview_month_2017",
        "interview_year_2017",
        "age_2017",
        "interview_month_2019",
        "interview_year_2019",
        "age_2019",
        "interview_month_2021",
        "interview_year_2021",
        "age_2021",
    ]
    slot_recode_cols = [
        col
        for col in df.columns
        if col.startswith("occupation_code_2000_slot")
        or col.startswith("occupation_code_2011_slot")
        or col.startswith("occupation_code_2013_slot")
        or col.startswith("occupation_code_2015_slot")
        or col.startswith("occupation_code_2017_slot")
        or col.startswith("occupation_code_2019_slot")
        or col.startswith("occupation_code_2021_slot")
    ]
    extra_recode_cols = list(dict.fromkeys([col for col in extra_recode_candidates if col in df.columns] + slot_recode_cols))
    if extra_recode_cols:
        df = recode_missing_in_columns(df, extra_recode_cols, missing_codes=missing_codes)
    df = _coalesce_prefixed_columns(df, target="occupation_code_2000", prefix="occupation_code_2000_slot")
    df = _coalesce_prefixed_columns(df, target="occupation_code_2011", prefix="occupation_code_2011_slot")
    df = _coalesce_prefixed_columns(df, target="occupation_code_2013", prefix="occupation_code_2013_slot")
    df = _coalesce_prefixed_columns(df, target="occupation_code_2015", prefix="occupation_code_2015_slot")
    df = _coalesce_prefixed_columns(df, target="occupation_code_2017", prefix="occupation_code_2017_slot")
    df = _coalesce_prefixed_columns(df, target="occupation_code_2019", prefix="occupation_code_2019_slot")
    df = _coalesce_prefixed_columns(df, target="occupation_code_2021", prefix="occupation_code_2021_slot")
    df = _coalesce_named_columns(df, target="wage_income_2014", candidates=["wage_income_2014_raw"])
    df = _coalesce_named_columns(df, target="family_income_2014", candidates=["family_income_2014_raw"])

    if "race_ethnicity_raw" in df.columns and "race_ethnicity_3cat" not in df.columns:
        df["race_ethnicity_3cat"] = harmonize_race_ethnicity_3cat(cohort, df["race_ethnicity_raw"])

    if "mother_education" in df.columns and "parent_education" not in df.columns:
        father = df["father_education"] if "father_education" in df.columns else None
        df["parent_education"] = compute_parent_education(df["mother_education"], father)

    if age_col and age_col in df.columns:
        age_unit = str(sample_cfg.get("age_unit", "years")).strip().lower()
        if age_unit in {"month", "months"}:
            df[age_col] = pd.to_numeric(df[age_col], errors="coerce") / 12.0

    age_range = cohort_cfg.get("expected_age_range", {})
    age_min = age_range.get("min")
    age_max = age_range.get("max")
    if age_col and age_col in df.columns:
        df = filter_age_range(df, age_col=age_col, age_min=age_min, age_max=age_max)
    n_after_age = len(df)

    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    min_tests = int(sample_cfg.get("min_tests", cohort_cfg.get("minimum_tests_observed", len(subtests))))
    if cohort == "cnlsy":
        coverage_path = tables_dir / "cnlsy_indicator_coverage.csv"
        scenarios_path = tables_dir / "cnlsy_test_rule_scenarios.csv"
        _cnlsy_indicator_coverage(df, cohort=cohort, subtests=subtests).to_csv(coverage_path, index=False)
        _cnlsy_test_rule_scenarios(df, cohort=cohort, subtests=subtests).to_csv(scenarios_path, index=False)

        df_filtered = require_min_tests_observed(df, subtests, min_tests=min_tests)
        cnlsy_long = df_filtered.copy()
    else:
        df_filtered = require_complete_tests(df, subtests)
        cnlsy_long = None
    n_after_test_rule = len(df_filtered)

    cfa_df = deduplicate_people(df_filtered, id_col=id_col)
    n_after_dedupe = len(cfa_df)

    if cohort == "nlsy97" and harmonization_enabled:
        unique_outputs = list(dict.fromkeys(harmonized_outputs))
        balance_path = tables_dir / "nlsy97_harmonization_branch_balance_by_sex.csv"
        continuity_path = tables_dir / "nlsy97_harmonization_continuity_near_zero.csv"
        balance = _branch_source_balance_by_sex(cfa_df, cohort=cohort, outputs=unique_outputs, sex_col=sex_col)
        continuity = _continuity_near_zero_summary(cfa_df, cohort=cohort, outputs=unique_outputs)
        balance.to_csv(balance_path, index=False)
        continuity.to_csv(continuity_path, index=False)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{cohort}_cfa.csv"
    cfa_df.to_csv(out_path, index=False)

    if cohort == "cnlsy" and cnlsy_long is not None:
        cnlsy_long.to_csv(processed_dir / "cnlsy_long.csv", index=False)

    n_male, n_female = _normalize_sex_counts(cfa_df[sex_col])
    adequacy = _information_adequacy(
        cohort=cohort,
        sample_cfg=sample_cfg,
        n_total=n_after_dedupe,
        n_male=n_male,
        n_female=n_female,
    )
    age_min_obs = float(cfa_df[age_col].min()) if age_col and age_col in cfa_df.columns and not cfa_df.empty else float("nan")
    age_max_obs = float(cfa_df[age_col].max()) if age_col and age_col in cfa_df.columns and not cfa_df.empty else float("nan")

    counts = {
        "cohort": cohort,
        "n_input": n_input,
        "n_after_age": n_after_age,
        "n_after_test_rule": n_after_test_rule,
        "n_after_dedupe": n_after_dedupe,
        "n_male": n_male,
        "n_female": n_female,
        "age_min": age_min_obs,
        "age_max": age_max_obs,
        "test_rule": f"{min_tests}/{len(subtests)}",
        "input_path": relative_path(root, input_path),
        "output_path": relative_path(root, out_path),
        **adequacy,
    }

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Construct cohort-specific CFA samples.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--source-path", type=Path, help="Optional explicit input file path (single-cohort use).")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")

    cohorts = _cohorts_from_args(args)
    if args.source_path is not None and len(cohorts) != 1:
        raise ValueError("--source-path is only supported when exactly one --cohort is provided.")
    source_override = args.source_path.resolve() if args.source_path else None

    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        row = _construct_for_cohort(
            root=root,
            cohort=cohort,
            paths_cfg=paths_cfg,
            models_cfg=models_cfg,
            source_override=source_override,
        )
        rows.append(row)
        print(f"[ok] {cohort}: n_after_dedupe={row['n_after_dedupe']}")

    outputs_dir = _resolve_path(paths_cfg["outputs_dir"], root)
    sample_counts_path = outputs_dir / "tables" / "sample_counts.csv"
    pd.DataFrame(rows).to_csv(sample_counts_path, index=False)
    print(f"[ok] wrote sample counts: {sample_counts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
