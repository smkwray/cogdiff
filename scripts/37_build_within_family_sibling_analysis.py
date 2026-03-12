#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

LINKS_PATH_BY_COHORT = {
    "nlsy79": "data/interim/links/links79_links.csv",
}

SUMMARY_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "n_total",
    "n_male",
    "n_female",
    "n_pairs_total",
    "n_pairs_opposite_sex",
    "between_d_g",
    "between_log_vr_g",
    "between_mean_male",
    "between_mean_female",
    "between_pooled_sd",
    "within_pair_mean_diff",
    "within_pair_SE_diff",
    "within_pair_ci_low",
    "within_pair_ci_high",
    "within_pair_d_z",
    "within_pair_d_pooled",
    "within_between_delta_d",
    "source_data",
    "source_links",
]

PAIR_COLUMNS = [
    "cohort",
    "family_id",
    "pair_id",
    "male_person_id",
    "female_person_id",
    "male_g",
    "female_g",
    "diff_male_minus_female",
    "source_data",
    "source_links",
]


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _as_label(value: Any) -> str:
    if isinstance(value, bool):
        return "NO" if value is False else "YES"
    return str(value)


def _normalize_sex(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"m", "male", "1", "man", "boy"}:
        return "male"
    if token in {"f", "female", "2", "woman", "girl"}:
        return "female"
    return "unknown"


def _zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mean = vals.mean(skipna=True)
    sd = vals.std(skipna=True, ddof=1)
    if pd.isna(sd) or float(sd) <= 0.0:
        return pd.Series([pd.NA] * len(vals), index=vals.index, dtype="float64")
    return (vals - mean) / sd


def _g_proxy(df: pd.DataFrame, indicators: list[str]) -> pd.Series:
    existing = [col for col in indicators if col in df.columns]
    if not existing:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
    z = pd.DataFrame({col: _zscore(df[col]) for col in existing}, index=df.index)
    return z.mean(axis=1, skipna=False)


def _empty_summary_row(cohort: str, reason: str, source_data: str, source_links: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "status": "not_feasible",
        "reason": reason,
        "n_total": 0,
        "n_male": 0,
        "n_female": 0,
        "n_pairs_total": 0,
        "n_pairs_opposite_sex": 0,
        "source_data": source_data,
        "source_links": source_links,
    }
    for col in SUMMARY_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _between_stats(values: pd.Series, sex: pd.Series) -> dict[str, Any]:
    clean = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "sex": sex.map(_normalize_sex)}).dropna()
    clean = clean[clean["sex"].isin({"male", "female"})].copy()

    male = clean.loc[clean["sex"] == "male", "value"]
    female = clean.loc[clean["sex"] == "female", "value"]
    n_male = int(len(male))
    n_female = int(len(female))
    n_total = int(len(clean))
    if n_male < 2 or n_female < 2:
        return {
            "n_total": n_total,
            "n_male": n_male,
            "n_female": n_female,
            "between_d_g": pd.NA,
            "between_log_vr_g": pd.NA,
            "between_mean_male": pd.NA,
            "between_mean_female": pd.NA,
            "between_pooled_sd": pd.NA,
        }

    mean_male = float(male.mean())
    mean_female = float(female.mean())
    var_male = float(male.var(ddof=1))
    var_female = float(female.var(ddof=1))
    pooled_sd = math.sqrt((var_male + var_female) / 2.0) if var_male > 0.0 and var_female > 0.0 else float("nan")
    d_g = (mean_male - mean_female) / pooled_sd if math.isfinite(pooled_sd) and pooled_sd > 0.0 else float("nan")
    log_vr = math.log(var_male / var_female) if var_male > 0.0 and var_female > 0.0 else float("nan")

    return {
        "n_total": n_total,
        "n_male": n_male,
        "n_female": n_female,
        "between_d_g": d_g if math.isfinite(d_g) else pd.NA,
        "between_log_vr_g": log_vr if math.isfinite(log_vr) else pd.NA,
        "between_mean_male": mean_male,
        "between_mean_female": mean_female,
        "between_pooled_sd": pooled_sd if math.isfinite(pooled_sd) else pd.NA,
    }


def _pair_rows(
    *,
    cohort: str,
    links: pd.DataFrame,
    person_map: pd.DataFrame,
    source_data: str,
    source_links: str,
) -> pd.DataFrame:
    lookup = person_map.set_index("person_id")[["sex_norm", "g_proxy"]]
    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()

    for _, row in links.iterrows():
        try:
            sid = int(float(row.get("SubjectTag")))
            pid = int(float(row.get("PartnerTag")))
        except (TypeError, ValueError):
            continue
        key = tuple(sorted((sid, pid)))
        if sid == pid or key in seen_pairs:
            continue
        seen_pairs.add(key)
        if sid not in lookup.index or pid not in lookup.index:
            continue

        s_row = lookup.loc[sid]
        p_row = lookup.loc[pid]
        s_sex = str(s_row["sex_norm"])
        p_sex = str(p_row["sex_norm"])
        if {s_sex, p_sex} != {"male", "female"}:
            continue
        s_val = pd.to_numeric(pd.Series([s_row["g_proxy"]]), errors="coerce").iloc[0]
        p_val = pd.to_numeric(pd.Series([p_row["g_proxy"]]), errors="coerce").iloc[0]
        if pd.isna(s_val) or pd.isna(p_val):
            continue

        if s_sex == "male":
            male_id, female_id = sid, pid
            male_g, female_g = float(s_val), float(p_val)
        else:
            male_id, female_id = pid, sid
            male_g, female_g = float(p_val), float(s_val)

        rows.append(
            {
                "cohort": cohort,
                "family_id": str(row.get("family_id", "")),
                "pair_id": str(row.get("pair_id", f"{sid}|{pid}")),
                "male_person_id": male_id,
                "female_person_id": female_id,
                "male_g": male_g,
                "female_g": female_g,
                "diff_male_minus_female": float(male_g - female_g),
                "source_data": source_data,
                "source_links": source_links,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=PAIR_COLUMNS)
    for col in PAIR_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[PAIR_COLUMNS].copy()


def run_within_family_sibling_analysis(
    *,
    root: Path,
    cohorts: list[str],
    summary_output_path: Path = Path("outputs/tables/within_family_sibling_analysis.csv"),
    pairs_output_path: Path = Path("outputs/tables/within_family_sibling_pairs.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_cfg = load_yaml(root / "config" / "paths.yml")
    models_cfg = load_yaml(root / "config" / "models.yml")
    processed_dir = _resolve_path(paths_cfg.get("processed_dir", "data/processed"), root)

    summary_rows: list[dict[str, Any]] = []
    all_pairs: list[pd.DataFrame] = []

    for cohort in cohorts:
        config_rel = COHORT_CONFIGS.get(cohort)
        if config_rel is None:
            summary_rows.append(_empty_summary_row(cohort, "unknown_cohort", "", ""))
            continue

        cohort_cfg = load_yaml(root / config_rel)
        sample_cfg = cohort_cfg.get("sample_construct", {}) if isinstance(cohort_cfg.get("sample_construct", {}), dict) else {}
        sex_col = str(sample_cfg.get("sex_col", "sex"))
        id_col = str(sample_cfg.get("id_col", "person_id"))
        indicators = [_as_label(x) for x in sample_cfg.get("subtests", [])]
        if not indicators:
            indicators = [_as_label(x) for x in (models_cfg.get("cnlsy_single_factor", []) if cohort == "cnlsy" else hierarchical_subtests(models_cfg))]

        data_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not data_path.exists():
            data_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(data_path.relative_to(root)) if data_path.exists() else ""

        links_rel = LINKS_PATH_BY_COHORT.get(cohort)
        links_path = (root / links_rel) if links_rel else None
        source_links = links_rel or ""

        if not data_path.exists():
            summary_rows.append(_empty_summary_row(cohort, "missing_source_data", source_data, source_links))
            continue
        if links_path is None or not links_path.exists():
            summary_rows.append(_empty_summary_row(cohort, "missing_links_file", source_data, source_links))
            continue

        df = pd.read_csv(data_path, low_memory=False)
        if id_col not in df.columns or sex_col not in df.columns:
            summary_rows.append(_empty_summary_row(cohort, "missing_id_or_sex_column", source_data, source_links))
            continue

        g_proxy = _g_proxy(df, indicators)
        person_map = pd.DataFrame(
            {
                "person_id": pd.to_numeric(df[id_col], errors="coerce"),
                "sex_norm": df[sex_col].map(_normalize_sex),
                "g_proxy": pd.to_numeric(g_proxy, errors="coerce"),
            }
        ).dropna(subset=["person_id"]).copy()
        person_map["person_id"] = person_map["person_id"].astype(int)

        between = _between_stats(person_map["g_proxy"], person_map["sex_norm"])

        links = pd.read_csv(links_path, low_memory=False)
        if "R" in links.columns:
            r_target = float(cohort_cfg.get("pair_rules", {}).get("relatedness_r", 0.5))
            links = links[pd.to_numeric(links["R"], errors="coerce").round(6) == round(r_target, 6)].copy()
        rel_path = str(cohort_cfg.get("pair_rules", {}).get("relationship_path", "any")).strip()
        if rel_path and rel_path.lower() != "any" and "RelationshipPath" in links.columns:
            links = links[links["RelationshipPath"].astype(str) == rel_path].copy()

        n_pairs_total = int(len(links))
        pair_df = _pair_rows(
            cohort=cohort,
            links=links,
            person_map=person_map,
            source_data=source_data,
            source_links=source_links,
        )
        n_pairs_opposite = int(len(pair_df))

        if n_pairs_opposite < 2:
            summary = _empty_summary_row(cohort, "insufficient_opposite_sex_pairs", source_data, source_links) | between | {
                "n_pairs_total": n_pairs_total,
                "n_pairs_opposite_sex": n_pairs_opposite,
            }
            summary_rows.append(summary)
            all_pairs.append(pair_df)
            continue

        diffs = pd.to_numeric(pair_df["diff_male_minus_female"], errors="coerce").dropna()
        n_diffs = int(len(diffs))
        if n_diffs < 2:
            summary = _empty_summary_row(cohort, "insufficient_nonmissing_pair_diffs", source_data, source_links) | between | {
                "n_pairs_total": n_pairs_total,
                "n_pairs_opposite_sex": n_pairs_opposite,
            }
            summary_rows.append(summary)
            all_pairs.append(pair_df)
            continue

        mean_diff = float(diffs.mean())
        sd_diff = float(diffs.std(ddof=1))
        se_diff = float(sd_diff / math.sqrt(n_diffs)) if sd_diff > 0.0 else float("nan")
        ci_low = float(mean_diff - 1.96 * se_diff) if math.isfinite(se_diff) else float("nan")
        ci_high = float(mean_diff + 1.96 * se_diff) if math.isfinite(se_diff) else float("nan")
        within_d_z = float(mean_diff / sd_diff) if sd_diff > 0.0 else float("nan")

        pooled_sd = between.get("between_pooled_sd")
        pooled_sd_f = float(pooled_sd) if pd.notna(pooled_sd) else float("nan")
        within_d_pooled = float(mean_diff / pooled_sd_f) if math.isfinite(pooled_sd_f) and pooled_sd_f > 0.0 else float("nan")
        between_d = between.get("between_d_g")
        between_d_f = float(between_d) if pd.notna(between_d) else float("nan")
        delta_d = float(within_d_pooled - between_d_f) if math.isfinite(within_d_pooled) and math.isfinite(between_d_f) else float("nan")

        summary = {
            "cohort": cohort,
            "status": "computed",
            "reason": pd.NA,
            "n_pairs_total": n_pairs_total,
            "n_pairs_opposite_sex": n_pairs_opposite,
            "within_pair_mean_diff": mean_diff,
            "within_pair_SE_diff": se_diff if math.isfinite(se_diff) else pd.NA,
            "within_pair_ci_low": ci_low if math.isfinite(ci_low) else pd.NA,
            "within_pair_ci_high": ci_high if math.isfinite(ci_high) else pd.NA,
            "within_pair_d_z": within_d_z if math.isfinite(within_d_z) else pd.NA,
            "within_pair_d_pooled": within_d_pooled if math.isfinite(within_d_pooled) else pd.NA,
            "within_between_delta_d": delta_d if math.isfinite(delta_d) else pd.NA,
            "source_data": source_data,
            "source_links": source_links,
            **between,
        }
        for col in SUMMARY_COLUMNS:
            summary.setdefault(col, pd.NA)
        summary_rows.append(summary)
        all_pairs.append(pair_df)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
    for col in SUMMARY_COLUMNS:
        if col not in summary_df.columns:
            summary_df[col] = pd.NA
    summary_df = summary_df[SUMMARY_COLUMNS].copy()

    pairs_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(columns=PAIR_COLUMNS)
    for col in PAIR_COLUMNS:
        if col not in pairs_df.columns:
            pairs_df[col] = pd.NA
    pairs_df = pairs_df[PAIR_COLUMNS].copy()

    summary_output = summary_output_path if summary_output_path.is_absolute() else root / summary_output_path
    pairs_output = pairs_output_path if pairs_output_path.is_absolute() else root / pairs_output_path
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    pairs_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output, index=False)
    pairs_df.to_csv(pairs_output, index=False)
    return summary_df, pairs_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build within-family opposite-sex pair analysis and compare to between-family sex difference."
    )
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=Path("outputs/tables/within_family_sibling_analysis.csv"),
        help="Summary output CSV path (relative to project-root if not absolute).",
    )
    parser.add_argument(
        "--pairs-output-path",
        type=Path,
        default=Path("outputs/tables/within_family_sibling_pairs.csv"),
        help="Pair-level output CSV path (relative to project-root if not absolute).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    cohorts = _cohorts_from_args(args)
    try:
        summary, pairs = run_within_family_sibling_analysis(
            root=root,
            cohorts=cohorts,
            summary_output_path=args.summary_output_path,
            pairs_output_path=args.pairs_output_path,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    summary_output = args.summary_output_path if args.summary_output_path.is_absolute() else root / args.summary_output_path
    pairs_output = args.pairs_output_path if args.pairs_output_path.is_absolute() else root / args.pairs_output_path
    computed = int((summary["status"] == "computed").sum()) if "status" in summary.columns else 0
    print(f"[ok] wrote {summary_output}")
    print(f"[ok] wrote {pairs_output}")
    print(f"[ok] computed cohort rows: {computed}")
    print(f"[ok] pair rows: {len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
