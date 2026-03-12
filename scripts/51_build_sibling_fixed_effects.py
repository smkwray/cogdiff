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

from nls_pipeline.exploratory import g_proxy, ols_fit, pick_col
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.links import filter_links_by_relationship, load_links_csv
from nls_pipeline.sem import hierarchical_subtests

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

LINK_FILES = {
    "nlsy79": "data/interim/links/links79_links.csv",
    "nlsy97": "data/interim/links/links97_links.csv",
    "cnlsy": "data/interim/links/links_cnlsy_links.csv",
}

OUTCOME_CANDIDATES: dict[str, tuple[str, ...]] = {
    "education": ("education_years", "years_education", "highest_grade_completed", "education"),
    "household_income": ("household_income", "family_income", "hh_income", "income"),
    "net_worth": ("net_worth", "wealth", "net_assets", "assets_net"),
    "earnings": ("annual_earnings", "earnings", "labor_income", "wage_income"),
}

OUTPUT_COLUMNS = [
    "cohort",
    "outcome",
    "status",
    "reason",
    "family_id_col",
    "n_families",
    "n_individuals",
    "beta_g_total",
    "beta_g_within",
    "se_within",
    "p_within",
    "r2_within",
    "source_data",
    "source_links",
]


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _empty_row(
    cohort: str,
    outcome: str,
    reason: str,
    source_data: str,
    source_links: str,
    *,
    family_id_col: str = "",
    n_families: int = 0,
    n_individuals: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "outcome": outcome,
        "status": "not_feasible",
        "reason": reason,
        "family_id_col": family_id_col,
        "n_families": int(n_families),
        "n_individuals": int(n_individuals),
        "source_data": source_data,
        "source_links": source_links,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _family_id_string(row: pd.Series) -> str | None:
    for col in ("family_id", "ExtendedID", "MPUBID", "pair_id"):
        if col not in row.index:
            continue
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return None


def _family_lookup(links: pd.DataFrame) -> tuple[dict[int, str], str]:
    family_id_col = "family_id" if "family_id" in links.columns else ("ExtendedID" if "ExtendedID" in links.columns else "MPUBID")
    lookup: dict[int, str] = {}
    for _, row in links.iterrows():
        fam = _family_id_string(row)
        if fam is None:
            continue
        for col in ("SubjectTag", "PartnerTag"):
            try:
                pid = int(float(row.get(col)))
            except (TypeError, ValueError):
                continue
            lookup[pid] = fam
    return lookup, family_id_col


def _load_links(root: Path, cohort: str) -> tuple[pd.DataFrame | None, str]:
    link_path = root / LINK_FILES[cohort]
    source_links = str(link_path.relative_to(root))
    if not link_path.exists():
        return None, source_links

    links = load_links_csv(link_path)
    cohort_cfg = load_yaml(root / COHORT_CONFIGS[cohort])
    pair_rules = cohort_cfg.get("pair_rules", {}) if isinstance(cohort_cfg.get("pair_rules", {}), dict) else {}
    relatedness = pair_rules.get("relatedness_r", 0.5)
    relationship_path = pair_rules.get("relationship_path", "any")
    try:
        filtered = filter_links_by_relationship(
            links,
            relatedness_r=float(relatedness) if relatedness is not None else None,
            relationship_path=str(relationship_path) if relationship_path is not None else "any",
        )
    except Exception:
        filtered = links.copy()
    return filtered, source_links


def _total_fit(work: pd.DataFrame) -> tuple[dict[str, Any] | None, str | None]:
    design: dict[str, Any] = {"intercept": 1.0, "g": pd.to_numeric(work["g"], errors="coerce")}
    if "age_proxy" in work.columns:
        design["age_proxy"] = pd.to_numeric(work["age_proxy"], errors="coerce")
    x = pd.DataFrame(design, index=work.index)
    return ols_fit(work["outcome"], x)


def _within_fit(work: pd.DataFrame) -> tuple[dict[str, Any] | None, str | None, pd.DataFrame]:
    centered = work.copy()
    centered["y_within"] = centered["outcome"] - centered.groupby("family_id")["outcome"].transform("mean")
    centered["g_within"] = centered["g"] - centered.groupby("family_id")["g"].transform("mean")
    if "age_proxy" in centered.columns:
        centered["age_within"] = centered["age_proxy"] - centered.groupby("family_id")["age_proxy"].transform("mean")
    centered = centered.dropna(subset=["y_within", "g_within"]).copy()
    if centered.empty:
        return None, "empty_after_demeaning", centered
    if float(pd.to_numeric(centered["g_within"], errors="coerce").std(ddof=1)) <= 0.0:
        return None, "no_within_family_g_variation", centered
    design: dict[str, Any] = {"g_within": pd.to_numeric(centered["g_within"], errors="coerce")}
    if "age_within" in centered.columns:
        design["age_within"] = pd.to_numeric(centered["age_within"], errors="coerce")
    x = pd.DataFrame(design, index=centered.index)
    fit, reason = ols_fit(centered["y_within"], x)
    return fit, reason, centered


def run_sibling_fixed_effects(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path = Path("outputs/tables/sibling_fe_g_outcome.csv"),
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_dir = processed_dir if processed_dir.is_absolute() else root / processed_dir

    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        source_data = str(source_path.relative_to(root)) if source_path.exists() else f"{cohort}_cfa_resid_or_cfa.csv"
        links, source_links = _load_links(root, cohort)
        if not source_path.exists():
            for outcome in OUTCOME_CANDIDATES:
                rows.append(_empty_row(cohort, outcome, "missing_source_data", source_data, source_links))
            continue
        if links is None or links.empty:
            for outcome in OUTCOME_CANDIDATES:
                rows.append(_empty_row(cohort, outcome, "missing_or_empty_links", source_data, source_links))
            continue

        df = pd.read_csv(source_path, low_memory=False)
        indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", [])] if cohort == "cnlsy" else hierarchical_subtests(models_cfg)
        df = df.copy()
        df["__g_proxy"] = g_proxy(df, indicators)
        family_lookup, family_id_col = _family_lookup(links)
        if "person_id" not in df.columns:
            for outcome in OUTCOME_CANDIDATES:
                rows.append(_empty_row(cohort, outcome, "missing_person_id", source_data, source_links, family_id_col=family_id_col))
            continue

        person_ids = pd.to_numeric(df["person_id"], errors="coerce")
        df["__family_id"] = person_ids.map(lambda x: family_lookup.get(int(x)) if pd.notna(x) and int(x) in family_lookup else pd.NA)
        if int(df["__family_id"].notna().sum()) == 0:
            for outcome in OUTCOME_CANDIDATES:
                rows.append(_empty_row(cohort, outcome, "no_linked_individuals", source_data, source_links, family_id_col=family_id_col))
            continue

        for outcome, candidates in OUTCOME_CANDIDATES.items():
            out_col = pick_col(df, candidates)
            if out_col is None:
                rows.append(_empty_row(cohort, outcome, "missing_outcome_column", source_data, source_links, family_id_col=family_id_col))
                continue

            selected_cols = ["__family_id", "__g_proxy", out_col]
            if "birth_year" in df.columns:
                selected_cols.append("birth_year")
            work = df[selected_cols].copy()
            work = work.rename(columns={"__family_id": "family_id", "__g_proxy": "g", out_col: "outcome", "birth_year": "age_proxy"})
            work["g"] = pd.to_numeric(work["g"], errors="coerce")
            work["outcome"] = pd.to_numeric(work["outcome"], errors="coerce")
            if "age_proxy" in work.columns:
                work["age_proxy"] = pd.to_numeric(work["age_proxy"], errors="coerce")
            work = work.dropna(subset=["family_id", "g", "outcome"]).copy()
            if work.empty:
                rows.append(_empty_row(cohort, outcome, "no_valid_rows_after_cleaning", source_data, source_links, family_id_col=family_id_col))
                continue

            family_sizes = work.groupby("family_id")["family_id"].transform("size")
            work = work.loc[family_sizes >= 2].copy()
            n_families = int(work["family_id"].nunique())
            n_individuals = int(len(work))
            if n_families < 1 or n_individuals < 3:
                rows.append(
                    _empty_row(
                        cohort,
                        outcome,
                        "insufficient_linked_family_rows",
                        source_data,
                        source_links,
                        family_id_col=family_id_col,
                        n_families=n_families,
                        n_individuals=n_individuals,
                    )
                )
                continue

            total_fit, total_reason = _total_fit(work)
            if total_fit is None:
                rows.append(
                    _empty_row(
                        cohort,
                        outcome,
                        f"total_ols_failed:{total_reason or 'unknown'}",
                        source_data,
                        source_links,
                        family_id_col=family_id_col,
                        n_families=n_families,
                        n_individuals=n_individuals,
                    )
                )
                continue

            within_fit, within_reason, centered = _within_fit(work)
            if within_fit is None:
                rows.append(
                    _empty_row(
                        cohort,
                        outcome,
                        f"within_ols_failed:{within_reason or 'unknown'}",
                        source_data,
                        source_links,
                        family_id_col=family_id_col,
                        n_families=n_families,
                        n_individuals=int(len(centered)) if not centered.empty else n_individuals,
                    )
                )
                continue

            beta_total = total_fit["beta"]
            beta_within = within_fit["beta"]
            se_within = within_fit["se"]
            p_within = within_fit["p"]
            row = {
                "cohort": cohort,
                "outcome": outcome,
                "status": "computed",
                "reason": pd.NA,
                "family_id_col": family_id_col,
                "n_families": n_families,
                "n_individuals": int(within_fit["n_used"]),
                "beta_g_total": float(beta_total[1]),
                "beta_g_within": float(beta_within[0]),
                "se_within": float(se_within[0]),
                "p_within": float(p_within[0]) if math.isfinite(float(p_within[0])) else pd.NA,
                "r2_within": float(within_fit["r2"]) if math.isfinite(float(within_fit["r2"])) else pd.NA,
                "source_data": source_data,
                "source_links": source_links,
            }
            for col in OUTPUT_COLUMNS:
                row.setdefault(col, pd.NA)
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=OUTPUT_COLUMNS)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS].copy()
    target = output_path if output_path.is_absolute() else root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build sibling fixed-effects regressions of outcomes on g_proxy.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/sibling_fe_g_outcome.csv"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        out = run_sibling_fixed_effects(
            root=root,
            cohorts=_cohorts_from_args(args),
            output_path=args.output_path,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.output_path if args.output_path.is_absolute() else root / args.output_path}")
    print(f"[ok] computed rows: {int((out['status'] == 'computed').sum()) if 'status' in out.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
