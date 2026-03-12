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

from nls_pipeline.exploratory import g_proxy, ols_fit, pick_col, safe_corr
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.links import filter_links_by_relationship, load_links_csv
from nls_pipeline.sem import hierarchical_subtests

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
}

LINK_FILES = {
    "nlsy79": "data/interim/links/links79_links.csv",
    "nlsy97": "data/interim/links/links97_links.csv",
}

OUTCOME_CANDIDATES: dict[str, dict[str, tuple[str, ...]]] = {
    "nlsy79": {
        "education": ("education_years",),
        "household_income": ("household_income",),
        "net_worth": ("net_worth",),
        "earnings": ("annual_earnings",),
        "employment": ("employment_2000",),
    },
    "nlsy97": {
        "education": ("education_years",),
        "household_income": ("household_income_2021", "household_income_2019", "household_income"),
        "net_worth": ("net_worth",),
        "earnings": ("annual_earnings_2021", "annual_earnings_2019"),
        "employment": ("employment_2021", "employment_2019", "employment_2011"),
    },
}

OUTPUT_COLUMNS = [
    "cohort",
    "outcome",
    "status",
    "reason",
    "outcome_col",
    "n_pairs",
    "n_families",
    "mean_abs_g_diff",
    "mean_abs_outcome_diff",
    "corr_abs_diff",
    "beta_abs_g_diff",
    "SE_beta_abs_g_diff",
    "p_value_abs_g_diff",
    "r2",
    "source_data",
    "source_links",
]


def _empty_row(
    cohort: str,
    outcome: str,
    reason: str,
    source_data: str,
    source_links: str,
    *,
    outcome_col: str = "",
    n_pairs: int = 0,
    n_families: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "outcome": outcome,
        "status": "not_feasible",
        "reason": reason,
        "outcome_col": outcome_col,
        "n_pairs": int(n_pairs),
        "n_families": int(n_families),
        "source_data": source_data,
        "source_links": source_links,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


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


def _subject_partner_cols(links: pd.DataFrame) -> tuple[str | None, str | None]:
    if {"SubjectTag", "PartnerTag"}.issubset(links.columns):
        return "SubjectTag", "PartnerTag"
    if {"SubjectTag_S1", "SubjectTag_S2"}.issubset(links.columns):
        return "SubjectTag_S1", "SubjectTag_S2"
    return None, None


def _family_col(links: pd.DataFrame) -> str | None:
    for col in ("family_id", "ExtendedID", "MPUBID"):
        if col in links.columns:
            return col
    return None


def run_sibling_discordance(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path = Path("outputs/tables/sibling_discordance.csv"),
    min_pairs: int = 30,
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
            for outcome in OUTCOME_CANDIDATES[cohort]:
                rows.append(_empty_row(cohort, outcome, "missing_source_data", source_data, source_links))
            continue
        if links is None or links.empty:
            for outcome in OUTCOME_CANDIDATES[cohort]:
                rows.append(_empty_row(cohort, outcome, "missing_or_empty_links", source_data, source_links))
            continue

        left_col, right_col = _subject_partner_cols(links)
        fam_col = _family_col(links)
        if left_col is None or right_col is None:
            for outcome in OUTCOME_CANDIDATES[cohort]:
                rows.append(_empty_row(cohort, outcome, "missing_pair_identifier_columns", source_data, source_links))
            continue

        df = pd.read_csv(source_path, low_memory=False)
        if "person_id" not in df.columns:
            for outcome in OUTCOME_CANDIDATES[cohort]:
                rows.append(_empty_row(cohort, outcome, "missing_person_id", source_data, source_links))
            continue

        indicators = hierarchical_subtests(models_cfg)
        df = df.copy()
        df["__g_proxy"] = g_proxy(df, indicators)
        pid = pd.to_numeric(df["person_id"], errors="coerce")
        df = df.loc[pid.notna()].copy()
        df["person_id"] = pid.loc[df.index].astype(int)

        links = links.copy()
        links[left_col] = pd.to_numeric(links[left_col], errors="coerce")
        links[right_col] = pd.to_numeric(links[right_col], errors="coerce")
        links = links.dropna(subset=[left_col, right_col]).copy()
        links[left_col] = links[left_col].astype(int)
        links[right_col] = links[right_col].astype(int)
        links["pair_id"] = links.apply(
            lambda r: f"{min(int(r[left_col]), int(r[right_col]))}|{max(int(r[left_col]), int(r[right_col]))}",
            axis=1,
        )
        links = links.drop_duplicates(subset=["pair_id"]).copy()

        base_cols = ["person_id", "__g_proxy"]
        for outcome, candidates in OUTCOME_CANDIDATES[cohort].items():
            out_col = pick_col(df, candidates)
            if out_col is None:
                rows.append(_empty_row(cohort, outcome, "missing_outcome_column", source_data, source_links))
                continue

            left = df[base_cols + [out_col]].rename(
                columns={"person_id": left_col, "__g_proxy": "g_left", out_col: "outcome_left"}
            )
            right = df[base_cols + [out_col]].rename(
                columns={"person_id": right_col, "__g_proxy": "g_right", out_col: "outcome_right"}
            )
            pair_df = links.merge(left, on=left_col, how="inner").merge(right, on=right_col, how="inner")
            if fam_col and fam_col in pair_df.columns:
                family_series = pair_df[fam_col]
            else:
                family_series = pd.Series(pd.NA, index=pair_df.index)

            work = pd.DataFrame(
                {
                    "pair_id": pair_df["pair_id"],
                    "family_id": family_series,
                    "g_left": pd.to_numeric(pair_df["g_left"], errors="coerce"),
                    "g_right": pd.to_numeric(pair_df["g_right"], errors="coerce"),
                    "outcome_left": pd.to_numeric(pair_df["outcome_left"], errors="coerce"),
                    "outcome_right": pd.to_numeric(pair_df["outcome_right"], errors="coerce"),
                }
            ).dropna()
            if work.empty:
                rows.append(_empty_row(cohort, outcome, "no_valid_pairs_after_cleaning", source_data, source_links, outcome_col=out_col))
                continue

            work["abs_g_diff"] = (work["g_left"] - work["g_right"]).abs()
            work["abs_outcome_diff"] = (work["outcome_left"] - work["outcome_right"]).abs()
            work = work.loc[work["abs_g_diff"].notna() & work["abs_outcome_diff"].notna()].copy()
            n_pairs = int(len(work))
            n_families = int(work["family_id"].dropna().astype(str).nunique()) if fam_col else 0
            if n_pairs < min_pairs:
                rows.append(
                    _empty_row(
                        cohort,
                        outcome,
                        "insufficient_pairs",
                        source_data,
                        source_links,
                        outcome_col=out_col,
                        n_pairs=n_pairs,
                        n_families=n_families,
                    )
                )
                continue

            x = pd.DataFrame({"intercept": 1.0, "abs_g_diff": work["abs_g_diff"]}, index=work.index)
            fit, reason = ols_fit(work["abs_outcome_diff"], x)
            if fit is None:
                rows.append(
                    _empty_row(
                        cohort,
                        outcome,
                        f"ols_failed:{reason or 'unknown'}",
                        source_data,
                        source_links,
                        outcome_col=out_col,
                        n_pairs=n_pairs,
                        n_families=n_families,
                    )
                )
                continue

            row = {
                "cohort": cohort,
                "outcome": outcome,
                "status": "computed",
                "reason": pd.NA,
                "outcome_col": out_col,
                "n_pairs": n_pairs,
                "n_families": n_families,
                "mean_abs_g_diff": float(work["abs_g_diff"].mean()),
                "mean_abs_outcome_diff": float(work["abs_outcome_diff"].mean()),
                "corr_abs_diff": safe_corr(work["abs_g_diff"], work["abs_outcome_diff"]),
                "beta_abs_g_diff": float(fit["beta"][1]),
                "SE_beta_abs_g_diff": float(fit["se"][1]),
                "p_value_abs_g_diff": float(fit["p"][1]),
                "r2": float(fit["r2"]),
                "source_data": source_data,
                "source_links": source_links,
            }
            for col in OUTPUT_COLUMNS:
                row.setdefault(col, pd.NA)
            rows.append(row)

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build sibling-pair discordance models beyond education.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/sibling_discordance.csv"))
    parser.add_argument("--min-pairs", type=int, default=30)
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    cohorts = list(COHORT_CONFIGS.keys()) if args.all or not args.cohort else args.cohort
    out = run_sibling_discordance(
        root=Path(args.project_root),
        cohorts=cohorts,
        output_path=args.output_path,
        min_pairs=int(args.min_pairs),
    )
    print(f"[ok] sibling discordance rows computed: {int((out['status'] == 'computed').sum())}")
    print(f"[ok] wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
