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
from nls_pipeline.links import filter_links_by_relationship, load_links_csv, normalize_family_pairs

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

RAW_LINK_FILES = {
    "nlsy79": "links79_pair_expanded.csv",
    "nlsy97": "links97_pair_expanded.csv",
    "cnlsy": "links_cnlsy_pair_expanded.csv",
}

OUTPUT_LINK_FILES = {
    "nlsy79": "links79_links.csv",
    "nlsy97": "links97_links.csv",
    "cnlsy": "links_cnlsy_links.csv",
}
PLACEHOLDER_LINK_COLUMNS = [
    "SubjectTag",
    "PartnerTag",
    "R",
    "RelationshipPath",
    "ExtendedID",
    "MPUBID",
    "pair_id",
    "family_id",
]


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _load_cohort_cfg(project_root_path: Path, cohort: str) -> dict[str, Any]:
    cohort_cfg_path = project_root_path / COHORT_CONFIGS[cohort]
    if not cohort_cfg_path.exists():
        raise FileNotFoundError(f"Missing cohort config: {cohort_cfg_path}")
    return load_yaml(cohort_cfg_path)


def _resolve_pair_rules(cohort_cfg: dict[str, Any]) -> tuple[float | None, str | None]:
    pair_rules = cohort_cfg.get("pair_rules", {})
    if not isinstance(pair_rules, dict):
        return None, "any"

    relatedness_r = pair_rules.get("relatedness_r")
    if relatedness_r is None:
        relatedness = None
    else:
        relatedness = float(relatedness_r)

    relationship_path = pair_rules.get("relationship_path")
    if relationship_path is None:
        relationship_path = "any"

    return relatedness, str(relationship_path)


def _normalize_raw_links_schema(links: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    if "SubjectTag" not in links.columns and "SubjectTag_S1" in links.columns:
        rename_map["SubjectTag_S1"] = "SubjectTag"
    if "PartnerTag" not in links.columns and "SubjectTag_S2" in links.columns:
        rename_map["SubjectTag_S2"] = "PartnerTag"
    if rename_map:
        links = links.rename(columns=rename_map)
    return links


def _find_raw_links_path(links_dir: Path, cohort: str) -> Path:
    primary = links_dir / RAW_LINK_FILES[cohort]
    if primary.exists():
        return primary

    if cohort == "cnlsy":
        # Fallback per replication note if package does not expose CNLSY expanded links.
        fallback = links_dir / RAW_LINK_FILES["nlsy79"]
        if fallback.exists():
            return fallback

    raise FileNotFoundError(f"Missing raw links source for {cohort}: {primary}")


def _process_cohort(root: Path, links_dir: Path, cohort: str, cohort_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_path = _find_raw_links_path(links_dir, cohort)
    links = pd.read_csv(raw_path)
    links = _normalize_raw_links_schema(links)
    validate_required = ["SubjectTag", "PartnerTag", "R", "RelationshipPath"]
    missing = [col for col in validate_required if col not in links.columns]
    if missing:
        raise ValueError(f"Missing required links columns: {missing}")

    relatedness_r, relationship_path = _resolve_pair_rules(cohort_cfg)
    filtered = filter_links_by_relationship(
        links,
        relatedness_r=relatedness_r,
        relationship_path=relationship_path,
    )
    normalized = normalize_family_pairs(filtered)
    normalized = normalized.sort_values(["family_id", "pair_id", "SubjectTag", "PartnerTag"]).reset_index(drop=True)

    out_file = links_dir / OUTPUT_LINK_FILES[cohort]
    normalized.to_csv(out_file, index=False)

    return {
        "cohort": cohort,
        "raw_path": relative_path(root, raw_path),
        "output_path": relative_path(root, out_file),
        "n_raw": len(links),
        "n_filtered": len(filtered),
        "n_output": len(normalized),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize and export NlsyLinks pair CSVs for each cohort.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all configured cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    paths_cfg = load_yaml(root / "config/paths.yml")
    links_dir = _resolve_path(paths_cfg["links_interim_dir"], root)
    links_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for cohort in _cohorts_from_args(args):
        cohort_cfg = _load_cohort_cfg(root, cohort)
        try:
            rows.append(_process_cohort(root, links_dir, cohort, cohort_cfg))
            print(f"[ok] {cohort}: wrote {OUTPUT_LINK_FILES[cohort]}")
        except FileNotFoundError as exc:
            out_file = links_dir / OUTPUT_LINK_FILES[cohort]
            pd.DataFrame(columns=PLACEHOLDER_LINK_COLUMNS).to_csv(out_file, index=False)
            rows.append(
                {
                    "cohort": cohort,
                    "raw_path": "",
                    "output_path": relative_path(root, out_file),
                    "n_raw": 0,
                    "n_filtered": 0,
                    "n_output": 0,
                    "status": "missing_source",
                    "error": str(exc),
                }
            )
            print(f"[warn] {cohort}: missing raw links source; wrote placeholder {out_file}")

    status_path = links_dir / "link_exports.csv"
    pd.DataFrame(rows).to_csv(status_path, index=False)
    print(f"[ok] wrote status: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
