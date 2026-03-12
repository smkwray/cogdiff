#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root, relative_path

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

CATEGORY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "employment": (
        re.compile(r"\bemploy"),
        re.compile(r"\bunemploy"),
        re.compile(r"labor force"),
        re.compile(r"hours worked"),
        re.compile(r"weeks worked"),
        re.compile(r"\bwages?\b"),
        re.compile(r"\bsalary\b"),
        re.compile(r"\bjob\b"),
        re.compile(r"laid off"),
        re.compile(r"worked for pay"),
    ),
    "occupation": (
        re.compile(r"\boccup"),
        re.compile(r"\bindustry\b"),
        re.compile(r"job title"),
        re.compile(r"employer type"),
        re.compile(r"census occupation"),
    ),
    "age": (
        re.compile(r"age at interview"),
        re.compile(r"current age"),
        re.compile(r"\bage of respondent\b"),
        re.compile(r"\bage of child\b"),
        re.compile(r"\bage of youth\b"),
        re.compile(r"\bintdate\b"),
        re.compile(r"interview date"),
        re.compile(r"date of interview"),
    ),
}

OUTPUT_COLUMNS = [
    "cohort",
    "category",
    "raw_id",
    "alias_name",
    "label",
    "in_panel_extract",
    "in_column_map",
    "mapped_name",
    "in_processed",
    "source_sas",
]

LABEL_RE = re.compile(r'^\s*label\s+([A-Z0-9]+)\s*=\s*"(.*)";\s*$')
ALIAS_RE = re.compile(r"^\s*([A-Z0-9]+)\s*=\s*'([^']+)'n")


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _find_sas_file(root: Path, cohort: str) -> Path:
    base = root / "data" / "interim" / cohort / "raw_files"
    candidates = sorted(base.glob("*.sas"))
    if not candidates:
        raise FileNotFoundError(f"Missing SAS source for {cohort}: expected under {base}")
    return candidates[0]


def _scan_sas_candidates(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = LABEL_RE.match(line)
            if match:
                raw_id = match.group(1).strip()
                label = match.group(2).strip()
                entry = out.setdefault(raw_id, {})
                entry["label"] = label
                continue
            alias_match = ALIAS_RE.match(line)
            if alias_match:
                raw_id = alias_match.group(1).strip()
                alias_name = alias_match.group(2).strip()
                entry = out.setdefault(raw_id, {})
                entry["alias_name"] = alias_name
    return out


def _matching_categories(label: str) -> list[str]:
    text = label.strip().lower()
    matches: list[str] = []
    for category, patterns in CATEGORY_PATTERNS.items():
        if not any(pattern.search(text) for pattern in patterns):
            continue
        if category == "age":
            # Keep age/date candidates narrowly focused on interview timing, not
            # employment-status items that mention "at interview date".
            has_age_token = "age" in text
            has_interview_date_token = ("intdate" in text) or ("interview date" in text) or ("date of interview" in text)
            if has_interview_date_token and any(token in text for token in ("working at job", "work at interview", "job at interview")):
                continue
            if not has_age_token and not has_interview_date_token:
                continue
        matches.append(category)
    return matches


def _read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.astype(str).tolist()


def _cohort_rows(root: Path, cohort: str) -> list[dict[str, Any]]:
    cohort_cfg = load_yaml(root / COHORT_CONFIGS[cohort])
    sample_cfg = cohort_cfg.get("sample_construct", {}) if isinstance(cohort_cfg.get("sample_construct", {}), dict) else {}
    column_map = sample_cfg.get("column_map", {}) if isinstance(sample_cfg.get("column_map", {}), dict) else {}
    mapped_names = {str(raw): str(mapped) for raw, mapped in column_map.items()}

    panel_extract_path = root / "data" / "interim" / cohort / "panel_extract.csv"
    processed_path = root / "data" / "processed" / f"{cohort}_cfa_resid.csv"
    if not processed_path.exists():
        processed_path = root / "data" / "processed" / f"{cohort}_cfa.csv"
    panel_cols = set(_read_header(panel_extract_path)) if panel_extract_path.exists() else set()
    processed_cols = set(_read_header(processed_path)) if processed_path.exists() else set()

    sas_path = _find_sas_file(root, cohort)
    scanned = _scan_sas_candidates(sas_path)
    rows: list[dict[str, Any]] = []
    for raw_id, info in sorted(scanned.items()):
        label = str(info.get("label", "")).strip()
        if not label:
            continue
        categories = _matching_categories(label)
        if not categories:
            continue
        mapped_name = mapped_names.get(raw_id, "")
        for category in categories:
            rows.append(
                {
                    "cohort": cohort,
                    "category": category,
                    "raw_id": raw_id,
                    "alias_name": info.get("alias_name", pd.NA),
                    "label": label,
                    "in_panel_extract": raw_id in panel_cols,
                    "in_column_map": raw_id in mapped_names,
                    "mapped_name": mapped_name if mapped_name else pd.NA,
                    "in_processed": bool(mapped_name and mapped_name in processed_cols),
                    "source_sas": relative_path(root, sas_path),
                }
            )
    return rows


def _summary_markdown(project_root_path: Path, audit: pd.DataFrame, output_csv: Path) -> str:
    lines = ["# Upstream Field Audit", "", f"- source table: `{relative_path(project_root_path, output_csv)}`", ""]
    if audit.empty:
        lines.append("- no candidate variables found")
        lines.append("")
        return "\n".join(lines)

    for cohort in sorted(audit["cohort"].dropna().astype(str).unique().tolist()):
        lines.append(f"## {cohort}")
        lines.append("")
        cohort_df = audit.loc[audit["cohort"].astype(str) == cohort].copy()
        for category in ("employment", "occupation", "age"):
            subset = cohort_df.loc[cohort_df["category"].astype(str) == category].copy()
            if subset.empty:
                continue
            lines.append(f"### {category}")
            lines.append("")
            headers = ["raw_id", "alias_name", "mapped_name", "in_panel_extract", "in_processed", "label"]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            ranked = subset.sort_values(
                ["in_processed", "in_panel_extract", "in_column_map", "raw_id"],
                ascending=[False, False, False, True],
            ).head(12)
            for _, row in ranked.iterrows():
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("raw_id", "")),
                            str(row.get("alias_name", "") or ""),
                            str(row.get("mapped_name", "") or ""),
                            str(bool(row.get("in_panel_extract", False))).lower(),
                            str(bool(row.get("in_processed", False))).lower(),
                            str(row.get("label", "")).replace("|", "\\|"),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    return "\n".join(lines)


def run_upstream_field_audit(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path = Path("outputs/tables/upstream_field_audit.csv"),
    markdown_output_path: Path = Path("outputs/tables/upstream_field_audit.md"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        rows.extend(_cohort_rows(root, cohort))

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=OUTPUT_COLUMNS)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS].copy()
    csv_target = output_path if output_path.is_absolute() else root / output_path
    md_target = markdown_output_path if markdown_output_path.is_absolute() else root / markdown_output_path
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_target, index=False)
    md_target.write_text(_summary_markdown(root, out, csv_target), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit upstream raw metadata for employment, occupation, and age candidate fields.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all supported cohorts.")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/upstream_field_audit.csv"))
    parser.add_argument("--markdown-output-path", type=Path, default=Path("outputs/tables/upstream_field_audit.md"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        out = run_upstream_field_audit(
            root=root,
            cohorts=_cohorts_from_args(args),
            output_path=args.output_path,
            markdown_output_path=args.markdown_output_path,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.output_path if args.output_path.is_absolute() else root / args.output_path}")
    print(f"[ok] wrote {args.markdown_output_path if args.markdown_output_path.is_absolute() else root / args.markdown_output_path}")
    print(f"[ok] candidate rows: {len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
