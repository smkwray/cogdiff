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

from nls_pipeline.io import project_root

DEFAULT_CLAIM_VERDICTS_PATH = Path("outputs/tables/claim_verdicts.csv")
DEFAULT_ANALYSIS_TIERS_PATH = Path("outputs/tables/analysis_tiers.csv")
DEFAULT_SPECIFICATION_STABILITY_PATH = Path("outputs/tables/specification_stability_summary.csv")
DEFAULT_INFERENCE_CI_COHERENCE_PATH = Path("outputs/tables/inference_ci_coherence.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/tables/publication_report_sections.md")


def _resolve_path(root: Path, value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _safe_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", ""}:
        return False
    return None


def _cohort_count(frame: pd.DataFrame) -> int:
    if frame.empty or "cohort" not in frame.columns:
        if "claim_id" in frame.columns:
            cohorts = frame["claim_id"].astype(str).str.split(":", n=1).str[0]
            cohorts = cohorts.replace({"nan": pd.NA, "": pd.NA}).dropna()
            return int(cohorts.nunique())
        if "claim_label" in frame.columns:
            cohorts = frame["claim_label"].astype(str).str.split(":", n=1).str[1]
            cohorts = cohorts.str.split(":", n=1).str[0].replace({"nan": pd.NA, "": pd.NA}).dropna()
            return int(cohorts.nunique())
        return 0
    return int(
        frame["cohort"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": pd.NA})
        .dropna()
        .nunique()
    )


def _normalize_verdict(value: Any) -> str:
    text = _safe_text(value).strip().lower().replace(" ", "_")
    if text in {"not_confirmed", "notconfirmed", "no", "reject", "rejected"}:
        return "not_confirmed"
    if text in {"confirmed", "yes", "true", "pass", "ok", "confirmed_with caveat", "confirmed_with_notes"}:
        return "confirmed"
    return text


def _methods_snapshot(
    claim_verdicts: pd.DataFrame,
    analysis_tiers: pd.DataFrame,
    specification_summary: pd.DataFrame,
    inference_ci: pd.DataFrame,
    claim_exists: bool,
    analysis_exists: bool,
    specification_exists: bool,
    inference_exists: bool,
) -> list[str]:
    lines = ["## Methods snapshot", ""]

    if claim_exists:
        lines.append(f"- claim_verdicts.csv: {len(claim_verdicts)} row(s); {_cohort_count(claim_verdicts)} cohort(s)")
    else:
        lines.append("- Missing input table: outputs/tables/claim_verdicts.csv")
        lines.append("- claim_verdicts.csv: unavailable; claims summary will note missing input")

    if analysis_exists:
        lines.append(f"- analysis_tiers.csv: {len(analysis_tiers)} row(s); {_cohort_count(analysis_tiers)} cohort(s)")
    else:
        lines.append("- Missing input table: outputs/tables/analysis_tiers.csv")
        lines.append("- analysis_tiers.csv: unavailable; exclusion inference is limited to available claims")

    if specification_exists:
        lines.append(
            f"- specification_stability_summary.csv: {len(specification_summary)} row(s); {_cohort_count(specification_summary)} cohort(s)"
        )
    else:
        lines.append("- Missing input table: outputs/tables/specification_stability_summary.csv")
        lines.append("- specification_stability_summary.csv: unavailable; robustness eligibility notes are limited")

    if inference_exists:
        lines.append(f"- inference_ci_coherence.csv: {len(inference_ci)} row(s); {_cohort_count(inference_ci)} cohort(s)")
    else:
        lines.append("- Missing input table: outputs/tables/inference_ci_coherence.csv")
        lines.append("- inference_ci_coherence.csv: unavailable; CI checks cannot be reviewed")

    if not claim_verdicts.empty:
        verdict = claim_verdicts.get("verdict", pd.Series(dtype=object)).map(_normalize_verdict)
        for label in ("confirmed", "not_confirmed", "inconclusive"):
            lines.append(f"- claim verdict count ({label}): {(verdict == label).sum()}")

    if not analysis_tiers.empty and "blocked_confirmatory" in analysis_tiers.columns and "analysis_tier" in analysis_tiers.columns:
        blocked = analysis_tiers[analysis_tiers["blocked_confirmatory"].map(_safe_bool).fillna(False)]
        lines.append(f"- confirmatory exclusions in analysis_tiers: {len(blocked)} row(s)")

    if not specification_summary.empty:
        if "robust_claim_eligible" in specification_summary.columns:
            elig = specification_summary["robust_claim_eligible"].map(_safe_bool)
            lines.append(f"- specification_stability_summary robust_claim_eligible true: {int(elig.fillna(False).sum())}")
            lines.append(f"- specification_stability_summary robust_claim_eligible false: {int((elig == False).sum())}")

    if not inference_ci.empty:
        status = inference_ci.get("status", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
        issue = inference_ci.get("issue", pd.Series(dtype=object)).astype(str)
        computed = (status == "computed").sum()
        lines.append(f"- inference_ci_coherence computed rows: {int(computed)}")
        if "ci_contains_estimate" in inference_ci.columns:
            ci_ok = inference_ci.loc[status == "computed", "ci_contains_estimate"].eq(True).sum()
            lines.append(f"- inferred CI coherence pass rows: {int(ci_ok)}")
        if "issue" in inference_ci.columns:
            missing_text = (issue == "computed_missing_ci_or_estimate").sum()
            lines.append(f"- inference_ci_coherence rows with missing CI/estimate fields: {int(missing_text)}")

    if not lines:
        lines.append("- No usable source tables were available to populate this snapshot.")

    return lines


def _claim_verdict_summary(claim_verdicts: pd.DataFrame, exists: bool) -> list[str]:
    lines = ["## Claim verdict summary", ""]
    if not exists:
        lines.append("- Missing claim_verdicts.csv; cannot draft claim verdict summary.")
        return lines
    if claim_verdicts.empty:
        lines.append("- claim_verdicts.csv was present but empty.")
        return lines

    verdict = claim_verdicts.get("verdict", pd.Series(dtype=object)).map(_normalize_verdict)
    for label in ("confirmed", "not_confirmed", "inconclusive"):
        lines.append(f"- {label.replace('_', ' ').title()}: {int((verdict == label).sum())}")

    confirmed = claim_verdicts.loc[verdict == "confirmed", "claim_id"].dropna().astype(str).tolist()[:3]
    if confirmed:
        lines.append("- Example confirmed claims: " + ", ".join(sorted(confirmed)))

    not_confirmed = claim_verdicts.loc[verdict == "not_confirmed", "claim_id"].dropna().astype(str).tolist()[:3]
    if not_confirmed:
        lines.append("- Example non-confirmed claims: " + ", ".join(sorted(not_confirmed)))

    return lines


def _exclusions_and_inconclusive(
    analysis_tiers: pd.DataFrame,
    claim_verdicts: pd.DataFrame,
    specification_summary: pd.DataFrame,
    analysis_exists: bool,
    claim_exists: bool,
    spec_exists: bool,
) -> list[str]:
    lines = ["## Exclusions / inconclusive notes", ""]
    added = False

    if not claim_exists:
        lines.append("- claim_verdicts.csv missing: can only provide source-level exclusion notices where available.")
        added = True

    if claim_exists and not claim_verdicts.empty:
        verdict_series = claim_verdicts.get("verdict")
        if isinstance(verdict_series, pd.Series):
            inrows = claim_verdicts.loc[verdict_series.map(lambda v: _normalize_verdict(v) == "inconclusive")]
        else:
            inrows = pd.DataFrame()
        if not inrows.empty:
            added = True
            lines.append(f"- Inconclusive claim rows: {len(inrows)}")
            for row in inrows.head(5).to_dict("records"):
                lines.append(
                    f"  - {row.get('claim_id')}: {_safe_text(row.get('reason')) or 'no_reason'}"
                )

    if analysis_exists and not analysis_tiers.empty and "blocked_confirmatory" in analysis_tiers.columns:
        blocked = analysis_tiers[analysis_tiers["blocked_confirmatory"].map(_safe_bool).fillna(False)]
        if not blocked.empty:
            added = True
            lines.append(f"- analysis_tiers exclusions: {len(blocked)}")
            for row in blocked.to_dict("records"):
                row_cohort = _safe_text(row.get("cohort"))
                row_estimand = _safe_text(row.get("estimand"))
                row_reason = _safe_text(row.get("reason"))
                if not row_cohort:
                    continue
                label = row_cohort if not row_estimand else f"{row_cohort}:{row_estimand}"
                reason_text = row_reason or "blocked_confirmatory"
                lines.append(f"  - {label} -> {reason_text}")

    if spec_exists and not specification_summary.empty and "weight_concordance_reason" in specification_summary.columns:
        nonconfirm = specification_summary[
            specification_summary["weight_concordance_reason"].astype(str).str.startswith("nonconfirmatory_", na=False)
        ]
        if not nonconfirm.empty:
            added = True
            lines.append(f"- specification_stability_summary exclusion notes: {len(nonconfirm)}")
            for row in nonconfirm.head(5).to_dict("records"):
                row_cohort = _safe_text(row.get("cohort"))
                row_estimand = _safe_text(row.get("estimand"))
                row_reason = _safe_text(row.get("weight_concordance_reason"))
                label = row_cohort if not row_estimand else f"{row_cohort}:{row_estimand}"
                lines.append(f"  - {label}: {row_reason}")

    if not added:
        lines.append("- No exclusions or inconclusive notes detected from available lock artifacts.")
    return lines


def _ci_coherence_summary(ci_frame: pd.DataFrame, exists: bool) -> list[str]:
    lines = ["## CI coherence summary", ""]
    if not exists:
        lines.append("- CI coherence unavailable: inference_ci_coherence.csv is missing.")
        lines.append("- Run `30_check_inference_ci_coherence.py` to regenerate.")
        return lines

    if ci_frame.empty:
        lines.append("- inference_ci_coherence.csv was present but empty.")
        return lines

    status = ci_frame.get("status", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    issue = ci_frame.get("issue", pd.Series(dtype=object)).fillna("").astype(str)
    cohort = ci_frame.get("cohort", pd.Series(dtype=object)).fillna("_")
    estimate_type = ci_frame.get("estimate_type", pd.Series(dtype=object)).fillna("_")

    computed_mask = status == "computed"
    computed = int(computed_mask.sum())
    missing = int((~computed_mask).sum())
    violations = int((computed_mask & (issue != "contains_estimate")).sum())
    lines.append(f"- CI rows evaluated: {len(ci_frame)}")
    lines.append(f"- Computed rows reviewed for coherence: {computed}")
    lines.append(f"- Non-computed rows: {missing}")
    lines.append(f"- Estimated-containment mismatches: {violations}")
    if violations:
        mismatch = ci_frame[computed_mask & (issue != "contains_estimate")]
        for row in mismatch.head(5).to_dict("records"):
            row_cohort = _safe_text(row.get("cohort"))
            row_estimand = _safe_text(row.get("estimate_type"))
            row_issue = _safe_text(row.get("issue"))
            lines.append(f"  - {row_cohort}:{row_estimand} -> {row_issue}")
    return lines


def _build_sections(
    claim_verdicts: pd.DataFrame,
    analysis_tiers: pd.DataFrame,
    specification_summary: pd.DataFrame,
    inference_ci: pd.DataFrame,
    claim_exists: bool,
    analysis_exists: bool,
    specification_exists: bool,
    inference_exists: bool,
) -> str:
    sections = []
    sections.extend(_methods_snapshot(
        claim_verdicts,
        analysis_tiers,
        specification_summary,
        inference_ci,
        claim_exists,
        analysis_exists,
        specification_exists,
        inference_exists,
    ))
    sections.append("")
    sections.extend(_claim_verdict_summary(claim_verdicts, claim_exists))
    sections.append("")
    sections.extend(_exclusions_and_inconclusive(
        analysis_tiers,
        claim_verdicts,
        specification_summary,
        analysis_exists,
        claim_exists,
        specification_exists,
    ))
    sections.append("")
    sections.extend(_ci_coherence_summary(inference_ci, inference_exists))
    return "\n".join(sections)


def _write_markdown(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def build_publication_report_sections(
    *,
    project_root_path: Path,
    claim_verdicts_path: Path = DEFAULT_CLAIM_VERDICTS_PATH,
    analysis_tiers_path: Path = DEFAULT_ANALYSIS_TIERS_PATH,
    specification_stability_summary_path: Path = DEFAULT_SPECIFICATION_STABILITY_PATH,
    inference_ci_coherence_path: Path = DEFAULT_INFERENCE_CI_COHERENCE_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> str:
    root = Path(project_root_path).resolve()
    claim_path = _resolve_path(root, claim_verdicts_path)
    analysis_path = _resolve_path(root, analysis_tiers_path)
    specification_path = _resolve_path(root, specification_stability_summary_path)
    inference_path = _resolve_path(root, inference_ci_coherence_path)
    output_file = _resolve_path(root, output_path)

    claim_verdicts = _safe_read_csv(claim_path)
    analysis_tiers = _safe_read_csv(analysis_path)
    specification_summary = _safe_read_csv(specification_path)
    inference_ci = _safe_read_csv(inference_path)

    sections = _build_sections(
        claim_verdicts,
        analysis_tiers,
        specification_summary,
        inference_ci,
        claim_path.exists(),
        analysis_path.exists(),
        specification_path.exists(),
        inference_path.exists(),
    )
    markdown = "# Publication report sections\n\n" + sections
    _write_markdown(markdown, output_file)
    return markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Export publication report sections from lock artifacts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--claim-verdicts",
        type=Path,
        default=DEFAULT_CLAIM_VERDICTS_PATH,
        help="Path to claim_verdicts.csv (relative to project root).",
    )
    parser.add_argument(
        "--analysis-tiers",
        type=Path,
        default=DEFAULT_ANALYSIS_TIERS_PATH,
        help="Path to analysis_tiers.csv (relative to project root).",
    )
    parser.add_argument(
        "--specification-stability-summary",
        type=Path,
        default=DEFAULT_SPECIFICATION_STABILITY_PATH,
        help="Path to specification_stability_summary.csv (relative to project root).",
    )
    parser.add_argument(
        "--inference-ci-coherence",
        type=Path,
        default=DEFAULT_INFERENCE_CI_COHERENCE_PATH,
        help="Path to inference_ci_coherence.csv (relative to project root).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output markdown path (relative to project root).",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    build_publication_report_sections(
        project_root_path=root,
        claim_verdicts_path=args.claim_verdicts,
        analysis_tiers_path=args.analysis_tiers,
        specification_stability_summary_path=args.specification_stability_summary,
        inference_ci_coherence_path=args.inference_ci_coherence,
        output_path=args.output,
    )
    print(f"[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
