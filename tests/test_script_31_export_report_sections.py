from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "31_export_report_sections.py"
    spec = importlib.util.spec_from_file_location("script31_export_report_sections", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_publication_report_sections_generates_clear_sections_with_all_inputs(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    output_path = root / "outputs/tables/publication_report_sections.md"

    _write_csv(
        root / "outputs/tables/claim_verdicts.csv",
        [
            {
                "claim_id": "nlsy79:d_g:unweighted_vs_weighted",
                "claim_label": "claim:nlsy79:d_g:unweighted_vs_weighted",
                "verdict": "confirmed",
                "reason": "robust_claim_eligible=true",
                "cohorts_evaluated": "nlsy79",
                "thresholds_used": "{}",
            },
            {
                "claim_id": "nlsy97:d_g:unweighted_vs_weighted",
                "claim_label": "claim:nlsy97:d_g:unweighted_vs_weighted",
                "verdict": "not_confirmed",
                "reason": "blocked_confirmatory",
                "cohorts_evaluated": "nlsy97",
                "thresholds_used": "{}",
            },
            {
                "claim_id": "cnlsy:d_g:unweighted_vs_weighted",
                "claim_label": "claim:cnlsy:d_g:unweighted_vs_weighted",
                "verdict": "inconclusive",
                "reason": "missing_artifact: specification",
                "cohorts_evaluated": "cnlsy",
                "thresholds_used": "{}",
            },
        ],
    )

    _write_csv(
        root / "outputs/tables/analysis_tiers.csv",
        [
            {
                "cohort": "nlsy79",
                "estimand": "d_g",
                "analysis_tier": "confirmatory",
                "blocked_confirmatory": False,
                "reason": pd.NA,
            },
            {
                "cohort": "nlsy97",
                "estimand": "d_g",
                "analysis_tier": "exploratory_low_information",
                "blocked_confirmatory": True,
                "reason": "partial_replicability_guard",
            },
            {
                "cohort": "cnlsy",
                "estimand": "vr_g",
                "analysis_tier": "exploratory_low_information",
                "blocked_confirmatory": False,
                "reason": "n/a",
            },
        ],
    )

    _write_csv(
        root / "outputs/tables/specification_stability_summary.csv",
        [
            {
                "cohort": "nlsy79",
                "estimand": "d_g",
                "robust_claim_eligible": True,
                "weight_concordance_reason": "none",
            },
            {
                "cohort": "nlsy97",
                "estimand": "d_g",
                "robust_claim_eligible": False,
                "weight_concordance_reason": "nonconfirmatory_missing_unweighted_baseline",
            },
        ],
    )

    _write_csv(
        root / "outputs/tables/inference_ci_coherence.csv",
        [
            {
                "cohort": "nlsy79",
                "inference_method": "robust_cluster",
                "estimate_type": "d_g",
                "status": "computed",
                "estimate": 0.2,
                "ci_low": 0.1,
                "ci_high": 0.3,
                "ci_contains_estimate": True,
                "issue": "contains_estimate",
            },
            {
                "cohort": "nlsy97",
                "inference_method": "robust_cluster",
                "estimate_type": "d_g",
                "status": "computed",
                "estimate": 0.8,
                "ci_low": 0.1,
                "ci_high": 0.2,
                "ci_contains_estimate": False,
                "issue": "computed_outside_ci",
            },
            {
                "cohort": "cnlsy",
                "inference_method": "robust_cluster",
                "estimate_type": "d_g",
                "status": "baseline_missing",
                "estimate": 0.1,
                "ci_low": 0.0,
                "ci_high": 0.2,
                "ci_contains_estimate": None,
                "issue": "non_computed_status_baseline_missing",
            },
        ],
    )

    markdown = module.build_publication_report_sections(project_root_path=root, output_path=output_path)

    assert output_path.exists()
    assert "## Methods snapshot" in markdown
    assert "## Claim verdict summary" in markdown
    assert "## Exclusions / inconclusive notes" in markdown
    assert "## CI coherence summary" in markdown
    assert "- claim verdict count (confirmed): 1" in markdown
    assert "nlsy79:d_g:unweighted_vs_weighted" in markdown
    assert "- CI rows evaluated: 3" in markdown
    assert "- Inconclusive claim rows: 1" in markdown
    assert "nonconfirmatory_missing_unweighted_baseline" in markdown
    assert "- Estimated-containment mismatches: 1" in markdown


def test_build_publication_report_sections_degrades_with_missing_tables(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    output_path = root / "outputs/tables/publication_report_sections.md"

    _write_csv(
        root / "outputs/tables/claim_verdicts.csv",
        [
            {
                "claim_id": "nlsy79:d_g:unweighted_vs_weighted",
                "claim_label": "claim:nlsy79:d_g:unweighted_vs_weighted",
                "verdict": "inconclusive",
                "reason": "missing_artifact: x",
                "cohorts_evaluated": "nlsy79",
                "thresholds_used": "{}",
            }
        ],
    )

    # Intentionally omit the other core lock tables to verify graceful fallback behavior.
    markdown = module.build_publication_report_sections(
        project_root_path=root,
        output_path=output_path,
    )

    assert output_path.exists()
    assert "Missing input table: outputs/tables/analysis_tiers.csv" in markdown
    assert "Missing input table: outputs/tables/specification_stability_summary.csv" in markdown
    assert "Missing input table: outputs/tables/inference_ci_coherence.csv" in markdown
    assert "CI coherence unavailable" in markdown
    assert "## Exclusions / inconclusive notes" in markdown
    assert "- claim_verdicts.csv: 1 row(s);" in markdown
