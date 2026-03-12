#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import project_root

KEY_ARTIFACTS: tuple[str, ...] = (
    # Primary results tables
    "outputs/tables/g_mean_diff.csv",
    "outputs/tables/g_variance_ratio.csv",
    "outputs/tables/g_mean_diff_family_bootstrap.csv",
    "outputs/tables/g_variance_ratio_family_bootstrap.csv",
    "outputs/tables/group_factor_diffs.csv",
    "outputs/tables/g_variance_ratio_vr0.csv",
    # Key figures (stage-09 outputs)
    "outputs/figures/g_mean_diff_forestplot.png",
    "outputs/figures/vr_forestplot.png",
    "outputs/figures/robustness_forestplot.png",
    "outputs/figures/group_factor_gaps.png",
    "outputs/figures/group_factor_vr.png",
    "outputs/figures/missingness_heatmap.png",
    "outputs/figures/cnlsy_age_trends_mean.png",
    "outputs/figures/cnlsy_age_trends_vr.png",
    "outputs/figures/cnlsy_subtest_d_profile.png",
    "outputs/figures/cnlsy_subtest_log_vr_profile.png",
    "outputs/figures/nlsy79_subtest_d_profile.png",
    "outputs/figures/nlsy79_subtest_log_vr_profile.png",
    "outputs/figures/nlsy97_subtest_d_profile.png",
    "outputs/figures/nlsy97_subtest_log_vr_profile.png",
    # Publication QA / contracts
    "outputs/tables/analysis_tiers.csv",
    "outputs/tables/specification_stability_summary.csv",
    "outputs/tables/confirmatory_exclusions.csv",
    "outputs/tables/publication_snapshot_manifest.csv",
    "outputs/tables/claim_verdicts.csv",
    "outputs/tables/inference_ci_coherence.csv",
)
OPTIONAL_ARTIFACTS: tuple[str, ...] = (
    "outputs/tables/results_snapshot.md",
    "outputs/tables/g_proxy_mean_diff_family_bootstrap.csv",
    "outputs/tables/g_proxy_variance_ratio_family_bootstrap.csv",
    # Exploratory extensions (g_proxy-based), when available.
    "outputs/tables/race_sex_group_estimates.csv",
    "outputs/tables/race_sex_interaction_summary.csv",
    "outputs/tables/racial_changes_over_time.csv",
    "outputs/tables/ses_moderation_summary.csv",
    "outputs/tables/ses_moderation_group_estimates.csv",
    "outputs/tables/g_income_wealth_associations.csv",
    "outputs/tables/g_employment_outcomes.csv",
    "outputs/tables/asvab_life_outcomes_by_sex.csv",
    "outputs/tables/g_sat_act_validity.csv",
    "outputs/tables/g_sat_act_validity_by_race.csv",
    "outputs/tables/g_sat_act_validity_by_ses.csv",
    "outputs/tables/g_sat_act_validity_by_race_sex.csv",
    "outputs/tables/g_outcome_associations_by_ses_summary.csv",
    "outputs/tables/g_outcome_associations_by_ses.csv",
    "outputs/tables/subtest_predictive_validity.csv",
    "outputs/tables/cnlsy_nonlinear_age_patterns.csv",
    "outputs/tables/overall_outcome_validity.csv",
    "outputs/tables/cross_cohort_predictive_validity_contrasts.csv",
    "outputs/tables/cross_cohort_pattern_stability.csv",
    "outputs/tables/sibling_fe_g_outcome.csv",
    "outputs/tables/sibling_fe_cross_cohort_contrasts.csv",
    "outputs/tables/intergenerational_g_transmission.csv",
    "outputs/tables/intergenerational_g_attenuation.csv",
    "outputs/tables/subtest_profile_tilt.csv",
    "outputs/tables/subtest_profile_tilt_summary.csv",
    "outputs/tables/degree_threshold_outcomes.csv",
    "outputs/tables/explicit_degree_outcomes.csv",
    "outputs/tables/race_invariance_summary.csv",
    "outputs/tables/race_invariance_transition_checks.csv",
    "outputs/tables/race_invariance_eligibility.csv",
    "outputs/tables/nlsy79_occupation_major_group_summary.csv",
    "outputs/tables/nlsy79_high_skill_occupation_outcome.csv",
    "outputs/tables/nlsy79_job_zone_mapping_quality.csv",
    "outputs/tables/nlsy79_job_zone_complexity_outcome.csv",
    "outputs/tables/nlsy79_job_pay_mismatch_summary.csv",
    "outputs/tables/nlsy79_job_pay_mismatch_models.csv",
    "outputs/tables/nlsy79_occupation_education_mapping_quality.csv",
    "outputs/tables/nlsy79_occupation_education_requirement_outcome.csv",
    "outputs/tables/nlsy79_education_job_mismatch_summary.csv",
    "outputs/tables/nlsy79_education_job_mismatch_models.csv",
    "outputs/tables/nlsy97_adult_occupation_major_group_summary.csv",
    "outputs/tables/nlsy97_high_skill_occupation_outcome.csv",
    "outputs/tables/age_matched_outcome_validity.csv",
    "outputs/tables/age_matched_cross_cohort_contrasts.csv",
    "outputs/tables/nlsy97_income_earnings_trajectories.csv",
    "outputs/tables/nlsy97_employment_persistence.csv",
    "outputs/tables/cnlsy_adult_outcome_associations.csv",
    "outputs/tables/cnlsy_carryover_net_mother_ses.csv",
    "outputs/tables/cnlsy_carryover_net_mother_ses_summary.csv",
    "outputs/tables/nonlinear_threshold_outcome_models.csv",
    "outputs/tables/nonlinear_threshold_outcome_summary.csv",
    "outputs/tables/nlsy79_mediation_models.csv",
    "outputs/tables/nlsy79_mediation_summary.csv",
    "outputs/tables/nlsy97_income_earnings_volatility.csv",
    "outputs/tables/nlsy97_employment_instability.csv",
    "outputs/tables/nlsy97_unemployment_insurance.csv",
    "outputs/tables/sibling_discordance.csv",
    "outputs/tables/nlsy97_occupational_mobility_summary.csv",
    "outputs/tables/nlsy97_occupational_mobility_models.csv",
    # Stage-09 optional plots derived from stage-20 bootstrap inference (if present).
    "outputs/figures/g_mean_diff_family_bootstrap_forestplot.png",
    "outputs/figures/vr_family_bootstrap_forestplot.png",
    "outputs/figures/g_proxy_mean_diff_family_bootstrap_forestplot.png",
    "outputs/figures/g_proxy_vr_family_bootstrap_forestplot.png",
)
WEIGHT_PAIR_POLICY = "replication_unweighted_primary_weighted_sensitivity"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_none_"
    subset = df[columns].fillna("")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for _, row in subset.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_bundle_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "generated_utc",
        "source_path",
        "bundle_path",
        "sha256",
        "size_bytes",
        "mtime_utc",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(src_dir.rglob("*")):
            if not file_path.is_file():
                continue
            archive.write(file_path, arcname=str(file_path.relative_to(src_dir)))


def build_publication_results_lock(root: Path) -> dict[str, str]:
    generated_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    tables_dir = root / "outputs" / "tables"
    bundle_dir = tables_dir / "publication_results_lock"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copied_rows: list[dict[str, Any]] = []
    def _copy_to_bundle(relative_path: str, *, required: bool) -> None:
        src = root / relative_path
        if not src.exists():
            if required:
                raise FileNotFoundError(f"Missing required artifact: {relative_path}")
            return
        dst = bundle_dir / Path(relative_path).name
        shutil.copy2(src, dst)
        mtime = datetime.fromtimestamp(src.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )
        copied_rows.append(
            {
                "generated_utc": generated_utc,
                "source_path": _relative(root, src),
                "bundle_path": _relative(root, dst),
                "sha256": _sha256(src),
                "size_bytes": int(src.stat().st_size),
                "mtime_utc": mtime,
            }
        )

    for relative_path in KEY_ARTIFACTS:
        _copy_to_bundle(relative_path, required=True)
    for relative_path in OPTIONAL_ARTIFACTS:
        _copy_to_bundle(relative_path, required=False)

    analysis_tiers = _read_csv(tables_dir / "analysis_tiers.csv")
    nonconfirm_tiers = analysis_tiers.loc[
        (~analysis_tiers["analysis_tier"].astype(str).str.strip().str.lower().eq("confirmatory"))
        | (analysis_tiers["blocked_confirmatory"].fillna(False).astype(bool))
    ].copy() if not analysis_tiers.empty else pd.DataFrame()

    spec_summary = _read_csv(tables_dir / "specification_stability_summary.csv")
    nonconfirm_weights = spec_summary.loc[
        spec_summary["weight_concordance_reason"].astype(str).str.startswith("nonconfirmatory_")
    ].copy() if not spec_summary.empty else pd.DataFrame()

    methods_md = bundle_dir / "manuscript_results_lock.md"
    methods_md.write_text(
        "\n".join(
            [
                "# Manuscript Results Lock",
                "",
                f"Generated UTC: {generated_utc}",
                "",
                "## Weight Policy",
                f"- `weight_pair_policy={WEIGHT_PAIR_POLICY}`",
                "- Weighted/unweighted concordance is primary-eligible only when a primary unweighted baseline exists.",
                "- Any `weight_concordance_reason` beginning with `nonconfirmatory_` is excluded from primary robustness claims.",
                "",
                "## Non-Primary Estimands",
                _markdown_table(
                    nonconfirm_tiers,
                    ["cohort", "estimand", "analysis_tier", "blocked_confirmatory", "reason"],
                ),
                "",
                "## Non-Primary Weight Concordance Cases",
                _markdown_table(
                    nonconfirm_weights,
                    [
                        "cohort",
                        "estimand",
                        "weight_concordance_reason",
                        "weight_unweighted_status",
                        "weight_weighted_status",
                        "robust_claim_eligible",
                    ],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle_manifest = bundle_dir / "publication_results_lock_manifest.csv"
    _write_bundle_manifest(bundle_manifest, copied_rows)

    zip_path = tables_dir / "publication_results_lock.zip"
    _zip_dir(bundle_dir, zip_path)

    return {
        "bundle_dir": _relative(root, bundle_dir),
        "bundle_manifest": _relative(root, bundle_manifest),
        "methods_markdown": _relative(root, methods_md),
        "zip_path": _relative(root, zip_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build publication results lock bundle.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        outputs = build_publication_results_lock(root)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {outputs['bundle_dir']}")
    print(f"[ok] wrote {outputs['bundle_manifest']}")
    print(f"[ok] wrote {outputs['methods_markdown']}")
    print(f"[ok] wrote {outputs['zip_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
