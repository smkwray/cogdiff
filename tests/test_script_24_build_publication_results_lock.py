from __future__ import annotations

import subprocess
import zipfile
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_data(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_script_24_builds_results_lock_bundle(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    tables = root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    # Primary results tables (minimal stubs; lock builder only copies + hashes).
    _write_data(
        tables / "g_mean_diff.csv",
        pd.DataFrame([{"cohort": "nlsy79", "d_g": 0.3, "SE_d_g": 0.03, "ci_low_d_g": 0.24, "ci_high_d_g": 0.36}]),
    )
    _write_data(
        tables / "g_variance_ratio.csv",
        pd.DataFrame([{"cohort": "nlsy79", "VR_g": 1.31, "SE_logVR": 0.03, "ci_low": 1.16, "ci_high": 1.33}]),
    )
    _write_data(
        tables / "g_mean_diff_family_bootstrap.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "reason": "", "d_g": 0.28, "SE_d_g": 0.19, "ci_low_d_g": -0.10, "ci_high_d_g": 0.35}]),
    )
    _write_data(
        tables / "g_variance_ratio_family_bootstrap.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "reason": "", "VR_g": 1.29, "SE_logVR": 0.16, "ci_low": 0.79, "ci_high": 1.31}]),
    )
    _write_data(
        tables / "group_factor_diffs.csv",
        pd.DataFrame([{"cohort": "nlsy79", "factor": "Speed", "mean_diff": 0.1}]),
    )
    _write_data(
        tables / "g_variance_ratio_vr0.csv",
        pd.DataFrame([{"cohort": "nlsy79", "VR0_g": 1.0}]),
    )

    # Key figures (dummy bytes; lock builder only copies + hashes).
    figures = root / "outputs" / "figures"
    png = b"\x89PNG\r\n\x1a\n"
    for name in (
        "g_mean_diff_forestplot.png",
        "vr_forestplot.png",
        "robustness_forestplot.png",
        "group_factor_gaps.png",
        "group_factor_vr.png",
        "missingness_heatmap.png",
        "cnlsy_age_trends_mean.png",
        "cnlsy_age_trends_vr.png",
        "cnlsy_subtest_d_profile.png",
        "cnlsy_subtest_log_vr_profile.png",
        "nlsy79_subtest_d_profile.png",
        "nlsy79_subtest_log_vr_profile.png",
        "nlsy97_subtest_d_profile.png",
        "nlsy97_subtest_log_vr_profile.png",
    ):
        _write_bytes(figures / name, png)

    _write_data(
        tables / "analysis_tiers.csv",
        pd.DataFrame(
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
                    "analysis_tier": "exploratory_sensitivity",
                    "blocked_confirmatory": True,
                    "reason": "invariance:partial_replicability_guard",
                },
            ]
        ),
    )
    _write_data(
        tables / "specification_stability_summary.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy97",
                    "estimand": "d_g",
                    "weight_concordance_reason": "nonconfirmatory_missing_unweighted_baseline",
                    "weight_unweighted_status": "baseline_missing",
                    "weight_weighted_status": "computed",
                    "robust_claim_eligible": False,
                }
            ]
        ),
    )
    _write_data(
        tables / "confirmatory_exclusions.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy97",
                    "blocked_confirmatory_d_g": True,
                    "blocked_confirmatory_vr_g": False,
                    "reason_d_g": "invariance:partial_replicability_guard",
                    "reason_vr_g": pd.NA,
                }
            ]
        ),
    )
    _write_data(
        tables / "publication_snapshot_manifest.csv",
        pd.DataFrame(
            [
                {
                    "snapshot_utc": "2026-02-21T00:00:00Z",
                    "path": "outputs/tables/analysis_tiers.csv",
                    "sha256": "x",
                    "size_bytes": 1,
                    "mtime_utc": "2026-02-21T00:00:00Z",
                }
            ]
        ),
    )
    _write_data(
        tables / "claim_verdicts.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "claim_id": "C1",
                    "verdict": "confirmed",
                    "evidence_note": "stub",
                }
            ]
        ),
    )
    _write_data(
        tables / "inference_ci_coherence.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "estimand": "d_g",
                    "is_strict": True,
                    "violations": 0,
                }
            ]
        ),
    )
    _write_data(
        tables / "sibling_fe_g_outcome.csv",
        pd.DataFrame([{"cohort": "nlsy79", "outcome": "education", "status": "computed", "beta_g_within": 1.2}]),
    )
    _write_data(
        tables / "intergenerational_g_transmission.csv",
        pd.DataFrame([{"model": "bivariate", "status": "computed", "beta_mother_g": 0.4}]),
    )
    _write_data(
        tables / "subtest_profile_tilt.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "d_tilt": -0.2}]),
    )
    _write_data(
        tables / "sibling_fe_cross_cohort_contrasts.csv",
        pd.DataFrame([{"outcome": "education", "status": "computed", "diff_b_minus_a": -0.8}]),
    )
    _write_data(
        tables / "intergenerational_g_attenuation.csv",
        pd.DataFrame([{"status": "computed", "attenuation_pct": 20.0}]),
    )
    _write_data(
        tables / "subtest_profile_tilt_summary.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "interpretation": "small_add_on"}]),
    )
    _write_data(
        tables / "degree_threshold_outcomes.csv",
        pd.DataFrame([{"cohort": "nlsy79", "threshold": "ba_or_more", "status": "computed", "odds_ratio_g": 4.9}]),
    )
    _write_data(
        tables / "explicit_degree_outcomes.csv",
        pd.DataFrame([{"cohort": "nlsy79", "threshold": "ba_or_more_explicit", "status": "computed", "odds_ratio_g": 4.2}]),
    )
    _write_data(
        tables / "g_employment_outcomes.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "odds_ratio_g": 1.9}]),
    )
    _write_data(
        tables / "race_invariance_summary.csv",
        pd.DataFrame([{"cohort": "nlsy79", "model_step": "configural", "status": "computed"}]),
    )
    _write_data(
        tables / "race_invariance_transition_checks.csv",
        pd.DataFrame([{"cohort": "nlsy79", "transition": "configural->metric", "status": "computed"}]),
    )
    _write_data(
        tables / "race_invariance_eligibility.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "metric_pass": True, "scalar_pass": True}]),
    )
    _write_data(
        tables / "nlsy79_occupation_major_group_summary.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "occupation_group": "management_professional_related"}]),
    )
    _write_data(
        tables / "nlsy79_high_skill_occupation_outcome.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "outcome": "management_professional_related", "odds_ratio_g": 2.4}]),
    )
    _write_data(
        tables / "nlsy79_job_zone_mapping_quality.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "n_matched_any": 3113, "pct_matched_any": 0.59}]),
    )
    _write_data(
        tables / "nlsy79_job_zone_complexity_outcome.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "outcome": "job_zone", "beta_g": 0.34}]),
    )
    _write_data(
        tables / "nlsy79_job_pay_mismatch_summary.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "group": "overall", "n_used": 2500, "mean_pay_residual_z": 0.0}]),
    )
    _write_data(
        tables / "nlsy79_job_pay_mismatch_models.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "outcome": "pay_residual_z", "model_family": "ols_residual", "beta_g": 0.15}]),
    )
    _write_data(
        tables / "nlsy79_occupation_education_mapping_quality.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "n_matched_any": 4120, "mean_required_education_years": 14.8}]),
    )
    _write_data(
        tables / "nlsy79_occupation_education_requirement_outcome.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "outcome": "required_education_years", "beta_g": 0.52}]),
    )
    _write_data(
        tables / "nlsy79_education_job_mismatch_summary.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "group": "overall", "n_used": 3000, "mean_mismatch_years": 0.4}]),
    )
    _write_data(
        tables / "nlsy79_education_job_mismatch_models.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "outcome": "mismatch_years", "model_family": "ols", "beta_g": 0.11}]),
    )
    _write_data(
        tables / "nlsy97_adult_occupation_major_group_summary.csv",
        pd.DataFrame([{"cohort": "nlsy97", "status": "computed", "occupation_group": "management_professional_related"}]),
    )
    _write_data(
        tables / "nlsy97_high_skill_occupation_outcome.csv",
        pd.DataFrame([{"cohort": "nlsy97", "status": "computed", "outcome": "management_professional_related", "odds_ratio_g": 2.1}]),
    )
    _write_data(
        tables / "age_matched_outcome_validity.csv",
        pd.DataFrame([{"cohort": "nlsy79", "outcome": "employment", "status": "computed", "beta_g": 0.66}]),
    )
    _write_data(
        tables / "age_matched_cross_cohort_contrasts.csv",
        pd.DataFrame([{"outcome": "employment", "status": "computed", "diff_b_minus_a": -0.004}]),
    )
    _write_data(
        tables / "nlsy97_income_earnings_trajectories.csv",
        pd.DataFrame([{"cohort": "nlsy97", "outcome": "annual_earnings", "model": "annualized_log_change", "status": "computed", "beta_g": 0.03}]),
    )
    _write_data(
        tables / "nlsy97_employment_persistence.csv",
        pd.DataFrame([{"cohort": "nlsy97", "model": "persistent_employment_2019_2021", "status": "computed", "odds_ratio_g": 1.4}]),
    )
    _write_data(
        tables / "cnlsy_adult_outcome_associations.csv",
        pd.DataFrame([{"cohort": "cnlsy", "outcome": "education_years_2014", "model_type": "ols", "status": "computed", "beta_g": 0.07}]),
    )
    _write_data(
        tables / "cnlsy_carryover_net_mother_ses.csv",
        pd.DataFrame([{"cohort": "cnlsy", "outcome": "log_wage_income_2014", "model": "baseline", "status": "computed", "beta_g": 0.2}]),
    )
    _write_data(
        tables / "cnlsy_carryover_net_mother_ses_summary.csv",
        pd.DataFrame([{"cohort": "cnlsy", "outcome": "log_wage_income_2014", "status": "computed", "attenuation_pct": 10.0}]),
    )
    _write_data(
        tables / "nonlinear_threshold_outcome_models.csv",
        pd.DataFrame([{"cohort": "nlsy79", "outcome": "log_annual_earnings", "model": "linear", "status": "computed", "beta_g": 0.2}]),
    )
    _write_data(
        tables / "nonlinear_threshold_outcome_summary.csv",
        pd.DataFrame([{"cohort": "nlsy79", "outcome": "log_annual_earnings", "status": "computed", "p_value_g_sq": 0.03}]),
    )
    _write_data(
        tables / "nlsy79_mediation_models.csv",
        pd.DataFrame([{"cohort": "nlsy79", "outcome": "log_annual_earnings", "model": "plus_all_mediators", "status": "computed", "beta_g": 0.17}]),
    )
    _write_data(
        tables / "nlsy79_mediation_summary.csv",
        pd.DataFrame([{"cohort": "nlsy79", "outcome": "log_annual_earnings", "model": "plus_all_mediators", "status": "computed", "pct_attenuation_g": 46.9}]),
    )
    _write_data(
        tables / "nlsy97_income_earnings_volatility.csv",
        pd.DataFrame([{"cohort": "nlsy97", "outcome": "annual_earnings", "model": "abs_annualized_log_change", "status": "computed", "beta_g": -0.03}]),
    )
    _write_data(
        tables / "nlsy97_employment_instability.csv",
        pd.DataFrame([{"cohort": "nlsy97", "model": "any_transition_2011_2021", "status": "computed", "odds_ratio_g": 0.81}]),
    )
    _write_data(
        tables / "nlsy97_unemployment_insurance.csv",
        pd.DataFrame([{"cohort": "nlsy97", "year": 2019, "model": "any_ui_receipt", "status": "computed", "odds_ratio_g": 0.74}]),
    )
    _write_data(
        tables / "sibling_discordance.csv",
        pd.DataFrame([{"cohort": "nlsy79", "outcome": "earnings", "status": "computed", "beta_abs_g_diff": 9500.0}]),
    )
    _write_data(
        tables / "nlsy97_occupational_mobility_summary.csv",
        pd.DataFrame([{"cohort": "nlsy97", "status": "computed", "n_with_2plus_occupation_waves": 73}]),
    )
    _write_data(
        tables / "nlsy97_occupational_mobility_models.csv",
        pd.DataFrame([{"cohort": "nlsy97", "model": "any_major_group_change", "status": "computed", "odds_ratio_g": 0.88}]),
    )
    (tables / "results_snapshot.md").write_text("# snapshot\n", encoding="utf-8")

    script = _repo_root() / "scripts" / "24_build_publication_results_lock.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        cwd=_repo_root(),
        check=True,
    )

    bundle_dir = root / "outputs" / "tables" / "publication_results_lock"
    assert bundle_dir.exists()
    assert (bundle_dir / "g_mean_diff.csv").exists()
    assert (bundle_dir / "g_variance_ratio.csv").exists()
    assert (bundle_dir / "g_mean_diff_family_bootstrap.csv").exists()
    assert (bundle_dir / "g_variance_ratio_family_bootstrap.csv").exists()
    assert (bundle_dir / "group_factor_diffs.csv").exists()
    assert (bundle_dir / "g_variance_ratio_vr0.csv").exists()
    assert (bundle_dir / "g_mean_diff_forestplot.png").exists()
    assert (bundle_dir / "vr_forestplot.png").exists()
    assert (bundle_dir / "robustness_forestplot.png").exists()
    assert (bundle_dir / "analysis_tiers.csv").exists()
    assert (bundle_dir / "specification_stability_summary.csv").exists()
    assert (bundle_dir / "confirmatory_exclusions.csv").exists()
    assert (bundle_dir / "publication_snapshot_manifest.csv").exists()
    assert (bundle_dir / "claim_verdicts.csv").exists()
    assert (bundle_dir / "inference_ci_coherence.csv").exists()
    assert (bundle_dir / "sibling_fe_g_outcome.csv").exists()
    assert (bundle_dir / "sibling_fe_cross_cohort_contrasts.csv").exists()
    assert (bundle_dir / "intergenerational_g_transmission.csv").exists()
    assert (bundle_dir / "intergenerational_g_attenuation.csv").exists()
    assert (bundle_dir / "subtest_profile_tilt.csv").exists()
    assert (bundle_dir / "subtest_profile_tilt_summary.csv").exists()
    assert (bundle_dir / "degree_threshold_outcomes.csv").exists()
    assert (bundle_dir / "explicit_degree_outcomes.csv").exists()
    assert (bundle_dir / "g_employment_outcomes.csv").exists()
    assert (bundle_dir / "race_invariance_summary.csv").exists()
    assert (bundle_dir / "race_invariance_transition_checks.csv").exists()
    assert (bundle_dir / "race_invariance_eligibility.csv").exists()
    assert (bundle_dir / "nlsy79_occupation_major_group_summary.csv").exists()
    assert (bundle_dir / "nlsy79_high_skill_occupation_outcome.csv").exists()
    assert (bundle_dir / "nlsy79_job_zone_mapping_quality.csv").exists()
    assert (bundle_dir / "nlsy79_job_zone_complexity_outcome.csv").exists()
    assert (bundle_dir / "nlsy79_job_pay_mismatch_summary.csv").exists()
    assert (bundle_dir / "nlsy79_job_pay_mismatch_models.csv").exists()
    assert (bundle_dir / "nlsy79_occupation_education_mapping_quality.csv").exists()
    assert (bundle_dir / "nlsy79_occupation_education_requirement_outcome.csv").exists()
    assert (bundle_dir / "nlsy79_education_job_mismatch_summary.csv").exists()
    assert (bundle_dir / "nlsy79_education_job_mismatch_models.csv").exists()
    assert (bundle_dir / "nlsy97_adult_occupation_major_group_summary.csv").exists()
    assert (bundle_dir / "nlsy97_high_skill_occupation_outcome.csv").exists()
    assert (bundle_dir / "age_matched_outcome_validity.csv").exists()
    assert (bundle_dir / "age_matched_cross_cohort_contrasts.csv").exists()
    assert (bundle_dir / "nlsy97_income_earnings_trajectories.csv").exists()
    assert (bundle_dir / "nlsy97_employment_persistence.csv").exists()
    assert (bundle_dir / "cnlsy_adult_outcome_associations.csv").exists()
    assert (bundle_dir / "cnlsy_carryover_net_mother_ses.csv").exists()
    assert (bundle_dir / "cnlsy_carryover_net_mother_ses_summary.csv").exists()
    assert (bundle_dir / "nonlinear_threshold_outcome_models.csv").exists()
    assert (bundle_dir / "nonlinear_threshold_outcome_summary.csv").exists()
    assert (bundle_dir / "nlsy79_mediation_models.csv").exists()
    assert (bundle_dir / "nlsy79_mediation_summary.csv").exists()
    assert (bundle_dir / "nlsy97_income_earnings_volatility.csv").exists()
    assert (bundle_dir / "nlsy97_employment_instability.csv").exists()
    assert (bundle_dir / "nlsy97_unemployment_insurance.csv").exists()
    assert (bundle_dir / "sibling_discordance.csv").exists()
    assert (bundle_dir / "nlsy97_occupational_mobility_summary.csv").exists()
    assert (bundle_dir / "nlsy97_occupational_mobility_models.csv").exists()
    assert (bundle_dir / "publication_results_lock_manifest.csv").exists()
    assert (bundle_dir / "manuscript_results_lock.md").exists()
    assert (bundle_dir / "results_snapshot.md").exists()
    assert (root / "outputs" / "tables" / "publication_results_lock.zip").exists()

    lock_zip = root / "outputs" / "tables" / "publication_results_lock.zip"
    with zipfile.ZipFile(lock_zip) as zf:
        zipped_filenames = {Path(name).name for name in zf.namelist()}

    expected_zip_members = {
        "g_mean_diff.csv",
        "g_variance_ratio.csv",
        "g_mean_diff_family_bootstrap.csv",
        "g_variance_ratio_family_bootstrap.csv",
        "group_factor_diffs.csv",
        "g_variance_ratio_vr0.csv",
        "g_mean_diff_forestplot.png",
        "vr_forestplot.png",
        "robustness_forestplot.png",
        "analysis_tiers.csv",
        "specification_stability_summary.csv",
        "confirmatory_exclusions.csv",
        "publication_snapshot_manifest.csv",
        "publication_results_lock_manifest.csv",
        "manuscript_results_lock.md",
        "claim_verdicts.csv",
        "inference_ci_coherence.csv",
        "sibling_fe_g_outcome.csv",
        "sibling_fe_cross_cohort_contrasts.csv",
        "intergenerational_g_transmission.csv",
        "intergenerational_g_attenuation.csv",
        "subtest_profile_tilt.csv",
        "subtest_profile_tilt_summary.csv",
        "degree_threshold_outcomes.csv",
        "explicit_degree_outcomes.csv",
        "g_employment_outcomes.csv",
        "race_invariance_summary.csv",
        "race_invariance_transition_checks.csv",
        "race_invariance_eligibility.csv",
        "nlsy79_occupation_major_group_summary.csv",
        "nlsy79_high_skill_occupation_outcome.csv",
        "nlsy79_job_zone_mapping_quality.csv",
        "nlsy79_job_zone_complexity_outcome.csv",
        "nlsy79_job_pay_mismatch_summary.csv",
        "nlsy79_job_pay_mismatch_models.csv",
        "nlsy79_occupation_education_mapping_quality.csv",
        "nlsy79_occupation_education_requirement_outcome.csv",
        "nlsy79_education_job_mismatch_summary.csv",
        "nlsy79_education_job_mismatch_models.csv",
        "nlsy97_adult_occupation_major_group_summary.csv",
        "nlsy97_high_skill_occupation_outcome.csv",
        "age_matched_outcome_validity.csv",
        "age_matched_cross_cohort_contrasts.csv",
        "nlsy97_income_earnings_trajectories.csv",
        "nlsy97_employment_persistence.csv",
        "cnlsy_adult_outcome_associations.csv",
        "cnlsy_carryover_net_mother_ses.csv",
        "cnlsy_carryover_net_mother_ses_summary.csv",
        "nonlinear_threshold_outcome_models.csv",
        "nonlinear_threshold_outcome_summary.csv",
        "nlsy79_mediation_models.csv",
        "nlsy79_mediation_summary.csv",
        "nlsy97_income_earnings_volatility.csv",
        "nlsy97_employment_instability.csv",
        "nlsy97_unemployment_insurance.csv",
        "sibling_discordance.csv",
        "nlsy97_occupational_mobility_summary.csv",
        "nlsy97_occupational_mobility_models.csv",
        "results_snapshot.md",
    }
    assert expected_zip_members.issubset(zipped_filenames)

    methods = (bundle_dir / "manuscript_results_lock.md").read_text(encoding="utf-8")
    assert "weight_pair_policy=replication_unweighted_primary_weighted_sensitivity" in methods
    assert "nlsy97" in methods


def test_script_24_removes_stale_bundle_files_on_rebuild(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    tables = root / "outputs" / "tables"
    figures = root / "outputs" / "figures"
    bundle_dir = tables / "publication_results_lock"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "cnlsy_employment_2014.csv").write_text("stale\n", encoding="utf-8")

    _write_data(
        tables / "g_mean_diff.csv",
        pd.DataFrame([{"cohort": "nlsy79", "d_g": 0.3, "SE_d_g": 0.03, "ci_low_d_g": 0.24, "ci_high_d_g": 0.36}]),
    )
    _write_data(
        tables / "g_variance_ratio.csv",
        pd.DataFrame([{"cohort": "nlsy79", "VR_g": 1.31, "SE_logVR": 0.03, "ci_low": 1.16, "ci_high": 1.33}]),
    )
    _write_data(
        tables / "g_mean_diff_family_bootstrap.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "reason": "", "d_g": 0.28, "SE_d_g": 0.19, "ci_low_d_g": -0.10, "ci_high_d_g": 0.35}]),
    )
    _write_data(
        tables / "g_variance_ratio_family_bootstrap.csv",
        pd.DataFrame([{"cohort": "nlsy79", "status": "computed", "reason": "", "VR_g": 1.29, "SE_logVR": 0.16, "ci_low": 0.79, "ci_high": 1.31}]),
    )
    _write_data(tables / "group_factor_diffs.csv", pd.DataFrame([{"cohort": "nlsy79", "factor": "Speed", "mean_diff": 0.1}]))
    _write_data(tables / "g_variance_ratio_vr0.csv", pd.DataFrame([{"cohort": "nlsy79", "VR0_g": 1.0}]))
    for name in (
        "g_mean_diff_forestplot.png",
        "vr_forestplot.png",
        "robustness_forestplot.png",
        "group_factor_gaps.png",
        "group_factor_vr.png",
        "missingness_heatmap.png",
        "cnlsy_age_trends_mean.png",
        "cnlsy_age_trends_vr.png",
        "cnlsy_subtest_d_profile.png",
        "cnlsy_subtest_log_vr_profile.png",
        "nlsy79_subtest_d_profile.png",
        "nlsy79_subtest_log_vr_profile.png",
        "nlsy97_subtest_d_profile.png",
        "nlsy97_subtest_log_vr_profile.png",
    ):
        _write_bytes(figures / name, b"\x89PNG\r\n\x1a\n")
    _write_data(tables / "analysis_tiers.csv", pd.DataFrame([{"cohort": "nlsy79", "estimand": "d_g", "analysis_tier": "confirmatory", "blocked_confirmatory": False, "reason": pd.NA}]))
    _write_data(tables / "specification_stability_summary.csv", pd.DataFrame([{"cohort": "nlsy79", "estimand": "d_g", "weight_concordance_reason": "primary", "weight_unweighted_status": "computed", "weight_weighted_status": "computed", "robust_claim_eligible": True}]))
    _write_data(tables / "confirmatory_exclusions.csv", pd.DataFrame([{"cohort": "nlsy79", "blocked_confirmatory_d_g": False, "blocked_confirmatory_vr_g": False, "reason_d_g": pd.NA, "reason_vr_g": pd.NA}]))
    _write_data(tables / "publication_snapshot_manifest.csv", pd.DataFrame([{"snapshot_utc": "2026-02-21T00:00:00Z", "path": "outputs/tables/analysis_tiers.csv", "sha256": "x", "size_bytes": 1, "mtime_utc": "2026-02-21T00:00:00Z"}]))
    _write_data(tables / "claim_verdicts.csv", pd.DataFrame([{"cohort": "nlsy79", "claim_id": "C1", "verdict": "confirmed", "evidence_note": "stub"}]))
    _write_data(tables / "inference_ci_coherence.csv", pd.DataFrame([{"cohort": "nlsy79", "estimand": "d_g", "is_strict": True, "violations": 0}]))

    script = _repo_root() / "scripts" / "24_build_publication_results_lock.py"
    subprocess.run([sys.executable, str(script), "--project-root", str(root)], cwd=_repo_root(), check=True)

    assert not (bundle_dir / "cnlsy_employment_2014.csv").exists()
    with zipfile.ZipFile(root / "outputs" / "tables" / "publication_results_lock.zip") as zf:
        assert "cnlsy_employment_2014.csv" not in {Path(name).name for name in zf.namelist()}
