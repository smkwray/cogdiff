from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["cohort"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    script = _repo_root() / "scripts" / "93_build_results_snapshot.py"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_script_93_writes_snapshot_with_bootstrap_and_publication_lock(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    tables = root / "outputs/tables"
    _write_csv(
        tables / "g_mean_diff.csv",
        [
            {
                "cohort": "nlsy79",
                "d_g": "0.3",
                "SE_d_g": "0.03",
                "ci_low_d_g": "0.24",
                "ci_high_d_g": "0.36",
            }
        ],
    )
    _write_csv(
        tables / "g_variance_ratio.csv",
        [
            {
                "cohort": "nlsy79",
                "VR_g": "1.31",
                "SE_logVR": "0.03",
                "ci_low": "1.16",
                "ci_high": "1.33",
            }
        ],
    )
    _write_csv(
        tables / "g_mean_diff_family_bootstrap.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "",
                "d_g": "0.28",
                "SE_d_g": "0.19",
                "ci_low_d_g": "-0.10",
                "ci_high_d_g": "0.35",
            }
        ],
    )
    _write_csv(
        tables / "g_variance_ratio_family_bootstrap.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "",
                "VR_g": "1.29",
                "SE_logVR": "0.16",
                "ci_low": "0.79",
                "ci_high": "1.31",
            }
        ],
    )
    _write_csv(
        tables / "sibling_fe_g_outcome.csv",
        [
            {
                "cohort": "nlsy79",
                "outcome": "education",
                "status": "computed",
                "n_families": "25",
                "n_individuals": "54",
                "beta_g_total": "1.59",
                "beta_g_within": "2.07",
                "p_within": "0.0001",
                "r2_within": "0.46",
            }
        ],
    )
    _write_csv(
        tables / "intergenerational_g_transmission.csv",
        [
            {
                "model": "bivariate",
                "status": "computed",
                "n_pairs": "115",
                "beta_mother_g": "0.45",
                "beta_parent_ed": "",
                "p": "0.0001",
                "r2": "0.23",
            }
        ],
    )
    _write_csv(
        tables / "subtest_profile_tilt.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "n_used_education": "9258",
                "d_tilt": "-0.43",
                "tilt_g_corr": "-0.01",
                "tilt_incremental_r2_education": "0.0024",
                "p_tilt_incremental": "0.0001",
            }
        ],
    )
    _write_csv(
        tables / "sibling_fe_cross_cohort_contrasts.csv",
        [
            {
                "outcome": "education",
                "cohort_a": "nlsy79",
                "cohort_b": "nlsy97",
                "status": "computed",
                "diff_b_minus_a": "-0.8",
                "p_value_diff": "0.04",
            }
        ],
    )
    _write_csv(
        tables / "intergenerational_g_attenuation.csv",
        [
            {
                "status": "computed",
                "n_pairs_bivariate": "115",
                "n_pairs_ses_controlled": "114",
                "attenuation_abs": "0.05",
                "attenuation_pct": "10.1",
                "delta_r2": "-0.002",
                "beta_parent_ed": "0.04",
            }
        ],
    )
    _write_csv(
        tables / "degree_threshold_outcomes.csv",
        [
            {
                "cohort": "nlsy79",
                "threshold": "ba_or_more",
                "status": "computed",
                "n_used": "9000",
                "n_positive": "2500",
                "prevalence": "0.28",
                "odds_ratio_g": "4.9",
                "p_value_beta_g": "0.0",
                "pseudo_r2": "0.20",
            }
        ],
    )
    _write_csv(
        tables / "explicit_degree_outcomes.csv",
        [
            {
                "cohort": "nlsy79",
                "threshold": "ba_or_more_explicit",
                "degree_col": "degree_2000",
                "age_col": "age_2000",
                "status": "computed",
                "n_used": "5200",
                "n_positive": "1700",
                "prevalence": "0.33",
                "odds_ratio_g": "4.2",
                "p_value_beta_g": "0.0",
                "pseudo_r2": "0.18",
            }
        ],
    )
    _write_csv(
        tables / "age_matched_outcome_validity.csv",
        [
            {
                "outcome": "employment",
                "cohort": "nlsy79",
                "age_col": "age_2000",
                "status": "computed",
                "n_used": "3290",
                "model_type": "logit",
                "beta_g": "0.66",
                "p_value_beta_g": "0.0",
                "r2_or_pseudo_r2": "0.04",
            }
        ],
    )
    _write_csv(
        tables / "age_matched_cross_cohort_contrasts.csv",
        [
            {
                "outcome": "employment",
                "cohort_a": "nlsy79",
                "age_col_a": "age_2000",
                "cohort_b": "nlsy97",
                "age_col_b": "age_2021",
                "status": "computed",
                "overlap_min": "37",
                "overlap_max": "42",
                "beta_a": "0.67",
                "beta_b": "0.66",
                "diff_b_minus_a": "-0.004",
                "p_value_diff": "0.95",
            }
        ],
    )
    _write_csv(
        tables / "nlsy97_income_earnings_trajectories.csv",
        [
            {
                "cohort": "nlsy97",
                "outcome": "annual_earnings",
                "model": "annualized_log_change",
                "status": "computed",
                "n_used": "3800",
                "mean_annualized_log_change": "0.03",
                "beta_g": "0.02",
                "p_value_beta_g": "0.0001",
                "r2": "0.01",
            }
        ],
    )
    _write_csv(
        tables / "nlsy97_employment_persistence.csv",
        [
            {
                "cohort": "nlsy97",
                "model": "persistent_employment_2019_2021",
                "status": "computed",
                "n_used": "4200",
                "n_positive": "3000",
                "prevalence": "0.71",
                "odds_ratio_g": "1.40",
                "p_value_beta_g": "0.0001",
                "pseudo_r2": "0.03",
            }
        ],
    )
    _write_csv(
        tables / "cnlsy_adult_outcome_associations.csv",
        [
            {
                "cohort": "cnlsy",
                "outcome": "education_years_2014",
                "model_type": "ols",
                "status": "computed",
                "n_used": "171",
                "mean_outcome": "2.07",
                "beta_g": "0.07",
                "p_value_beta_g": "0.09",
                "r2_or_pseudo_r2": "0.09",
            }
        ],
    )
    _write_csv(
        tables / "cnlsy_carryover_net_mother_ses_summary.csv",
        [
            {
                "cohort": "cnlsy",
                "outcome": "log_wage_income_2014",
                "model_type": "ols_log_income",
                "status": "computed",
                "n_baseline": "150",
                "n_mother_ses": "140",
                "beta_g_baseline": "0.22",
                "beta_g_mother_ses": "0.18",
                "attenuation_pct": "18.2",
                "delta_r2_or_pseudo_r2": "0.02",
            }
        ],
    )
    _write_csv(
        tables / "nonlinear_threshold_outcome_summary.csv",
        [
            {
                "cohort": "nlsy79",
                "outcome": "log_annual_earnings",
                "outcome_type": "continuous",
                "status": "computed",
                "n_linear": "5000",
                "beta_g_sq": "0.04",
                "p_value_g_sq": "0.03",
                "delta_fit_linear_to_quadratic": "0.002",
                "threshold_odds_ratio": "",
                "threshold_beta": "0.18",
                "p_value_threshold": "0.001",
            }
        ],
    )
    _write_csv(
        tables / "nlsy79_mediation_summary.csv",
        [
            {
                "cohort": "nlsy79",
                "outcome": "log_annual_earnings",
                "model": "plus_all_mediators",
                "status": "computed",
                "n_used": "2400",
                "beta_g_baseline": "0.32",
                "beta_g_model": "0.17",
                "pct_attenuation_g": "46.9",
                "delta_r2": "0.08",
                "mediators_in_model": "education_years;employment_2000;__job_zone",
            }
        ],
    )
    _write_csv(
        tables / "sibling_discordance.csv",
        [
            {
                "cohort": "nlsy79",
                "outcome": "earnings",
                "status": "computed",
                "n_pairs": "320",
                "n_families": "200",
                "mean_abs_g_diff": "0.44",
                "mean_abs_outcome_diff": "12000",
                "corr_abs_diff": "0.18",
                "beta_abs_g_diff": "9500",
                "p_value_abs_g_diff": "0.0002",
                "r2": "0.04",
            }
        ],
    )
    _write_csv(
        tables / "nlsy97_occupational_mobility_summary.csv",
        [
            {
                "cohort": "nlsy97",
                "status": "computed",
                "n_with_2plus_occupation_waves": "73",
                "n_with_major_group_start_end": "69",
                "n_changed_major_group": "38",
                "pct_changed_major_group": "0.55",
                "n_upward_to_management_professional": "4",
                "n_downward_from_management_professional": "10",
                "mean_year_gap": "5.8",
                "top_wave_pair": "2013->2021",
            }
        ],
    )
    _write_csv(
        tables / "nlsy97_occupational_mobility_models.csv",
        [
            {
                "cohort": "nlsy97",
                "model": "any_major_group_change",
                "status": "computed",
                "n_used": "69",
                "n_positive": "38",
                "prevalence": "0.55",
                "odds_ratio_g": "0.88",
                "p_value_beta_g": "0.40",
                "pseudo_r2": "0.02",
            }
        ],
    )
    _write_csv(
        tables / "nlsy97_income_earnings_volatility.csv",
        [
            {
                "cohort": "nlsy97",
                "outcome": "annual_earnings",
                "model": "abs_annualized_log_change",
                "status": "computed",
                "n_used": "3000",
                "mean_abs_annualized_log_change": "0.14",
                "instability_cutoff": "0.19",
                "prevalence": "",
                "beta_g": "-0.03",
                "odds_ratio_g": "",
                "p_value_beta_g": "0.002",
                "r2_or_pseudo_r2": "0.04",
            }
        ],
    )
    _write_csv(
        tables / "nlsy97_employment_instability.csv",
        [
            {
                "cohort": "nlsy97",
                "model": "any_transition_2011_2021",
                "status": "computed",
                "n_used": "4200",
                "n_positive": "1500",
                "prevalence": "0.36",
                "odds_ratio_g": "0.81",
                "p_value_beta_g": "0.0003",
                "pseudo_r2": "0.02",
            }
        ],
    )
    _write_csv(
        tables / "nlsy97_unemployment_insurance.csv",
        [
            {
                "cohort": "nlsy97",
                "year": "2019",
                "model": "any_ui_receipt",
                "status": "computed",
                "n_used": "4300",
                "n_positive": "510",
                "prevalence": "0.119",
                "beta_g": "-0.25",
                "odds_ratio_g": "0.78",
                "p_value_beta_g": "0.0001",
                "r2_or_pseudo_r2": "0.03",
            }
        ],
    )
    _write_csv(
        tables / "g_employment_outcomes.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "outcome_col": "employment_2000",
                "age_col": "age_2000",
                "n_used": "5781",
                "n_employed": "4752",
                "prevalence": "0.82",
                "odds_ratio_g": "1.90",
                "p_value_beta_g": "0.0",
                "pseudo_r2": "0.04",
            }
        ],
    )
    _write_csv(
        tables / "cnlsy_employment_2014.csv",
        [
            {
                "cohort": "cnlsy",
                "status": "computed",
                "n_used": "93",
                "n_employed": "64",
                "prevalence": "0.688",
                "age_min_used": "14.9",
                "age_max_used": "18.0",
                "odds_ratio_g": "1.7",
                "p_value_beta_g": "0.03",
                "pseudo_r2": "0.07",
            }
        ],
    )
    _write_csv(
        tables / "subtest_profile_tilt_summary.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "d_g_proxy": "0.21",
                "d_tilt": "-0.43",
                "tilt_to_g_ratio_abs": "2.0",
                "incremental_r2_band": "very_small",
                "interpretation": "small_add_on",
            }
        ],
    )
    _write_csv(
        tables / "race_invariance_eligibility.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "n_groups": "3",
                "smallest_group_n": "120",
                "metric_pass": "True",
                "scalar_pass": "False",
                "reason_d_g": "scalar_gate:failed_delta_srmr",
            }
        ],
    )
    _write_csv(
        tables / "nlsy79_occupation_major_group_summary.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "",
                "occupation_group": "management_professional_related",
                "occupation_group_label": "Management/professional/related",
                "n_used": "2500",
                "share_used": "0.48",
                "mean_g_proxy": "0.42",
                "mean_education_years": "15.4",
                "mean_household_income": "54000",
            }
        ],
    )
    _write_csv(
        tables / "nlsy79_high_skill_occupation_outcome.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "",
                "outcome": "management_professional_related",
                "n_used": "5200",
                "n_positive": "2500",
                "prevalence": "0.48",
                "odds_ratio_g": "2.4",
                "p_value_beta_g": "0.0001",
                "pseudo_r2": "0.08",
            }
        ],
    )
    _write_csv(
        tables / "nlsy79_job_zone_mapping_quality.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "n_occ_nonmissing": "5233",
                "n_matched_exact": "2213",
                "n_matched_prefix_only": "900",
                "n_matched_any": "3113",
                "pct_matched_any": "0.595",
                "mean_job_zone": "3.14",
            }
        ],
    )
    _write_csv(
        tables / "nlsy79_job_zone_complexity_outcome.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "outcome": "job_zone",
                "n_used": "3113",
                "beta_g": "0.338",
                "p_value_beta_g": "0.0000000001",
                "beta_age": "0.010",
                "p_value_beta_age": "0.31",
                "r2": "0.068",
            }
        ],
    )
    _write_csv(
        tables / "nlsy79_job_pay_mismatch_summary.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "group": "overall",
                "n_used": "2500",
                "share_used": "1.0",
                "mean_job_zone": "3.1",
                "mean_annual_earnings": "42000",
                "mean_pay_residual_z": "0.0",
                "mean_g_proxy": "0.08",
                "mean_education_years": "13.7",
            }
        ],
    )
    _write_csv(
        tables / "nlsy79_job_pay_mismatch_models.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "outcome": "pay_residual_z",
                "model_family": "ols_residual",
                "n_used": "2500",
                "beta_g": "0.15",
                "p_value_beta_g": "0.01",
                "r2_or_pseudo_r2": "0.02",
            }
        ],
    )
    _write_csv(
        tables / "nlsy79_occupation_education_mapping_quality.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "n_occ_nonmissing": "5233",
                "n_matched_exact": "2800",
                "n_matched_prefix_only": "1320",
                "n_matched_any": "4120",
                "pct_matched_any": "0.787",
                "mean_required_education_years": "14.8",
                "mean_bachelor_plus_share": "0.43",
                "modal_required_education_label": "Bachelor's Degree",
            }
        ],
    )
    _write_csv(
        tables / "nlsy79_occupation_education_requirement_outcome.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "outcome": "required_education_years",
                "n_used": "4120",
                "beta_g": "0.52",
                "p_value_beta_g": "0.0000000001",
                "beta_age": "0.010",
                "p_value_beta_age": "0.31",
                "r2": "0.11",
                "mean_outcome": "14.8",
            },
            {
                "cohort": "nlsy79",
                "status": "computed",
                "outcome": "bachelor_plus_share",
                "n_used": "4120",
                "beta_g": "0.07",
                "p_value_beta_g": "0.00000001",
                "beta_age": "0.001",
                "p_value_beta_age": "0.40",
                "r2": "0.09",
                "mean_outcome": "0.43",
            },
        ],
    )
    _write_csv(
        tables / "nlsy79_education_job_mismatch_summary.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "group": "overall",
                "n_used": "2500",
                "share_used": "1.0",
                "mean_mismatch_years": "0.45",
                "mean_g_proxy": "0.12",
                "mean_education_years": "13.8",
                "mean_required_education_years": "13.3",
                "mean_annual_earnings": "41200",
            }
        ],
    )
    _write_csv(
        tables / "nlsy79_education_job_mismatch_models.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "outcome": "mismatch_years",
                "model_family": "ols",
                "n_used": "2500",
                "beta_g": "0.11",
                "p_value_beta_g": "0.02",
                "beta_age": "0.01",
                "p_value_beta_age": "0.30",
                "r2_or_pseudo_r2": "0.01",
            }
        ],
    )
    _write_csv(
        tables / "nlsy97_adult_occupation_major_group_summary.csv",
        [
            {
                "cohort": "nlsy97",
                "status": "computed",
                "reason": "",
                "occupation_group": "management_professional_related",
                "occupation_group_label": "Management/professional/related",
                "n_total": "6992",
                "n_with_any_occupation": "779",
                "n_used": "197",
                "share_used": "0.262",
                "mean_g_proxy": "0.36",
                "mean_education_years": "15.8",
                "top_source_wave": "2013",
            }
        ],
    )
    _write_csv(
        tables / "nlsy97_high_skill_occupation_outcome.csv",
        [
            {
                "cohort": "nlsy97",
                "status": "computed",
                "reason": "",
                "outcome": "management_professional_related",
                "n_used": "752",
                "n_positive": "197",
                "prevalence": "0.262",
                "odds_ratio_g": "2.09",
                "p_value_beta_g": "0.000000002",
                "pseudo_r2": "0.045",
            }
        ],
    )

    lock_dir = tables / "publication_results_lock"
    lock_dir.mkdir(parents=True, exist_ok=True)
    (tables / "publication_results_lock.zip").write_text("zip", encoding="utf-8")
    (lock_dir / "manuscript_results_lock.md").write_text("# lock", encoding="utf-8")
    (lock_dir / "publication_results_lock_manifest.csv").write_text(
        "path,sha256,bytes\nx,abc,1\n", encoding="utf-8"
    )
    (tables / "publication_snapshot_manifest.csv").write_text(
        "path,sha256,bytes\nx,abc,1\n", encoding="utf-8"
    )

    model_root = root / "outputs/model_fits/bootstrap_inference/nlsy79"
    (model_root / "full_sample").mkdir(parents=True, exist_ok=True)
    for idx in range(3):
        (model_root / f"rep_{idx:04d}").mkdir(parents=True, exist_ok=True)

    output_path = tables / "results_snapshot.md"
    result = _run_script(
        "--project-root",
        str(root),
        "--output",
        str(output_path),
        "--include-run-summary",
    )
    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    text = output_path.read_text(encoding="utf-8")
    assert "# sexg results snapshot" in text
    assert "## Warnings" in text
    assert "## Bootstrap vs baseline deltas" in text
    assert "## Bootstrap inference coverage" in text
    assert "in_bootstrap_tables" in text
    assert "manifest_status" in text
    assert "manifest_attempted" in text
    assert "### g_mean_diff" in text
    assert "### g_variance_ratio" in text
    assert "| cohort | baseline | bootstrap | delta |" in text
    assert "nlsy79" in text
    assert "| nlsy79 | 0.3 | 0.28 | -0.02 |" in text
    assert "## Publication lock artifacts" in text
    assert "publication_results_lock.zip" in text
    assert "publication_snapshot_manifest.csv" in text
    assert "### Sibling fixed-effects outcome associations" in text
    assert "### Within-family cross-cohort contrasts" in text
    assert "### Intergenerational mother-child `g_proxy` transmission" in text
    assert "### SES attenuation of mother-child `g_proxy` transmission" in text
    assert "### Verbal-quantitative subtest profile tilt" in text
    assert "### Tilt interpretation relative to `g_proxy`" in text
    assert "### Degree-threshold proxy outcomes" in text
    assert "### Explicit coded degree outcomes" in text
    assert "### Age-matched cross-cohort outcome validity" in text
    assert "### Age-matched cross-cohort contrasts" in text
    assert "### NLSY97 two-wave income and earnings trajectories" in text
    assert "### NLSY97 labor-force persistence and employment transitions" in text
    assert "### NLSY97 multi-wave employment instability" in text
    assert "### CNLSY 2014 late-adolescent/adult outcome associations" in text
    assert "### CNLSY child `g_proxy` carryover net mother SES" in text
    assert "### Nonlinear and threshold outcome models" in text
    assert "### Sibling discordance beyond education" in text
    assert "### NLSY79 mediation of earnings and income associations" in text
    assert "### NLSY97 two-wave income and earnings volatility" in text
    assert "### NLSY79 occupation major-group summary" in text
    assert "### NLSY79 management/professional occupation proxy" in text
    assert "### NLSY79 Job Zone mapping quality" in text
    assert "### NLSY79 Job Zone complexity association" in text
    assert "### NLSY79 job-complexity vs pay mismatch summary" in text
    assert "### NLSY79 job-complexity vs pay mismatch associations" in text
    assert "### NLSY79 occupation education-requirement mapping quality" in text
    assert "### NLSY79 occupation education-requirement associations" in text
    assert "### NLSY79 education-job mismatch summary" in text
    assert "### NLSY79 education-job mismatch associations" in text
    assert "### NLSY97 latest-adult occupation major-group summary" in text
    assert "### NLSY97 bounded occupational mobility" in text
    assert "### NLSY97 occupational mobility associations" in text
    assert "### NLSY97 latest-adult management/professional occupation proxy" in text
    assert "### `g_proxy` associations with employment status" in text
    assert "### CNLSY 2014 employment-status association" in text
    assert "### Race/ethnicity measurement invariance" in text


def test_script_93_succeeds_and_notes_missing_bootstrap(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    tables = root / "outputs/tables"
    _write_csv(
        tables / "g_mean_diff.csv",
        [{"cohort": "nlsy79", "d_g": "0.3", "SE_d_g": "0.03", "ci_low_d_g": "0.24", "ci_high_d_g": "0.36"}],
    )
    _write_csv(
        tables / "g_variance_ratio.csv",
        [{"cohort": "nlsy79", "VR_g": "1.31", "SE_logVR": "0.03", "ci_low": "1.16", "ci_high": "1.33"}],
    )

    output_path = tables / "results_snapshot.md"
    result = _run_script("--project-root", str(root), "--output", str(output_path))
    assert result.returncode == 0

    text = output_path.read_text(encoding="utf-8")
    assert "## Warnings" in text
    assert "g_mean_diff bootstrap table missing" in text
    assert "g_variance_ratio bootstrap table missing" in text
    assert "bootstrap: missing" in text


def test_script_93_fails_cleanly_when_baseline_missing(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    tables = root / "outputs/tables"
    _write_csv(
        tables / "g_mean_diff.csv",
        [{"cohort": "nlsy79", "d_g": "0.3", "SE_d_g": "0.03", "ci_low_d_g": "0.24", "ci_high_d_g": "0.36"}],
    )
    output_path = tables / "results_snapshot.md"

    result = _run_script("--project-root", str(root), "--output", str(output_path))
    assert result.returncode == 1
    assert "missing baseline table(s)" in result.stderr
    assert not output_path.exists()


def test_script_93_warnings_cover_baseline_and_bootstrap_coverage_gaps(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    tables = root / "outputs/tables"
    _write_csv(
        tables / "g_mean_diff.csv",
        [
            {
                "cohort": "nlsy79",
                "d_g": "0.3",
                "SE_d_g": "0.03",
                "ci_low_d_g": "0.24",
                "ci_high_d_g": "0.36",
            }
        ],
    )
    _write_csv(
        tables / "g_variance_ratio.csv",
        [
            {
                "cohort": "nlsy79",
                "VR_g": "1.31",
                "SE_logVR": "0.03",
                "ci_low": "1.16",
                "ci_high": "1.33",
            }
        ],
    )
    _write_csv(
        tables / "g_mean_diff_family_bootstrap.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "",
                "d_g": "0.28",
                "SE_d_g": "0.19",
                "ci_low_d_g": "-0.10",
                "ci_high_d_g": "0.35",
            },
            {
                "cohort": "nlsy97",
                "status": "computed",
                "reason": "",
                "d_g": "0.25",
                "SE_d_g": "0.20",
                "ci_low_d_g": "-0.11",
                "ci_high_d_g": "0.33",
            },
        ],
    )
    _write_csv(
        tables / "g_variance_ratio_family_bootstrap.csv",
        [
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "",
                "VR_g": "1.29",
                "SE_logVR": "0.16",
                "ci_low": "0.79",
                "ci_high": "1.31",
            },
            {
                "cohort": "nlsy97",
                "status": "computed",
                "reason": "",
                "VR_g": "1.21",
                "SE_logVR": "0.19",
                "ci_low": "0.74",
                "ci_high": "1.25",
            },
        ],
    )

    model_root = root / "outputs/model_fits/bootstrap_inference"
    nlsy79 = model_root / "nlsy79"
    (nlsy79 / "full_sample").mkdir(parents=True, exist_ok=True)
    (nlsy79 / "rep_0000").mkdir(parents=True, exist_ok=True)
    (nlsy79 / "rep_0002").mkdir(parents=True, exist_ok=True)
    # Minimal manifest structure needed for snapshot warnings.
    (tables / "inference_rerun_manifest_family_bootstrap.json").write_text(
        "{\n"
        '  "variant_token": "family_bootstrap",\n'
        '  "engine": "proxy",\n'
        '  "n_bootstrap": 2,\n'
        '  "cohorts": ["nlsy79"],\n'
        '  "cohort_details": [\n'
        '    {"cohort": "nlsy79", "status": "computed", "attempted": 2, "success": 2, "success_share": 1.0}\n'
        "  ]\n"
        "}\n",
        encoding="utf-8",
    )

    output_path = tables / "results_snapshot.md"
    result = _run_script(
        "--project-root",
        str(root),
        "--output",
        str(output_path),
        "--expected-bootstrap-reps",
        "5",
    )
    assert result.returncode == 0, result.stderr

    text = output_path.read_text(encoding="utf-8")
    assert "## Warnings" in text
    assert "bootstrap manifest gaps: nlsy79 (attempted=2 < expected=5)" in text
    assert "g_mean_diff baseline missing cohort(s): nlsy97" in text
    assert "g_variance_ratio baseline missing cohort(s): nlsy97" in text


def test_script_93_warns_when_bootstrap_table_mentions_cohort_with_zero_rep_dirs(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    tables = root / "outputs/tables"
    _write_csv(
        tables / "g_mean_diff.csv",
        [{"cohort": "nlsy79", "d_g": "0.3", "SE_d_g": "0.03", "ci_low_d_g": "0.24", "ci_high_d_g": "0.36"}],
    )
    _write_csv(
        tables / "g_variance_ratio.csv",
        [{"cohort": "nlsy79", "VR_g": "1.31", "SE_logVR": "0.03", "ci_low": "1.16", "ci_high": "1.33"}],
    )
    _write_csv(
        tables / "g_mean_diff_family_bootstrap.csv",
        [
            {
                "cohort": "cnlsy",
                "status": "computed",
                "reason": "",
                "d_g": "0.28",
                "SE_d_g": "0.19",
                "ci_low_d_g": "-0.10",
                "ci_high_d_g": "0.35",
            }
        ],
    )
    _write_csv(
        tables / "g_variance_ratio_family_bootstrap.csv",
        [
            {
                "cohort": "cnlsy",
                "status": "computed",
                "reason": "",
                "VR_g": "1.29",
                "SE_logVR": "0.16",
                "ci_low": "0.79",
                "ci_high": "1.31",
            }
        ],
    )

    model_root = root / "outputs/model_fits/bootstrap_inference"
    (model_root / "cnlsy" / "full_sample").mkdir(parents=True, exist_ok=True)

    output_path = tables / "results_snapshot.md"
    result = _run_script(
        "--project-root",
        str(root),
        "--output",
        str(output_path),
        "--expected-bootstrap-reps",
        "5",
    )
    assert result.returncode == 0, result.stderr
    text = output_path.read_text(encoding="utf-8")
    assert "bootstrap manifest gaps: cnlsy (attempted=missing)" in text


def test_script_93_expands_missing_g_mean_diff_warning_when_confirmatory_exclusions_present(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    tables = root / "outputs/tables"
    _write_csv(
        tables / "g_mean_diff.csv",
        [{"cohort": "nlsy79", "d_g": "0.3", "SE_d_g": "0.03", "ci_low_d_g": "0.24", "ci_high_d_g": "0.36"}],
    )
    _write_csv(
        tables / "g_variance_ratio.csv",
        [{"cohort": "nlsy79", "VR_g": "1.31", "SE_logVR": "0.03", "ci_low": "1.16", "ci_high": "1.33"}],
    )
    _write_csv(
        tables / "g_mean_diff_family_bootstrap.csv",
        [
            {
                "cohort": "nlsy97",
                "status": "computed",
                "reason": "",
                "d_g": "0.25",
                "SE_d_g": "0.20",
                "ci_low_d_g": "-0.11",
                "ci_high_d_g": "0.33",
            }
        ],
    )
    _write_csv(
        tables / "g_variance_ratio_family_bootstrap.csv",
        [
            {
                "cohort": "nlsy97",
                "status": "computed",
                "reason": "",
                "VR_g": "1.21",
                "SE_logVR": "0.19",
                "ci_low": "0.74",
                "ci_high": "1.25",
            }
        ],
    )
    _write_csv(
        tables / "confirmatory_exclusions.csv",
        [
            {
                "cohort": "nlsy97",
                "blocked_confirmatory": "False",
                "blocked_confirmatory_d_g": "True",
                "blocked_confirmatory_vr_g": "False",
                "reason": "invariance:scalar_gate:failed_delta_cfi",
                "reason_d_g": "invariance:scalar_gate:failed_delta_cfi",
                "reason_vr_g": "",
            }
        ],
    )

    output_path = tables / "results_snapshot.md"
    result = _run_script("--project-root", str(root), "--output", str(output_path))
    assert result.returncode == 0, result.stderr
    text = output_path.read_text(encoding="utf-8")
    assert "## Gating exclusions" in text
    assert "g_mean_diff baseline missing cohort(s): nlsy97 (primary d_g excluded: nlsy97: invariance:scalar_gate:failed_delta_cfi)" in text
