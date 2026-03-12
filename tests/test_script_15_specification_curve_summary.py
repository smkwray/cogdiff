from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ALLOWED_WEIGHT_CONCORDANCE_REASON_EXACT = {
    "ok",
    "discordant_weight_estimates",
    "nonpositive_vr_weight_pair",
    "nonconfirmatory_missing_weight_pair",
    "nonconfirmatory_missing_unweighted_baseline",
    "nonconfirmatory_missing_weighted_estimate",
    "nonconfirmatory_invalid_weight_pair_estimate",
}
ALLOWED_WEIGHT_CONCORDANCE_REASON_PREFIXES = (
    "nonconfirmatory_weighted_not_feasible:",
    "nonconfirmatory_unweighted_status_",
    "nonconfirmatory_weighted_",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_data(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _is_allowed_weight_reason(reason: object) -> bool:
    if reason is None or pd.isna(reason):
        return True
    token = str(reason).strip()
    if not token:
        return True
    if token in ALLOWED_WEIGHT_CONCORDANCE_REASON_EXACT:
        return True
    return any(token.startswith(prefix) for prefix in ALLOWED_WEIGHT_CONCORDANCE_REASON_PREFIXES)


def test_script_15_writes_specification_stability_summary(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(
        project_root / "config/paths.yml",
        "outputs_dir: outputs\n",
    )
    _write(
        project_root / "config/models.yml",
        """
reporting:
  specification_stability:
    sign_stability_threshold:
      robust: 0.9
      mixed: 0.6
    se_soi:
      d_g: 0.15
      vr_g: 0.2
    se_soi_center:
      d_g: 0.0
      vr_g: 1.0
    primary_estimate:
      d_g: 0.18
      vr_g: 1.1
    primary_deviation:
      d_g: 0.1
      vr_g: 0.2
""",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    _write_data(
        tables_dir / "robustness_inference.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "inference_method": "robust_cluster",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": 0.20,
                },
                {
                    "cohort": "nlsy79",
                    "inference_method": "family_bootstrap",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": 0.30,
                },
                {
                    "cohort": "nlsy79",
                    "inference_method": "robust_cluster",
                    "estimate_type": "vr_g",
                    "status": "computed",
                    "estimate": 1.10,
                },
                {
                    "cohort": "nlsy79",
                    "inference_method": "family_bootstrap",
                    "estimate_type": "vr_g",
                    "status": "computed",
                    "estimate": 1.15,
                },
                {
                    "cohort": "nlsy97",
                    "inference_method": "robust_cluster",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": -0.02,
                },
                {
                    "cohort": "nlsy97",
                    "inference_method": "family_bootstrap",
                    "estimate_type": "d_g",
                    "status": "not_run_placeholder",
                    "estimate": 0.10,
                },
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_weights.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "weight_mode": "unweighted",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": 0.22,
                },
                {
                    "cohort": "nlsy79",
                    "weight_mode": "weighted",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": 0.18,
                },
                {
                    "cohort": "nlsy79",
                    "weight_mode": "unweighted",
                    "estimate_type": "vr_g",
                    "status": "computed",
                    "estimate": 1.06,
                },
                {
                    "cohort": "nlsy79",
                    "weight_mode": "weighted",
                    "estimate_type": "vr_g",
                    "status": "computed",
                    "estimate": 1.16,
                },
                {
                    "cohort": "nlsy97",
                    "weight_mode": "unweighted",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": -0.01,
                },
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_sampling.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "sampling_scheme": "sibling_restricted",
                    "status": "computed",
                    "d_g": 0.18,
                },
                {
                    "cohort": "nlsy79",
                    "sampling_scheme": "full_cohort",
                    "status": "computed",
                    "d_g": 0.16,
                },
                {
                    "cohort": "nlsy79",
                    "sampling_scheme": "one_pair_per_family",
                    "status": "computed",
                    "d_g": 0.12,
                },
                {
                    "cohort": "nlsy97",
                    "sampling_scheme": "sibling_restricted",
                    "status": "computed",
                    "d_g": -0.01,
                },
                {
                    "cohort": "nlsy97",
                    "sampling_scheme": "full_cohort",
                    "status": "not_run_placeholder",
                    "d_g": 0.04,
                },
            ]
        ),
    )
    _write_data(
        tables_dir / "g_mean_diff.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "d_g": 0.18},
                {"cohort": "nlsy97", "d_g": -0.01},
            ]
        ),
    )
    _write_data(
        tables_dir / "g_variance_ratio.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "vr_g": 1.1},
                {"cohort": "nlsy97", "vr_g": 1.06},
            ]
        ),
    )

    script = _repo_root() / "scripts" / "15_specification_curve_summary.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root)],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(project_root / "outputs/tables/specification_stability_summary.csv")
    assert set(
        [
            "cohort",
            "estimand",
            "spec_count",
            "expected_spec_count",
            "spec_coverage_share",
            "spec_coverage_label",
            "not_feasible_count",
            "not_feasible_share",
            "median",
            "p2_5",
            "p97_5",
            "sign_stability_count",
            "sign_stability_share",
            "sign_stability_label",
            "magnitude_stability_count",
            "magnitude_stability_share",
            "robust_claim_sign_eligible",
            "robust_claim_primary_eligible",
            "robust_claim_weight_eligible",
            "robust_claim_warning_eligible",
            "robust_claim_eligible",
            "weight_pair_policy",
            "weight_concordance_checked",
            "weight_concordance_reason",
            "weight_sign_match",
            "weight_abs_diff",
            "weight_log_diff",
            "weight_diff_threshold",
            "weight_unweighted_status",
            "weight_unweighted_reason",
            "weight_weighted_status",
            "weight_weighted_reason",
            "primary_estimate",
            "median_deviation_from_primary",
            "primary_deviation_threshold",
        ]
    ).issubset(summary.columns)

    nlsy79_dg = summary[(summary["cohort"] == "nlsy79") & (summary["estimand"] == "d_g")].iloc[0]
    assert int(nlsy79_dg["spec_count"]) == 7
    assert int(nlsy79_dg["expected_spec_count"]) == 7
    assert float(nlsy79_dg["spec_coverage_share"]) == 1.0
    assert nlsy79_dg["spec_coverage_label"] == "complete"
    assert int(nlsy79_dg["sign_stability_count"]) == 7
    assert float(nlsy79_dg["sign_stability_share"]) == 1.0
    assert nlsy79_dg["sign_stability_label"] == "robust_positive"
    assert int(nlsy79_dg["magnitude_stability_count"]) == 1
    assert abs(float(nlsy79_dg["magnitude_stability_share"]) - (1 / 7)) < 1e-12
    assert nlsy79_dg["robust_claim_sign_eligible"] == True  # noqa: E712
    assert nlsy79_dg["robust_claim_primary_eligible"] == True  # noqa: E712
    assert nlsy79_dg["robust_claim_weight_eligible"] == True  # noqa: E712
    assert nlsy79_dg["robust_claim_warning_eligible"] == True  # noqa: E712
    assert nlsy79_dg["robust_claim_eligible"] == True  # noqa: E712
    assert int(nlsy79_dg["not_feasible_count"]) == 0
    assert float(nlsy79_dg["not_feasible_share"]) == 0.0
    assert float(nlsy79_dg["p2_5"]) < float(nlsy79_dg["median"]) < float(nlsy79_dg["p97_5"])

    nlsy79_vr = summary[(summary["cohort"] == "nlsy79") & (summary["estimand"] == "vr_g")].iloc[0]
    assert int(nlsy79_vr["spec_count"]) == 4
    assert int(nlsy79_vr["expected_spec_count"]) == 4
    assert float(nlsy79_vr["spec_coverage_share"]) == 1.0
    assert nlsy79_vr["spec_coverage_label"] == "complete"
    assert int(nlsy79_vr["sign_stability_count"]) == 4
    assert float(nlsy79_vr["sign_stability_share"]) == 1.0
    assert nlsy79_vr["sign_stability_label"] == "robust_positive"
    assert int(nlsy79_vr["magnitude_stability_count"]) == 4
    assert float(nlsy79_vr["magnitude_stability_share"]) == 1.0
    assert nlsy79_vr["robust_claim_sign_eligible"] == True  # noqa: E712
    assert nlsy79_vr["robust_claim_primary_eligible"] == True  # noqa: E712
    assert nlsy79_vr["robust_claim_weight_eligible"] == True  # noqa: E712
    assert nlsy79_vr["robust_claim_warning_eligible"] == True  # noqa: E712
    assert nlsy79_vr["robust_claim_eligible"] == True  # noqa: E712

    assert set(summary["cohort"]) >= {"nlsy79", "nlsy97"}


def test_script_15_supports_sign_bands_and_variance_ratio_log_support(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(
        project_root / "config/paths.yml",
        "outputs_dir: outputs\n",
    )
    _write(
        project_root / "config/models.yml",
        """
reporting:
  specification_stability:
    sign_stability_threshold:
      robust: 0.95
      mixed: 0.8
    se_soi:
      d_g: 0.1
      vr_g: 0.2
    se_soi_center:
      d_g: 0.0
      vr_g: 1.0
    primary_estimate:
      d_g: 0.2
      vr_g: 1.02
    primary_deviation:
      d_g: 0.1
      vr_g: 0.05
""",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    _write_data(
        tables_dir / "robustness_inference.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "inference_method": "robust_cluster",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": 0.30,
                },
                {
                    "cohort": "nlsy79",
                    "inference_method": "family_bootstrap",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": 0.20,
                },
                {
                    "cohort": "nlsy79",
                    "inference_method": "bootstrap_residual",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": 0.10,
                },
                {
                    "cohort": "nlsy79",
                    "inference_method": "residualization",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": -0.05,
                },
                {
                    "cohort": "nlsy79",
                    "inference_method": "robust_cluster",
                    "estimate_type": "vr_g",
                    "status": "computed",
                    "estimate": 1.12,
                },
                {
                    "cohort": "nlsy79",
                    "inference_method": "family_bootstrap",
                    "estimate_type": "vr_g",
                    "status": "computed",
                    "estimate": 1.04,
                },
                {
                    "cohort": "nlsy79",
                    "inference_method": "bootstrap_residual",
                    "estimate_type": "vr_g",
                    "status": "computed",
                    "estimate": 1.01,
                },
                {
                    "cohort": "nlsy79",
                    "inference_method": "residualization",
                    "estimate_type": "vr_g",
                    "status": "computed",
                    "estimate": 0.83,
                },
            ]
        ),
    )
    _write_data(
        tables_dir / "g_mean_diff.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "d_g": 0.16},
            ]
        ),
    )
    _write_data(
        tables_dir / "g_variance_ratio.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "vr_g": 1.02},
            ]
        ),
    )

    script = _repo_root() / "scripts" / "15_specification_curve_summary.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root)],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(project_root / "outputs/tables/specification_stability_summary.csv")

    nlsy79_dg = summary[(summary["cohort"] == "nlsy79") & (summary["estimand"] == "d_g")].iloc[0]
    assert nlsy79_dg["sign_stability_label"] == "unstable"
    assert int(nlsy79_dg["sign_stability_count"]) == 3
    assert float(nlsy79_dg["sign_stability_share"]) == 0.75
    assert int(nlsy79_dg["expected_spec_count"]) == 7
    assert abs(float(nlsy79_dg["spec_coverage_share"]) - (4 / 7)) < 1e-12
    assert nlsy79_dg["spec_coverage_label"] == "partial"
    assert int(nlsy79_dg["magnitude_stability_count"]) == 2
    assert float(nlsy79_dg["magnitude_stability_share"]) == 0.5
    assert nlsy79_dg["robust_claim_sign_eligible"] == False  # noqa: E712
    assert nlsy79_dg["robust_claim_primary_eligible"] == True  # noqa: E712
    assert nlsy79_dg["robust_claim_weight_eligible"] == False  # noqa: E712
    assert nlsy79_dg["robust_claim_warning_eligible"] == True  # noqa: E712
    assert nlsy79_dg["robust_claim_eligible"] == False  # noqa: E712

    nlsy79_vr = summary[(summary["cohort"] == "nlsy79") & (summary["estimand"] == "vr_g")].iloc[0]
    assert nlsy79_vr["sign_stability_label"] == "unstable"
    assert int(nlsy79_vr["sign_stability_count"]) == 3
    assert float(nlsy79_vr["sign_stability_share"]) == 0.75
    assert int(nlsy79_vr["expected_spec_count"]) == 4
    assert float(nlsy79_vr["spec_coverage_share"]) == 1.0
    assert nlsy79_vr["spec_coverage_label"] == "complete"
    assert int(nlsy79_vr["magnitude_stability_count"]) == 3
    assert float(nlsy79_vr["magnitude_stability_share"]) == 0.75
    assert nlsy79_vr["robust_claim_sign_eligible"] == False  # noqa: E712
    assert nlsy79_vr["robust_claim_primary_eligible"] == True  # noqa: E712
    assert nlsy79_vr["robust_claim_weight_eligible"] == False  # noqa: E712
    assert nlsy79_vr["robust_claim_warning_eligible"] == True  # noqa: E712
    assert nlsy79_vr["robust_claim_eligible"] == False  # noqa: E712


def test_script_15_handles_missing_robustness_inputs(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(
        project_root / "config/paths.yml",
        "outputs_dir: outputs\n",
    )

    script = _repo_root() / "scripts" / "15_specification_curve_summary.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root)],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(project_root / "outputs/tables/specification_stability_summary.csv")
    assert set(
        [
            "cohort",
            "estimand",
            "spec_count",
            "expected_spec_count",
            "spec_coverage_share",
            "spec_coverage_label",
            "median",
            "p2_5",
            "p97_5",
            "sign_stability_share",
            "sign_stability_label",
            "magnitude_stability_share",
        ]
    ).issubset(summary.columns)
    assert summary.empty


def test_script_15_counts_not_feasible_statuses(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    _write_data(
        tables_dir / "robustness_inference.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy97",
                    "inference_method": "robust_cluster",
                    "estimate_type": "d_g",
                    "status": "baseline_missing",
                    "estimate": None,
                },
                {
                    "cohort": "nlsy97",
                    "inference_method": "family_bootstrap",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": -0.05,
                },
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_weights.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy97",
                    "weight_mode": "unweighted",
                    "estimate_type": "d_g",
                    "status": "missing_source",
                    "estimate": None,
                },
                {
                    "cohort": "nlsy97",
                    "weight_mode": "weighted",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": -0.06,
                },
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_sampling.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy97",
                    "sampling_scheme": "sibling_restricted",
                    "status": "baseline_missing",
                    "d_g": None,
                },
                {
                    "cohort": "nlsy97",
                    "sampling_scheme": "full_cohort",
                    "status": "computed",
                    "d_g": -0.04,
                },
                {
                    "cohort": "nlsy97",
                    "sampling_scheme": "one_pair_per_family",
                    "status": "computed",
                    "d_g": -0.04,
                },
            ]
        ),
    )

    script = _repo_root() / "scripts" / "15_specification_curve_summary.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--cohort", "nlsy97"],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(project_root / "outputs/tables/specification_stability_summary.csv")
    row = summary[(summary["cohort"] == "nlsy97") & (summary["estimand"] == "d_g")].iloc[0]
    assert int(row["spec_count"]) == 4
    assert int(row["expected_spec_count"]) == 7
    assert int(row["not_feasible_count"]) == 3
    assert abs(float(row["not_feasible_share"]) - (3 / 7)) < 1e-12
    assert row["weight_concordance_reason"] == "nonconfirmatory_missing_unweighted_baseline"


def test_script_15_defaults_robust_claim_thresholds_when_missing_config(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(
        project_root / "config/paths.yml",
        "outputs_dir: outputs\n",
    )
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    _write_data(
        tables_dir / "robustness_inference.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "inference_method": "robust_cluster",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": 0.2,
                }
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_weights.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "weight_mode": "unweighted",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": 0.19,
                }
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_sampling.csv",
        pd.DataFrame(
            [
                {
                    "cohort": "nlsy79",
                    "sampling_scheme": "sibling_restricted",
                    "status": "computed",
                    "d_g": 0.18,
                }
            ]
        ),
    )

    script = _repo_root() / "scripts" / "15_specification_curve_summary.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root)],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(project_root / "outputs/tables/specification_stability_summary.csv")
    row = summary[(summary["cohort"] == "nlsy79") & (summary["estimand"] == "d_g")].iloc[0]
    assert row["robust_claim_sign_eligible"] == True  # noqa: E712
    assert row["robust_claim_primary_eligible"] == False  # noqa: E712
    assert row["robust_claim_weight_eligible"] == False  # noqa: E712
    assert row["robust_claim_warning_eligible"] == True  # noqa: E712
    assert row["robust_claim_eligible"] == False  # noqa: E712
    assert pd.isna(row["primary_estimate"])
    assert pd.isna(row["primary_deviation_threshold"])


def test_script_15_blocks_robust_claim_when_sem_warning_status_is_fail(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    _write_data(
        tables_dir / "robustness_inference.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "inference_method": "robust_cluster", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
                {"cohort": "nlsy79", "inference_method": "family_bootstrap", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_weights.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "weight_mode": "unweighted", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
                {"cohort": "nlsy79", "weight_mode": "weighted", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_sampling.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "sampling_scheme": "sibling_restricted", "status": "computed", "d_g": 0.2},
                {"cohort": "nlsy79", "sampling_scheme": "full_cohort", "status": "computed", "d_g": 0.2},
                {"cohort": "nlsy79", "sampling_scheme": "one_pair_per_family", "status": "computed", "d_g": 0.2},
            ]
        ),
    )
    _write_data(
        tables_dir / "g_mean_diff.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "status": "computed", "d_g": 0.2},
            ]
        ),
    )
    _write_data(
        tables_dir / "sem_run_status.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "status": "ok", "warning_policy_status": "fail"},
            ]
        ),
    )

    script = _repo_root() / "scripts" / "15_specification_curve_summary.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root)],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(project_root / "outputs/tables/specification_stability_summary.csv")
    row = summary[(summary["cohort"] == "nlsy79") & (summary["estimand"] == "d_g")].iloc[0]
    assert row["robust_claim_sign_eligible"] == True  # noqa: E712
    assert row["robust_claim_primary_eligible"] == True  # noqa: E712
    assert row["robust_claim_weight_eligible"] == True  # noqa: E712
    assert row["robust_claim_warning_eligible"] == False  # noqa: E712
    assert row["robust_claim_eligible"] == False  # noqa: E712


def test_script_15_blocks_robust_claim_when_weighted_unweighted_discordant(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    _write_data(
        tables_dir / "robustness_inference.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "inference_method": "robust_cluster", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
                {"cohort": "nlsy79", "inference_method": "family_bootstrap", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_weights.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "weight_mode": "unweighted", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
                {"cohort": "nlsy79", "weight_mode": "weighted", "estimate_type": "d_g", "status": "computed", "estimate": -0.2},
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_sampling.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "sampling_scheme": "sibling_restricted", "status": "computed", "d_g": 0.2},
                {"cohort": "nlsy79", "sampling_scheme": "full_cohort", "status": "computed", "d_g": 0.2},
                {"cohort": "nlsy79", "sampling_scheme": "one_pair_per_family", "status": "computed", "d_g": 0.2},
            ]
        ),
    )
    _write_data(
        tables_dir / "g_mean_diff.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "status": "computed", "d_g": 0.2},
            ]
        ),
    )

    script = _repo_root() / "scripts" / "15_specification_curve_summary.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root)],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(project_root / "outputs/tables/specification_stability_summary.csv")
    row = summary[(summary["cohort"] == "nlsy79") & (summary["estimand"] == "d_g")].iloc[0]
    assert row["robust_claim_sign_eligible"] == False  # noqa: E712
    assert row["robust_claim_primary_eligible"] == True  # noqa: E712
    assert row["weight_concordance_checked"] == True  # noqa: E712
    assert row["weight_sign_match"] == False  # noqa: E712
    assert row["robust_claim_weight_eligible"] == False  # noqa: E712
    assert row["robust_claim_eligible"] == False  # noqa: E712


def test_script_15_labels_weighted_not_feasible_reason(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    _write_data(
        tables_dir / "robustness_inference.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "inference_method": "robust_cluster", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
                {"cohort": "nlsy79", "inference_method": "family_bootstrap", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_weights.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "weight_mode": "unweighted", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
                {
                    "cohort": "nlsy79",
                    "weight_mode": "weighted",
                    "estimate_type": "d_g",
                    "status": "not_feasible",
                    "reason": "weight_quality_gate:effective_n_total_below_threshold",
                    "estimate": None,
                },
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_sampling.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "sampling_scheme": "sibling_restricted", "status": "computed", "d_g": 0.2},
                {"cohort": "nlsy79", "sampling_scheme": "full_cohort", "status": "computed", "d_g": 0.2},
                {"cohort": "nlsy79", "sampling_scheme": "one_pair_per_family", "status": "computed", "d_g": 0.2},
            ]
        ),
    )
    _write_data(
        tables_dir / "g_mean_diff.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "status": "computed", "d_g": 0.2},
            ]
        ),
    )

    script = _repo_root() / "scripts" / "15_specification_curve_summary.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root)],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(project_root / "outputs/tables/specification_stability_summary.csv")
    row = summary[(summary["cohort"] == "nlsy79") & (summary["estimand"] == "d_g")].iloc[0]
    assert str(row["weight_concordance_reason"]).startswith("nonconfirmatory_weighted_not_feasible:")
    assert row["weight_pair_policy"] == "replication_unweighted_primary_weighted_sensitivity"
    assert row["robust_claim_weight_eligible"] == False  # noqa: E712
    assert row["robust_claim_eligible"] == True  # noqa: E712


def test_script_15_weight_reason_taxonomy_is_restricted(tmp_path: Path) -> None:
    project_root = tmp_path.resolve()
    _write(project_root / "config/paths.yml", "outputs_dir: outputs\n")
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    _write_data(
        tables_dir / "robustness_inference.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "inference_method": "robust_cluster", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
                {"cohort": "nlsy79", "inference_method": "family_bootstrap", "estimate_type": "vr_g", "status": "computed", "estimate": 1.1},
                {"cohort": "nlsy97", "inference_method": "robust_cluster", "estimate_type": "d_g", "status": "computed", "estimate": 0.1},
                {"cohort": "nlsy97", "inference_method": "family_bootstrap", "estimate_type": "vr_g", "status": "computed", "estimate": 1.05},
                {"cohort": "cnlsy", "inference_method": "robust_cluster", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_weights.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "weight_mode": "unweighted", "estimate_type": "d_g", "status": "computed", "estimate": 0.2},
                {"cohort": "nlsy79", "weight_mode": "weighted", "estimate_type": "d_g", "status": "computed", "estimate": -0.2},
                {"cohort": "nlsy79", "weight_mode": "unweighted", "estimate_type": "vr_g", "status": "computed", "estimate": 1.1},
                {"cohort": "nlsy79", "weight_mode": "weighted", "estimate_type": "vr_g", "status": "computed", "estimate": 1.11},
                {"cohort": "nlsy97", "weight_mode": "unweighted", "estimate_type": "d_g", "status": "baseline_missing", "estimate": None},
                {"cohort": "nlsy97", "weight_mode": "weighted", "estimate_type": "d_g", "status": "computed", "estimate": 0.11},
                {"cohort": "nlsy97", "weight_mode": "unweighted", "estimate_type": "vr_g", "status": "computed", "estimate": 1.05},
                {
                    "cohort": "cnlsy",
                    "weight_mode": "unweighted",
                    "estimate_type": "d_g",
                    "status": "computed",
                    "estimate": 0.2,
                },
                {
                    "cohort": "cnlsy",
                    "weight_mode": "weighted",
                    "estimate_type": "d_g",
                    "status": "not_feasible",
                    "reason": "weight_quality_gate:effective_n_total_below_threshold",
                    "estimate": None,
                },
            ]
        ),
    )
    _write_data(
        tables_dir / "robustness_sampling.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "sampling_scheme": "sibling_restricted", "status": "computed", "d_g": 0.2},
                {"cohort": "nlsy97", "sampling_scheme": "sibling_restricted", "status": "computed", "d_g": 0.1},
                {"cohort": "cnlsy", "sampling_scheme": "sibling_restricted", "status": "computed", "d_g": 0.2},
            ]
        ),
    )
    _write_data(tables_dir / "g_mean_diff.csv", pd.DataFrame([{"cohort": "nlsy79", "d_g": 0.2}]))
    _write_data(tables_dir / "g_variance_ratio.csv", pd.DataFrame([{"cohort": "nlsy79", "vr_g": 1.1}]))

    script = _repo_root() / "scripts" / "15_specification_curve_summary.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(project_root), "--all"],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(project_root / "outputs/tables/specification_stability_summary.csv")
    assert not summary.empty
    reasons = summary["weight_concordance_reason"].dropna().astype(str).tolist()
    assert reasons
    assert all(_is_allowed_weight_reason(reason) for reason in reasons)
