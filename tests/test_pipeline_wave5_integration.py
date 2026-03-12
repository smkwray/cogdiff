from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _run_script(repo_root: Path, script: Path, args: list[str]) -> None:
    subprocess.run([sys.executable, str(script), *args], cwd=repo_root, check=True)


def test_wave5_scripts_run_sequentially_with_synthetic_artifacts(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    repo_root = _repo_root()
    script_04 = repo_root / "scripts/04_export_links.py"
    script_07 = repo_root / "scripts/07_fit_sem_models.py"
    script_08 = repo_root / "scripts/08_invariance_and_partial.py"
    script_09 = repo_root / "scripts/09_results_and_figures.py"
    script_10 = repo_root / "scripts/10_cnlsy_development.py"
    script_11 = repo_root / "scripts/11_robustness_suite.py"

    _write_file(
        root / "config/paths.yml",
        """
links_interim_dir: data/interim/links
processed_dir: data/processed
outputs_dir: outputs
sem_interim_dir: data/interim/sem
""",
    )
    _write_file(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: ["NO", CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
invariance:
  steps: [configural, metric, scalar, strict]
  partial_intercepts:
    nlsy79: [GS~1]
""",
    )
    _write_file(
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
pair_rules:
  relatedness_r: 0.5
  relationship_path: Gen1Housemates
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  age_resid_col: birth_year
  subtests: [NO, CS, AR, MK, WK, PC, GS, AS]
""",
    )
    _write_file(
        root / "config/nlsy97.yml",
        """
cohort: nlsy97
pair_rules:
  relatedness_r: 0.5
  relationship_path: any
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  age_resid_col: birth_year
  subtests: [NO, CS, AR, MK, WK, PC, GS, AS]
""",
    )
    _write_file(
        root / "config/cnlsy.yml",
        """
cohort: cnlsy
expected_age_range:
  min: 5
  max: 18
pair_rules:
  relatedness_r: 0.5
  relationship_path: Gen2Siblings
sample_construct:
  id_col: person_id
  sex_col: sex
  age_col: age
  subtests: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )

    _write_csv(
        root / "data/interim/links/links79_pair_expanded.csv",
        pd.DataFrame(
            [
                {
                    "SubjectTag": 11,
                    "PartnerTag": 12,
                    "R": 0.5,
                    "RelationshipPath": "Gen1Housemates",
                    "ExtendedID": "F1",
                },
                {
                    "SubjectTag": 12,
                    "PartnerTag": 11,
                    "R": 0.5,
                    "RelationshipPath": "Gen1Housemates",
                    "ExtendedID": "F1",
                },
                {
                    "SubjectTag": 13,
                    "PartnerTag": 14,
                    "R": 0.25,
                    "RelationshipPath": "Gen1Housemates",
                    "ExtendedID": "F2",
                },
            ]
        ),
    )
    _write_csv(
        root / "data/interim/links/links97_pair_expanded.csv",
        pd.DataFrame(
            [
                {
                    "SubjectTag": 21,
                    "PartnerTag": 22,
                    "R": 0.5,
                    "RelationshipPath": "Gen2Siblings",
                    "ExtendedID": "G1",
                },
                {
                    "SubjectTag": 23,
                    "PartnerTag": 24,
                    "R": 0.5,
                    "RelationshipPath": "Gen1Housemates",
                    "ExtendedID": "G2",
                },
            ]
        ),
    )
    _write_csv(
        root / "data/interim/links/links_cnlsy_pair_expanded.csv",
        pd.DataFrame(
            [
                {
                    "SubjectTag": 31,
                    "PartnerTag": 32,
                    "R": 0.5,
                    "RelationshipPath": "Gen2Siblings",
                    "ExtendedID": "C1",
                },
            ]
        ),
    )

    # Script 04: normalize sibling links before downstream pipeline stages.
    _run_script(root, script_04, ["--project-root", str(root), "--all"])
    assert (root / "data/interim/links/links79_links.csv").exists()
    assert (root / "data/interim/links/links97_links.csv").exists()
    assert (root / "data/interim/links/links_cnlsy_links.csv").exists()
    assert (root / "data/interim/links/link_exports.csv").exists()

    _write_csv(
        root / "data/processed/nlsy79_cfa_resid.csv",
        pd.DataFrame(
            {
                "person_id": [1, 2],
                "sex": ["F", "M"],
                "NO": [1.0, 2.0],
                "CS": [3.0, 4.0],
                "AR": [1.1, 1.2],
                "MK": [2.1, 2.2],
                "WK": [1.3, 1.4],
                "PC": [2.3, 2.4],
                "GS": [1.5, 1.6],
                "AS": [2.5, 2.6],
            }
        ),
    )
    _write_csv(
        root / "data/processed/nlsy97_cfa_resid.csv",
        pd.DataFrame(
            {
                "person_id": [3, 4],
                "sex": ["M", "F"],
                "NO": [1.0, 2.0],
                "CS": [3.0, 4.0],
                "AR": [1.1, 1.2],
                "MK": [2.1, 2.2],
                "WK": [1.3, 1.4],
                "PC": [2.3, 2.4],
                "GS": [1.5, 1.6],
                "AS": [2.5, 2.6],
            }
        ),
    )
    _write_csv(
        root / "data/processed/cnlsy_cfa_resid.csv",
        pd.DataFrame(
            {
                "person_id": [5, 6, 7, 8, 9, 10],
                "sex": ["M", "F", "M", "F", "M", "F"],
                "age": [5, 6, 7, 8, 9, 10],
                "PPVT": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                "PIAT_RR": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
                "PIAT_RC": [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
                "PIAT_MATH": [4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
                "DIGITSPAN": [5.0, 5.1, 5.2, 5.3, 5.4, 5.5],
                "g": [9.0, 10.0, 9.5, 10.5, 9.2, 10.2],
            }
        ),
    )

    # Script 07: dry-run SEM orchestration should only write request/model artifacts.
    _run_script(root, script_07, ["--project-root", str(root), "--all", "--dry-run"])
    status = pd.read_csv(root / "outputs/tables/sem_run_status.csv")
    assert set(status["cohort"]) == {"nlsy79", "nlsy97", "cnlsy"}
    assert set(status["status"]) == {"dry-run"}
    for cohort in ("nlsy79", "nlsy97", "cnlsy"):
        assert (root / f"data/interim/sem/{cohort}/request.json").exists()
        assert (root / f"data/interim/sem/{cohort}/model.lavaan").exists()
        assert (root / f"outputs/model_fits/{cohort}/run_status.json").exists()

    # Synthetic SEM artifacts for downstream results scripts.
    for cohort in ("nlsy79", "nlsy97", "cnlsy"):
        fit_dir = root / "outputs" / "model_fits" / cohort
        fit_dir.mkdir(parents=True, exist_ok=True)
        factors = ["g", "Speed", "Math", "Verbal", "Tech"] if cohort != "cnlsy" else ["g_cnlsy"]
        params_rows: list[dict[str, object]] = []
        latent_rows: list[dict[str, object]] = []
        for group in ["F", "M"]:
            for factor in factors:
                params_rows.extend(
                    [
                        {
                            "cohort": cohort,
                            "model_step": "configural",
                            "group": group,
                            "lhs": factor,
                            "op": "~1",
                            "rhs": "",
                            "est": 0.1 if factor == "g" or factor == "g_cnlsy" else 1.0,
                            "se": 0.05,
                        },
                        {
                            "cohort": cohort,
                            "model_step": "configural",
                            "group": group,
                            "lhs": factor,
                            "op": "~~",
                            "rhs": factor,
                            "est": 1.8 if group == "F" else 2.4,
                            "se": 0.2,
                        },
                    ]
                )
                latent_rows.append(
                    {
                        "cohort": cohort,
                        "group": group,
                        "factor": factor,
                        "mean": 0.0 if group == "F" else 0.2,
                        "var": 1.8 if group == "F" else 2.4,
                        "sd": 1.34,
                    }
                )
        _write_csv(fit_dir / "params.csv", pd.DataFrame(params_rows))
        _write_csv(fit_dir / "latent_summary.csv", pd.DataFrame(latent_rows))
        _write_csv(
            fit_dir / "fit_indices.csv",
            pd.DataFrame(
                {
                    "cohort": [cohort] * 4,
                    "model_step": ["configural", "metric", "scalar", "strict"],
                    "cfi": [0.96, 0.94, 0.92, 0.90],
                    "rmsea": [0.04, 0.042, 0.044, 0.046],
                    "srmr": [0.03, 0.032, 0.034, 0.036],
                }
            ),
        )

    _write_csv(
        root / "outputs/tables/sample_counts.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "n_male": 1, "n_female": 1, "n_input": 2, "n_after_age": 2, "n_after_test_rule": 2, "n_after_dedupe": 2},
                {"cohort": "nlsy97", "n_male": 1, "n_female": 1, "n_input": 2, "n_after_age": 2, "n_after_test_rule": 2, "n_after_dedupe": 2},
                {"cohort": "cnlsy", "n_male": 3, "n_female": 3, "n_input": 6, "n_after_age": 6, "n_after_test_rule": 6, "n_after_dedupe": 6},
            ]
        ),
    )
    _write_csv(
        root / "outputs/tables/residualization_diagnostics_nlsy79.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "subtest": "GS", "n_used": 1, "r2": 0.8, "resid_sd": 1.0},
                {"cohort": "nlsy79", "subtest": "AS", "n_used": 1, "r2": 0.7, "resid_sd": 1.0},
            ]
        ),
    )
    _write_csv(
        root / "outputs/tables/residualization_diagnostics_nlsy97.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy97", "subtest": "GS", "n_used": 1, "r2": 0.8, "resid_sd": 1.0},
                {"cohort": "nlsy97", "subtest": "AS", "n_used": 1, "r2": 0.7, "resid_sd": 1.0},
            ]
        ),
    )
    _write_csv(
        root / "outputs/tables/residualization_diagnostics_all.csv",
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "subtest": "GS", "n_used": 1, "r2": 0.8, "resid_sd": 1.0},
                {"cohort": "nlsy79", "subtest": "AS", "n_used": 1, "r2": 0.7, "resid_sd": 1.0},
                {"cohort": "nlsy97", "subtest": "GS", "n_used": 1, "r2": 0.8, "resid_sd": 1.1},
                {"cohort": "nlsy97", "subtest": "AS", "n_used": 1, "r2": 0.7, "resid_sd": 1.1},
                {"cohort": "cnlsy", "subtest": "g", "n_used": 1, "r2": 0.6, "resid_sd": 1.2},
            ]
        ),
    )

    # Script 08: build invariance and partial-parameter summaries.
    _run_script(root, script_08, ["--project-root", str(root), "--all"])
    for cohort in ("nlsy79", "nlsy97", "cnlsy"):
        assert (root / f"outputs/tables/{cohort}_invariance_summary.csv").exists()
        assert (root / f"outputs/tables/{cohort}_freed_parameters.csv").exists()
    assert (root / "outputs/tables/invariance_summary_all.csv").exists()

    # Script 09: produce canonical g tables and figures.
    _run_script(root, script_09, ["--project-root", str(root), "--all"])
    assert (root / "outputs/tables/g_mean_diff.csv").exists()
    assert (root / "outputs/tables/g_variance_ratio.csv").exists()
    assert (root / "outputs/tables/group_factor_diffs.csv").exists()
    for fig in (
        "g_mean_diff_forestplot.png",
        "vr_forestplot.png",
        "group_factor_gaps.png",
        "group_factor_vr.png",
    ):
        assert (root / "outputs/figures" / fig).exists()

    # Script 10: generate CNLSY age-bin and longitudinal outputs.
    _run_script(root, script_10, ["--project-root", str(root)])
    assert (root / "outputs/tables/cnlsy_agebin_summary.csv").exists()
    assert (root / "outputs/tables/cnlsy_longitudinal_model.csv").exists()
    assert (root / "outputs/figures/cnlsy_age_trends_mean.png").exists()
    assert (root / "outputs/figures/cnlsy_age_trends_vr.png").exists()

    # Script 11: build robustness placeholders from artifact tables.
    _run_script(root, script_11, ["--project-root", str(root), "--all"])
    assert (root / "outputs/tables/robustness_sampling.csv").exists()
    assert (root / "outputs/tables/robustness_age_adjustment.csv").exists()
    assert (root / "outputs/tables/robustness_model_forms.csv").exists()

    sampling = pd.read_csv(root / "outputs/tables/robustness_sampling.csv")
    age = pd.read_csv(root / "outputs/tables/robustness_age_adjustment.csv")
    forms = pd.read_csv(root / "outputs/tables/robustness_model_forms.csv")
    assert set(sampling["sampling_scheme"]) == {"sibling_restricted", "full_cohort", "one_pair_per_family"}
    assert set(age["age_adjustment"]) == {"quadratic", "cubic", "spline"}
    assert set(forms["model_form"]) == {"baseline", "single_factor_alt", "bifactor_alt"}
