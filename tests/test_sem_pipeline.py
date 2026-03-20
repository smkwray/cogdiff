from __future__ import annotations

import json
import importlib.util
import subprocess as sp
import subprocess
import sys
from pathlib import Path

import pandas as pd

from nls_pipeline.sem import cnlsy_model_syntax, hierarchical_model_syntax


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _script_07_module():
    path = _repo_root() / "scripts" / "07_fit_sem_models.py"
    spec = importlib.util.spec_from_file_location("stage07_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _script_08_module():
    path = _repo_root() / "scripts" / "08_invariance_and_partial.py"
    spec = importlib.util.spec_from_file_location("stage08_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeRResult:
    def __init__(self, stderr: str = "", stdout: str = "") -> None:
        self.stderr = stderr
        self.stdout = stdout


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_sem_model_syntax_builders() -> None:
    models_cfg = {
        "hierarchical_factors": {
            "speed": ["NO", "CS"],
            "math": ["AR", "MK"],
            "verbal": ["WK", "PC"],
            "technical": ["GS", "AS", "MC", "EI"],
        },
        "cnlsy_single_factor": ["PPVT", "PIAT_RR", "PIAT_RC", "PIAT_MATH", "DIGITSPAN"],
    }
    hier = hierarchical_model_syntax(models_cfg)
    cnlsy = cnlsy_model_syntax(models_cfg)
    assert "g =~ Speed + Math + Verbal + Tech" in hier
    assert "g_cnlsy =~ PPVT + PIAT_RR + PIAT_RC + PIAT_MATH + DIGITSPAN" == cnlsy


def test_script_07_dry_run_writes_sem_inputs(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(
        root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\nsem_interim_dir: data/interim/sem\n",
    )
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
invariance:
  steps: [configural, metric, scalar, strict]
  partial_intercepts:
    nlsy79: [GS~1, AS~1]
""",
    )
    _write(
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  sex_col: sex
""",
    )

    df = pd.DataFrame(
        {
            "person_id": [1, 2],
            "sex": ["F", "M"],
            "NO": [0.1, 0.2],
            "CS": [0.3, 0.4],
            "AR": [0.1, 0.1],
            "MK": [0.2, 0.2],
            "WK": [0.1, 0.2],
            "PC": [0.2, 0.3],
            "GS": [0.3, 0.4],
            "AS": [0.1, 0.2],
            "MC": [0.2, 0.4],
            "EI": [0.1, 0.2],
        }
    )
    in_path = root / "data/processed/nlsy79_cfa_resid.csv"
    in_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(in_path, index=False)

    script = _repo_root() / "scripts/07_fit_sem_models.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--dry-run",
            "--se-mode",
            "robust.cluster",
            "--cluster-col",
            "family_id",
            "--weight-col",
            "wgt",
        ],
        cwd=_repo_root(),
        check=True,
    )

    request_path = root / "data/interim/sem/nlsy79/request.json"
    assert request_path.exists()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["cohort"] == "nlsy79"
    assert request["group_col"] == "sex"
    assert request["data_csv"] == "sem_input.csv"
    assert not Path(request["data_csv"]).is_absolute()
    assert request["se_mode"] == "robust.cluster"
    assert request["cluster_col"] == "family_id"
    assert request["weight_col"] == "wgt"

    status = pd.read_csv(root / "outputs/tables/sem_run_status.csv")
    assert status.loc[0, "status"] == "dry-run"
    assert status.loc[0, "warning_policy_status"] == "clean"
    assert bool(status.loc[0, "warning_policy_enforced"]) is False
    assert bool(status.loc[0, "warning_policy_violated"]) is False
    assert status.loc[0, "input_data"] == "data/processed/nlsy79_cfa_resid.csv"
    assert status.loc[0, "request_file"] == "data/interim/sem/nlsy79/request.json"
    assert status.loc[0, "model_syntax_file"] == "data/interim/sem/nlsy79/model.lavaan"
    assert pd.isna(status.loc[0, "warning_policy_violation_reason"]) or status.loc[0, "warning_policy_violation_reason"] == ""
    assert int(status.loc[0, "warning_count"]) == 0
    triage = pd.read_csv(root / "outputs/tables/sem_warning_triage.csv")
    assert set(triage["warning_class"]) == {
        "se_not_computed",
        "non_convergence",
        "robust_fit_failed",
        "vcov_not_posdef_or_identification",
        "negative_variance_heywood",
        "gradient_warning",
        "modindices_unavailable",
        "vcov_not_posdef",
        "modindices_constraints_notice",
        "runtime_sem_failure",
        "unclassified_warning",
    }
    assert triage["count"].sum() == 0


def test_script_07_python_fallback_writes_sem_outputs(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(
        root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\nsem_interim_dir: data/interim/sem\n",
    )
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
invariance:
  steps: [configural, metric, scalar, strict]
  partial_intercepts:
    nlsy79: [GS~1, AS~1]
""",
    )
    _write(
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  sex_col: sex
""",
    )

    df = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "sex": ["F", "F", "F", "M", "M", "M"],
            "NO": [0.1, 0.2, 0.0, 0.4, 0.5, 0.6],
            "CS": [0.3, 0.1, 0.2, 0.6, 0.5, 0.4],
            "AR": [0.1, 0.1, 0.0, 0.2, 0.3, 0.4],
            "MK": [0.2, 0.2, 0.1, 0.3, 0.4, 0.5],
            "WK": [0.1, 0.2, 0.1, 0.2, 0.3, 0.4],
            "PC": [0.2, 0.3, 0.2, 0.3, 0.4, 0.5],
            "GS": [0.3, 0.4, 0.3, 0.4, 0.5, 0.6],
            "AS": [0.1, 0.2, 0.1, 0.2, 0.3, 0.4],
            "MC": [0.2, 0.4, 0.3, 0.4, 0.6, 0.7],
            "EI": [0.1, 0.2, 0.1, 0.2, 0.3, 0.4],
        }
    )
    in_path = root / "data/processed/nlsy79_cfa_resid.csv"
    in_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(in_path, index=False)

    script = _repo_root() / "scripts/07_fit_sem_models.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--skip-r",
            "--python-fallback",
        ],
        cwd=_repo_root(),
        check=True,
    )

    fit_dir = root / "outputs/model_fits/nlsy79"
    assert (fit_dir / "fit_indices.csv").exists()
    assert (fit_dir / "params.csv").exists()
    assert (fit_dir / "latent_summary.csv").exists()
    assert (fit_dir / "modindices.csv").exists()
    assert (fit_dir / "lavtestscore.csv").exists()
    assert (fit_dir / "SEM_FALLBACK_USED.flag").exists()

    status = pd.read_csv(root / "outputs/tables/sem_run_status.csv")
    assert status.loc[0, "status"] == "python-fallback"
    assert status.loc[0, "warning_policy_status"] == "clean"

    latent = pd.read_csv(fit_dir / "latent_summary.csv")
    assert {"cohort", "group", "factor", "mean", "var", "sd"}.issubset(latent.columns)
    assert set(latent["factor"]) >= {"g", "Speed", "Math", "Verbal", "Tech"}


def test_script_07_errors_when_r_unavailable_without_fallback_flag(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(
        root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\nsem_interim_dir: data/interim/sem\n",
    )
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
invariance:
  steps: [configural, metric, scalar, strict]
  partial_intercepts:
    nlsy79: [GS~1, AS~1]
""",
    )
    _write(
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  sex_col: sex
""",
    )

    df = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "sex": ["F", "F", "M", "M"],
            "NO": [0.1, 0.2, 0.4, 0.5],
            "CS": [0.3, 0.1, 0.6, 0.5],
            "AR": [0.1, 0.1, 0.2, 0.3],
            "MK": [0.2, 0.2, 0.3, 0.4],
            "WK": [0.1, 0.2, 0.2, 0.3],
            "PC": [0.2, 0.3, 0.3, 0.4],
            "GS": [0.3, 0.4, 0.4, 0.5],
            "AS": [0.1, 0.2, 0.2, 0.3],
            "MC": [0.2, 0.4, 0.4, 0.6],
            "EI": [0.1, 0.2, 0.2, 0.3],
        }
    )
    in_path = root / "data/processed/nlsy79_cfa_resid.csv"
    in_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(in_path, index=False)

    script = _repo_root() / "scripts/07_fit_sem_models.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
        ],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PATH": ""},  # Ensure Rscript not found
    )
    assert result.returncode != 0
    assert "python-fallback" in result.stderr or "Rscript" in result.stderr


def test_script_08_builds_invariance_outputs(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
invariance:
  steps: [configural, metric, scalar, strict]
  partial_intercepts:
    nlsy79: [GS~1, AS~1]
""",
    )
    fit = pd.DataFrame(
        {
            "cohort": ["nlsy79"] * 4,
            "model_step": ["configural", "metric", "scalar", "strict"],
            "cfi": [0.96, 0.955, 0.952, 0.950],
            "rmsea": [0.041, 0.042, 0.043, 0.044],
            "srmr": [0.031, 0.032, 0.033, 0.034],
        }
    )
    fit_path = root / "outputs/model_fits/nlsy79/fit_indices.csv"
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    fit.to_csv(fit_path, index=False)

    script = _repo_root() / "scripts/08_invariance_and_partial.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy79"],
        cwd=_repo_root(),
        check=True,
    )

    summary = pd.read_csv(root / "outputs/tables/nlsy79_invariance_summary.csv")
    freed = pd.read_csv(root / "outputs/tables/nlsy79_freed_parameters.csv")
    decisions = pd.read_csv(root / "outputs/tables/invariance_confirmatory_eligibility.csv")
    transitions = pd.read_csv(root / "outputs/tables/invariance_transition_checks.csv")
    replicability = pd.read_csv(root / "outputs/tables/partial_replicability_guard.csv")
    assert "delta_cfi" in summary.columns
    assert set(freed["parameter"]) == {"GS~1", "AS~1"}
    assert {
        "cohort",
        "confirmatory_d_g_eligible",
        "confirmatory_vr_g_eligible",
        "invariance_ok_for_d",
        "invariance_ok_for_vr",
    }.issubset(decisions.columns)
    assert {"transition", "passed", "reason"}.issubset(transitions.columns)
    assert {"cohort", "partial_replicability_checked", "partial_replicability_reason"}.issubset(replicability.columns)
    row = decisions.loc[decisions["cohort"] == "nlsy79"].iloc[0]
    assert bool(row["invariance_ok_for_d"]) == bool(row["confirmatory_d_g_eligible"])
    assert bool(row["invariance_ok_for_vr"]) == bool(row["confirmatory_vr_g_eligible"])


def test_script_08_selects_metric_gap_intercepts_under_policy_caps() -> None:
    module = _script_08_module()
    params = pd.DataFrame(
        [
            {"model_step": "metric", "group": "1", "lhs": "CS", "op": "~1", "est": 0.35},
            {"model_step": "metric", "group": "2", "lhs": "CS", "op": "~1", "est": -0.05},
            {"model_step": "metric", "group": "1", "lhs": "NO", "op": "~1", "est": 0.10},
            {"model_step": "metric", "group": "2", "lhs": "NO", "op": "~1", "est": -0.08},
            {"model_step": "metric", "group": "1", "lhs": "MK", "op": "~1", "est": 0.20},
            {"model_step": "metric", "group": "2", "lhs": "MK", "op": "~1", "est": -0.10},
            {"model_step": "metric", "group": "1", "lhs": "AR", "op": "~1", "est": 0.02},
            {"model_step": "metric", "group": "2", "lhs": "AR", "op": "~1", "est": -0.01},
            {"model_step": "metric", "group": "1", "lhs": "PC", "op": "~1", "est": 0.30},
            {"model_step": "metric", "group": "2", "lhs": "PC", "op": "~1", "est": -0.12},
            {"model_step": "metric", "group": "1", "lhs": "WK", "op": "~1", "est": 0.04},
            {"model_step": "metric", "group": "2", "lhs": "WK", "op": "~1", "est": -0.01},
            {"model_step": "metric", "group": "1", "lhs": "AS", "op": "~1", "est": 0.45},
            {"model_step": "metric", "group": "2", "lhs": "AS", "op": "~1", "est": -0.20},
            {"model_step": "metric", "group": "1", "lhs": "EI", "op": "~1", "est": 0.22},
            {"model_step": "metric", "group": "2", "lhs": "EI", "op": "~1", "est": -0.14},
            {"model_step": "metric", "group": "1", "lhs": "MC", "op": "~1", "est": 0.15},
            {"model_step": "metric", "group": "2", "lhs": "MC", "op": "~1", "est": -0.05},
            {"model_step": "metric", "group": "1", "lhs": "GS", "op": "~1", "est": 0.12},
            {"model_step": "metric", "group": "2", "lhs": "GS", "op": "~1", "est": -0.03},
        ]
    )
    indicator_map = {
        "Speed": ["NO", "CS"],
        "Math": ["AR", "MK"],
        "Verbal": ["WK", "PC"],
        "Tech": ["GS", "AS", "MC", "EI"],
    }
    partial_policy = {
        "max_free_per_factor": 2.0,
        "max_free_share_per_factor": 0.2,
        "min_invariant_share_per_factor": 0.5,
        "min_free_allowance_per_factor": 2.0,
    }
    candidates = module._metric_intercept_gap_candidates(
        params=params,
        indicator_map=indicator_map,
        reference_group="female",
    )
    selected, proposal = module._select_partial_intercepts(
        indicator_map=indicator_map,
        partial_policy=partial_policy,
        configured_partial=["PC~1"],
        candidates=candidates,
        candidate_source="metric_intercept_gap",
        selected_reason="metric_selected",
        exceeds_reason="metric_exceeds_factor_capacity",
        candidate_score_col="abs_intercept_gap",
    )
    assert set(selected) == {"CS~1", "MK~1", "PC~1", "AS~1", "EI~1"}
    selected_rows = proposal.loc[proposal["selected"] == True]
    assert len(selected_rows) == 5
    assert set(selected_rows["factor"]) == {"Speed", "Math", "Verbal", "Tech"}


def test_script_08_score_test_candidates_prioritize_scalar_intercepts() -> None:
    module = _script_08_module()
    indicator_map = {
        "Speed": ["NO", "CS"],
        "Math": ["AR", "MK"],
        "Verbal": ["WK", "PC"],
        "Tech": ["GS", "AS", "MC", "EI"],
    }
    score = pd.DataFrame(
        [
            {
                "model_step": "metric",
                "constraint_type": "intercept",
                "mapped_lhs": "AS",
                "x2": 999.0,
                "p_value": 0.0,
            },
            {
                "model_step": "scalar",
                "constraint_type": "loading",
                "mapped_lhs": "CS",
                "x2": 123.0,
                "p_value": 0.0,
            },
            {
                "model_step": "scalar",
                "constraint_type": "intercept",
                "mapped_lhs": "AS",
                "x2": 88.0,
                "p_value": 1e-6,
            },
            {
                "model_step": "scalar",
                "constraint_type": "intercept",
                "mapped_lhs": "CS",
                "x2": 45.0,
                "p_value": 1e-4,
            },
        ]
    )
    ranked = module._score_test_intercept_candidates(score_tests=score, indicator_map=indicator_map)
    assert ranked["indicator"].tolist() == ["AS", "CS"]
    assert ranked["rank"].tolist() == [1, 2]
    assert float(ranked.iloc[0]["score_test_x2"]) == 88.0


def test_script_08_partial_replicability_guard_downgrades_partial_refit_when_not_stable() -> None:
    module = _script_08_module()
    decisions = pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "gatekeeping_enabled": True,
                "confirmatory_d_g_eligible": True,
                "reason_d_g": pd.NA,
                "partial_refit_used": False,
                "total_freed_intercepts": 2,
            },
            {
                "cohort": "nlsy97",
                "gatekeeping_enabled": True,
                "confirmatory_d_g_eligible": True,
                "reason_d_g": pd.NA,
                "partial_refit_used": True,
                "total_freed_intercepts": 2,
            },
        ]
    )
    partial = pd.DataFrame(
        [
            {"cohort": "nlsy79", "indicator": "GS", "selected": True},
            {"cohort": "nlsy79", "indicator": "AS", "selected": True},
            {"cohort": "nlsy97", "indicator": "PC", "selected": True},
            {"cohort": "nlsy97", "indicator": "WK", "selected": True},
        ]
    )
    policy = {
        "enabled": True,
        "mode": "cross_cohort_overlap",
        "apply_when_partial_refit": True,
        "min_overlap_share": 0.7,
        "comparison_cohorts": ["nlsy79", "nlsy97"],
    }
    guarded, report = module._apply_partial_replicability_guard(
        decisions=decisions,
        partial_rows=partial,
        policy=policy,
    )
    row79 = guarded.loc[guarded["cohort"] == "nlsy79"].iloc[0]
    row97 = guarded.loc[guarded["cohort"] == "nlsy97"].iloc[0]
    assert bool(row79["confirmatory_d_g_eligible"]) is True
    assert bool(row79["partial_replicability_checked"]) is False
    assert bool(row97["confirmatory_d_g_eligible"]) is False
    assert bool(row97["invariance_ok_for_d"]) is False
    assert bool(row97["partial_replicability_checked"]) is True
    assert bool(row97["partial_replicability_pass"]) is False
    assert "partial_replicability_guard" in str(row97["reason_d_g"])
    assert "below_overlap_threshold" in set(report["partial_replicability_reason"])


def test_script_08_partial_replicability_guard_keeps_stable_partial_refit_confirmatory() -> None:
    module = _script_08_module()
    decisions = pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "gatekeeping_enabled": True,
                "confirmatory_d_g_eligible": True,
                "reason_d_g": pd.NA,
                "partial_refit_used": False,
                "total_freed_intercepts": 2,
            },
            {
                "cohort": "nlsy97",
                "gatekeeping_enabled": True,
                "confirmatory_d_g_eligible": True,
                "reason_d_g": pd.NA,
                "partial_refit_used": True,
                "total_freed_intercepts": 2,
            },
        ]
    )
    partial = pd.DataFrame(
        [
            {"cohort": "nlsy79", "indicator": "GS", "selected": True},
            {"cohort": "nlsy79", "indicator": "AS", "selected": True},
            {"cohort": "nlsy97", "indicator": "GS", "selected": True},
            {"cohort": "nlsy97", "indicator": "AS", "selected": True},
        ]
    )
    policy = {
        "enabled": True,
        "mode": "cross_cohort_overlap",
        "apply_when_partial_refit": True,
        "min_overlap_share": 0.7,
        "comparison_cohorts": ["nlsy79", "nlsy97"],
    }
    guarded, _ = module._apply_partial_replicability_guard(
        decisions=decisions,
        partial_rows=partial,
        policy=policy,
    )
    row97 = guarded.loc[guarded["cohort"] == "nlsy97"].iloc[0]
    assert bool(row97["confirmatory_d_g_eligible"]) is True
    assert bool(row97["invariance_ok_for_d"]) is True
    assert bool(row97["partial_replicability_checked"]) is True
    assert bool(row97["partial_replicability_pass"]) is True
    assert pd.isna(row97["reason_d_g"])


def test_script_08_partial_replicability_policy_parses_combined_bootstrap_fields() -> None:
    module = _script_08_module()
    parsed = module._partial_replicability_policy(
        {
            "invariance": {
                "gatekeeping": {
                    "partial_replicability": {
                        "enabled": True,
                        "mode": "combined",
                        "bootstrap_replicates": 120,
                        "bootstrap_seed": 99,
                        "min_bootstrap_indicator_share": 0.75,
                        "min_bootstrap_success_reps": 50,
                        "bootstrap_min_group_n": 20,
                        "bootstrap_cluster_cols": {
                            "nlsy97": ["R9708601", "R9708602"],
                        },
                    }
                }
            }
        }
    )
    assert parsed["mode"] == "combined"
    assert int(parsed["bootstrap_replicates"]) == 120
    assert int(parsed["bootstrap_seed"]) == 99
    assert float(parsed["min_bootstrap_indicator_share"]) == 0.75
    assert int(parsed["min_bootstrap_success_reps"]) == 50
    assert int(parsed["bootstrap_min_group_n"]) == 20
    assert parsed["bootstrap_cluster_cols"]["nlsy97"] == ["R9708601", "R9708602"]


def test_script_08_partial_replicability_guard_combined_mode_requires_bootstrap_pass() -> None:
    module = _script_08_module()
    decisions = pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "gatekeeping_enabled": True,
                "confirmatory_d_g_eligible": True,
                "reason_d_g": pd.NA,
                "partial_refit_used": False,
                "total_freed_intercepts": 2,
            },
            {
                "cohort": "nlsy97",
                "gatekeeping_enabled": True,
                "confirmatory_d_g_eligible": True,
                "reason_d_g": pd.NA,
                "partial_refit_used": True,
                "total_freed_intercepts": 2,
            },
        ]
    )
    partial = pd.DataFrame(
        [
            {"cohort": "nlsy79", "indicator": "GS", "selected": True},
            {"cohort": "nlsy79", "indicator": "AS", "selected": True},
            {"cohort": "nlsy97", "indicator": "GS", "selected": True},
            {"cohort": "nlsy97", "indicator": "AS", "selected": True},
        ]
    )
    policy = {
        "enabled": True,
        "mode": "combined",
        "apply_when_partial_refit": True,
        "min_overlap_share": 0.7,
        "comparison_cohorts": ["nlsy79", "nlsy97"],
    }
    bootstrap_support = pd.DataFrame(
        [
            {
                "cohort": "nlsy97",
                "bootstrap_checked": True,
                "bootstrap_pass": False,
                "bootstrap_reason": "below_bootstrap_threshold",
                "bootstrap_replicates_requested": 200,
                "bootstrap_successful_reps": 160,
                "bootstrap_min_indicator_share": 0.55,
                "bootstrap_threshold_share": 0.7,
                "bootstrap_threshold_success_reps": 100,
            }
        ]
    )
    guarded, report = module._apply_partial_replicability_guard(
        decisions=decisions,
        partial_rows=partial,
        policy=policy,
        bootstrap_support=bootstrap_support,
    )
    row97 = guarded.loc[guarded["cohort"] == "nlsy97"].iloc[0]
    assert bool(row97["confirmatory_d_g_eligible"]) is False
    assert bool(row97["invariance_ok_for_d"]) is False
    assert "partial_replicability_guard" in str(row97["reason_d_g"])
    assert bool(row97["partial_replicability_bootstrap_checked"]) is True
    assert bool(row97["partial_replicability_bootstrap_pass"]) is False
    assert "below_bootstrap_threshold" in set(report["partial_replicability_reason"])


def test_script_07_warning_classifier_detects_fail_and_caution() -> None:
    module = _script_07_module()
    stderr = """
Warning messages:
1: lavaan->lav_model_vcov():
   Could not compute standard errors! The information matrix could not be inverted.
2: lavaan->lav_fit_cfi_lavobject():
   computation of robust CFI failed.
3: In value[[3L]](cond):
   Modification indices unavailable for step 'configural'
"""
    parsed = module._classify_sem_warnings(stderr)
    assert parsed["warning_policy_status"] == "fail"
    assert parsed["warning_count"] == 3
    assert parsed["warning_class_counts"]["se_not_computed"] >= 1
    assert parsed["warning_class_counts"]["robust_fit_failed"] >= 1
    assert parsed["warning_class_counts"]["modindices_unavailable"] >= 1


def test_script_07_warning_classifier_detects_serious_classes_without_explicit_warning_count() -> None:
    module = _script_07_module()
    stderr = """
The model did not converge in the last iterations.
The latent covariance matrix is not positive definite.
A negative variance was found for a latent variable.
    Gradient was flat and optimization problem remains.
"""
    parsed = module._classify_sem_warnings(stderr)
    assert parsed["warning_policy_status"] == "fail"
    assert parsed["warning_count"] == 0
    assert parsed["warning_class_counts"]["non_convergence"] >= 1
    assert parsed["warning_class_counts"]["vcov_not_posdef_or_identification"] >= 1
    assert parsed["warning_class_counts"]["negative_variance_heywood"] >= 1
    assert parsed["warning_class_counts"]["gradient_warning"] >= 1


def test_script_07_warning_classifier_marks_runtime_failure() -> None:
    module = _script_07_module()
    parsed = module._classify_sem_warnings("", status="r-failed")
    assert parsed["warning_class_counts"]["runtime_sem_failure"] >= 1
    assert parsed["warning_policy_status"] == "fail"
    applied = module._apply_warning_policy(parsed, module.WarningPolicyConfig(enforced=True, threshold="fail"))
    assert applied["warning_policy_violated"] is True
    assert "classes=runtime_sem_failure" in str(applied["warning_policy_violation_reason"])


def test_script_07_warning_policy_non_enforced(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path.resolve()
    module = _script_07_module()
    _write(
        root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\nsem_interim_dir: data/interim/sem\n",
    )
    _write(
        root / "config/models.yml",
        """
reporting:
  warning_policy:
    enabled: false
    threshold: caution
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
invariance:
  steps: [configural, metric, scalar, strict]
  partial_intercepts:
    nlsy79: [GS~1, AS~1]
""",
    )
    _write(
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  sex_col: sex
""",
    )
    _write(root / "scripts/sem_fit.R", "")
    df = pd.DataFrame(
        {
            "person_id": [1, 2],
            "sex": ["F", "M"],
            "NO": [0.1, 0.2],
            "CS": [0.3, 0.4],
            "AR": [0.1, 0.1],
            "MK": [0.2, 0.2],
            "WK": [0.1, 0.2],
            "PC": [0.2, 0.3],
            "GS": [0.3, 0.4],
            "AS": [0.1, 0.2],
            "MC": [0.2, 0.4],
            "EI": [0.1, 0.2],
        }
    )
    in_path = root / "data/processed/nlsy79_cfa_resid.csv"
    in_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(in_path, index=False)

    monkeypatch.setattr(module, "rscript_path", lambda: Path("/tmp/does-not-matter"))
    monkeypatch.setattr(
        module,
        "run_sem_r_script",
        lambda **kwargs: _FakeRResult(
            stdout="",
            stderr="Warning in lavTestLRT: could not compute modification indices for configural step",
        ),
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["07_fit_sem_models.py", "--project-root", str(root), "--cohort", "nlsy79"],
    )

    assert module.main() == 0
    status = pd.read_csv(root / "outputs/tables/sem_run_status.csv")
    assert status.loc[0, "warning_policy_enforced"] == False
    assert status.loc[0, "warning_policy_violated"] == False
    assert pd.isna(status.loc[0, "warning_policy_violation_reason"]) or status.loc[0, "warning_policy_violation_reason"] == ""


def test_script_07_warning_policy_enforced_fails_on_serious_class_without_warning_count(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path.resolve()
    module = _script_07_module()
    _write(
        root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\nsem_interim_dir: data/interim/sem\n",
    )
    _write(
        root / "config/models.yml",
        """
reporting:
  warning_policy:
    enabled: false
    threshold: caution
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
invariance:
  steps: [configural, metric, scalar, strict]
  partial_intercepts:
    nlsy79: [GS~1, AS~1]
""",
    )
    _write(
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  sex_col: sex
""",
    )
    _write(root / "scripts/sem_fit.R", "")
    df = pd.DataFrame(
        {
            "person_id": [1, 2],
            "sex": ["F", "M"],
            "NO": [0.1, 0.2],
            "CS": [0.3, 0.4],
            "AR": [0.1, 0.1],
            "MK": [0.2, 0.2],
            "WK": [0.1, 0.2],
            "PC": [0.2, 0.3],
            "GS": [0.3, 0.4],
            "AS": [0.1, 0.2],
            "MC": [0.2, 0.4],
            "EI": [0.1, 0.2],
        }
    )
    in_path = root / "data/processed/nlsy79_cfa_resid.csv"
    in_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(in_path, index=False)

    monkeypatch.setattr(module, "rscript_path", lambda: Path("/tmp/does-not-matter"))
    monkeypatch.setattr(
        module,
        "run_sem_r_script",
        lambda **kwargs: _FakeRResult(
            stdout="",
            stderr="The model did not converge and the latent covariance matrix is not positive definite.",
        ),
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "07_fit_sem_models.py",
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--enforce-warning-policy",
            "--warning-policy-threshold",
            "fail",
        ],
    )

    assert module.main() == 1
    status = pd.read_csv(root / "outputs/tables/sem_run_status.csv")
    assert status.loc[0, "warning_policy_enforced"] == True
    assert status.loc[0, "warning_policy_violated"] == True
    assert int(status.loc[0, "warning_count"]) == 0
    assert "classes=non_convergence" in str(status.loc[0, "warning_policy_violation_reason"])

    triage = pd.read_csv(root / "outputs/tables/sem_warning_triage.csv")
    non_conv = int(triage.loc[(triage["warning_class"] == "non_convergence"), "count"].iloc[0])
    assert non_conv >= 1


def test_script_07_warning_policy_enforced_failure(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path.resolve()
    module = _script_07_module()
    _write(
        root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\nsem_interim_dir: data/interim/sem\n",
    )
    _write(
        root / "config/models.yml",
        """
reporting:
  warning_policy:
    enabled: false
    threshold: fail
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
invariance:
  steps: [configural, metric, scalar, strict]
  partial_intercepts:
    nlsy79: [GS~1, AS~1]
""",
    )
    _write(
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  sex_col: sex
""",
    )
    _write(root / "scripts/sem_fit.R", "")
    df = pd.DataFrame(
        {
            "person_id": [1, 2],
            "sex": ["F", "M"],
            "NO": [0.1, 0.2],
            "CS": [0.3, 0.4],
            "AR": [0.1, 0.1],
            "MK": [0.2, 0.2],
            "WK": [0.1, 0.2],
            "PC": [0.2, 0.3],
            "GS": [0.3, 0.4],
            "AS": [0.1, 0.2],
            "MC": [0.2, 0.4],
            "EI": [0.1, 0.2],
        }
    )
    in_path = root / "data/processed/nlsy79_cfa_resid.csv"
    in_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(in_path, index=False)

    monkeypatch.setattr(module, "rscript_path", lambda: Path("/tmp/does-not-matter"))
    monkeypatch.setattr(
        module,
        "run_sem_r_script",
        lambda **kwargs: _FakeRResult(
            stdout="",
            stderr="Warning in lavModel: could not compute standard errors. information matrix could not be inverted.",
        ),
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "07_fit_sem_models.py",
            "--project-root",
            str(root),
            "--cohort",
            "nlsy79",
            "--enforce-warning-policy",
            "--warning-policy-threshold",
            "fail",
        ],
    )

    assert module.main() == 1
    status = pd.read_csv(root / "outputs/tables/sem_run_status.csv")
    assert status.loc[0, "warning_policy_enforced"] == True
    assert status.loc[0, "warning_policy_violated"] == True
    assert "threshold=fail" in str(status.loc[0, "warning_policy_violation_reason"])


def test_script_07_returns_nonzero_on_r_failed_and_captures_streams(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path.resolve()
    module = _script_07_module()
    _write(
        root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\nsem_interim_dir: data/interim/sem\n",
    )
    _write(
        root / "config/models.yml",
        """
reporting:
  warning_policy:
    enabled: false
    threshold: fail
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
invariance:
  steps: [configural, metric, scalar, strict]
  partial_intercepts:
    nlsy79: [GS~1, AS~1]
""",
    )
    _write(
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  sex_col: sex
""",
    )
    _write(root / "scripts/sem_fit.R", "")
    df = pd.DataFrame(
        {
            "person_id": [1, 2],
            "sex": ["F", "M"],
            "NO": [0.1, 0.2],
            "CS": [0.3, 0.4],
            "AR": [0.1, 0.1],
            "MK": [0.2, 0.2],
            "WK": [0.1, 0.2],
            "PC": [0.2, 0.3],
            "GS": [0.3, 0.4],
            "AS": [0.1, 0.2],
            "MC": [0.2, 0.4],
            "EI": [0.1, 0.2],
        }
    )
    in_path = root / "data/processed/nlsy79_cfa_resid.csv"
    in_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(in_path, index=False)

    failing_exc = sp.CalledProcessError(
        returncode=1,
        cmd=["Rscript", "sem_fit.R"],
        output="Warning: could not compute standard errors",
        stderr="Error: failed to fit metric",
    )

    monkeypatch.setattr(module, "rscript_path", lambda: Path("/tmp/does-not-matter"))

    def _raise_called_process_error(**kwargs):
        raise failing_exc

    monkeypatch.setattr(module, "run_sem_r_script", _raise_called_process_error)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["07_fit_sem_models.py", "--project-root", str(root), "--cohort", "nlsy79"],
    )

    assert module.main() == 1
    status = pd.read_csv(root / "outputs/tables/sem_run_status.csv")
    assert status.loc[0, "status"] == "r-failed"
    assert "Warning: could not compute standard errors" in str(status.loc[0, "r_stdout"])
    assert "Error: failed to fit metric" in str(status.loc[0, "r_stderr"])
    assert status.loc[0, "warning_policy_status"] == "fail"
