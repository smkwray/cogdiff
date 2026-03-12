from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_params(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_script_09_writes_core_tables_and_figures(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
invariance:
  steps: [configural, metric, scalar, strict]
""",
    )

    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"cohort": "nlsy79", "n_male": 100, "n_female": 120},
            {"cohort": "nlsy97", "n_male": 80, "n_female": 90},
            {"cohort": "cnlsy", "n_male": 60, "n_female": 70},
        ]
    ).to_csv(root / "outputs/tables/sample_counts.csv", index=False)

    params_nlsy79 = [
        {"cohort": "nlsy79", "model_step": "configural", "group": "F", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.05},
        {"cohort": "nlsy79", "model_step": "configural", "group": "M", "lhs": "g", "op": "~1", "rhs": "", "est": 0.25, "se": 0.08},
        {"cohort": "nlsy79", "model_step": "configural", "group": "F", "lhs": "g", "op": "~~", "rhs": "g", "est": 4.0, "se": 0.15},
        {"cohort": "nlsy79", "model_step": "configural", "group": "M", "lhs": "g", "op": "~~", "rhs": "g", "est": 6.0, "se": 0.2},
    ]
    params_nlsy97 = [
        {"cohort": "nlsy97", "model_step": "configural", "group": "F", "lhs": "g", "op": "~1", "rhs": "", "est": 0.05, "se": 0.06},
        {"cohort": "nlsy97", "model_step": "configural", "group": "M", "lhs": "g", "op": "~1", "rhs": "", "est": 0.18, "se": 0.09},
        {"cohort": "nlsy97", "model_step": "configural", "group": "F", "lhs": "g", "op": "~~", "rhs": "g", "est": 3.5, "se": 0.2},
        {"cohort": "nlsy97", "model_step": "configural", "group": "M", "lhs": "g", "op": "~~", "rhs": "g", "est": 5.2, "se": 0.21},
    ]
    params_cnlsy = [
        {"cohort": "cnlsy", "model_step": "configural", "group": "F", "lhs": "g_cnlsy", "op": "~1", "rhs": "", "est": 0.00, "se": 0.04},
        {"cohort": "cnlsy", "model_step": "configural", "group": "M", "lhs": "g_cnlsy", "op": "~1", "rhs": "", "est": 0.20, "se": 0.07},
        {"cohort": "cnlsy", "model_step": "configural", "group": "F", "lhs": "g_cnlsy", "op": "~~", "rhs": "g_cnlsy", "est": 7.2, "se": 0.2},
        {"cohort": "cnlsy", "model_step": "configural", "group": "M", "lhs": "g_cnlsy", "op": "~~", "rhs": "g_cnlsy", "est": 10.1, "se": 0.22},
    ]
    for cohort, rows in (("nlsy79", params_nlsy79), ("nlsy97", params_nlsy97), ("cnlsy", params_cnlsy)):
        _write_params(root / f"outputs/model_fits/{cohort}/params.csv", rows)

    factor_rows = []
    for cohort in ("nlsy79", "nlsy97"):
        for factor in ("Speed", "Math", "Verbal", "Tech", "g"):
            factor_rows.append({"cohort": cohort, "group": "F", "factor": factor, "mean": 0.0 if factor == "g" else 1.0, "var": 1.0, "sd": 1.0})
            factor_rows.append({"cohort": cohort, "group": "M", "factor": factor, "mean": 0.2 if factor == "g" else 1.2, "var": 1.5, "sd": 1.22})
    pd.DataFrame(factor_rows).to_csv(root / "outputs/model_fits/nlsy79/latent_summary.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(root / "outputs/model_fits/nlsy97/latent_summary.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(root / "outputs/model_fits/cnlsy/latent_summary.csv", index=False)

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--all"],
        check=True,
        cwd=repo_root,
    )

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio.csv")
    group_factor = pd.read_csv(root / "outputs/tables/group_factor_diffs.csv")
    exclusions = pd.read_csv(root / "outputs/tables/confirmatory_exclusions.csv")
    tiers = pd.read_csv(root / "outputs/tables/analysis_tiers.csv")
    assert set(["cohort", "d_g", "IQ_diff", "SE", "ci_low", "ci_high"]).issubset(g_mean.columns)
    assert set(["cohort", "VR_g", "SE_logVR", "ci_low", "ci_high"]).issubset(g_vr.columns)
    assert set(["cohort", "factor", "mean_diff_iq", "VR"]).issubset(group_factor.columns)
    assert set(g_mean["cohort"]) == {"nlsy79", "nlsy97", "cnlsy"}
    assert {"nlsy79", "nlsy97"}.issubset(set(group_factor["cohort"]))
    assert set(exclusions["cohort"]) == {"nlsy79", "nlsy97", "cnlsy"}
    assert exclusions["blocked_confirmatory"].astype(bool).sum() == 0
    assert set(tiers["cohort"]) == {"nlsy79", "nlsy97", "cnlsy"}
    assert set(tiers["analysis_tier"]) == {"confirmatory"}

    assert (root / "outputs/figures/g_mean_diff_forestplot.png").exists()
    assert (root / "outputs/figures/vr_forestplot.png").exists()
    assert (root / "outputs/figures/group_factor_gaps.png").exists()
    assert (root / "outputs/figures/group_factor_vr.png").exists()


def test_script_09_uses_partial_refit_outputs_when_flagged(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
invariance:
  steps: [configural, metric, scalar, strict]
""",
    )
    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"cohort": "nlsy97", "n_male": 100, "n_female": 120}]).to_csv(
        root / "outputs/tables/sample_counts.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "cohort": "nlsy97",
                "gatekeeping_enabled": True,
                "confirmatory_d_g_eligible": True,
                "confirmatory_vr_g_eligible": True,
                "reason_d_g": "",
                "reason_vr_g": "",
                "partial_refit_used": True,
                "partial_refit_dir": str(root / "outputs/model_fits/nlsy97/partial_scalar_refit"),
            }
        ]
    ).to_csv(root / "outputs/tables/invariance_confirmatory_eligibility.csv", index=False)

    baseline_params = [
        {"cohort": "nlsy97", "model_step": "scalar", "group": "female", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.05},
        {"cohort": "nlsy97", "model_step": "scalar", "group": "male", "lhs": "g", "op": "~1", "rhs": "", "est": 0.1, "se": 0.05},
        {"cohort": "nlsy97", "model_step": "metric", "group": "female", "lhs": "g", "op": "~~", "rhs": "g", "est": 1.0, "se": 0.1},
        {"cohort": "nlsy97", "model_step": "metric", "group": "male", "lhs": "g", "op": "~~", "rhs": "g", "est": 1.21, "se": 0.1},
    ]
    partial_params = [
        {"cohort": "nlsy97", "model_step": "scalar", "group": "female", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.05},
        {"cohort": "nlsy97", "model_step": "scalar", "group": "male", "lhs": "g", "op": "~1", "rhs": "", "est": 0.4, "se": 0.05},
        {"cohort": "nlsy97", "model_step": "metric", "group": "female", "lhs": "g", "op": "~~", "rhs": "g", "est": 1.0, "se": 0.1},
        {"cohort": "nlsy97", "model_step": "metric", "group": "male", "lhs": "g", "op": "~~", "rhs": "g", "est": 1.21, "se": 0.1},
    ]
    _write_params(root / "outputs/model_fits/nlsy97/params.csv", baseline_params)
    _write_params(root / "outputs/model_fits/nlsy97/partial_scalar_refit/params.csv", partial_params)

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy97"],
        check=True,
        cwd=repo_root,
    )

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff.csv")
    assert len(g_mean) == 1
    assert float(g_mean.iloc[0]["d_g"]) == 0.4


def test_script_09_handles_unknown_group_labels_with_missing_values(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
invariance:
  steps: [configural, metric, scalar, strict]
""",
    )

    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"cohort": "nlsy79", "n_male": 100, "n_female": 120}]
    ).to_csv(root / "outputs/tables/sample_counts.csv", index=False)

    _write_params(
        root / "outputs/model_fits/nlsy79/params.csv",
        [
            {
                "cohort": "nlsy79",
                "model_step": "configural",
                "group": "A",
                "lhs": "g",
                "op": "~1",
                "rhs": "",
                "est": 0.20,
                "se": 0.10,
            },
            {
                "cohort": "nlsy79",
                "model_step": "configural",
                "group": "B",
                "lhs": "g",
                "op": "~1",
                "rhs": "",
                "est": "",
                "se": "",
            },
            {
                "cohort": "nlsy79",
                "model_step": "configural",
                "group": "A",
                "lhs": "g",
                "op": "~~",
                "rhs": "g",
                "est": 4.0,
                "se": 0.12,
            },
            {
                "cohort": "nlsy79",
                "model_step": "configural",
                "group": "B",
                "lhs": "g",
                "op": "~~",
                "rhs": "g",
                "est": 5.0,
                "se": 0.18,
            },
        ],
    )

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy79"],
        check=True,
        cwd=repo_root,
    )

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio.csv")
    assert set(["cohort", "d_g", "IQ_diff", "SE", "ci_low", "ci_high"]).issubset(g_mean.columns)
    assert set(["cohort", "VR_g", "SE_logVR", "ci_low", "ci_high"]).issubset(g_vr.columns)
    assert g_mean.empty
    assert len(g_vr) == 1
    assert set(g_vr["cohort"]) == {"nlsy79"}
    assert (root / "outputs/figures/vr_forestplot.png").exists()


def test_script_09_skips_forestplot_when_all_ci_rows_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
invariance:
  steps: [configural, metric, scalar, strict]
""",
    )

    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"cohort": "nlsy79", "n_male": 10, "n_female": 12},
            {"cohort": "nlsy97", "n_male": 8, "n_female": 10},
            {"cohort": "cnlsy", "n_male": 4, "n_female": 6},
        ]
    ).to_csv(root / "outputs/tables/sample_counts.csv", index=False)

    params_nlsy79 = [
        {"cohort": "nlsy79", "model_step": "configural", "group": "F", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.05},
        {"cohort": "nlsy79", "model_step": "configural", "group": "M", "lhs": "g", "op": "~1", "rhs": "", "est": 0.20, "se": ""},
        {"cohort": "nlsy79", "model_step": "configural", "group": "F", "lhs": "g", "op": "~~", "rhs": "g", "est": 4.0, "se": 0.15},
        {"cohort": "nlsy79", "model_step": "configural", "group": "M", "lhs": "g", "op": "~~", "rhs": "g", "est": 7.0, "se": 0.2},
    ]
    params_nlsy97 = [
        {"cohort": "nlsy97", "model_step": "configural", "group": "F", "lhs": "g", "op": "~1", "rhs": "", "est": 0.05, "se": 0.06},
        {"cohort": "nlsy97", "model_step": "configural", "group": "M", "lhs": "g", "op": "~1", "rhs": "", "est": 0.18, "se": ""},
        {"cohort": "nlsy97", "model_step": "configural", "group": "F", "lhs": "g", "op": "~~", "rhs": "g", "est": 3.5, "se": 0.2},
        {"cohort": "nlsy97", "model_step": "configural", "group": "M", "lhs": "g", "op": "~~", "rhs": "g", "est": 5.2, "se": 0.21},
    ]
    params_cnlsy = [
        {"cohort": "cnlsy", "model_step": "configural", "group": "F", "lhs": "g_cnlsy", "op": "~1", "rhs": "", "est": 0.00, "se": 0.04},
        {"cohort": "cnlsy", "model_step": "configural", "group": "M", "lhs": "g_cnlsy", "op": "~1", "rhs": "", "est": 0.20, "se": ""},
        {"cohort": "cnlsy", "model_step": "configural", "group": "F", "lhs": "g_cnlsy", "op": "~~", "rhs": "g_cnlsy", "est": 7.2, "se": 0.2},
        {"cohort": "cnlsy", "model_step": "configural", "group": "M", "lhs": "g_cnlsy", "op": "~~", "rhs": "g_cnlsy", "est": 10.1, "se": 0.22},
    ]
    for cohort, rows in (("nlsy79", params_nlsy79), ("nlsy97", params_nlsy97), ("cnlsy", params_cnlsy)):
        _write_params(root / f"outputs/model_fits/{cohort}/params.csv", rows)

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--all"],
        check=True,
        cwd=repo_root,
    )

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff.csv")
    assert len(g_mean) == 3
    assert not (root / "outputs/figures/g_mean_diff_forestplot.png").exists()
    assert (root / "outputs/figures/vr_forestplot.png").exists()


def test_script_09_handles_missing_hierarchical_factor_config(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors: null
invariance:
  steps: [configural, metric, scalar, strict]
""",
    )

    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"cohort": "nlsy79", "n_male": 100, "n_female": 120}]).to_csv(
        root / "outputs/tables/sample_counts.csv", index=False
    )
    _write_params(
        root / "outputs/model_fits/nlsy79/params.csv",
        [
            {
                "cohort": "nlsy79",
                "model_step": "configural",
                "group": "F",
                "lhs": "g",
                "op": "~1",
                "rhs": "",
                "est": 0.0,
                "se": 0.05,
            },
            {
                "cohort": "nlsy79",
                "model_step": "configural",
                "group": "M",
                "lhs": "g",
                "op": "~1",
                "rhs": "",
                "est": 0.20,
                "se": 0.08,
            },
            {
                "cohort": "nlsy79",
                "model_step": "configural",
                "group": "F",
                "lhs": "g",
                "op": "~~",
                "rhs": "g",
                "est": 4.0,
                "se": 0.15,
            },
            {
                "cohort": "nlsy79",
                "model_step": "configural",
                "group": "M",
                "lhs": "g",
                "op": "~~",
                "rhs": "g",
                "est": 6.0,
                "se": 0.2,
            },
        ],
    )

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy79"],
        check=True,
        cwd=repo_root,
    )

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff.csv")
    assert list(g_mean["cohort"]) == ["nlsy79"]


def test_script_09_handles_missing_sem_outputs_with_empty_tables(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
invariance:
  steps: [configural, metric, scalar, strict]
""",
    )
    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"cohort": "nlsy79", "n_male": 100, "n_female": 120}]).to_csv(
        root / "outputs/tables/sample_counts.csv", index=False
    )

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy79"],
        check=True,
        cwd=repo_root,
    )

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio.csv")
    group_factor = pd.read_csv(root / "outputs/tables/group_factor_diffs.csv")
    assert g_mean.empty
    assert g_vr.empty
    assert group_factor.empty


def test_script_09_handles_integer_group_labels_in_params(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
invariance:
  steps: [configural, metric, scalar, strict]
""",
    )
    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"cohort": "nlsy79", "n_male": 100, "n_female": 120}]).to_csv(
        root / "outputs/tables/sample_counts.csv", index=False
    )

    rows = []
    for step in ("configural", "metric", "scalar", "strict"):
        rows.extend(
            [
                {"cohort": "nlsy79", "model_step": step, "group": 2, "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.05},
                {"cohort": "nlsy79", "model_step": step, "group": 1, "lhs": "g", "op": "~1", "rhs": "", "est": 0.2, "se": 0.08},
                {"cohort": "nlsy79", "model_step": step, "group": 2, "lhs": "g", "op": "~~", "rhs": "g", "est": 4.0, "se": 0.15},
                {"cohort": "nlsy79", "model_step": step, "group": 1, "lhs": "g", "op": "~~", "rhs": "g", "est": 6.0, "se": 0.2},
            ]
        )
    _write_params(root / "outputs/model_fits/nlsy79/params.csv", rows)

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy79"],
        check=True,
        cwd=repo_root,
    )

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio.csv")
    assert len(g_mean) == 1
    assert len(g_vr) == 1


def test_script_09_uses_scalar_for_means_and_metric_for_variance(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
invariance:
  steps: [configural, metric, scalar, strict]
""",
    )
    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"cohort": "nlsy79", "n_male": 100, "n_female": 120}]).to_csv(
        root / "outputs/tables/sample_counts.csv", index=False
    )
    _write_params(
        root / "outputs/model_fits/nlsy79/params.csv",
        [
            {"cohort": "nlsy79", "model_step": "metric", "group": "F", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.05},
            {"cohort": "nlsy79", "model_step": "metric", "group": "M", "lhs": "g", "op": "~1", "rhs": "", "est": 0.50, "se": 0.08},
            {"cohort": "nlsy79", "model_step": "scalar", "group": "F", "lhs": "g", "op": "~1", "rhs": "", "est": 0.10, "se": 0.06},
            {"cohort": "nlsy79", "model_step": "scalar", "group": "M", "lhs": "g", "op": "~1", "rhs": "", "est": 0.30, "se": 0.09},
            {"cohort": "nlsy79", "model_step": "metric", "group": "F", "lhs": "g", "op": "~~", "rhs": "g", "est": 4.0, "se": 0.15},
            {"cohort": "nlsy79", "model_step": "metric", "group": "M", "lhs": "g", "op": "~~", "rhs": "g", "est": 9.0, "se": 0.2},
            {"cohort": "nlsy79", "model_step": "scalar", "group": "F", "lhs": "g", "op": "~~", "rhs": "g", "est": 5.0, "se": 0.15},
            {"cohort": "nlsy79", "model_step": "scalar", "group": "M", "lhs": "g", "op": "~~", "rhs": "g", "est": 10.0, "se": 0.2},
        ],
    )

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy79"],
        check=True,
        cwd=repo_root,
    )

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio.csv")
    assert len(g_mean) == 1
    assert len(g_vr) == 1
    assert abs(float(g_mean.loc[0, "d_g"]) - 0.2) < 1e-9
    assert abs(float(g_vr.loc[0, "VR_g"]) - 2.25) < 1e-9


def test_script_09_excludes_fail_warning_policy_cohorts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
invariance:
  steps: [configural, metric, scalar, strict]
""",
    )
    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"cohort": "nlsy79", "n_male": 100, "n_female": 120},
            {"cohort": "nlsy97", "n_male": 80, "n_female": 90},
        ]
    ).to_csv(root / "outputs/tables/sample_counts.csv", index=False)
    pd.DataFrame(
        [
            {"cohort": "nlsy79", "status": "ok", "warning_policy_status": "clean"},
            {"cohort": "nlsy97", "status": "ok", "warning_policy_status": "fail"},
        ]
    ).to_csv(root / "outputs/tables/sem_run_status.csv", index=False)

    _write_params(
        root / "outputs/model_fits/nlsy79/params.csv",
        [
            {"cohort": "nlsy79", "model_step": "configural", "group": "F", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.05},
            {"cohort": "nlsy79", "model_step": "configural", "group": "M", "lhs": "g", "op": "~1", "rhs": "", "est": 0.2, "se": 0.08},
            {"cohort": "nlsy79", "model_step": "configural", "group": "F", "lhs": "g", "op": "~~", "rhs": "g", "est": 4.0, "se": 0.15},
            {"cohort": "nlsy79", "model_step": "configural", "group": "M", "lhs": "g", "op": "~~", "rhs": "g", "est": 6.0, "se": 0.2},
        ],
    )
    _write_params(
        root / "outputs/model_fits/nlsy97/params.csv",
        [
            {"cohort": "nlsy97", "model_step": "configural", "group": "F", "lhs": "g", "op": "~1", "rhs": "", "est": 0.05, "se": 0.06},
            {"cohort": "nlsy97", "model_step": "configural", "group": "M", "lhs": "g", "op": "~1", "rhs": "", "est": 0.18, "se": 0.09},
            {"cohort": "nlsy97", "model_step": "configural", "group": "F", "lhs": "g", "op": "~~", "rhs": "g", "est": 3.5, "se": 0.2},
            {"cohort": "nlsy97", "model_step": "configural", "group": "M", "lhs": "g", "op": "~~", "rhs": "g", "est": 5.2, "se": 0.21},
        ],
    )

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--all"],
        check=True,
        cwd=repo_root,
    )

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio.csv")
    exclusions = pd.read_csv(root / "outputs/tables/confirmatory_exclusions.csv")
    tiers = pd.read_csv(root / "outputs/tables/analysis_tiers.csv")
    assert set(g_mean["cohort"]) == {"nlsy79"}
    assert set(g_vr["cohort"]) == {"nlsy79"}
    row = exclusions.loc[exclusions["cohort"] == "nlsy97"].iloc[0]
    assert bool(row["blocked_confirmatory"]) is True
    assert "warning_policy_status=fail" in str(row["reason"])
    nlsy97_tiers = tiers.loc[tiers["cohort"] == "nlsy97"]
    assert set(nlsy97_tiers["analysis_tier"]) == {"non_inferential"}


def test_script_09_applies_estimand_specific_invariance_and_information_gates(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
invariance:
  steps: [configural, metric, scalar]
""",
    )

    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy97",
                "n_male": 80,
                "n_female": 90,
                "information_adequacy_status": "ok",
                "information_adequacy_reason": "",
            },
            {
                "cohort": "cnlsy",
                "n_male": 10,
                "n_female": 12,
                "information_adequacy_status": "low_information",
                "information_adequacy_reason": "n_total<100",
            },
        ]
    ).to_csv(root / "outputs/tables/sample_counts.csv", index=False)
    pd.DataFrame(
        [
            {"cohort": "nlsy97", "status": "ok", "warning_policy_status": "caution"},
            {"cohort": "cnlsy", "status": "ok", "warning_policy_status": "caution"},
        ]
    ).to_csv(root / "outputs/tables/sem_run_status.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy97",
                "confirmatory_d_g_eligible": False,
                "confirmatory_vr_g_eligible": True,
                "reason_d_g": "scalar_gate:failed_delta_cfi",
                "reason_vr_g": "",
            },
            {
                "cohort": "cnlsy",
                "confirmatory_d_g_eligible": True,
                "confirmatory_vr_g_eligible": True,
                "reason_d_g": "",
                "reason_vr_g": "",
            },
        ]
    ).to_csv(root / "outputs/tables/invariance_confirmatory_eligibility.csv", index=False)

    _write_params(
        root / "outputs/model_fits/nlsy97/params.csv",
        [
            {"cohort": "nlsy97", "model_step": "metric", "group": "F", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.05},
            {"cohort": "nlsy97", "model_step": "metric", "group": "M", "lhs": "g", "op": "~1", "rhs": "", "est": 0.2, "se": 0.08},
            {"cohort": "nlsy97", "model_step": "metric", "group": "F", "lhs": "g", "op": "~~", "rhs": "g", "est": 4.0, "se": 0.15},
            {"cohort": "nlsy97", "model_step": "metric", "group": "M", "lhs": "g", "op": "~~", "rhs": "g", "est": 6.0, "se": 0.2},
        ],
    )
    _write_params(
        root / "outputs/model_fits/cnlsy/params.csv",
        [
            {"cohort": "cnlsy", "model_step": "metric", "group": "F", "lhs": "g_cnlsy", "op": "~1", "rhs": "", "est": 0.0, "se": 0.2},
            {"cohort": "cnlsy", "model_step": "metric", "group": "M", "lhs": "g_cnlsy", "op": "~1", "rhs": "", "est": 0.1, "se": 0.3},
            {"cohort": "cnlsy", "model_step": "metric", "group": "F", "lhs": "g_cnlsy", "op": "~~", "rhs": "g_cnlsy", "est": 2.0, "se": 0.4},
            {"cohort": "cnlsy", "model_step": "metric", "group": "M", "lhs": "g_cnlsy", "op": "~~", "rhs": "g_cnlsy", "est": 3.0, "se": 0.5},
        ],
    )

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--all"],
        check=True,
        cwd=repo_root,
    )

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio.csv")
    exclusions = pd.read_csv(root / "outputs/tables/confirmatory_exclusions.csv")
    tiers = pd.read_csv(root / "outputs/tables/analysis_tiers.csv")

    # nlsy97 d_g blocked by invariance scalar gate, vr_g still allowed.
    assert set(g_mean["cohort"]) == set()
    assert set(g_vr["cohort"]) == {"nlsy97"}
    nlsy97_row = exclusions.loc[exclusions["cohort"] == "nlsy97"].iloc[0]
    assert bool(nlsy97_row["blocked_confirmatory_d_g"]) is True
    assert bool(nlsy97_row["blocked_confirmatory_vr_g"]) is False
    assert "invariance:scalar_gate:failed_delta_cfi" in str(nlsy97_row["reason_d_g"])
    nlsy97_tier_d = tiers.loc[(tiers["cohort"] == "nlsy97") & (tiers["estimand"] == "d_g")].iloc[0]
    nlsy97_tier_vr = tiers.loc[(tiers["cohort"] == "nlsy97") & (tiers["estimand"] == "vr_g")].iloc[0]
    assert str(nlsy97_tier_d["analysis_tier"]) == "exploratory_sensitivity"
    assert str(nlsy97_tier_vr["analysis_tier"]) == "confirmatory"

    # cnlsy blocked by information adequacy gate.
    cnlsy_row = exclusions.loc[exclusions["cohort"] == "cnlsy"].iloc[0]
    assert bool(cnlsy_row["blocked_confirmatory"]) is True
    assert "information_adequacy:n_total<100" in str(cnlsy_row["reason"])
    cnlsy_tiers = tiers.loc[tiers["cohort"] == "cnlsy"]
    assert set(cnlsy_tiers["analysis_tier"]) == {"exploratory_low_information"}


def test_script_09_prefers_invariance_alias_fields_for_invariance_gates(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
invariance:
  steps: [configural, metric, scalar]
""",
    )

    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy97",
                "n_male": 80,
                "n_female": 90,
            },
        ]
    ).to_csv(root / "outputs/tables/sample_counts.csv", index=False)
    pd.DataFrame([{"cohort": "nlsy97", "status": "ok", "warning_policy_status": "caution"}]).to_csv(
        root / "outputs/tables/sem_run_status.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "cohort": "nlsy97",
                "invariance_ok_for_d": False,
                "invariance_ok_for_vr": True,
                "reason_d_g": "alias_gate:failed_delta_cfi",
                "reason_vr_g": "",
            },
        ]
    ).to_csv(root / "outputs/tables/invariance_confirmatory_eligibility.csv", index=False)

    _write_params(
        root / "outputs/model_fits/nlsy97/params.csv",
        [
            {"cohort": "nlsy97", "model_step": "metric", "group": "F", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.05},
            {"cohort": "nlsy97", "model_step": "metric", "group": "M", "lhs": "g", "op": "~1", "rhs": "", "est": 0.2, "se": 0.08},
            {"cohort": "nlsy97", "model_step": "metric", "group": "F", "lhs": "g", "op": "~~", "rhs": "g", "est": 4.0, "se": 0.15},
            {"cohort": "nlsy97", "model_step": "metric", "group": "M", "lhs": "g", "op": "~~", "rhs": "g", "est": 6.0, "se": 0.2},
        ],
    )

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy97"],
        check=True,
        cwd=repo_root,
    )

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio.csv")
    exclusions = pd.read_csv(root / "outputs/tables/confirmatory_exclusions.csv")
    tiers = pd.read_csv(root / "outputs/tables/analysis_tiers.csv")

    assert set(g_mean["cohort"]) == set()
    assert set(g_vr["cohort"]) == {"nlsy97"}
    row = exclusions.loc[exclusions["cohort"] == "nlsy97"].iloc[0]
    assert bool(row["blocked_confirmatory_d_g"]) is True
    assert bool(row["blocked_confirmatory_vr_g"]) is False
    assert "alias_gate:failed_delta_cfi" in str(row["reason_d_g"])
    tier_d = tiers.loc[(tiers["cohort"] == "nlsy97") & (tiers["estimand"] == "d_g")].iloc[0]
    tier_vr = tiers.loc[(tiers["cohort"] == "nlsy97") & (tiers["estimand"] == "vr_g")].iloc[0]
    assert str(tier_d["analysis_tier"]) == "exploratory_sensitivity"
    assert str(tier_vr["analysis_tier"]) == "confirmatory"


def test_script_09_falls_back_to_confirmatory_fields_when_alias_fields_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
invariance:
  steps: [configural, metric, scalar]
""",
    )

    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "cohort": "nlsy97",
                "n_male": 80,
                "n_female": 90,
            },
        ]
    ).to_csv(root / "outputs/tables/sample_counts.csv", index=False)
    pd.DataFrame([{"cohort": "nlsy97", "status": "ok", "warning_policy_status": "caution"}]).to_csv(
        root / "outputs/tables/sem_run_status.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "cohort": "nlsy97",
                "confirmatory_d_g_eligible": False,
                "confirmatory_vr_g_eligible": True,
                "reason_d_g": "legacy_gate:failed_delta_cfi",
                "reason_vr_g": "",
            },
        ]
    ).to_csv(root / "outputs/tables/invariance_confirmatory_eligibility.csv", index=False)

    _write_params(
        root / "outputs/model_fits/nlsy97/params.csv",
        [
            {"cohort": "nlsy97", "model_step": "metric", "group": "F", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.05},
            {"cohort": "nlsy97", "model_step": "metric", "group": "M", "lhs": "g", "op": "~1", "rhs": "", "est": 0.2, "se": 0.08},
            {"cohort": "nlsy97", "model_step": "metric", "group": "F", "lhs": "g", "op": "~~", "rhs": "g", "est": 4.0, "se": 0.15},
            {"cohort": "nlsy97", "model_step": "metric", "group": "M", "lhs": "g", "op": "~~", "rhs": "g", "est": 6.0, "se": 0.2},
        ],
    )

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy97"],
        check=True,
        cwd=repo_root,
    )

    exclusions = pd.read_csv(root / "outputs/tables/confirmatory_exclusions.csv")
    assert bool(exclusions.loc[exclusions["cohort"] == "nlsy97", "blocked_confirmatory_d_g"].iloc[0]) is True
    assert bool(exclusions.loc[exclusions["cohort"] == "nlsy97", "blocked_confirmatory_vr_g"].iloc[0]) is False


def test_script_09_uses_sex_estimand_export_and_preserves_inversion_identity(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "outputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
iq_sd_points: 15
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS]
invariance:
  steps: [configural, metric, scalar, strict]
""",
    )
    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"cohort": "nlsy79", "n_male": 120, "n_female": 130}]).to_csv(
        root / "outputs/tables/sample_counts.csv",
        index=False,
    )
    # Keep params ambiguous so the script must consume sex_group_estimands.csv.
    _write_params(
        root / "outputs/model_fits/nlsy79/params.csv",
        [
            {"cohort": "nlsy79", "model_step": "metric", "group": "A", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.1},
            {"cohort": "nlsy79", "model_step": "metric", "group": "B", "lhs": "g", "op": "~1", "rhs": "", "est": 0.1, "se": 0.1},
            {"cohort": "nlsy79", "model_step": "metric", "group": "A", "lhs": "g", "op": "~~", "rhs": "g", "est": 5.0, "se": 0.2},
            {"cohort": "nlsy79", "model_step": "metric", "group": "B", "lhs": "g", "op": "~~", "rhs": "g", "est": 5.1, "se": 0.2},
        ],
    )

    estimands_path = root / "outputs/model_fits/nlsy79/sex_group_estimands.csv"
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "model_step": "scalar",
                "factor": "g",
                "female_group": "female",
                "male_group": "male",
                "mean_female": 0.0,
                "mean_male": 0.2,
                "var_female": 4.0,
                "var_male": 9.0,
                "se_mean_female": 0.05,
                "se_mean_male": 0.08,
            },
            {
                "cohort": "nlsy79",
                "model_step": "metric",
                "factor": "g",
                "female_group": "female",
                "male_group": "male",
                "mean_female": 0.0,
                "mean_male": 0.2,
                "var_female": 4.0,
                "var_male": 9.0,
                "se_mean_female": 0.05,
                "se_mean_male": 0.08,
            },
        ]
    ).to_csv(estimands_path, index=False)

    script = repo_root / "scripts/09_results_and_figures.py"
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy79"],
        check=True,
        cwd=repo_root,
    )
    vr_one = float(pd.read_csv(root / "outputs/tables/g_variance_ratio.csv").iloc[0]["VR_g"])

    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "model_step": "scalar",
                "factor": "g",
                "female_group": "female",
                "male_group": "male",
                "mean_female": 0.2,
                "mean_male": 0.0,
                "var_female": 9.0,
                "var_male": 4.0,
                "se_mean_female": 0.05,
                "se_mean_male": 0.08,
            },
            {
                "cohort": "nlsy79",
                "model_step": "metric",
                "factor": "g",
                "female_group": "female",
                "male_group": "male",
                "mean_female": 0.2,
                "mean_male": 0.0,
                "var_female": 9.0,
                "var_male": 4.0,
                "se_mean_female": 0.05,
                "se_mean_male": 0.08,
            },
        ]
    ).to_csv(estimands_path, index=False)
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--cohort", "nlsy79"],
        check=True,
        cwd=repo_root,
    )
    vr_two = float(pd.read_csv(root / "outputs/tables/g_variance_ratio.csv").iloc[0]["VR_g"])

    assert vr_one == pytest.approx(9.0 / 4.0)
    assert vr_two == pytest.approx(4.0 / 9.0)
    assert vr_one * vr_two == pytest.approx(1.0)
