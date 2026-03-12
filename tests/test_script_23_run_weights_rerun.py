from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _script_module():
    path = _repo_root() / "scripts" / "23_run_weights_rerun.py"
    spec = importlib.util.spec_from_file_location("stage23_weights_rerun", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_stage23_marks_not_feasible_when_weight_column_missing(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, DIGITSPAN]
""",
    )
    _write(root / "config/robustness.yml", "weights_rerun: {}\n")
    df = pd.DataFrame(
        {
            "sex": ["F", "F", "M", "M"],
            "NO": [1, 2, 3, 4],
            "CS": [1, 2, 3, 4],
            "AR": [1, 2, 3, 4],
            "MK": [1, 2, 3, 4],
            "WK": [1, 2, 3, 4],
            "PC": [1, 2, 3, 4],
            "GS": [1, 2, 3, 4],
            "AS": [1, 2, 3, 4],
            "MC": [1, 2, 3, 4],
            "EI": [1, 2, 3, 4],
        }
    )
    source = root / "data/processed/nlsy79_cfa_resid.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(source, index=False)

    module = _script_module()
    manifest = module.run_weights_rerun(root=root, variant_token="weighted")
    assert manifest["variant_token"] == "weighted"

    mean_path = root / "outputs/tables/g_mean_diff_weighted.csv"
    vr_path = root / "outputs/tables/g_variance_ratio_weighted.csv"
    quality_path = root / "outputs/tables/weights_quality_diagnostics.csv"
    assert mean_path.exists()
    assert vr_path.exists()
    assert quality_path.exists()
    mean_df = pd.read_csv(mean_path)
    vr_df = pd.read_csv(vr_path)
    quality_df = pd.read_csv(quality_path)
    assert mean_df.iloc[0]["status"] == "not_feasible"
    assert vr_df.iloc[0]["status"] == "not_feasible"
    assert "weight_column_unavailable" in str(mean_df.iloc[0]["reason"])
    assert "selection_reason" in quality_df.columns
    assert quality_df.iloc[0]["selection_reason"] in {"no_candidate_columns", "missing_from_dataframe"}


def test_stage23_computes_weighted_rows_when_weight_column_available(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, DIGITSPAN]
""",
    )
    _write(
        root / "config/robustness.yml",
        """
weights_rerun:
  candidate_cols:
    nlsy79: [survey_weight]
""",
    )
    df = pd.DataFrame(
        {
            "sex": ["F", "F", "F", "M", "M", "M"],
            "survey_weight": [1.0, 1.2, 0.9, 1.1, 1.0, 1.3],
            "NO": [1, 2, 2, 3, 4, 4],
            "CS": [1, 2, 2, 3, 4, 5],
            "AR": [1, 2, 2, 3, 4, 4],
            "MK": [1, 2, 2, 3, 4, 5],
            "WK": [1, 2, 2, 3, 4, 4],
            "PC": [1, 2, 2, 3, 4, 5],
            "GS": [1, 2, 2, 3, 4, 4],
            "AS": [1, 2, 2, 3, 4, 5],
            "MC": [1, 2, 2, 3, 4, 4],
            "EI": [1, 2, 2, 3, 4, 5],
        }
    )
    source = root / "data/processed/nlsy79_cfa_resid.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(source, index=False)

    module = _script_module()
    manifest = module.run_weights_rerun(root=root, variant_token="weighted")

    mean_df = pd.read_csv(root / "outputs/tables/g_mean_diff_weighted.csv")
    vr_df = pd.read_csv(root / "outputs/tables/g_variance_ratio_weighted.csv")
    quality_df = pd.read_csv(root / "outputs/tables/weights_quality_diagnostics.csv")
    assert mean_df.iloc[0]["status"] == "computed"
    assert vr_df.iloc[0]["status"] == "computed"
    assert pd.notna(mean_df.iloc[0]["d_g"])
    assert pd.notna(vr_df.iloc[0]["VR_g"])
    assert manifest["artifacts"]["weights_quality_diagnostics"] == "outputs/tables/weights_quality_diagnostics.csv"
    assert "min_positive_share" in manifest
    assert "min_effective_n_total" in manifest
    assert "min_effective_n_by_sex" in manifest
    selected = quality_df.loc[quality_df["selected"] == True].copy()  # noqa: E712
    assert not selected.empty
    assert selected.iloc[0]["weight_col"] == "survey_weight"
    assert selected.iloc[0]["quality_gate_checked"] == True  # noqa: E712
    assert selected.iloc[0]["quality_gate_passed"] == True  # noqa: E712
    assert selected.iloc[0]["quality_gate_reason"] == "ok"


def test_stage23_marks_not_feasible_when_quality_gate_fails(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, DIGITSPAN]
""",
    )
    _write(
        root / "config/robustness.yml",
        """
weights_rerun:
  candidate_cols:
    nlsy79: [survey_weight]
  min_effective_n_total: 100
  min_effective_n_by_sex: 50
  min_positive_share: 0.5
""",
    )
    df = pd.DataFrame(
        {
            "sex": ["F", "F", "F", "M", "M", "M"],
            "survey_weight": [1.0, 1.2, 0.9, 1.1, 1.0, 1.3],
            "NO": [1, 2, 2, 3, 4, 4],
            "CS": [1, 2, 2, 3, 4, 5],
            "AR": [1, 2, 2, 3, 4, 4],
            "MK": [1, 2, 2, 3, 4, 5],
            "WK": [1, 2, 2, 3, 4, 4],
            "PC": [1, 2, 2, 3, 4, 5],
            "GS": [1, 2, 2, 3, 4, 4],
            "AS": [1, 2, 2, 3, 4, 5],
            "MC": [1, 2, 2, 3, 4, 4],
            "EI": [1, 2, 2, 3, 4, 5],
        }
    )
    source = root / "data/processed/nlsy79_cfa_resid.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(source, index=False)

    module = _script_module()
    manifest = module.run_weights_rerun(root=root, variant_token="weighted")

    mean_df = pd.read_csv(root / "outputs/tables/g_mean_diff_weighted.csv")
    vr_df = pd.read_csv(root / "outputs/tables/g_variance_ratio_weighted.csv")
    quality_df = pd.read_csv(root / "outputs/tables/weights_quality_diagnostics.csv")
    assert mean_df.iloc[0]["status"] == "not_feasible"
    assert vr_df.iloc[0]["status"] == "not_feasible"
    assert "weight_quality_gate:" in str(mean_df.iloc[0]["reason"])
    selected = quality_df.loc[quality_df["selected"] == True].copy()  # noqa: E712
    assert not selected.empty
    assert selected.iloc[0]["quality_gate_checked"] == True  # noqa: E712
    assert selected.iloc[0]["quality_gate_passed"] == False  # noqa: E712
    assert selected.iloc[0]["quality_gate_reason"] == "effective_n_total_below_threshold"
    assert manifest["min_effective_n_total"] == 100
