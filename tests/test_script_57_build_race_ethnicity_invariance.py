from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "57_build_race_ethnicity_invariance.py"
    spec = importlib.util.spec_from_file_location("script57_build_race_ethnicity_invariance", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_script_57_builds_race_invariance_outputs_with_python_fallback(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\nsem_interim_dir: data/interim/sem\n")
    _write(
        root / "config/models.yml",
        """
reference_group: female
invariance:
  steps: [configural, metric, scalar]
  gatekeeping:
    thresholds:
      metric: {delta_cfi_min: -0.01, delta_rmsea_max: 0.015, delta_srmr_max: 0.03}
      scalar: {delta_cfi_min: -0.01, delta_rmsea_max: 0.015, delta_srmr_max: 0.015}
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write(root / "config/nlsy79.yml", "cohort: nlsy79\nsample_construct: {sex_col: sex}\n")

    rows = []
    for idx, group in enumerate(["BLACK"] * 6 + ["HISPANIC"] * 6 + ["NON-BLACK, NON-HISPANIC"] * 6):
        rows.append(
            {
                "race_ethnicity_3cat": group,
                "NO": 0.1 + idx,
                "CS": 0.2 + idx,
                "AR": 0.3 + idx,
                "MK": 0.4 + idx,
                "WK": 0.5 + idx,
                "PC": 0.6 + idx,
                "GS": 0.7 + idx,
                "AS": 0.8 + idx,
                "MC": 0.9 + idx,
                "EI": 1.0 + idx,
            }
        )
    _write_csv(root / "data/processed/nlsy79_cfa_resid.csv", pd.DataFrame(rows))

    summary, transitions, eligibility = module.run_race_ethnicity_invariance(
        root=root,
        cohorts=["nlsy79"],
        min_group_n=5,
        python_fallback=True,
    )
    assert set(eligibility["status"]) == {"computed"}
    row = eligibility.iloc[0]
    assert row["n_groups"] == 3
    assert bool(row["metric_pass"]) is True
    assert bool(row["race_invariance_ok_for_d"]) is True
    assert set(transitions["transition"]) == {"configural->metric", "metric->scalar"}
    assert set(summary["model_step"]) == {"configural", "metric", "scalar"}


def test_script_57_marks_insufficient_group_counts_not_feasible(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\nsem_interim_dir: data/interim/sem\n")
    _write(
        root / "config/models.yml",
        """
invariance:
  steps: [configural, metric, scalar]
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    _write(root / "config/nlsy79.yml", "cohort: nlsy79\nsample_construct: {sex_col: sex}\n")
    _write_csv(
        root / "data/processed/nlsy79_cfa_resid.csv",
        pd.DataFrame(
            [
                {"race_ethnicity_3cat": "BLACK", "NO": 1, "CS": 1, "AR": 1, "MK": 1, "WK": 1, "PC": 1, "GS": 1, "AS": 1, "MC": 1, "EI": 1},
                {"race_ethnicity_3cat": "HISPANIC", "NO": 2, "CS": 2, "AR": 2, "MK": 2, "WK": 2, "PC": 2, "GS": 2, "AS": 2, "MC": 2, "EI": 2},
                {"race_ethnicity_3cat": "NON-BLACK, NON-HISPANIC", "NO": 3, "CS": 3, "AR": 3, "MK": 3, "WK": 3, "PC": 3, "GS": 3, "AS": 3, "MC": 3, "EI": 3},
            ]
        ),
    )

    _, transitions, eligibility = module.run_race_ethnicity_invariance(
        root=root,
        cohorts=["nlsy79"],
        min_group_n=5,
        python_fallback=True,
    )
    assert eligibility.loc[0, "status"] == "not_feasible"
    assert eligibility.loc[0, "reason"] == "insufficient_group_n"
    assert set(transitions["status"]) == {"not_feasible"}
