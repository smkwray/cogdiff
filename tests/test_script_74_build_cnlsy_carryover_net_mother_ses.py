from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "74_build_cnlsy_carryover_net_mother_ses.py"
    spec = importlib.util.spec_from_file_location("script74_build_cnlsy_carryover_net_mother_ses", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_cnlsy_carryover_net_mother_ses_computes_outputs(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", "cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]\n")
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(180):
        g = (i - 90) / 25.0
        mother_ed = 10 + (i % 8)
        age = 14 + (i % 5)
        wage = float(np.expm1(6.0 + 0.22 * g + 0.08 * mother_ed + 0.03 * age))
        family = float(np.expm1(7.0 + 0.10 * g + 0.10 * mother_ed + 0.02 * age))
        degree = 2 if (0.35 * g + 0.18 * mother_ed + 0.05 * age) > 2.9 else 0
        rows.append(
            {
                "PPVT": g + 0.1,
                "PIAT_RR": g - 0.1,
                "PIAT_RC": g + 0.2,
                "PIAT_MATH": g - 0.2,
                "DIGITSPAN": g + 0.05,
                "age_2014": age,
                "mother_education": mother_ed,
                "wage_income_2014": wage,
                "family_income_2014": family,
                "num_current_jobs_2014": float(i % 3),
                "degree_2014": degree,
            }
        )
    pd.DataFrame(rows).to_csv(processed / "cnlsy_cfa_resid.csv", index=False)

    detail, summary = module.run_cnlsy_carryover_net_mother_ses(root=root, min_n_continuous=60, min_class_n_binary=20)

    assert (root / "outputs" / "tables" / "cnlsy_carryover_net_mother_ses.csv").exists()
    assert (root / "outputs" / "tables" / "cnlsy_carryover_net_mother_ses_summary.csv").exists()
    assert "computed" in set(detail["status"])
    assert "computed" in set(summary["status"])
    assert set(summary["outcome"]) == {"log_wage_income_2014", "log_family_income_2014", "num_current_jobs_2014", "degree_any_2014"}


def test_run_cnlsy_carryover_net_mother_ses_handles_missing_columns(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", "cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]\n")
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"PPVT": 1.0, "PIAT_RR": 1.0, "PIAT_RC": 1.0, "PIAT_MATH": 1.0, "DIGITSPAN": 1.0}]).to_csv(processed / "cnlsy_cfa_resid.csv", index=False)

    detail, summary = module.run_cnlsy_carryover_net_mother_ses(root=root)
    assert set(detail["status"]) == {"not_feasible"}
    assert set(summary["status"]) == {"not_feasible"}
