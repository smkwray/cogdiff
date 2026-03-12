from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "73_build_nlsy79_job_pay_mismatch.py"
    spec = importlib.util.spec_from_file_location("script73_build_nlsy79_job_pay_mismatch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_nlsy79_job_pay_mismatch_computes_outputs(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    census_text = root / "fixtures/census_codes.txt"
    _write(
        census_text,
        "\n".join(
            [
                "Management analysts                       0800  13-1111",
                "Computer programmers                     1010  15-1021",
                "Computer software engineers              1020  15-1030",
            ]
        ),
    )
    job_zones = root / "fixtures/job_zones.txt"
    _write(
        job_zones,
        "\n".join(
            [
                "O*NET-SOC Code\tTitle\tJob Zone",
                "13-1111.00\tManagement analysts\t4",
                "15-1021.00\tComputer programmers\t3",
                "15-1031.00\tComputer software engineers\t5",
            ]
        ),
    )
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    rows = []
    codes = [80, 101, 102]
    job_zone_effects = [4.0, 3.0, 5.0]
    for i in range(900):
        g = (i - 450) / 90.0
        idx = i % len(codes)
        age = 36 + (i % 8)
        noise = 0.9 if i % 9 == 0 else (-0.9 if i % 7 == 0 else 0.0)
        log_pay = 9.0 + 0.22 * job_zone_effects[idx] + 0.18 * g + 0.01 * age + noise
        rows.append(
            {
                "GS": g + 0.1,
                "AR": g - 0.1,
                "WK": g + 0.2,
                "PC": g - 0.2,
                "NO": g + 0.15,
                "CS": g - 0.15,
                "AS": g + 0.05,
                "MK": g - 0.05,
                "MC": g + 0.03,
                "EI": g - 0.03,
                "occupation_code_2000": codes[idx],
                "age_2000": age,
                "education_years": 12 + idx * 2,
                "annual_earnings": float(np.expm1(log_pay)),
            }
        )
    pd.DataFrame(rows).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    summary, model = module.run_nlsy79_job_pay_mismatch(
        root=root,
        census_text_path=census_text,
        job_zones_source=str(job_zones),
        min_n=100,
        min_class_n=50,
    )

    assert (root / "outputs" / "tables" / "nlsy79_job_pay_mismatch_summary.csv").exists()
    assert (root / "outputs" / "tables" / "nlsy79_job_pay_mismatch_models.csv").exists()
    assert set(summary["status"]) == {"computed"}
    assert set(summary["group"]) == {"overall", "underpaid_for_complexity", "aligned_band", "overpaid_for_complexity"}
    assert set(model["outcome"]) == {"pay_residual_z", "overpaid_for_complexity", "underpaid_for_complexity"}
    assert "computed" in set(model["status"])


def test_run_nlsy79_job_pay_mismatch_handles_missing_columns(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: ['NO', CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC, PIAT_MATH, DIGITSPAN]
""",
    )
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"GS": 1.0, "AR": 2.0}]).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    summary, model = module.run_nlsy79_job_pay_mismatch(root=root, min_n=100)
    assert set(summary["status"]) == {"not_feasible"}
    assert set(model["status"]) == {"not_feasible"}
