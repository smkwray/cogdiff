from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "54_build_degree_threshold_outcomes.py"
    spec = importlib.util.spec_from_file_location("script54_build_degree_threshold_outcomes", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_script_54_computes_degree_threshold_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "hierarchical_factors:\n  speed: [GS, AR]\n  math: [MK]\n  verbal: [WK, PC]\n  technical: [NO]\ncnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n",
    )

    rows: list[dict[str, object]] = []
    for idx in range(180):
        g = float(idx)
        if idx < 40:
            education = 12.0
        elif idx < 70:
            education = 16.0 if idx % 3 != 0 else 12.0
        elif idx < 110:
            education = 16.0
        elif idx < 145:
            education = 18.0 if idx % 4 != 0 else 16.0
        else:
            education = 18.0
        rows.append(
            {
                "GS": g,
                "AR": g + 0.1,
                "WK": g + 0.2,
                "PC": g + 0.3,
                "NO": g + 0.4,
                "MK": g + 0.5,
                "education_years": education,
            }
        )

    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", rows)
    out = module.run_degree_threshold_outcomes(root=root, cohorts=["nlsy79"], min_class_n=20)
    assert set(out["status"]) == {"computed"}
    assert set(out["threshold"]) == {"ba_or_more", "graduate_or_more"}
    assert pd.to_numeric(out["odds_ratio_g"], errors="coerce").gt(1).all()


def test_script_54_marks_insufficient_class_counts(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "hierarchical_factors:\n  speed: [GS]\n  math: [AR]\n  verbal: [WK, PC]\n  technical: [NO]\ncnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n",
    )

    rows = [
        {"GS": i, "AR": i, "WK": i, "PC": i, "NO": i, "education_years": 12.0}
        for i in range(40)
    ]
    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", rows)
    out = module.run_degree_threshold_outcomes(root=root, cohorts=["nlsy79"], min_class_n=5)
    assert set(out["status"]) == {"not_feasible"}
    assert set(out["reason"]) == {"insufficient_outcome_class_counts"}
