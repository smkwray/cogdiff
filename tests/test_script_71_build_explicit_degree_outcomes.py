from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "71_build_explicit_degree_outcomes.py"
    spec = importlib.util.spec_from_file_location("script71_build_explicit_degree_outcomes", path)
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


def test_script_71_computes_explicit_degree_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "hierarchical_factors:\n  speed: [GS, AR]\n  math: [MK]\n  verbal: [WK, PC]\n  technical: [NO]\ncnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n",
    )

    rows79: list[dict[str, object]] = []
    for idx in range(180):
        if idx < 50:
            degree = 1 if idx % 7 else 3
        elif idx < 110:
            degree = 3 if idx % 5 else 1
        else:
            degree = 5 if idx % 6 else 3
        rows79.append(
            {
                "GS": idx,
                "AR": idx + 1,
                "WK": idx + 2,
                "PC": idx + 3,
                "NO": idx + 4,
                "MK": idx + 5,
                "CS": idx + 6,
                "AS": idx + 7,
                "MC": idx + 8,
                "EI": idx + 9,
                "highest_degree_ever": degree,
            }
        )
    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", rows79)

    rows97: list[dict[str, object]] = []
    for idx in range(180):
        if idx < 40:
            degree = 3 if idx % 8 else 5
        elif idx < 120:
            degree = 5 if idx % 6 else 3
        else:
            degree = 6 if idx % 7 else 5
        rows97.append(
            {
                "GS": idx,
                "AR": idx + 1,
                "WK": idx + 2,
                "PC": idx + 3,
                "NO": idx + 4,
                "MK": idx + 5,
                "CS": idx + 6,
                "AS": idx + 7,
                "MC": idx + 8,
                "EI": idx + 9,
                "degree_2021": degree,
                "age_2021": 39 + (idx % 3),
            }
        )
    _write_csv(root / "data" / "processed" / "nlsy97_cfa_resid.csv", rows97)

    out = module.run_explicit_degree_outcomes(root=root, cohorts=["nlsy79", "nlsy97"], min_class_n=20)
    assert set(out["status"]) == {"computed"}
    assert set(out["threshold"]) == {"ba_or_more_explicit", "graduate_or_more_explicit"}
    assert set(out["cohort"]) == {"nlsy79", "nlsy97"}
    assert pd.to_numeric(out["odds_ratio_g"], errors="coerce").gt(1).all()
    assert set(out.loc[out["cohort"] == "nlsy79", "degree_col"]) == {"highest_degree_ever"}


def test_script_71_marks_missing_columns(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "hierarchical_factors:\n  speed: [GS]\n  math: [AR]\n  verbal: [WK, PC]\n  technical: [NO]\ncnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n",
    )
    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", [{"GS": 1, "AR": 2, "WK": 3, "PC": 4, "NO": 5, "MK": 6, "CS": 7, "AS": 8, "MC": 9, "EI": 10}])
    out = module.run_explicit_degree_outcomes(root=root, cohorts=["nlsy79"], min_class_n=5)
    assert set(out["status"]) == {"not_feasible"}
    assert set(out["reason"]) == {"missing_required_columns:highest_degree_ever"}
