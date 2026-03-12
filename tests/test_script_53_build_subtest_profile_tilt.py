from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "53_build_subtest_profile_tilt.py"
    spec = importlib.util.spec_from_file_location("script53_build_subtest_profile_tilt", path)
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


def test_script_53_computes_tilt_outputs_for_two_cohorts(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "hierarchical_factors:\n  speed: [GS, AR]\n  math: [MK]\n  verbal: [WK, PC]\n  technical: [NO]\n",
    )

    for cohort in ("nlsy79", "nlsy97"):
        rows: list[dict[str, object]] = []
        for idx in range(120):
            sex = 1 if idx % 2 == 0 else 2
            verbal_boost = 1.0 if sex == 1 else -1.0
            quant_boost = -0.5 if sex == 1 else 0.5
            rows.append(
                {
                    "sex": sex,
                    "GS": float(idx),
                    "WK": float(idx + verbal_boost + 1.0),
                    "PC": float(idx + verbal_boost + 1.5),
                    "AR": float(idx + quant_boost + 0.5),
                    "MK": float(idx + quant_boost),
                    "NO": float(idx + 0.25),
                    "education_years": float(12 + (0.02 * idx) + (0.8 * verbal_boost) - (0.3 * quant_boost)),
                }
            )
        _write_csv(root / "data" / "processed" / f"{cohort}_cfa_resid.csv", rows)

    out = module.run_subtest_profile_tilt(root=root, cohorts=["nlsy79", "nlsy97"], min_n=30)
    assert set(out["status"]) == {"computed"}
    assert set(out["cohort"]) == {"nlsy79", "nlsy97"}
    assert pd.to_numeric(out["tilt_incremental_r2_education"], errors="coerce").gt(0).all()
    assert pd.to_numeric(out["tilt_g_corr"], errors="coerce").notna().all()
