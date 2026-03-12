from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "49_build_cnlsy_nonlinear_age_patterns.py"
    spec = importlib.util.spec_from_file_location("script49_build_cnlsy_nonlinear_age_patterns", path)
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


def test_script_49_computes_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config" / "models.yml", "cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n")
    _write(root / "config" / "cnlsy.yml", "expected_age_range:\n  min: 5\n  max: 12\nsample_construct:\n  id_col: person_id\n  age_col: csage\n  sex_col: sex\n  subtests: [PPVT, PIAT_RR, PIAT_RC]\n")

    rows: list[dict[str, object]] = []
    for age in range(5, 13):
        for person in range(8):
            rows.append(
                {
                    "person_id": f"{age}_{person}",
                    "csage": float(age),
                    "sex": 1 if person % 2 == 0 else 0,
                    "PPVT": float(age + person + (0.5 * age if person % 2 == 0 else -0.25 * age)),
                    "PIAT_RR": float(age + person + 1 + (0.3 * age if person % 2 == 0 else -0.15 * age)),
                    "PIAT_RC": float(age + person + 2 + (0.2 * age if person % 2 == 0 else -0.1 * age)),
                }
            )
    _write_csv(root / "data" / "processed" / "cnlsy_cfa_resid.csv", rows)

    out = module.run_cnlsy_nonlinear_age_patterns(root=root)
    assert (out["status"] == "computed").any()
