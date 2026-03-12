from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "46_build_g_sat_act_validity_by_group.py"
    spec = importlib.util.spec_from_file_location("script46_build_g_sat_act_validity_by_group", path)
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


def test_script_46_computes_grouped_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config" / "models.yml", "hierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: []\n  technical: []\n")
    _write(root / "config" / "nlsy97.yml", "sample_construct:\n  sex_col: sex\n")

    rows: list[dict[str, object]] = []
    for i in range(150):
        rows.append(
            {
                "sex": 1 if i % 2 == 0 else 2,
                "race_ethnicity_3cat": "BLACK" if i < 50 else ("HISPANIC" if i < 100 else "NON-BLACK, NON-HISPANIC"),
                "parent_education": float(8 + (i % 9)),
                "AR": float(i),
                "WK": float(i + 2),
                "sat_math_2007_bin": 1 + (i % 6),
                "sat_verbal_2007_bin": 1 + (i % 6),
                "act_2007_bin": 1 + (i % 6),
            }
        )
    _write_csv(root / "data" / "processed" / "nlsy97_cfa_resid.csv", rows)

    race_df, ses_df, race_sex_df = module.run_grouped_validity(root=root, min_group_n=20)
    assert set(race_df["status"]) == {"computed"}
    assert set(ses_df["status"]) == {"computed"}
    assert set(race_sex_df["status"]) == {"computed"}

