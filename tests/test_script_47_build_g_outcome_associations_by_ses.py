from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "47_build_g_outcome_associations_by_ses.py"
    spec = importlib.util.spec_from_file_location("script47_build_g_outcome_associations_by_ses", path)
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


def test_script_47_computes_ses_rows(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config" / "models.yml", "hierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n")
    _write(root / "config" / "nlsy79.yml", "sample_construct:\n  sex_col: sex\n")

    rows: list[dict[str, object]] = []
    for i in range(120):
        rows.append(
            {
                "sex": 1 if i % 2 == 0 else 2,
                "parent_education": float(8 + (i % 9)),
                "AR": float(i),
                "WK": float(i + 1),
                "annual_earnings": float(10000 + 50 * i),
                "household_income": float(20000 + 100 * i),
                "net_worth": float(5000 + 80 * i),
                "education_years": float(10 + (i % 6)),
            }
        )
    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", rows)

    summary_df, detail_df = module.run_g_outcome_by_ses(root=root, cohorts=["nlsy79"], min_group_n=20)
    assert (summary_df["status"] == "computed").any()
    assert (detail_df["status"] == "computed").any()

