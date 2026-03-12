from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "58_audit_upstream_field_candidates.py"
    spec = importlib.util.spec_from_file_location("script58_audit_upstream_field_candidates", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_script_58_flags_candidate_raw_fields_and_current_presence(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  column_map:
    R0001000: current_age
    R0002000: annual_earnings
    R0003000: occupation_code
""",
    )
    _write(root / "data/interim/nlsy79/raw_files/mock.sas", "label R0001000 = \"CURRENT AGE OF R\";\nR0001000 = 'CURRENT_AGE'n\nlabel R0002000 = \"WAGES AND SALARY LAST YEAR\";\nR0002000 = 'WAGES_LAST_YEAR'n\nlabel R0003000 = \"CENSUS OCCUPATION CODE - CURRENT JOB\";\nR0003000 = 'CENSUS_OCCUPATION_CURRENT_JOB'n\n")
    pd.DataFrame(columns=["R0002000"]).to_csv(root / "data/interim/nlsy79/panel_extract.csv", index=False)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["annual_earnings", "current_age"]).to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)

    out = module.run_upstream_field_audit(root=root, cohorts=["nlsy79"])
    assert set(out["category"]) == {"age", "employment", "occupation"}
    age_row = out.loc[out["category"] == "age"].iloc[0]
    assert bool(age_row["in_column_map"]) is True
    assert bool(age_row["in_processed"]) is True
    employment_row = out.loc[out["category"] == "employment"].iloc[0]
    assert bool(employment_row["in_panel_extract"]) is True
    occupation_row = out.loc[out["category"] == "occupation"].iloc[0]
    assert str(occupation_row["mapped_name"]) == "occupation_code"
