from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "60_build_occupation_code_availability.py"
    spec = importlib.util.spec_from_file_location("script60_build_occupation_code_availability", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_occupation_code_availability_computes_rows(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"occupation_code_2000": [245, 245, 503, None, 95]}).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame(
        {
            "occupation_code_2011": [pd.NA, 1020, pd.NA, pd.NA],
            "occupation_code_2013": [1020, pd.NA, 4210, pd.NA],
            "occupation_code_2015": [pd.NA, 3600, 4210, pd.NA],
            "occupation_code_2017": [1100, pd.NA, pd.NA, 5240],
            "occupation_code_2019": [1020, 3600, 3600, 4210],
            "occupation_code_2021": [1100, 1100, 4210, pd.NA],
        }
    ).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    out = module.run_occupation_code_availability(root=root, cohorts=["nlsy79", "nlsy97"], top_n=2)

    assert set(out["status"]) == {"computed"}
    nlsy79 = out.loc[out["cohort"] == "nlsy79"].iloc[0]
    assert int(nlsy79["n_nonmissing"]) == 4
    assert int(nlsy79["n_unique_codes"]) == 3
    assert "245:2" in str(nlsy79["top_codes"])
    nlsy97 = out.loc[out["cohort"] == "nlsy97"].sort_values("occupation_col").reset_index(drop=True)
    assert list(nlsy97["occupation_col"]) == [
        "occupation_code_2011",
        "occupation_code_2013",
        "occupation_code_2015",
        "occupation_code_2017",
        "occupation_code_2019",
        "occupation_code_2021",
    ]
    assert list(nlsy97["n_nonmissing"].astype(int)) == [1, 2, 2, 2, 4, 3]
    assert (root / "outputs" / "tables" / "occupation_code_availability.csv").exists()


def test_run_occupation_code_availability_handles_missing_column(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [1]}).to_csv(processed / "nlsy79_cfa_resid.csv", index=False)

    out = module.run_occupation_code_availability(root=root, cohorts=["nlsy79"])
    assert out.loc[0, "status"] == "not_feasible"
    assert out.loc[0, "reason"] == "missing_occupation_column"


def test_run_occupation_code_availability_marks_missing_wave_columns(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"occupation_code_2019": [1020, 3600]}).to_csv(processed / "nlsy97_cfa_resid.csv", index=False)

    out = module.run_occupation_code_availability(root=root, cohorts=["nlsy97"])
    reasons = dict(zip(out["occupation_col"], out["reason"], strict=False))
    assert reasons["occupation_code_2011"] == "missing_occupation_column"
    assert reasons["occupation_code_2013"] == "missing_occupation_column"
    assert reasons["occupation_code_2015"] == "missing_occupation_column"
    assert reasons["occupation_code_2017"] == "missing_occupation_column"
    assert reasons["occupation_code_2021"] == "missing_occupation_column"
    assert pd.isna(reasons["occupation_code_2019"])
