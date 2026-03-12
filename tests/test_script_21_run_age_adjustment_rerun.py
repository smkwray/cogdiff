from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _script_module():
    path = _repo_root() / "scripts" / "21_run_age_adjustment_rerun.py"
    spec = importlib.util.spec_from_file_location("stage21_age_adjustment_rerun", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_stage21_writes_age_adjustment_variant_outputs(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(
        root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: [GS]
  math: [AR]
  verbal: []
  technical: []
""",
    )
    _write(
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  sex_col: sex
  age_resid_col: birth_year
  subtests: [GS, AR]
  standardize_output: false
""",
    )
    df = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "sex": ["F", "F", "F", "M", "M", "M"],
            "birth_year": [1960, 1961, 1962, 1960, 1961, 1962],
            "GS": [20, 22, 25, 19, 21, 24],
            "AR": [18, 19, 23, 17, 20, 22],
        }
    )
    source = root / "data/processed/nlsy79_cfa.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(source, index=False)

    module = _script_module()
    manifest = module.run_age_adjustment_rerun(
        root=root,
        cohort="nlsy79",
        variant_token="cubic_within_sex",
    )

    diagnostics_path = root / "outputs/tables/residualization_diagnostics_nlsy79_cubic_within_sex.csv"
    resid_path = root / "data/processed/nlsy79_cfa_resid_cubic_within_sex.csv"
    manifest_path = root / "outputs/tables/age_adjustment_rerun_manifest_nlsy79_cubic_within_sex.json"
    assert diagnostics_path.exists()
    assert resid_path.exists()
    assert manifest_path.exists()
    assert manifest["variant_token"] == "cubic_within_sex"
    assert manifest["source_path"] == "data/processed/nlsy79_cfa.csv"
    assert manifest["residualized_path"] == "data/processed/nlsy79_cfa_resid_cubic_within_sex.csv"
    assert manifest["diagnostics_path"] == "outputs/tables/residualization_diagnostics_nlsy79_cubic_within_sex.csv"

    diag = pd.read_csv(diagnostics_path)
    assert set(diag["subtest"]) == {"GS", "AR"}
    assert set(diag["age_adjustment"]) == {"cubic"}
    assert set(diag["residualization_mode"]) == {"within_sex"}
    assert set(diag["variant_token"]) == {"cubic_within_sex"}
    assert diag["n_used"].min() > 0


def test_stage21_rejects_unsupported_variant(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", "hierarchical_factors: {}\n")
    _write(root / "config/nlsy79.yml", "cohort: nlsy79\nsample_construct: {sex_col: sex, age_resid_col: age, subtests: [GS]}\n")
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sex": ["F"], "age": [20], "GS": [10]}).to_csv(root / "data/processed/nlsy79_cfa.csv", index=False)

    module = _script_module()
    try:
        module.run_age_adjustment_rerun(root=root, cohort="nlsy79", variant_token="unsupported")
        raise AssertionError("expected ValueError for unsupported variant token")
    except ValueError as exc:
        assert "Unsupported variant token" in str(exc)
