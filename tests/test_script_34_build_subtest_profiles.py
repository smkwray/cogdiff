from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "34_build_subtest_profiles.py"
    spec = importlib.util.spec_from_file_location("script34_build_subtest_profiles", path)
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


def test_run_subtest_profiles_computes_expected_stats(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(
        root / "config" / "nlsy79.yml",
        "sample_construct:\n  sex_col: sex\n  subtests: [AR, WK]\n",
    )

    _write_csv(
        root / "data" / "processed" / "nlsy79_cfa_resid.csv",
        [
            {"sex": 1, "AR": 1.0, "WK": 2.0},
            {"sex": 1, "AR": 2.0, "WK": 4.0},
            {"sex": 1, "AR": 3.0, "WK": 6.0},
            {"sex": 2, "AR": 0.0, "WK": 1.0},
            {"sex": 2, "AR": 1.0, "WK": 3.0},
            {"sex": 2, "AR": 2.0, "WK": 5.0},
        ],
    )

    out = module.run_subtest_profiles(
        root=root,
        cohorts=["nlsy79"],
        output_path=Path("outputs/tables/subtest_sex_profiles.csv"),
        make_plots=False,
    )

    assert out.shape[0] == 2
    assert set(out["status"]) == {"computed"}
    ar = out[out["subtest"] == "AR"].iloc[0]
    assert int(ar["n_male"]) == 3
    assert int(ar["n_female"]) == 3
    assert float(ar["d_subtest"]) == pytest.approx(1.0)
    assert float(ar["vr_subtest"]) == pytest.approx(1.0)
    assert float(ar["log_vr_subtest"]) == pytest.approx(0.0)

    out_path = root / "outputs" / "tables" / "subtest_sex_profiles.csv"
    assert out_path.exists()
    written = pd.read_csv(out_path)
    assert written.shape[0] == 2


def test_run_subtest_profiles_records_missing_subtest(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: []\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(
        root / "config" / "nlsy79.yml",
        "sample_construct:\n  sex_col: sex\n  subtests: [AR, NOPE]\n",
    )
    _write_csv(
        root / "data" / "processed" / "nlsy79_cfa_resid.csv",
        [
            {"sex": 1, "AR": 1.0},
            {"sex": 1, "AR": 2.0},
            {"sex": 2, "AR": 0.0},
            {"sex": 2, "AR": 1.0},
        ],
    )

    out = module.run_subtest_profiles(
        root=root,
        cohorts=["nlsy79"],
        output_path=Path("outputs/tables/subtest_sex_profiles.csv"),
        make_plots=False,
    )

    missing = out[out["subtest"] == "NOPE"].iloc[0]
    assert missing["status"] == "not_feasible"
    assert missing["reason"] == "missing_subtest_column"
