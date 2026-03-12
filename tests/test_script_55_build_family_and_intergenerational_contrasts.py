from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "55_build_family_and_intergenerational_contrasts.py"
    spec = importlib.util.spec_from_file_location("script55_build_family_and_intergenerational_contrasts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_script_55_builds_sibling_and_intergen_contrasts(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write_csv(
        root / "outputs" / "tables" / "sibling_fe_g_outcome.csv",
        [
            {"cohort": "nlsy79", "outcome": "education", "status": "computed", "beta_g_within": 2.0, "se_within": 0.4},
            {"cohort": "nlsy97", "outcome": "education", "status": "computed", "beta_g_within": 1.2, "se_within": 0.2},
        ],
    )
    _write_csv(
        root / "outputs" / "tables" / "intergenerational_g_transmission.csv",
        [
            {"model": "bivariate", "status": "computed", "n_pairs": 100, "beta_mother_g": 0.50, "r2": 0.20},
            {"model": "ses_controlled", "status": "computed", "n_pairs": 98, "beta_mother_g": 0.40, "r2": 0.22, "beta_parent_ed": 0.05},
        ],
    )

    sibling, intergen = module.run_family_and_intergenerational_contrasts(root=root)
    assert set(sibling["status"]) == {"computed"}
    assert sibling.loc[0, "outcome"] == "education"
    assert round(float(sibling.loc[0, "diff_b_minus_a"]), 3) == -0.8
    assert set(intergen["status"]) == {"computed"}
    assert round(float(intergen.loc[0, "attenuation_abs"]), 3) == 0.1
    assert round(float(intergen.loc[0, "attenuation_pct"]), 1) == 20.0
