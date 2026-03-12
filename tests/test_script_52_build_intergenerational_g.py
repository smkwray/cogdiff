from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "52_build_intergenerational_g.py"
    spec = importlib.util.spec_from_file_location("script52_build_intergenerational_g", path)
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


def test_script_52_computes_intergenerational_rows_from_panel_link(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(
        root / "config" / "models.yml",
        "hierarchical_factors:\n  speed: [GS, AR]\n  math: [MK]\n  verbal: [WK, PC]\n  technical: [NO, CS, AS, EI]\ncnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n",
    )

    mother_rows: list[dict[str, object]] = []
    child_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, object]] = []
    for idx in range(1, 7):
        mother_id = 100 + idx
        mother_base = float(idx)
        mother_rows.append(
            {
                "person_id": mother_id,
                "GS": mother_base,
                "AR": mother_base + 0.1,
                "WK": mother_base + 0.2,
                "PC": mother_base + 0.3,
                "NO": mother_base + 0.4,
                "CS": mother_base + 0.5,
                "AS": mother_base + 0.6,
                "MK": mother_base + 0.7,
                "EI": mother_base + 0.8,
            }
        )
        child_id = 200 + idx
        parent_ed = 10 + idx
        child_signal = (0.6 * mother_base) + (0.1 * parent_ed)
        child_rows.append(
            {
                "person_id": child_id,
                "PPVT": child_signal,
                "PIAT_RR": child_signal + 0.2,
                "PIAT_RC": child_signal + 0.4,
                "parent_education": parent_ed,
            }
        )
        link_rows.append({"C0000100": child_id, "C0000200": mother_id})

    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", mother_rows)
    _write_csv(root / "data" / "processed" / "cnlsy_cfa_resid.csv", child_rows)
    _write_csv(root / "data" / "interim" / "cnlsy" / "panel_extract.csv", link_rows)

    out = module.run_intergenerational_g(root=root, min_pairs=4)
    assert set(out["status"]) == {"computed"}
    assert set(out["model"]) == {"bivariate", "ses_controlled"}
    controlled = out.loc[out["model"] == "ses_controlled"].iloc[0]
    assert pd.notna(controlled["beta_parent_ed"])
    assert float(controlled["n_pairs"]) == 6


def test_script_52_marks_missing_mpubid_as_not_feasible(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _write(
        root / "config" / "models.yml",
        "hierarchical_factors:\n  speed: [GS]\n  math: [AR]\n  verbal: [WK, PC]\n  technical: [NO]\ncnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n",
    )
    _write_csv(
        root / "data" / "processed" / "nlsy79_cfa_resid.csv",
        [
            {"person_id": 1, "GS": 1, "AR": 1, "WK": 1, "PC": 1, "NO": 1},
            {"person_id": 2, "GS": 2, "AR": 2, "WK": 2, "PC": 2, "NO": 2},
        ],
    )
    _write_csv(
        root / "data" / "processed" / "cnlsy_cfa_resid.csv",
        [
            {"person_id": 3, "PPVT": 1, "PIAT_RR": 1, "PIAT_RC": 1, "parent_education": 12},
            {"person_id": 4, "PPVT": 2, "PIAT_RR": 2, "PIAT_RC": 2, "parent_education": 13},
        ],
    )

    out = module.run_intergenerational_g(root=root, min_pairs=1)
    assert set(out["status"]) == {"not_feasible"}
    assert set(out["reason"]) == {"missing_mother_link_column"}
