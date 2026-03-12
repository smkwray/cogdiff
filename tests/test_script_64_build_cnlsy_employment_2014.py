from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "64_build_cnlsy_employment_2014.py"
    spec = importlib.util.spec_from_file_location("script64_build_cnlsy_employment_2014", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_cnlsy_employment_2014_computes_row(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", "cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n")

    rows = []
    for i in range(80):
        g = (i - 40) / 10.0
        age = 14.5 + (i % 5)
        noise = (((i * 5) % 9) - 4) / 10.0
        employed = 1 if (0.8 * g + 0.2 * age + noise) > 2.8 else 0
        rows.append(
            {
                "PPVT": g + 0.2,
                "PIAT_RR": g - 0.1,
                "PIAT_RC": g + 0.1,
                "employment_2014": employed,
                "csage": age,
            }
        )
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(processed / "cnlsy_cfa_resid.csv", index=False)

    out = module.run_cnlsy_employment_2014(root=root)
    assert len(out) == 1
    assert out.loc[0, "status"] == "computed"
    assert float(out.loc[0, "odds_ratio_g"]) > 1.0
    assert (root / "outputs" / "tables" / "cnlsy_employment_2014.csv").exists()


def test_run_cnlsy_employment_2014_handles_missing_columns(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path.resolve()
    _write(root / "config/paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(root / "config/models.yml", "cnlsy_single_factor: [PPVT, PIAT_RR, PIAT_RC]\n")

    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"PPVT": 1.0, "PIAT_RR": 2.0, "PIAT_RC": 3.0}]).to_csv(processed / "cnlsy_cfa_resid.csv", index=False)

    out = module.run_cnlsy_employment_2014(root=root)
    assert out.loc[0, "status"] == "not_feasible"
    assert str(out.loc[0, "reason"]).startswith("missing_required_columns:")
