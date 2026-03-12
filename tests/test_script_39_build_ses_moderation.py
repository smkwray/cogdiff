from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "39_build_ses_moderation.py"
    spec = importlib.util.spec_from_file_location("script39_build_ses_moderation", path)
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


def test_run_ses_moderation_computes_heterogeneity(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: [WK]\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(root / "config" / "nlsy79.yml", "sample_construct:\n  sex_col: sex\n  subtests: [AR, WK]\n")

    rows: list[dict[str, object]] = []
    for i in range(30):
        ses = float(i + 1)
        # Male score grows faster with SES than female score for a moderation signal.
        rows.append({"sex": 1, "ses_index": ses, "AR": 5.0 + 0.30 * ses, "WK": 5.0 + 0.30 * ses})
        rows.append({"sex": 2, "ses_index": ses, "AR": 5.0 + 0.10 * ses, "WK": 5.0 + 0.10 * ses})
    _write_csv(root / "data" / "processed" / "nlsy79_cfa_resid.csv", rows)

    summary, detail = module.run_ses_moderation(
        root=root,
        cohorts=["nlsy79"],
        summary_output_path=Path("outputs/tables/ses_moderation_summary.csv"),
        detail_output_path=Path("outputs/tables/ses_moderation_group_estimates.csv"),
    )

    assert summary.shape[0] == 1
    s = summary.iloc[0]
    assert s["status"] == "computed"
    assert s["ses_col"] == "ses_index"
    assert int(s["n_bins_used"]) >= 2
    assert float(s["heterogeneity_Q"]) >= 0.0
    assert 0.0 <= float(s["heterogeneity_p_value"]) <= 1.0

    d = detail[detail["status"] == "computed"].copy()
    assert d.shape[0] >= 2
    assert d["ses_bin"].isin({"low", "mid", "high"}).all()
    assert set(d["g_construct"]) == {"g_proxy"}
    assert (d["d_g_proxy"] == d["d_g"]).all()

    assert (root / "outputs" / "tables" / "ses_moderation_summary.csv").exists()
    assert (root / "outputs" / "tables" / "ses_moderation_group_estimates.csv").exists()


def test_run_ses_moderation_handles_missing_ses_column(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()

    _write(root / "config" / "paths.yml", "processed_dir: data/processed\noutputs_dir: outputs\n")
    _write(
        root / "config" / "models.yml",
        "reference_group: female\nhierarchical_factors:\n  speed: [AR]\n  math: []\n  verbal: []\n  technical: []\ncnlsy_single_factor: [PPVT]\n",
    )
    _write(root / "config" / "nlsy79.yml", "sample_construct:\n  sex_col: sex\n  subtests: [AR]\n")
    _write_csv(
        root / "data" / "processed" / "nlsy79_cfa_resid.csv",
        [
            {"sex": 1, "AR": 1.0},
            {"sex": 2, "AR": 0.0},
            {"sex": 1, "AR": 2.0},
            {"sex": 2, "AR": 1.0},
        ],
    )

    summary, detail = module.run_ses_moderation(
        root=root,
        cohorts=["nlsy79"],
        summary_output_path=Path("outputs/tables/ses_moderation_summary.csv"),
        detail_output_path=Path("outputs/tables/ses_moderation_group_estimates.csv"),
    )

    assert summary.shape[0] == 1
    s = summary.iloc[0]
    assert s["status"] == "not_feasible"
    assert s["reason"] == "missing_ses_column"
    assert detail.shape[0] == 1
    assert detail.iloc[0]["status"] == "not_feasible"
    assert detail.iloc[0]["reason"] == "missing_ses_column"
