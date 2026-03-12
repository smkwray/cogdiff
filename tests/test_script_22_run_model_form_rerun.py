from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _script_module():
    path = _repo_root() / "scripts" / "22_run_model_form_rerun.py"
    spec = importlib.util.spec_from_file_location("stage22_model_form_rerun", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_stage22_writes_variant_summary_with_r_path(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path.resolve()
    _write(
        root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
invariance:
  steps: [configural, metric, scalar]
cnlsy_single_factor: [PPVT, DIGITSPAN]
""",
    )
    _write(
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  sex_col: sex
""",
    )
    df = pd.DataFrame(
        {
            "sex": ["F", "F", "M", "M"],
            "NO": [1, 2, 2, 3],
            "CS": [1, 1, 2, 3],
            "AR": [2, 2, 3, 3],
            "MK": [2, 3, 3, 4],
            "WK": [1, 2, 2, 2],
            "PC": [2, 2, 3, 3],
            "GS": [1, 1, 2, 2],
            "AS": [2, 2, 3, 3],
            "MC": [1, 2, 2, 3],
            "EI": [1, 1, 2, 2],
        }
    )
    source = root / "data/processed/nlsy79_cfa_resid.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(source, index=False)

    module = _script_module()

    def _fake_run_sem_r_script(*, r_script, request_file, outdir):
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "model_step": "configural", "cfi": 0.95, "rmsea": 0.04, "srmr": 0.03},
                {"cohort": "nlsy79", "model_step": "scalar", "cfi": 0.93, "rmsea": 0.05, "srmr": 0.035},
            ]
        ).to_csv(outdir / "fit_indices.csv", index=False)
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(module, "rscript_path", lambda: "/usr/bin/Rscript")
    monkeypatch.setattr(module, "run_sem_r_script", _fake_run_sem_r_script)

    manifest = module.run_model_form_rerun(
        root=root,
        cohort="nlsy79",
        variant_token="single_factor_alt",
        skip_r=False,
    )

    summary_path = root / "outputs/tables/nlsy79_invariance_summary_single_factor_alt.csv"
    manifest_path = root / "outputs/tables/model_form_rerun_manifest_nlsy79_single_factor_alt.json"
    assert summary_path.exists()
    assert manifest_path.exists()
    assert manifest["used_python_fallback"] is False
    assert manifest["source_path"] == "data/processed/nlsy79_cfa_resid.csv"
    assert manifest["variant_summary_path"] == "outputs/tables/nlsy79_invariance_summary_single_factor_alt.csv"
    assert manifest["model_syntax_path"].startswith("outputs/logs/robustness/model_form_runs/")
    assert manifest["request_path"].startswith("outputs/logs/robustness/model_form_runs/")
    assert manifest["fit_dir"].startswith("outputs/logs/robustness/model_form_runs/")
    assert manifest["model_syntax_path"].endswith("/model.lavaan")
    assert manifest["request_path"].endswith("/request.json")
    assert manifest["fit_dir"].endswith("/fit")
    assert not Path(manifest["model_syntax_path"]).is_absolute()
    assert not Path(manifest["request_path"]).is_absolute()
    assert not Path(manifest["fit_dir"]).is_absolute()
    out = pd.read_csv(summary_path)
    assert set(out["model_step"]) == {"configural", "scalar"}


def test_stage22_python_fallback_path(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    _write(
        root / "config/paths.yml",
        "processed_dir: data/processed\noutputs_dir: outputs\n",
    )
    _write(
        root / "config/models.yml",
        """
hierarchical_factors:
  speed: [NO, CS]
  math: [AR, MK]
  verbal: [WK, PC]
  technical: [GS, AS, MC, EI]
invariance:
  steps: [configural, metric, scalar]
cnlsy_single_factor: [PPVT, DIGITSPAN]
""",
    )
    _write(
        root / "config/nlsy79.yml",
        """
cohort: nlsy79
sample_construct:
  sex_col: sex
""",
    )
    df = pd.DataFrame(
        {
            "sex": ["F", "F", "M", "M"],
            "NO": [1, 2, 2, 3],
            "CS": [1, 1, 2, 3],
            "AR": [2, 2, 3, 3],
            "MK": [2, 3, 3, 4],
            "WK": [1, 2, 2, 2],
            "PC": [2, 2, 3, 3],
            "GS": [1, 1, 2, 2],
            "AS": [2, 2, 3, 3],
            "MC": [1, 2, 2, 3],
            "EI": [1, 1, 2, 2],
        }
    )
    source = root / "data/processed/nlsy79_cfa_resid.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(source, index=False)

    module = _script_module()
    manifest = module.run_model_form_rerun(
        root=root,
        cohort="nlsy79",
        variant_token="bifactor_alt",
        skip_r=True,
    )
    summary_path = root / "outputs/tables/nlsy79_invariance_summary_bifactor_alt.csv"
    assert summary_path.exists()
    assert manifest["used_python_fallback"] is True
    assert manifest["source_path"] == "data/processed/nlsy79_cfa_resid.csv"
    assert manifest["variant_summary_path"] == "outputs/tables/nlsy79_invariance_summary_bifactor_alt.csv"
    assert manifest["model_syntax_path"].startswith("outputs/logs/robustness/model_form_runs/")
    assert manifest["request_path"].startswith("outputs/logs/robustness/model_form_runs/")
    assert manifest["fit_dir"].startswith("outputs/logs/robustness/model_form_runs/")
    assert manifest["model_syntax_path"].endswith("/model.lavaan")
    assert manifest["request_path"].endswith("/request.json")
    assert manifest["fit_dir"].endswith("/fit")
    assert not Path(manifest["model_syntax_path"]).is_absolute()
    assert not Path(manifest["request_path"]).is_absolute()
    assert not Path(manifest["fit_dir"]).is_absolute()
    out = pd.read_csv(summary_path)
    assert not out.empty
