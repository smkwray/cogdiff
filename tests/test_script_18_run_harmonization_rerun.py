from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _module():
    path = _repo_root() / "scripts" / "18_run_harmonization_rerun.py"
    spec = importlib.util.spec_from_file_location("stage18_harmonization_rerun", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _minimal_config(root: Path, *, with_harmonization: bool = True) -> None:
    _write(
        root / "config" / "paths.yml",
        "raw_dir: data/raw\n"
        "interim_dir: data/interim\n"
        "processed_dir: data/processed\n"
        "outputs_dir: outputs\n"
        "logs_dir: outputs/logs\n"
        "manifest_file: data/raw/manifest.json\n"
        "sem_interim_dir: data/interim/sem\n"
        "links_interim_dir: data/interim/links\n",
    )
    nlsy97_payload = {"sample_construct": {"id_col": "person_id", "sex_col": "sex"}}
    if with_harmonization:
        nlsy97_payload["sample_construct"]["branch_harmonization"] = {
            "enabled": True,
            "method": "signed_merge",
            "pairs": [],
        }
    _write(root / "config" / "nlsy97.yml", yaml.safe_dump(nlsy97_payload, sort_keys=False))
    _write(root / "config" / "nlsy79.yml", "sample_construct: {id_col: person_id, sex_col: sex}\n")
    _write(root / "config" / "cnlsy.yml", "sample_construct: {id_col: person_id, sex_col: sex}\n")
    _write(root / "config" / "models.yml", "reporting:\n  warning_policy:\n    enabled: true\n    threshold: fail\n")
    _write(root / "scripts" / "sem_fit.R", "cat('ok\\n')\n")
    _write(root / "data/raw/manifest.json", "{}\n")


def test_stage18_exports_variant_tables_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    root = tmp_path.resolve()
    _minimal_config(root, with_harmonization=True)

    def fake_run_stage_sequence(*, root: Path, isolated_root: Path, cohort: str, log_path: Path) -> None:
        tables = isolated_root / "outputs" / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"cohort": "nlsy97", "d_g": 0.11, "SE_d_g": 0.03, "ci_low_d_g": 0.05, "ci_high_d_g": 0.17},
                {"cohort": "nlsy79", "d_g": 0.22, "SE_d_g": 0.04, "ci_low_d_g": 0.14, "ci_high_d_g": 0.30},
            ]
        ).to_csv(tables / "g_mean_diff.csv", index=False)
        pd.DataFrame(
            [
                {"cohort": "nlsy97", "VR_g": 1.10, "SE_logVR": 0.06, "ci_low": 0.98, "ci_high": 1.23},
                {"cohort": "nlsy79", "VR_g": 1.20, "SE_logVR": 0.07, "ci_low": 1.02, "ci_high": 1.41},
            ]
        ).to_csv(tables / "g_variance_ratio.csv", index=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run_stage_sequence", fake_run_stage_sequence)

    manifest = module.run_harmonization_rerun(
        root=root,
        cohort="nlsy97",
        variant_token="zscore_by_branch",
        keep_workspace=False,
    )

    mean_path = root / "outputs" / "tables" / "g_mean_diff_zscore_by_branch.csv"
    vr_path = root / "outputs" / "tables" / "g_variance_ratio_zscore_by_branch.csv"
    manifest_path = root / "outputs" / "tables" / "harmonization_rerun_manifest_nlsy97_zscore_by_branch.json"

    assert mean_path.exists()
    assert vr_path.exists()
    assert manifest_path.exists()
    mean = pd.read_csv(mean_path)
    vr = pd.read_csv(vr_path)
    assert set(mean["cohort"]) == {"nlsy97"}
    assert set(vr["cohort"]) == {"nlsy97"}
    assert float(mean.iloc[0]["d_g"]) == pytest.approx(0.11)
    assert float(vr.iloc[0]["VR_g"]) == pytest.approx(1.10)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["cohort"] == "nlsy97"
    assert payload["variant_token"] == "zscore_by_branch"
    assert manifest["workspace_kept"] is False
    workspace_path = Path(payload["workspace_root"])
    if workspace_path.is_absolute():
        assert not workspace_path.exists()
    else:
        assert not (root / workspace_path).exists()


def test_stage18_requires_enabled_branch_harmonization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    root = tmp_path.resolve()
    _minimal_config(root, with_harmonization=False)

    with pytest.raises(ValueError, match="branch_harmonization"):
        module.run_harmonization_rerun(
            root=root,
            cohort="nlsy97",
            variant_token="zscore_by_branch",
            keep_workspace=True,
        )


def test_stage18_derives_exploratory_rows_when_confirmatory_tables_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root = tmp_path.resolve()
    _minimal_config(root, with_harmonization=True)

    def fake_run_stage_sequence(*, root: Path, isolated_root: Path, cohort: str, log_path: Path) -> None:
        tables = isolated_root / "outputs" / "tables"
        fit_dir = isolated_root / "outputs" / "model_fits" / cohort
        tables.mkdir(parents=True, exist_ok=True)
        fit_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["cohort", "d_g", "SE_d_g", "ci_low_d_g", "ci_high_d_g", "IQ_diff", "SE", "ci_low", "ci_high"]).to_csv(
            tables / "g_mean_diff.csv", index=False
        )
        pd.DataFrame(columns=["cohort", "VR_g", "SE_logVR", "ci_low", "ci_high"]).to_csv(
            tables / "g_variance_ratio.csv", index=False
        )
        pd.DataFrame([{"cohort": cohort, "n_male": 100, "n_female": 100}]).to_csv(
            tables / "sample_counts.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {"cohort": cohort, "model_step": "metric", "group": "male", "lhs": "g", "op": "~1", "rhs": "", "est": 0.20, "se": 0.05},
                {"cohort": cohort, "model_step": "metric", "group": "female", "lhs": "g", "op": "~1", "rhs": "", "est": 0.00, "se": 0.05},
                {"cohort": cohort, "model_step": "metric", "group": "male", "lhs": "g", "op": "~~", "rhs": "g", "est": 1.10, "se": 0.10},
                {"cohort": cohort, "model_step": "metric", "group": "female", "lhs": "g", "op": "~~", "rhs": "g", "est": 1.00, "se": 0.10},
            ]
        ).to_csv(fit_dir / "params.csv", index=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run_stage_sequence", fake_run_stage_sequence)

    module.run_harmonization_rerun(
        root=root,
        cohort="nlsy97",
        variant_token="zscore_by_branch",
        keep_workspace=False,
    )

    mean = pd.read_csv(root / "outputs/tables/g_mean_diff_zscore_by_branch.csv")
    vr = pd.read_csv(root / "outputs/tables/g_variance_ratio_zscore_by_branch.csv")
    assert float(mean.iloc[0]["d_g"]) == pytest.approx(0.20)
    assert float(vr.iloc[0]["VR_g"]) == pytest.approx(1.10)
    manifest = json.loads(
        (root / "outputs/tables/harmonization_rerun_manifest_nlsy97_zscore_by_branch.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["exploratory_override_used"] is True
