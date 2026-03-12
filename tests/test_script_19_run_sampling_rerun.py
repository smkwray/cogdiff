from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _module():
    path = _repo_root() / "scripts" / "19_run_sampling_rerun.py"
    spec = importlib.util.spec_from_file_location("stage19_sampling_rerun", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _bootstrap_config(root: Path) -> None:
    _write(
        root / "config/paths.yml",
        "raw_dir: data/raw\n"
        "interim_dir: data/interim\n"
        "processed_dir: data/processed\n"
        "outputs_dir: outputs\n"
        "logs_dir: outputs/logs\n"
        "manifest_file: data/raw/manifest.json\n"
        "sem_interim_dir: data/interim/sem\n"
        "links_interim_dir: data/interim/links\n",
    )
    _write(
        root / "config/models.yml",
        "hierarchical_factors:\n"
        "  speed: [GS]\n"
        "  math: [AR]\n"
        "  verbal: [WK]\n"
        "  technical: [MC]\n"
        "cnlsy_single_factor: [PPVT, DIGITSPAN]\n"
        "reporting:\n"
        "  warning_policy:\n"
        "    enabled: true\n"
        "    threshold: fail\n",
    )
    for cohort in ("nlsy79", "nlsy97", "cnlsy"):
        _write(root / "config" / f"{cohort}.yml", "sample_construct: {id_col: person_id, sex_col: sex}\n")
    _write(root / "scripts/sem_fit.R", "cat('ok\\n')\n")
    _write(root / "data/raw/manifest.json", "{}\n")


def test_stage19_one_pair_sampling_reduces_nlsy79_and_exports_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    # source processed inputs
    pd.DataFrame(
        [
            {"person_id": 1000300, "sex": "M", "GS": 0.1},
            {"person_id": 1000400, "sex": "F", "GS": 0.2},
            {"person_id": 1000500, "sex": "M", "GS": 0.3},
        ]
    ).to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame(
        [
            {"person_id": 1, "sex": "M", "R9708601": 11, "GS": 0.1},
            {"person_id": 2, "sex": "F", "R9708601": 11, "GS": 0.2},
            {"person_id": 3, "sex": "M", "R9708601": 12, "GS": 0.3},
        ]
    ).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)
    pd.DataFrame(
        [
            {"person_id": 10, "sex": "M", "PPVT": 1.0},
            {"person_id": 11, "sex": "F", "PPVT": 1.1},
        ]
    ).to_csv(root / "data/processed/cnlsy_cfa_resid.csv", index=False)

    # nlsy79 family map merges 1000300 + 1000400 into one family
    pd.DataFrame(
        [
            {"SubjectTag": 1000300, "PartnerTag": 1000400, "family_id": "F10003"},
        ]
    ).to_csv(root / "data/interim/links/links79_links.csv", index=False)

    def fake_run_stage_sequence(root: Path, isolated_root: Path, log_path: Path) -> None:
        tables = isolated_root / "outputs" / "tables"
        fits = isolated_root / "outputs" / "model_fits"
        tables.mkdir(parents=True, exist_ok=True)
        (fits / "nlsy97").mkdir(parents=True, exist_ok=True)
        (fits / "cnlsy").mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [{"cohort": "nlsy79", "d_g": 0.10, "SE_d_g": 0.05, "ci_low_d_g": 0.00, "ci_high_d_g": 0.20, "IQ_diff": 1.5, "SE": 0.75, "ci_low": 0.0, "ci_high": 3.0}]
        ).to_csv(tables / "g_mean_diff.csv", index=False)

        # fallback rows for missing cohorts
        pd.DataFrame(
            [
                {"cohort": "nlsy97", "model_step": "metric", "group": "male", "lhs": "g", "op": "~1", "rhs": "", "est": 0.2, "se": 0.1},
                {"cohort": "nlsy97", "model_step": "metric", "group": "female", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.1},
            ]
        ).to_csv(fits / "nlsy97" / "params.csv", index=False)
        pd.DataFrame(
            [
                {"cohort": "cnlsy", "model_step": "metric", "group": "male", "lhs": "g", "op": "~1", "rhs": "", "est": -0.1, "se": 0.1},
                {"cohort": "cnlsy", "model_step": "metric", "group": "female", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 0.1},
            ]
        ).to_csv(fits / "cnlsy" / "params.csv", index=False)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run_stage_sequence", fake_run_stage_sequence)

    manifest = module.run_sampling_rerun(
        root=root,
        variant_token="one_pair_per_family",
        keep_workspace=False,
    )

    counts = pd.read_csv(root / "outputs/tables/sample_counts_one_pair_per_family.csv")
    nlsy79_row = counts[counts["cohort"] == "nlsy79"].iloc[0]
    assert int(nlsy79_row["n_input"]) == 3
    assert int(nlsy79_row["n_after_dedupe"]) == 2

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff_one_pair_per_family.csv")
    assert set(g_mean["cohort"]) == {"nlsy79", "nlsy97", "cnlsy"}

    manifest_path = root / "outputs/tables/sampling_rerun_manifest_one_pair_per_family.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["variant_token"] == "one_pair_per_family"
    assert manifest["workspace_kept"] is False
    workspace_path = Path(payload["workspace_root"])
    if workspace_path.is_absolute():
        assert not workspace_path.exists()
    else:
        assert not (root / workspace_path).exists()


def test_stage19_falls_back_when_model_se_is_implausibly_large(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    # nlsy79 only is enough for this check; keep other cohorts minimal.
    pd.DataFrame(
        [
            {"person_id": 1, "sex": "M", "GS": 1.0},
            {"person_id": 2, "sex": "F", "GS": -1.0},
            {"person_id": 3, "sex": "M", "GS": 1.1},
            {"person_id": 4, "sex": "F", "GS": -1.1},
        ]
    ).to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame([{"person_id": 1, "sex": "M"}]).to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)
    pd.DataFrame([{"person_id": 1, "sex": "M"}]).to_csv(root / "data/processed/cnlsy_cfa_resid.csv", index=False)
    pd.DataFrame(columns=["SubjectTag", "PartnerTag", "family_id"]).to_csv(
        root / "data/interim/links/links79_links.csv",
        index=False,
    )

    def fake_run_stage_sequence(root: Path, isolated_root: Path, log_path: Path) -> None:
        tables = isolated_root / "outputs" / "tables"
        fits = isolated_root / "outputs" / "model_fits" / "nlsy79"
        tables.mkdir(parents=True, exist_ok=True)
        fits.mkdir(parents=True, exist_ok=True)
        # no g_mean table row -> force model-fit derivation path
        pd.DataFrame(columns=["cohort", "d_g"]).to_csv(tables / "g_mean_diff.csv", index=False)
        # very large SE should trigger sample fallback
        pd.DataFrame(
            [
                {"cohort": "nlsy79", "model_step": "metric", "group": "male", "lhs": "g", "op": "~1", "rhs": "", "est": 0.2, "se": 50.0},
                {"cohort": "nlsy79", "model_step": "metric", "group": "female", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0, "se": 50.0},
            ]
        ).to_csv(fits / "params.csv", index=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run_stage_sequence", fake_run_stage_sequence)

    module.run_sampling_rerun(root=root, variant_token="full_cohort", keep_workspace=False)
    out = pd.read_csv(root / "outputs/tables/g_mean_diff_full_cohort.csv")
    nlsy79 = out[out["cohort"] == "nlsy79"].iloc[0]
    assert float(nlsy79["SE_d_g"]) < 5.0
