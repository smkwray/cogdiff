from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _module():
    path = _repo_root() / "scripts" / "20_run_inference_bootstrap.py"
    spec = importlib.util.spec_from_file_location("stage20_inference_bootstrap", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_sem_rep_outputs(rep_dir: Path, *, cohort: str, d_g: float, vr_g: float) -> None:
    rep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"cohort": cohort, "model_step": "metric", "group": "female", "lhs": "g", "op": "~~", "rhs": "g", "est": 1.0},
            {"cohort": cohort, "model_step": "metric", "group": "male", "lhs": "g", "op": "~~", "rhs": "g", "est": vr_g},
            {"cohort": cohort, "model_step": "scalar", "group": "female", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0},
            {"cohort": cohort, "model_step": "scalar", "group": "male", "lhs": "g", "op": "~1", "rhs": "", "est": d_g},
        ]
    ).to_csv(rep_dir / "params.csv", index=False)
    pd.DataFrame(
        [
            {
                "cohort": cohort,
                "model_step": "scalar",
                "factor": "g",
                "female_group": "female",
                "male_group": "male",
                "mean_female": 0.0,
                "mean_male": d_g,
                "var_female": 1.0,
                "var_male": vr_g,
            }
        ]
    ).to_csv(rep_dir / "sex_group_estimands.csv", index=False)


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
        "cnlsy_single_factor: [PPVT, DIGITSPAN]\n",
    )
    _write(root / "data/raw/manifest.json", "{}\n")


def _cohort_full_sample_estimate(root: Path, cohort: str, module, models_cfg: dict) -> dict[str, float]:
    if cohort == "nlsy79":
        source_path = root / "data/processed/nlsy79_cfa_resid.csv"
    elif cohort == "nlsy97":
        source_path = root / "data/processed/nlsy97_cfa_resid.csv"
    else:
        source_path = root / "data/processed/cnlsy_cfa_resid.csv"

    df = pd.read_csv(source_path, low_memory=False)
    if cohort == "cnlsy":
        indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", [])]
    else:
        indicators = [str(x) for x in module.hierarchical_subtests(models_cfg)]

    df = df.copy()
    df["sex_label"] = module._sex_labels(df["sex"])
    df["g_proxy"] = module._composite_score(df, indicators)
    d_g, vr = module._estimate_stats(df)
    if d_g is None or vr is None:
        raise AssertionError(f"Could not estimate full-sample stats for {cohort}")

    return {"d_g": float(d_g), "VR_g": float(vr)}


def _write_bootstrap_outputs(
    root: Path,
    *,
    mean_rows: list[dict[str, float | int | str]],
    vr_rows: list[dict[str, float | int | str]],
) -> None:
    tables_dir = root / "outputs/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mean_rows).to_csv(tables_dir / "g_mean_diff_family_bootstrap.csv", index=False)
    pd.DataFrame(vr_rows).to_csv(tables_dir / "g_variance_ratio_family_bootstrap.csv", index=False)


def _fake_rows(n: int, *, offset: float = 0.0) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for i in range(n):
        male = i % 2 == 0
        shift = 0.4 if male else -0.4
        rows.append(
            {
                "person_id": i + 1,
                "sex": "M" if male else "F",
                "GS": offset + shift + i * 0.01,
                "AR": offset + shift + i * 0.02,
                "WK": offset + shift + i * 0.03,
                "MC": offset + shift + i * 0.04,
            }
        )
    return rows


def test_stage20_writes_bootstrap_variant_outputs(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.DataFrame(_fake_rows(30, offset=0.1))
    nlsy79.to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)

    nlsy97 = pd.DataFrame(_fake_rows(30, offset=0.2))
    nlsy97["R9708601"] = [10 + (i % 8) for i in range(len(nlsy97))]
    nlsy97.to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    cnlsy = pd.DataFrame(
        [
            {
                "person_id": i + 1,
                "sex": "M" if i % 2 == 0 else "F",
                "PPVT": (0.5 if i % 2 == 0 else -0.5) + i * 0.01,
                "DIGITSPAN": (0.5 if i % 2 == 0 else -0.5) + i * 0.02,
            }
            for i in range(30)
        ]
    )
    cnlsy.to_csv(root / "data/processed/cnlsy_cfa_resid.csv", index=False)

    pd.DataFrame(
        [
            {"SubjectTag": 1, "PartnerTag": 2, "family_id": "F1"},
            {"SubjectTag": 3, "PartnerTag": 4, "family_id": "F2"},
        ]
    ).to_csv(root / "data/interim/links/links79_links.csv", index=False)

    manifest = module.run_inference_bootstrap(
        root=root,
        variant_token="family_bootstrap",
        n_bootstrap=80,
        seed=123,
    )

    g_mean_path = root / "outputs/tables/g_mean_diff_family_bootstrap.csv"
    g_vr_path = root / "outputs/tables/g_variance_ratio_family_bootstrap.csv"
    manifest_path = root / "outputs/tables/inference_rerun_manifest_family_bootstrap.json"

    assert g_mean_path.exists()
    assert g_vr_path.exists()
    assert manifest_path.exists()

    g_mean = pd.read_csv(g_mean_path)
    g_vr = pd.read_csv(g_vr_path)
    assert set(g_mean["cohort"]) == {"nlsy79", "nlsy97", "cnlsy"}
    assert set(g_vr["cohort"]) == {"nlsy79", "nlsy97", "cnlsy"}
    assert (g_mean["SE_d_g"] > 0).all()
    assert (g_vr["SE_logVR"] > 0).all()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["variant_token"] == "family_bootstrap"
    assert payload["n_bootstrap"] == 80
    assert payload["engine"] == "proxy"
    assert payload["point_estimate_source"] == "full_sample_fit"
    assert "full-sample" in payload["point_estimate_detail"]
    assert set(payload["cohorts"]) == {"nlsy79", "nlsy97", "cnlsy"}
    assert isinstance(payload["cohort_details"], list)
    assert len(payload["cohort_details"]) == 3
    assert manifest["seed"] == 123

    models_cfg = module.load_yaml(root / "config/models.yml")
    for cohort in ("nlsy79", "nlsy97", "cnlsy"):
        expected = _cohort_full_sample_estimate(root, cohort, module, models_cfg)
        mean_row = g_mean.loc[g_mean["cohort"] == cohort].iloc[0]
        vr_row = g_vr.loc[g_vr["cohort"] == cohort].iloc[0]
        assert str(mean_row["status"]).strip().lower() == "computed"
        assert str(vr_row["status"]).strip().lower() == "computed"
        assert float(mean_row["d_g"]) == pytest.approx(expected["d_g"], rel=1e-12, abs=1e-12)
    assert float(vr_row["VR_g"]) == pytest.approx(expected["VR_g"], rel=1e-12, abs=1e-12)


def test_stage20_supports_custom_artifact_prefix(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.DataFrame(_fake_rows(30, offset=0.1))
    nlsy79.to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)

    pd.DataFrame(
        [
            {"SubjectTag": 1, "PartnerTag": 2, "family_id": "F1"},
            {"SubjectTag": 3, "PartnerTag": 4, "family_id": "F2"},
        ]
    ).to_csv(root / "data/interim/links/links79_links.csv", index=False)

    module.run_inference_bootstrap(
        root=root,
        variant_token="family_bootstrap",
        artifact_prefix="g_proxy",
        n_bootstrap=50,
        seed=123,
    )

    g_mean_path = root / "outputs/tables/g_proxy_mean_diff_family_bootstrap.csv"
    g_vr_path = root / "outputs/tables/g_proxy_variance_ratio_family_bootstrap.csv"
    manifest_path = root / "outputs/tables/inference_rerun_manifest_g_proxy_family_bootstrap.json"

    assert g_mean_path.exists()
    assert g_vr_path.exists()
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["artifact_prefix"] == "g_proxy"


def test_stage20_skips_successful_cohorts_when_flag_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.DataFrame(_fake_rows(30, offset=0.1))
    nlsy79.to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)

    _write_bootstrap_outputs(
        root,
        mean_rows=[
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "precomputed",
                "d_g": 42.0,
                "SE_d_g": 0.5,
                "ci_low_d_g": 41.0,
                "ci_high_d_g": 43.0,
                "IQ_diff": 630.0,
                "SE": 0.5,
                "ci_low": 41.0,
                "ci_high": 43.0,
            }
        ],
        vr_rows=[
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "precomputed",
                "VR_g": 2.0,
                "SE_logVR": 0.01,
                "ci_low": 1.9,
                "ci_high": 2.1,
            }
        ],
    )

    monkeypatch.setattr(module, "_cluster_bootstrap_proxy", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected recompute")))

    manifest = module.run_inference_bootstrap(
        root=root,
        variant_token="family_bootstrap",
        n_bootstrap=80,
        seed=11,
        cohorts=["nlsy79"],
        skip_successful=True,
    )

    assert manifest["skip_successful_requested"] is True
    assert manifest["skipped_existing_cohorts"] == ["nlsy79"]
    assert len(manifest["cohort_details"]) == 1
    detail = manifest["cohort_details"][0]
    assert detail["cohort"] == "nlsy79"
    assert detail["status"] == "computed"
    assert detail["skipped_existing"] is True

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff_family_bootstrap.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio_family_bootstrap.csv")
    assert g_mean.loc[0, "d_g"] == 42.0
    assert g_vr.loc[0, "VR_g"] == 2.0


def test_stage20_skip_successful_does_not_skip_when_requested_bootstrap_increases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.DataFrame(_fake_rows(30, offset=0.1))
    nlsy79.to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame(
        [
            {"SubjectTag": 1, "PartnerTag": 2, "family_id": "F1"},
            {"SubjectTag": 3, "PartnerTag": 4, "family_id": "F2"},
        ]
    ).to_csv(root / "data/interim/links/links79_links.csv", index=False)

    # Existing computed tables + prior manifest says attempted=80.
    _write_bootstrap_outputs(
        root,
        mean_rows=[
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "precomputed",
                "d_g": 42.0,
                "SE_d_g": 0.5,
                "ci_low_d_g": 41.0,
                "ci_high_d_g": 43.0,
                "IQ_diff": 630.0,
                "SE": 0.5,
                "ci_low": 41.0,
                "ci_high": 43.0,
            }
        ],
        vr_rows=[
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "precomputed",
                "VR_g": 2.0,
                "SE_logVR": 0.01,
                "ci_low": 1.9,
                "ci_high": 2.1,
            }
        ],
    )
    manifest_path = root / "outputs/tables/inference_rerun_manifest_family_bootstrap.json"
    _write(
        manifest_path,
        json.dumps(
            {
                "variant_token": "family_bootstrap",
                "engine": "proxy",
                "n_bootstrap": 80,
                "cohort_details": [
                    {
                        "cohort": "nlsy79",
                        "status": "computed",
                        "reason": "",
                        "attempted": 80,
                        "success": 80,
                        "success_share": 1.0,
                    }
                ],
            }
        )
        + "\n",
    )

    called = {"ran": False}

    def _fake_cluster(*args, **kwargs):
        called["ran"] = True
        return [0.0] * 100, [1.0] * 100

    monkeypatch.setattr(module, "_cluster_bootstrap_proxy", _fake_cluster)

    module.run_inference_bootstrap(
        root=root,
        variant_token="family_bootstrap",
        n_bootstrap=100,
        seed=11,
        cohorts=["nlsy79"],
        skip_successful=True,
    )
    assert called["ran"] is True

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff_family_bootstrap.csv")
    assert float(g_mean.loc[0, "d_g"]) != 42.0


def test_stage20_skip_successful_requires_both_tables_to_be_computed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.DataFrame(_fake_rows(30, offset=0.1))
    nlsy79.to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)

    _write_bootstrap_outputs(
        root,
        mean_rows=[
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "precomputed",
                "d_g": 42.0,
                "SE_d_g": 0.5,
                "ci_low_d_g": 41.0,
                "ci_high_d_g": 43.0,
                "IQ_diff": 630.0,
                "SE": 0.5,
                "ci_low": 41.0,
                "ci_high": 43.0,
            }
        ],
        vr_rows=[
            {
                "cohort": "nlsy79",
                "status": "not_feasible",
                "reason": "missing",
                "VR_g": pd.NA,
                "SE_logVR": pd.NA,
                "ci_low": pd.NA,
                "ci_high": pd.NA,
            }
        ],
    )

    monkeypatch.setattr(module, "_cluster_bootstrap_proxy", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected recompute")))

    with pytest.raises(AssertionError):
        module.run_inference_bootstrap(
            root=root,
            variant_token="family_bootstrap",
            n_bootstrap=50,
            seed=11,
            cohorts=["nlsy79"],
            skip_successful=True,
        )


def test_stage20_no_skip_without_flag_recomputes_existing_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.DataFrame(_fake_rows(30, offset=0.1))
    nlsy79.to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)

    _write_bootstrap_outputs(
        root,
        mean_rows=[
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "precomputed",
                "d_g": 42.0,
                "SE_d_g": 0.5,
                "ci_low_d_g": 41.0,
                "ci_high_d_g": 43.0,
                "IQ_diff": 630.0,
                "SE": 0.5,
                "ci_low": 41.0,
                "ci_high": 43.0,
            }
        ],
        vr_rows=[
            {
                "cohort": "nlsy79",
                "status": "computed",
                "reason": "precomputed",
                "VR_g": 2.0,
                "SE_logVR": 0.01,
                "ci_low": 1.9,
                "ci_high": 2.1,
            }
        ],
    )

    calls = {"n": 0}

    def _proxy(_df: pd.DataFrame, *, n_boot: int, seed: int) -> tuple[list[float], list[float]]:
        calls["n"] += 1
        return [0.1] * n_boot, [1.0] * n_boot

    monkeypatch.setattr(module, "_cluster_bootstrap_proxy", _proxy)

    manifest = module.run_inference_bootstrap(
        root=root,
        variant_token="family_bootstrap",
        n_bootstrap=50,
        seed=11,
        cohorts=["nlsy79"],
    )

    assert manifest["skip_successful_requested"] is False
    assert manifest["skipped_existing_cohorts"] == []
    assert calls["n"] == 1


def test_stage20_extract_sem_estimands_handles_scalar_metric_rows(tmp_path: Path) -> None:
    module = _module()
    params_path = tmp_path / "params.csv"
    pd.DataFrame(
        [
            {"cohort": "nlsy79", "model_step": "metric", "group": "female", "lhs": "g", "op": "~~", "rhs": "g", "est": 1.0},
            {"cohort": "nlsy79", "model_step": "metric", "group": "male", "lhs": "g", "op": "~~", "rhs": "g", "est": 1.2},
            {"cohort": "nlsy79", "model_step": "scalar", "group": "female", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0},
            {"cohort": "nlsy79", "model_step": "scalar", "group": "male", "lhs": "g", "op": "~1", "rhs": "", "est": 0.2},
        ]
    ).to_csv(params_path, index=False)

    d_g, vr, reason = module._extract_sem_estimands(
        params_path=params_path,
        cohort="nlsy79",
        reference_group="female",
    )
    assert reason is None
    assert d_g == pytest.approx(0.2)
    assert vr == pytest.approx(1.2)


def test_stage20_extract_sem_estimands_prefers_sex_estimand_export(tmp_path: Path) -> None:
    module = _module()
    params_path = tmp_path / "params.csv"
    pd.DataFrame(
        [
            {"cohort": "nlsy79", "model_step": "metric", "group": "female", "lhs": "g", "op": "~~", "rhs": "g", "est": 4.0},
            {"cohort": "nlsy79", "model_step": "metric", "group": "male", "lhs": "g", "op": "~~", "rhs": "g", "est": 4.0},
            {"cohort": "nlsy79", "model_step": "scalar", "group": "female", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0},
            {"cohort": "nlsy79", "model_step": "scalar", "group": "male", "lhs": "g", "op": "~1", "rhs": "", "est": 0.0},
        ]
    ).to_csv(params_path, index=False)
    sex_estimands_path = tmp_path / "sex_group_estimands.csv"
    pd.DataFrame(
        [
            {
                "cohort": "nlsy79",
                "model_step": "scalar",
                "factor": "g",
                "female_group": "female",
                "male_group": "male",
                "mean_female": 0.0,
                "mean_male": 0.3,
                "var_female": 4.0,
                "var_male": 9.0,
            },
            {
                "cohort": "nlsy79",
                "model_step": "metric",
                "factor": "g",
                "female_group": "female",
                "male_group": "male",
                "mean_female": 0.0,
                "mean_male": 0.3,
                "var_female": 4.0,
                "var_male": 9.0,
            },
        ]
    ).to_csv(sex_estimands_path, index=False)

    d_g, vr, reason = module._extract_sem_estimands(
        params_path=params_path,
        sex_estimands_path=sex_estimands_path,
        cohort="nlsy79",
        reference_group="female",
    )
    assert reason is None
    assert d_g == pytest.approx(0.3)
    assert vr == pytest.approx(2.25)


def test_stage20_sem_refit_engine_flow_without_r(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.DataFrame(_fake_rows(24, offset=0.1))
    nlsy79.to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)

    nlsy97 = pd.DataFrame(_fake_rows(24, offset=0.2))
    nlsy97["R9708601"] = [100 + (i % 6) for i in range(len(nlsy97))]
    nlsy97.to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    cnlsy = pd.DataFrame(
        [
            {
                "person_id": i + 1,
                "sex": "M" if i % 2 == 0 else "F",
                "PPVT": (0.4 if i % 2 == 0 else -0.4) + i * 0.01,
                "DIGITSPAN": (0.4 if i % 2 == 0 else -0.4) + i * 0.02,
                "MPUBID": 200 + (i % 5),
            }
            for i in range(24)
        ]
    )
    cnlsy.to_csv(root / "data/processed/cnlsy_cfa_resid.csv", index=False)

    pd.DataFrame(
        [
            {"SubjectTag": 1, "PartnerTag": 2, "family_id": "F1"},
            {"SubjectTag": 3, "PartnerTag": 4, "family_id": "F2"},
        ]
    ).to_csv(root / "data/interim/links/links79_links.csv", index=False)

    counter = {"calls": 0}

    def _fake_refit(**kwargs):
        counter["calls"] += 1
        work_dir = kwargs.get("work_dir")
        if work_dir is not None and "rep_" in str(work_dir) and (counter["calls"] % 5 == 0):
            return None, None, "simulated_failure"
        return 0.12, 1.18, None

    monkeypatch.setattr(module, "_run_sem_refit_once", _fake_refit)

    manifest = module.run_inference_bootstrap(
        root=root,
        variant_token="family_bootstrap",
        n_bootstrap=50,
        seed=11,
        engine="sem_refit",
        min_success_share=0.70,
        sem_timeout_seconds=1.0,
    )

    assert manifest["engine"] == "sem_refit"
    details = manifest["cohort_details"]
    assert len(details) == 3
    assert all(str(row.get("status")) == "computed" for row in details)

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff_family_bootstrap.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio_family_bootstrap.csv")
    assert (g_mean["status"].astype(str).str.lower() == "computed").all()
    assert (g_vr["status"].astype(str).str.lower() == "computed").all()


def test_stage20_sem_refit_engine_fails_when_bootstrap_success_share_below_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.DataFrame(_fake_rows(30, offset=0.1))
    nlsy79.to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)

    pd.DataFrame(
        [
            {"SubjectTag": 1, "PartnerTag": 2, "family_id": "F1"},
            {"SubjectTag": 3, "PartnerTag": 4, "family_id": "F2"},
        ]
    ).to_csv(root / "data/interim/links/links79_links.csv", index=False)

    counter = {"calls": 0}

    def _fake_refit(**kwargs):
        call_idx = counter["calls"]
        counter["calls"] += 1
        if call_idx == 0:
            return 0.15, 1.12, None
        if call_idx % 2 == 0:
            return None, None, "simulated_failure"
        return 0.15, 1.12, None

    monkeypatch.setattr(module, "_run_sem_refit_once", _fake_refit)

    manifest = module.run_inference_bootstrap(
        root=root,
        variant_token="family_bootstrap",
        n_bootstrap=50,
        seed=11,
        engine="sem_refit",
        min_success_share=0.80,
        sem_timeout_seconds=1.0,
        cohorts=["nlsy79"],
    )

    assert manifest["engine"] == "sem_refit"
    assert len(manifest["cohorts"]) == 1
    assert manifest["cohorts"] == ["nlsy79"]
    assert len(manifest["cohort_details"]) == 1

    detail = manifest["cohort_details"][0]
    assert detail["cohort"] == "nlsy79"
    assert detail["status"] == "not_feasible"
    assert str(detail["reason"]).startswith("bootstrap_success_below_threshold")

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff_family_bootstrap.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio_family_bootstrap.csv")

    assert (g_mean["cohort"] == "nlsy79").all()
    assert (g_vr["cohort"] == "nlsy79").all()
    assert str(g_mean.loc[0, "status"]).lower() == "not_feasible"
    assert str(g_vr.loc[0, "status"]).lower() == "not_feasible"
    assert str(g_mean.loc[0, "reason"]).startswith("bootstrap_success_below_threshold")
    assert str(g_vr.loc[0, "reason"]).startswith("bootstrap_success_below_threshold")


def test_stage20_sem_refit_parallel_workers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.DataFrame(_fake_rows(30, offset=0.1))
    nlsy79.to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame(
        [
            {"SubjectTag": 1, "PartnerTag": 2, "family_id": "F1"},
            {"SubjectTag": 3, "PartnerTag": 4, "family_id": "F2"},
        ]
    ).to_csv(root / "data/interim/links/links79_links.csv", index=False)

    def _fake_refit(**kwargs):
        work_dir = Path(str(kwargs.get("work_dir", "")))
        if work_dir.name == "full_sample":
            return 0.15, 1.12, None
        if work_dir.name.startswith("rep_"):
            rep_idx = int(work_dir.name.split("_")[1])
            if rep_idx % 13 == 0:
                return None, None, "simulated_failure"
        return 0.15, 1.12, None

    monkeypatch.setattr(module, "_run_sem_refit_once", _fake_refit)

    manifest = module.run_inference_bootstrap(
        root=root,
        variant_token="family_bootstrap",
        n_bootstrap=50,
        seed=11,
        engine="sem_refit",
        min_success_share=0.70,
        sem_timeout_seconds=1.0,
        sem_jobs=4,
        cohorts=["nlsy79"],
    )

    assert manifest["engine"] == "sem_refit"
    assert manifest["sem_jobs"] == 4
    assert manifest["sem_threads_per_job"] == 1
    assert len(manifest["cohort_details"]) == 1
    assert manifest["cohort_details"][0]["status"] == "computed"

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff_family_bootstrap.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio_family_bootstrap.csv")
    assert str(g_mean.loc[0, "status"]).lower() == "computed"
    assert str(g_vr.loc[0, "status"]).lower() == "computed"


def test_stage20_sem_refit_resume_existing_reps_reuses_rep_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.DataFrame(_fake_rows(30, offset=0.1))
    nlsy79.to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame(
        [
            {"SubjectTag": 1, "PartnerTag": 2, "family_id": "F1"},
            {"SubjectTag": 3, "PartnerTag": 4, "family_id": "F2"},
        ]
    ).to_csv(root / "data/interim/links/links79_links.csv", index=False)

    cohort_root = root / "outputs/model_fits/bootstrap_inference/nlsy79"
    _write_sem_rep_outputs(cohort_root / "rep_0000", cohort="nlsy79", d_g=0.31, vr_g=1.21)
    _write_sem_rep_outputs(cohort_root / "rep_0001", cohort="nlsy79", d_g=0.32, vr_g=1.22)

    calls = {"full_sample": 0, "rep_calls": 0}

    def _fake_refit(**kwargs):
        work_dir = Path(str(kwargs.get("work_dir", "")))
        if work_dir.name == "full_sample":
            calls["full_sample"] += 1
            return 0.15, 1.12, None
        calls["rep_calls"] += 1
        return 0.15, 1.12, None

    monkeypatch.setattr(module, "_run_sem_refit_once", _fake_refit)

    manifest = module.run_inference_bootstrap(
        root=root,
        variant_token="family_bootstrap",
        n_bootstrap=50,
        seed=11,
        engine="sem_refit",
        resume_existing_reps=True,
        cohorts=["nlsy79"],
    )

    assert manifest["resume_existing_reps_requested"] is True
    detail = manifest["cohort_details"][0]
    assert detail["cohort"] == "nlsy79"
    assert detail["status"] == "computed"
    assert detail["reused_rep_count"] == 2
    assert detail["computed_rep_count"] == 48
    assert calls["full_sample"] == 1
    assert calls["rep_calls"] == 48


def test_stage20_sem_refit_without_resume_existing_reps_recomputes_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.DataFrame(_fake_rows(30, offset=0.1))
    nlsy79.to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame(
        [
            {"SubjectTag": 1, "PartnerTag": 2, "family_id": "F1"},
            {"SubjectTag": 3, "PartnerTag": 4, "family_id": "F2"},
        ]
    ).to_csv(root / "data/interim/links/links79_links.csv", index=False)

    cohort_root = root / "outputs/model_fits/bootstrap_inference/nlsy79"
    _write_sem_rep_outputs(cohort_root / "rep_0000", cohort="nlsy79", d_g=0.31, vr_g=1.21)
    _write_sem_rep_outputs(cohort_root / "rep_0001", cohort="nlsy79", d_g=0.32, vr_g=1.22)

    calls = {"full_sample": 0, "rep_calls": 0}

    def _fake_refit(**kwargs):
        work_dir = Path(str(kwargs.get("work_dir", "")))
        if work_dir.name == "full_sample":
            calls["full_sample"] += 1
            return 0.15, 1.12, None
        calls["rep_calls"] += 1
        return 0.15, 1.12, None

    monkeypatch.setattr(module, "_run_sem_refit_once", _fake_refit)

    manifest = module.run_inference_bootstrap(
        root=root,
        variant_token="family_bootstrap",
        n_bootstrap=50,
        seed=11,
        engine="sem_refit",
        cohorts=["nlsy79"],
    )

    assert manifest["resume_existing_reps_requested"] is False
    detail = manifest["cohort_details"][0]
    assert detail["cohort"] == "nlsy79"
    assert detail["status"] == "computed"
    assert detail["reused_rep_count"] == 0
    assert detail["computed_rep_count"] == 50
    assert calls["full_sample"] == 1
    assert calls["rep_calls"] == 50


def test_stage20_sem_refit_request_uses_relative_sem_input_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/sem_fit.R").write_text("cat('ok\\n')\n", encoding="utf-8")

    models_cfg = module.load_yaml(root / "config/models.yml")
    cohort_cfg = {"sample_construct": {"sex_col": "sex"}, "sem_fit": {"std_lv": True}}
    data = pd.DataFrame(
        {
            "sex": ["F", "M", "F", "M"],
            "GS": [0.1, 0.2, 0.3, 0.4],
            "AR": [0.2, 0.3, 0.4, 0.5],
            "WK": [0.3, 0.4, 0.5, 0.6],
            "MC": [0.4, 0.5, 0.6, 0.7],
        }
    )
    work_dir = root / "outputs/model_fits/bootstrap_inference/nlsy79/full_sample"

    monkeypatch.setattr(module, "rscript_path", lambda: "/usr/bin/Rscript")
    monkeypatch.setattr(module, "_extract_sem_estimands", lambda **kwargs: (0.15, 1.12, None))

    def _fake_run(args, check, text, capture_output, timeout, env):
        assert check is True
        assert text is True
        assert capture_output is True
        request_path = Path(args[args.index("--request") + 1])
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        assert payload["data_csv"] == "sem_input.csv"
        assert not Path(payload["data_csv"]).is_absolute()
        assert (request_path.parent / payload["data_csv"]).exists()
        return None

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    d_g, vr_g, reason = module._run_sem_refit_once(
        root=root,
        cohort="nlsy79",
        data=data,
        cohort_cfg=cohort_cfg,
        models_cfg=models_cfg,
        work_dir=work_dir,
        timeout_seconds=10.0,
    )

    assert d_g == pytest.approx(0.15)
    assert vr_g == pytest.approx(1.12)
    assert reason is None


def test_stage20_rejects_invalid_sem_jobs(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(ValueError, match="sem-jobs"):
        module.run_inference_bootstrap(
            root=tmp_path.resolve(),
            variant_token="family_bootstrap",
            n_bootstrap=50,
            seed=1,
            sem_jobs=0,
        )


def test_stage20_supports_cohort_filter(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.DataFrame(_fake_rows(30, offset=0.1))
    nlsy79.to_csv(root / "data/processed/nlsy79_cfa_resid.csv", index=False)
    pd.DataFrame(
        [
            {"SubjectTag": 1, "PartnerTag": 2, "family_id": "F1"},
            {"SubjectTag": 3, "PartnerTag": 4, "family_id": "F2"},
        ]
    ).to_csv(root / "data/interim/links/links79_links.csv", index=False)

    manifest = module.run_inference_bootstrap(
        root=root,
        variant_token="family_bootstrap",
        n_bootstrap=60,
        seed=321,
        cohorts=["nlsy79"],
    )
    assert manifest["cohorts"] == ["nlsy79"]

    g_mean = pd.read_csv(root / "outputs/tables/g_mean_diff_family_bootstrap.csv")
    g_vr = pd.read_csv(root / "outputs/tables/g_variance_ratio_family_bootstrap.csv")
    assert set(g_mean["cohort"]) == {"nlsy79"}
    assert set(g_vr["cohort"]) == {"nlsy79"}


def test_stage20_sem_refit_can_resume_existing_rep_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    root = tmp_path.resolve()
    _bootstrap_config(root)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/links").mkdir(parents=True, exist_ok=True)

    nlsy97 = pd.DataFrame(_fake_rows(60, offset=0.2))
    nlsy97["R9708601"] = [10 + (i % 8) for i in range(len(nlsy97))]
    nlsy97.to_csv(root / "data/processed/nlsy97_cfa_resid.csv", index=False)

    rep0 = root / "outputs/model_fits/bootstrap_inference/nlsy97/rep_0000"
    rep0.mkdir(parents=True, exist_ok=True)
    _write(
        rep0 / "params.csv",
        "model_step,lhs,op,rhs,group,est\n"
        "scalar,g,~1,,female,0.0\n"
        "scalar,g,~1,,male,0.2\n"
        "metric,g,~~,g,female,1.0\n"
        "metric,g,~~,g,male,1.2\n",
    )

    rep1 = root / "outputs/model_fits/bootstrap_inference/nlsy97/rep_0001"
    rep1.mkdir(parents=True, exist_ok=True)
    _write(
        rep1 / "sex_group_estimands.csv",
        "model_step,factor,mean_female,mean_male,var_female,var_male\n"
        "scalar,g,0.0,0.2,1.0,1.2\n",
    )

    def _fake_indices(family_labels: np.ndarray, *, n_boot: int, seed: int) -> list[np.ndarray]:
        _ = (seed, family_labels)
        out: list[np.ndarray] = []
        for i in range(n_boot):
            out.append(np.array([0, 1, 2], dtype=int) if i < 25 else np.array([], dtype=int))
        return out

    monkeypatch.setattr(module, "_cluster_bootstrap_indices", _fake_indices)

    calls: list[str] = []

    def _fake_sem_refit_once(*, work_dir: Path, **kwargs):
        calls.append(work_dir.name)
        if work_dir.name in {"rep_0000", "rep_0001"}:
            raise AssertionError("unexpected recompute of reused rep dir")
        return 0.1, 1.1, None

    monkeypatch.setattr(module, "_run_sem_refit_once", _fake_sem_refit_once)

    manifest = module.run_inference_bootstrap(
        root=root,
        variant_token="family_bootstrap",
        n_bootstrap=50,
        seed=123,
        engine="sem_refit",
        sem_jobs=1,
        sem_timeout_seconds=0.1,
        min_success_share=0.5,
        cohorts=["nlsy97"],
        resume_existing_reps=True,
    )

    assert manifest["engine"] == "sem_refit"
    assert manifest["resume_existing_reps_requested"] is True

    assert "full_sample" in calls
    assert "rep_0000" not in calls
    assert "rep_0001" not in calls
    assert len([name for name in calls if name.startswith("rep_")]) == 23

    details = manifest["cohort_details"][0]
    assert details["cohort"] == "nlsy97"
    assert int(details.get("reused_reps", -1)) == 2
    assert int(details.get("computed_reps", -1)) == 23


def test_stage20_sem_refit_writes_failure_logs_on_called_process_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root = tmp_path.resolve()
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/sem_fit.R").write_text("# stub\n", encoding="utf-8")

    # Force rscript discovery and a deterministic failure from subprocess.run.
    monkeypatch.setattr(module, "rscript_path", lambda: "/usr/bin/Rscript")

    def _fail_run(*args, **kwargs):
        raise module.subprocess.CalledProcessError(
            returncode=1,
            cmd=["Rscript", "sem_fit.R"],
            output="stdout-stub",
            stderr="stderr-stub",
        )

    monkeypatch.setattr(module.subprocess, "run", _fail_run)

    work_dir = root / "tmp_work/full_sample"
    df = pd.DataFrame(
        [
            {"sex": "F", "GS": 0.1},
            {"sex": "M", "GS": 0.2},
        ]
    )
    d_g, vr_g, reason = module._run_sem_refit_once(
        root=root,
        cohort="nlsy79",
        data=df,
        cohort_cfg={},
        models_cfg={"hierarchical_factors": {"speed": ["GS"]}, "reference_group": "female"},
        work_dir=work_dir,
        timeout_seconds=0.1,
        thread_limit=1,
    )
    assert d_g is None
    assert vr_g is None
    assert reason is not None and "sem_refit_failed" in reason
    assert (work_dir / "sem_fit_stderr.txt").exists()


def test_stage20_sem_refit_request_sets_force_standard_se(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    root = tmp_path.resolve()
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/sem_fit.R").write_text("# stub\n", encoding="utf-8")

    monkeypatch.setattr(module, "rscript_path", lambda: "/usr/bin/Rscript")

    def _ok_run(*args, **kwargs):
        class Dummy:
            stdout = ""
            stderr = ""
        return Dummy()

    monkeypatch.setattr(module.subprocess, "run", _ok_run)
    monkeypatch.setattr(module, "_extract_sem_estimands", lambda **kwargs: (0.1, 1.1, None))

    work_dir = root / "tmp_work/full_sample"
    df = pd.DataFrame(
        [
            {"sex": "F", "GS": 0.1},
            {"sex": "M", "GS": 0.2},
        ]
    )
    d_g, vr_g, reason = module._run_sem_refit_once(
        root=root,
        cohort="nlsy79",
        data=df,
        cohort_cfg={},
        models_cfg={"hierarchical_factors": {"speed": ["GS"]}, "reference_group": "female"},
        work_dir=work_dir,
        timeout_seconds=0.1,
        thread_limit=1,
    )
    assert reason is None
    assert d_g == 0.1
    assert vr_g == 1.1

    payload = json.loads((work_dir / "request.json").read_text(encoding="utf-8"))
    assert payload.get("force_standard_se") is True
