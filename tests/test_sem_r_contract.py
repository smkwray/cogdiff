import json
import random
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sem_fit.R"


def _has_r_package(package_name: str) -> bool:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False
    check = subprocess.run(
        [rscript, "-e", f"if (!requireNamespace('{package_name}', quietly = TRUE)) quit(status = 1)"],
        capture_output=True,
        text=True,
    )
    return check.returncode == 0


HAS_JSONLITE = _has_r_package("jsonlite")
HAS_LAVAAN = _has_r_package("lavaan")


def _build_inputs(
    tmp_path: Path,
    request_steps=None,
    partial_intercepts=None,
    se_mode: str = "standard",
    cluster_col: str | None = None,
    weight_col: str | None = None,
):
    if request_steps is None:
        request_steps = ["configural", "metric", "scalar", "strict"]
    if partial_intercepts is None:
        partial_intercepts = ["x1~1", "x2~1"]

    outdir = tmp_path / "interim"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "model.lavaan").write_text("g =~ x1 + x2 + x3\n", encoding="utf-8")

    rng = random.Random(1729)
    rows = []
    for sex, offset in [("F", 0), ("M", 1)]:
        for idx in range(40):
            base = 50 + offset + idx * 0.25
            row = {
                "sex": sex,
                "x1": base + rng.gauss(0.0, 0.9),
                "x2": 0.8 * base + rng.gauss(0.0, 1.1),
                "x3": 1.2 * base + rng.gauss(0.0, 1.0),
            }
            if cluster_col:
                row[cluster_col] = idx
            if weight_col:
                row[weight_col] = 1.0 + 0.5 * idx
            rows.append(row)
    data = pd.DataFrame(rows)
    data_csv = tmp_path / "sem_input.csv"
    data.to_csv(data_csv, index=False)

    request = {
        "cohort": "nlsy79",
        "data_csv": str(data_csv),
        "group_col": "sex",
        "estimator": "MLR",
        "missing": "fiml",
        "invariance_steps": request_steps,
        "partial_intercepts": partial_intercepts,
        "observed_tests": ["x1", "x2", "x3"],
        "se_mode": se_mode,
        "cluster_col": cluster_col,
        "weight_col": weight_col,
    }
    request_path = outdir / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return request_path, outdir


@pytest.mark.skipif(
    not (HAS_JSONLITE and HAS_LAVAAN),
    reason="Rscript/jsonlite/lavaan unavailable in this environment.",
)
def test_sem_fit_writes_expected_contract_files(tmp_path: Path) -> None:
    request_path, outdir = _build_inputs(tmp_path)
    run = subprocess.run(
        ["Rscript", str(SCRIPT), "--request", str(request_path), "--outdir", str(outdir)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr

    fit_indices = pd.read_csv(outdir / "fit_indices.csv")
    params = pd.read_csv(outdir / "params.csv")
    latent_summary = pd.read_csv(outdir / "latent_summary.csv")
    modindices = pd.read_csv(outdir / "modindices.csv")
    lavtestscore = pd.read_csv(outdir / "lavtestscore.csv")
    group_audit = pd.read_csv(outdir / "group_label_audit.csv")
    sex_estimands = pd.read_csv(outdir / "sex_group_estimands.csv")

    expected_fit_cols = {"cohort", "model_step", "cfi", "tli", "rmsea", "srmr", "chisq_scaled", "df", "aic", "bic"}
    expected_param_cols = {"cohort", "model_step", "group", "lhs", "op", "rhs", "est", "se", "z", "p", "std_all"}
    expected_latent_cols = {"cohort", "group", "factor", "mean", "var", "sd"}
    expected_mod_cols = {"cohort", "model_step", "lhs", "op", "rhs", "mi", "epc"}
    expected_score_cols = {
        "cohort",
        "model_step",
        "lhs",
        "op",
        "rhs",
        "x2",
        "df",
        "p_value",
        "mapped_lhs",
        "mapped_op",
        "mapped_rhs",
        "mapped_group_lhs",
        "mapped_group_rhs",
        "constraint_type",
    }
    expected_group_audit_cols = {
        "cohort",
        "model_step",
        "group_index",
        "group_label",
        "lavaan_group_label",
        "reference_group",
        "female_group",
        "male_group",
    }
    expected_estimand_cols = {
        "cohort",
        "model_step",
        "factor",
        "female_group",
        "male_group",
        "mean_female",
        "mean_male",
        "var_female",
        "var_male",
        "d_g",
        "vr",
        "log_vr",
    }

    assert expected_fit_cols.issubset(set(fit_indices.columns))
    assert expected_param_cols.issubset(set(params.columns))
    assert expected_latent_cols.issubset(set(latent_summary.columns))
    assert expected_mod_cols.issubset(set(modindices.columns))
    assert expected_score_cols.issubset(set(lavtestscore.columns))
    assert expected_group_audit_cols.issubset(set(group_audit.columns))
    assert expected_estimand_cols.issubset(set(sex_estimands.columns))

    assert len(fit_indices) == 4
    assert set(fit_indices["model_step"]) == {"configural", "metric", "scalar", "strict"}
    assert set(latent_summary["group"]) == {"F", "M"}
    assert "~1" in params["op"].astype(str).values
    assert set(latent_summary["factor"]) == {"g"}


@pytest.mark.skipif(
    not (HAS_JSONLITE and HAS_LAVAAN),
    reason="Rscript/jsonlite/lavaan unavailable in this environment.",
)
def test_sem_fit_rejects_invalid_se_mode(tmp_path: Path) -> None:
    request_path, outdir = _build_inputs(
        tmp_path,
        request_steps=["configural"],
        partial_intercepts=[],
        se_mode="unsupported_mode",
    )
    run = subprocess.run(
        ["Rscript", str(SCRIPT), "--request", str(request_path), "--outdir", str(outdir)],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    combined = (run.stdout + run.stderr).lower()
    assert "unsupported se_mode" in combined


@pytest.mark.skipif(
    not (HAS_JSONLITE and HAS_LAVAAN),
    reason="Rscript/jsonlite/lavaan unavailable in this environment.",
)
def test_sem_fit_rejects_missing_cluster_col_for_cluster_mode(tmp_path: Path) -> None:
    request_path, outdir = _build_inputs(
        tmp_path,
        request_steps=["configural"],
        partial_intercepts=[],
        se_mode="robust.cluster",
        cluster_col=None,
        weight_col=None,
    )
    run = subprocess.run(
        ["Rscript", str(SCRIPT), "--request", str(request_path), "--outdir", str(outdir)],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    combined = (run.stdout + run.stderr).lower()
    assert "robust.cluster se_mode requires cluster_col" in combined


@pytest.mark.skipif(
    not HAS_JSONLITE or HAS_LAVAAN,
    reason="R package availability does not exercise missing-lavaan path.",
)
def test_sem_fit_errors_without_lavaan(tmp_path: Path) -> None:
    request_path, outdir = _build_inputs(
        tmp_path,
        request_steps=["configural"],
        partial_intercepts=[],
    )
    run = subprocess.run(
        ["Rscript", str(SCRIPT), "--request", str(request_path), "--outdir", str(outdir)],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    combined = (run.stdout + run.stderr).lower()
    assert "install.packages('lavaan')" in combined
