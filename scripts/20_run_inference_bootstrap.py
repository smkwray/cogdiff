#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import dump_json, load_yaml, project_root, relative_path
from nls_pipeline.sem import cnlsy_model_syntax, hierarchical_model_syntax, hierarchical_subtests, rscript_path

COHORTS: tuple[str, ...] = ("nlsy79", "nlsy97", "cnlsy")
DEFAULT_ENGINE = "proxy"
DEFAULT_INVARIANCE_STEPS: tuple[str, ...] = ("metric", "scalar")
DEFAULT_MIN_SUCCESS_SHARE = 0.90


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _status_is_computed(value: Any) -> bool:
    return str(value).strip().lower() == "computed"


def _coalesce_int(value: Any, default: int) -> int:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return default


def _coalesce_float(value: Any, default: float) -> float:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _read_inference_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _cohort_row(rows: pd.DataFrame, cohort: str) -> dict[str, Any] | None:
    if rows.empty or "cohort" not in rows.columns:
        return None
    cohort_values = rows["cohort"].astype(str)
    matches = rows[cohort_values == str(cohort)]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def _safe_float(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    try:
        number_f = float(number)
    except Exception:
        return None
    if not math.isfinite(number_f):
        return None
    return number_f


def _nlsy79_family_lookup(root: Path) -> dict[int, str]:
    links_path = root / "data" / "interim" / "links" / "links79_links.csv"
    if not links_path.exists():
        return {}
    links = pd.read_csv(links_path, low_memory=False)
    required = {"SubjectTag", "PartnerTag", "family_id"}
    if links.empty or not required.issubset(set(links.columns)):
        return {}

    stacked = pd.concat(
        [
            links[["SubjectTag", "family_id"]].rename(columns={"SubjectTag": "person_id"}),
            links[["PartnerTag", "family_id"]].rename(columns={"PartnerTag": "person_id"}),
        ],
        ignore_index=True,
    )
    stacked["person_id"] = pd.to_numeric(stacked["person_id"], errors="coerce")
    stacked = stacked.dropna(subset=["person_id", "family_id"]).copy()
    if stacked.empty:
        return {}

    stacked["person_id"] = stacked["person_id"].astype(int)
    stacked["family_id"] = stacked["family_id"].astype(str)
    lookup: dict[int, str] = {}
    for pid, grp in stacked.groupby("person_id"):
        vals = grp["family_id"].dropna().astype(str)
        if vals.empty:
            continue
        lookup[int(pid)] = vals.mode().iloc[0]
    return lookup


def _family_ids(root: Path, cohort: str, df: pd.DataFrame) -> pd.Series:
    if cohort == "nlsy79":
        lookup = _nlsy79_family_lookup(root)
        pids = pd.to_numeric(df.get("person_id"), errors="coerce")
        fam = pids.map(lambda x: lookup.get(int(x), None) if pd.notna(x) else None)
        return fam.where(pd.notna(fam), pids.map(lambda x: f"singleton_{int(x)}" if pd.notna(x) else "singleton_missing")).astype(str)

    if cohort == "nlsy97":
        fam = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
        if "R9708601" in df.columns:
            fam = pd.to_numeric(df["R9708601"], errors="coerce")
        if "R9708602" in df.columns:
            fallback = pd.to_numeric(df["R9708602"], errors="coerce")
            fam = fam.where(pd.notna(fam), fallback)
        fam = fam.map(lambda x: f"hh_{int(x)}" if pd.notna(x) else pd.NA)
        pids = pd.to_numeric(df.get("person_id"), errors="coerce")
        return fam.where(pd.notna(fam), pids.map(lambda x: f"singleton_{int(x)}" if pd.notna(x) else "singleton_missing")).astype(str)

    # CNLSY: prefer MPUBID if present; otherwise fallback to singleton person IDs.
    if "MPUBID" in df.columns:
        fam = pd.to_numeric(df["MPUBID"], errors="coerce")
        fam = fam.map(lambda x: f"mpubid_{int(x)}" if pd.notna(x) else pd.NA)
        pids = pd.to_numeric(df.get("person_id"), errors="coerce")
        return fam.where(pd.notna(fam), pids.map(lambda x: f"singleton_{int(x)}" if pd.notna(x) else "singleton_missing")).astype(str)

    pids = pd.to_numeric(df.get("person_id"), errors="coerce")
    return pids.map(lambda x: f"singleton_{int(x)}" if pd.notna(x) else "singleton_missing").astype(str)


def _sex_labels(series: pd.Series) -> pd.Series:
    vals = series.astype(str).str.strip().str.lower()
    out = pd.Series(["unknown"] * len(vals), index=vals.index, dtype="object")
    out[vals.isin({"m", "male", "1", "man", "boy"})] = "male"
    out[vals.isin({"f", "female", "2", "woman", "girl"})] = "female"
    return out


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=1)
    if pd.isna(sd) or float(sd) <= 0.0:
        return pd.Series([np.nan] * len(s), index=s.index)
    return (s - mean) / sd


def _composite_score(df: pd.DataFrame, indicators: list[str]) -> pd.Series:
    existing = [col for col in indicators if col in df.columns]
    if not existing:
        return pd.Series([np.nan] * len(df), index=df.index)
    z = pd.DataFrame({col: _zscore(df[col]) for col in existing}, index=df.index)
    return z.mean(axis=1, skipna=False)


def _estimate_proxy_stats(df: pd.DataFrame) -> tuple[float | None, float | None]:
    male = pd.to_numeric(df.loc[df["sex_label"] == "male", "g_proxy"], errors="coerce").dropna()
    female = pd.to_numeric(df.loc[df["sex_label"] == "female", "g_proxy"], errors="coerce").dropna()
    if len(male) < 2 or len(female) < 2:
        return None, None
    male_var = float(male.var(ddof=1))
    female_var = float(female.var(ddof=1))
    if male_var <= 0.0 or female_var <= 0.0:
        return None, None
    pooled_var = (((len(male) - 1) * male_var) + ((len(female) - 1) * female_var)) / float(len(male) + len(female) - 2)
    if not math.isfinite(pooled_var) or pooled_var <= 0.0:
        return None, None
    pooled_sd = math.sqrt(pooled_var)
    d_g = float(male.mean() - female.mean()) / pooled_sd
    vr = float(male_var / female_var)
    return d_g, vr


def _estimate_stats(df: pd.DataFrame) -> tuple[float | None, float | None]:
    # Backward-compatible alias retained for existing tests and callers.
    return _estimate_proxy_stats(df)


def _cluster_bootstrap_indices(
    family_labels: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> list[np.ndarray]:
    families = pd.Series(family_labels).dropna().unique().tolist()
    if not families:
        return []

    family_rows: list[np.ndarray] = []
    for fam in families:
        idx = np.flatnonzero(family_labels == fam)
        if idx.size > 0:
            family_rows.append(idx)
    if not family_rows:
        return []

    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    n_fam = len(family_rows)
    for _ in range(n_boot):
        chosen = rng.integers(0, n_fam, size=n_fam)
        row_idx = np.concatenate([family_rows[i] for i in chosen])
        out.append(row_idx)
    return out


def _cluster_bootstrap_proxy(df: pd.DataFrame, *, n_boot: int, seed: int) -> tuple[list[float], list[float]]:
    family_labels = df["__family_id"].astype(str).to_numpy()
    row_idx_list = _cluster_bootstrap_indices(family_labels, n_boot=n_boot, seed=seed)
    if not row_idx_list:
        return [], []

    g_vals = pd.to_numeric(df["g_proxy"], errors="coerce").to_numpy(dtype=float)
    sex_vals = df["sex_label"].astype(str).to_numpy()

    d_samples: list[float] = []
    vr_samples: list[float] = []
    for row_idx in row_idx_list:
        if row_idx.size == 0:
            continue
        g = g_vals[row_idx]
        s = sex_vals[row_idx]
        valid = np.isfinite(g)
        if not np.any(valid):
            continue
        male = g[(s == "male") & valid]
        female = g[(s == "female") & valid]
        if male.size < 2 or female.size < 2:
            continue
        male_var = float(np.var(male, ddof=1))
        female_var = float(np.var(female, ddof=1))
        if male_var <= 0.0 or female_var <= 0.0:
            continue
        d_samples.append(float(np.mean(male) - np.mean(female)))
        vr_samples.append(float(male_var / female_var))
    return d_samples, vr_samples


def _normalize_group(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    token = str(value).strip().lower()
    if token in {"f", "female", "2", "w", "woman", "girl"}:
        return "female"
    if token in {"m", "male", "1", "boy", "man"}:
        return "male"
    return token


def _group_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        numeric_f = float(numeric)
        if numeric_f.is_integer():
            return str(int(numeric_f))
        return str(numeric_f)
    return str(value).strip()


def _infer_group_labels(groups: list[str], means_by_group: dict[str, float | None], reference_group: str) -> tuple[str | None, str | None]:
    if not groups:
        return None, None
    norm_reference = _normalize_group(reference_group)
    female_candidates = [g for g in groups if _normalize_group(g) == norm_reference]
    if len(female_candidates) == 1:
        female = female_candidates[0]
    else:
        female_tokens = {"female", "f"}
        female_candidates = [g for g in groups if _normalize_group(g) in female_tokens]
        male_tokens = {"male", "m"}
        male_candidates = [g for g in groups if _normalize_group(g) in male_tokens]
        if len(female_candidates) == 1:
            female = female_candidates[0]
        elif len(male_candidates) == 1 and len(groups) == 2:
            female = next((g for g in groups if g != male_candidates[0]), None)
        else:
            female = None
            if means_by_group:
                for g, mean in means_by_group.items():
                    if mean is not None and abs(float(mean)) <= 1e-8:
                        female = g
                        break
            if female is None and len(groups) == 2:
                female = groups[0]
    if female is not None:
        male = next((g for g in groups if g != female), None)
    else:
        if len(groups) == 2:
            male, female = groups[0], groups[1]
        else:
            return None, None
    return female, male


def _pick_step(df: pd.DataFrame, preferred_steps: tuple[str, ...]) -> pd.DataFrame:
    if df.empty or "model_step" not in df.columns:
        return df
    for step in preferred_steps:
        rows = df[df["model_step"] == step]
        if not rows.empty:
            return rows
    return df


def _extract_sem_estimands(
    *,
    params_path: Path,
    sex_estimands_path: Path | None = None,
    cohort: str,
    reference_group: str,
) -> tuple[float | None, float | None, str | None]:
    if sex_estimands_path is not None and sex_estimands_path.exists():
        sex_estimands = pd.read_csv(sex_estimands_path, low_memory=False)
        if not sex_estimands.empty:
            if "cohort" in sex_estimands.columns:
                cohort_rows = sex_estimands[sex_estimands["cohort"].astype(str).str.strip() == cohort]
                if not cohort_rows.empty:
                    sex_estimands = cohort_rows
            factor_name = "g_cnlsy" if cohort == "cnlsy" else "g"
            if "factor" in sex_estimands.columns:
                factor_rows = sex_estimands[sex_estimands["factor"].astype(str).str.strip() == factor_name]
                if not factor_rows.empty:
                    sex_estimands = factor_rows
            var_subset = _pick_step(sex_estimands, ("metric", "scalar", "strict", "configural"))
            mean_subset = _pick_step(sex_estimands, ("scalar", "strict", "metric", "configural"))
            if not mean_subset.empty and not var_subset.empty:
                mean_row = mean_subset.iloc[0]
                var_row = var_subset.iloc[0]
                male_mean = _safe_float(mean_row.get("mean_male"))
                female_mean = _safe_float(mean_row.get("mean_female"))
                male_var = _safe_float(var_row.get("var_male"))
                female_var = _safe_float(var_row.get("var_female"))
                if (
                    male_mean is not None
                    and female_mean is not None
                    and male_var is not None
                    and female_var is not None
                    and male_var > 0.0
                    and female_var > 0.0
                ):
                    return float(male_mean - female_mean), float(male_var / female_var), None

    if not params_path.exists():
        return None, None, "missing_params"

    params = pd.read_csv(params_path, low_memory=False)
    if params.empty:
        return None, None, "empty_params"

    params = params.copy()
    if "cohort" in params.columns:
        cohort_rows = params[params["cohort"].astype(str).str.strip() == cohort]
        if not cohort_rows.empty:
            params = cohort_rows

    factor_name = "g_cnlsy" if cohort == "cnlsy" else "g"

    mean_subset = _pick_step(params, ("scalar", "strict", "metric", "configural"))
    var_subset = _pick_step(params, ("metric", "scalar", "strict", "configural"))

    mean_rows = mean_subset.loc[
        (mean_subset.get("lhs") == factor_name) & (mean_subset.get("op") == "~1"),
        ["group", "est"],
    ].copy()
    var_rows = var_subset.loc[
        (var_subset.get("lhs") == factor_name) & (var_subset.get("rhs") == factor_name) & (var_subset.get("op") == "~~"),
        ["group", "est"],
    ].copy()
    if mean_rows.empty or var_rows.empty:
        return None, None, "missing_factor_rows"

    mean_rows["group_key"] = mean_rows["group"].map(_group_key)
    var_rows["group_key"] = var_rows["group"].map(_group_key)
    means = {str(r["group_key"]): _safe_float(r["est"]) for _, r in mean_rows.iterrows()}
    vars_ = {str(r["group_key"]): _safe_float(r["est"]) for _, r in var_rows.iterrows()}

    groups = sorted(set(means.keys()) | set(vars_.keys()))
    female_group, male_group = _infer_group_labels(groups, means, reference_group)
    if female_group is None or male_group is None:
        return None, None, "cannot_identify_groups"

    male_mean = means.get(male_group)
    female_mean = means.get(female_group)
    male_var = vars_.get(male_group)
    female_var = vars_.get(female_group)
    if male_mean is None or female_mean is None:
        return None, None, "missing_group_means"
    if male_var is None or female_var is None:
        return None, None, "missing_group_vars"
    if male_var <= 0.0 or female_var <= 0.0:
        return None, None, "nonpositive_group_var"

    d_g = float(male_mean - female_mean)
    vr = float(male_var / female_var)
    return d_g, vr, None


def _sem_model_syntax(cohort: str, models_cfg: dict[str, Any]) -> str:
    return cnlsy_model_syntax(models_cfg) if cohort == "cnlsy" else hierarchical_model_syntax(models_cfg)


def _sem_observed_tests(cohort: str, models_cfg: dict[str, Any]) -> list[str]:
    if cohort == "cnlsy":
        return [str(x) for x in models_cfg.get("cnlsy_single_factor", [])]
    return hierarchical_subtests(models_cfg)


def _sem_group_col(cohort_cfg: dict[str, Any]) -> str:
    sample_cfg = cohort_cfg.get("sample_construct", {}) if isinstance(cohort_cfg.get("sample_construct", {}), dict) else {}
    return str(sample_cfg.get("sex_col", "sex"))


def _sem_std_lv(cohort_cfg: dict[str, Any]) -> bool:
    sem_fit_cfg = cohort_cfg.get("sem_fit", {}) if isinstance(cohort_cfg.get("sem_fit", {}), dict) else {}
    return bool(sem_fit_cfg.get("std_lv", True))


def _sem_partial_intercepts(cohort: str, models_cfg: dict[str, Any]) -> list[str]:
    inv_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    partial_cfg = inv_cfg.get("partial_intercepts", {})
    if not isinstance(partial_cfg, dict):
        return []
    values = partial_cfg.get(cohort, [])
    if not isinstance(values, list):
        return []
    return [str(x) for x in values if str(x).strip()]


def _thread_limited_env(thread_limit: int | None) -> dict[str, str] | None:
    if thread_limit is None:
        return None
    limit = int(thread_limit)
    if limit <= 0:
        return None
    env = os.environ.copy()
    value = str(limit)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = value
    return env


def _run_sem_refit_once(
    *,
    root: Path,
    cohort: str,
    data: pd.DataFrame,
    cohort_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    work_dir: Path,
    timeout_seconds: float,
    thread_limit: int | None = None,
) -> tuple[float | None, float | None, str | None]:
    rscript = rscript_path()
    if rscript is None:
        return None, None, "rscript_missing"

    sem_script = root / "scripts" / "sem_fit.R"
    if not sem_script.exists():
        return None, None, "sem_fit_script_missing"

    work_dir.mkdir(parents=True, exist_ok=True)
    data_path = work_dir / "sem_input.csv"
    request_path = work_dir / "request.json"
    model_path = work_dir / "model.lavaan"

    data.to_csv(data_path, index=False)
    model_path.write_text(_sem_model_syntax(cohort, models_cfg).strip() + "\n", encoding="utf-8")

    request_payload = {
        "cohort": cohort,
        "data_csv": data_path.name,
        "group_col": _sem_group_col(cohort_cfg),
        "reference_group": str(models_cfg.get("reference_group", "female")),
        "estimator": "MLR",
        "missing": "fiml",
        # Bootstrap inference uses the bootstrap distribution for SE/CI; force standard
        # SE/test inside lavaan to avoid robust-MLR edge cases in small resamples.
        "force_standard_se": True,
        "std_lv": _sem_std_lv(cohort_cfg),
        "invariance_steps": list(DEFAULT_INVARIANCE_STEPS),
        "partial_intercepts": _sem_partial_intercepts(cohort, models_cfg),
        "observed_tests": _sem_observed_tests(cohort, models_cfg),
    }
    request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_failure_logs(*, label: str, stdout: str | None, stderr: str | None) -> None:
        try:
            if stdout:
                (work_dir / f"{label}_stdout.txt").write_text(str(stdout), encoding="utf-8", errors="ignore")
            if stderr:
                (work_dir / f"{label}_stderr.txt").write_text(str(stderr), encoding="utf-8", errors="ignore")
        except Exception:
            return

    try:
        subprocess.run(
            [str(rscript), str(sem_script), "--request", str(request_path), "--outdir", str(work_dir)],
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=_thread_limited_env(thread_limit),
        )
    except subprocess.TimeoutExpired as exc:
        _write_failure_logs(label="sem_fit", stdout=getattr(exc, "stdout", None), stderr=getattr(exc, "stderr", None))
        return None, None, "sem_refit_timeout"
    except subprocess.CalledProcessError as exc:
        _write_failure_logs(label="sem_fit", stdout=exc.stdout, stderr=exc.stderr)
        return None, None, f"sem_refit_failed:rc={exc.returncode};stderr=sem_fit_stderr.txt"

    reference_group = str(models_cfg.get("reference_group", "female"))
    return _extract_sem_estimands(
        params_path=work_dir / "params.csv",
        sex_estimands_path=work_dir / "sex_group_estimands.csv",
        cohort=cohort,
        reference_group=reference_group,
    )


def _summarize_bootstrap(
    *,
    d_samples: list[float],
    vr_samples: list[float],
    base_d: float,
    base_vr: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    d_arr = np.asarray(d_samples, dtype=float)
    vr_arr = np.asarray(vr_samples, dtype=float)
    log_vr_arr = np.log(vr_arr)

    se_d = float(np.std(d_arr, ddof=1)) if len(d_arr) > 1 else np.nan
    se_log_vr = float(np.std(log_vr_arr, ddof=1)) if len(log_vr_arr) > 1 else np.nan

    d_low, d_high = float(np.percentile(d_arr, 2.5)), float(np.percentile(d_arr, 97.5))
    log_vr_low, log_vr_high = float(np.percentile(log_vr_arr, 2.5)), float(np.percentile(log_vr_arr, 97.5))
    vr_low, vr_high = float(np.exp(log_vr_low)), float(np.exp(log_vr_high))

    mean_row = {
        "status": "computed",
        "reason": pd.NA,
        "d_g": float(base_d),
        "SE_d_g": se_d,
        "ci_low_d_g": d_low,
        "ci_high_d_g": d_high,
        "IQ_diff": float(base_d) * 15.0,
        "SE": se_d * 15.0,
        "ci_low": d_low * 15.0,
        "ci_high": d_high * 15.0,
    }
    vr_row = {
        "status": "computed",
        "reason": pd.NA,
        "VR_g": float(base_vr),
        "SE_logVR": se_log_vr,
        "ci_low": vr_low,
        "ci_high": vr_high,
    }
    return mean_row, vr_row


def _reuse_existing_sem_rep(
    *,
    cohort: str,
    cohort_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    rep_dir: Path,
) -> tuple[float | None, float | None]:
    params_path = rep_dir / "params.csv"
    sex_estimands_path = rep_dir / "sex_group_estimands.csv"
    if not params_path.exists() and not sex_estimands_path.exists():
        return None, None
    reference_group = str(cohort_cfg.get("reference_group", models_cfg.get("reference_group", "female")))
    d_rep, vr_rep, reason = _extract_sem_estimands(
        params_path=params_path,
        sex_estimands_path=sex_estimands_path if sex_estimands_path.exists() else None,
        cohort=cohort,
        reference_group=reference_group,
    )
    if d_rep is None or vr_rep is None or reason is not None:
        return None, None
    return float(d_rep), float(vr_rep)


def run_inference_bootstrap(
    *,
    root: Path,
    variant_token: str,
    artifact_prefix: str = "g",
    n_bootstrap: int,
    seed: int,
    engine: str = DEFAULT_ENGINE,
    min_success_share: float = DEFAULT_MIN_SUCCESS_SHARE,
    sem_timeout_seconds: float = 60.0,
    sem_jobs: int = 1,
    sem_threads_per_job: int | None = None,
    skip_successful: bool = False,
    resume_existing_reps: bool = False,
    cohorts: list[str] | None = None,
) -> dict[str, Any]:
    if variant_token != "family_bootstrap":
        raise ValueError("variant-token must be family_bootstrap")
    if n_bootstrap < 50:
        raise ValueError("n-bootstrap must be >= 50")
    if engine not in {"proxy", "sem_refit"}:
        raise ValueError("engine must be one of: proxy, sem_refit")
    if resume_existing_reps and engine != "sem_refit":
        raise ValueError("resume-existing-reps is only supported when engine=sem_refit")
    if min_success_share <= 0 or min_success_share > 1:
        raise ValueError("min-success-share must be in (0, 1]")
    if sem_jobs < 1:
        raise ValueError("sem-jobs must be >= 1")
    if sem_threads_per_job is not None and sem_threads_per_job < 1:
        raise ValueError("sem-threads-per-job must be >= 1 when provided")
    if sem_threads_per_job is None and sem_jobs > 1:
        sem_threads_per_job = 1

    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    processed_dir = _resolve_path(paths_cfg.get("processed_dir", "data/processed"), root)
    outputs_dir = _resolve_path(paths_cfg.get("outputs_dir", "outputs"), root)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir = outputs_dir / "model_fits" / "bootstrap_inference"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = str(artifact_prefix or "g").strip()
    if not artifact_prefix:
        raise ValueError("artifact_prefix must be non-empty.")
    g_mean_path = tables_dir / f"{artifact_prefix}_mean_diff_{variant_token}.csv"
    g_vr_path = tables_dir / f"{artifact_prefix}_variance_ratio_{variant_token}.csv"
    existing_mean_rows = _read_inference_rows(g_mean_path) if skip_successful else pd.DataFrame()
    existing_vr_rows = _read_inference_rows(g_vr_path) if skip_successful else pd.DataFrame()
    if artifact_prefix == "g":
        prior_manifest_path = tables_dir / f"inference_rerun_manifest_{variant_token}.json"
    else:
        prior_manifest_path = tables_dir / f"inference_rerun_manifest_{artifact_prefix}_{variant_token}.json"
    prior_attempted: dict[str, int] = {}
    prior_success: dict[str, int] = {}
    prior_status: dict[str, str] = {}
    prior_reason: dict[str, str] = {}
    prior_success_share: dict[str, float] = {}
    if skip_successful and prior_manifest_path.exists():
        try:
            payload = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for row in (payload.get("cohort_details") or []) if isinstance(payload, dict) else []:
            if not isinstance(row, dict):
                continue
            cohort = str(row.get("cohort", "")).strip()
            if not cohort:
                continue
            attempted_raw = row.get("attempted")
            success_raw = row.get("success")
            share_raw = row.get("success_share")
            try:
                prior_attempted[cohort] = int(attempted_raw) if attempted_raw is not None else prior_attempted.get(cohort, 0)
            except (TypeError, ValueError):
                pass
            try:
                prior_success[cohort] = int(success_raw) if success_raw is not None else prior_success.get(cohort, 0)
            except (TypeError, ValueError):
                pass
            try:
                prior_success_share[cohort] = float(share_raw) if share_raw is not None else prior_success_share.get(cohort, 0.0)
            except (TypeError, ValueError):
                pass
            prior_status[cohort] = str(row.get("status", "")).strip()
            prior_reason[cohort] = str(row.get("reason", "")).strip()

    selected_cohorts = [c for c in COHORTS if cohorts is None or c in set(cohorts)]
    if not selected_cohorts:
        raise ValueError("No cohorts selected for inference bootstrap run.")

    g_mean_rows: list[dict[str, Any]] = []
    g_vr_rows: list[dict[str, Any]] = []
    cohort_manifest: list[dict[str, Any]] = []
    skipped_existing_cohorts: list[str] = []

    for idx, cohort in enumerate(selected_cohorts):
        cohort_cfg_path = root / "config" / f"{cohort}.yml"
        cohort_cfg = load_yaml(cohort_cfg_path) if cohort_cfg_path.exists() else {}

        if skip_successful:
            existing_mean_row = _cohort_row(existing_mean_rows, cohort)
            existing_vr_row = _cohort_row(existing_vr_rows, cohort)
            prior_ok = True
            if prior_manifest_path.exists():
                prior_ok = (
                    prior_attempted.get(cohort, 0) >= int(n_bootstrap)
                    and _status_is_computed(prior_status.get(cohort, ""))
                )
            if (
                existing_mean_row is not None
                and existing_vr_row is not None
                and _status_is_computed(existing_mean_row.get("status"))
                and _status_is_computed(existing_vr_row.get("status"))
                and prior_ok
            ):
                g_mean_rows.append(existing_mean_row)
                g_vr_rows.append(existing_vr_row)
                skipped_existing_cohorts.append(cohort)
                attempted = prior_attempted.get(cohort)
                success = prior_success.get(cohort)
                success_share = prior_success_share.get(cohort)
                cohort_manifest.append(
                    {
                        "cohort": cohort,
                        "status": "computed",
                        "reason": prior_reason.get(cohort, "") or existing_mean_row.get("reason", "reused_existing_computed"),
                        "attempted": int(attempted) if isinstance(attempted, int) and attempted > 0 else int(n_bootstrap),
                        "success": int(success) if isinstance(success, int) and success >= 0 else int(n_bootstrap),
                        "success_share": float(success_share) if isinstance(success_share, float) and success_share >= 0 else 1.0,
                        "skipped_existing": True,
                        "reused_from_outputs": True,
                    }
                )
                continue

        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        if not source_path.exists():
            g_mean_rows.append({"cohort": cohort, "status": "not_feasible", "reason": "missing_source", "d_g": pd.NA, "SE_d_g": pd.NA, "ci_low_d_g": pd.NA, "ci_high_d_g": pd.NA, "IQ_diff": pd.NA, "SE": pd.NA, "ci_low": pd.NA, "ci_high": pd.NA})
            g_vr_rows.append({"cohort": cohort, "status": "not_feasible", "reason": "missing_source", "VR_g": pd.NA, "SE_logVR": pd.NA, "ci_low": pd.NA, "ci_high": pd.NA})
            cohort_manifest.append({"cohort": cohort, "status": "not_feasible", "reason": "missing_source", "attempted": n_bootstrap, "success": 0, "success_share": 0.0})
            continue

        df = pd.read_csv(source_path, low_memory=False)
        if "sex" not in df.columns:
            g_mean_rows.append({"cohort": cohort, "status": "not_feasible", "reason": "missing_sex_column", "d_g": pd.NA, "SE_d_g": pd.NA, "ci_low_d_g": pd.NA, "ci_high_d_g": pd.NA, "IQ_diff": pd.NA, "SE": pd.NA, "ci_low": pd.NA, "ci_high": pd.NA})
            g_vr_rows.append({"cohort": cohort, "status": "not_feasible", "reason": "missing_sex_column", "VR_g": pd.NA, "SE_logVR": pd.NA, "ci_low": pd.NA, "ci_high": pd.NA})
            cohort_manifest.append({"cohort": cohort, "status": "not_feasible", "reason": "missing_sex_column", "attempted": n_bootstrap, "success": 0, "success_share": 0.0})
            continue

        indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", [])] if cohort == "cnlsy" else hierarchical_subtests(models_cfg)
        df = df.copy()
        df["sex_label"] = _sex_labels(df["sex"])
        df["g_proxy"] = _composite_score(df, indicators)
        df["__family_id"] = _family_ids(root, cohort, df)

        if engine == "proxy":
            base_d, base_vr = _estimate_proxy_stats(df)
            if base_d is None or base_vr is None:
                reason = "proxy_base_estimate_failed"
                g_mean_rows.append({"cohort": cohort, "status": "not_feasible", "reason": reason, "d_g": pd.NA, "SE_d_g": pd.NA, "ci_low_d_g": pd.NA, "ci_high_d_g": pd.NA, "IQ_diff": pd.NA, "SE": pd.NA, "ci_low": pd.NA, "ci_high": pd.NA})
                g_vr_rows.append({"cohort": cohort, "status": "not_feasible", "reason": reason, "VR_g": pd.NA, "SE_logVR": pd.NA, "ci_low": pd.NA, "ci_high": pd.NA})
                cohort_manifest.append({"cohort": cohort, "status": "not_feasible", "reason": reason, "attempted": n_bootstrap, "success": 0, "success_share": 0.0})
                continue
            d_samples, vr_samples = _cluster_bootstrap_proxy(df, n_boot=n_bootstrap, seed=seed + idx)
            success_count = min(len(d_samples), len(vr_samples))
            success_share = success_count / float(n_bootstrap)
            if success_count < max(25, int(math.ceil(min_success_share * n_bootstrap))):
                reason = f"bootstrap_success_below_threshold:{success_count}/{n_bootstrap}"
                g_mean_rows.append({"cohort": cohort, "status": "not_feasible", "reason": reason, "d_g": pd.NA, "SE_d_g": pd.NA, "ci_low_d_g": pd.NA, "ci_high_d_g": pd.NA, "IQ_diff": pd.NA, "SE": pd.NA, "ci_low": pd.NA, "ci_high": pd.NA})
                g_vr_rows.append({"cohort": cohort, "status": "not_feasible", "reason": reason, "VR_g": pd.NA, "SE_logVR": pd.NA, "ci_low": pd.NA, "ci_high": pd.NA})
                cohort_manifest.append({"cohort": cohort, "status": "not_feasible", "reason": reason, "attempted": n_bootstrap, "success": success_count, "success_share": success_share})
                continue
            mean_row, vr_row = _summarize_bootstrap(d_samples=d_samples, vr_samples=vr_samples, base_d=float(base_d), base_vr=float(base_vr))
            g_mean_rows.append({"cohort": cohort, **mean_row})
            g_vr_rows.append({"cohort": cohort, **vr_row})
            cohort_manifest.append({"cohort": cohort, "status": "computed", "reason": "", "attempted": n_bootstrap, "success": success_count, "success_share": success_share})
            continue

        # SEM-refit engine
        required_cols = [_sem_group_col(cohort_cfg), *_sem_observed_tests(cohort, models_cfg)]
        required_cols = [c for c in required_cols if c in df.columns]
        sem_df = df[required_cols].copy()
        sem_df["__family_id"] = df["__family_id"].astype(str)

        cohort_work = bootstrap_dir / cohort
        cohort_work.mkdir(parents=True, exist_ok=True)

        base_d, base_vr, base_reason = _run_sem_refit_once(
            root=root,
            cohort=cohort,
            data=sem_df.drop(columns=["__family_id"]),
            cohort_cfg=cohort_cfg,
            models_cfg=models_cfg,
            work_dir=cohort_work / "full_sample",
            timeout_seconds=sem_timeout_seconds,
            thread_limit=sem_threads_per_job,
        )
        if base_d is None or base_vr is None:
            reason = f"sem_base_estimate_failed:{base_reason or 'unknown'}"
            g_mean_rows.append({"cohort": cohort, "status": "not_feasible", "reason": reason, "d_g": pd.NA, "SE_d_g": pd.NA, "ci_low_d_g": pd.NA, "ci_high_d_g": pd.NA, "IQ_diff": pd.NA, "SE": pd.NA, "ci_low": pd.NA, "ci_high": pd.NA})
            g_vr_rows.append({"cohort": cohort, "status": "not_feasible", "reason": reason, "VR_g": pd.NA, "SE_logVR": pd.NA, "ci_low": pd.NA, "ci_high": pd.NA})
            cohort_manifest.append({"cohort": cohort, "status": "not_feasible", "reason": reason, "attempted": n_bootstrap, "success": 0, "success_share": 0.0})
            continue

        family_labels = sem_df["__family_id"].astype(str).to_numpy()
        row_idx_list = _cluster_bootstrap_indices(family_labels, n_boot=n_bootstrap, seed=seed + idx)
        d_samples: list[float] = []
        vr_samples: list[float] = []
        fail_reasons: Counter[str] = Counter()
        pending_reps: list[tuple[int, np.ndarray]] = []
        reused_rep_count = 0
        computed_rep_count = 0
        for rep_idx, row_idx in enumerate(row_idx_list):
            if row_idx.size == 0:
                fail_reasons["empty_resample"] += 1
                continue
            if resume_existing_reps:
                rep_dir = cohort_work / f"rep_{rep_idx:04d}"
                reused_d, reused_vr = _reuse_existing_sem_rep(
                    cohort=cohort,
                    cohort_cfg=cohort_cfg,
                    models_cfg=models_cfg,
                    rep_dir=rep_dir,
                )
                if reused_d is not None and reused_vr is not None:
                    d_samples.append(float(reused_d))
                    vr_samples.append(float(reused_vr))
                    reused_rep_count += 1
                    continue
            pending_reps.append((rep_idx, row_idx))

        def _run_rep(rep_idx: int, row_idx: np.ndarray) -> tuple[float | None, float | None, str | None]:
            try:
                boot = sem_df.iloc[row_idx].drop(columns=["__family_id"]).copy()
                return _run_sem_refit_once(
                    root=root,
                    cohort=cohort,
                    data=boot,
                    cohort_cfg=cohort_cfg,
                    models_cfg=models_cfg,
                    work_dir=cohort_work / f"rep_{rep_idx:04d}",
                    timeout_seconds=sem_timeout_seconds,
                    thread_limit=sem_threads_per_job,
                )
            except Exception:
                return None, None, "sem_refit_python_exception"

        if sem_jobs == 1:
            for rep_idx, row_idx in pending_reps:
                d_rep, vr_rep, rep_reason = _run_rep(rep_idx, row_idx)
                if d_rep is None or vr_rep is None:
                    fail_reasons[rep_reason or "sem_refit_unknown_failure"] += 1
                    continue
                d_samples.append(float(d_rep))
                vr_samples.append(float(vr_rep))
                computed_rep_count += 1
        else:
            with ThreadPoolExecutor(max_workers=sem_jobs) as executor:
                futures = [executor.submit(_run_rep, rep_idx, row_idx) for rep_idx, row_idx in pending_reps]
                for fut in as_completed(futures):
                    d_rep, vr_rep, rep_reason = fut.result()
                    if d_rep is None or vr_rep is None:
                        fail_reasons[rep_reason or "sem_refit_unknown_failure"] += 1
                        continue
                    d_samples.append(float(d_rep))
                    vr_samples.append(float(vr_rep))
                    computed_rep_count += 1

        success_count = min(len(d_samples), len(vr_samples))
        success_share = success_count / float(n_bootstrap) if n_bootstrap > 0 else 0.0
        required_success = int(math.ceil(min_success_share * n_bootstrap))
        if success_count < max(25, required_success):
            fail_desc = ";".join([f"{k}:{v}" for k, v in sorted(fail_reasons.items())])
            reason = f"bootstrap_success_below_threshold:{success_count}/{n_bootstrap}" + (f";{fail_desc}" if fail_desc else "")
            g_mean_rows.append({"cohort": cohort, "status": "not_feasible", "reason": reason, "d_g": pd.NA, "SE_d_g": pd.NA, "ci_low_d_g": pd.NA, "ci_high_d_g": pd.NA, "IQ_diff": pd.NA, "SE": pd.NA, "ci_low": pd.NA, "ci_high": pd.NA})
            g_vr_rows.append({"cohort": cohort, "status": "not_feasible", "reason": reason, "VR_g": pd.NA, "SE_logVR": pd.NA, "ci_low": pd.NA, "ci_high": pd.NA})
            cohort_manifest.append({
                "cohort": cohort,
                "status": "not_feasible",
                "reason": reason,
                "attempted": n_bootstrap,
                "success": success_count,
                "success_share": success_share,
                "fail_reasons": dict(fail_reasons),
                "reused_reps": int(reused_rep_count),
                "computed_reps": int(computed_rep_count),
                "reused_rep_count": int(reused_rep_count),
                "computed_rep_count": int(computed_rep_count),
            })
            continue

        mean_row, vr_row = _summarize_bootstrap(d_samples=d_samples, vr_samples=vr_samples, base_d=float(base_d), base_vr=float(base_vr))
        g_mean_rows.append({"cohort": cohort, **mean_row})
        g_vr_rows.append({"cohort": cohort, **vr_row})
        cohort_manifest.append(
            {
                "cohort": cohort,
                "status": "computed",
                "reason": "",
                "attempted": n_bootstrap,
                "success": success_count,
                "success_share": success_share,
                "fail_reasons": dict(fail_reasons),
                "reused_reps": int(reused_rep_count),
                "computed_reps": int(computed_rep_count),
                "reused_rep_count": int(reused_rep_count),
                "computed_rep_count": int(computed_rep_count),
            }
        )

    if not g_mean_rows or not g_vr_rows:
        raise ValueError("No bootstrap inference rows could be produced.")

    pd.DataFrame(g_mean_rows).to_csv(g_mean_path, index=False)
    pd.DataFrame(g_vr_rows).to_csv(g_vr_path, index=False)

    manifest = {
        "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "variant_token": variant_token,
        "engine": engine,
        "artifact_prefix": artifact_prefix,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "min_success_share": float(min_success_share),
        "sem_timeout_seconds": float(sem_timeout_seconds),
        "sem_jobs": int(sem_jobs),
        "sem_threads_per_job": int(sem_threads_per_job) if sem_threads_per_job is not None else None,
        "skip_successful_requested": bool(skip_successful),
        "resume_existing_reps_requested": bool(resume_existing_reps),
        "skipped_existing_cohorts": sorted(skipped_existing_cohorts),
        "point_estimate_source": "full_sample_fit",
        "point_estimate_detail": (
            "d_g and VR_g are computed from full-sample fits; bootstrap is used only for SE/CI estimation"
        ),
        "artifacts": {
            "g_mean_diff": relative_path(root, g_mean_path),
            "g_variance_ratio": relative_path(root, g_vr_path),
        },
        "cohorts": sorted({str(row["cohort"]) for row in g_mean_rows}),
        "cohort_details": cohort_manifest,
    }
    dump_json(prior_manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run family-cluster bootstrap inference rerun.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--variant-token", required=True, choices=("family_bootstrap",))
    parser.add_argument(
        "--artifact-prefix",
        default="g",
        help=(
            "Artifact filename prefix for outputs/tables. "
            "Default 'g' writes g_mean_diff_* and g_variance_ratio_* tables. "
            "Use 'g_proxy' to write g_proxy_mean_diff_* tables, etc."
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=150, help="Bootstrap repetitions (>=50). Default 150 for development; use 499 for publication-quality results.")
    parser.add_argument("--seed", type=int, default=20260221, help="Random seed.")
    parser.add_argument(
        "--engine",
        choices=("proxy", "sem_refit"),
        default=DEFAULT_ENGINE,
        help="Bootstrap inference engine.",
    )
    parser.add_argument(
        "--min-success-share",
        type=float,
        default=DEFAULT_MIN_SUCCESS_SHARE,
        help="Minimum required converged bootstrap share per cohort.",
    )
    parser.add_argument(
        "--sem-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-refit timeout in seconds for sem_refit engine.",
    )
    parser.add_argument(
        "--sem-jobs",
        type=int,
        default=1,
        help="Concurrent SEM-refit workers per cohort (sem_refit engine only).",
    )
    parser.add_argument(
        "--sem-threads-per-job",
        type=int,
        help="Thread cap per SEM worker. Defaults to 1 when --sem-jobs > 1, else inherited env.",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        choices=COHORTS,
        help="Optional cohort selector (repeatable). Defaults to all cohorts.",
    )
    parser.add_argument(
        "--skip-successful",
        action="store_true",
        help="Reuse cohorts already present as computed in existing stage-20 outputs.",
    )
    parser.add_argument(
        "--resume-existing-reps",
        action="store_true",
        help=(
            "SEM-refit engine only: reuse parseable existing rep_#### outputs "
            "(params.csv + sex_group_estimands.csv) instead of recomputing."
        ),
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        manifest = run_inference_bootstrap(
            root=root,
            variant_token=str(args.variant_token),
            artifact_prefix=str(args.artifact_prefix),
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed),
            engine=str(args.engine),
            min_success_share=float(args.min_success_share),
            sem_timeout_seconds=float(args.sem_timeout_seconds),
            sem_jobs=int(args.sem_jobs),
            sem_threads_per_job=int(args.sem_threads_per_job) if args.sem_threads_per_job is not None else None,
            skip_successful=bool(args.skip_successful),
            resume_existing_reps=bool(args.resume_existing_reps),
            cohorts=[str(c) for c in args.cohort] if args.cohort else None,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] inference bootstrap rerun complete for variant={manifest['variant_token']}")
    print(f"[ok] wrote {manifest['artifacts']['g_mean_diff']}")
    print(f"[ok] wrote {manifest['artifacts']['g_variance_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
