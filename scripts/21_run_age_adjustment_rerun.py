#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
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
from nls_pipeline.sampling import standardize_series

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}
METHOD_BY_TOKEN: dict[str, tuple[str, str]] = {
    "quadratic_within_sex": ("quadratic", "within_sex"),
    "cubic_pooled": ("cubic", "pooled"),
    "cubic_within_sex": ("cubic", "within_sex"),
    "spline_pooled": ("spline", "pooled"),
    "spline_within_sex": ("spline", "within_sex"),
}


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _as_label(value: Any) -> str:
    return str(value)


def _sample_cfg(cohort_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = cohort_cfg.get("sample_construct", {})
    return cfg if isinstance(cfg, dict) else {}


def _default_subtests(cohort: str, models_cfg: dict[str, Any]) -> list[str]:
    if cohort == "cnlsy":
        return [_as_label(x) for x in models_cfg.get("cnlsy_single_factor", [])]
    groups = models_cfg.get("hierarchical_factors", {})
    ordered: list[str] = []
    for key in ("speed", "math", "verbal", "technical"):
        for val in groups.get(key, []):
            sval = _as_label(val)
            if sval not in ordered:
                ordered.append(sval)
    return ordered


def _normalize_sex(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"f", "female", "2", "woman", "girl"}:
        return "female"
    if token in {"m", "male", "1", "man", "boy"}:
        return "male"
    return token


def _design_matrix(x: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "quadratic":
        design = np.column_stack([np.ones(len(x)), x, x**2])
        return design, np.array([])
    if method == "cubic":
        design = np.column_stack([np.ones(len(x)), x, x**2, x**3])
        return design, np.array([])

    unique = np.unique(x)
    if len(unique) < 6:
        design = np.column_stack([np.ones(len(x)), x, x**2, x**3])
        return design, np.array([])
    knots = np.quantile(unique, [0.25, 0.5, 0.75])
    spline_terms = [np.maximum(0.0, x - knot) ** 3 for knot in knots]
    design = np.column_stack([np.ones(len(x)), x, x**2, x**3, *spline_terms])
    return design, knots


def _fit_residual_model(y: pd.Series, x: pd.Series, method: str) -> tuple[pd.Series, dict[str, Any]]:
    work = pd.DataFrame({"y": y, "x": x}).dropna().copy()
    residuals = pd.Series(np.nan, index=y.index, dtype=float)
    if work.empty:
        return residuals, {
            "n_used": 0,
            "r2": np.nan,
            "beta0": np.nan,
            "beta1": np.nan,
            "beta2": np.nan,
            "resid_sd": np.nan,
            "outliers_3sd": 0,
            "method_detail": "empty",
        }

    x_arr = work["x"].astype(float).to_numpy()
    y_arr = work["y"].astype(float).to_numpy()
    design, knots = _design_matrix(x_arr, method)
    beta, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    fitted = design @ beta
    resid = y_arr - fitted

    residuals.loc[work.index] = resid
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    resid_sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    outliers_3sd = 0 if resid_sd == 0.0 else int(np.sum(np.abs(resid) > (3.0 * resid_sd)))

    return residuals, {
        "n_used": int(len(work)),
        "r2": r2,
        "beta0": float(beta[0]) if len(beta) > 0 else np.nan,
        "beta1": float(beta[1]) if len(beta) > 1 else np.nan,
        "beta2": float(beta[2]) if len(beta) > 2 else np.nan,
        "resid_sd": resid_sd,
        "outliers_3sd": outliers_3sd,
        "method_detail": f"{method}({','.join([f'{k:.3f}' for k in knots])})" if len(knots) else method,
    }


def _combine_group_diags(diags: list[dict[str, Any]], residuals: pd.Series) -> dict[str, Any]:
    if not diags:
        return {
            "n_used": 0,
            "r2": np.nan,
            "beta0": np.nan,
            "beta1": np.nan,
            "beta2": np.nan,
            "resid_sd": np.nan,
            "outliers_3sd": 0,
            "method_detail": "within_sex_empty",
        }

    n_total = int(sum(int(d.get("n_used", 0)) for d in diags))
    if n_total <= 0:
        return {
            "n_used": 0,
            "r2": np.nan,
            "beta0": np.nan,
            "beta1": np.nan,
            "beta2": np.nan,
            "resid_sd": np.nan,
            "outliers_3sd": 0,
            "method_detail": "within_sex_empty",
        }

    weighted_r2_sum = 0.0
    for d in diags:
        n = int(d.get("n_used", 0))
        r2 = pd.to_numeric(pd.Series([d.get("r2")]), errors="coerce").iloc[0]
        if n > 0 and pd.notna(r2):
            weighted_r2_sum += float(r2) * float(n)
    weighted_r2 = weighted_r2_sum / float(n_total) if n_total > 0 else np.nan

    used_resid = pd.to_numeric(residuals, errors="coerce").dropna()
    resid_sd = float(used_resid.std(ddof=1)) if len(used_resid) > 1 else 0.0
    outliers_3sd = 0 if resid_sd == 0.0 else int((used_resid.abs() > 3.0 * resid_sd).sum())

    return {
        "n_used": n_total,
        "r2": weighted_r2,
        "beta0": np.nan,
        "beta1": np.nan,
        "beta2": np.nan,
        "resid_sd": resid_sd,
        "outliers_3sd": outliers_3sd,
        "method_detail": "within_sex",
    }


def _variant_residualize(
    *,
    values: pd.Series,
    predictor: pd.Series,
    sex: pd.Series,
    method: str,
    mode: str,
) -> tuple[pd.Series, dict[str, Any]]:
    if mode == "pooled":
        return _fit_residual_model(values, predictor, method)

    residuals = pd.Series(np.nan, index=values.index, dtype=float)
    group_diags: list[dict[str, Any]] = []
    sex_norm = sex.map(_normalize_sex)
    for grp in ["female", "male"]:
        mask = sex_norm == grp
        if int(mask.sum()) == 0:
            continue
        grp_resid, grp_diag = _fit_residual_model(values.loc[mask], predictor.loc[mask], method)
        residuals.loc[grp_resid.index] = grp_resid
        group_diags.append(grp_diag)
    combined = _combine_group_diags(group_diags, residuals)
    return residuals, combined


def run_age_adjustment_rerun(
    *,
    root: Path,
    cohort: str,
    variant_token: str,
) -> dict[str, Any]:
    token = str(variant_token).strip()
    if token not in METHOD_BY_TOKEN:
        raise ValueError(f"Unsupported variant token: {token}")
    method, mode = METHOD_BY_TOKEN[token]

    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    cohort_cfg = load_yaml(root / COHORT_CONFIGS[cohort])
    sample_cfg = _sample_cfg(cohort_cfg)

    processed_dir = _resolve_path(paths_cfg.get("processed_dir", "data/processed"), root)
    outputs_dir = _resolve_path(paths_cfg.get("outputs_dir", "outputs"), root)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    source_path = processed_dir / f"{cohort}_cfa.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing cohort source file: {source_path}")

    df = pd.read_csv(source_path, low_memory=False)
    sex_col = str(sample_cfg.get("sex_col", "sex"))
    predictor_col = str(sample_cfg.get("age_resid_col", sample_cfg.get("age_col", "age")))
    if sex_col not in df.columns:
        raise ValueError(f"{cohort}: sex column missing in source data: {sex_col}")
    if predictor_col not in df.columns:
        raise ValueError(f"{cohort}: predictor column missing in source data: {predictor_col}")

    subtests = [_as_label(x) for x in sample_cfg.get("subtests", _default_subtests(cohort, models_cfg))]
    missing = [c for c in subtests if c not in df.columns]
    if missing:
        raise ValueError(f"{cohort}: missing subtests for variant residualization: {missing}")

    standardize_output = bool(sample_cfg.get("standardize_output", False))
    out = df.copy()
    diagnostics: list[dict[str, Any]] = []
    for subtest in subtests:
        resid, diag = _variant_residualize(
            values=out[subtest],
            predictor=out[predictor_col],
            sex=out[sex_col],
            method=method,
            mode=mode,
        )
        if standardize_output:
            resid = standardize_series(resid)
        out[subtest] = resid
        diagnostics.append(
            {
                "cohort": cohort,
                "subtest": subtest,
                "predictor_col": predictor_col,
                "n_used": int(diag["n_used"]),
                "r2": diag["r2"],
                "beta0": diag["beta0"],
                "beta1": diag["beta1"],
                "beta2": diag["beta2"],
                "resid_sd": diag["resid_sd"],
                "outliers_3sd": int(diag["outliers_3sd"]),
                "age_adjustment": method,
                "residualization_mode": mode,
                "variant_token": token,
                "method_detail": diag["method_detail"],
            }
        )

    resid_path = processed_dir / f"{cohort}_cfa_resid_{token}.csv"
    out.to_csv(resid_path, index=False)

    diag_path = tables_dir / f"residualization_diagnostics_{cohort}_{token}.csv"
    pd.DataFrame(diagnostics).to_csv(diag_path, index=False)

    manifest = {
        "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "cohort": cohort,
        "variant_token": token,
        "age_adjustment": method,
        "residualization_mode": mode,
        "source_path": relative_path(root, source_path),
        "residualized_path": relative_path(root, resid_path),
        "diagnostics_path": relative_path(root, diag_path),
        "n_rows": int(len(df)),
        "n_subtests": int(len(subtests)),
    }
    manifest_path = tables_dir / f"age_adjustment_rerun_manifest_{cohort}_{token}.json"
    dump_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real age-adjustment residualization variant rerun.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", required=True, choices=tuple(COHORT_CONFIGS.keys()))
    parser.add_argument("--variant-token", required=True, choices=tuple(METHOD_BY_TOKEN.keys()))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        manifest = run_age_adjustment_rerun(
            root=root,
            cohort=str(args.cohort),
            variant_token=str(args.variant_token),
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] age-adjustment rerun complete: cohort={manifest['cohort']} variant={manifest['variant_token']}")
    print(f"[ok] wrote {manifest['diagnostics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
