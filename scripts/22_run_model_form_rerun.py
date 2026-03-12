#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import dump_json, load_yaml, project_root, relative_path
from nls_pipeline.sem import hierarchical_subtests, rscript_path, run_python_sem_fallback, run_sem_r_script

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}
VARIANTS = ("single_factor_alt", "bifactor_alt")


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    for val in values:
        if val not in out:
            out.append(val)
    return out


def _input_data_path(root: Path, paths_cfg: dict[str, Any], cohort: str) -> Path:
    processed_dir = _resolve_path(paths_cfg["processed_dir"], root)
    preferred = processed_dir / f"{cohort}_cfa_resid.csv"
    fallback = processed_dir / f"{cohort}_cfa.csv"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing cohort input for model-form rerun: expected {preferred} or {fallback}")


def _factor_map(models_cfg: dict[str, Any], available_cols: set[str]) -> dict[str, list[str]]:
    groups = models_cfg.get("hierarchical_factors", {})
    if not isinstance(groups, dict):
        groups = {}
    factor_map = {
        "Speed": [str(x) for x in groups.get("speed", [])],
        "Math": [str(x) for x in groups.get("math", [])],
        "Verbal": [str(x) for x in groups.get("verbal", [])],
        "Tech": [str(x) for x in groups.get("technical", [])],
    }
    return {
        factor: [x for x in indicators if x in available_cols]
        for factor, indicators in factor_map.items()
    }


def _single_factor_syntax(latent: str, indicators: list[str]) -> str:
    if len(indicators) < 2:
        raise ValueError(f"Single-factor model requires at least 2 indicators, got {len(indicators)}")
    return f"{latent} =~ {' + '.join(indicators)}"


def _bifactor_syntax(indicators: list[str], factor_map: dict[str, list[str]]) -> str:
    if len(indicators) < 3:
        return _single_factor_syntax("g", indicators)
    lines = [f"g =~ {' + '.join(indicators)}"]
    valid_factors = [name for name, cols in factor_map.items() if len(cols) >= 2]
    for factor in valid_factors:
        lines.append(f"{factor} =~ {' + '.join(factor_map[factor])}")

    if not valid_factors:
        return "\n".join(lines)

    for factor in valid_factors:
        lines.append(f"g ~~ 0*{factor}")
    for i, left in enumerate(valid_factors):
        for right in valid_factors[i + 1 :]:
            lines.append(f"{left} ~~ 0*{right}")
    return "\n".join(lines)


def _model_syntax_for_variant(
    *,
    cohort: str,
    variant_token: str,
    models_cfg: dict[str, Any],
    data_cols: set[str],
) -> tuple[str, list[str]]:
    if cohort == "cnlsy":
        indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", []) if str(x) in data_cols]
        latent = "g_cnlsy"
        return _single_factor_syntax(latent, indicators), indicators

    indicators = [x for x in hierarchical_subtests(models_cfg) if x in data_cols]
    if variant_token == "single_factor_alt":
        return _single_factor_syntax("g", indicators), indicators

    factor_map = _factor_map(models_cfg, data_cols)
    return _bifactor_syntax(indicators, factor_map), indicators


def run_model_form_rerun(
    *,
    root: Path,
    cohort: str,
    variant_token: str,
    skip_r: bool = False,
) -> dict[str, Any]:
    variant = str(variant_token).strip()
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant token: {variant}")

    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    cohort_cfg = load_yaml(root / COHORT_CONFIGS[cohort])
    sample_cfg = cohort_cfg.get("sample_construct", {}) if isinstance(cohort_cfg.get("sample_construct", {}), dict) else {}
    sem_fit_cfg = cohort_cfg.get("sem_fit", {}) if isinstance(cohort_cfg.get("sem_fit", {}), dict) else {}
    group_col = str(sample_cfg.get("sex_col", "sex"))
    std_lv = bool(sem_fit_cfg.get("std_lv", True))

    source_path = _input_data_path(root, paths_cfg, cohort)
    df = pd.read_csv(source_path, low_memory=False)
    if group_col not in df.columns:
        raise ValueError(f"{cohort}: group column missing for model-form rerun: {group_col}")

    model_syntax, observed_tests = _model_syntax_for_variant(
        cohort=cohort,
        variant_token=variant,
        models_cfg=models_cfg,
        data_cols=set(df.columns),
    )

    invariance_cfg = models_cfg.get("invariance", {}) if isinstance(models_cfg.get("invariance", {}), dict) else {}
    steps = [str(x) for x in invariance_cfg.get("steps", ["configural", "metric", "scalar"])]
    request_payload = {
        "cohort": cohort,
        "data_csv": str(source_path),
        "group_col": group_col,
        "estimator": "MLR",
        "missing": "fiml",
        "std_lv": std_lv,
        "invariance_steps": steps,
        "partial_intercepts": [],
        "observed_tests": observed_tests,
        "se_mode": "standard",
        "cluster_col": None,
        "weight_col": None,
    }

    outputs_dir = _resolve_path(paths_cfg["outputs_dir"], root)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    workspace = outputs_dir / "logs" / "robustness" / "model_form_runs" / f"{run_stamp}_{cohort}_{variant}"
    workspace.mkdir(parents=True, exist_ok=True)
    request_path = workspace / "request.json"
    model_path = workspace / "model.lavaan"
    fit_dir = workspace / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True), encoding="utf-8")
    model_path.write_text(model_syntax.strip() + "\n", encoding="utf-8")

    used_python_fallback = False
    rerun_status = "ok"
    rerun_error = ""
    if not skip_r and rscript_path() is not None:
        try:
            run_sem_r_script(r_script=root / "scripts" / "sem_fit.R", request_file=request_path, outdir=fit_dir)
        except Exception as exc:
            used_python_fallback = True
            rerun_status = "r_failed_python_fallback"
            rerun_error = str(exc)
            run_python_sem_fallback(
                cohort=cohort,
                data_csv=source_path,
                outdir=fit_dir,
                group_col=group_col,
                models_cfg=models_cfg,
                invariance_steps=steps,
                observed_tests=observed_tests,
            )
    else:
        used_python_fallback = True
        rerun_status = "python_fallback"
        run_python_sem_fallback(
            cohort=cohort,
            data_csv=source_path,
            outdir=fit_dir,
            group_col=group_col,
            models_cfg=models_cfg,
            invariance_steps=steps,
            observed_tests=observed_tests,
        )

    fit_path = fit_dir / "fit_indices.csv"
    if not fit_path.exists():
        raise FileNotFoundError(f"Model-form rerun did not produce fit_indices.csv: {fit_path}")
    fit_df = pd.read_csv(fit_path)
    if "cohort" not in fit_df.columns:
        fit_df["cohort"] = cohort

    variant_summary_path = tables_dir / f"{cohort}_invariance_summary_{variant}.csv"
    fit_df.to_csv(variant_summary_path, index=False)

    manifest = {
        "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "cohort": cohort,
        "variant_token": variant,
        "source_path": relative_path(root, source_path),
        "model_syntax_path": relative_path(root, model_path),
        "request_path": relative_path(root, request_path),
        "fit_dir": relative_path(root, fit_dir),
        "variant_summary_path": relative_path(root, variant_summary_path),
        "used_python_fallback": bool(used_python_fallback),
        "rerun_status": rerun_status,
        "rerun_error": rerun_error if rerun_error else None,
    }
    manifest_path = tables_dir / f"model_form_rerun_manifest_{cohort}_{variant}.json"
    dump_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real model-form rerun and emit variant invariance summary.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", required=True, choices=tuple(COHORT_CONFIGS.keys()))
    parser.add_argument("--variant-token", required=True, choices=VARIANTS)
    parser.add_argument("--skip-r", action="store_true", help="Force Python fallback without R.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        manifest = run_model_form_rerun(
            root=root,
            cohort=str(args.cohort),
            variant_token=str(args.variant_token),
            skip_r=bool(args.skip_r),
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] model-form rerun complete: cohort={manifest['cohort']} variant={manifest['variant_token']}")
    print(f"[ok] wrote {manifest['variant_summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
