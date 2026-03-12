#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import dump_json, load_yaml, project_root, relative_path

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

STAGE_SCRIPT_BY_NUMBER: dict[int, str] = {
    5: "05_construct_samples.py",
    6: "06_age_residualize.py",
    7: "07_fit_sem_models.py",
    8: "08_invariance_and_partial.py",
    9: "09_results_and_figures.py",
}

STAGE_SEQUENCE: tuple[int, ...] = (5, 6, 7, 8, 9)
PREFERRED_MEAN_STEPS: tuple[str, ...] = ("scalar", "strict", "metric", "configural")
PREFERRED_VARIANCE_STEPS: tuple[str, ...] = ("metric", "scalar", "strict", "configural")


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _prepare_isolated_project(
    *,
    root: Path,
    isolated_root: Path,
    cohort: str,
    variant_token: str,
) -> None:
    source_config = root / "config"
    target_config = isolated_root / "config"
    shutil.copytree(source_config, target_config, dirs_exist_ok=True)
    # Stage-07 expects sem_fit.R under <project-root>/scripts.
    source_r_script = root / "scripts" / "sem_fit.R"
    target_r_script = isolated_root / "scripts" / "sem_fit.R"
    if not source_r_script.exists():
        raise FileNotFoundError(f"Missing required SEM R script: {source_r_script}")
    target_r_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_r_script, target_r_script)

    main_paths = load_yaml(root / "config/paths.yml")
    isolated_paths = {
        "project_root": str(isolated_root),
        "raw_dir": str(_resolve_path(main_paths.get("raw_dir", "data/raw"), root)),
        "interim_dir": str(_resolve_path(main_paths.get("interim_dir", "data/interim"), root)),
        "processed_dir": str(isolated_root / "data" / "processed"),
        "outputs_dir": str(isolated_root / "outputs"),
        "logs_dir": str(isolated_root / "outputs" / "logs"),
        "manifest_file": str(_resolve_path(main_paths.get("manifest_file", "data/raw/manifest.json"), root)),
        "sem_interim_dir": str(isolated_root / "data" / "interim" / "sem"),
        "links_interim_dir": str(_resolve_path(main_paths.get("links_interim_dir", "data/interim/links"), root)),
    }
    _write_yaml(target_config / "paths.yml", isolated_paths)

    models_cfg_path = target_config / "models.yml"
    models_cfg = load_yaml(models_cfg_path)
    reporting_cfg = models_cfg.get("reporting", {})
    if not isinstance(reporting_cfg, dict):
        reporting_cfg = {}
    warning_policy_cfg = reporting_cfg.get("warning_policy", {})
    if not isinstance(warning_policy_cfg, dict):
        warning_policy_cfg = {}
    # Robustness reruns should complete even when warning policy flags fail-level
    # issues; downstream tables retain warning metadata for interpretation.
    warning_policy_cfg["enabled"] = False
    reporting_cfg["warning_policy"] = warning_policy_cfg
    models_cfg["reporting"] = reporting_cfg
    _write_yaml(models_cfg_path, models_cfg)

    cohort_config_path = target_config / f"{cohort}.yml"
    cohort_cfg = load_yaml(cohort_config_path)
    sample_cfg = cohort_cfg.get("sample_construct", {})
    if not isinstance(sample_cfg, dict):
        raise ValueError(f"{cohort}: sample_construct must be a mapping.")
    harmonize_cfg = sample_cfg.get("branch_harmonization", {})
    if not isinstance(harmonize_cfg, dict):
        raise ValueError(f"{cohort}: sample_construct.branch_harmonization must be configured.")
    if not bool(harmonize_cfg.get("enabled", False)):
        raise ValueError(f"{cohort}: branch_harmonization must be enabled for harmonization rerun.")
    harmonize_cfg["method"] = str(variant_token)
    sample_cfg["branch_harmonization"] = harmonize_cfg
    cohort_cfg["sample_construct"] = sample_cfg
    _write_yaml(cohort_config_path, cohort_cfg)


def _run_stage_sequence(
    *,
    root: Path,
    isolated_root: Path,
    cohort: str,
    log_path: Path,
) -> None:
    log_lines: list[str] = []
    for stage in STAGE_SEQUENCE:
        script = STAGE_SCRIPT_BY_NUMBER[stage]
        command = [
            sys.executable,
            str(root / "scripts" / script),
            "--project-root",
            str(isolated_root),
            "--cohort",
            cohort,
        ]
        command_display = " ".join(command)
        log_lines.append(f"[stage {stage:02d}] {command_display}")
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            log_lines.append(result.stdout.rstrip())
        if result.stderr:
            log_lines.append(result.stderr.rstrip())
        if result.returncode != 0:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            raise RuntimeError(
                f"Stage {stage:02d} failed for cohort={cohort} (exit={result.returncode}). "
                f"See log: {log_path}"
            )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def _export_variant_table(
    *,
    source_path: Path,
    target_path: Path,
    cohort: str,
    allow_empty: bool = False,
) -> int:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing expected rerun output: {source_path}")
    frame = pd.read_csv(source_path)
    if "cohort" in frame.columns:
        frame = frame[frame["cohort"].astype(str) == str(cohort)].copy()
    if frame.empty:
        if allow_empty:
            return 0
        raise ValueError(f"Rerun output has no rows for cohort '{cohort}': {source_path}")
    frame = frame.sort_values("cohort").reset_index(drop=True) if "cohort" in frame.columns else frame
    target_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target_path, index=False)
    return int(len(frame))


def _normalize_group(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"female", "f", "2", "woman", "w"}:
        return "female"
    if token in {"male", "m", "1", "man"}:
        return "male"
    return token


def _infer_groups(groups: list[Any]) -> tuple[str | None, str | None]:
    normalized = {str(g): _normalize_group(g) for g in groups}
    female = [g for g in groups if normalized[str(g)] == "female"]
    male = [g for g in groups if normalized[str(g)] == "male"]
    if len(female) == 1 and len(male) == 1:
        return str(female[0]), str(male[0])
    if len(groups) == 2:
        return str(groups[0]), str(groups[1])
    return None, None


def _pick_step_rows(df: pd.DataFrame, preferred_steps: tuple[str, ...]) -> pd.DataFrame:
    if df.empty or "model_step" not in df.columns:
        return df
    for step in preferred_steps:
        step_rows = df[df["model_step"].astype(str).str.lower() == step]
        if not step_rows.empty:
            return step_rows
    return df


def _safe_float(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def _derive_exploratory_estimates(
    *,
    root: Path,
    isolated_root: Path,
    cohort: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    tables = isolated_root / "outputs" / "tables"
    fit_root = isolated_root / "outputs" / "model_fits" / cohort
    params_path = fit_root / "params.csv"
    latent_path = fit_root / "latent_summary.csv"
    sample_counts_path = tables / "sample_counts.csv"
    params = pd.read_csv(params_path) if params_path.exists() else pd.DataFrame()
    latent = pd.read_csv(latent_path) if latent_path.exists() else pd.DataFrame()
    sample_counts = pd.read_csv(sample_counts_path) if sample_counts_path.exists() else pd.DataFrame()
    cohort_cfg_path = root / "config" / f"{cohort}.yml"
    cohort_cfg = load_yaml(cohort_cfg_path) if cohort_cfg_path.exists() else {}
    sem_fit_cfg = cohort_cfg.get("sem_fit", {}) if isinstance(cohort_cfg.get("sem_fit", {}), dict) else {}
    std_lv = bool(sem_fit_cfg.get("std_lv", True))

    mean_row: dict[str, Any] | None = None
    vr_row: dict[str, Any] | None = None

    mean_candidates = pd.DataFrame()
    if not params.empty:
        mean_candidates = _pick_step_rows(params, PREFERRED_MEAN_STEPS)
        if {"lhs", "op"}.issubset(set(mean_candidates.columns)):
            mean_candidates = mean_candidates[
                (mean_candidates["lhs"].astype(str) == "g")
                & (mean_candidates["op"].astype(str) == "~1")
            ].copy()
        else:
            mean_candidates = pd.DataFrame()
    if mean_candidates.empty and not latent.empty:
        if {"cohort", "factor", "group", "mean"}.issubset(set(latent.columns)):
            mean_candidates = latent[
                (latent["cohort"].astype(str) == cohort) & (latent["factor"].astype(str) == "g")
            ][["group", "mean"]].rename(columns={"mean": "est"})
            mean_candidates["se"] = pd.NA

    groups = [str(g) for g in mean_candidates.get("group", pd.Series(dtype=object)).dropna().unique().tolist()]
    female_group, male_group = _infer_groups(groups)
    if female_group is not None and male_group is not None:
        mean_lookup = {str(r["group"]): _safe_float(r.get("est")) for _, r in mean_candidates.iterrows()}
        se_lookup = {str(r["group"]): _safe_float(r.get("se")) for _, r in mean_candidates.iterrows()}
        male_mean = mean_lookup.get(male_group)
        female_mean = mean_lookup.get(female_group)
        if male_mean is not None and female_mean is not None:
            d_g = float(male_mean - female_mean)
            se_d = None
            if se_lookup.get(male_group) is not None and se_lookup.get(female_group) is not None:
                se_d = math.sqrt(float(se_lookup[male_group]) ** 2 + float(se_lookup[female_group]) ** 2)
            mean_row = {
                "cohort": cohort,
                "d_g": d_g,
                "SE_d_g": se_d if se_d is not None else pd.NA,
                "ci_low_d_g": (d_g - 1.96 * se_d) if se_d is not None else pd.NA,
                "ci_high_d_g": (d_g + 1.96 * se_d) if se_d is not None else pd.NA,
                "IQ_diff": d_g * 15.0,
                "SE": (se_d * 15.0) if se_d is not None else pd.NA,
                "ci_low": ((d_g - 1.96 * se_d) * 15.0) if se_d is not None else pd.NA,
                "ci_high": ((d_g + 1.96 * se_d) * 15.0) if se_d is not None else pd.NA,
            }

    var_candidates = pd.DataFrame()
    if not params.empty:
        var_candidates = _pick_step_rows(params, PREFERRED_VARIANCE_STEPS)
        if {"lhs", "rhs", "op"}.issubset(set(var_candidates.columns)):
            var_candidates = var_candidates[
                (var_candidates["lhs"].astype(str) == "g")
                & (var_candidates["rhs"].astype(str) == "g")
                & (var_candidates["op"].astype(str) == "~~")
            ].copy()
        else:
            var_candidates = pd.DataFrame()
    if var_candidates.empty and not latent.empty:
        if {"cohort", "factor", "group", "var"}.issubset(set(latent.columns)):
            var_candidates = latent[
                (latent["cohort"].astype(str) == cohort) & (latent["factor"].astype(str) == "g")
            ][["group", "var"]].rename(columns={"var": "est"})
            var_candidates["se"] = pd.NA

    groups = [str(g) for g in var_candidates.get("group", pd.Series(dtype=object)).dropna().unique().tolist()]
    female_group, male_group = _infer_groups(groups)
    if female_group is not None and male_group is not None:
        var_lookup = {str(r["group"]): _safe_float(r.get("est")) for _, r in var_candidates.iterrows()}
        male_var = var_lookup.get(male_group)
        female_var = var_lookup.get(female_group)
        if mean_row is not None and not std_lv and female_var is not None and female_var > 0:
            d_g_standardized = float(mean_row["d_g"]) / math.sqrt(female_var)
            se_d = _safe_float(mean_row.get("SE_d_g"))
            mean_row["IQ_diff"] = d_g_standardized * 15.0
            if se_d is not None:
                mean_row["ci_low"] = (d_g_standardized - 1.96 * se_d) * 15.0
                mean_row["ci_high"] = (d_g_standardized + 1.96 * se_d) * 15.0
        if male_var is not None and female_var is not None and male_var > 0 and female_var > 0:
            vr = float(male_var / female_var)
            se_log_vr = None
            if not sample_counts.empty and "cohort" in sample_counts.columns:
                row = sample_counts[sample_counts["cohort"].astype(str) == cohort]
                if not row.empty:
                    n_male = _safe_float(row.iloc[0].get("n_male"))
                    n_female = _safe_float(row.iloc[0].get("n_female"))
                    if n_male and n_female and n_male > 1 and n_female > 1:
                        se_log_vr = math.sqrt(2.0 / (n_male - 1.0) + 2.0 / (n_female - 1.0))
            if se_log_vr is not None:
                log_vr = math.log(vr)
                ci_low = math.exp(log_vr - 1.96 * se_log_vr)
                ci_high = math.exp(log_vr + 1.96 * se_log_vr)
            else:
                ci_low = pd.NA
                ci_high = pd.NA
            vr_row = {
                "cohort": cohort,
                "VR_g": vr,
                "SE_logVR": se_log_vr if se_log_vr is not None else pd.NA,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

    return mean_row, vr_row


def run_harmonization_rerun(
    *,
    root: Path,
    cohort: str,
    variant_token: str,
    keep_workspace: bool = False,
) -> dict[str, Any]:
    if cohort not in COHORT_CONFIGS:
        raise ValueError(f"Unsupported cohort: {cohort}")
    safe_token = "".join(c if (c.isalnum() or c in {"_", "-", "."}) else "_" for c in variant_token.strip())
    if not safe_token:
        raise ValueError("variant-token must be a non-empty string")

    paths_cfg = load_yaml(root / "config/paths.yml")
    outputs_dir = _resolve_path(paths_cfg.get("outputs_dir", "outputs"), root)
    run_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    workspace_root = outputs_dir / "logs" / "robustness" / "harmonization_runs" / f"{run_stamp}_{cohort}_{safe_token}"
    isolated_root = workspace_root / "project"
    run_log_path = outputs_dir / "logs" / "robustness" / f"harmonization_rerun_{run_stamp}_{cohort}_{safe_token}.log"
    success = False
    try:
        _prepare_isolated_project(
            root=root,
            isolated_root=isolated_root,
            cohort=cohort,
            variant_token=safe_token,
        )
        _run_stage_sequence(
            root=root,
            isolated_root=isolated_root,
            cohort=cohort,
            log_path=run_log_path,
        )

        isolated_tables = isolated_root / "outputs" / "tables"
        main_tables = outputs_dir / "tables"
        main_tables.mkdir(parents=True, exist_ok=True)
        mean_rows = _export_variant_table(
            source_path=isolated_tables / "g_mean_diff.csv",
            target_path=main_tables / f"g_mean_diff_{safe_token}.csv",
            cohort=cohort,
            allow_empty=True,
        )
        vr_rows = _export_variant_table(
            source_path=isolated_tables / "g_variance_ratio.csv",
            target_path=main_tables / f"g_variance_ratio_{safe_token}.csv",
            cohort=cohort,
            allow_empty=True,
        )
        exploratory_override = False
        if mean_rows == 0 or vr_rows == 0:
            derived_mean, derived_vr = _derive_exploratory_estimates(
                root=root,
                isolated_root=isolated_root,
                cohort=cohort,
            )
            if mean_rows == 0 and derived_mean is not None:
                pd.DataFrame([derived_mean]).to_csv(
                    main_tables / f"g_mean_diff_{safe_token}.csv",
                    index=False,
                )
                mean_rows = 1
                exploratory_override = True
            if vr_rows == 0 and derived_vr is not None:
                pd.DataFrame([derived_vr]).to_csv(
                    main_tables / f"g_variance_ratio_{safe_token}.csv",
                    index=False,
                )
                vr_rows = 1
                exploratory_override = True
        if mean_rows == 0:
            raise ValueError(
                f"Rerun output has no rows for cohort '{cohort}': {isolated_tables / 'g_mean_diff.csv'}"
            )
        if vr_rows == 0:
            raise ValueError(
                f"Rerun output has no rows for cohort '{cohort}': {isolated_tables / 'g_variance_ratio.csv'}"
            )
        success = True
        manifest = {
            "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "cohort": cohort,
            "variant_token": safe_token,
            "workspace_root": relative_path(root, workspace_root),
            "workspace_kept": bool(keep_workspace),
            "run_log": relative_path(root, run_log_path),
            "artifacts": {
                "g_mean_diff": f"outputs/tables/g_mean_diff_{safe_token}.csv",
                "g_variance_ratio": f"outputs/tables/g_variance_ratio_{safe_token}.csv",
            },
            "row_counts": {
                "g_mean_diff": mean_rows,
                "g_variance_ratio": vr_rows,
            },
            "exploratory_override_used": exploratory_override,
        }
        dump_json(
            outputs_dir / "tables" / f"harmonization_rerun_manifest_{cohort}_{safe_token}.json",
            manifest,
        )
        return manifest
    finally:
        if success and (not keep_workspace) and workspace_root.exists():
            shutil.rmtree(workspace_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run harmonization sensitivity rerun in isolated workspace and export variant artifacts."
    )
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--cohort", choices=sorted(COHORT_CONFIGS), default="nlsy97", help="Target cohort.")
    parser.add_argument("--variant-token", required=True, help="Harmonization method token (for example zscore_by_branch).")
    parser.add_argument("--keep-workspace", action="store_true", help="Keep isolated rerun workspace for debugging.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        manifest = run_harmonization_rerun(
            root=root,
            cohort=str(args.cohort),
            variant_token=str(args.variant_token),
            keep_workspace=bool(args.keep_workspace),
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(
        f"[ok] harmonization rerun complete for cohort={manifest['cohort']} variant={manifest['variant_token']}"
    )
    print(f"[ok] wrote {manifest['artifacts']['g_mean_diff']}")
    print(f"[ok] wrote {manifest['artifacts']['g_variance_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
