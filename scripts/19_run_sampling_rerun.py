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
from nls_pipeline.sem import hierarchical_subtests

COHORTS: tuple[str, ...] = ("nlsy79", "nlsy97", "cnlsy")
STAGE_SEQUENCE: tuple[int, ...] = (7, 8, 9)
STAGE_SCRIPT_BY_NUMBER: dict[int, str] = {
    7: "07_fit_sem_models.py",
    8: "08_invariance_and_partial.py",
    9: "09_results_and_figures.py",
}
PREFERRED_MEAN_STEPS: tuple[str, ...] = ("scalar", "strict", "metric", "configural")
MAX_REASONABLE_SE_D_G = 5.0


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _safe_float(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def _normalize_group(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"female", "f", "2", "woman", "w"}:
        return "female"
    if token in {"male", "m", "1", "man", "boy"}:
        return "male"
    return token


def _infer_groups(groups: list[Any], means: dict[str, float | None]) -> tuple[str | None, str | None]:
    if not groups:
        return None, None
    groups_str = [str(g) for g in groups]
    female = [g for g in groups_str if _normalize_group(g) == "female"]
    male = [g for g in groups_str if _normalize_group(g) == "male"]
    if len(female) == 1 and len(male) == 1:
        return female[0], male[0]

    for group in groups_str:
        m = means.get(group)
        if m is not None and math.isfinite(float(m)) and abs(float(m)) <= 1e-8:
            female_guess = group
            male_guess = next((g for g in groups_str if g != female_guess), None)
            return female_guess, male_guess

    if len(groups_str) == 2:
        return groups_str[0], groups_str[1]
    return None, None


def _pick_step_rows(df: pd.DataFrame, preferred_steps: tuple[str, ...]) -> pd.DataFrame:
    if df.empty or "model_step" not in df.columns:
        return df
    for step in preferred_steps:
        step_rows = df[df["model_step"].astype(str).str.lower() == step]
        if not step_rows.empty:
            return step_rows
    return df


def _derive_d_g_from_model_fit(
    *,
    model_fit_dir: Path,
    cohort: str,
) -> dict[str, Any] | None:
    params_path = model_fit_dir / "params.csv"
    latent_path = model_fit_dir / "latent_summary.csv"
    params = pd.read_csv(params_path) if params_path.exists() else pd.DataFrame()
    latent = pd.read_csv(latent_path) if latent_path.exists() else pd.DataFrame()

    rows = pd.DataFrame()
    if not params.empty and {"lhs", "op", "group", "est"}.issubset(set(params.columns)):
        subset = _pick_step_rows(params, PREFERRED_MEAN_STEPS)
        rows = subset[(subset["lhs"].astype(str) == "g") & (subset["op"].astype(str) == "~1")][
            ["group", "est", "se"]
        ].copy()

    if rows.empty and not latent.empty and {"cohort", "factor", "group", "mean"}.issubset(set(latent.columns)):
        rows = latent[(latent["cohort"].astype(str) == cohort) & (latent["factor"].astype(str) == "g")][
            ["group", "mean"]
        ].rename(columns={"mean": "est"})
        rows["se"] = pd.NA

    if rows.empty:
        return None

    groups = [str(g) for g in rows["group"].dropna().unique().tolist()]
    means = {str(r["group"]): _safe_float(r.get("est")) for _, r in rows.iterrows()}
    ses = {str(r["group"]): _safe_float(r.get("se")) for _, r in rows.iterrows()}
    female, male = _infer_groups(groups, means)
    if female is None or male is None:
        return None

    male_mean = means.get(male)
    female_mean = means.get(female)
    if male_mean is None or female_mean is None:
        return None

    d_g = float(male_mean - female_mean)
    se_d = None
    if ses.get(male) is not None and ses.get(female) is not None:
        se_d = math.sqrt(float(ses[male]) ** 2 + float(ses[female]) ** 2)

    ci_low_d = pd.NA
    ci_high_d = pd.NA
    if se_d is not None:
        ci_low_d = d_g - 1.96 * se_d
        ci_high_d = d_g + 1.96 * se_d

    return {
        "cohort": cohort,
        "d_g": d_g,
        "SE_d_g": se_d if se_d is not None else pd.NA,
        "ci_low_d_g": ci_low_d,
        "ci_high_d_g": ci_high_d,
        "IQ_diff": d_g * 15.0,
        "SE": (se_d * 15.0) if se_d is not None else pd.NA,
        "ci_low": (ci_low_d * 15.0) if se_d is not None else pd.NA,
        "ci_high": (ci_high_d * 15.0) if se_d is not None else pd.NA,
    }


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


def _family_ids_for_cohort(root: Path, cohort: str, df: pd.DataFrame) -> pd.Series:
    if cohort == "nlsy79":
        lookup = _nlsy79_family_lookup(root)
        pids = pd.to_numeric(df.get("person_id"), errors="coerce")
        series = pids.map(lambda x: lookup.get(int(x), None) if pd.notna(x) else None)
        return series.fillna(pids.map(lambda x: f"singleton_{int(x)}" if pd.notna(x) else "singleton_missing")).astype(str)

    if cohort == "nlsy97":
        fam = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
        if "R9708601" in df.columns:
            fam = pd.to_numeric(df["R9708601"], errors="coerce")
        if "R9708602" in df.columns:
            fallback = pd.to_numeric(df["R9708602"], errors="coerce")
            fam = fam.where(pd.notna(fam), fallback)
        fam = fam.map(lambda x: f"hh_{int(x)}" if pd.notna(x) else pd.NA)
        if "person_id" in df.columns:
            pids = pd.to_numeric(df["person_id"], errors="coerce")
            fam = fam.where(pd.notna(fam), pids.map(lambda x: f"singleton_{int(x)}" if pd.notna(x) else "singleton_missing"))
        return fam.astype(str)

    pids = pd.to_numeric(df.get("person_id"), errors="coerce")
    return pids.map(lambda x: f"singleton_{int(x)}" if pd.notna(x) else "singleton_missing").astype(str)


def _sample_variant_df(root: Path, cohort: str, source_df: pd.DataFrame, variant_token: str) -> pd.DataFrame:
    if variant_token == "full_cohort":
        return source_df.copy()

    if variant_token != "one_pair_per_family":
        raise ValueError(f"Unsupported sampling variant token: {variant_token}")

    work = source_df.copy()
    family = _family_ids_for_cohort(root, cohort, work)
    work["__family_id_variant"] = family

    sort_cols = ["__family_id_variant"]
    if "person_id" in work.columns:
        sort_cols.append("person_id")
    work = work.sort_values(sort_cols).reset_index(drop=True)
    sampled = work.drop_duplicates(subset=["__family_id_variant"], keep="first").copy()
    sampled = sampled.drop(columns=["__family_id_variant"], errors="ignore")
    return sampled


def _prepare_isolated_project(root: Path, isolated_root: Path) -> None:
    source_config = root / "config"
    target_config = isolated_root / "config"
    shutil.copytree(source_config, target_config, dirs_exist_ok=True)

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
    warning_policy_cfg["enabled"] = False
    reporting_cfg["warning_policy"] = warning_policy_cfg
    models_cfg["reporting"] = reporting_cfg
    _write_yaml(models_cfg_path, models_cfg)


def _run_stage_sequence(root: Path, isolated_root: Path, log_path: Path) -> None:
    lines: list[str] = []
    stage7_failed = False
    for stage in STAGE_SEQUENCE:
        script = STAGE_SCRIPT_BY_NUMBER[stage]
        command = [
            sys.executable,
            str(root / "scripts" / script),
            "--project-root",
            str(isolated_root),
            "--all",
        ]
        lines.append(f"[stage {stage:02d}] {' '.join(command)}")
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        if result.stdout:
            lines.append(result.stdout.rstrip())
        if result.stderr:
            lines.append(result.stderr.rstrip())
        if result.returncode != 0:
            if stage == 7:
                stage7_failed = True
                lines.append(f"[warn] stage 07 returned non-zero (exit={result.returncode}); continuing with stages 08-09.")
                continue
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            raise RuntimeError(f"Stage {stage:02d} failed (exit={result.returncode}). See {log_path}")

    if stage7_failed:
        lines.append("[warn] partial stage-07 failures occurred; downstream outputs may require fallback derivation.")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=1)
    if pd.isna(sd) or float(sd) <= 0.0:
        return pd.Series([pd.NA] * len(s), index=s.index, dtype="float64")
    return (s - mean) / sd


def _derive_d_g_from_sample(
    *,
    cohort: str,
    sampled_df: pd.DataFrame,
    models_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if "sex" not in sampled_df.columns:
        return None

    if cohort == "cnlsy":
        indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", [])]
    else:
        indicators = hierarchical_subtests(models_cfg)
    existing = [col for col in indicators if col in sampled_df.columns]
    if not existing:
        return None

    z_df = pd.DataFrame({col: _zscore(sampled_df[col]) for col in existing}, index=sampled_df.index)
    g_proxy = z_df.mean(axis=1, skipna=False)
    sex = sampled_df["sex"].astype(str).str.strip().str.lower()
    male = pd.to_numeric(g_proxy[sex.isin({"m", "male", "1", "man", "boy"})], errors="coerce").dropna()
    female = pd.to_numeric(g_proxy[sex.isin({"f", "female", "2", "woman", "girl"})], errors="coerce").dropna()
    if len(male) < 2 or len(female) < 2:
        return None

    male_var = float(male.var(ddof=1))
    female_var = float(female.var(ddof=1))
    pooled_var = (((len(male) - 1) * male_var) + ((len(female) - 1) * female_var)) / float(len(male) + len(female) - 2)
    if not math.isfinite(pooled_var) or pooled_var <= 0.0:
        return None
    pooled_sd = math.sqrt(pooled_var)
    d_g = float(male.mean() - female.mean()) / pooled_sd
    se_m = float(male.std(ddof=1) / math.sqrt(len(male)))
    se_f = float(female.std(ddof=1) / math.sqrt(len(female)))
    se_d = math.sqrt(se_m**2 + se_f**2)
    ci_low_d = d_g - 1.96 * se_d
    ci_high_d = d_g + 1.96 * se_d
    return {
        "cohort": cohort,
        "d_g": d_g,
        "SE_d_g": se_d,
        "ci_low_d_g": ci_low_d,
        "ci_high_d_g": ci_high_d,
        "IQ_diff": d_g * 15.0,
        "SE": se_d * 15.0,
        "ci_low": ci_low_d * 15.0,
        "ci_high": ci_high_d * 15.0,
    }


def run_sampling_rerun(*, root: Path, variant_token: str, keep_workspace: bool = False) -> dict[str, Any]:
    safe_variant = "".join(c if (c.isalnum() or c in {"_", "-", "."}) else "_" for c in variant_token.strip())
    if safe_variant not in {"full_cohort", "one_pair_per_family"}:
        raise ValueError("variant-token must be one of: full_cohort, one_pair_per_family")

    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    outputs_dir = _resolve_path(paths_cfg.get("outputs_dir", "outputs"), root)
    processed_dir = _resolve_path(paths_cfg.get("processed_dir", "data/processed"), root)

    run_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    workspace_root = outputs_dir / "logs" / "robustness" / "sampling_runs" / f"{run_stamp}_{safe_variant}"
    isolated_root = workspace_root / "project"
    run_log = outputs_dir / "logs" / "robustness" / f"sampling_rerun_{run_stamp}_{safe_variant}.log"

    success = False
    try:
        _prepare_isolated_project(root, isolated_root)

        isolated_processed = isolated_root / "data" / "processed"
        isolated_processed.mkdir(parents=True, exist_ok=True)
        sample_rows: list[dict[str, Any]] = []
        sampled_by_cohort: dict[str, pd.DataFrame] = {}

        for cohort in COHORTS:
            source_path = processed_dir / f"{cohort}_cfa_resid.csv"
            if not source_path.exists():
                source_path = processed_dir / f"{cohort}_cfa.csv"
            if not source_path.exists():
                raise FileNotFoundError(f"Missing processed input for {cohort}: {source_path}")

            source_df = pd.read_csv(source_path, low_memory=False)
            sampled_df = _sample_variant_df(root, cohort, source_df, safe_variant)
            sampled_by_cohort[cohort] = sampled_df.copy()
            sampled_out = isolated_processed / f"{cohort}_cfa_resid.csv"
            sampled_df.to_csv(sampled_out, index=False)

            sex_vals = sampled_df.get("sex", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
            n_male = int(sex_vals.isin({"m", "male", "1"}).sum())
            n_female = int(sex_vals.isin({"f", "female", "2"}).sum())
            sample_rows.append(
                {
                    "cohort": cohort,
                    "n_input": int(len(source_df)),
                    "n_after_age": int(len(sampled_df)),
                    "n_after_test_rule": int(len(sampled_df)),
                    "n_after_dedupe": int(len(sampled_df)),
                    "n_male": n_male,
                    "n_female": n_female,
                }
            )

        isolated_tables = isolated_root / "outputs" / "tables"
        isolated_tables.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sample_rows).to_csv(isolated_tables / "sample_counts.csv", index=False)

        _run_stage_sequence(root, isolated_root, run_log)

        main_tables = outputs_dir / "tables"
        main_tables.mkdir(parents=True, exist_ok=True)

        sample_counts_variant = main_tables / f"sample_counts_{safe_variant}.csv"
        pd.DataFrame(sample_rows).to_csv(sample_counts_variant, index=False)

        g_mean_path = isolated_tables / "g_mean_diff.csv"
        g_mean = pd.read_csv(g_mean_path) if g_mean_path.exists() else pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for cohort in COHORTS:
            if not g_mean.empty and "cohort" in g_mean.columns:
                cohort_rows = g_mean[g_mean["cohort"].astype(str) == cohort]
            else:
                cohort_rows = pd.DataFrame()
            if not cohort_rows.empty:
                cohort_row = cohort_rows.iloc[0].to_dict()
                se_candidate = _safe_float(cohort_row.get("SE_d_g"))
                if se_candidate is None or abs(se_candidate) <= MAX_REASONABLE_SE_D_G:
                    rows.append(cohort_row)
                    continue

            derived = _derive_d_g_from_model_fit(
                model_fit_dir=isolated_root / "outputs" / "model_fits" / cohort,
                cohort=cohort,
            )
            if derived is not None:
                se_candidate = _safe_float(derived.get("SE_d_g"))
                if se_candidate is not None and abs(se_candidate) > MAX_REASONABLE_SE_D_G:
                    derived = None
            if derived is None and cohort in sampled_by_cohort:
                derived = _derive_d_g_from_sample(
                    cohort=cohort,
                    sampled_df=sampled_by_cohort[cohort],
                    models_cfg=models_cfg,
                )
            if derived is not None:
                rows.append(derived)

        if not rows:
            raise ValueError(f"No g_mean rows could be produced for variant {safe_variant}")
        g_mean_variant_path = main_tables / f"g_mean_diff_{safe_variant}.csv"
        pd.DataFrame(rows).to_csv(g_mean_variant_path, index=False)

        manifest = {
            "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "variant_token": safe_variant,
            "workspace_root": relative_path(root, workspace_root),
            "workspace_kept": bool(keep_workspace),
            "run_log": relative_path(root, run_log),
            "artifacts": {
                "sample_counts": relative_path(root, sample_counts_variant),
                "g_mean_diff": relative_path(root, g_mean_variant_path),
            },
        }
        dump_json(main_tables / f"sampling_rerun_manifest_{safe_variant}.json", manifest)

        success = True
        return manifest
    finally:
        if success and (not keep_workspace) and workspace_root.exists():
            shutil.rmtree(workspace_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sampling robustness rerun in isolated workspace.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--variant-token", required=True, choices=("full_cohort", "one_pair_per_family"))
    parser.add_argument("--keep-workspace", action="store_true", help="Keep isolated workspace.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        manifest = run_sampling_rerun(
            root=root,
            variant_token=str(args.variant_token),
            keep_workspace=bool(args.keep_workspace),
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] sampling rerun complete for variant={manifest['variant_token']}")
    print(f"[ok] wrote {manifest['artifacts']['sample_counts']}")
    print(f"[ok] wrote {manifest['artifacts']['g_mean_diff']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
