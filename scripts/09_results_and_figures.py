#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_model_syntax
from nls_pipeline.plots import save_bar_plot, save_forest_plot
from nls_pipeline.stats import mean_diff_ci_iq, variance_ratio_ci

COHORTS = ["nlsy79", "nlsy97", "cnlsy"]
# Prefer scalar-constrained estimates for latent means and metric-constrained estimates
# for latent variances, while allowing deterministic fallbacks for exploratory rows.
PREFERRED_MEAN_STEPS = ("scalar", "strict", "metric", "configural")
PREFERRED_VARIANCE_STEPS = ("metric", "scalar", "strict", "configural")

GMEAN_COLUMNS = ["cohort", "d_g", "SE_d_g", "ci_low_d_g", "ci_high_d_g", "IQ_diff", "SE", "ci_low", "ci_high"]
GVR_COLUMNS = ["cohort", "VR_g", "SE_logVR", "ci_low", "ci_high"]
GVR0_COLUMNS = ["cohort", "VR0_g", "SE_logVR0", "ci_low", "ci_high", "mean_diff_at_vr0"]
GROUP_FACTOR_COLUMNS = [
    "cohort",
    "factor",
    "male_mean",
    "female_mean",
    "mean_diff",
    "mean_diff_iq",
    "SE",
    "mean_ci_low",
    "mean_ci_high",
    "VR",
    "SE_logVR",
    "vr_ci_low",
    "vr_ci_high",
]
CONFIRMATORY_EXCLUSIONS_COLUMNS = [
    "cohort",
    "blocked_confirmatory",
    "blocked_confirmatory_d_g",
    "blocked_confirmatory_vr_g",
    "reason",
    "reason_d_g",
    "reason_vr_g",
]
ANALYSIS_TIER_COLUMNS = [
    "cohort",
    "estimand",
    "analysis_tier",
    "blocked_confirmatory",
    "reason",
]


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return COHORTS
    return args.cohort


def _normalize_num(value: object) -> float | None:
    value_f = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value_f):
        return None
    return float(value_f)


def _load_csv_if_exists(path: Path, required_columns: set[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    if required_columns is not None:
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}.")
    return frame


def _pick_preferred_step(df: pd.DataFrame, preferred_steps: tuple[str, ...]) -> pd.DataFrame:
    if df.empty or "model_step" not in df.columns:
        return df
    for step in preferred_steps:
        step_rows = df[df["model_step"] == step]
        if not step_rows.empty:
            return step_rows
    return df


def _normalize_group(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    token = str(value).strip().lower()
    if token in {"f", "female", "2", "w", "woman"}:
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


def _infer_group_labels(
    groups: list[str],
    means_by_group: dict[str, float | None],
    reference_group: str,
) -> tuple[str | None, str | None]:
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
                    if _is_real_finite_number(mean) and abs(float(mean)) <= 1e-8:
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


def _cohort_group_factors(models_cfg: dict[str, Any]) -> list[str]:
    try:
        syntax = hierarchical_model_syntax(models_cfg)
    except Exception:
        syntax = ""
    factors: list[str] = []
    for line in str(syntax).splitlines():
        if "=~" not in line:
            continue
        lhs = str(line.split("=~", 1)[0]).strip()
        if lhs and lhs != "g" and lhs not in factors:
            factors.append(lhs)
    if factors:
        return factors

    hierarchical_factors = models_cfg.get("hierarchical_factors", {})
    if not isinstance(hierarchical_factors, dict):
        hierarchical_factors = {}

    fallback: list[str] = []
    factor_fields = (
        ("speed", "Speed"),
        ("math", "Math"),
        ("verbal", "Verbal"),
        ("technical", "Tech"),
    )
    for key, factor_name in factor_fields:
        values = hierarchical_factors.get(key, [])
        if isinstance(values, list) and values:
            if factor_name not in fallback:
                fallback.append(factor_name)
    if fallback:
        return fallback
    return ["Speed", "Math", "Verbal", "Tech"]


def _load_stats_for_factor(
    params: pd.DataFrame,
    latent: pd.DataFrame,
    factor: str,
    cohort: str,
    reference_group: str,
    mean_steps: tuple[str, ...],
    variance_steps: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_rows = pd.DataFrame(columns=["group", "estimate", "se"])
    var_rows = pd.DataFrame(columns=["group", "estimate", "se"])

    if not params.empty:
        mean_subset = _pick_preferred_step(params.copy(), mean_steps)
        mean_df = mean_subset.loc[
            (mean_subset.get("lhs") == factor) & (mean_subset.get("op") == "~1"),
            ["group", "est", "se"],
        ].copy()
        mean_df["estimate"] = pd.to_numeric(mean_df["est"], errors="coerce")
        mean_df["se"] = pd.to_numeric(mean_df["se"], errors="coerce")
        if not mean_df.empty:
            mean_rows = mean_df.dropna(subset=["estimate"])[["group", "estimate", "se"]]

        var_subset = _pick_preferred_step(params.copy(), variance_steps)
        var_df = var_subset.loc[
            (var_subset.get("lhs") == factor)
            & (var_subset.get("rhs") == factor)
            & (var_subset.get("op") == "~~"),
            ["group", "est", "se"],
        ].copy()
        var_df["estimate"] = pd.to_numeric(var_df["est"], errors="coerce")
        var_df["se"] = pd.to_numeric(var_df["se"], errors="coerce")
        if not var_df.empty:
            var_rows = var_df.dropna(subset=["estimate"])[["group", "estimate", "se"]]

    if mean_rows.empty and not latent.empty:
        latent_mean = latent[(latent["cohort"] == cohort) & (latent["factor"] == factor)]
        if not latent_mean.empty:
            mean_rows = latent_mean[["group", "mean"]].rename(columns={"mean": "estimate"})
            mean_rows["se"] = pd.NA

    if var_rows.empty and not latent.empty:
        latent_var = latent[(latent["cohort"] == cohort) & (latent["factor"] == factor)]
        if not latent_var.empty:
            var_rows = latent_var[["group", "var"]].rename(columns={"var": "estimate"})
            var_rows["se"] = pd.NA

    return mean_rows, var_rows


def _coerce_value(value: object) -> float | pd.NAType:
    value_f = _normalize_num(value)
    if value_f is None:
        return pd.NA
    return value_f


def _is_real_finite_number(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean_diff_row(
    factor_stats: pd.DataFrame,
    iq_sd: float,
    reference_group: str,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    if factor_stats.empty:
        return None, None, None

    groups = [_group_key(g) for g in factor_stats["group"].dropna().unique().tolist()]
    means = {_group_key(r["group"]): _coerce_value(r["estimate"]) for _, r in factor_stats.iterrows()}
    se_by_group = {_group_key(r["group"]): _coerce_value(r["se"]) for _, r in factor_stats.iterrows()}
    female_group, male_group = _infer_group_labels(groups, means, reference_group)
    if female_group is None or male_group is None:
        return None, None, None

    male_mean = means.get(male_group)
    female_mean = means.get(female_group)
    if male_mean is pd.NA or female_mean is pd.NA:
        return female_group, male_group, None

    male_se = se_by_group.get(male_group)
    female_se = se_by_group.get(female_group)
    se_diff: float | pd.NAType = pd.NA
    if isinstance(male_se, float) and isinstance(female_se, float):
        se_diff = math.sqrt(male_se**2 + female_se**2)

    diff = male_mean - female_mean
    se_d_g = se_diff
    d_g = diff
    iq_diff = diff * iq_sd
    if isinstance(se_d_g, float):
        try:
            ci_low, ci_high = mean_diff_ci_iq(d_g, se_d_g, iq_sd=iq_sd, z=1.96)
            ci_low_d_g = d_g - 1.96 * se_d_g
            ci_high_d_g = d_g + 1.96 * se_d_g
        except Exception:
            ci_low, ci_high = (pd.NA, pd.NA)
            ci_low_d_g, ci_high_d_g = (pd.NA, pd.NA)
    else:
        ci_low, ci_high = (pd.NA, pd.NA)
        ci_low_d_g, ci_high_d_g = (pd.NA, pd.NA)

    return female_group, male_group, {
        "mean_diff": diff,
        "mean_diff_iq": iq_diff,
        "d_g": d_g,
        "SE_d_g": se_d_g if isinstance(se_d_g, float) else pd.NA,
        "ci_low_d_g": ci_low_d_g,
        "ci_high_d_g": ci_high_d_g,
        "SE": se_d_g * iq_sd if isinstance(se_d_g, float) else pd.NA,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _variance_ratio_row(
    cohort: str,
    var_stats: pd.DataFrame,
    group_counts: dict[str, dict[str, int]],
    reference_group: str,
) -> tuple[str | None, str | None, float | pd.NAType, float | pd.NAType, float | pd.NAType, float | pd.NAType] | None:
    if var_stats.empty:
        return None
    groups = [_group_key(g) for g in var_stats["group"].dropna().unique().tolist()]
    vars_ = {_group_key(r["group"]): _coerce_value(r["estimate"]) for _, r in var_stats.iterrows()}
    female_group, male_group = _infer_group_labels(groups, {g: None for g in vars_}, reference_group)
    if female_group is None or male_group is None:
        return None
    male_var = vars_.get(male_group)
    female_var = vars_.get(female_group)
    if not _is_real_finite_number(male_var) or not _is_real_finite_number(female_var):
        return female_group, male_group, pd.NA, pd.NA, pd.NA, pd.NA
    male_var_f = float(male_var)
    female_var_f = float(female_var)
    if female_var_f <= 0.0 or male_var_f <= 0.0:
        return female_group, male_group, pd.NA, pd.NA, pd.NA, pd.NA

    vr = male_var_f / female_var_f
    n_male = group_counts.get(cohort, {}).get("n_male")
    n_female = group_counts.get(cohort, {}).get("n_female")
    se_log_vr: float | pd.NAType = pd.NA
    if isinstance(n_male, int) and isinstance(n_female, int) and n_male > 1 and n_female > 1:
        se_log_vr = math.sqrt(2.0 / (n_male - 1) + 2.0 / (n_female - 1))

    ci_low: float | pd.NAType = pd.NA
    ci_high: float | pd.NAType = pd.NA
    if isinstance(se_log_vr, float):
        try:
            ci_low, ci_high = variance_ratio_ci(
                male_var=male_var_f,
                female_var=female_var_f,
                se_log_vr=se_log_vr,
                z=1.96,
            )
        except Exception:
            ci_low, ci_high = (pd.NA, pd.NA)
    return female_group, male_group, vr, se_log_vr, ci_low, ci_high


def _load_sex_estimands_for_factor(fit_dir: Path, cohort: str, factor: str) -> pd.DataFrame:
    path = fit_dir / "sex_group_estimands.csv"
    rows = _load_csv_if_exists(path)
    if rows.empty:
        return rows
    required = {
        "model_step",
        "factor",
        "female_group",
        "male_group",
        "mean_female",
        "mean_male",
        "var_female",
        "var_male",
    }
    missing = required - set(rows.columns)
    if missing:
        print(f"[warn] Ignoring {path}: missing columns {', '.join(sorted(missing))}.")
        return pd.DataFrame()
    if "cohort" in rows.columns:
        cohort_rows = rows[rows["cohort"].astype(str).str.strip() == cohort]
        if not cohort_rows.empty:
            rows = cohort_rows
    rows = rows[rows["factor"].astype(str).str.strip() == factor]
    return rows.copy()


def _mean_diff_row_from_estimands(
    estimands: pd.DataFrame,
    iq_sd: float,
    mean_steps: tuple[str, ...],
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    if estimands.empty:
        return None, None, None
    chosen = _pick_preferred_step(estimands.copy(), mean_steps)
    if chosen.empty:
        return None, None, None
    row = chosen.iloc[0]
    female_group = str(row.get("female_group", "")).strip() or None
    male_group = str(row.get("male_group", "")).strip() or None
    male_mean = _normalize_num(row.get("mean_male"))
    female_mean = _normalize_num(row.get("mean_female"))
    if male_mean is None or female_mean is None:
        return female_group, male_group, None

    male_se = _normalize_num(row.get("se_mean_male"))
    female_se = _normalize_num(row.get("se_mean_female"))
    se_diff: float | pd.NAType = pd.NA
    if male_se is not None and female_se is not None:
        se_diff = math.sqrt(male_se**2 + female_se**2)

    diff = float(male_mean - female_mean)
    se_d_g = se_diff
    iq_diff = diff * iq_sd
    if isinstance(se_d_g, float):
        try:
            ci_low, ci_high = mean_diff_ci_iq(diff, se_d_g, iq_sd=iq_sd, z=1.96)
            ci_low_d_g = diff - 1.96 * se_d_g
            ci_high_d_g = diff + 1.96 * se_d_g
        except Exception:
            ci_low, ci_high = (pd.NA, pd.NA)
            ci_low_d_g, ci_high_d_g = (pd.NA, pd.NA)
    else:
        ci_low, ci_high = (pd.NA, pd.NA)
        ci_low_d_g, ci_high_d_g = (pd.NA, pd.NA)

    return female_group, male_group, {
        "mean_diff": diff,
        "mean_diff_iq": iq_diff,
        "d_g": diff,
        "SE_d_g": se_d_g if isinstance(se_d_g, float) else pd.NA,
        "ci_low_d_g": ci_low_d_g,
        "ci_high_d_g": ci_high_d_g,
        "SE": se_d_g * iq_sd if isinstance(se_d_g, float) else pd.NA,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _variance_ratio_row_from_estimands(
    cohort: str,
    estimands: pd.DataFrame,
    group_counts: dict[str, dict[str, int]],
    variance_steps: tuple[str, ...],
) -> tuple[str | None, str | None, float | pd.NAType, float | pd.NAType, float | pd.NAType, float | pd.NAType] | None:
    if estimands.empty:
        return None
    chosen = _pick_preferred_step(estimands.copy(), variance_steps)
    if chosen.empty:
        return None
    row = chosen.iloc[0]
    female_group = str(row.get("female_group", "")).strip() or None
    male_group = str(row.get("male_group", "")).strip() or None
    male_var = _normalize_num(row.get("var_male"))
    female_var = _normalize_num(row.get("var_female"))
    if male_var is None or female_var is None or male_var <= 0.0 or female_var <= 0.0:
        return female_group, male_group, pd.NA, pd.NA, pd.NA, pd.NA

    vr = float(male_var / female_var)
    n_male = group_counts.get(cohort, {}).get("n_male")
    n_female = group_counts.get(cohort, {}).get("n_female")
    se_log_vr: float | pd.NAType = pd.NA
    if isinstance(n_male, int) and isinstance(n_female, int) and n_male > 1 and n_female > 1:
        se_log_vr = math.sqrt(2.0 / (n_male - 1) + 2.0 / (n_female - 1))

    ci_low: float | pd.NAType = pd.NA
    ci_high: float | pd.NAType = pd.NA
    if isinstance(se_log_vr, float):
        try:
            ci_low, ci_high = variance_ratio_ci(
                male_var=float(male_var),
                female_var=float(female_var),
                se_log_vr=se_log_vr,
                z=1.96,
            )
        except Exception:
            ci_low, ci_high = (pd.NA, pd.NA)
    return female_group, male_group, vr, se_log_vr, ci_low, ci_high


def _load_group_counts(root: Path, paths_cfg: dict[str, Any]) -> dict[str, dict[str, int]]:
    sample_path = _resolve_path(paths_cfg["outputs_dir"], root) / "tables" / "sample_counts.csv"
    counts = _load_csv_if_exists(sample_path)
    out: dict[str, dict[str, int]] = {}
    if counts.empty or "cohort" not in counts.columns:
        return out
    if "n_male" not in counts.columns or "n_female" not in counts.columns:
        return out
    for _, row in counts.iterrows():
        cohort = str(row["cohort"])
        try:
            out[cohort] = {"n_male": int(row["n_male"]), "n_female": int(row["n_female"])}
        except (TypeError, ValueError):
            continue
    return out


def _load_invariance_eligibility(root: Path, paths_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    table_path = _resolve_path(paths_cfg["outputs_dir"], root) / "tables" / "invariance_confirmatory_eligibility.csv"
    rows = _load_csv_if_exists(table_path)
    if rows.empty or "cohort" not in rows.columns:
        return {}

    def _as_bool(value: object, default: bool = True) -> bool:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        token = str(value).strip().lower()
        if token in {"true", "1", "yes", "y"}:
            return True
        if token in {"false", "0", "no", "n"}:
            return False
        return default

    def _coerce_field(
        row: dict[str, Any],
        preferred: str,
        legacy: str,
        default: bool = True,
    ) -> bool:
        if preferred in row:
            raw = row.get(preferred)
            if raw is not None and not pd.isna(raw):
                raw_token = str(raw).strip().lower()
                if raw_token and raw_token not in {"na", "nan"}:
                    return _as_bool(raw, default)
        return _as_bool(row.get(legacy), default)

    out: dict[str, dict[str, Any]] = {}
    for row in rows.to_dict(orient="records"):
        cohort = str(row.get("cohort", "")).strip()
        if not cohort:
            continue
        gatekeeping_enabled = _as_bool(row.get("gatekeeping_enabled"), True)
        out[cohort] = {
            "gatekeeping_enabled": gatekeeping_enabled,
            "d_g_eligible": _coerce_field(row, "invariance_ok_for_d", "confirmatory_d_g_eligible", True),
            "vr_g_eligible": _coerce_field(row, "invariance_ok_for_vr", "confirmatory_vr_g_eligible", True),
            "reason_d_g": str(row.get("reason_d_g", "")).strip(),
            "reason_vr_g": str(row.get("reason_vr_g", "")).strip(),
            "partial_refit_used": _as_bool(row.get("partial_refit_used"), False),
            "partial_refit_dir": str(row.get("partial_refit_dir", "")).strip(),
        }
    return out


def _low_information_blocks(root: Path, paths_cfg: dict[str, Any]) -> dict[str, str]:
    sample_path = _resolve_path(paths_cfg["outputs_dir"], root) / "tables" / "sample_counts.csv"
    rows = _load_csv_if_exists(sample_path)
    if rows.empty or "cohort" not in rows.columns or "information_adequacy_status" not in rows.columns:
        return {}

    out: dict[str, str] = {}
    for row in rows.to_dict(orient="records"):
        cohort = str(row.get("cohort", "")).strip()
        if not cohort:
            continue
        status = str(row.get("information_adequacy_status", "")).strip().lower()
        if status in {"", "ok"}:
            continue
        reason = str(row.get("information_adequacy_reason", "")).strip()
        out[cohort] = reason if reason else status
    return out


def _blocked_cohorts_for_confirmatory(root: Path, paths_cfg: dict[str, Any]) -> dict[str, str]:
    status_path = _resolve_path(paths_cfg["outputs_dir"], root) / "tables" / "sem_run_status.csv"
    status = _load_csv_if_exists(status_path)
    if status.empty or "cohort" not in status.columns:
        return {}

    blocked: dict[str, list[str]] = {}
    for row in status.to_dict(orient="records"):
        cohort = str(row.get("cohort", "")).strip()
        if not cohort:
            continue
        reasons: list[str] = []
        warning_status = str(row.get("warning_policy_status", "")).strip().lower()
        if warning_status == "fail":
            reasons.append("warning_policy_status=fail")
        run_status = str(row.get("status", "")).strip().lower()
        if run_status in {"r-failed", "failed", "error", "no-output", "no_output"}:
            reasons.append(f"status={run_status}")
        if reasons:
            blocked.setdefault(cohort, [])
            blocked[cohort].extend(reasons)

    return {cohort: ";".join(sorted(set(reasons))) for cohort, reasons in blocked.items()}


def _build_plot_path(base: Path, name: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def _analysis_tier_from_reasons(reasons: list[str]) -> str:
    if not reasons:
        return "confirmatory"
    lowered = [str(r).strip().lower() for r in reasons if str(r).strip()]
    if any("warning_policy_status=fail" in r or "status=r-failed" in r or "status=failed" in r for r in lowered):
        return "non_inferential"
    if any("information_adequacy:" in r for r in lowered):
        return "exploratory_low_information"
    return "exploratory_sensitivity"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build core results tables and figures from SEM outputs.")
    parser.add_argument(
        "--cohort",
        action="append",
        choices=COHORTS,
        help="Cohort(s) to process.",
    )
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root(),
        help="Project root path.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    cohorts = _cohorts_from_args(args)
    outputs_root = _resolve_path(paths_cfg["outputs_dir"], root)
    tables_dir = outputs_root / "tables"
    figures_dir = outputs_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reference_group = str(models_cfg.get("reference_group", "female"))
    iq_sd = float(models_cfg.get("iq_sd_points", 15.0))
    factor_name_map = {"cnlsy": "g_cnlsy"}
    group_factors = _cohort_group_factors(models_cfg)
    sample_counts = _load_group_counts(root=root, paths_cfg=paths_cfg)
    hard_blocked_cohorts = _blocked_cohorts_for_confirmatory(root=root, paths_cfg=paths_cfg)
    invariance_eligibility = _load_invariance_eligibility(root=root, paths_cfg=paths_cfg)
    low_information = _low_information_blocks(root=root, paths_cfg=paths_cfg)

    g_mean_rows: list[dict[str, Any]] = []
    g_vr_rows: list[dict[str, Any]] = []
    g_vr0_rows: list[dict[str, Any]] = []
    group_factor_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    tier_rows: list[dict[str, Any]] = []

    confirmatory_blocks: dict[str, dict[str, Any]] = {}
    for cohort in cohorts:
        d_reasons: list[str] = []
        vr_reasons: list[str] = []
        hard_reason = hard_blocked_cohorts.get(cohort)
        if hard_reason:
            d_reasons.append(hard_reason)
            vr_reasons.append(hard_reason)

        eligibility = invariance_eligibility.get(cohort)
        if eligibility and bool(eligibility.get("gatekeeping_enabled", True)):
            if not bool(eligibility.get("d_g_eligible", True)):
                detail = str(eligibility.get("reason_d_g", "")).strip() or "invariance_gate_failed"
                d_reasons.append(f"invariance:{detail}")
            if not bool(eligibility.get("vr_g_eligible", True)):
                detail = str(eligibility.get("reason_vr_g", "")).strip() or "invariance_gate_failed"
                vr_reasons.append(f"invariance:{detail}")

        info_reason = low_information.get(cohort)
        if info_reason:
            tag = f"information_adequacy:{info_reason}"
            d_reasons.append(tag)
            vr_reasons.append(tag)

        blocked_d = bool(d_reasons)
        blocked_vr = bool(vr_reasons)
        blocked_all = blocked_d and blocked_vr
        merged_reasons = sorted(set(d_reasons + vr_reasons))

        exclusion_rows.append(
            {
                "cohort": cohort,
                "blocked_confirmatory": blocked_all,
                "blocked_confirmatory_d_g": blocked_d,
                "blocked_confirmatory_vr_g": blocked_vr,
                "reason": ";".join(merged_reasons) if merged_reasons else pd.NA,
                "reason_d_g": ";".join(sorted(set(d_reasons))) if d_reasons else pd.NA,
                "reason_vr_g": ";".join(sorted(set(vr_reasons))) if vr_reasons else pd.NA,
            }
        )
        tier_rows.append(
            {
                "cohort": cohort,
                "estimand": "d_g",
                "analysis_tier": _analysis_tier_from_reasons(d_reasons),
                "blocked_confirmatory": blocked_d,
                "reason": ";".join(sorted(set(d_reasons))) if d_reasons else pd.NA,
            }
        )
        tier_rows.append(
            {
                "cohort": cohort,
                "estimand": "vr_g",
                "analysis_tier": _analysis_tier_from_reasons(vr_reasons),
                "blocked_confirmatory": blocked_vr,
                "reason": ";".join(sorted(set(vr_reasons))) if vr_reasons else pd.NA,
            }
        )
        confirmatory_blocks[cohort] = {
            "blocked_d_g": blocked_d,
            "blocked_vr_g": blocked_vr,
            "reason_d_g": ";".join(sorted(set(d_reasons))),
            "reason_vr_g": ";".join(sorted(set(vr_reasons))),
        }

    for cohort in cohorts:
        block_info = confirmatory_blocks.get(cohort, {})
        blocked_d_g = bool(block_info.get("blocked_d_g", False))
        blocked_vr_g = bool(block_info.get("blocked_vr_g", False))
        if blocked_d_g and blocked_vr_g:
            print(
                f"[warn] Skipping confirmatory extraction for {cohort}: "
                f"{str(block_info.get('reason_d_g', '') or block_info.get('reason_vr_g', '')).strip()}."
            )
            continue
        fit_dir = outputs_root / "model_fits" / cohort
        eligibility = invariance_eligibility.get(cohort, {})
        if bool(eligibility.get("partial_refit_used", False)):
            refit_token = str(eligibility.get("partial_refit_dir", "")).strip()
            if refit_token:
                refit_dir = Path(refit_token)
                if not refit_dir.is_absolute():
                    refit_dir = root / refit_dir
                if refit_dir.exists():
                    fit_dir = refit_dir
                else:
                    print(f"[warn] Partial-refit directory missing for {cohort}: {refit_dir}; using baseline SEM outputs.")
        params = _load_csv_if_exists(fit_dir / "params.csv", required_columns={"cohort", "lhs", "op", "group", "est"})
        latent = _load_csv_if_exists(
            fit_dir / "latent_summary.csv",
            required_columns={"cohort", "group", "factor", "mean", "var"},
        )
        g_factor = factor_name_map.get(cohort, "g")
        g_estimands = _load_sex_estimands_for_factor(fit_dir=fit_dir, cohort=cohort, factor=g_factor)
        if params.empty and latent.empty and g_estimands.empty:
            print(
                f"[warn] Missing SEM outputs for {cohort}: expected params.csv, latent_summary.csv, or sex_group_estimands.csv in {fit_dir}; "
                "writing placeholder/empty result rows."
            )
            continue

        if not g_estimands.empty:
            g_female, g_male, g_row = _mean_diff_row_from_estimands(
                g_estimands,
                iq_sd=iq_sd,
                mean_steps=PREFERRED_MEAN_STEPS,
            )
            vr = _variance_ratio_row_from_estimands(
                cohort=cohort,
                estimands=g_estimands,
                group_counts=sample_counts,
                variance_steps=PREFERRED_VARIANCE_STEPS,
            )
        else:
            g_means, g_vars = _load_stats_for_factor(
                params,
                latent,
                g_factor,
                cohort,
                reference_group,
                mean_steps=PREFERRED_MEAN_STEPS,
                variance_steps=PREFERRED_VARIANCE_STEPS,
            )
            g_female, g_male, g_row = _mean_diff_row(g_means, iq_sd, reference_group)
            vr = _variance_ratio_row(cohort, g_vars, sample_counts, reference_group)

        if g_row is not None and not blocked_d_g:
            g_row = {
                "cohort": cohort,
                "d_g": g_row["d_g"],
                "SE_d_g": g_row["SE_d_g"],
                "ci_low_d_g": g_row["ci_low_d_g"],
                "ci_high_d_g": g_row["ci_high_d_g"],
                "IQ_diff": g_row["mean_diff_iq"],
                "SE": g_row["SE"],
                "ci_low": g_row["ci_low"],
                "ci_high": g_row["ci_high"],
            }
            g_mean_rows.append(g_row)
        if vr is not None and not blocked_vr_g:
            _, _, vr_val, se_log_vr, vr_ci_low, vr_ci_high = vr
            g_vr_rows.append(
                {
                    "cohort": cohort,
                    "VR_g": vr_val,
                    "SE_logVR": se_log_vr,
                    "ci_low": vr_ci_low,
                    "ci_high": vr_ci_high,
                }
            )
            if not g_estimands.empty:
                _, _, metric_mean_row = _mean_diff_row_from_estimands(
                    g_estimands,
                    iq_sd=iq_sd,
                    mean_steps=("metric", "configural", "scalar", "strict"),
                )
                vr0 = _variance_ratio_row_from_estimands(
                    cohort=cohort,
                    estimands=g_estimands,
                    group_counts=sample_counts,
                    variance_steps=("metric", "configural", "scalar", "strict"),
                )
            else:
                g_metric_means, g_metric_vars = _load_stats_for_factor(
                    params,
                    latent,
                    g_factor,
                    cohort,
                    reference_group,
                    mean_steps=("metric", "configural", "scalar", "strict"),
                    variance_steps=("metric", "configural", "scalar", "strict"),
                )
                _, _, metric_mean_row = _mean_diff_row(g_metric_means, iq_sd, reference_group)
                vr0 = _variance_ratio_row(cohort, g_metric_vars, sample_counts, reference_group)
            if vr0 is not None:
                _, _, vr0_val, se_log_vr0, vr0_ci_low, vr0_ci_high = vr0
                g_vr0_rows.append(
                    {
                        "cohort": cohort,
                        "VR0_g": vr0_val,
                        "SE_logVR0": se_log_vr0,
                        "ci_low": vr0_ci_low,
                        "ci_high": vr0_ci_high,
                        "mean_diff_at_vr0": metric_mean_row["mean_diff"] if metric_mean_row else pd.NA,
                    }
                )
        if blocked_d_g and not blocked_vr_g:
            print(f"[warn] Skipping d_g confirmatory extraction for {cohort}: {block_info.get('reason_d_g')}")
        if blocked_vr_g and not blocked_d_g:
            print(f"[warn] Skipping VR_g confirmatory extraction for {cohort}: {block_info.get('reason_vr_g')}")

        if cohort != "cnlsy":
            for factor in group_factors:
                f_means, f_vars = _load_stats_for_factor(
                    params,
                    latent,
                    factor,
                    cohort,
                    reference_group,
                    mean_steps=PREFERRED_MEAN_STEPS,
                    variance_steps=PREFERRED_VARIANCE_STEPS,
                )
                if f_means.empty or f_vars.empty:
                    continue
                _, _, f_diff = _mean_diff_row(f_means, iq_sd, reference_group)
                vr_factor = _variance_ratio_row(cohort, f_vars, sample_counts, reference_group)
                groups = [str(g) for g in f_means["group"].dropna().unique().tolist()]
                mean_lookup = {str(r["group"]): _coerce_value(r["estimate"]) for _, r in f_means.iterrows()}
                f_female, f_male = _infer_group_labels(groups, mean_lookup, reference_group)
                male_mean = mean_lookup.get(f_male, pd.NA) if f_female is not None and f_male is not None else pd.NA
                female_mean = mean_lookup.get(f_female, pd.NA) if f_female is not None else pd.NA
                vr_diff = vr_factor[2] if vr_factor else pd.NA
                se_logvr = vr_factor[3] if vr_factor else pd.NA
                vr_ci_l = vr_factor[4] if vr_factor else pd.NA
                vr_ci_h = vr_factor[5] if vr_factor else pd.NA
                group_factor_rows.append(
                    {
                        "cohort": cohort,
                        "factor": factor,
                        "male_mean": male_mean,
                        "female_mean": female_mean,
                        "mean_diff": f_diff["mean_diff"] if f_diff else pd.NA,
                        "mean_diff_iq": f_diff["mean_diff_iq"] if f_diff else pd.NA,
                        "SE": f_diff["SE"] if f_diff else pd.NA,
                        "mean_ci_low": f_diff["ci_low"] if f_diff else pd.NA,
                        "mean_ci_high": f_diff["ci_high"] if f_diff else pd.NA,
                        "VR": vr_diff,
                        "SE_logVR": se_logvr,
                        "vr_ci_low": vr_ci_l,
                        "vr_ci_high": vr_ci_h,
                    }
                )

    g_mean_df = pd.DataFrame(g_mean_rows, columns=GMEAN_COLUMNS)
    g_vr_df = pd.DataFrame(g_vr_rows, columns=GVR_COLUMNS)
    g_vr0_df = pd.DataFrame(g_vr0_rows, columns=GVR0_COLUMNS)
    group_factor_df = pd.DataFrame(group_factor_rows, columns=GROUP_FACTOR_COLUMNS)
    exclusions_df = pd.DataFrame(exclusion_rows, columns=CONFIRMATORY_EXCLUSIONS_COLUMNS)
    tiers_df = pd.DataFrame(tier_rows, columns=ANALYSIS_TIER_COLUMNS)
    g_mean_df.to_csv(tables_dir / "g_mean_diff.csv", index=False)
    g_vr_df.to_csv(tables_dir / "g_variance_ratio.csv", index=False)
    g_vr0_df.to_csv(tables_dir / "g_variance_ratio_vr0.csv", index=False)
    group_factor_df.to_csv(tables_dir / "group_factor_diffs.csv", index=False)
    exclusions_df.to_csv(tables_dir / "confirmatory_exclusions.csv", index=False)
    tiers_df.to_csv(tables_dir / "analysis_tiers.csv", index=False)

    if not g_mean_df.empty:
        mean_plot_df = g_mean_df.dropna(subset=["IQ_diff", "ci_low", "ci_high"]).rename(
            columns={"cohort": "label", "IQ_diff": "estimate", "ci_low": "ci_lower", "ci_high": "ci_upper"}
        )
        if not mean_plot_df.empty:
            save_forest_plot(
                mean_plot_df,
                _build_plot_path(figures_dir, "g_mean_diff_forestplot.png"),
                label_col="label",
                estimate_col="estimate",
                lower_col="ci_lower",
                upper_col="ci_upper",
                title="g mean difference (male - female)",
                xlabel="IQ points",
            )
    # If stage-20 bootstrap inference exists, also emit bootstrap-based forest plots that
    # cover cohorts where confirmatory baseline mean differences are withheld.
    family_bootstrap_mean = tables_dir / "g_mean_diff_family_bootstrap.csv"
    if family_bootstrap_mean.exists():
        boot = pd.read_csv(family_bootstrap_mean)
        if "status" in boot.columns:
            boot = boot.loc[boot["status"].astype(str).str.strip().eq("computed")].copy()
        boot_plot_df = boot.dropna(subset=["IQ_diff", "ci_low", "ci_high"]).rename(
            columns={"cohort": "label", "IQ_diff": "estimate", "ci_low": "ci_lower", "ci_high": "ci_upper"}
        )
        if not boot_plot_df.empty:
            save_forest_plot(
                boot_plot_df,
                _build_plot_path(figures_dir, "g_mean_diff_family_bootstrap_forestplot.png"),
                label_col="label",
                estimate_col="estimate",
                lower_col="ci_lower",
                upper_col="ci_upper",
                title="g mean difference (male - female), family bootstrap",
                xlabel="IQ points",
            )
    if not g_vr_df.empty:
        vr_plot_df = g_vr_df.dropna(subset=["VR_g", "ci_low", "ci_high"]).rename(
            columns={"cohort": "label", "VR_g": "estimate", "ci_low": "ci_lower", "ci_high": "ci_upper"}
        )
        if not vr_plot_df.empty:
            save_forest_plot(
                vr_plot_df,
                _build_plot_path(figures_dir, "vr_forestplot.png"),
                label_col="label",
                estimate_col="estimate",
                lower_col="ci_lower",
                upper_col="ci_upper",
                title="g variance ratio (male / female)",
                xlabel="Variance ratio",
            )
    family_bootstrap_vr = tables_dir / "g_variance_ratio_family_bootstrap.csv"
    if family_bootstrap_vr.exists():
        boot = pd.read_csv(family_bootstrap_vr)
        if "status" in boot.columns:
            boot = boot.loc[boot["status"].astype(str).str.strip().eq("computed")].copy()
        boot_plot_df = boot.dropna(subset=["VR_g", "ci_low", "ci_high"]).rename(
            columns={"cohort": "label", "VR_g": "estimate", "ci_low": "ci_lower", "ci_high": "ci_upper"}
        )
        if not boot_plot_df.empty:
            save_forest_plot(
                boot_plot_df,
                _build_plot_path(figures_dir, "vr_family_bootstrap_forestplot.png"),
                label_col="label",
                estimate_col="estimate",
                lower_col="ci_lower",
                upper_col="ci_upper",
                title="g variance ratio (male / female), family bootstrap",
                xlabel="Variance ratio",
            )
    proxy_bootstrap_mean = tables_dir / "g_proxy_mean_diff_family_bootstrap.csv"
    if proxy_bootstrap_mean.exists():
        boot = pd.read_csv(proxy_bootstrap_mean)
        if "status" in boot.columns:
            boot = boot.loc[boot["status"].astype(str).str.strip().eq("computed")].copy()
        boot_plot_df = boot.dropna(subset=["IQ_diff", "ci_low", "ci_high"]).rename(
            columns={"cohort": "label", "IQ_diff": "estimate", "ci_low": "ci_lower", "ci_high": "ci_upper"}
        )
        if not boot_plot_df.empty:
            save_forest_plot(
                boot_plot_df,
                _build_plot_path(figures_dir, "g_proxy_mean_diff_family_bootstrap_forestplot.png"),
                label_col="label",
                estimate_col="estimate",
                lower_col="ci_lower",
                upper_col="ci_upper",
                title="Observed g proxy mean difference (male - female), family bootstrap",
                xlabel="IQ points",
            )
    proxy_bootstrap_vr = tables_dir / "g_proxy_variance_ratio_family_bootstrap.csv"
    if proxy_bootstrap_vr.exists():
        boot = pd.read_csv(proxy_bootstrap_vr)
        if "status" in boot.columns:
            boot = boot.loc[boot["status"].astype(str).str.strip().eq("computed")].copy()
        boot_plot_df = boot.dropna(subset=["VR_g", "ci_low", "ci_high"]).rename(
            columns={"cohort": "label", "VR_g": "estimate", "ci_low": "ci_lower", "ci_high": "ci_upper"}
        )
        if not boot_plot_df.empty:
            save_forest_plot(
                boot_plot_df,
                _build_plot_path(figures_dir, "g_proxy_vr_family_bootstrap_forestplot.png"),
                label_col="label",
                estimate_col="estimate",
                lower_col="ci_lower",
                upper_col="ci_upper",
                title="Observed g proxy variance ratio (male / female), family bootstrap",
                xlabel="Variance ratio",
            )
    if not group_factor_df.empty:
        gap_plot = group_factor_df.assign(
            cohort_factor=group_factor_df["cohort"] + ":" + group_factor_df["factor"]
        )
        gap_mean_df = gap_plot.dropna(subset=["mean_diff_iq"])
        if not gap_mean_df.empty:
            save_bar_plot(
                gap_mean_df,
                _build_plot_path(figures_dir, "group_factor_gaps.png"),
                category_col="cohort_factor",
                value_col="mean_diff_iq",
                title="Group factor gaps (IQ scale)",
                xlabel="Cohort:factor",
                ylabel="IQ points (male - female)",
            )
        gap_vr_df = gap_plot.dropna(subset=["VR"])
        if not gap_vr_df.empty:
            save_bar_plot(
                gap_vr_df,
                _build_plot_path(figures_dir, "group_factor_vr.png"),
                category_col="cohort_factor",
                value_col="VR",
                title="Group factor variance ratios",
                xlabel="Cohort:factor",
                ylabel="Variance ratio (male/female)",
            )

    print(f"[ok] wrote {outputs_root/'tables'/'g_mean_diff.csv'}")
    print(f"[ok] wrote {outputs_root/'tables'/'g_variance_ratio.csv'}")
    print(f"[ok] wrote {outputs_root/'tables'/'g_variance_ratio_vr0.csv'}")
    print(f"[ok] wrote {outputs_root/'tables'/'group_factor_diffs.csv'}")
    print(f"[ok] wrote {outputs_root/'tables'/'confirmatory_exclusions.csv'}")
    print(f"[ok] wrote {outputs_root/'tables'/'analysis_tiers.csv'}")
    print(f"[ok] wrote {outputs_root/'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
