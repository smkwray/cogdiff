#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
from nls_pipeline.sem import hierarchical_subtests

COHORTS: tuple[str, ...] = ("nlsy79", "nlsy97", "cnlsy")
WEIGHT_PATTERN_TOKENS: tuple[str, ...] = ("weight", "wgt", "wt", "finalwt", "pweight", "sample_weight")
DEFAULT_MIN_POSITIVE_ROWS = 5
DEFAULT_MIN_POSITIVE_SHARE = 0.0
DEFAULT_MIN_EFFECTIVE_N_TOTAL = 2
DEFAULT_MIN_EFFECTIVE_N_BY_SEX = 2


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _safe_float(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


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
    z_df = pd.DataFrame({col: _zscore(df[col]) for col in existing}, index=df.index)
    return z_df.mean(axis=1, skipna=False)


def _sex_labels(series: pd.Series) -> pd.Series:
    vals = series.astype(str).str.strip().str.lower()
    out = pd.Series(["unknown"] * len(vals), index=vals.index, dtype="object")
    out[vals.isin({"m", "male", "1", "man", "boy"})] = "male"
    out[vals.isin({"f", "female", "2", "woman", "girl"})] = "female"
    return out


def _candidate_weight_cols(
    *,
    df: pd.DataFrame,
    cohort: str,
    configured_candidates: dict[str, list[str]],
) -> list[str]:
    candidates = [str(x) for x in configured_candidates.get(cohort, []) if str(x).strip()]
    if not candidates:
        pattern_hits = [
            col
            for col in df.columns
            if any(token in str(col).strip().lower() for token in WEIGHT_PATTERN_TOKENS)
        ]
        candidates = [str(x) for x in pattern_hits]
    deduped: list[str] = []
    seen: set[str] = set()
    for col in candidates:
        token = str(col).strip()
        if token and token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def _detect_weight_col(
    *,
    df: pd.DataFrame,
    candidates: list[str],
    min_positive_rows: int,
) -> str | None:
    min_positive_rows = max(1, int(min_positive_rows))
    for col in candidates:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        valid = values.replace([np.inf, -np.inf], np.nan).dropna()
        valid = valid[valid > 0]
        if len(valid) >= min_positive_rows:
            return col
    return None


def _weight_quality_for_col(
    *,
    df: pd.DataFrame,
    cohort: str,
    weight_col: str | None,
    selected_col: str | None,
    min_positive_rows: int,
    min_positive_share: float,
    min_effective_n_total: int,
    min_effective_n_by_sex: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "cohort": cohort,
        "weight_col": weight_col if weight_col is not None else pd.NA,
        "selected_weight_col": selected_col if selected_col is not None else pd.NA,
        "selected": bool(weight_col is not None and selected_col is not None and weight_col == selected_col),
        "min_positive_rows_threshold": int(min_positive_rows),
        "present_in_data": bool(weight_col is not None and weight_col in df.columns),
        "n_rows_total": int(len(df)),
        "n_non_missing": 0,
        "n_positive": 0,
        "positive_share": pd.NA,
        "n_pos_with_g_and_sex": 0,
        "n_male_pos_with_g": 0,
        "n_female_pos_with_g": 0,
        "neff_total": pd.NA,
        "neff_male": pd.NA,
        "neff_female": pd.NA,
        "weight_min_pos": pd.NA,
        "weight_p01_pos": pd.NA,
        "weight_p50_pos": pd.NA,
        "weight_p99_pos": pd.NA,
        "weight_max_pos": pd.NA,
        "selection_reason": "not_selected",
        "min_positive_share_threshold": float(min_positive_share),
        "min_effective_n_total_threshold": int(min_effective_n_total),
        "min_effective_n_by_sex_threshold": int(min_effective_n_by_sex),
        "quality_gate_checked": bool(weight_col is not None and selected_col is not None and weight_col == selected_col),
        "quality_gate_passed": pd.NA,
        "quality_gate_reason": "not_selected",
    }
    if weight_col is None:
        out["selection_reason"] = "no_candidate_columns"
        return out
    if weight_col not in df.columns:
        out["selection_reason"] = "missing_from_dataframe"
        return out

    raw = pd.to_numeric(df[weight_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    non_missing = raw.dropna()
    positive = non_missing[non_missing > 0]
    out["n_non_missing"] = int(non_missing.shape[0])
    out["n_positive"] = int(positive.shape[0])
    if len(df) > 0:
        out["positive_share"] = float(out["n_positive"] / float(len(df)))
    if not positive.empty:
        out["weight_min_pos"] = float(positive.min())
        out["weight_p01_pos"] = float(positive.quantile(0.01))
        out["weight_p50_pos"] = float(positive.quantile(0.50))
        out["weight_p99_pos"] = float(positive.quantile(0.99))
        out["weight_max_pos"] = float(positive.max())

    work = pd.DataFrame(
        {
            "sex_label": df.get("sex_label", pd.Series(["unknown"] * len(df), index=df.index)),
            "g_proxy": pd.to_numeric(df.get("g_proxy"), errors="coerce"),
            "weight": raw,
        },
        index=df.index,
    ).replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["sex_label", "g_proxy", "weight"])
    work = work.loc[work["weight"] > 0].copy()
    out["n_pos_with_g_and_sex"] = int(len(work))
    if not work.empty:
        male = work.loc[work["sex_label"] == "male", "weight"].to_numpy(dtype=float)
        female = work.loc[work["sex_label"] == "female", "weight"].to_numpy(dtype=float)
        out["n_male_pos_with_g"] = int(len(male))
        out["n_female_pos_with_g"] = int(len(female))
        out["neff_total"] = _effective_n(work["weight"].to_numpy(dtype=float))
        out["neff_male"] = _effective_n(male) if len(male) > 0 else pd.NA
        out["neff_female"] = _effective_n(female) if len(female) > 0 else pd.NA

    if out["selected"]:
        out["selection_reason"] = "selected_first_passing_candidate"
    else:
        if out["n_positive"] < int(min_positive_rows):
            out["selection_reason"] = "below_positive_threshold"
        elif selected_col is None:
            out["selection_reason"] = "no_candidate_passed_threshold"
        else:
            out["selection_reason"] = "not_first_passing_candidate"
    if bool(out["quality_gate_checked"]):
        gate_reason = _weight_quality_gate_failure_reason(
            positive_share=_safe_float(out.get("positive_share")),
            neff_total=_safe_float(out.get("neff_total")),
            neff_male=_safe_float(out.get("neff_male")),
            neff_female=_safe_float(out.get("neff_female")),
            min_positive_share=min_positive_share,
            min_effective_n_total=min_effective_n_total,
            min_effective_n_by_sex=min_effective_n_by_sex,
        )
        if gate_reason is None:
            out["quality_gate_passed"] = True
            out["quality_gate_reason"] = "ok"
        else:
            out["quality_gate_passed"] = False
            out["quality_gate_reason"] = gate_reason
    return out


def _effective_n(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w * w))
    if denom <= 0.0:
        return float("nan")
    return float((np.sum(w) ** 2) / denom)


def _weight_quality_gate_failure_reason(
    *,
    positive_share: float | None,
    neff_total: float | None,
    neff_male: float | None,
    neff_female: float | None,
    min_positive_share: float,
    min_effective_n_total: int,
    min_effective_n_by_sex: int,
) -> str | None:
    if positive_share is None or not math.isfinite(float(positive_share)):
        return "invalid_positive_share"
    if float(positive_share) < float(min_positive_share):
        return "positive_share_below_threshold"
    if neff_total is None or not math.isfinite(float(neff_total)):
        return "invalid_effective_n_total"
    if float(neff_total) < float(min_effective_n_total):
        return "effective_n_total_below_threshold"
    if neff_male is None or neff_female is None:
        return "invalid_effective_n_by_sex"
    if not math.isfinite(float(neff_male)) or not math.isfinite(float(neff_female)):
        return "invalid_effective_n_by_sex"
    if float(neff_male) < float(min_effective_n_by_sex) or float(neff_female) < float(min_effective_n_by_sex):
        return "effective_n_by_sex_below_threshold"
    return None


def _weighted_mean_var(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.size == 0 or w.size == 0:
        return float("nan"), float("nan")
    mu = float(np.sum(w * x) / np.sum(w))
    var = float(np.sum(w * ((x - mu) ** 2)) / np.sum(w))
    return mu, var


def _estimate_weighted(
    *,
    df: pd.DataFrame,
    weight_col: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    work = df.copy()
    work["weight"] = pd.to_numeric(work[weight_col], errors="coerce")
    work["g_proxy"] = pd.to_numeric(work["g_proxy"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["sex_label", "g_proxy", "weight"]).copy()
    work = work.loc[work["weight"] > 0].copy()
    if work.empty:
        return None, None, "no_positive_weight_rows"

    male = work.loc[work["sex_label"] == "male", ["g_proxy", "weight"]].copy()
    female = work.loc[work["sex_label"] == "female", ["g_proxy", "weight"]].copy()
    if len(male) < 2 or len(female) < 2:
        return None, None, "insufficient_sex_rows_after_weight_filter"

    mu_m, var_m = _weighted_mean_var(male["g_proxy"].to_numpy(), male["weight"].to_numpy())
    mu_f, var_f = _weighted_mean_var(female["g_proxy"].to_numpy(), female["weight"].to_numpy())
    if not math.isfinite(var_m) or not math.isfinite(var_f) or var_m <= 0.0 or var_f <= 0.0:
        return None, None, "nonpositive_weighted_variance"

    neff_m = _effective_n(male["weight"].to_numpy())
    neff_f = _effective_n(female["weight"].to_numpy())
    if not math.isfinite(neff_m) or not math.isfinite(neff_f) or neff_m <= 1.0 or neff_f <= 1.0:
        return None, None, "invalid_effective_sample_size"

    pooled_var = (((neff_m - 1.0) * var_m) + ((neff_f - 1.0) * var_f)) / max((neff_m + neff_f - 2.0), 1.0)
    if not math.isfinite(pooled_var) or pooled_var <= 0.0:
        return None, None, "nonpositive_weighted_pooled_variance"
    pooled_sd = math.sqrt(pooled_var)
    d_g = float(mu_m - mu_f) / pooled_sd
    se_m = math.sqrt(var_m / neff_m)
    se_f = math.sqrt(var_f / neff_f)
    se_d = math.sqrt(se_m**2 + se_f**2)
    ci_low_d = float(d_g - 1.96 * se_d)
    ci_high_d = float(d_g + 1.96 * se_d)

    vr = float(var_m / var_f)
    log_vr = math.log(vr)
    se_log_vr = math.sqrt(max(0.0, (2.0 / max(neff_m - 1.0, 1.0)) + (2.0 / max(neff_f - 1.0, 1.0))))
    ci_low_vr = float(math.exp(log_vr - 1.96 * se_log_vr))
    ci_high_vr = float(math.exp(log_vr + 1.96 * se_log_vr))

    mean_row = {
        "d_g": d_g,
        "SE_d_g": se_d,
        "ci_low_d_g": ci_low_d,
        "ci_high_d_g": ci_high_d,
        "IQ_diff": d_g * 15.0,
        "SE": se_d * 15.0,
        "ci_low": ci_low_d * 15.0,
        "ci_high": ci_high_d * 15.0,
    }
    vr_row = {
        "VR_g": vr,
        "SE_logVR": se_log_vr,
        "ci_low": ci_low_vr,
        "ci_high": ci_high_vr,
    }
    return mean_row, vr_row, None


def run_weights_rerun(
    *,
    root: Path,
    variant_token: str,
) -> dict[str, Any]:
    if str(variant_token).strip() != "weighted":
        raise ValueError("variant-token must be weighted")

    paths_cfg = load_yaml(root / "config/paths.yml")
    models_cfg = load_yaml(root / "config/models.yml")
    robustness_cfg = load_yaml(root / "config/robustness.yml") if (root / "config/robustness.yml").exists() else {}
    weights_cfg = robustness_cfg.get("weights_rerun", {}) if isinstance(robustness_cfg, dict) else {}
    if not isinstance(weights_cfg, dict):
        weights_cfg = {}
    candidate_cols_cfg = weights_cfg.get("candidate_cols", {})
    if not isinstance(candidate_cols_cfg, dict):
        candidate_cols_cfg = {}
    min_positive_rows = int(
        max(
            1,
            round(
                pd.to_numeric(
                    pd.Series([weights_cfg.get("min_positive_rows", DEFAULT_MIN_POSITIVE_ROWS)]),
                    errors="coerce",
                ).fillna(DEFAULT_MIN_POSITIVE_ROWS).iloc[0]
            ),
        )
    )
    min_positive_share = float(
        max(
            0.0,
            pd.to_numeric(
                pd.Series([weights_cfg.get("min_positive_share", DEFAULT_MIN_POSITIVE_SHARE)]),
                errors="coerce",
            )
            .fillna(DEFAULT_MIN_POSITIVE_SHARE)
            .iloc[0],
        )
    )
    min_effective_n_total = int(
        max(
            2,
            round(
                pd.to_numeric(
                    pd.Series([weights_cfg.get("min_effective_n_total", DEFAULT_MIN_EFFECTIVE_N_TOTAL)]),
                    errors="coerce",
                )
                .fillna(DEFAULT_MIN_EFFECTIVE_N_TOTAL)
                .iloc[0]
            ),
        )
    )
    min_effective_n_by_sex = int(
        max(
            2,
            round(
                pd.to_numeric(
                    pd.Series([weights_cfg.get("min_effective_n_by_sex", DEFAULT_MIN_EFFECTIVE_N_BY_SEX)]),
                    errors="coerce",
                )
                .fillna(DEFAULT_MIN_EFFECTIVE_N_BY_SEX)
                .iloc[0]
            ),
        )
    )
    configured_candidates = {
        str(cohort): [str(x) for x in cols] if isinstance(cols, list) else []
        for cohort, cols in candidate_cols_cfg.items()
    }

    processed_dir = _resolve_path(paths_cfg.get("processed_dir", "data/processed"), root)
    outputs_dir = _resolve_path(paths_cfg.get("outputs_dir", "outputs"), root)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    g_mean_rows: list[dict[str, Any]] = []
    g_vr_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for cohort in COHORTS:
        source_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not source_path.exists():
            source_path = processed_dir / f"{cohort}_cfa.csv"
        if not source_path.exists():
            continue

        df = pd.read_csv(source_path, low_memory=False)
        if "sex" not in df.columns:
            continue

        indicators = (
            [str(x) for x in models_cfg.get("cnlsy_single_factor", [])]
            if cohort == "cnlsy"
            else hierarchical_subtests(models_cfg)
        )
        df = df.copy()
        df["sex_label"] = _sex_labels(df["sex"])
        df["g_proxy"] = _composite_score(df, indicators)
        candidates = _candidate_weight_cols(df=df, cohort=cohort, configured_candidates=configured_candidates)
        weight_col = _detect_weight_col(df=df, candidates=candidates, min_positive_rows=min_positive_rows)

        selected_quality: dict[str, Any] | None = None
        if candidates:
            for candidate in candidates:
                quality_row = _weight_quality_for_col(
                    df=df,
                    cohort=cohort,
                    weight_col=candidate,
                    selected_col=weight_col,
                    min_positive_rows=min_positive_rows,
                    min_positive_share=min_positive_share,
                    min_effective_n_total=min_effective_n_total,
                    min_effective_n_by_sex=min_effective_n_by_sex,
                )
                quality_rows.append(quality_row)
                if weight_col is not None and str(candidate) == str(weight_col):
                    selected_quality = quality_row
        else:
            quality_row = _weight_quality_for_col(
                df=df,
                cohort=cohort,
                weight_col=None,
                selected_col=weight_col,
                min_positive_rows=min_positive_rows,
                min_positive_share=min_positive_share,
                min_effective_n_total=min_effective_n_total,
                min_effective_n_by_sex=min_effective_n_by_sex,
            )
            quality_rows.append(quality_row)
            selected_quality = quality_row if bool(quality_row.get("selected", False)) else None

        if weight_col is None:
            status = "not_feasible"
            reason = "weight_column_unavailable"
            mean_payload = {}
            vr_payload = {}
        elif selected_quality is not None and selected_quality.get("quality_gate_passed") is False:
            status = "not_feasible"
            gate_reason = str(selected_quality.get("quality_gate_reason", "")).strip() or "quality_gate_failed"
            reason = f"weight_quality_gate:{gate_reason}"
            mean_payload = {}
            vr_payload = {}
        else:
            mean_payload, vr_payload, error_reason = _estimate_weighted(df=df, weight_col=weight_col)
            if mean_payload is None or vr_payload is None:
                status = "not_feasible"
                reason = str(error_reason or "weight_estimation_failed")
                mean_payload = {}
                vr_payload = {}
            else:
                status = "computed"
                reason = ""

        g_mean_rows.append(
            {
                "cohort": cohort,
                "status": status,
                "reason": reason if reason else pd.NA,
                "weight_mode": "weighted",
                "weight_col": weight_col if weight_col is not None else pd.NA,
                **{
                    "d_g": mean_payload.get("d_g", pd.NA),
                    "SE_d_g": mean_payload.get("SE_d_g", pd.NA),
                    "ci_low_d_g": mean_payload.get("ci_low_d_g", pd.NA),
                    "ci_high_d_g": mean_payload.get("ci_high_d_g", pd.NA),
                    "IQ_diff": mean_payload.get("IQ_diff", pd.NA),
                    "SE": mean_payload.get("SE", pd.NA),
                    "ci_low": mean_payload.get("ci_low", pd.NA),
                    "ci_high": mean_payload.get("ci_high", pd.NA),
                },
            }
        )
        g_vr_rows.append(
            {
                "cohort": cohort,
                "status": status,
                "reason": reason if reason else pd.NA,
                "weight_mode": "weighted",
                "weight_col": weight_col if weight_col is not None else pd.NA,
                **{
                    "VR_g": vr_payload.get("VR_g", pd.NA),
                    "SE_logVR": vr_payload.get("SE_logVR", pd.NA),
                    "ci_low": vr_payload.get("ci_low", pd.NA),
                    "ci_high": vr_payload.get("ci_high", pd.NA),
                },
            }
        )

    if not g_mean_rows or not g_vr_rows:
        raise ValueError("No rows produced for weighted rerun.")

    mean_path = tables_dir / "g_mean_diff_weighted.csv"
    vr_path = tables_dir / "g_variance_ratio_weighted.csv"
    quality_path = tables_dir / "weights_quality_diagnostics.csv"
    pd.DataFrame(g_mean_rows).to_csv(mean_path, index=False)
    pd.DataFrame(g_vr_rows).to_csv(vr_path, index=False)
    pd.DataFrame(quality_rows).to_csv(quality_path, index=False)

    mean_df = pd.DataFrame(g_mean_rows)
    manifest = {
        "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "variant_token": "weighted",
        "artifacts": {
            "g_mean_diff": relative_path(root, mean_path),
            "g_variance_ratio": relative_path(root, vr_path),
            "weights_quality_diagnostics": relative_path(root, quality_path),
        },
        "cohorts": [str(x) for x in mean_df["cohort"].astype(str).tolist()],
        "computed_count": int((mean_df["status"].astype(str) == "computed").sum()),
        "not_feasible_count": int((mean_df["status"].astype(str) == "not_feasible").sum()),
        "min_positive_rows": int(min_positive_rows),
        "min_positive_share": float(min_positive_share),
        "min_effective_n_total": int(min_effective_n_total),
        "min_effective_n_by_sex": int(min_effective_n_by_sex),
    }
    manifest_path = tables_dir / "weights_rerun_manifest_weighted.json"
    dump_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run weighted robustness rerun.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--variant-token", required=True, choices=("weighted",))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        manifest = run_weights_rerun(
            root=root,
            variant_token=str(args.variant_token),
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] weights rerun complete for variant={manifest['variant_token']}")
    print(f"[ok] wrote {manifest['artifacts']['g_mean_diff']}")
    print(f"[ok] wrote {manifest['artifacts']['g_variance_ratio']}")
    print(f"[ok] wrote {manifest['artifacts']['weights_quality_diagnostics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
