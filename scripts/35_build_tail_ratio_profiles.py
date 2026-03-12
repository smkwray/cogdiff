#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

DEFAULT_TAIL_QUANTILES = (0.95, 0.99)
DEFAULT_N_BOOTSTRAP = 200
DEFAULT_SEED = 20260224

OUTPUT_COLUMNS = [
    "cohort",
    "tail",
    "quantile",
    "threshold",
    "status",
    "reason",
    "n_total",
    "n_male",
    "n_female",
    "n_tail_total",
    "n_tail_male",
    "n_tail_female",
    "male_tail_rate",
    "female_tail_rate",
    "male_female_tail_rate_ratio",
    "ci_low",
    "ci_high",
    "n_bootstrap",
    "n_bootstrap_success",
    "source_data",
]


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _as_label(value: Any) -> str:
    if isinstance(value, bool):
        return "NO" if value is False else "YES"
    return str(value)


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def _normalize_sex(value: Any) -> str:
    token = str(value).strip().lower()
    if token in {"m", "male", "1", "man", "boy"}:
        return "male"
    if token in {"f", "female", "2", "woman", "girl"}:
        return "female"
    return "unknown"


def _zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mean = vals.mean(skipna=True)
    sd = vals.std(skipna=True, ddof=1)
    if pd.isna(sd) or float(sd) <= 0.0:
        return pd.Series([pd.NA] * len(vals), index=vals.index, dtype="float64")
    return (vals - mean) / sd


def _g_proxy(df: pd.DataFrame, indicators: list[str]) -> pd.Series:
    existing = [col for col in indicators if col in df.columns]
    if not existing:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")
    z = pd.DataFrame({col: _zscore(df[col]) for col in existing}, index=df.index)
    return z.mean(axis=1, skipna=False)


def _tail_ratio_once(values: pd.Series, sex: pd.Series, quantile: float, tail: str) -> dict[str, Any]:
    clean = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "sex": sex.map(_normalize_sex)}).dropna()
    clean = clean[clean["sex"].isin({"male", "female"})].copy()
    n_total = int(len(clean))
    if n_total == 0:
        return {
            "status": "not_feasible",
            "reason": "no_valid_rows",
            "threshold": pd.NA,
            "n_total": 0,
            "n_male": 0,
            "n_female": 0,
            "n_tail_total": 0,
            "n_tail_male": 0,
            "n_tail_female": 0,
            "male_tail_rate": pd.NA,
            "female_tail_rate": pd.NA,
            "male_female_tail_rate_ratio": pd.NA,
        }

    male = clean[clean["sex"] == "male"]
    female = clean[clean["sex"] == "female"]
    n_male = int(len(male))
    n_female = int(len(female))
    if n_male < 2 or n_female < 2:
        return {
            "status": "not_feasible",
            "reason": "insufficient_group_n",
            "threshold": pd.NA,
            "n_total": n_total,
            "n_male": n_male,
            "n_female": n_female,
            "n_tail_total": 0,
            "n_tail_male": 0,
            "n_tail_female": 0,
            "male_tail_rate": pd.NA,
            "female_tail_rate": pd.NA,
            "male_female_tail_rate_ratio": pd.NA,
        }

    if tail == "top":
        threshold = float(clean["value"].quantile(quantile))
        in_tail = clean["value"] >= threshold
    else:
        threshold = float(clean["value"].quantile(1.0 - quantile))
        in_tail = clean["value"] <= threshold

    tail_df = clean[in_tail]
    n_tail_total = int(len(tail_df))
    n_tail_male = int((tail_df["sex"] == "male").sum())
    n_tail_female = int((tail_df["sex"] == "female").sum())
    male_tail_rate = n_tail_male / float(n_male) if n_male > 0 else pd.NA
    female_tail_rate = n_tail_female / float(n_female) if n_female > 0 else pd.NA
    if not isinstance(female_tail_rate, float) or female_tail_rate <= 0.0:
        return {
            "status": "not_feasible",
            "reason": "zero_female_tail_rate",
            "threshold": threshold,
            "n_total": n_total,
            "n_male": n_male,
            "n_female": n_female,
            "n_tail_total": n_tail_total,
            "n_tail_male": n_tail_male,
            "n_tail_female": n_tail_female,
            "male_tail_rate": male_tail_rate,
            "female_tail_rate": female_tail_rate,
            "male_female_tail_rate_ratio": pd.NA,
        }
    ratio = float(male_tail_rate / female_tail_rate)
    return {
        "status": "computed",
        "reason": pd.NA,
        "threshold": threshold,
        "n_total": n_total,
        "n_male": n_male,
        "n_female": n_female,
        "n_tail_total": n_tail_total,
        "n_tail_male": n_tail_male,
        "n_tail_female": n_tail_female,
        "male_tail_rate": male_tail_rate,
        "female_tail_rate": female_tail_rate,
        "male_female_tail_rate_ratio": ratio,
    }


def _bootstrap_ci(
    *,
    values: pd.Series,
    sex: pd.Series,
    quantile: float,
    tail: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float | pd.NAType, float | pd.NAType, int]:
    clean = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "sex": sex.map(_normalize_sex)}).dropna()
    clean = clean[clean["sex"].isin({"male", "female"})].copy()
    if clean.empty:
        return pd.NA, pd.NA, 0

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    n_rows = len(clean)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_rows, size=n_rows)
        boot = clean.iloc[idx]
        row = _tail_ratio_once(boot["value"], boot["sex"], quantile=quantile, tail=tail)
        ratio = row.get("male_female_tail_rate_ratio")
        if isinstance(ratio, float) and math.isfinite(ratio):
            samples.append(ratio)
    if not samples:
        return pd.NA, pd.NA, 0
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return lo, hi, int(len(samples))


def run_tail_ratio_profiles(
    *,
    root: Path,
    cohorts: list[str],
    quantiles: tuple[float, ...] = DEFAULT_TAIL_QUANTILES,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    output_path: Path = Path("outputs/tables/tail_ratio_profiles.csv"),
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config" / "paths.yml")
    models_cfg = load_yaml(root / "config" / "models.yml")
    processed_dir = _resolve_path(paths_cfg.get("processed_dir", "data/processed"), root)

    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        cohort_cfg = load_yaml(root / COHORT_CONFIGS[cohort])
        sample_cfg = cohort_cfg.get("sample_construct", {}) if isinstance(cohort_cfg.get("sample_construct", {}), dict) else {}
        sex_col = str(sample_cfg.get("sex_col", "sex"))
        indicators = [_as_label(x) for x in sample_cfg.get("subtests", [])]
        if not indicators:
            indicators = [_as_label(x) for x in (models_cfg.get("cnlsy_single_factor", []) if cohort == "cnlsy" else hierarchical_subtests(models_cfg))]

        data_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not data_path.exists():
            data_path = processed_dir / f"{cohort}_cfa.csv"
        if not data_path.exists():
            rows.append(
                {
                    "cohort": cohort,
                    "tail": pd.NA,
                    "quantile": pd.NA,
                    "status": "not_feasible",
                    "reason": "missing_source_data",
                    "source_data": "",
                }
            )
            continue

        source_data = str(data_path.relative_to(root))
        df = pd.read_csv(data_path, low_memory=False)
        if sex_col not in df.columns:
            rows.append(
                {
                    "cohort": cohort,
                    "tail": pd.NA,
                    "quantile": pd.NA,
                    "status": "not_feasible",
                    "reason": "missing_sex_column",
                    "source_data": source_data,
                }
            )
            continue

        values = _g_proxy(df, indicators)
        sex = df[sex_col]
        for q in quantiles:
            for tail in ("top", "bottom"):
                one = _tail_ratio_once(values, sex, quantile=q, tail=tail)
                ci_low, ci_high, n_boot_ok = _bootstrap_ci(
                    values=values,
                    sex=sex,
                    quantile=q,
                    tail=tail,
                    n_bootstrap=n_bootstrap,
                    seed=seed + int(round(q * 1000)) + (17 if tail == "top" else 31),
                )
                row = {
                    "cohort": cohort,
                    "tail": tail,
                    "quantile": q,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_bootstrap": int(n_bootstrap),
                    "n_bootstrap_success": int(n_boot_ok),
                    "source_data": source_data,
                    **one,
                }
                for col in OUTPUT_COLUMNS:
                    row.setdefault(col, pd.NA)
                rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=OUTPUT_COLUMNS)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS].copy()

    output_file = output_path if output_path.is_absolute() else root / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build male:female tail-rate ratio profiles with bootstrap CIs.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP, help="Bootstrap replicates.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Bootstrap seed.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables/tail_ratio_profiles.csv"),
        help="Output CSV path (relative to project-root if not absolute).",
    )
    args = parser.parse_args()

    if int(args.n_bootstrap) < 1:
        print("[error] --n-bootstrap must be >= 1", file=sys.stderr)
        return 1

    root = Path(args.project_root).resolve()
    cohorts = _cohorts_from_args(args)
    try:
        out = run_tail_ratio_profiles(
            root=root,
            cohorts=cohorts,
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed),
            output_path=args.output_path,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    output_file = args.output_path if args.output_path.is_absolute() else root / args.output_path
    computed = int((out["status"] == "computed").sum()) if "status" in out.columns else 0
    print(f"[ok] wrote {output_file}")
    print(f"[ok] computed rows: {computed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
