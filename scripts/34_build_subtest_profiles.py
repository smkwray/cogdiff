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
from nls_pipeline.plots import save_forest_plot

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}

OUTPUT_COLUMNS = [
    "cohort",
    "subtest",
    "status",
    "reason",
    "n_male",
    "n_female",
    "mean_male",
    "mean_female",
    "var_male",
    "var_female",
    "d_subtest",
    "SE_d_subtest",
    "ci_low_d_subtest",
    "ci_high_d_subtest",
    "vr_subtest",
    "log_vr_subtest",
    "SE_log_vr_subtest",
    "ci_low_vr_subtest",
    "ci_high_vr_subtest",
    "ci_low_log_vr_subtest",
    "ci_high_log_vr_subtest",
    "source_data",
]


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _as_label(value: Any) -> str:
    # YAML 1.1 may parse "NO" as False.
    if isinstance(value, bool):
        return "NO" if value is False else "YES"
    return str(value)


def _cohorts_from_args(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


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
    if token in {"m", "male", "1", "man", "boy"}:
        return "male"
    if token in {"f", "female", "2", "woman", "girl"}:
        return "female"
    return "unknown"


def _empty_row(cohort: str, subtest: str, reason: str, source_data: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "subtest": subtest,
        "status": "not_feasible",
        "reason": reason,
        "n_male": 0,
        "n_female": 0,
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _compute_subtest_stats(
    *,
    cohort: str,
    subtest: str,
    values: pd.Series,
    sex_series: pd.Series,
    source_data: str,
) -> dict[str, Any]:
    sex_norm = sex_series.map(_normalize_sex)
    numeric = pd.to_numeric(values, errors="coerce")
    male = numeric[sex_norm == "male"].dropna()
    female = numeric[sex_norm == "female"].dropna()

    n_male = int(len(male))
    n_female = int(len(female))
    if n_male < 2 or n_female < 2:
        return _empty_row(cohort, subtest, "insufficient_group_n", source_data) | {
            "n_male": n_male,
            "n_female": n_female,
        }

    mean_male = float(male.mean())
    mean_female = float(female.mean())
    var_male = float(male.var(ddof=1))
    var_female = float(female.var(ddof=1))
    if var_male <= 0.0 or var_female <= 0.0:
        return _empty_row(cohort, subtest, "nonpositive_group_variance", source_data) | {
            "n_male": n_male,
            "n_female": n_female,
            "mean_male": mean_male,
            "mean_female": mean_female,
            "var_male": var_male,
            "var_female": var_female,
        }

    diff = mean_male - mean_female
    pooled_sd = math.sqrt((var_male + var_female) / 2.0)
    if pooled_sd <= 0.0:
        return _empty_row(cohort, subtest, "nonpositive_pooled_sd", source_data) | {
            "n_male": n_male,
            "n_female": n_female,
            "mean_male": mean_male,
            "mean_female": mean_female,
            "var_male": var_male,
            "var_female": var_female,
        }

    d_subtest = diff / pooled_sd
    n_total = n_male + n_female
    se_d = math.sqrt((n_total / (n_male * n_female)) + (d_subtest**2 / (2.0 * (n_total - 2))))
    ci_low_d = d_subtest - 1.96 * se_d
    ci_high_d = d_subtest + 1.96 * se_d

    vr = var_male / var_female
    log_vr = math.log(vr)
    se_log_vr = math.sqrt(2.0 / (n_male - 1) + 2.0 / (n_female - 1))
    ci_low_log_vr = log_vr - 1.96 * se_log_vr
    ci_high_log_vr = log_vr + 1.96 * se_log_vr
    ci_low_vr = math.exp(ci_low_log_vr)
    ci_high_vr = math.exp(ci_high_log_vr)

    row = {
        "cohort": cohort,
        "subtest": subtest,
        "status": "computed",
        "reason": pd.NA,
        "n_male": n_male,
        "n_female": n_female,
        "mean_male": mean_male,
        "mean_female": mean_female,
        "var_male": var_male,
        "var_female": var_female,
        "d_subtest": d_subtest,
        "SE_d_subtest": se_d,
        "ci_low_d_subtest": ci_low_d,
        "ci_high_d_subtest": ci_high_d,
        "vr_subtest": vr,
        "log_vr_subtest": log_vr,
        "SE_log_vr_subtest": se_log_vr,
        "ci_low_vr_subtest": ci_low_vr,
        "ci_high_vr_subtest": ci_high_vr,
        "ci_low_log_vr_subtest": ci_low_log_vr,
        "ci_high_log_vr_subtest": ci_high_log_vr,
        "source_data": source_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _plot_profiles(rows: pd.DataFrame, figures_dir: Path, cohort: str) -> None:
    computed = rows[(rows["cohort"] == cohort) & (rows["status"] == "computed")].copy()
    if computed.empty:
        return

    d_plot = computed[["subtest", "d_subtest", "ci_low_d_subtest", "ci_high_d_subtest"]].rename(
        columns={
            "subtest": "label",
            "d_subtest": "estimate",
            "ci_low_d_subtest": "ci_lower",
            "ci_high_d_subtest": "ci_upper",
        }
    )
    save_forest_plot(
        d_plot,
        figures_dir / f"{cohort}_subtest_d_profile.png",
        title=f"{cohort.upper()} subtest d profile",
        xlabel="d (male - female)",
    )

    log_vr_plot = computed[["subtest", "log_vr_subtest", "ci_low_log_vr_subtest", "ci_high_log_vr_subtest"]].rename(
        columns={
            "subtest": "label",
            "log_vr_subtest": "estimate",
            "ci_low_log_vr_subtest": "ci_lower",
            "ci_high_log_vr_subtest": "ci_upper",
        }
    )
    save_forest_plot(
        log_vr_plot,
        figures_dir / f"{cohort}_subtest_log_vr_profile.png",
        title=f"{cohort.upper()} subtest log(VR) profile",
        xlabel="log(VR male/female)",
    )


def run_subtest_profiles(
    *,
    root: Path,
    cohorts: list[str],
    output_path: Path = Path("outputs/tables/subtest_sex_profiles.csv"),
    make_plots: bool = True,
) -> pd.DataFrame:
    paths_cfg = load_yaml(root / "config" / "paths.yml")
    models_cfg = load_yaml(root / "config" / "models.yml")

    processed_dir = _resolve_path(paths_cfg.get("processed_dir", "data/processed"), root)
    outputs_dir = _resolve_path(paths_cfg.get("outputs_dir", "outputs"), root)
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        config_rel = COHORT_CONFIGS.get(cohort)
        if config_rel is None:
            rows.append(_empty_row(cohort, pd.NA, "unknown_cohort", ""))
            continue

        cohort_cfg = load_yaml(root / config_rel)
        sample_cfg = cohort_cfg.get("sample_construct", {}) if isinstance(cohort_cfg.get("sample_construct", {}), dict) else {}
        sex_col = str(sample_cfg.get("sex_col", "sex"))
        subtests = [_as_label(x) for x in sample_cfg.get("subtests", _default_subtests(cohort, models_cfg))]

        data_path = processed_dir / f"{cohort}_cfa_resid.csv"
        if not data_path.exists():
            data_path = processed_dir / f"{cohort}_cfa.csv"
        if not data_path.exists():
            rows.append(_empty_row(cohort, pd.NA, "missing_source_data", ""))
            continue

        source_data = str(data_path.relative_to(root))
        df = pd.read_csv(data_path, low_memory=False)
        if sex_col not in df.columns:
            rows.append(_empty_row(cohort, pd.NA, "missing_sex_column", source_data))
            continue

        for subtest in subtests:
            if subtest not in df.columns:
                rows.append(_empty_row(cohort, subtest, "missing_subtest_column", source_data))
                continue
            rows.append(
                _compute_subtest_stats(
                    cohort=cohort,
                    subtest=subtest,
                    values=df[subtest],
                    sex_series=df[sex_col],
                    source_data=source_data,
                )
            )

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

    if make_plots:
        for cohort in cohorts:
            _plot_profiles(out, figures_dir, cohort)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build subtest-level sex difference and variance-ratio profiles.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Cohort(s) to process.")
    parser.add_argument("--all", action="store_true", help="Process all cohorts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables/subtest_sex_profiles.csv"),
        help="Output CSV path (relative to project-root if not absolute).",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Do not render profile plots.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    cohorts = _cohorts_from_args(args)
    try:
        out = run_subtest_profiles(
            root=root,
            cohorts=cohorts,
            output_path=args.output_path,
            make_plots=not bool(args.skip_plots),
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
