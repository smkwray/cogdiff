#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact markdown snapshot summarizing baseline vs "
            "family-bootstrap stage-20 outputs and publication lock artifacts."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_repo_root(),
        help="Project root containing outputs/ (default: repo root).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output markdown path (default: <project-root>/outputs/tables/results_snapshot.md)."
        ),
    )
    parser.add_argument(
        "--include-run-summary",
        action="store_true",
        help="If available, include the latest pipeline run summary CSV path.",
    )
    parser.add_argument(
        "--expected-bootstrap-reps",
        type=int,
        default=None,
        help=(
            "Expected bootstrap replicate count used for warnings. "
            "Default is dynamic: 499 if any cohort has >=400 reps, else 100."
        ),
    )
    return parser.parse_args()


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"missing header in {path}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            cohort = (row.get("cohort") or "").strip()
            if not cohort:
                continue
            rows[cohort] = {k: (v or "").strip() for k, v in row.items()}
    return rows


def _read_table_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"missing header in {path}")
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def _fmt_ci(low_raw: str | None, high_raw: str | None) -> str:
    low = _to_float(low_raw)
    high = _to_float(high_raw)
    if low is None or high is None:
        return "-"
    return f"[{_fmt_num(low)}, {_fmt_num(high)}]"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows) if rows else ""
    return "\n".join([header_row, sep_row, body]).rstrip() + "\n"


def _rel(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _format_mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _find_latest_run_summary(project_root: Path) -> Path | None:
    base = project_root / "outputs/logs/pipeline"
    if not base.exists():
        return None
    candidates = sorted(base.glob("*_pipeline_run_summary.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))


def _render_metric_section(
    *,
    project_root: Path,
    label: str,
    baseline_path: Path,
    bootstrap_path: Path,
    estimate_col: str,
    se_col: str,
    ci_low_col: str,
    ci_high_col: str,
) -> str:
    baseline_rows = _read_rows(baseline_path)
    bootstrap_rows = _read_rows(bootstrap_path) if bootstrap_path.exists() else {}

    cohorts = sorted(set(baseline_rows) | set(bootstrap_rows))
    md = [f"### {label}", ""]
    md.append(f"- baseline: `{_rel(project_root, baseline_path)}`")
    if bootstrap_path.exists():
        md.append(f"- bootstrap: `{_rel(project_root, bootstrap_path)}`")
    else:
        md.append(f"- bootstrap: missing (`{_rel(project_root, bootstrap_path)}`)")
    md.append("")

    headers = [
        "cohort",
        "baseline",
        "bootstrap",
        "delta",
        "baseline_se",
        "bootstrap_se",
        "baseline_ci",
        "bootstrap_ci",
        "status",
    ]
    rows: list[list[str]] = []
    for cohort in cohorts:
        base = baseline_rows.get(cohort, {})
        boot = bootstrap_rows.get(cohort, {})
        base_est = _to_float(base.get(estimate_col))
        boot_est = _to_float(boot.get(estimate_col))
        delta = None if base_est is None or boot_est is None else boot_est - base_est
        status = (boot.get("status") or "").strip()
        reason = (boot.get("reason") or "").strip()
        status_cell = "-"
        if status or reason:
            status_cell = f"{status}:{reason}" if status and reason else (status or reason)

        rows.append(
            [
                cohort,
                _fmt_num(base_est),
                _fmt_num(boot_est),
                _fmt_num(delta),
                _fmt_num(_to_float(base.get(se_col))),
                _fmt_num(_to_float(boot.get(se_col))),
                _fmt_ci(base.get(ci_low_col), base.get(ci_high_col)),
                _fmt_ci(boot.get(ci_low_col), boot.get(ci_high_col)),
                status_cell,
            ]
        )
    md.append(_md_table(headers, rows))
    return "\n".join(md).rstrip() + "\n"


def _render_publication_lock_section(project_root: Path) -> str:
    lock_dir = project_root / "outputs/tables/publication_results_lock"
    lock_zip = project_root / "outputs/tables/publication_results_lock.zip"
    lock_md = lock_dir / "manuscript_results_lock.md"
    lock_manifest = lock_dir / "publication_results_lock_manifest.csv"
    snapshot_manifest = project_root / "outputs/tables/publication_snapshot_manifest.csv"

    md = ["## Publication lock artifacts", ""]
    if not lock_dir.exists() and not lock_zip.exists():
        md.append("- status: not present")
        md.append("")
        return "\n".join(md)

    def _describe(path: Path) -> str:
        if not path.exists():
            return f"- `{_rel(project_root, path)}`: missing"
        size = path.stat().st_size
        return (
            f"- `{_rel(project_root, path)}`: present "
            f"(bytes={size}, mtime_utc={_format_mtime_utc(path)})"
        )

    md.append(_describe(lock_dir))
    md.append(_describe(lock_zip))
    md.append(_describe(lock_md))
    md.append(_describe(lock_manifest))
    md.append(_describe(snapshot_manifest))
    md.append("")
    return "\n".join(md)


def _render_bootstrap_coverage_section(project_root: Path) -> str:
    root = project_root / "outputs/model_fits/bootstrap_inference"
    md = ["## Bootstrap inference coverage", ""]
    rows_data = _collect_bootstrap_coverage_rows(project_root)
    if rows_data is None:
        md.append("- status: not found (`outputs/model_fits/bootstrap_inference/` missing)")
        md.append("")
        return "\n".join(md)

    if not rows_data:
        md.append("- status: empty")
        md.append("")
        return "\n".join(md)

    headers = [
        "cohort",
        "rep_dirs",
        "max_rep_index",
        "has_full_sample",
        "in_bootstrap_tables",
        "manifest_status",
        "manifest_attempted",
    ]
    rows: list[list[str]] = []
    for row_data in rows_data:
        rows.append(
            [
                row_data["cohort"],
                str(int(row_data["rep_dirs"])),
                (
                    str(int(row_data["max_rep_index"]))
                    if row_data["max_rep_index"] is not None
                    else "-"
                ),
                "yes" if bool(row_data["has_full_sample"]) else "no",
                "yes" if bool(row_data["in_bootstrap_tables"]) else "no",
                str(row_data["manifest_status"]) if row_data["manifest_status"] else "-",
                (
                    str(int(row_data["manifest_attempted"]))
                    if row_data["manifest_attempted"] is not None
                    else "-"
                ),
            ]
        )
    md.append(_md_table(headers, rows))
    return "\n".join(md).rstrip() + "\n"


def _render_confirmatory_exclusions_section(project_root: Path) -> str:
    exclusions_path = project_root / "outputs/tables/confirmatory_exclusions.csv"
    if not exclusions_path.exists():
        return ""

    rows_by_cohort = _read_rows(exclusions_path)
    if not rows_by_cohort:
        return ""

    headers = ["cohort", "blocked_confirmatory_d_g", "reason_d_g"]
    rows: list[list[str]] = []
    for cohort in sorted(rows_by_cohort):
        row = rows_by_cohort[cohort]
        rows.append(
            [
                cohort,
                (row.get("blocked_confirmatory_d_g") or "").strip() or "-",
                (row.get("reason_d_g") or "").strip() or "-",
            ]
        )

    md = ["## Gating exclusions", ""]
    md.append(f"- source: `{_rel(project_root, exclusions_path)}`")
    md.append("")
    md.append(_md_table(headers, rows))
    return "\n".join(md).rstrip() + "\n"


def _render_g_proxy_section(project_root: Path) -> str:
    mean_path = project_root / "outputs/tables/g_proxy_mean_diff_family_bootstrap.csv"
    vr_path = project_root / "outputs/tables/g_proxy_variance_ratio_family_bootstrap.csv"
    if not mean_path.exists() and not vr_path.exists():
        return ""

    md = ["## Observed g proxy (bootstrap inference)", ""]
    if mean_path.exists():
        mean_rows = _read_rows(mean_path)
        headers = ["cohort", "d_g_proxy", "SE", "CI"]
        rows: list[list[str]] = []
        for cohort in sorted(mean_rows):
            row = mean_rows[cohort]
            d = _fmt_num(_to_float(row.get("d_g")))
            se = _fmt_num(_to_float(row.get("SE_d_g")))
            ci = _fmt_ci(row.get("ci_low_d_g"), row.get("ci_high_d_g"))
            rows.append([cohort, d, se, ci])
        md.append(f"- mean diff table: `{_rel(project_root, mean_path)}`")
        md.append(_md_table(headers, rows))
        md.append("")

    if vr_path.exists():
        vr_rows = _read_rows(vr_path)
        headers = ["cohort", "VR_g_proxy", "SE_logVR", "CI"]
        rows = []
        for cohort in sorted(vr_rows):
            row = vr_rows[cohort]
            vr = _fmt_num(_to_float(row.get("VR_g")))
            se = _fmt_num(_to_float(row.get("SE_logVR")))
            ci = _fmt_ci(row.get("ci_low"), row.get("ci_high"))
            rows.append([cohort, vr, se, ci])
        md.append(f"- variance ratio table: `{_rel(project_root, vr_path)}`")
        md.append(_md_table(headers, rows))
        md.append("")

    return "\n".join(md).rstrip() + "\n"


def _render_exploratory_extensions_section(project_root: Path) -> str:
    """Render optional exploratory tables (race/SES/outcomes) when computed rows exist."""
    blocks: list[str] = []

    def _estimate_from_row(row: dict[str, str]) -> float | None:
        for key in ("d_g_proxy", "d_g"):
            val = _to_float(row.get(key))
            if val is not None:
                return val
        return None

    def _ci_from_row(row: dict[str, str]) -> str:
        return _fmt_ci(row.get("ci_low_d_g"), row.get("ci_high_d_g"))

    # Race disaggregation
    race_summary_path = project_root / "outputs/tables/race_sex_interaction_summary.csv"
    if race_summary_path.exists():
        try:
            rows = _read_table_rows(race_summary_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "heterogeneity_p", "min_d_g_proxy", "max_d_g_proxy"]
            table_rows: list[list[str]] = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        _fmt_num(_to_float(r.get("heterogeneity_p_value"))),
                        _fmt_num(_to_float(r.get("min_d_race_group"))),
                        _fmt_num(_to_float(r.get("max_d_race_group"))),
                    ]
                )
            blocks.extend(
                [
                    "### Race × sex interaction (heterogeneity across race groups)",
                    "",
                    f"- summary: `{_rel(project_root, race_summary_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    race_detail_path = project_root / "outputs/tables/race_sex_group_estimates.csv"
    if race_detail_path.exists():
        try:
            rows = _read_table_rows(race_detail_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "race_group", "n_total", "mean_all", "var_all", "d_g_proxy", "SE", "CI"]
            table_rows: list[list[str]] = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("race_group") or "-") or "-",
                        (r.get("n_total") or "-") or "-",
                        _fmt_num(_to_float(r.get("mean_all"))),
                        _fmt_num(_to_float(r.get("var_all"))),
                        _fmt_num(_estimate_from_row(r)),
                        _fmt_num(_to_float(r.get("SE_d_g"))),
                        _ci_from_row(r),
                    ]
                )
            blocks.extend(
                [
                    "### Race/ethnicity-disaggregated `g_proxy` differences",
                    "",
                    f"- detail: `{_rel(project_root, race_detail_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    # SES moderation
    ses_detail_path = project_root / "outputs/tables/ses_moderation_group_estimates.csv"
    if ses_detail_path.exists():
        try:
            rows = _read_table_rows(ses_detail_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "ses_bin", "d_g_proxy", "SE", "CI"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("ses_bin") or "-") or "-",
                        _fmt_num(_estimate_from_row(r)),
                        _fmt_num(_to_float(r.get("SE_d_g"))),
                        _ci_from_row(r),
                    ]
                )
            blocks.extend(
                [
                    "### SES moderation (parent education bins)",
                    "",
                    f"- detail: `{_rel(project_root, ses_detail_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    # g associations with outcomes
    income_path = project_root / "outputs/tables/g_income_wealth_associations.csv"
    if income_path.exists():
        try:
            rows = _read_table_rows(income_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "outcome", "n_used", "corr_all", "beta_g", "p_beta_g", "r2"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("outcome") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("corr_all"))),
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### `g_proxy` associations with income/wealth",
                    "",
                    f"- source: `{_rel(project_root, income_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    employment_path = project_root / "outputs/tables/g_employment_outcomes.csv"
    if employment_path.exists():
        try:
            rows = _read_table_rows(employment_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "outcome_col", "age_col", "n_used", "n_employed", "prevalence", "odds_ratio_g", "p_beta_g", "pseudo_r2"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("outcome_col") or "-") or "-",
                        (r.get("age_col") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        (r.get("n_employed") or "-") or "-",
                        _fmt_num(_to_float(r.get("prevalence"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### `g_proxy` associations with employment status",
                    "",
                    f"- source: `{_rel(project_root, employment_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    life_path = project_root / "outputs/tables/asvab_life_outcomes_by_sex.csv"
    if life_path.exists():
        try:
            rows = _read_table_rows(life_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "outcome", "n_used", "beta_g_male", "beta_g_female", "p_delta_beta"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("outcome") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("beta_g_male"))),
                        _fmt_num(_to_float(r.get("beta_g_female"))),
                        _fmt_num(_to_float(r.get("p_value_delta_beta"))),
                    ]
                )
            blocks.extend(
                [
                    "### `g_proxy` associations by sex (selected outcomes)",
                    "",
                    f"- source: `{_rel(project_root, life_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    cnlsy_employment_path = project_root / "outputs/tables/cnlsy_employment_2014.csv"
    if cnlsy_employment_path.exists():
        try:
            rows = _read_table_rows(cnlsy_employment_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "n_used", "n_employed", "prevalence", "age_min_used", "age_max_used", "odds_ratio_g", "p_value_beta_g", "pseudo_r2"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        (r.get("n_employed") or "-") or "-",
                        _fmt_num(_to_float(r.get("prevalence"))),
                        _fmt_num(_to_float(r.get("age_min_used"))),
                        _fmt_num(_to_float(r.get("age_max_used"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### CNLSY 2014 employment-status association",
                    "",
                    f"- source: `{_rel(project_root, cnlsy_employment_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    sat_act_path = project_root / "outputs/tables/g_sat_act_validity.csv"
    if sat_act_path.exists():
        try:
            rows = _read_table_rows(sat_act_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "outcome", "n_used", "pearson_all", "spearman_all", "beta_g", "p_beta_g"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("outcome") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("pearson_all"))),
                        _fmt_num(_to_float(r.get("spearman_all"))),
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY97 SAT/ACT validity (bins)",
                    "",
                    f"- source: `{_rel(project_root, sat_act_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    sat_act_race_path = project_root / "outputs/tables/g_sat_act_validity_by_race.csv"
    if sat_act_race_path.exists():
        try:
            rows = _read_table_rows(sat_act_race_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "group", "n_used", "pearson", "beta_g", "r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("outcome") or ""), (x.get("group_value") or ""))):
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("group_value") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("pearson"))),
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY97 SAT/ACT validity by race/ethnicity",
                    "",
                    f"- source: `{_rel(project_root, sat_act_race_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    sat_act_ses_path = project_root / "outputs/tables/g_sat_act_validity_by_ses.csv"
    if sat_act_ses_path.exists():
        try:
            rows = _read_table_rows(sat_act_ses_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "ses_bin", "n_used", "pearson", "beta_g", "r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("outcome") or ""), (x.get("group_value") or ""))):
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("group_value") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("pearson"))),
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY97 SAT/ACT validity by SES bins",
                    "",
                    f"- source: `{_rel(project_root, sat_act_ses_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    outcome_ses_summary_path = project_root / "outputs/tables/g_outcome_associations_by_ses_summary.csv"
    if outcome_ses_summary_path.exists():
        try:
            rows = _read_table_rows(outcome_ses_summary_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "outcome", "n_bins_used", "heterogeneity_p", "min_beta_g", "max_beta_g"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("cohort") or ""), (x.get("outcome") or ""))):
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("outcome") or "-") or "-",
                        (r.get("n_bins_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("heterogeneity_p_value"))),
                        _fmt_num(_to_float(r.get("min_beta_g"))),
                        _fmt_num(_to_float(r.get("max_beta_g"))),
                    ]
                )
            blocks.extend(
                [
                    "### `g_proxy` outcome validity by SES bins",
                    "",
                    f"- source: `{_rel(project_root, outcome_ses_summary_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    subtest_validity_path = project_root / "outputs/tables/subtest_predictive_validity.csv"
    if subtest_validity_path.exists():
        try:
            rows = _read_table_rows(subtest_validity_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            best_by_key: dict[tuple[str, str], dict[str, str]] = {}
            for r in computed:
                key = ((r.get("cohort") or ""), (r.get("outcome") or ""))
                current = best_by_key.get(key)
                current_r2 = abs(_to_float(current.get("r2")) or 0.0) if current else -1.0
                new_r2 = abs(_to_float(r.get("r2")) or 0.0)
                if current is None or new_r2 > current_r2:
                    best_by_key[key] = r
            headers = ["cohort", "outcome", "predictor_type", "predictor", "beta", "r2"]
            table_rows = []
            for _, r in sorted(best_by_key.items(), key=lambda item: item[0]):
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("outcome") or "-") or "-",
                        (r.get("predictor_type") or "-") or "-",
                        (r.get("predictor") or "-") or "-",
                        _fmt_num(_to_float(r.get("beta_predictor"))),
                        _fmt_num(_to_float(r.get("r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### Strongest subtest/factor predictors by outcome",
                    "",
                    f"- source: `{_rel(project_root, subtest_validity_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    cnlsy_nonlinear_path = project_root / "outputs/tables/cnlsy_nonlinear_age_patterns.csv"
    if cnlsy_nonlinear_path.exists():
        try:
            rows = _read_table_rows(cnlsy_nonlinear_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["metric", "n_bins", "linear_r2", "quadratic_r2", "delta_r2", "turning_point_age"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("metric") or "-") or "-",
                        (r.get("n_bins") or "-") or "-",
                        _fmt_num(_to_float(r.get("linear_r2"))),
                        _fmt_num(_to_float(r.get("quadratic_r2"))),
                        _fmt_num(_to_float(r.get("quadratic_delta_r2"))),
                        _fmt_num(_to_float(r.get("turning_point_age"))),
                    ]
                )
            blocks.extend(
                [
                    "### CNLSY nonlinear age-pattern checks",
                    "",
                    f"- source: `{_rel(project_root, cnlsy_nonlinear_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    contrasts_path = project_root / "outputs/tables/cross_cohort_predictive_validity_contrasts.csv"
    if contrasts_path.exists():
        try:
            rows = _read_table_rows(contrasts_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "cohort_a", "cohort_b", "diff_b_minus_a", "p_value"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("cohort_a") or "-") or "-",
                        (r.get("cohort_b") or "-") or "-",
                        _fmt_num(_to_float(r.get("diff_b_minus_a"))),
                        _fmt_num(_to_float(r.get("p_value_diff"))),
                    ]
                )
            blocks.extend(
                [
                    "### Cross-cohort predictive-validity contrasts",
                    "",
                    f"- source: `{_rel(project_root, contrasts_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    stability_path = project_root / "outputs/tables/cross_cohort_pattern_stability.csv"
    if stability_path.exists():
        try:
            rows = _read_table_rows(stability_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["estimand", "n_cohorts", "mean_estimate", "sd_estimate", "range_estimate", "cv_estimate"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("estimand") or "-") or "-",
                        (r.get("n_cohorts") or "-") or "-",
                        _fmt_num(_to_float(r.get("mean_estimate"))),
                        _fmt_num(_to_float(r.get("sd_estimate"))),
                        _fmt_num(_to_float(r.get("range_estimate"))),
                        _fmt_num(_to_float(r.get("cv_estimate"))),
                    ]
                )
            blocks.extend(
                [
                    "### Cross-cohort pattern stability",
                    "",
                    f"- source: `{_rel(project_root, stability_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    sibling_fe_path = project_root / "outputs/tables/sibling_fe_g_outcome.csv"
    if sibling_fe_path.exists():
        try:
            rows = _read_table_rows(sibling_fe_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "outcome", "n_families", "n_individuals", "beta_between", "beta_within", "p_within", "r2_within"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("cohort") or ""), (x.get("outcome") or ""))):
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("outcome") or "-") or "-",
                        (r.get("n_families") or "-") or "-",
                        (r.get("n_individuals") or "-") or "-",
                        _fmt_num(_to_float(r.get("beta_g_total") if r.get("beta_g_total") not in {None, ""} else r.get("beta_g_between"))),
                        _fmt_num(_to_float(r.get("beta_g_within"))),
                        _fmt_num(_to_float(r.get("p_within"))),
                        _fmt_num(_to_float(r.get("r2_within"))),
                    ]
                )
            blocks.extend(
                [
                    "### Sibling fixed-effects outcome associations",
                    "",
                    f"- source: `{_rel(project_root, sibling_fe_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    sibling_contrast_path = project_root / "outputs/tables/sibling_fe_cross_cohort_contrasts.csv"
    if sibling_contrast_path.exists():
        try:
            rows = _read_table_rows(sibling_contrast_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "cohort_a", "cohort_b", "diff_b_minus_a", "p_diff"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("cohort_a") or "-") or "-",
                        (r.get("cohort_b") or "-") or "-",
                        _fmt_num(_to_float(r.get("diff_b_minus_a"))),
                        _fmt_num(_to_float(r.get("p_value_diff"))),
                    ]
                )
            blocks.extend(
                [
                    "### Within-family cross-cohort contrasts",
                    "",
                    f"- source: `{_rel(project_root, sibling_contrast_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    intergen_path = project_root / "outputs/tables/intergenerational_g_transmission.csv"
    if intergen_path.exists():
        try:
            rows = _read_table_rows(intergen_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["model", "n_pairs", "beta_mother_g", "beta_parent_ed", "p", "r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: (x.get("model") or "")):
                table_rows.append(
                    [
                        (r.get("model") or "-") or "-",
                        (r.get("n_pairs") or "-") or "-",
                        _fmt_num(_to_float(r.get("beta_mother_g"))),
                        _fmt_num(_to_float(r.get("beta_parent_ed"))),
                        _fmt_num(_to_float(r.get("p"))),
                        _fmt_num(_to_float(r.get("r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### Intergenerational mother-child `g_proxy` transmission",
                    "",
                    f"- source: `{_rel(project_root, intergen_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    intergen_attn_path = project_root / "outputs/tables/intergenerational_g_attenuation.csv"
    if intergen_attn_path.exists():
        try:
            rows = _read_table_rows(intergen_attn_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["n_pairs_bivariate", "n_pairs_ses_controlled", "attenuation_abs", "attenuation_pct", "delta_r2", "beta_parent_ed"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("n_pairs_bivariate") or "-") or "-",
                        (r.get("n_pairs_ses_controlled") or "-") or "-",
                        _fmt_num(_to_float(r.get("attenuation_abs"))),
                        _fmt_num(_to_float(r.get("attenuation_pct"))),
                        _fmt_num(_to_float(r.get("delta_r2"))),
                        _fmt_num(_to_float(r.get("beta_parent_ed"))),
                    ]
                )
            blocks.extend(
                [
                    "### SES attenuation of mother-child `g_proxy` transmission",
                    "",
                    f"- source: `{_rel(project_root, intergen_attn_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    tilt_path = project_root / "outputs/tables/subtest_profile_tilt.csv"
    if tilt_path.exists():
        try:
            rows = _read_table_rows(tilt_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "n_used_education", "d_tilt", "tilt_g_corr", "incremental_r2", "p_tilt"]
            table_rows = []
            for r in sorted(computed, key=lambda x: (x.get("cohort") or "")):
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("n_used_education") or "-") or "-",
                        _fmt_num(_to_float(r.get("d_tilt"))),
                        _fmt_num(_to_float(r.get("tilt_g_corr"))),
                        _fmt_num(_to_float(r.get("tilt_incremental_r2_education"))),
                        _fmt_num(_to_float(r.get("p_tilt_incremental"))),
                    ]
                )
            blocks.extend(
                [
                    "### Verbal-quantitative subtest profile tilt",
                    "",
                    f"- source: `{_rel(project_root, tilt_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    tilt_summary_path = project_root / "outputs/tables/subtest_profile_tilt_summary.csv"
    if tilt_summary_path.exists():
        try:
            rows = _read_table_rows(tilt_summary_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "d_g_proxy", "d_tilt", "tilt_to_g_ratio_abs", "incremental_r2_band", "interpretation"]
            table_rows = []
            for r in sorted(computed, key=lambda x: (x.get("cohort") or "")):
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        _fmt_num(_to_float(r.get("d_g_proxy"))),
                        _fmt_num(_to_float(r.get("d_tilt"))),
                        _fmt_num(_to_float(r.get("tilt_to_g_ratio_abs"))),
                        (r.get("incremental_r2_band") or "-") or "-",
                        (r.get("interpretation") or "-") or "-",
                    ]
                )
            blocks.extend(
                [
                    "### Tilt interpretation relative to `g_proxy`",
                    "",
                    f"- source: `{_rel(project_root, tilt_summary_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    degree_path = project_root / "outputs/tables/degree_threshold_outcomes.csv"
    if degree_path.exists():
        try:
            rows = _read_table_rows(degree_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "threshold", "n_used", "n_positive", "prevalence", "odds_ratio_g", "p_value_beta_g", "pseudo_r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("cohort") or ""), (x.get("threshold") or ""))):
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("threshold") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        (r.get("n_positive") or "-") or "-",
                        _fmt_num(_to_float(r.get("prevalence"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### Degree-threshold proxy outcomes",
                    "",
                    f"- source: `{_rel(project_root, degree_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    explicit_degree_path = project_root / "outputs/tables/explicit_degree_outcomes.csv"
    if explicit_degree_path.exists():
        try:
            rows = _read_table_rows(explicit_degree_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "threshold", "degree_col", "age_col", "n_used", "n_positive", "prevalence", "odds_ratio_g", "p_value_beta_g", "pseudo_r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("cohort") or ""), (x.get("threshold") or ""))):
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("threshold") or "-") or "-",
                        (r.get("degree_col") or "-") or "-",
                        (r.get("age_col") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        (r.get("n_positive") or "-") or "-",
                        _fmt_num(_to_float(r.get("prevalence"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### Explicit coded degree outcomes",
                    "",
                    f"- source: `{_rel(project_root, explicit_degree_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    age_matched_validity_path = project_root / "outputs/tables/age_matched_outcome_validity.csv"
    if age_matched_validity_path.exists():
        try:
            rows = _read_table_rows(age_matched_validity_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "cohort", "age_col", "n_used", "model_type", "beta_g", "p_value_beta_g", "r2_or_pseudo_r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("outcome") or ""), (x.get("cohort") or ""), (x.get("age_col") or ""))):
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("cohort") or "-") or "-",
                        (r.get("age_col") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        (r.get("model_type") or "-") or "-",
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("r2_or_pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### Age-matched cross-cohort outcome validity",
                    "",
                    f"- source: `{_rel(project_root, age_matched_validity_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    age_matched_contrast_path = project_root / "outputs/tables/age_matched_cross_cohort_contrasts.csv"
    if age_matched_contrast_path.exists():
        try:
            rows = _read_table_rows(age_matched_contrast_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "age_col_b", "overlap_min", "overlap_max", "beta_a", "beta_b", "diff_b_minus_a", "p_value_diff"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("outcome") or ""), (x.get("age_col_b") or ""))):
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("age_col_b") or "-") or "-",
                        _fmt_num(_to_float(r.get("overlap_min"))),
                        _fmt_num(_to_float(r.get("overlap_max"))),
                        _fmt_num(_to_float(r.get("beta_a"))),
                        _fmt_num(_to_float(r.get("beta_b"))),
                        _fmt_num(_to_float(r.get("diff_b_minus_a"))),
                        _fmt_num(_to_float(r.get("p_value_diff"))),
                    ]
                )
            blocks.extend(
                [
                    "### Age-matched cross-cohort contrasts",
                    "",
                    f"- source: `{_rel(project_root, age_matched_contrast_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    trajectories_path = project_root / "outputs/tables/nlsy97_income_earnings_trajectories.csv"
    if trajectories_path.exists():
        try:
            rows = _read_table_rows(trajectories_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "model", "n_used", "mean_annualized_log_change", "beta_g", "p_value_beta_g", "r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("outcome") or ""), (x.get("model") or ""))):
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("model") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("mean_annualized_log_change"))),
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY97 two-wave income and earnings trajectories",
                    "",
                    f"- source: `{_rel(project_root, trajectories_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    volatility_path = project_root / "outputs/tables/nlsy97_income_earnings_volatility.csv"
    if volatility_path.exists():
        try:
            rows = _read_table_rows(volatility_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "model", "n_used", "mean_abs_annualized_log_change", "instability_cutoff", "prevalence", "beta_g", "odds_ratio_g", "p_value_beta_g", "r2_or_pseudo_r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("outcome") or ""), (x.get("model") or ""))):
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("model") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("mean_abs_annualized_log_change"))),
                        _fmt_num(_to_float(r.get("instability_cutoff"))),
                        _fmt_num(_to_float(r.get("prevalence"))),
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("r2_or_pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY97 two-wave income and earnings volatility",
                    "",
                    f"- source: `{_rel(project_root, volatility_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    employment_persistence_path = project_root / "outputs/tables/nlsy97_employment_persistence.csv"
    if employment_persistence_path.exists():
        try:
            rows = _read_table_rows(employment_persistence_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["model", "n_used", "n_positive", "prevalence", "odds_ratio_g", "p_value_beta_g", "pseudo_r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: (x.get("model") or "")):
                table_rows.append(
                    [
                        (r.get("model") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        (r.get("n_positive") or "-") or "-",
                        _fmt_num(_to_float(r.get("prevalence"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY97 labor-force persistence and employment transitions",
                    "",
                    f"- source: `{_rel(project_root, employment_persistence_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    employment_instability_path = project_root / "outputs/tables/nlsy97_employment_instability.csv"
    if employment_instability_path.exists():
        try:
            rows = _read_table_rows(employment_instability_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["model", "n_used", "n_positive", "prevalence", "odds_ratio_g", "p_value_beta_g", "pseudo_r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: (x.get("model") or "")):
                table_rows.append(
                    [
                        (r.get("model") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        (r.get("n_positive") or "-") or "-",
                        _fmt_num(_to_float(r.get("prevalence"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY97 multi-wave employment instability",
                    "",
                    f"- source: `{_rel(project_root, employment_instability_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    ui_path = project_root / "outputs/tables/nlsy97_unemployment_insurance.csv"
    if ui_path.exists():
        try:
            rows = _read_table_rows(ui_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["year", "model", "n_used", "n_positive", "prevalence", "beta_g", "odds_ratio_g", "p_value_beta_g", "r2_or_pseudo_r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("year") or ""), (x.get("model") or ""))):
                table_rows.append(
                    [
                        (r.get("year") or "-") or "-",
                        (r.get("model") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        (r.get("n_positive") or "-") or "-",
                        _fmt_num(_to_float(r.get("prevalence"))),
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("r2_or_pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY97 unemployment-insurance receipt and intensity",
                    "",
                    f"- source: `{_rel(project_root, ui_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    cnlsy_adult_path = project_root / "outputs/tables/cnlsy_adult_outcome_associations.csv"
    if cnlsy_adult_path.exists():
        try:
            rows = _read_table_rows(cnlsy_adult_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "model_type", "n_used", "mean_outcome", "beta_g", "p_value_beta_g", "r2_or_pseudo_r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("model_type") or ""), (x.get("outcome") or ""))):
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("model_type") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("mean_outcome"))),
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("r2_or_pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### CNLSY 2014 late-adolescent/adult outcome associations",
                    "",
                    f"- source: `{_rel(project_root, cnlsy_adult_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    cnlsy_carryover_path = project_root / "outputs/tables/cnlsy_carryover_net_mother_ses_summary.csv"
    if cnlsy_carryover_path.exists():
        try:
            rows = _read_table_rows(cnlsy_carryover_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "model_type", "n_baseline", "n_mother_ses", "beta_g_baseline", "beta_g_mother_ses", "attenuation_pct", "delta_r2_or_pseudo_r2"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("model_type") or "-") or "-",
                        (r.get("n_baseline") or "-") or "-",
                        (r.get("n_mother_ses") or "-") or "-",
                        _fmt_num(_to_float(r.get("beta_g_baseline"))),
                        _fmt_num(_to_float(r.get("beta_g_mother_ses"))),
                        _fmt_num(_to_float(r.get("attenuation_pct"))),
                        _fmt_num(_to_float(r.get("delta_r2_or_pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### CNLSY child `g_proxy` carryover net mother SES",
                    "",
                    f"- source: `{_rel(project_root, cnlsy_carryover_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nonlinear_summary_path = project_root / "outputs/tables/nonlinear_threshold_outcome_summary.csv"
    if nonlinear_summary_path.exists():
        try:
            rows = _read_table_rows(nonlinear_summary_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "outcome", "outcome_type", "n_linear", "beta_g_sq", "p_value_g_sq", "delta_fit_linear_to_quadratic", "threshold_odds_ratio", "threshold_beta", "p_value_threshold"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("outcome") or "-") or "-",
                        (r.get("outcome_type") or "-") or "-",
                        (r.get("n_linear") or "-") or "-",
                        _fmt_num(_to_float(r.get("beta_g_sq"))),
                        _fmt_num(_to_float(r.get("p_value_g_sq"))),
                        _fmt_num(_to_float(r.get("delta_fit_linear_to_quadratic"))),
                        _fmt_num(_to_float(r.get("threshold_odds_ratio"))),
                        _fmt_num(_to_float(r.get("threshold_beta"))),
                        _fmt_num(_to_float(r.get("p_value_threshold"))),
                    ]
                )
            blocks.extend(
                [
                    "### Nonlinear and threshold outcome models",
                    "",
                    f"- source: `{_rel(project_root, nonlinear_summary_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    sibling_discordance_path = project_root / "outputs/tables/sibling_discordance.csv"
    if sibling_discordance_path.exists():
        try:
            rows = _read_table_rows(sibling_discordance_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["cohort", "outcome", "n_pairs", "n_families", "mean_abs_g_diff", "mean_abs_outcome_diff", "corr_abs_diff", "beta_abs_g_diff", "p_value_abs_g_diff", "r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: ((x.get("cohort") or ""), (x.get("outcome") or ""))):
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("outcome") or "-") or "-",
                        (r.get("n_pairs") or "-") or "-",
                        (r.get("n_families") or "-") or "-",
                        _fmt_num(_to_float(r.get("mean_abs_g_diff"))),
                        _fmt_num(_to_float(r.get("mean_abs_outcome_diff"))),
                        _fmt_num(_to_float(r.get("corr_abs_diff"))),
                        _fmt_num(_to_float(r.get("beta_abs_g_diff"))),
                        _fmt_num(_to_float(r.get("p_value_abs_g_diff"))),
                        _fmt_num(_to_float(r.get("r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### Sibling discordance beyond education",
                    "",
                    f"- source: `{_rel(project_root, sibling_discordance_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    mediation_summary_path = project_root / "outputs/tables/nlsy79_mediation_summary.csv"
    if mediation_summary_path.exists():
        try:
            rows = _read_table_rows(mediation_summary_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "model", "n_used", "beta_g_baseline", "beta_g_model", "pct_attenuation_g", "delta_r2", "mediators_in_model"]
            table_rows = []
            sort_order = {
                "baseline": 0,
                "plus_education_years": 1,
                "plus_employment_2000": 2,
                "plus_job_zone": 3,
                "plus_all_mediators": 4,
            }
            for r in sorted(computed, key=lambda x: ((x.get("outcome") or ""), sort_order.get((x.get("model") or ""), 99))):
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("model") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("beta_g_baseline"))),
                        _fmt_num(_to_float(r.get("beta_g_model"))),
                        _fmt_num(_to_float(r.get("pct_attenuation_g"))),
                        _fmt_num(_to_float(r.get("delta_r2"))),
                        (r.get("mediators_in_model") or "-") or "-",
                    ]
                )
            blocks.extend(
                [
                    "### NLSY79 mediation of earnings and income associations",
                    "",
                    f"- source: `{_rel(project_root, mediation_summary_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    occupation_summary_path = project_root / "outputs/tables/nlsy79_occupation_major_group_summary.csv"
    if occupation_summary_path.exists():
        try:
            rows = _read_table_rows(occupation_summary_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["occupation_group", "n_used", "share_used", "mean_g_proxy", "mean_education_years", "mean_household_income"]
            table_rows = []
            for r in sorted(computed, key=lambda x: -(_to_float(x.get("share_used")) or 0.0)):
                table_rows.append(
                    [
                        (r.get("occupation_group_label") or r.get("occupation_group") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("share_used"))),
                        _fmt_num(_to_float(r.get("mean_g_proxy"))),
                        _fmt_num(_to_float(r.get("mean_education_years"))),
                        _fmt_num(_to_float(r.get("mean_household_income"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY79 occupation major-group summary",
                    "",
                    f"- source: `{_rel(project_root, occupation_summary_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    occupation_model_path = project_root / "outputs/tables/nlsy79_high_skill_occupation_outcome.csv"
    if occupation_model_path.exists():
        try:
            rows = _read_table_rows(occupation_model_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "n_used", "n_positive", "prevalence", "odds_ratio_g", "p_value_beta_g", "pseudo_r2"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        (r.get("n_positive") or "-") or "-",
                        _fmt_num(_to_float(r.get("prevalence"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY79 management/professional occupation proxy",
                    "",
                    f"- source: `{_rel(project_root, occupation_model_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nlsy97_occupation_summary_path = project_root / "outputs/tables/nlsy97_adult_occupation_major_group_summary.csv"
    if nlsy97_occupation_summary_path.exists():
        try:
            rows = _read_table_rows(nlsy97_occupation_summary_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["occupation_group", "n_used", "share_used", "mean_g_proxy", "mean_education_years", "top_source_wave"]
            table_rows = []
            n_with_any = computed[0].get("n_with_any_occupation") or "-"
            n_total = computed[0].get("n_total") or "-"
            for r in sorted(computed, key=lambda x: -(_to_float(x.get("share_used")) or 0.0)):
                table_rows.append(
                    [
                        (r.get("occupation_group_label") or r.get("occupation_group") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("share_used"))),
                        _fmt_num(_to_float(r.get("mean_g_proxy"))),
                        _fmt_num(_to_float(r.get("mean_education_years"))),
                        (r.get("top_source_wave") or "-") or "-",
                    ]
                )
            blocks.extend(
                [
                    "### NLSY97 latest-adult occupation major-group summary",
                    "",
                    f"- source: `{_rel(project_root, nlsy97_occupation_summary_path)}`",
                    f"- coverage: `{n_with_any}` with any adult occupation code out of `{n_total}` total respondents",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nlsy97_occupation_mobility_summary_path = project_root / "outputs/tables/nlsy97_occupational_mobility_summary.csv"
    if nlsy97_occupation_mobility_summary_path.exists():
        try:
            rows = _read_table_rows(nlsy97_occupation_mobility_summary_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["n_with_2plus_occupation_waves", "n_with_major_group_start_end", "n_changed_major_group", "pct_changed_major_group", "n_upward_to_management_professional", "n_downward_from_management_professional", "mean_year_gap", "top_wave_pair"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("n_with_2plus_occupation_waves") or "-") or "-",
                        (r.get("n_with_major_group_start_end") or "-") or "-",
                        (r.get("n_changed_major_group") or "-") or "-",
                        _fmt_num(_to_float(r.get("pct_changed_major_group"))),
                        (r.get("n_upward_to_management_professional") or "-") or "-",
                        (r.get("n_downward_from_management_professional") or "-") or "-",
                        _fmt_num(_to_float(r.get("mean_year_gap"))),
                        (r.get("top_wave_pair") or "-") or "-",
                    ]
                )
            blocks.extend(
                [
                    "### NLSY97 bounded occupational mobility",
                    "",
                    f"- source: `{_rel(project_root, nlsy97_occupation_mobility_summary_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nlsy97_occupation_mobility_models_path = project_root / "outputs/tables/nlsy97_occupational_mobility_models.csv"
    if nlsy97_occupation_mobility_models_path.exists():
        try:
            rows = _read_table_rows(nlsy97_occupation_mobility_models_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["model", "n_used", "n_positive", "prevalence", "odds_ratio_g", "p_value_beta_g", "pseudo_r2"]
            table_rows = []
            for r in sorted(computed, key=lambda x: (x.get("model") or "")):
                table_rows.append(
                    [
                        (r.get("model") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        (r.get("n_positive") or "-") or "-",
                        _fmt_num(_to_float(r.get("prevalence"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY97 occupational mobility associations",
                    "",
                    f"- source: `{_rel(project_root, nlsy97_occupation_mobility_models_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nlsy97_occupation_model_path = project_root / "outputs/tables/nlsy97_high_skill_occupation_outcome.csv"
    if nlsy97_occupation_model_path.exists():
        try:
            rows = _read_table_rows(nlsy97_occupation_model_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "n_used", "n_positive", "prevalence", "odds_ratio_g", "p_value_beta_g", "pseudo_r2"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        (r.get("n_positive") or "-") or "-",
                        _fmt_num(_to_float(r.get("prevalence"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("pseudo_r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY97 latest-adult management/professional occupation proxy",
                    "",
                    f"- source: `{_rel(project_root, nlsy97_occupation_model_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nlsy79_job_zone_mapping_path = project_root / "outputs/tables/nlsy79_job_zone_mapping_quality.csv"
    if nlsy79_job_zone_mapping_path.exists():
        try:
            rows = _read_table_rows(nlsy79_job_zone_mapping_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["n_occ_nonmissing", "n_matched_exact", "n_matched_prefix_only", "n_matched_any", "pct_matched_any", "mean_job_zone"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("n_occ_nonmissing") or "-") or "-",
                        (r.get("n_matched_exact") or "-") or "-",
                        (r.get("n_matched_prefix_only") or "-") or "-",
                        (r.get("n_matched_any") or "-") or "-",
                        _fmt_num(_to_float(r.get("pct_matched_any"))),
                        _fmt_num(_to_float(r.get("mean_job_zone"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY79 Job Zone mapping quality",
                    "",
                    f"- source: `{_rel(project_root, nlsy79_job_zone_mapping_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nlsy79_job_zone_model_path = project_root / "outputs/tables/nlsy79_job_zone_complexity_outcome.csv"
    if nlsy79_job_zone_model_path.exists():
        try:
            rows = _read_table_rows(nlsy79_job_zone_model_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "n_used", "beta_g", "p_value_beta_g", "beta_age", "p_value_beta_age", "r2"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("beta_age"))),
                        _fmt_num(_to_float(r.get("p_value_beta_age"))),
                        _fmt_num(_to_float(r.get("r2"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY79 Job Zone complexity association",
                    "",
                    f"- source: `{_rel(project_root, nlsy79_job_zone_model_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nlsy79_job_pay_summary_path = project_root / "outputs/tables/nlsy79_job_pay_mismatch_summary.csv"
    if nlsy79_job_pay_summary_path.exists():
        try:
            rows = _read_table_rows(nlsy79_job_pay_summary_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["group", "n_used", "share_used", "mean_job_zone", "mean_annual_earnings", "mean_pay_residual_z", "mean_g_proxy", "mean_education_years"]
            table_rows = []
            sort_order = {"overall": 0, "underpaid_for_complexity": 1, "aligned_band": 2, "overpaid_for_complexity": 3}
            for r in sorted(computed, key=lambda x: sort_order.get((x.get("group") or ""), 99)):
                table_rows.append(
                    [
                        (r.get("group") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("share_used"))),
                        _fmt_num(_to_float(r.get("mean_job_zone"))),
                        _fmt_num(_to_float(r.get("mean_annual_earnings"))),
                        _fmt_num(_to_float(r.get("mean_pay_residual_z"))),
                        _fmt_num(_to_float(r.get("mean_g_proxy"))),
                        _fmt_num(_to_float(r.get("mean_education_years"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY79 job-complexity vs pay mismatch summary",
                    "",
                    f"- source: `{_rel(project_root, nlsy79_job_pay_summary_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nlsy79_job_pay_model_path = project_root / "outputs/tables/nlsy79_job_pay_mismatch_models.csv"
    if nlsy79_job_pay_model_path.exists():
        try:
            rows = _read_table_rows(nlsy79_job_pay_model_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "model_family", "n_used", "beta_g", "p_value_beta_g", "odds_ratio_g", "r2_or_pseudo_r2", "prevalence"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("model_family") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("r2_or_pseudo_r2"))),
                        _fmt_num(_to_float(r.get("prevalence"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY79 job-complexity vs pay mismatch associations",
                    "",
                    f"- source: `{_rel(project_root, nlsy79_job_pay_model_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nlsy79_occ_edu_mapping_path = project_root / "outputs/tables/nlsy79_occupation_education_mapping_quality.csv"
    if nlsy79_occ_edu_mapping_path.exists():
        try:
            rows = _read_table_rows(nlsy79_occ_edu_mapping_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = [
                "n_occ_nonmissing",
                "n_matched_exact",
                "n_matched_prefix_only",
                "n_matched_any",
                "pct_matched_any",
                "mean_required_education_years",
                "mean_bachelor_plus_share",
                "modal_required_education_label",
            ]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("n_occ_nonmissing") or "-") or "-",
                        (r.get("n_matched_exact") or "-") or "-",
                        (r.get("n_matched_prefix_only") or "-") or "-",
                        (r.get("n_matched_any") or "-") or "-",
                        _fmt_num(_to_float(r.get("pct_matched_any"))),
                        _fmt_num(_to_float(r.get("mean_required_education_years"))),
                        _fmt_num(_to_float(r.get("mean_bachelor_plus_share"))),
                        (r.get("modal_required_education_label") or "-") or "-",
                    ]
                )
            blocks.extend(
                [
                    "### NLSY79 occupation education-requirement mapping quality",
                    "",
                    f"- source: `{_rel(project_root, nlsy79_occ_edu_mapping_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nlsy79_occ_edu_model_path = project_root / "outputs/tables/nlsy79_occupation_education_requirement_outcome.csv"
    if nlsy79_occ_edu_model_path.exists():
        try:
            rows = _read_table_rows(nlsy79_occ_edu_model_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "n_used", "beta_g", "p_value_beta_g", "beta_age", "p_value_beta_age", "r2", "mean_outcome"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("beta_age"))),
                        _fmt_num(_to_float(r.get("p_value_beta_age"))),
                        _fmt_num(_to_float(r.get("r2"))),
                        _fmt_num(_to_float(r.get("mean_outcome"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY79 occupation education-requirement associations",
                    "",
                    f"- source: `{_rel(project_root, nlsy79_occ_edu_model_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nlsy79_mismatch_summary_path = project_root / "outputs/tables/nlsy79_education_job_mismatch_summary.csv"
    if nlsy79_mismatch_summary_path.exists():
        try:
            rows = _read_table_rows(nlsy79_mismatch_summary_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["group", "n_used", "share_used", "mean_mismatch_years", "mean_g_proxy", "mean_education_years", "mean_required_education_years", "mean_annual_earnings"]
            table_rows = []
            sort_order = {"overall": 0, "undereducated": 1, "matched_band": 2, "overeducated": 3}
            for r in sorted(computed, key=lambda x: sort_order.get((x.get("group") or ""), 99)):
                table_rows.append(
                    [
                        (r.get("group") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("share_used"))),
                        _fmt_num(_to_float(r.get("mean_mismatch_years"))),
                        _fmt_num(_to_float(r.get("mean_g_proxy"))),
                        _fmt_num(_to_float(r.get("mean_education_years"))),
                        _fmt_num(_to_float(r.get("mean_required_education_years"))),
                        _fmt_num(_to_float(r.get("mean_annual_earnings"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY79 education-job mismatch summary",
                    "",
                    f"- source: `{_rel(project_root, nlsy79_mismatch_summary_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    nlsy79_mismatch_model_path = project_root / "outputs/tables/nlsy79_education_job_mismatch_models.csv"
    if nlsy79_mismatch_model_path.exists():
        try:
            rows = _read_table_rows(nlsy79_mismatch_model_path)
        except Exception:
            rows = []
        computed = [r for r in rows if (r.get("status") or "").strip().lower() == "computed"]
        if computed:
            headers = ["outcome", "model_family", "n_used", "beta_g", "p_value_beta_g", "odds_ratio_g", "beta_age", "p_value_beta_age", "r2_or_pseudo_r2", "prevalence"]
            table_rows = []
            for r in computed:
                table_rows.append(
                    [
                        (r.get("outcome") or "-") or "-",
                        (r.get("model_family") or "-") or "-",
                        (r.get("n_used") or "-") or "-",
                        _fmt_num(_to_float(r.get("beta_g"))),
                        _fmt_num(_to_float(r.get("p_value_beta_g"))),
                        _fmt_num(_to_float(r.get("odds_ratio_g"))),
                        _fmt_num(_to_float(r.get("beta_age"))),
                        _fmt_num(_to_float(r.get("p_value_beta_age"))),
                        _fmt_num(_to_float(r.get("r2_or_pseudo_r2"))),
                        _fmt_num(_to_float(r.get("prevalence"))),
                    ]
                )
            blocks.extend(
                [
                    "### NLSY79 education-job mismatch associations",
                    "",
                    f"- source: `{_rel(project_root, nlsy79_mismatch_model_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    race_invariance_path = project_root / "outputs/tables/race_invariance_eligibility.csv"
    if race_invariance_path.exists():
        try:
            rows = _read_table_rows(race_invariance_path)
        except Exception:
            rows = []
        if rows:
            headers = ["cohort", "status", "n_groups", "smallest_group_n", "metric_pass", "scalar_pass", "reason_d_g"]
            table_rows = []
            for r in sorted(rows, key=lambda x: (x.get("cohort") or "")):
                table_rows.append(
                    [
                        (r.get("cohort") or "-") or "-",
                        (r.get("status") or "-") or "-",
                        (r.get("n_groups") or "-") or "-",
                        (r.get("smallest_group_n") or "-") or "-",
                        (r.get("metric_pass") or "-") or "-",
                        (r.get("scalar_pass") or "-") or "-",
                        (r.get("reason_d_g") or "-") or "-",
                    ]
                )
            blocks.extend(
                [
                    "### Race/ethnicity measurement invariance",
                    "",
                    f"- source: `{_rel(project_root, race_invariance_path)}`",
                    _md_table(headers, table_rows).rstrip(),
                    "",
                ]
            )

    if not blocks:
        return ""

    header = [
        "## Exploratory extensions (observed `g_proxy`)",
        "",
        "These modules use an observed composite `g_proxy` (not latent `d_g`).",
        "",
    ]
    return "\n".join(header + blocks).rstrip() + "\n"


def _collect_bootstrap_coverage_rows(project_root: Path) -> list[dict[str, object]] | None:
    root = project_root / "outputs/model_fits/bootstrap_inference"
    if not root.exists():
        return None

    table_cohorts: set[str] = set()
    for table_name in ("g_mean_diff_family_bootstrap.csv", "g_variance_ratio_family_bootstrap.csv"):
        table_path = project_root / "outputs/tables" / table_name
        if table_path.exists():
            try:
                table_cohorts.update(_read_rows(table_path).keys())
            except Exception:
                continue

    manifest_path = project_root / "outputs/tables/inference_rerun_manifest_family_bootstrap.json"
    manifest_by_cohort: dict[str, dict[str, object]] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for detail in manifest.get("cohort_details", []) if isinstance(manifest, dict) else []:
                if not isinstance(detail, dict):
                    continue
                cohort = str(detail.get("cohort") or "").strip()
                if cohort:
                    manifest_by_cohort[cohort] = detail
        except Exception:
            manifest_by_cohort = {}

    cohort_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    rows: list[dict[str, object]] = []
    for cohort_dir in cohort_dirs:
        rep_dirs = sorted(
            [
                p
                for p in cohort_dir.iterdir()
                if p.is_dir() and p.name.startswith("rep_") and p.name[4:].isdigit()
            ]
        )
        indices = [int(p.name[4:]) for p in rep_dirs]
        max_idx = max(indices) if indices else None
        full_sample = cohort_dir / "full_sample"
        detail = manifest_by_cohort.get(cohort_dir.name, {})
        rows.append(
            {
                "cohort": cohort_dir.name,
                "rep_dirs": len(rep_dirs),
                "max_rep_index": max_idx,
                "has_full_sample": full_sample.exists(),
                "in_bootstrap_tables": cohort_dir.name in table_cohorts,
                "manifest_status": (detail.get("status") if isinstance(detail, dict) else None),
                "manifest_attempted": (detail.get("attempted") if isinstance(detail, dict) else None),
            }
        )
    return rows


def _expected_bootstrap_reps(
    coverage_rows: list[dict[str, object]], explicit_expected: int | None
) -> int:
    if explicit_expected is not None:
        return explicit_expected
    attempted = [
        int(row["manifest_attempted"])
        for row in coverage_rows
        if row.get("manifest_attempted") is not None
    ]
    if any(count >= 400 for count in attempted):
        return 499
    return 100


def _warning_lines(
    *,
    project_root: Path,
    baseline_mean: Path,
    baseline_vr: Path,
    bootstrap_mean: Path,
    bootstrap_vr: Path,
    explicit_expected_reps: int | None,
) -> list[str]:
    warnings: list[str] = []

    coverage_rows = _collect_bootstrap_coverage_rows(project_root)
    if coverage_rows is None:
        coverage_rows = []
    if coverage_rows:
        expected = _expected_bootstrap_reps(coverage_rows, explicit_expected_reps)
        manifest_gaps: list[str] = []
        for row in coverage_rows:
            if not bool(row.get("in_bootstrap_tables")):
                continue
            cohort = str(row["cohort"])
            status = str(row.get("manifest_status") or "").strip().lower()
            attempted_raw = row.get("manifest_attempted")
            attempted = None
            try:
                attempted = int(attempted_raw) if attempted_raw is not None else None
            except (TypeError, ValueError):
                attempted = None
            reasons: list[str] = []
            if status and status != "computed":
                reasons.append(f"status={status}")
            if attempted is None:
                reasons.append("attempted=missing")
            elif attempted < expected:
                reasons.append(f"attempted={attempted} < expected={expected}")
            if reasons:
                manifest_gaps.append(f"{cohort} ({'; '.join(reasons)})")
        if manifest_gaps:
            warnings.append("bootstrap manifest gaps: " + "; ".join(manifest_gaps))

    for label, baseline_path, bootstrap_path in (
        ("g_mean_diff", baseline_mean, bootstrap_mean),
        ("g_variance_ratio", baseline_vr, bootstrap_vr),
    ):
        if not bootstrap_path.exists():
            warnings.append(
                f"{label} bootstrap table missing: `{_rel(project_root, bootstrap_path)}`"
            )
            continue
        baseline_rows = _read_rows(baseline_path)
        bootstrap_rows = _read_rows(bootstrap_path)
        missing = sorted(set(bootstrap_rows) - set(baseline_rows))
        if missing:
            extra = ""
            if label == "g_mean_diff":
                exclusions_path = project_root / "outputs/tables/confirmatory_exclusions.csv"
                if exclusions_path.exists():
                    try:
                        exclusions = _read_rows(exclusions_path)
                    except Exception:
                        exclusions = {}
                    reasons: list[str] = []
                    for cohort in missing:
                        row = exclusions.get(cohort, {})
                        blocked = (row.get("blocked_confirmatory_d_g") or "").strip().lower()
                        reason = (row.get("reason_d_g") or "").strip()
                        if blocked == "true" and reason:
                            reasons.append(f"{cohort}: {reason}")
                    if reasons:
                        extra = " (primary d_g excluded: " + "; ".join(reasons) + ")"
            warnings.append(f"{label} baseline missing cohort(s): {', '.join(missing)}{extra}")

    return warnings


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.expanduser().resolve()

    tables = project_root / "outputs/tables"
    baseline_mean = tables / "g_mean_diff.csv"
    baseline_vr = tables / "g_variance_ratio.csv"
    bootstrap_mean = tables / "g_mean_diff_family_bootstrap.csv"
    bootstrap_vr = tables / "g_variance_ratio_family_bootstrap.csv"

    missing_baseline = [str(p) for p in (baseline_mean, baseline_vr) if not p.exists()]
    if missing_baseline:
        print(
            "[error] missing baseline table(s): " + ", ".join(sorted(missing_baseline)),
            file=sys.stderr,
        )
        return 1

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (tables / "results_snapshot.md").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    md: list[str] = []
    md.append("# sexg results snapshot")
    md.append("")
    md.append(f"- generated_utc: `{generated_utc}`")
    md.append(f"- project_root: `{_rel(project_root, project_root)}`")
    md.append("")

    warnings = _warning_lines(
        project_root=project_root,
        baseline_mean=baseline_mean,
        baseline_vr=baseline_vr,
        bootstrap_mean=bootstrap_mean,
        bootstrap_vr=bootstrap_vr,
        explicit_expected_reps=args.expected_bootstrap_reps,
    )
    md.append("## Warnings")
    md.append("")
    if warnings:
        for line in warnings:
            md.append(f"- {line}")
    else:
        md.append("- none")
    md.append("")

    md.append("## Primary tables")
    md.append("")
    md.append(f"- baseline: `{_rel(project_root, baseline_mean)}`")
    md.append(f"- baseline: `{_rel(project_root, baseline_vr)}`")
    if bootstrap_mean.exists():
        md.append(f"- bootstrap: `{_rel(project_root, bootstrap_mean)}`")
    else:
        md.append(f"- bootstrap: missing (`{_rel(project_root, bootstrap_mean)}`)")
    if bootstrap_vr.exists():
        md.append(f"- bootstrap: `{_rel(project_root, bootstrap_vr)}`")
    else:
        md.append(f"- bootstrap: missing (`{_rel(project_root, bootstrap_vr)}`)")
    md.append("")

    confirmatory_section = _render_confirmatory_exclusions_section(project_root).rstrip()
    if confirmatory_section:
        md.append(confirmatory_section)
        md.append("")

    proxy_section = _render_g_proxy_section(project_root).rstrip()
    if proxy_section:
        md.append(proxy_section)
        md.append("")

    exploratory_section = _render_exploratory_extensions_section(project_root).rstrip()
    if exploratory_section:
        md.append(exploratory_section)
        md.append("")

    md.append(_render_bootstrap_coverage_section(project_root).rstrip())
    md.append("")

    if args.include_run_summary:
        latest = _find_latest_run_summary(project_root)
        md.append("## Latest pipeline run summary")
        md.append("")
        if latest is None:
            md.append("- status: not found under `outputs/logs/pipeline/`")
        else:
            md.append(f"- path: `{_rel(project_root, latest)}`")
            md.append(f"- mtime_utc: `{_format_mtime_utc(latest)}`")
        md.append("")

    md.append("## Bootstrap vs baseline deltas")
    md.append("")
    md.append(
        _render_metric_section(
            project_root=project_root,
            label="g_mean_diff",
            baseline_path=baseline_mean,
            bootstrap_path=bootstrap_mean,
            estimate_col="d_g",
            se_col="SE_d_g",
            ci_low_col="ci_low_d_g",
            ci_high_col="ci_high_d_g",
        ).rstrip()
    )
    md.append(
        _render_metric_section(
            project_root=project_root,
            label="g_variance_ratio",
            baseline_path=baseline_vr,
            bootstrap_path=bootstrap_vr,
            estimate_col="VR_g",
            se_col="SE_logVR",
            ci_low_col="ci_low",
            ci_high_col="ci_high",
        ).rstrip()
    )
    md.append("")

    md.append(_render_publication_lock_section(project_root).rstrip())
    md.append("")

    output_path.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(f"[ok] wrote snapshot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
