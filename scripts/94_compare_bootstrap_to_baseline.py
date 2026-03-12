#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline vs family-bootstrap tables for stage-20 outputs and "
            "print concise cohort-level deltas."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_repo_root(),
        help="Project root containing outputs/tables (default: repo root).",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"missing header in {path}")
        rows = {}
        for row in reader:
            cohort = (row.get("cohort") or "").strip()
            if not cohort:
                continue
            rows[cohort] = {k: (v or "").strip() for k, v in row.items()}
    return list(reader.fieldnames), rows


def _to_float(raw: str) -> float | None:
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


def _fmt_ci(low_raw: str, high_raw: str) -> str:
    low = _to_float(low_raw)
    high = _to_float(high_raw)
    if low is None or high is None:
        return "-"
    return f"[{_fmt_num(low)}, {_fmt_num(high)}]"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = []
    for col_idx, header in enumerate(headers):
        max_cell = max((len(row[col_idx]) for row in rows), default=0)
        widths.append(max(len(header), max_cell))
    print("  " + "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  " + "  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  " + "  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def _render_comparison(
    label: str,
    baseline_path: Path,
    bootstrap_path: Path,
    estimate_col: str,
    se_col: str,
    ci_low_col: str,
    ci_high_col: str,
) -> int:
    try:
        _, baseline_rows = _read_rows(baseline_path)
        _, bootstrap_rows = _read_rows(bootstrap_path)
    except (OSError, ValueError) as exc:
        print(f"[error] failed reading comparison inputs: {exc}", file=sys.stderr)
        return 1

    cohorts = sorted(set(baseline_rows) | set(bootstrap_rows))
    print(f"comparison: {label}")
    print(f"baseline: {baseline_path}")
    print(f"bootstrap: {bootstrap_path}")

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
        base_est = _to_float(base.get(estimate_col, ""))
        boot_est = _to_float(boot.get(estimate_col, ""))
        delta = None if base_est is None or boot_est is None else boot_est - base_est
        status = boot.get("status", "")
        if boot.get("reason", ""):
            status = f"{status}:{boot.get('reason', '')}" if status else boot["reason"]
        rows.append(
            [
                cohort,
                _fmt_num(base_est),
                _fmt_num(boot_est),
                _fmt_num(delta),
                _fmt_num(_to_float(base.get(se_col, ""))),
                _fmt_num(_to_float(boot.get(se_col, ""))),
                _fmt_ci(base.get(ci_low_col, ""), base.get(ci_high_col, "")),
                _fmt_ci(boot.get(ci_low_col, ""), boot.get(ci_high_col, "")),
                status or "-",
            ]
        )

    _print_table(headers, rows)
    print()
    return 0


def main() -> int:
    args = _parse_args()
    root = args.project_root.expanduser().resolve()
    tables = root / "outputs" / "tables"
    paths = {
        "g_mean_diff_baseline": tables / "g_mean_diff.csv",
        "g_mean_diff_bootstrap": tables / "g_mean_diff_family_bootstrap.csv",
        "g_variance_ratio_baseline": tables / "g_variance_ratio.csv",
        "g_variance_ratio_bootstrap": tables / "g_variance_ratio_family_bootstrap.csv",
    }

    missing_bootstrap = [
        str(path)
        for key, path in paths.items()
        if key.endswith("_bootstrap") and not path.exists()
    ]
    if missing_bootstrap:
        print(
            "[error] missing bootstrap comparison file(s): "
            + ", ".join(sorted(missing_bootstrap)),
            file=sys.stderr,
        )
        return 1

    missing_baseline = [
        str(path)
        for key, path in paths.items()
        if key.endswith("_baseline") and not path.exists()
    ]
    if missing_baseline:
        print(
            "[error] missing baseline comparison file(s): "
            + ", ".join(sorted(missing_baseline)),
            file=sys.stderr,
        )
        return 1

    rc = 0
    rc |= _render_comparison(
        label="g_mean_diff",
        baseline_path=paths["g_mean_diff_baseline"],
        bootstrap_path=paths["g_mean_diff_bootstrap"],
        estimate_col="d_g",
        se_col="SE_d_g",
        ci_low_col="ci_low_d_g",
        ci_high_col="ci_high_d_g",
    )
    rc |= _render_comparison(
        label="g_variance_ratio",
        baseline_path=paths["g_variance_ratio_baseline"],
        bootstrap_path=paths["g_variance_ratio_bootstrap"],
        estimate_col="VR_g",
        se_col="SE_logVR",
        ci_low_col="ci_low",
        ci_high_col="ci_high",
    )
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
