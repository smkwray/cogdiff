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

from nls_pipeline.io import project_root

OUTPUT_COLUMNS = [
    "cohort",
    "status",
    "reason",
    "d_tilt",
    "se_d_tilt",
    "d_g_proxy",
    "tilt_to_g_ratio_abs",
    "tilt_g_corr",
    "tilt_incremental_r2_education",
    "p_tilt_incremental",
    "incremental_r2_band",
    "tilt_vs_g_band",
    "interpretation",
    "source_tables",
]


def _to_float(raw: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _empty_row(cohort: str, reason: str, source_tables: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cohort": cohort,
        "status": "not_feasible",
        "reason": reason,
        "source_tables": source_tables,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _r2_band(value: float | None) -> str | pd._libs.missing.NAType:
    if value is None or not math.isfinite(value):
        return pd.NA
    if value < 0.001:
        return "negligible"
    if value < 0.005:
        return "very_small"
    if value < 0.02:
        return "modest"
    return "large"


def _ratio_band(value: float | None) -> str | pd._libs.missing.NAType:
    if value is None or not math.isfinite(value):
        return pd.NA
    if value < 0.25:
        return "much_smaller_than_g"
    if value < 0.75:
        return "smaller_than_g"
    if value < 1.25:
        return "similar_to_g"
    return "larger_than_g"


def _interpretation(
    *,
    incremental_r2: float | None,
    p_value: float | None,
    ratio_abs: float | None,
) -> str | pd._libs.missing.NAType:
    if incremental_r2 is None or not math.isfinite(incremental_r2):
        return pd.NA
    statistically_detectable = p_value is not None and math.isfinite(p_value) and p_value < 0.05
    if incremental_r2 < 0.005:
        return "small_add_on" if statistically_detectable else "small_or_null_add_on"
    if incremental_r2 < 0.02:
        if ratio_abs is not None and math.isfinite(ratio_abs) and ratio_abs >= 0.75:
            return "meaningful_secondary_dimension"
        return "modest_secondary_dimension"
    return "meaningful_secondary_dimension"


def run_tilt_interpretation(
    *,
    root: Path,
    output_path: Path = Path("outputs/tables/subtest_profile_tilt_summary.csv"),
) -> pd.DataFrame:
    tilt_path = root / "outputs" / "tables" / "subtest_profile_tilt.csv"
    g_proxy_path = root / "outputs" / "tables" / "g_proxy_mean_diff_family_bootstrap.csv"
    source_tables = f"{tilt_path.relative_to(root)},{g_proxy_path.relative_to(root)}"

    if not tilt_path.exists() or not g_proxy_path.exists():
        missing = []
        if not tilt_path.exists():
            missing.append("missing_tilt_table")
        if not g_proxy_path.exists():
            missing.append("missing_g_proxy_table")
        out = pd.DataFrame([_empty_row("", "+".join(missing), source_tables)], columns=OUTPUT_COLUMNS)
    else:
        tilt_df = pd.read_csv(tilt_path)
        g_df = pd.read_csv(g_proxy_path)
        g_map = {
            str(row["cohort"]): row
            for _, row in g_df.loc[g_df["status"].astype("string").str.lower() == "computed"].iterrows()
        }
        rows: list[dict[str, Any]] = []
        for _, row in tilt_df.iterrows():
            cohort = str(row.get("cohort") or "").strip()
            if not cohort:
                continue
            if str(row.get("status") or "").strip().lower() != "computed":
                rows.append(_empty_row(cohort, str(row.get("reason") or "tilt_not_computed"), source_tables))
                continue
            g_row = g_map.get(cohort)
            if g_row is None:
                rows.append(_empty_row(cohort, "missing_g_proxy_row", source_tables))
                continue
            d_tilt = _to_float(row.get("d_tilt"))
            d_g_proxy = _to_float(g_row.get("d_g"))
            ratio_abs = None
            if d_tilt is not None and d_g_proxy is not None and d_g_proxy != 0.0:
                ratio_abs = abs(d_tilt) / abs(d_g_proxy)
            incremental_r2 = _to_float(row.get("tilt_incremental_r2_education"))
            p_value = _to_float(row.get("p_tilt_incremental"))
            out_row = {
                "cohort": cohort,
                "status": "computed",
                "reason": pd.NA,
                "d_tilt": d_tilt if d_tilt is not None and math.isfinite(d_tilt) else pd.NA,
                "se_d_tilt": _to_float(row.get("se_d_tilt")),
                "d_g_proxy": d_g_proxy if d_g_proxy is not None and math.isfinite(d_g_proxy) else pd.NA,
                "tilt_to_g_ratio_abs": ratio_abs if ratio_abs is not None and math.isfinite(ratio_abs) else pd.NA,
                "tilt_g_corr": _to_float(row.get("tilt_g_corr")),
                "tilt_incremental_r2_education": incremental_r2 if incremental_r2 is not None and math.isfinite(incremental_r2) else pd.NA,
                "p_tilt_incremental": p_value if p_value is not None and math.isfinite(p_value) else pd.NA,
                "incremental_r2_band": _r2_band(incremental_r2),
                "tilt_vs_g_band": _ratio_band(ratio_abs),
                "interpretation": _interpretation(incremental_r2=incremental_r2, p_value=p_value, ratio_abs=ratio_abs),
                "source_tables": source_tables,
            }
            for col in OUTPUT_COLUMNS:
                out_row.setdefault(col, pd.NA)
            rows.append(out_row)
        out = pd.DataFrame(rows)
        if out.empty:
            out = pd.DataFrame([_empty_row("", "no_rows", source_tables)], columns=OUTPUT_COLUMNS)

    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OUTPUT_COLUMNS].copy()
    target = output_path if output_path.is_absolute() else root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize whether verbal-quantitative tilt adds meaningfully beyond g_proxy.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/subtest_profile_tilt_summary.csv"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        out = run_tilt_interpretation(root=root, output_path=args.output_path)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.output_path if args.output_path.is_absolute() else root / args.output_path}")
    print(f"[ok] computed rows: {int((out['status'] == 'computed').sum()) if 'status' in out.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
