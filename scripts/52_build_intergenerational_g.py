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

from nls_pipeline.exploratory import g_proxy, ols_fit
from nls_pipeline.io import load_yaml, project_root
from nls_pipeline.sem import hierarchical_subtests

OUTPUT_COLUMNS = [
    "model",
    "status",
    "reason",
    "n_pairs",
    "beta_mother_g",
    "se",
    "p",
    "r2",
    "beta_parent_ed",
    "source_child_data",
    "source_mother_data",
]


def _empty_row(model: str, reason: str, source_child_data: str, source_mother_data: str, *, n_pairs: int = 0) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "status": "not_feasible",
        "reason": reason,
        "n_pairs": int(n_pairs),
        "source_child_data": source_child_data,
        "source_mother_data": source_mother_data,
    }
    for col in OUTPUT_COLUMNS:
        row.setdefault(col, pd.NA)
    return row


def _load_processed(root: Path, cohort: str) -> tuple[pd.DataFrame | None, str]:
    source_path = root / "data" / "processed" / f"{cohort}_cfa_resid.csv"
    if not source_path.exists():
        source_path = root / "data" / "processed" / f"{cohort}_cfa.csv"
    source_data = str(source_path.relative_to(root)) if source_path.exists() else f"data/processed/{cohort}_cfa_resid_or_cfa.csv"
    if not source_path.exists():
        return None, source_data
    return pd.read_csv(source_path, low_memory=False), source_data


def _child_mother_link(child_df: pd.DataFrame, root: Path) -> tuple[pd.Series | None, str | None]:
    for col in ("MPUBID", "mpubid", "mother_id", "mother_person_id"):
        if col in child_df.columns:
            return pd.to_numeric(child_df[col], errors="coerce"), col

    panel_path = root / "data" / "interim" / "cnlsy" / "panel_extract.csv"
    if panel_path.exists():
        panel = pd.read_csv(panel_path, low_memory=False)
        person_col = "C0000100" if "C0000100" in panel.columns else ("person_id" if "person_id" in panel.columns else None)
        mother_col = None
        for col in ("C0000200", "MPUBID", "mpubid", "mother_id", "mother_person_id"):
            if col in panel.columns:
                mother_col = col
                break
        if person_col is not None and mother_col is not None and "person_id" in child_df.columns:
            lookup = (
                panel[[person_col, mother_col]]
                .rename(columns={person_col: "person_id", mother_col: "mother_id"})
                .dropna(subset=["person_id", "mother_id"])
                .copy()
            )
            lookup["person_id"] = pd.to_numeric(lookup["person_id"], errors="coerce")
            lookup["mother_id"] = pd.to_numeric(lookup["mother_id"], errors="coerce")
            lookup = lookup.dropna(subset=["person_id", "mother_id"]).drop_duplicates(subset=["person_id"])
            merged = child_df[["person_id"]].copy()
            merged["person_id"] = pd.to_numeric(merged["person_id"], errors="coerce")
            merged = merged.merge(lookup, on="person_id", how="left")
            return pd.to_numeric(merged["mother_id"], errors="coerce"), mother_col

    return None, None


def run_intergenerational_g(
    *,
    root: Path,
    output_path: Path = Path("outputs/tables/intergenerational_g_transmission.csv"),
    min_pairs: int = 30,
) -> pd.DataFrame:
    models_cfg = load_yaml(root / "config" / "models.yml")
    child_df, source_child_data = _load_processed(root, "cnlsy")
    mother_df, source_mother_data = _load_processed(root, "nlsy79")

    rows: list[dict[str, Any]] = []
    if child_df is None or mother_df is None:
        reason = "missing_child_source_data" if child_df is None else "missing_mother_source_data"
        for model in ("bivariate", "ses_controlled"):
            rows.append(_empty_row(model, reason, source_child_data, source_mother_data))
        out = pd.DataFrame(rows)
        out = out[OUTPUT_COLUMNS].copy()
        target = output_path if output_path.is_absolute() else root / output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)
        return out

    mother_indicators = hierarchical_subtests(models_cfg)
    child_indicators = [str(x) for x in models_cfg.get("cnlsy_single_factor", [])]

    child = child_df.copy()
    mother = mother_df.copy()
    child["__child_g"] = g_proxy(child, child_indicators)
    mother["__mother_g"] = g_proxy(mother, mother_indicators)

    mother_ids, mother_id_col = _child_mother_link(child, root)
    if mother_ids is None:
        for model in ("bivariate", "ses_controlled"):
            rows.append(_empty_row(model, "missing_mother_link_column", source_child_data, source_mother_data))
        out = pd.DataFrame(rows)
        out = out[OUTPUT_COLUMNS].copy()
        target = output_path if output_path.is_absolute() else root / output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)
        return out

    child["__mother_id"] = mother_ids
    mother_lookup = mother[["person_id", "__mother_g"]].copy()
    mother_lookup["person_id"] = pd.to_numeric(mother_lookup["person_id"], errors="coerce")
    mother_lookup = mother_lookup.rename(columns={"person_id": "__mother_id"})

    merged = child.merge(mother_lookup, on="__mother_id", how="left")
    merged["parent_ed"] = pd.to_numeric(merged["parent_education"], errors="coerce") if "parent_education" in merged.columns else pd.NA
    merged["__child_g"] = pd.to_numeric(merged["__child_g"], errors="coerce")
    merged["__mother_g"] = pd.to_numeric(merged["__mother_g"], errors="coerce")

    bivariate = merged.dropna(subset=["__child_g", "__mother_g"]).copy()
    if len(bivariate) < min_pairs:
        rows.append(_empty_row("bivariate", "insufficient_linked_pairs", source_child_data, source_mother_data, n_pairs=len(bivariate)))
    else:
        # CNLSY factor scores are built from the age-residualized child indicators,
        # so there is no additional child-age covariate to add here.
        x = pd.DataFrame({"intercept": 1.0, "mother_g": bivariate["__mother_g"]}, index=bivariate.index)
        fit, reason = ols_fit(bivariate["__child_g"], x)
        if fit is None:
            rows.append(_empty_row("bivariate", f"ols_failed:{reason or 'unknown'}", source_child_data, source_mother_data, n_pairs=len(bivariate)))
        else:
            beta = fit["beta"]
            se = fit["se"]
            p = fit["p"]
            row = {
                "model": "bivariate",
                "status": "computed",
                "reason": pd.NA,
                "n_pairs": int(fit["n_used"]),
                "beta_mother_g": float(beta[1]),
                "se": float(se[1]),
                "p": float(p[1]) if math.isfinite(float(p[1])) else pd.NA,
                "r2": float(fit["r2"]) if math.isfinite(float(fit["r2"])) else pd.NA,
                "beta_parent_ed": pd.NA,
                "source_child_data": source_child_data,
                "source_mother_data": source_mother_data,
            }
            for col in OUTPUT_COLUMNS:
                row.setdefault(col, pd.NA)
            rows.append(row)

    controlled = merged.dropna(subset=["__child_g", "__mother_g", "parent_ed"]).copy()
    if len(controlled) < min_pairs:
        rows.append(_empty_row("ses_controlled", "insufficient_linked_pairs", source_child_data, source_mother_data, n_pairs=len(controlled)))
    else:
        # Child-age variation is already removed upstream during CNLSY residualization.
        x = pd.DataFrame(
            {
                "intercept": 1.0,
                "mother_g": controlled["__mother_g"],
                "parent_ed": controlled["parent_ed"],
            },
            index=controlled.index,
        )
        fit, reason = ols_fit(controlled["__child_g"], x)
        if fit is None:
            rows.append(_empty_row("ses_controlled", f"ols_failed:{reason or 'unknown'}", source_child_data, source_mother_data, n_pairs=len(controlled)))
        else:
            beta = fit["beta"]
            se = fit["se"]
            p = fit["p"]
            row = {
                "model": "ses_controlled",
                "status": "computed",
                "reason": pd.NA,
                "n_pairs": int(fit["n_used"]),
                "beta_mother_g": float(beta[1]),
                "se": float(se[1]),
                "p": float(p[1]) if math.isfinite(float(p[1])) else pd.NA,
                "r2": float(fit["r2"]) if math.isfinite(float(fit["r2"])) else pd.NA,
                "beta_parent_ed": float(beta[2]),
                "source_child_data": source_child_data,
                "source_mother_data": source_mother_data,
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
    target = output_path if output_path.is_absolute() else root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build intergenerational CNLSY child-on-mother g_proxy regressions.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/tables/intergenerational_g_transmission.csv"))
    parser.add_argument("--min-pairs", type=int, default=30, help="Minimum linked pairs required per model.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    try:
        out = run_intergenerational_g(root=root, output_path=args.output_path, min_pairs=int(args.min_pairs))
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[ok] wrote {args.output_path if args.output_path.is_absolute() else root / args.output_path}")
    print(f"[ok] computed rows: {int((out['status'] == 'computed').sum()) if 'status' in out.columns else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
