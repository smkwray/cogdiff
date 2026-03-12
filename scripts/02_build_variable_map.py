#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.codebook import build_variable_map_from_header
from nls_pipeline.io import load_yaml, resolve_from_project

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap variable map from extracted cohort CSV header.")
    parser.add_argument("--cohort", required=True, choices=sorted(COHORT_CONFIGS))
    parser.add_argument("--csv", help="Optional explicit CSV path")
    args = parser.parse_args()

    paths_cfg = load_yaml(resolve_from_project("config/paths.yml"))
    interim_dir = resolve_from_project(paths_cfg["interim_dir"])

    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_candidates = sorted((interim_dir / args.cohort / "raw_files").rglob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(f"No CSV found under {interim_dir / args.cohort / 'raw_files'}")
        csv_path = csv_candidates[0]

    out_path = interim_dir / args.cohort / "varmap.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame = build_variable_map_from_header(csv_path)
    frame.to_csv(out_path, index=False)

    print(f"[ok] wrote {len(frame)} variables to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
