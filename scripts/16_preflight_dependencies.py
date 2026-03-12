#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import project_root
from nls_pipeline import preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Check runtime dependencies required by the pipeline.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root(),
        help="Project root path.",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        default=[],
        help="Repeatable cohort filter. Example: --cohort nlsy79.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all cohorts' file requirements.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code on critical prerequisite failures.",
    )

    args = parser.parse_args()
    root = args.project_root.resolve()
    code, status_path, summary_path, summary = preflight.run_preflight(
        root,
        cohorts=args.cohort,
        all_cohorts=args.all,
        strict=args.strict,
    )
    print(f"[ok] wrote {status_path}")
    print(f"[ok] wrote {summary_path}")
    if summary["overall"] == "fail":
        print("[warn] critical checks failed")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
