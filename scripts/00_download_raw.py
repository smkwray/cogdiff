#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import dump_json, load_yaml, project_root, relative_path, resolve_from_project, sha256_file, utc_timestamp

COHORT_CONFIGS = {
    "nlsy79": "config/nlsy79.yml",
    "nlsy97": "config/nlsy97.yml",
    "cnlsy": "config/cnlsy.yml",
}


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        with out_path.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    pbar.update(len(chunk))


def _selected_cohorts(args: argparse.Namespace) -> list[str]:
    if args.all or not args.cohort:
        return list(COHORT_CONFIGS.keys())
    return args.cohort


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NLS cohort raw zip files and write manifest.")
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS), help="Specific cohort(s) to download.")
    parser.add_argument("--all", action="store_true", help="Download all cohorts.")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    args = parser.parse_args()

    root = project_root()
    paths_cfg = load_yaml(resolve_from_project("config/paths.yml"))
    raw_dir = resolve_from_project(paths_cfg["raw_dir"])
    manifest_path = resolve_from_project(paths_cfg["manifest_file"])

    cohorts = _selected_cohorts(args)
    manifest: dict[str, dict[str, object]] = {}

    for cohort in cohorts:
        cfg = load_yaml(resolve_from_project(COHORT_CONFIGS[cohort]))
        url = cfg["download_url"]
        out_path = raw_dir / cfg["raw_zip_name"]

        if out_path.exists() and not args.force:
            print(f"[skip] {cohort}: {out_path} already exists")
        else:
            print(f"[download] {cohort}: {url}")
            _download(url, out_path)

        manifest[cohort] = {
            "url": url,
            "path": relative_path(root, out_path),
            "size_bytes": out_path.stat().st_size,
            "sha256": sha256_file(out_path),
            "recorded_at_utc": utc_timestamp(),
        }

    dump_json(manifest_path, manifest)
    print(f"[ok] wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
