#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import dump_json, load_yaml, project_root, relative_path, sha256_file

DETERMINISM_KEY_OUTPUTS: tuple[str, ...] = (
    "outputs/tables/sample_counts.csv",
    "outputs/tables/g_mean_diff.csv",
    "outputs/tables/g_variance_ratio.csv",
    "outputs/tables/specification_stability_summary.csv",
)


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _collect_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        if not path.exists():
            continue
        files.extend([p for p in path.rglob("*") if p.is_file()])
    files.sort()
    return files


def _hash_records(
    *,
    root: Path,
    files: list[Path],
    artifact_group: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_path in files:
        stat = file_path.stat()
        records.append(
            {
                "artifact_group": artifact_group,
                "relative_path": relative_path(root, file_path),
                "size_bytes": int(stat.st_size),
                "modified_utc": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
                "sha256": sha256_file(file_path),
            }
        )
    return records


def _latest_pipeline_summary(logs_pipeline_dir: Path) -> Path | None:
    summaries = sorted(logs_pipeline_dir.glob("*_pipeline_run_summary.csv"))
    if not summaries:
        return None
    return summaries[-1]


def _sanitize_provenance_command(root: Path, command: Any) -> str:
    text = str(command or "").strip()
    if not text or text.lower() == "nan":
        return ""
    sanitized = text.replace(str(root.resolve()), "<PROJECT_ROOT>")
    return re.sub(r"^/\S*/(python(?:\d+(?:\.\d+)*)?)\b", r"\1", sanitized)


def _pipeline_provenance_rows(root: Path, summary_path: Path | None) -> list[dict[str, Any]]:
    if summary_path is None or (not summary_path.exists()):
        return []

    summary = pd.read_csv(summary_path)
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "source": "pipeline_summary",
                "relative_source_path": relative_path(root, summary_path),
                "command": _sanitize_provenance_command(root, row.get("command", "")),
                "status": str(row.get("status", "")),
                "stage": row.get("stage", pd.NA),
                "script": row.get("script", pd.NA),
                "cohort_scope": row.get("cohort_scope", pd.NA),
                "return_code": row.get("returncode", pd.NA),
                "details": row.get("note", pd.NA),
            }
        )
    return rows


def _rerun_provenance_rows(root: Path, rerun_log_path: Path) -> list[dict[str, Any]]:
    if not rerun_log_path.exists():
        return []
    rerun_log = pd.read_csv(rerun_log_path)
    if rerun_log.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, row in rerun_log.iterrows():
        command = _sanitize_provenance_command(root, row.get("command", ""))
        if not command:
            continue
        rows.append(
            {
                "source": "robustness_rerun",
                "relative_source_path": relative_path(root, rerun_log_path),
                "command": command,
                "status": row.get("status", pd.NA),
                "stage": pd.NA,
                "script": "11_robustness_suite.py",
                "cohort_scope": row.get("cohort", pd.NA),
                "return_code": row.get("return_code", pd.NA),
                "details": f"{row.get('robustness_family', '')}:{row.get('variant_token', '')}",
            }
        )
    return rows


def _run_capture(command: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return 1, ""
    parts = [str(proc.stdout or "").strip(), str(proc.stderr or "").strip()]
    output = "\n".join([p for p in parts if p]).strip()
    return int(proc.returncode), output


def _git_metadata(root: Path) -> dict[str, Any]:
    commit_rc, commit = _run_capture(["git", "rev-parse", "HEAD"], cwd=root)
    dirty_rc, dirty_status = _run_capture(["git", "status", "--porcelain"], cwd=root)
    branch_rc, branch = _run_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    return {
        "available": bool(commit_rc == 0 and commit),
        "commit": commit if commit_rc == 0 and commit else None,
        "branch": branch if branch_rc == 0 and branch else None,
        "dirty": bool(dirty_status.strip()) if dirty_rc == 0 else None,
    }


def _environment_snapshot() -> dict[str, Any]:
    pip_rc, pip_freeze = _run_capture([sys.executable, "-m", "pip", "freeze"])
    rscript_path = shutil.which("Rscript")
    r_version_rc = 1
    r_version = ""
    r_session_rc = 1
    r_session = ""
    if rscript_path:
        r_version_rc, r_version = _run_capture([rscript_path, "--version"])
        r_session_rc, r_session = _run_capture([rscript_path, "-e", "sessionInfo()"])
    python_executable = Path(sys.executable).name if sys.executable else None
    return {
        "python_executable": python_executable,
        "python_version": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pip_freeze": pip_freeze.splitlines() if pip_rc == 0 and pip_freeze else [],
        "rscript_executable": rscript_path,
        "r_version": r_version if r_version_rc == 0 and r_version else None,
        "r_session_info": r_session.splitlines() if r_session_rc == 0 and r_session else [],
    }


def _hash_lookup_by_path(hash_df: pd.DataFrame) -> dict[str, str]:
    if hash_df.empty:
        return {}
    out: dict[str, str] = {}
    for row in hash_df.to_dict(orient="records"):
        rel = str(row.get("relative_path", "")).strip()
        sha = str(row.get("sha256", "")).strip()
        if rel and sha:
            out[rel] = sha
    return out


def _load_previous_hashes(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["artifact_group", "relative_path", "size_bytes", "modified_utc", "sha256"])
    try:
        previous = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["artifact_group", "relative_path", "size_bytes", "modified_utc", "sha256"])
    if previous.empty:
        return pd.DataFrame(columns=["artifact_group", "relative_path", "size_bytes", "modified_utc", "sha256"])
    return previous


def _build_determinism_audit(
    *,
    current_hash_df: pd.DataFrame,
    previous_hash_df: pd.DataFrame,
) -> pd.DataFrame:
    current_lookup = _hash_lookup_by_path(current_hash_df)
    previous_lookup = _hash_lookup_by_path(previous_hash_df)
    rows: list[dict[str, Any]] = []
    for relative_path in DETERMINISM_KEY_OUTPUTS:
        current_sha = current_lookup.get(relative_path)
        previous_sha = previous_lookup.get(relative_path)
        if not current_sha:
            status = "missing_current"
        elif not previous_sha:
            status = "no_previous"
        elif current_sha == previous_sha:
            status = "unchanged"
        else:
            status = "changed"
        rows.append(
            {
                "relative_path": relative_path,
                "status": status,
                "current_sha256": current_sha if current_sha else pd.NA,
                "previous_sha256": previous_sha if previous_sha else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _determinism_summary(audit_df: pd.DataFrame) -> dict[str, int]:
    if audit_df.empty or "status" not in audit_df.columns:
        return {
            "determinism_changed_count": 0,
            "determinism_unchanged_count": 0,
            "determinism_missing_current_count": 0,
            "determinism_no_previous_count": 0,
        }
    statuses = audit_df["status"].astype(str)
    return {
        "determinism_changed_count": int((statuses == "changed").sum()),
        "determinism_unchanged_count": int((statuses == "unchanged").sum()),
        "determinism_missing_current_count": int((statuses == "missing_current").sum()),
        "determinism_no_previous_count": int((statuses == "no_previous").sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reproducibility hash and command-provenance artifacts.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root path.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    paths_cfg = load_yaml(root / "config/paths.yml")
    outputs_dir = _resolve_path(paths_cfg["outputs_dir"], root)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [
        root / "config",
        _resolve_path(paths_cfg.get("raw_dir", "data/raw"), root),
        _resolve_path(paths_cfg.get("interim_dir", "data/interim"), root),
        _resolve_path(paths_cfg.get("processed_dir", "data/processed"), root),
    ]
    output_paths = [
        outputs_dir / "tables",
        outputs_dir / "figures",
        outputs_dir / "model_fits",
    ]
    log_paths = [outputs_dir / "logs"]

    hashes_path = tables_dir / "reproducibility_hashes.csv"
    previous_hash_df = _load_previous_hashes(hashes_path)

    hash_rows: list[dict[str, Any]] = []
    hash_rows.extend(_hash_records(root=root, files=_collect_files(input_paths), artifact_group="input"))
    hash_rows.extend(_hash_records(root=root, files=_collect_files(output_paths), artifact_group="output"))
    hash_rows.extend(_hash_records(root=root, files=_collect_files(log_paths), artifact_group="log"))
    hash_df = pd.DataFrame(hash_rows)
    if hash_df.empty:
        hash_df = pd.DataFrame(columns=["artifact_group", "relative_path", "size_bytes", "modified_utc", "sha256"])
    else:
        hash_df = hash_df.sort_values(["artifact_group", "relative_path"]).reset_index(drop=True)

    hash_df.to_csv(hashes_path, index=False)

    determinism_audit_df = _build_determinism_audit(
        current_hash_df=hash_df,
        previous_hash_df=previous_hash_df,
    )
    determinism_audit_path = tables_dir / "reproducibility_determinism_audit.csv"
    determinism_audit_df.to_csv(determinism_audit_path, index=False)
    determinism_counts = _determinism_summary(determinism_audit_df)

    logs_pipeline_dir = outputs_dir / "logs" / "pipeline"
    latest_summary_path = _latest_pipeline_summary(logs_pipeline_dir)
    rerun_log_path = tables_dir / "robustness_rerun_log.csv"
    provenance_rows: list[dict[str, Any]] = []
    provenance_rows.extend(_pipeline_provenance_rows(root, latest_summary_path))
    provenance_rows.extend(_rerun_provenance_rows(root, rerun_log_path))
    provenance_df = pd.DataFrame(provenance_rows)
    if provenance_df.empty:
        provenance_df = pd.DataFrame(
            columns=["source", "relative_source_path", "command", "status", "stage", "script", "cohort_scope", "return_code", "details"]
        )
    provenance_path = tables_dir / "reproducibility_command_provenance.csv"
    provenance_df.to_csv(provenance_path, index=False)

    environment_payload = {
        "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "environment": _environment_snapshot(),
        "git": _git_metadata(root),
    }
    environment_path = tables_dir / "reproducibility_environment.json"
    dump_json(environment_path, environment_payload)

    summary_payload = {
        "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "project_root": relative_path(root, root),
        "hashes_csv": relative_path(root, hashes_path),
        "command_provenance_csv": relative_path(root, provenance_path),
        "environment_json": relative_path(root, environment_path),
        "determinism_audit_csv": relative_path(root, determinism_audit_path),
        "n_hashed_files": int(len(hash_df)),
        "n_provenance_rows": int(len(provenance_df)),
        "latest_pipeline_summary": relative_path(root, latest_summary_path) if latest_summary_path else None,
        "git": environment_payload["git"],
        **determinism_counts,
    }
    summary_path = tables_dir / "reproducibility_report_summary.json"
    dump_json(summary_path, summary_payload)

    print(f"[ok] wrote {hashes_path}")
    print(f"[ok] wrote {determinism_audit_path}")
    print(f"[ok] wrote {provenance_path}")
    print(f"[ok] wrote {environment_path}")
    print(f"[ok] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
