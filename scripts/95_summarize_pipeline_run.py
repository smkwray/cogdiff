#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SUMMARY_GLOB = "*_pipeline_run_summary.csv"
PATH_FIELD_HINTS = {
    "log_file",
    "output_path",
    "output_file",
    "summary_path",
    "artifact_path",
    "result_path",
    "table_path",
    "figure_path",
}
PATH_FIELD_SUFFIXES = ("_path", "_file", "_csv", "_json")
STAGE20_BOOTSTRAP_OUTPUTS = (
    "outputs/tables/g_mean_diff_family_bootstrap.csv",
    "outputs/tables/g_variance_ratio_family_bootstrap.csv",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the latest pipeline run summary CSV and print a concise "
            "stage/status table with referenced output paths."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Explicit pipeline run summary CSV path.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_repo_root(),
        help="Project root used when locating latest summary (default: repo root).",
    )
    parser.add_argument(
        "--show-stage20-outputs",
        action="store_true",
        help=(
            "Optionally report a stage20_artifacts block with presence/mtime for "
            "stage-20 bootstrap output tables."
        ),
    )
    return parser.parse_args()


def _find_latest_summary(project_root: Path) -> Path:
    summaries = sorted((project_root / "outputs/logs/pipeline").glob(SUMMARY_GLOB))
    if not summaries:
        raise FileNotFoundError(
            f"no summary files found under {project_root / 'outputs/logs/pipeline'}"
        )
    return max(summaries, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _load_summary(summary_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"summary appears to have no header: {summary_path}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    return list(reader.fieldnames), rows


def _print_stage_table(rows: list[dict[str, str]]) -> None:
    columns = ("stage", "status", "returncode", "script")
    labels = ("stage", "status", "rc", "script")
    table_rows = [[row.get(col, "") for col in columns] for row in rows]
    widths = []
    for idx, label in enumerate(labels):
        max_cell = max((len(row[idx]) for row in table_rows), default=0)
        widths.append(max(len(label), max_cell))

    header = "  ".join(label.ljust(widths[idx]) for idx, label in enumerate(labels))
    print("stage_status_table:")
    print(header)
    print("  ".join("-" * widths[idx] for idx in range(len(widths))))
    for row in table_rows:
        print("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(widths))))


def _collect_referenced_paths(
    fieldnames: list[str], rows: list[dict[str, str]], summary_path: Path
) -> list[Path]:
    candidate_fields = [
        field
        for field in fieldnames
        if field in PATH_FIELD_HINTS or field.lower().endswith(PATH_FIELD_SUFFIXES)
    ]
    paths: list[Path] = []
    seen: set[Path] = set()
    for row in rows:
        for field in candidate_fields:
            raw = row.get(field, "")
            if not raw:
                continue
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = (summary_path.parent / candidate).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            paths.append(candidate)
    return paths


def _format_mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _print_stage20_artifacts(project_root: Path) -> None:
    print("stage20_artifacts:")
    for rel_path in STAGE20_BOOTSTRAP_OUTPUTS:
        candidate = project_root / rel_path
        if candidate.exists():
            print(f"  - {rel_path}: present mtime_utc={_format_mtime_utc(candidate)}")
        else:
            print(f"  - {rel_path}: missing")


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.expanduser().resolve()
    summary_path = (
        args.path.expanduser().resolve()
        if args.path is not None
        else _find_latest_summary(project_root)
    )

    if not summary_path.exists():
        print(f"[error] summary path does not exist: {summary_path}", file=sys.stderr)
        return 1

    try:
        fieldnames, rows = _load_summary(summary_path)
    except (OSError, ValueError) as exc:
        print(f"[error] failed reading summary: {exc}", file=sys.stderr)
        return 1

    status_counts = Counter(row.get("status", "") for row in rows if row.get("status", ""))
    print(f"run_summary: {summary_path}")
    print(f"row_count: {len(rows)}")
    if status_counts:
        compact = ", ".join(f"{key}={status_counts[key]}" for key in sorted(status_counts))
        print(f"status_counts: {compact}")
    else:
        print("status_counts: (none)")

    _print_stage_table(rows)

    referenced = _collect_referenced_paths(fieldnames, rows, summary_path)
    print("referenced_paths:")
    if referenced:
        for path in referenced:
            print(f"  - {path}")
    else:
        print("  (none)")

    if args.show_stage20_outputs:
        _print_stage20_artifacts(project_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
