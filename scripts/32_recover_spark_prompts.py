#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from nls_pipeline.io import project_root

TABLE_ROW_RE = re.compile(r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|?$")


@dataclass(frozen=True)
class ManifestRow:
    prompt_path: str
    spark_name: str
    status: str
    completion_state: str
    json_log: str
    stderr_log: str


@dataclass(frozen=True)
class RecoveryResult:
    prompt_path: Path
    row: ManifestRow


def parse_wave_manifest(manifest_path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        if line.startswith("| Prompt ") or line.startswith("|---"):
            continue

        match = TABLE_ROW_RE.match(line)
        if not match:
            continue

        prompt_path, spark_name, status, completion_state, json_log, stderr_log = [
            group.strip() for group in match.groups()
        ]
        if not prompt_path or not status or not completion_state:
            continue
        rows.append(
            ManifestRow(
                prompt_path=prompt_path,
                spark_name=spark_name,
                status=status,
                completion_state=completion_state,
                json_log=json_log,
                stderr_log=stderr_log,
            )
        )

    return rows


def _normalize_value(value: str) -> str:
    return value.strip().lower()


def _is_success_row(row: ManifestRow) -> bool:
    return _normalize_value(row.status) == "success"


def _is_completed_heading_state(row: ManifestRow) -> bool:
    return _normalize_value(row.completion_state) == "completed_heading"


def _resolve_prompt_path(value: str, root: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def _extract_task_id(prompt_text: str) -> str:
    heading_match = re.match(
        r"^#\s*Spark\s+Task\s+(.+?)\s*-\s*Completed$",
        prompt_text.strip(),
        flags=re.IGNORECASE,
    )
    if heading_match:
        return heading_match.group(1).strip()

    return "spark"


def _build_recovered_prompt(*, prompt_path: Path, row: ManifestRow, manifest_path: Path, root: Path) -> str:
    prompt_text = ""
    if prompt_path.exists():
        prompt_text = prompt_path.read_text(encoding="utf-8")
    heading = _extract_task_id(prompt_text)
    if heading == "spark":
        stem = prompt_path.stem
        heading = stem[5:] if stem.lower().startswith("spark") else stem
        if heading == "":
            heading = stem

    return (
        f"# Spark Task {heading} - Completed\n\n"
        "- Status: recovered\n"
        "- Recovery note: Spark runner reported success, but did not rewrite completed-heading status.\n"
        f"- Original completion state: `{row.completion_state}`\n"
        f"- Manifest path: `{manifest_path}`\n"
        f"- Prompt path: `{prompt_path}`\n"
        f"- JSON log: `{row.json_log}`\n"
        f"- STDERR log: `{row.stderr_log}`\n"
        f"- Project root: `{root}`\n"
    )


def collect_recoveries(*, manifest_rows: Iterable[ManifestRow], root: Path) -> list[RecoveryResult]:
    recoveries: list[RecoveryResult] = []
    for row in manifest_rows:
        if not _is_success_row(row):
            continue
        if _is_completed_heading_state(row):
            continue
        prompt_path = _resolve_prompt_path(row.prompt_path, root)
        recoveries.append(RecoveryResult(prompt_path=prompt_path, row=row))
    return recoveries


def recover_prompts(*, manifest_path: Path, project_root_path: Path, dry_run: bool) -> int:
    root = project_root_path.resolve()
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    rows = parse_wave_manifest(manifest_path)
    recoveries = collect_recoveries(manifest_rows=rows, root=root)

    if not recoveries:
        print("[spark-recover] no prompt rewrites required")
        return 0

    missing_files: list[Path] = []
    for recovery in sorted(recoveries, key=lambda item: str(item.prompt_path)):
        if not recovery.prompt_path.exists():
            missing_files.append(recovery.prompt_path)
            print(f"[spark-recover] skipping missing prompt: {recovery.prompt_path}", file=sys.stderr)
            continue

        new_content = _build_recovered_prompt(
            prompt_path=recovery.prompt_path,
            row=recovery.row,
            manifest_path=manifest_path,
            root=root,
        )

        if dry_run:
            print(f"[dry-run] would recover: {recovery.prompt_path}")
            print(f"         json_log: {recovery.row.json_log}")
            print(f"         stderr_log: {recovery.row.stderr_log}")
            continue

        recovery.prompt_path.write_text(new_content, encoding="utf-8")
        print(f"[spark-recover] recovered: {recovery.prompt_path}")

    if missing_files:
        return 2
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover spark prompt headings from a wave manifest."
    )
    parser.add_argument("manifest_path", type=Path, help="Path to spark wave manifest markdown file.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root(),
        help="Project root path (default: repo root).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print intended rewrites without editing prompt files.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    return recover_prompts(
        manifest_path=Path(args.manifest_path),
        project_root_path=Path(args.project_root),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
