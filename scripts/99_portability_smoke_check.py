#!/usr/bin/env python3
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCAN_PATHS: tuple[str, ...] = (
    "data/interim/links",
    "data/interim/sem",
    "outputs/model_fits",
    "outputs/tables",
)

DEFAULT_FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "/Users/",
    "CloudStorage/OneDrive-Personal",
    "CloudStorage/GoogleDrive-",
)

DEFAULT_EXCLUDE_GLOBS: tuple[str, ...] = (
    "logs/spark/**",
    "outputs/logs/**",
    "do/legacy_project_prompts_*/**",
)


def _resolve_path(root: Path, token: str) -> Path:
    path = Path(token)
    return path if path.is_absolute() else root / path


def _existing_scan_paths(root: Path, tokens: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for token in tokens:
        resolved = _resolve_path(root, token)
        if resolved.exists():
            paths.append(resolved)
    return paths


def _is_excluded(root: Path, path: Path, exclude_globs: Iterable[str]) -> bool:
    rel = path.resolve().relative_to(root.resolve())
    return any(rel.match(pattern) for pattern in exclude_globs)


def _scan_with_rg(
    *,
    root: Path,
    scan_paths: list[Path],
    forbidden_patterns: list[str],
    exclude_globs: list[str],
) -> list[str]:
    cmd = ["rg", "--line-number", "--with-filename", "--no-heading", "--color=never"]
    for pattern in forbidden_patterns:
        cmd.extend(["-e", pattern])
    for pattern in exclude_globs:
        cmd.extend(["--glob", f"!{pattern}"])
    cmd.extend([str(path) for path in scan_paths])
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=False)
    if proc.returncode == 1:
        return []
    if proc.returncode == 0:
        return [line for line in proc.stdout.splitlines() if line.strip()]
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    detail = stderr or stdout or "unknown rg error"
    raise RuntimeError(f"rg failed with exit code {proc.returncode}: {detail}")


def _scan_with_python(
    *,
    root: Path,
    scan_paths: list[Path],
    forbidden_patterns: list[str],
    exclude_globs: list[str],
) -> list[str]:
    hits: list[str] = []
    for base in scan_paths:
        files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        for file_path in files:
            if _is_excluded(root, file_path, exclude_globs):
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                if any(pattern in line for pattern in forbidden_patterns):
                    rel = file_path.resolve().relative_to(root.resolve())
                    hits.append(f"{rel}:{idx}:{line}")
    return hits


def scan_forbidden_paths(
    *,
    root: Path,
    scan_path_tokens: list[str],
    forbidden_patterns: list[str],
    exclude_globs: list[str],
) -> list[str]:
    scan_paths = _existing_scan_paths(root, scan_path_tokens)
    if not scan_paths:
        return []
    if shutil.which("rg"):
        return _scan_with_rg(
            root=root,
            scan_paths=scan_paths,
            forbidden_patterns=forbidden_patterns,
            exclude_globs=exclude_globs,
        )
    return _scan_with_python(
        root=root,
        scan_paths=scan_paths,
        forbidden_patterns=forbidden_patterns,
        exclude_globs=exclude_globs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Portability smoke-check for absolute/local-root paths in tracked artifacts.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root path.")
    parser.add_argument(
        "--scan-path",
        action="append",
        dest="scan_paths",
        help="Relative or absolute path to scan. Repeatable. Defaults to key artifact directories.",
    )
    parser.add_argument(
        "--forbidden-pattern",
        action="append",
        dest="forbidden_patterns",
        help="Regex/text pattern considered non-portable. Repeatable.",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        dest="exclude_globs",
        help="Glob pattern to exclude from scan. Repeatable.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    scan_paths = [str(item) for item in (args.scan_paths or list(DEFAULT_SCAN_PATHS))]
    forbidden_patterns = [str(item) for item in (args.forbidden_patterns or list(DEFAULT_FORBIDDEN_PATTERNS))]
    exclude_globs = [str(item) for item in (args.exclude_globs or list(DEFAULT_EXCLUDE_GLOBS))]

    matches = scan_forbidden_paths(
        root=root,
        scan_path_tokens=scan_paths,
        forbidden_patterns=forbidden_patterns,
        exclude_globs=exclude_globs,
    )
    if matches:
        print("[fail] portability smoke check found forbidden path matches:")
        for line in matches:
            print(line)
        return 1

    print("[ok] portability smoke check found no forbidden path matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
