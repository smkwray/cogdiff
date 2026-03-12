#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/96_clean_repo_caches.sh [--path <dir>] [--dry-run]

Removes common cache artifacts:
  - __pycache__/
  - .pytest_cache/
  - .ruff_cache/
  - .mypy_cache/
  - *.pyc
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path)
      TARGET_DIR="$2"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "[error] unknown arg: $1" >&2
      usage
      exit 2;;
  esac
done

cd "$TARGET_DIR"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] would remove:"
  find . -type d -name '__pycache__' -prune -print || true
  find . -type d -name '.pytest_cache' -prune -print || true
  find . -type d -name '.ruff_cache' -prune -print || true
  find . -type d -name '.mypy_cache' -prune -print || true
  find . -type f -name '*.pyc' -print || true
  exit 0
fi

echo "[run] removing cache directories/files under: $TARGET_DIR"

find . -type d -name '__pycache__' -prune -exec rm -rf {} + || true
find . -type d -name '.pytest_cache' -prune -exec rm -rf {} + || true
find . -type d -name '.ruff_cache' -prune -exec rm -rf {} + || true
find . -type d -name '.mypy_cache' -prune -exec rm -rf {} + || true
find . -type f -name '*.pyc' -delete || true

echo "[ok] cache cleanup complete"
