#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SEXG_VENV:-${VIRTUAL_ENV:-}}"

usage() {
  cat <<EOF
Usage:
  scripts/97_local_verify.sh [--fast] [--artifacts] [--self-test]

Runs local verification in a cache-clean way:
  - pytest (with cache provider disabled)
  - portability smoke check
  - optional publication-artifact doctor (`--artifacts`)

Environment:
  - uses venv at: $VENV (optional; set SEXG_VENV=/path/to/venv or activate a venv first)
  - sets PYTHONDONTWRITEBYTECODE=1 and PYTHONPATH=src
EOF
}

FAST=0
ARTIFACTS=0
SELF_TEST=0
if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --fast) FAST=1; shift 1;;
      --artifacts) ARTIFACTS=1; shift 1;;
      --self-test) SELF_TEST=1; shift 1;;
      -h|--help) usage; exit 0;;
      *) echo "[error] unknown arg: $1" >&2; usage; exit 2;;
    esac
  done
fi
if [[ "$SELF_TEST" -eq 1 ]]; then
  echo "[self-test] root=$ROOT"
  echo "[self-test] venv=$VENV"
  echo "[self-test] fast=$FAST"
  echo "[self-test] artifacts=$ARTIFACTS"
  echo "[self-test] would run:"
  echo "  scripts/96_clean_repo_caches.sh"
  if [[ "$FAST" -eq 1 ]]; then
    echo "  python -m pytest -q -p no:cacheprovider tests/test_script_99_portability_smoke_check.py"
  else
    echo "  python -m pytest -q -p no:cacheprovider"
  fi
  echo "  scripts/99_portability_smoke_check.py --project-root \"$ROOT\""
  if [[ "$ARTIFACTS" -eq 1 ]]; then
    echo "  python scripts/95_doctor_publication_artifacts.py --project-root \"$ROOT\""
  fi
  echo "  scripts/96_clean_repo_caches.sh"
  exit 0
fi

if [[ -n "${VENV}" ]]; then
  if [[ ! -d "$VENV" ]]; then
    echo "[error] missing venv: $VENV" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
else
  echo "[warn] No venv detected (set SEXG_VENV or activate a venv first). Using python from PATH." >&2
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src"

cd "$ROOT"

echo "[run] cache cleanup (pre)"
scripts/96_clean_repo_caches.sh >/dev/null

echo "[run] pytest"
if [[ "$FAST" -eq 1 ]]; then
  python -m pytest -q -p no:cacheprovider tests/test_script_99_portability_smoke_check.py
else
  python -m pytest -q -p no:cacheprovider
fi

echo "[run] portability smoke check"
scripts/99_portability_smoke_check.py --project-root "$ROOT"

if [[ "$ARTIFACTS" -eq 1 ]]; then
  echo "[run] publication artifact doctor"
  python scripts/95_doctor_publication_artifacts.py --project-root "$ROOT"
fi

echo "[run] cache cleanup (post)"
scripts/96_clean_repo_caches.sh >/dev/null

echo "[ok] local verify complete"
