#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/98_remote_heavy_run.sh --remote-dir <path> (--target <name> | --cmd <command>) [options]

Targets:
  pipeline_dry_run_07_15  Run scripts/13_run_pipeline.py --all --from-stage 07 --to-stage 15 --sem-dry-run.
  bootstrap_inference     Run scripts/20_run_inference_bootstrap.py family_bootstrap preset.
  publication_lock        Build scripts/93_build_results_snapshot.py then scripts/24_build_publication_results_lock.py then scripts/25_verify_publication_snapshot.py.
  spec_curve_summary      Run scripts/15_specification_curve_summary.py --all.
  repro_report            Run scripts/14_reproducibility_report.py.
  stage07_15_dry_run      Backward-compatible alias of pipeline_dry_run_07_15.

Safety:
  Dry-run by default. Use --confirm (or --run alias) to execute over SSH.

Defaults:
  --host shanewray@100.71.19.72
  --tag  remote_run
  --remote-venv '$HOME/venvs/sexg/bin/activate'
  --engine sem_refit
  --n-bootstrap 100
  --min-success-share 0.90
  --sem-timeout-seconds 60
  --seed 20260221

Ergonomics:
  --list-targets           Print available targets and exit 0.
  --print-log-path         Print the computed local log path and exit 0 (no log write).
  --dry-run                 Explicit dry-run alias (default behavior).
  --remote-python <path>    Force interpreter token replacement ("python " prefix or "$REMOTE_PYTHON" token).
  --env KEY=VALUE           Repeatable remote env exports added before execution.
  --sync-code               Rsync scripts/src/config to remote before running (run mode only).
  --sync-back               Rsync key remote artifacts back to local outputs/ (run mode only).
  --sync-back-bootstrap-dirs  Also rsync outputs/model_fits/bootstrap_inference/<cohort>/ dirs (large; run mode only).
  --self-test               Print computed remote script and exit 0 (no log write, no SSH).

Examples:
  scripts/98_remote_heavy_run.sh --remote-dir ~/proj/sexg --target pipeline_dry_run_07_15
  scripts/98_remote_heavy_run.sh --remote-dir ~/proj/sexg --target bootstrap_inference --confirm
  scripts/98_remote_heavy_run.sh --remote-dir ~/proj/sexg --cmd "python scripts/11_robustness_suite.py --project-root ~/proj/sexg" --confirm
EOF
}

quote_arg() {
  printf '%q' "$1"
}

quote_single() {
  printf "'%s'" "$(printf "%s" "$1" | sed "s/'/'\\\\''/g")"
}

HOST="shanewray@100.71.19.72"
REMOTE_DIR=""
REMOTE_CMD=""
TARGET=""
TAG="remote_run"
DO_RUN=0
SELF_TEST=0
LIST_TARGETS=0
PRINT_LOG_PATH=0
REMOTE_VENV='$HOME/venvs/sexg/bin/activate'
REMOTE_PYTHON_OVERRIDE=""
ENGINE="sem_refit"
N_BOOTSTRAP="100"
MIN_SUCCESS_SHARE="0.90"
SEM_TIMEOUT_SECONDS="60"
SEED="20260221"
SKIP_SUCCESSFUL="1"
SYNC_CODE="0"
SYNC_BACK="0"
SYNC_BACK_BOOTSTRAP_DIRS="0"
COHORTS=()
ENV_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list-targets)
      LIST_TARGETS=1; shift 1;;
    --print-log-path)
      PRINT_LOG_PATH=1; shift 1;;
    --host)
      HOST="$2"; shift 2;;
    --remote-dir)
      REMOTE_DIR="$2"; shift 2;;
    --cmd)
      REMOTE_CMD="$2"; shift 2;;
    --target)
      TARGET="$2"; shift 2;;
    --tag)
      TAG="$2"; shift 2;;
    --remote-venv)
      REMOTE_VENV="$2"; shift 2;;
    --remote-python)
      REMOTE_PYTHON_OVERRIDE="$2"; shift 2;;
    --engine)
      ENGINE="$2"; shift 2;;
    --n-bootstrap)
      N_BOOTSTRAP="$2"; shift 2;;
    --min-success-share)
      MIN_SUCCESS_SHARE="$2"; shift 2;;
    --sem-timeout-seconds)
      SEM_TIMEOUT_SECONDS="$2"; shift 2;;
    --seed)
      SEED="$2"; shift 2;;
    --skip-successful)
      SKIP_SUCCESSFUL="1"; shift 1;;
    --no-skip-successful)
      SKIP_SUCCESSFUL="0"; shift 1;;
    --sync-code)
      SYNC_CODE="1"; shift 1;;
    --sync-back)
      SYNC_BACK="1"; shift 1;;
    --sync-back-bootstrap-dirs)
      SYNC_BACK_BOOTSTRAP_DIRS="1"; shift 1;;
    --cohort)
      COHORTS+=("$2"); shift 2;;
    --env)
      ENV_OVERRIDES+=("$2"); shift 2;;
    --dry-run)
      DO_RUN=0; shift 1;;
    --self-test)
      SELF_TEST=1; DO_RUN=0; shift 1;;
    --confirm|--run)
      DO_RUN=1; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "[error] unknown arg: $1" >&2
      usage
      exit 2;;
  esac
done

if [[ "$LIST_TARGETS" -eq 1 ]]; then
  cat <<'EOF'
pipeline_dry_run_07_15  Run scripts/13_run_pipeline.py --all --from-stage 07 --to-stage 15 --sem-dry-run.
bootstrap_inference     Run scripts/20_run_inference_bootstrap.py family_bootstrap preset.
publication_lock        Build scripts/93_build_results_snapshot.py then scripts/24_build_publication_results_lock.py then scripts/25_verify_publication_snapshot.py.
spec_curve_summary      Run scripts/15_specification_curve_summary.py --all.
repro_report            Run scripts/14_reproducibility_report.py.
stage07_15_dry_run      Backward-compatible alias of pipeline_dry_run_07_15.
EOF
  exit 0
fi

if [[ -z "$REMOTE_DIR" ]]; then
  echo "[error] --remote-dir is required" >&2
  usage
  exit 2
fi
if [[ -n "$TARGET" && -n "$REMOTE_CMD" ]]; then
  echo "[error] pass either --target or --cmd, not both" >&2
  usage
  exit 2
fi
if [[ -z "$TARGET" && -z "$REMOTE_CMD" ]]; then
  echo "[error] one of --target or --cmd is required" >&2
  usage
  exit 2
fi
if [[ "$SYNC_BACK_BOOTSTRAP_DIRS" -eq 1 && "$SYNC_BACK" -ne 1 ]]; then
  echo "[error] --sync-back-bootstrap-dirs requires --sync-back" >&2
  exit 2
fi

if [[ -n "$TARGET" ]]; then
  case "$TARGET" in
    pipeline_dry_run_07_15|stage07_15_dry_run)
      REMOTE_CMD="\$REMOTE_PYTHON scripts/13_run_pipeline.py --all --from-stage 07 --to-stage 15 --sem-dry-run"
      ;;
    bootstrap_inference)
      REMOTE_CMD="\$REMOTE_PYTHON scripts/20_run_inference_bootstrap.py --variant-token family_bootstrap --engine $(quote_arg "$ENGINE") --n-bootstrap $(quote_arg "$N_BOOTSTRAP") --sem-jobs 16 --sem-threads-per-job 1 --cohort nlsy79 --cohort nlsy97 --min-success-share $(quote_arg "$MIN_SUCCESS_SHARE") --sem-timeout-seconds $(quote_arg "$SEM_TIMEOUT_SECONDS") --seed $(quote_arg "$SEED")"
      if [[ "$SKIP_SUCCESSFUL" -eq 1 ]]; then
        REMOTE_CMD="$REMOTE_CMD --skip-successful"
      fi
      if [[ "${#COHORTS[@]}" -gt 0 ]]; then
        REMOTE_CMD="\$REMOTE_PYTHON scripts/20_run_inference_bootstrap.py --variant-token family_bootstrap --engine $(quote_arg "$ENGINE") --n-bootstrap $(quote_arg "$N_BOOTSTRAP") --sem-jobs 16 --sem-threads-per-job 1 --min-success-share $(quote_arg "$MIN_SUCCESS_SHARE") --sem-timeout-seconds $(quote_arg "$SEM_TIMEOUT_SECONDS") --seed $(quote_arg "$SEED")"
        if [[ "$SKIP_SUCCESSFUL" -eq 1 ]]; then
          REMOTE_CMD="$REMOTE_CMD --skip-successful"
        fi
        for cohort in "${COHORTS[@]}"; do
          REMOTE_CMD="$REMOTE_CMD --cohort $(quote_arg "$cohort")"
        done
      fi
      ;;
    publication_lock)
      REMOTE_CMD="\$REMOTE_PYTHON scripts/09_results_and_figures.py --all && \$REMOTE_PYTHON scripts/93_build_results_snapshot.py --include-run-summary && \$REMOTE_PYTHON scripts/24_build_publication_results_lock.py && \$REMOTE_PYTHON scripts/25_verify_publication_snapshot.py"
      ;;
    spec_curve_summary)
      REMOTE_CMD="\$REMOTE_PYTHON scripts/15_specification_curve_summary.py --all"
      ;;
    repro_report)
      REMOTE_CMD="\$REMOTE_PYTHON scripts/14_reproducibility_report.py"
      ;;
    *)
      echo "[error] unknown --target: $TARGET" >&2
      echo "[error] allowed targets: pipeline_dry_run_07_15, bootstrap_inference, publication_lock, spec_curve_summary, repro_report, stage07_15_dry_run" >&2
      exit 2
      ;;
  esac
fi

if [[ -z "$REMOTE_CMD" ]]; then
  echo "[error] computed remote command is empty" >&2
  exit 2
fi
if [[ "$REMOTE_CMD" == *$'\n'* ]]; then
  echo "[error] remote command must be a single line" >&2
  exit 2
fi
if [[ -n "$REMOTE_PYTHON_OVERRIDE" ]]; then
  REMOTE_PYTHON_QUOTED="$(quote_arg "$REMOTE_PYTHON_OVERRIDE")"
  if [[ "$REMOTE_CMD" == *"\$REMOTE_PYTHON"* ]]; then
    REMOTE_CMD="${REMOTE_CMD//\$REMOTE_PYTHON/$REMOTE_PYTHON_QUOTED}"
  elif [[ "$REMOTE_CMD" == python\ * ]]; then
    REMOTE_CMD="$REMOTE_PYTHON_QUOTED ${REMOTE_CMD#python }"
  fi
fi

REMOTE_EXTRA_EXPORTS=""
if [[ "${#ENV_OVERRIDES[@]}" -gt 0 ]]; then
  for raw_env in "${ENV_OVERRIDES[@]}"; do
    if [[ "$raw_env" != *=* ]]; then
      echo "[error] --env must be KEY=VALUE (got: $raw_env)" >&2
      exit 2
    fi
    env_key="${raw_env%%=*}"
    env_val="${raw_env#*=}"
    if [[ ! "$env_key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "[error] invalid env key in --env: $env_key" >&2
      exit 2
    fi
    if [[ "$env_val" == *$'\n'* ]]; then
      echo "[error] env value for $env_key contains newline" >&2
      exit 2
    fi
    REMOTE_EXTRA_EXPORTS="${REMOTE_EXTRA_EXPORTS}export $env_key=$(quote_arg "$env_val")
"
  done
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date -u '+%Y%m%dT%H%M%SZ')"
LOG_DIR="$ROOT/outputs/logs/remote_runs"
LOG_PATH="$LOG_DIR/${TS}_${TAG}.log"

RSYNC_COMMANDS=""
if [[ "$SYNC_CODE" -eq 1 ]]; then
  REMOTE_BASE="${HOST}:$(quote_single "$REMOTE_DIR")/"
  RSYNC_COMMANDS="$(cat <<EOF
rsync -az $(quote_arg "$ROOT/scripts/") "${REMOTE_BASE}scripts/"
rsync -az $(quote_arg "$ROOT/src/") "${REMOTE_BASE}src/"
rsync -az $(quote_arg "$ROOT/config/") "${REMOTE_BASE}config/"
rsync -az $(quote_arg "$ROOT/requirements.txt") "${REMOTE_BASE}requirements.txt"
rsync -az $(quote_arg "$ROOT/pyproject.toml") "${REMOTE_BASE}pyproject.toml"
rsync -az $(quote_arg "$ROOT/pytest.ini") "${REMOTE_BASE}pytest.ini"
EOF
)"
fi

SYNC_BACK_COMMANDS=""
if [[ "$SYNC_BACK" -eq 1 ]]; then
  REMOTE_BASE="${HOST}:$(quote_single "$REMOTE_DIR")/"
	  SYNC_ITEMS=(
	    "outputs/tables/g_mean_diff_family_bootstrap.csv"
	    "outputs/tables/g_variance_ratio_family_bootstrap.csv"
	    "outputs/tables/inference_rerun_manifest_family_bootstrap.json"
	    "outputs/tables/g_proxy_mean_diff_family_bootstrap.csv"
	    "outputs/tables/g_proxy_variance_ratio_family_bootstrap.csv"
	    "outputs/tables/inference_rerun_manifest_g_proxy_family_bootstrap.json"
	    "outputs/tables/results_snapshot.md"
	    "outputs/tables/publication_snapshot_manifest.csv"
	    "outputs/tables/publication_results_lock.zip"
	    "outputs/tables/publication_results_lock/publication_results_lock_manifest.csv"
	    "outputs/tables/publication_results_lock/manuscript_results_lock.md"
	  )
  for rel_path in "${SYNC_ITEMS[@]}"; do
    local_target="$ROOT/$rel_path"
    local_parent="$(dirname "$local_target")"
    remote_source="${REMOTE_BASE}$rel_path"
    SYNC_BACK_COMMANDS="${SYNC_BACK_COMMANDS}mkdir -p $(quote_arg "$local_parent")
if rsync -az $(quote_arg "$remote_source") $(quote_arg "$local_target"); then
  :
else
  echo \"[sync-back][warn] skipped: $rel_path\" >&2
fi
"
  done

  if [[ "$SYNC_BACK_BOOTSTRAP_DIRS" -eq 1 ]]; then
    SYNC_COHORTS=()
    if [[ "${#COHORTS[@]}" -gt 0 ]]; then
      SYNC_COHORTS=("${COHORTS[@]}")
    elif [[ "$TARGET" == "bootstrap_inference" ]]; then
      SYNC_COHORTS=("nlsy79" "nlsy97")
    fi
    for cohort in "${SYNC_COHORTS[@]}"; do
      local_dir="$ROOT/outputs/model_fits/bootstrap_inference/$cohort/"
      remote_dir="${REMOTE_BASE}outputs/model_fits/bootstrap_inference/$cohort/"
      SYNC_BACK_COMMANDS="${SYNC_BACK_COMMANDS}mkdir -p $(quote_arg "$local_dir")
if rsync -az $(quote_arg "$remote_dir") $(quote_arg "$local_dir"); then
  :
else
  echo \"[sync-back][warn] skipped: outputs/model_fits/bootstrap_inference/$cohort/\" >&2
fi
"
    done
  fi
fi

REMOTE_SCRIPT=$(
  cat <<EOF
set -euo pipefail
cd $(quote_single "$REMOTE_DIR")
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
$REMOTE_EXTRA_EXPORTS
REMOTE_VENV_PATH="$REMOTE_VENV"
if [[ -n "\$REMOTE_VENV_PATH" ]] && [[ -f "\$REMOTE_VENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "\$REMOTE_VENV_PATH"
fi
if command -v python >/dev/null 2>&1; then
  REMOTE_PYTHON="python"
elif command -v python3 >/dev/null 2>&1; then
  REMOTE_PYTHON="python3"
else
  echo "[remote][error] python interpreter not found" >&2
  exit 127
fi
echo "[remote] host=\$(hostname) pwd=\$(pwd) python=\$REMOTE_PYTHON"
echo "[remote] cmd=$REMOTE_CMD"
$REMOTE_CMD
EOF
)

RUN_LABEL="${TARGET:-custom_cmd}"
MODE="dry-run"
if [[ "$DO_RUN" -eq 1 ]]; then
  MODE="run"
fi
if [[ "$SELF_TEST" -eq 1 ]]; then
  MODE="self-test"
fi

if [[ "$SELF_TEST" -eq 1 ]]; then
  printf '%s\n' "$REMOTE_SCRIPT"
  echo "[summary] mode=$MODE host=$HOST target=$RUN_LABEL remote_dir=$REMOTE_DIR log=<none>"
  exit 0
fi

if [[ "$PRINT_LOG_PATH" -eq 1 ]]; then
  echo "$LOG_PATH"
  exit 0
fi

mkdir -p "$LOG_DIR"
{
  echo "[local] timestamp_utc=$TS"
  echo "[local] host=$HOST"
  echo "[local] remote_dir=$REMOTE_DIR"
  echo "[local] target=$RUN_LABEL"
  echo "[local] mode=$MODE"
  echo "[local] tag=$TAG"
  echo "[local] remote_venv=$REMOTE_VENV"
  echo "[local] sync_code=$SYNC_CODE"
  echo "[local] sync_back=$SYNC_BACK"
  echo "[local] sync_back_bootstrap_dirs=$SYNC_BACK_BOOTSTRAP_DIRS"
  echo "[local] log_path=$LOG_PATH"
  if [[ "$SYNC_CODE" -eq 1 ]]; then
    echo
    echo "[local] planned rsync commands:"
    printf '%s\n' "$RSYNC_COMMANDS"
  fi
  if [[ "$SYNC_BACK" -eq 1 ]]; then
    echo
    echo "[local] planned sync-back commands:"
    printf '%s\n' "$SYNC_BACK_COMMANDS"
  fi
  echo
  echo "[local] remote script:"
  echo "----------------------------------------"
  printf '%s\n' "$REMOTE_SCRIPT"
  echo "----------------------------------------"
} >"$LOG_PATH"

if [[ "$DO_RUN" -eq 0 ]]; then
  echo "[dry-run] wrote: $LOG_PATH"
  echo "[summary] mode=$MODE host=$HOST target=$RUN_LABEL remote_dir=$REMOTE_DIR log=$LOG_PATH"
  exit 0
fi

echo "[run] streaming output to: $LOG_PATH"
if [[ "$SYNC_CODE" -eq 1 ]]; then
  if ! command -v rsync >/dev/null 2>&1; then
    echo "[error] rsync is required for --sync-code" >&2
    exit 127
  fi
  echo "[run] syncing code to remote via rsync..." | tee -a "$LOG_PATH"
  bash -lc "$RSYNC_COMMANDS" >>"$LOG_PATH" 2>&1
  echo "[run] rsync complete" | tee -a "$LOG_PATH"
fi
ssh "$HOST" "bash -lc $(printf '%q' "$REMOTE_SCRIPT")" 2>&1 | tee -a "$LOG_PATH"
if [[ "$SYNC_BACK" -eq 1 ]]; then
  if ! command -v rsync >/dev/null 2>&1; then
    echo "[error] rsync is required for --sync-back" >&2
    exit 127
  fi
  echo "[run] syncing selected artifacts back to local..." | tee -a "$LOG_PATH"
  bash -lc "$SYNC_BACK_COMMANDS" >>"$LOG_PATH" 2>&1
  echo "[run] sync-back complete" | tee -a "$LOG_PATH"
fi
echo "[ok] completed; log: $LOG_PATH"
echo "[summary] mode=$MODE host=$HOST target=$RUN_LABEL remote_dir=$REMOTE_DIR log=$LOG_PATH"
