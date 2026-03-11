#!/usr/bin/env bash
set -euo pipefail

# Always run from the harness directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Activate the conda environment for memcomp-agent in non-interactive shell.
if [[ "${CONDA_DEFAULT_ENV:-}" != "memcomp-agent" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "[run_main] error: conda command not found" >&2
        exit 1
    fi

    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate memcomp-agent
fi

# ===== Config: edit these values before running =====
RUN_NAME="test-dsv32"
CANDIDATES_FILE="candidates.json"
PROJECTS_FILE="projects_full.json"
COLLECT_ONLY="${COLLECT_ONLY:-0}"
VERBOSE=2
FILE_OWNER="zzr"
LITELLM_TRAJ_DIR="../play/trajs"
ROOTFS_DEVICE="/dev/sda"
DOCKER_IMAGE_BASE="jefzda/sweap-images"
EVAL_DOCKERHUB_USERNAME="jefzda"
EVAL_NUM_WORKERS=16
EVAL_DOCKER_API_TIMEOUT=600
DOCKER_API_TIMEOUT="${DOCKER_API_TIMEOUT:-${EVAL_DOCKER_API_TIMEOUT}}"
PYTHONUNBUFFERED=1

usage() {
    cat <<'EOF'
Usage: run_main.sh [--run-name <name>] [--candidates-file <file>]

Options:
  --run-name <name>         Override RUN_NAME (default: value defined in script)
  --candidates-file <file>  Override CANDIDATES_FILE (default: value defined in script)
  -h, --help                Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)
            [[ $# -ge 2 ]] || { echo "[run_main] error: --run-name requires a value" >&2; exit 2; }
            RUN_NAME="$2"
            shift 2
            ;;
        --candidates-file)
            [[ $# -ge 2 ]] || { echo "[run_main] error: --candidates-file requires a value" >&2; exit 2; }
            CANDIDATES_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[run_main] error: unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

export RUN_NAME VERBOSE FILE_OWNER LITELLM_TRAJ_DIR ROOTFS_DEVICE DOCKER_IMAGE_BASE COLLECT_ONLY PYTHONUNBUFFERED EVAL_DOCKERHUB_USERNAME EVAL_NUM_WORKERS EVAL_DOCKER_API_TIMEOUT DOCKER_API_TIMEOUT

LOG_DEST="results/${RUN_NAME}/run_main.log"
if [[ "${COLLECT_ONLY}" == "0" ]]; then
    mkdir -p "results/${RUN_NAME}"
    : > "${LOG_DEST}"
    exec > >(stdbuf -oL -eL tee -a "${LOG_DEST}") 2>&1
fi

echo "[run_main] RUN_NAME=${RUN_NAME}"
echo "[run_main] COLLECT_ONLY=${COLLECT_ONLY}"
echo "[run_main] PROJECTS_FILE=${PROJECTS_FILE}"
echo "[run_main] CANDIDATES_FILE=${CANDIDATES_FILE}"
echo "[run_main] VERBOSE=${VERBOSE}"
if [[ "${COLLECT_ONLY}" == "0" ]]; then
    echo "[run_main] LOG_DEST=${LOG_DEST}"
fi
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
echo "[run_main] PYTHON_BIN=${PYTHON_BIN}"
echo "[run_main] DOCKER_API_TIMEOUT=${DOCKER_API_TIMEOUT}"
echo "[run_main] start harness: sudo -E ${PYTHON_BIN} main.py --projects-file ${PROJECTS_FILE} --candidates-file ${CANDIDATES_FILE}"

sudo -E "${PYTHON_BIN}" main.py \
    --projects-file "${PROJECTS_FILE}" \
    --candidates-file "${CANDIDATES_FILE}"

PATCH_DIR="results/${RUN_NAME}"
echo "[run_main] start swe_bench_pro_eval.py: ${PYTHON_BIN} swe_bench_pro_eval.py --raw_sample_path swebench-pro.parquet --patch_dir ${PATCH_DIR} --dockerhub_username ${EVAL_DOCKERHUB_USERNAME} --scripts_dir run_scripts --num_workers ${EVAL_NUM_WORKERS} --docker_api_timeout ${EVAL_DOCKER_API_TIMEOUT}"
"${PYTHON_BIN}" swe_bench_pro_eval.py \
    --raw_sample_path swebench-pro.parquet \
    --patch_dir "${PATCH_DIR}" \
    --dockerhub_username "${EVAL_DOCKERHUB_USERNAME}" \
    --scripts_dir run_scripts \
    --num_workers "${EVAL_NUM_WORKERS}" \
    --docker_api_timeout "${EVAL_DOCKER_API_TIMEOUT}" \
    --redo
