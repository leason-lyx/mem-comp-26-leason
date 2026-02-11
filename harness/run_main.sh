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
RUN_NAME="test_run3"
VERBOSE=2
FILE_OWNER="zzr"
LITELLM_TRAJ_DIR="../play/trajs"
ROOTFS_DEVICE="/dev/sda"
DOCKER_IMAGE_BASE="jefzda/sweap-images"
COLLECT_ONLY=1

export RUN_NAME VERBOSE FILE_OWNER LITELLM_TRAJ_DIR ROOTFS_DEVICE DOCKER_IMAGE_BASE COLLECT_ONLY

echo "[run_main] RUN_NAME=${RUN_NAME}"
echo "[run_main] VERBOSE=${VERBOSE}"
echo "[run_main] FILE_OWNER=${FILE_OWNER}"
echo "[run_main] LITELLM_TRAJ_DIR=${LITELLM_TRAJ_DIR}"
echo "[run_main] ROOTFS_DEVICE=${ROOTFS_DEVICE}"
echo "[run_main] DOCKER_IMAGE_BASE=${DOCKER_IMAGE_BASE}"
echo "[run_main] COLLECT_ONLY=${COLLECT_ONLY}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
echo "[run_main] PYTHON_BIN=${PYTHON_BIN}"
echo "[run_main] start eval: sudo -E ${PYTHON_BIN} main.py"

sudo -E "${PYTHON_BIN}" main.py
