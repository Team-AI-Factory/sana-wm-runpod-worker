#!/usr/bin/env bash
set -Eeuo pipefail

echo "RUNPOD_START"
echo "Starting SANA-WM RunPod worker..."
echo "Mode: ${SANA_WM_MODE:-missing}"
echo "Workspace: ${WORKSPACE_DIR:-/workspace}"

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
SANA_HOME="${SANA_HOME:-/workspace/Sana}"
MINICONDA_DIR="${MINICONDA_DIR:-/workspace/miniconda}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

export HF_HOME
export CONDA_PLUGINS_AUTO_ACCEPT_TOS="${CONDA_PLUGINS_AUTO_ACCEPT_TOS:-yes}"
export PYTHONUNBUFFERED=1
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-2}"

mkdir -p "$WORKSPACE_DIR" "$HF_HOME" /workspace/job-input /workspace/results

echo "STAGE_CHECK_ENV_STARTED"

required_vars=(
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_BUCKET
  R2_ENDPOINT
  JOB_KEY
  SANA_WM_MODE
)

for var in "${required_vars[@]}"; do
  if [ -z "${!var:-}" ]; then
    echo "FAILED_AT_STAGE=CHECK_ENV"
    echo "Missing environment variable: $var"
    sleep infinity
  fi
done

if [ "${SANA_WM_MODE}" != "official_bidirectional_bf16" ]; then
  echo "FAILED_AT_STAGE=CHECK_MODE"
  echo "SANA_WM_MODE must be official_bidirectional_bf16"
  echo "Current value: ${SANA_WM_MODE}"
  sleep infinity
fi

echo "STAGE_CHECK_ENV_DONE"

echo "STAGE_GPU_CHECK_STARTED"
nvidia-smi || true
echo "STAGE_GPU_CHECK_DONE"

echo "STAGE_SYSTEM_PACKAGES_STARTED"
apt-get update
apt-get install -y git wget curl ca-certificates ffmpeg build-essential libgl1 libglib2.0-0
echo "STAGE_SYSTEM_PACKAGES_DONE"

echo "STAGE_MINICONDA_STARTED"
if [ ! -x "$MINICONDA_DIR/bin/conda" ]; then
  echo "Installing Miniconda into persistent volume: $MINICONDA_DIR"
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
  rm /tmp/miniconda.sh
else
  echo "Miniconda already exists. Reusing it."
fi

export PATH="$MINICONDA_DIR/bin:$PATH"
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
echo "STAGE_MINICONDA_DONE"

echo "STAGE_SANA_REPO_STARTED"
if [ ! -d "$SANA_HOME/.git" ]; then
  echo "Cloning official NVlabs/Sana repo into persistent volume..."
  git clone https://github.com/NVlabs/Sana.git "$SANA_HOME"
else
  echo "SANA repo already exists. Pulling latest updates..."
  git -C "$SANA_HOME" pull || true
fi
echo "STAGE_SANA_REPO_DONE"

cd "$SANA_HOME"

echo "STAGE_CONDA_TERMS_STARTED"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
echo "STAGE_CONDA_TERMS_DONE"

echo "STAGE_SANA_ENV_STARTED"
if conda env list | awk '{print $1}' | grep -qx "sana"; then
  echo "SANA conda env already exists. Reusing it."
else
  echo "Installing official SANA environment. This first run may take a long time."
  bash ./environment_setup.sh sana
fi
echo "STAGE_SANA_ENV_DONE"

echo "STAGE_HELPERS_STARTED"
conda activate sana
pip install --upgrade boto3 pillow imageio imageio-ffmpeg
echo "STAGE_HELPERS_DONE"

echo "STAGE_IMPORT_CHECK_STARTED"
python - <<'PY'
import torch
import boto3
from PIL import Image
import imageio.v3 as iio

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("Helper imports: OK")
PY
echo "STAGE_IMPORT_CHECK_DONE"

echo "ENV_READY"
echo "Running SANA-WM R2 job worker..."
python /workspace/sana-wm-runpod-worker/worker.py
