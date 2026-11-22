#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
export CUDA_VISIBLE_DEVICES=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_CUDNN_V8_API_DISABLED=1 CUDNN_FRONTEND_DISABLE=1 NVIDIA_TF32_OVERRIDE=0
unset PYTORCH_CUDA_ALLOC_CONF
cd /mnt/ojc/workplace1/4danyone/repo
trap 'code=$?; printf "%s\n" "$code" > ../waymo/logs/preflight.exit' EXIT
/mnt/ojc/miniconda3/envs/4danyone/bin/python ../waymo/scripts/preflight.py
