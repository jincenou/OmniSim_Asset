#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODE=${1:?smoke or turbo}
if [[ "$MODE" == smoke ]]; then export CUDA_VISIBLE_DEVICES=4; fi
cd /mnt/ojc/workplace1/4danyone/repo
trap 'code=$?; printf "%s\n" "$code" > ../waymo/logs/generate-${MODE}.exit' EXIT
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,memory.free --format=csv,noheader
/mnt/ojc/miniconda3/envs/4danyone/bin/python ../waymo/scripts/generate.py "$MODE"
/mnt/ojc/miniconda3/envs/4danyone/bin/python ../scripts/visualize.py "../waymo/outputs/$MODE"
/mnt/ojc/miniconda3/envs/4danyone/bin/python ../waymo/scripts/validate_output.py "$MODE"
