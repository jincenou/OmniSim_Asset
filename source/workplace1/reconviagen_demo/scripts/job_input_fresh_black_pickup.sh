#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/mnt/ojc/workplace1/reconviagen_demo/cache/hf
export TORCH_HOME=/mnt/ojc/workplace1/reconviagen_demo/cache/torch
export XDG_CACHE_HOME=/mnt/ojc/workplace1/reconviagen_demo/cache/xdg
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/segment_inputs.py --case fresh_black_pickup
code=$?
if [ "$code" -ne 0 ]; then echo "$code" > logs/input_fresh_black_pickup.exit; exit "$code"; fi
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/generate_fresh_flux.py --case fresh_black_pickup --item-index 1
code=$?
if [ "$code" -ne 0 ]; then echo "$code" > logs/input_fresh_black_pickup.exit; exit "$code"; fi
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/segment_flux.py --case fresh_black_pickup
code=$?
echo "$code" > logs/input_fresh_black_pickup.exit
exit "$code"
