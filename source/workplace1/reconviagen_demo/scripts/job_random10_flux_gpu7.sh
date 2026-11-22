#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/mnt/ojc/workplace1/reconviagen_demo/cache/hf TORCH_HOME=/mnt/ojc/workplace1/reconviagen_demo/cache/torch XDG_CACHE_HOME=/mnt/ojc/workplace1/reconviagen_demo/cache/xdg
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/generate_fresh_flux.py --case random10_08_white_sedan --item-index 7 --manifest inputs/random10_flux_manifest.json > logs/random10_08_white_sedan_flux.log 2>&1
env/bin/python scripts/segment_flux.py --case random10_08_white_sedan > logs/random10_08_white_sedan_flux_mask.log 2>&1
echo 0 > logs/random10_flux_gpu7.exit
