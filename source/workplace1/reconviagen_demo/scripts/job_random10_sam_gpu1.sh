#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/segment_inputs.py --case random10_02_white_pickup > logs/random10_02_white_pickup_sam2.log 2>&1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/segment_inputs.py --case random10_10_grey_electric_suv > logs/random10_10_grey_electric_suv_sam2.log 2>&1
echo 0 > logs/random10_sam_gpu1.exit
