#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/segment_inputs.py --case random10_01_bluegrey_suv > logs/random10_01_bluegrey_suv_sam2.log 2>&1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/segment_inputs.py --case random10_09_dark_sedan > logs/random10_09_dark_sedan_sam2.log 2>&1
echo 0 > logs/random10_sam_gpu0.exit
