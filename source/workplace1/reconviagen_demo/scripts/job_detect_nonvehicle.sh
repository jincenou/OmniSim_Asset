#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/mnt/ojc/workplace1/reconviagen_demo/cache/matplotlib
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/detect_nonvehicle_candidates.py > nonvehicle10/logs/detection.log 2>&1
echo $? > nonvehicle10/logs/detection.exit
