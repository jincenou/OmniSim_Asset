#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
env_refine/bin/python scripts/evaluate_pixel_metrics.py > logs/random10_pixel_metrics.log 2>&1
echo $? > logs/random10_pixel_metrics.exit
