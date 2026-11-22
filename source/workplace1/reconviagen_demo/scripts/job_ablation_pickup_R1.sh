#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
env/bin/python scripts/run_case.py --case fresh_black_pickup --variant R --real-count 1 > logs/ablation_pickup_R1.log 2>&1
code=$?
echo "$code" > logs/ablation_pickup_R1.exit
exit "$code"
