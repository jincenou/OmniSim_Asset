#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case fresh_silver_suv --variant B0_real4 --seed 7 --flip
