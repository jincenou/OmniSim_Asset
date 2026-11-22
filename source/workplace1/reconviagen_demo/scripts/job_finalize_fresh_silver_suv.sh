#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
set -e
env/bin/python scripts/finalize_heading.py --case fresh_silver_suv --variant F --flip
env/bin/python scripts/finalize_heading.py --case fresh_silver_suv --variant H --flip
env/bin/python scripts/compare_case.py --case fresh_silver_suv
