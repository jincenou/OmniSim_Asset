#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
set -e
env/bin/python scripts/finalize_heading.py --case fresh_blue_suv --variant R --flip
env/bin/python scripts/finalize_heading.py --case fresh_black_pickup --variant R --flip
env/bin/python scripts/finalize_heading.py --case fresh_silver_suv --variant R --flip
