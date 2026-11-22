#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
set -e
env/bin/python scripts/finalize_heading.py --case fresh_black_pickup --variant R1 --seed 42 
env/bin/python scripts/finalize_heading.py --case fresh_black_pickup --variant R2 --seed 42 
env/bin/python scripts/finalize_heading.py --case fresh_black_pickup --variant R --seed 7 --flip
