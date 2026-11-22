#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions_refine
export MAX_JOBS=4
export PATH=/mnt/ojc/workplace1/reconviagen_demo/env/bin:/home/wmj/miniconda3/envs/trellis/bin:$PATH
export PYTHONUNBUFFERED=1
env/bin/python scripts/refine_case.py --case fresh_black_pickup --variant R > logs/refine_pickup_R.log 2>&1
code=$?
echo "$code" > logs/refine_pickup_R.exit
exit "$code"
