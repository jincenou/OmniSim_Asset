#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/nonvehicle/segment_inputs.py --case nonvehicle_07_white_hydrant > nonvehicle10/logs/nonvehicle_07_white_hydrant_sam.log 2>&1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/nonvehicle/segment_inputs.py --case nonvehicle_08_yellow_hydrant > nonvehicle10/logs/nonvehicle_08_yellow_hydrant_sam.log 2>&1
