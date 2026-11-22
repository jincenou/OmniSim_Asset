#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/nonvehicle/segment_inputs.py --case nonvehicle_03_stop_sign > nonvehicle10/logs/nonvehicle_03_stop_sign_sam.log 2>&1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/nonvehicle/segment_inputs.py --case nonvehicle_04_speed30_sign > nonvehicle10/logs/nonvehicle_04_speed30_sign_sam.log 2>&1
