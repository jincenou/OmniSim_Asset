#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/nonvehicle/segment_inputs.py --case nonvehicle_05_cylindrical_bin > nonvehicle10/logs/nonvehicle_05_cylindrical_bin_sam.log 2>&1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/nonvehicle/segment_inputs.py --case nonvehicle_06_square_bin > nonvehicle10/logs/nonvehicle_06_square_bin_sam.log 2>&1
