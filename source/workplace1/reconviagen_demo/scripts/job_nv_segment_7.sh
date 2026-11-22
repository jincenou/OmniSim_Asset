#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/nonvehicle/segment_inputs.py --case nonvehicle_09_concrete_barrier > nonvehicle10/logs/nonvehicle_09_concrete_barrier_sam.log 2>&1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/nonvehicle/segment_inputs.py --case nonvehicle_10_orange_cone > nonvehicle10/logs/nonvehicle_10_orange_cone_sam.log 2>&1
