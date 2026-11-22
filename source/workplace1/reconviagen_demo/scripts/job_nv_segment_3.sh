#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/nonvehicle/segment_inputs.py --case nonvehicle_01_green_light_urban > nonvehicle10/logs/nonvehicle_01_green_light_urban_sam.log 2>&1
/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python scripts/nonvehicle/segment_inputs.py --case nonvehicle_02_red_light_suburban > nonvehicle10/logs/nonvehicle_02_red_light_suburban_sam.log 2>&1
