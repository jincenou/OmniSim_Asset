#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
if [ ! -f nonvehicle10/inputs/nonvehicle_03_stop_sign/flux_00.png ]; then
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/nonvehicle/generate_fresh_flux.py --case nonvehicle_03_stop_sign --item-index 2 > nonvehicle10/logs/nonvehicle_03_stop_sign_flux.log 2>&1
fi
env/bin/python scripts/nonvehicle/segment_flux.py --case nonvehicle_03_stop_sign > nonvehicle10/logs/nonvehicle_03_stop_sign_flux_mask.log 2>&1
if [ ! -f nonvehicle10/inputs/nonvehicle_04_speed30_sign/flux_00.png ]; then
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/nonvehicle/generate_fresh_flux.py --case nonvehicle_04_speed30_sign --item-index 3 > nonvehicle10/logs/nonvehicle_04_speed30_sign_flux.log 2>&1
fi
env/bin/python scripts/nonvehicle/segment_flux.py --case nonvehicle_04_speed30_sign > nonvehicle10/logs/nonvehicle_04_speed30_sign_flux_mask.log 2>&1
