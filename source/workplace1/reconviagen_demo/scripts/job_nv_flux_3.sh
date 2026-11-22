#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
if [ ! -f nonvehicle10/inputs/nonvehicle_01_green_light_urban/flux_00.png ]; then
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/nonvehicle/generate_fresh_flux.py --case nonvehicle_01_green_light_urban --item-index 0 > nonvehicle10/logs/nonvehicle_01_green_light_urban_flux.log 2>&1
fi
env/bin/python scripts/nonvehicle/segment_flux.py --case nonvehicle_01_green_light_urban > nonvehicle10/logs/nonvehicle_01_green_light_urban_flux_mask.log 2>&1
if [ ! -f nonvehicle10/inputs/nonvehicle_02_red_light_suburban/flux_00.png ]; then
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/nonvehicle/generate_fresh_flux.py --case nonvehicle_02_red_light_suburban --item-index 1 > nonvehicle10/logs/nonvehicle_02_red_light_suburban_flux.log 2>&1
fi
env/bin/python scripts/nonvehicle/segment_flux.py --case nonvehicle_02_red_light_suburban > nonvehicle10/logs/nonvehicle_02_red_light_suburban_flux_mask.log 2>&1
