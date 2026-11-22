#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
if [ ! -f nonvehicle10/inputs/nonvehicle_07_white_hydrant/flux_00.png ]; then
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/nonvehicle/generate_fresh_flux.py --case nonvehicle_07_white_hydrant --item-index 6 > nonvehicle10/logs/nonvehicle_07_white_hydrant_flux.log 2>&1
fi
env/bin/python scripts/nonvehicle/segment_flux.py --case nonvehicle_07_white_hydrant > nonvehicle10/logs/nonvehicle_07_white_hydrant_flux_mask.log 2>&1
if [ ! -f nonvehicle10/inputs/nonvehicle_08_yellow_hydrant/flux_00.png ]; then
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/nonvehicle/generate_fresh_flux.py --case nonvehicle_08_yellow_hydrant --item-index 7 > nonvehicle10/logs/nonvehicle_08_yellow_hydrant_flux.log 2>&1
fi
env/bin/python scripts/nonvehicle/segment_flux.py --case nonvehicle_08_yellow_hydrant > nonvehicle10/logs/nonvehicle_08_yellow_hydrant_flux_mask.log 2>&1
