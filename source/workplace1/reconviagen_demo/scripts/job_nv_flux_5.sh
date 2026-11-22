#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
if [ ! -f nonvehicle10/inputs/nonvehicle_05_cylindrical_bin/flux_00.png ]; then
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/nonvehicle/generate_fresh_flux.py --case nonvehicle_05_cylindrical_bin --item-index 4 > nonvehicle10/logs/nonvehicle_05_cylindrical_bin_flux.log 2>&1
fi
env/bin/python scripts/nonvehicle/segment_flux.py --case nonvehicle_05_cylindrical_bin > nonvehicle10/logs/nonvehicle_05_cylindrical_bin_flux_mask.log 2>&1
if [ ! -f nonvehicle10/inputs/nonvehicle_06_square_bin/flux_00.png ]; then
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/nonvehicle/generate_fresh_flux.py --case nonvehicle_06_square_bin --item-index 5 > nonvehicle10/logs/nonvehicle_06_square_bin_flux.log 2>&1
fi
env/bin/python scripts/nonvehicle/segment_flux.py --case nonvehicle_06_square_bin > nonvehicle10/logs/nonvehicle_06_square_bin_flux_mask.log 2>&1
