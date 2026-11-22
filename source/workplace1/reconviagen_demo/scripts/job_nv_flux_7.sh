#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
if [ ! -f nonvehicle10/inputs/nonvehicle_09_concrete_barrier/flux_00.png ]; then
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/nonvehicle/generate_fresh_flux.py --case nonvehicle_09_concrete_barrier --item-index 8 > nonvehicle10/logs/nonvehicle_09_concrete_barrier_flux.log 2>&1
fi
env/bin/python scripts/nonvehicle/segment_flux.py --case nonvehicle_09_concrete_barrier > nonvehicle10/logs/nonvehicle_09_concrete_barrier_flux_mask.log 2>&1
if [ ! -f nonvehicle10/inputs/nonvehicle_10_orange_cone/flux_00.png ]; then
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/nonvehicle/generate_fresh_flux.py --case nonvehicle_10_orange_cone --item-index 9 > nonvehicle10/logs/nonvehicle_10_orange_cone_flux.log 2>&1
fi
env/bin/python scripts/nonvehicle/segment_flux.py --case nonvehicle_10_orange_cone > nonvehicle10/logs/nonvehicle_10_orange_cone_flux_mask.log 2>&1
