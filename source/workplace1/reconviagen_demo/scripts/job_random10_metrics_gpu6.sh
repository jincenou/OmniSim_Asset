#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
while [ ! -f results/random10_04_white_suv/B0_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_04_white_suv/R_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_04_white_suv/F_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_04_white_suv/H_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_04_white_suv/B0_real1_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_04_white_suv/B0_real4_seed42/evaluation/render_audit.json ]; do sleep 5; done
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_04_white_suv --encoder clip > logs/random10_04_white_suv_all_clip.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_04_white_suv --encoder dino > logs/random10_04_white_suv_all_dino.log 2>&1
env/bin/python scripts/compare_random10_case.py --case random10_04_white_suv > logs/random10_04_white_suv_comparison.log 2>&1
while [ ! -f results/random10_09_dark_sedan/B0_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_09_dark_sedan/R_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_09_dark_sedan/F_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_09_dark_sedan/H_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_09_dark_sedan/B0_real1_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_09_dark_sedan/B0_real4_seed42/evaluation/render_audit.json ]; do sleep 5; done
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_09_dark_sedan --encoder clip > logs/random10_09_dark_sedan_all_clip.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_09_dark_sedan --encoder dino > logs/random10_09_dark_sedan_all_dino.log 2>&1
env/bin/python scripts/compare_random10_case.py --case random10_09_dark_sedan > logs/random10_09_dark_sedan_comparison.log 2>&1
echo 0 > logs/random10_metrics_gpu6.exit
