#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
while [ ! -f results/random10_02_white_pickup/R1_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_02_white_pickup/R2_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_02_white_pickup/B0_repeat_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_02_white_pickup/R_refine_base_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_02_white_pickup/R_refined_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_02_white_pickup/F_refine_base_seed42/evaluation/render_audit.json ]; do sleep 5; done
while [ ! -f results/random10_02_white_pickup/F_refined_seed42/evaluation/render_audit.json ]; do sleep 5; done
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_02_white_pickup --seed 42 --variants R1 R2 B0_repeat R_refine_base R_refined F_refine_base F_refined --encoder clip > logs/random10_extras_seed42_clip.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_02_white_pickup --seed 42 --variants R1 R2 B0_repeat R_refine_base R_refined F_refine_base F_refined --encoder dino > logs/random10_extras_seed42_dino.log 2>&1
echo 0 > logs/random10_extras_seed42_metrics.exit
