#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case fresh_blue_suv --variant B0 > logs/fresh_blue_suv_B0_metric_render.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case fresh_blue_suv --variant R > logs/fresh_blue_suv_R_metric_render.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case fresh_blue_suv --variant F > logs/fresh_blue_suv_F_metric_render.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case fresh_blue_suv --variant H > logs/fresh_blue_suv_H_metric_render.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case fresh_blue_suv --variant B0_real1 > logs/fresh_blue_suv_B0_real1_metric_render.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case fresh_blue_suv --variant B0_real4 > logs/fresh_blue_suv_B0_real4_metric_render.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case fresh_blue_suv --encoder clip > logs/fresh_blue_suv_all_clip.log 2>&1
