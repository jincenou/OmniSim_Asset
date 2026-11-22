#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
env/bin/python scripts/finalize_heading.py --case random10_01_bluegrey_suv --variant R  > logs/random10_01_bluegrey_suv_R_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_01_bluegrey_suv --variant R > logs/random10_01_bluegrey_suv_R_metric_render.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_01_bluegrey_suv --variants R --encoder clip > logs/random10_01_bluegrey_suv_R_clip.log 2>&1
env/bin/python scripts/finalize_heading.py --case random10_02_white_pickup --variant R --flip > logs/random10_02_white_pickup_R_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_02_white_pickup --variant R > logs/random10_02_white_pickup_R_metric_render.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_02_white_pickup --variants R --encoder clip > logs/random10_02_white_pickup_R_clip.log 2>&1
env/bin/python scripts/finalize_heading.py --case random10_03_silver_minivan --variant R  > logs/random10_03_silver_minivan_R_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_03_silver_minivan --variant R > logs/random10_03_silver_minivan_R_metric_render.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_03_silver_minivan --variants R --encoder clip > logs/random10_03_silver_minivan_R_clip.log 2>&1
