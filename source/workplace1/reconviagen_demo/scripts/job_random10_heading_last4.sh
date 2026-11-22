#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
env/bin/python scripts/finalize_heading.py --case random10_07_grey_minivan --variant B0 --flip > logs/random10_07_grey_minivan_B0_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_07_grey_minivan --variant B0 > logs/random10_07_grey_minivan_B0_metric_render.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_07_grey_minivan --variants B0 --encoder clip > logs/random10_07_grey_minivan_B0_clip.log 2>&1
env/bin/python scripts/finalize_heading.py --case random10_08_white_sedan --variant B0 --flip > logs/random10_08_white_sedan_B0_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_08_white_sedan --variant B0 > logs/random10_08_white_sedan_B0_metric_render.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_08_white_sedan --variants B0 --encoder clip > logs/random10_08_white_sedan_B0_clip.log 2>&1
env/bin/python scripts/finalize_heading.py --case random10_09_dark_sedan --variant B0 --flip > logs/random10_09_dark_sedan_B0_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_09_dark_sedan --variant B0 > logs/random10_09_dark_sedan_B0_metric_render.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_09_dark_sedan --variants B0 --encoder clip > logs/random10_09_dark_sedan_B0_clip.log 2>&1
env/bin/python scripts/finalize_heading.py --case random10_10_grey_electric_suv --variant B0 --flip > logs/random10_10_grey_electric_suv_B0_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_10_grey_electric_suv --variant B0 > logs/random10_10_grey_electric_suv_B0_metric_render.log 2>&1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python scripts/evaluate_asset_metrics.py --case random10_10_grey_electric_suv --variants B0 --encoder clip > logs/random10_10_grey_electric_suv_B0_clip.log 2>&1
