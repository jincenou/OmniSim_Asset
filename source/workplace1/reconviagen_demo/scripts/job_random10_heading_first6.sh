#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
env/bin/python scripts/finalize_heading.py --case random10_01_bluegrey_suv --variant B0 --flip > logs/random10_01_bluegrey_suv_B0_heading.log 2>&1
CPATH=/home/wmj/miniconda3/envs/splatad/include TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions /home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_01_bluegrey_suv --variant B0 > logs/random10_01_bluegrey_suv_B0_metric_render.log 2>&1
env/bin/python scripts/finalize_heading.py --case random10_02_white_pickup --variant B0 --flip > logs/random10_02_white_pickup_B0_heading.log 2>&1
CPATH=/home/wmj/miniconda3/envs/splatad/include TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions /home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_02_white_pickup --variant B0 > logs/random10_02_white_pickup_B0_metric_render.log 2>&1
env/bin/python scripts/finalize_heading.py --case random10_03_silver_minivan --variant B0 --flip > logs/random10_03_silver_minivan_B0_heading.log 2>&1
CPATH=/home/wmj/miniconda3/envs/splatad/include TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions /home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_03_silver_minivan --variant B0 > logs/random10_03_silver_minivan_B0_metric_render.log 2>&1
env/bin/python scripts/finalize_heading.py --case random10_04_white_suv --variant B0 --flip > logs/random10_04_white_suv_B0_heading.log 2>&1
CPATH=/home/wmj/miniconda3/envs/splatad/include TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions /home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_04_white_suv --variant B0 > logs/random10_04_white_suv_B0_metric_render.log 2>&1
env/bin/python scripts/finalize_heading.py --case random10_05_silver_crossover --variant B0 --flip > logs/random10_05_silver_crossover_B0_heading.log 2>&1
CPATH=/home/wmj/miniconda3/envs/splatad/include TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions /home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_05_silver_crossover --variant B0 > logs/random10_05_silver_crossover_B0_metric_render.log 2>&1
env/bin/python scripts/finalize_heading.py --case random10_06_blue_sedan --variant B0 --flip > logs/random10_06_blue_sedan_B0_heading.log 2>&1
CPATH=/home/wmj/miniconda3/envs/splatad/include TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions /home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_06_blue_sedan --variant B0 > logs/random10_06_blue_sedan_B0_metric_render.log 2>&1
