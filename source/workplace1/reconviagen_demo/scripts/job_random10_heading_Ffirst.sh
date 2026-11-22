#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_01_bluegrey_suv --variant F --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_01_bluegrey_suv_F_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_01_bluegrey_suv --variant F --seed 42 > logs/random10_01_bluegrey_suv_F_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_02_white_pickup --variant F --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_02_white_pickup_F_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_02_white_pickup --variant F --seed 42 > logs/random10_02_white_pickup_F_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_03_silver_minivan --variant F --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_03_silver_minivan_F_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_03_silver_minivan --variant F --seed 42 > logs/random10_03_silver_minivan_F_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_04_white_suv --variant F --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_04_white_suv_F_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_04_white_suv --variant F --seed 42 > logs/random10_04_white_suv_F_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_05_silver_crossover --variant F --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_05_silver_crossover_F_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_05_silver_crossover --variant F --seed 42 > logs/random10_05_silver_crossover_F_metric_render.log 2>&1
