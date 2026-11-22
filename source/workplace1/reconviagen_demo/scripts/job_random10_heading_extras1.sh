#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_02_white_pickup --variant R1 --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_02_white_pickup_R1_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_02_white_pickup --variant R1 --seed 42 > logs/random10_02_white_pickup_R1_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_02_white_pickup --variant R2 --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_02_white_pickup_R2_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_02_white_pickup --variant R2 --seed 42 > logs/random10_02_white_pickup_R2_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_02_white_pickup --variant R --seed 7 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_02_white_pickup_R_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_02_white_pickup --variant R --seed 7 > logs/random10_02_white_pickup_R_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_02_white_pickup --variant B0_real1 --seed 7 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_02_white_pickup_B0_real1_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_02_white_pickup --variant B0_real1 --seed 7 > logs/random10_02_white_pickup_B0_real1_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_02_white_pickup --variant R_refine_base --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_02_white_pickup_R_refine_base_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_02_white_pickup --variant R_refine_base --seed 42 > logs/random10_02_white_pickup_R_refine_base_metric_render.log 2>&1
