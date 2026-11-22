#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_06_blue_sedan --variant B0_real1 --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_06_blue_sedan_B0_real1_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_06_blue_sedan --variant B0_real1 --seed 42 > logs/random10_06_blue_sedan_B0_real1_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_07_grey_minivan --variant B0_real1 --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_07_grey_minivan_B0_real1_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_07_grey_minivan --variant B0_real1 --seed 42 > logs/random10_07_grey_minivan_B0_real1_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_08_white_sedan --variant B0_real1 --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' > logs/random10_08_white_sedan_B0_real1_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_08_white_sedan --variant B0_real1 --seed 42 > logs/random10_08_white_sedan_B0_real1_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_09_dark_sedan --variant B0_real1 --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_09_dark_sedan_B0_real1_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_09_dark_sedan --variant B0_real1 --seed 42 > logs/random10_09_dark_sedan_B0_real1_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_10_grey_electric_suv --variant B0_real1 --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_10_grey_electric_suv_B0_real1_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_10_grey_electric_suv --variant B0_real1 --seed 42 > logs/random10_10_grey_electric_suv_B0_real1_metric_render.log 2>&1
