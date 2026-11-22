#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_04_white_suv --variant R --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_04_white_suv_R_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_04_white_suv --variant R --seed 42 > logs/random10_04_white_suv_R_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_05_silver_crossover --variant R --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_05_silver_crossover_R_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_05_silver_crossover --variant R --seed 42 > logs/random10_05_silver_crossover_R_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_06_blue_sedan --variant R --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_06_blue_sedan_R_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_06_blue_sedan --variant R --seed 42 > logs/random10_06_blue_sedan_R_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_07_grey_minivan --variant R --seed 42 --confidence 1.0 --note 'Visual front/rear light, grille and cabin check' --flip > logs/random10_07_grey_minivan_R_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_07_grey_minivan --variant R --seed 42 > logs/random10_07_grey_minivan_R_metric_render.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case random10_08_white_sedan --variant R --seed 42 --confidence 0.5 --note 'Ambiguous frontal appearance: rear-like texture on both ends. Heading assigned from hood/cabin geometry, not metric optimization; inspect qualitative result.' --flip > logs/random10_08_white_sedan_R_heading.log 2>&1
/home/wmj/miniconda3/envs/sim/bin/python scripts/render_metric_views.py --case random10_08_white_sedan --variant R --seed 42 > logs/random10_08_white_sedan_R_metric_render.log 2>&1
