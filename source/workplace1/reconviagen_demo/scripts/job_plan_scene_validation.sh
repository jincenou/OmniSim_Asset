#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/mnt/ojc/workplace1/reconviagen_demo/cache/matplotlib OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
env_splatad/bin/python scripts/plan_scene_validation.py > scene_validation/logs/planning.log 2>&1
echo $? > scene_validation/logs/planning.exit
