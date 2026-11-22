#!/usr/bin/env bash
set -uo pipefail
project=/mnt/ojc/workplace1/4danyone/waymo/round2/alternatives/4dgaussians
/mnt/ojc/miniconda3/envs/4dgaussians/bin/python "$project/temporal_recipe_jobs.py" > "$project/logs/temporal-recipe-controller.log" 2>&1
recipe_exit=$?
printf '%s\n' "$recipe_exit" > "$project/logs/temporal-recipe-controller.exit"
exit "$recipe_exit"
