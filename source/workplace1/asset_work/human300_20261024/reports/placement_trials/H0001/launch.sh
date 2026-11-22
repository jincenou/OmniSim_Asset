#!/bin/bash
set -euo pipefail
cd /mnt/ojc/workplace1/human
/mnt/ojc/miniconda3/envs/grid4d/bin/python -u /mnt/ojc/workplace1/human/assets.py placement-trial H0001 > /mnt/ojc/workplace1/asset_work/human300_20261024/reports/placement_trials/H0001/render.log 2>&1
