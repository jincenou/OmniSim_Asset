#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
while [ ! -f results/random10_02_white_pickup/B0_real4_seed7/asset/heading_review.json ]; do sleep 5; done
env/bin/python scripts/compose_ablation.py --case random10_02_white_pickup
env/bin/python scripts/compare_refinement.py --case random10_02_white_pickup
env/bin/python scripts/compare_trellis_real_assist.py --case random10_02_white_pickup --seed-check
echo 0 > logs/random10_extras_videos.exit
