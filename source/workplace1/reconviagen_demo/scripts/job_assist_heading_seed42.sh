#!/bin/bash
set -e
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case fresh_black_pickup --variant B0_real1 --flip > logs/fresh_black_pickup_B0_real1_heading.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case fresh_black_pickup --variant B0_real4 --flip > logs/fresh_black_pickup_B0_real4_heading.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case fresh_blue_suv --variant B0_real1 --flip > logs/fresh_blue_suv_B0_real1_heading.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case fresh_blue_suv --variant B0_real4 --flip > logs/fresh_blue_suv_B0_real4_heading.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case fresh_silver_suv --variant B0_real1 --flip > logs/fresh_silver_suv_B0_real1_heading.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case fresh_silver_suv --variant B0_real4 --flip > logs/fresh_silver_suv_B0_real4_heading.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/compare_trellis_real_assist.py --case fresh_black_pickup
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/compare_trellis_real_assist.py --case fresh_blue_suv
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/compare_trellis_real_assist.py --case fresh_silver_suv
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/measure_trellis_real_assist.py > logs/trellis_assist_metrics.log
