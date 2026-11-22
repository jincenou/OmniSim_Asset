#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
set -e
python scripts/run_variant.py --case fresh_silver_suv --variant B0 --gpu 2 > logs/fresh_silver_suv_B0.log 2>&1
python scripts/run_variant.py --case fresh_silver_suv --variant F --gpu 2 > logs/fresh_silver_suv_F.log 2>&1
python scripts/run_variant.py --case fresh_silver_suv --variant H --gpu 2 > logs/fresh_silver_suv_H.log 2>&1
echo 0 > logs/fresh_silver_suv_remaining.exit
