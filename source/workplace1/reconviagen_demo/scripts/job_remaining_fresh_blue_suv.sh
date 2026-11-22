#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
set -e
python scripts/run_variant.py --case fresh_blue_suv --variant B0 --gpu 0 > logs/fresh_blue_suv_B0.log 2>&1
python scripts/run_variant.py --case fresh_blue_suv --variant F --gpu 0 > logs/fresh_blue_suv_F.log 2>&1
python scripts/run_variant.py --case fresh_blue_suv --variant H --gpu 0 > logs/fresh_blue_suv_H.log 2>&1
echo 0 > logs/fresh_blue_suv_remaining.exit
