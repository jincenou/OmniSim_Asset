#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
set -e
python scripts/run_variant.py --case fresh_black_pickup --variant B0 --gpu 1 > logs/fresh_black_pickup_B0.log 2>&1
python scripts/run_variant.py --case fresh_black_pickup --variant F --gpu 1 > logs/fresh_black_pickup_F.log 2>&1
python scripts/run_variant.py --case fresh_black_pickup --variant H --gpu 1 > logs/fresh_black_pickup_H.log 2>&1
echo 0 > logs/fresh_black_pickup_remaining.exit
