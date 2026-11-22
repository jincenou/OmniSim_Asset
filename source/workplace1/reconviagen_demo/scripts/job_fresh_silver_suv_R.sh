#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
python scripts/run_variant.py --case fresh_silver_suv --variant R --gpu 5 > logs/fresh_silver_suv_R.log 2>&1
code=$?
echo "$code" > logs/fresh_silver_suv_R.exit
exit "$code"
