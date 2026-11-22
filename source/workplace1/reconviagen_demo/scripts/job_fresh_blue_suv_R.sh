#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
python scripts/run_variant.py --case fresh_blue_suv --variant R --gpu 3 > logs/fresh_blue_suv_R.log 2>&1
code=$?
echo "$code" > logs/fresh_blue_suv_R.exit
exit "$code"
