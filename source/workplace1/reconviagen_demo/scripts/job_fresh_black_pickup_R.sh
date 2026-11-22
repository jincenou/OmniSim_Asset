#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
python scripts/run_variant.py --case fresh_black_pickup --variant R --gpu 4 > logs/fresh_black_pickup_R.log 2>&1
code=$?
echo "$code" > logs/fresh_black_pickup_R.exit
exit "$code"
