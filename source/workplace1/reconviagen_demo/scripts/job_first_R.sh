#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
python scripts/wait_and_run_first.py --variant R --gpu 2 > logs/first_R.log 2>&1
code=$?
printf "%s\n" "$code" > logs/first_R.exit
exit "$code"
