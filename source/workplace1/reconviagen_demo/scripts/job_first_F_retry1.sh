#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
python scripts/wait_and_run_first.py --variant F --gpu 3 > logs/first_F_retry1.log 2>&1
code=$?
printf "%s\n" "$code" > logs/first_F_retry1.exit
exit "$code"
