#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
python scripts/wait_and_run_first.py --variant H --gpu 4 > logs/first_H.log 2>&1
code=$?
printf "%s\n" "$code" > logs/first_H.exit
exit "$code"
