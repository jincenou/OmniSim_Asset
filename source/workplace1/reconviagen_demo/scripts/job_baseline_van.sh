#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
(CUDA_VISIBLE_DEVICES=0 /mnt/ojc/workplace1/reconviagen_demo/env/bin/python /mnt/ojc/workplace1/reconviagen_demo/scripts/baseline.py --case service_van) > /mnt/ojc/workplace1/reconviagen_demo/logs/baseline_van.log 2>&1
code=$?
printf "%s\n" "$code" > /mnt/ojc/workplace1/reconviagen_demo/logs/baseline_van.exit
exit "$code"
