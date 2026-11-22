#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
(CUDA_VISIBLE_DEVICES=2 /mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python /mnt/ojc/workplace1/reconviagen_demo/scripts/segment_inputs.py --case white_sedan && /mnt/ojc/workplace1/reconviagen_demo/env/bin/python /mnt/ojc/workplace1/reconviagen_demo/scripts/segment_flux.py --case white_sedan) > /mnt/ojc/workplace1/reconviagen_demo/logs/segment_white.log 2>&1
code=$?
printf "%s\n" "$code" > /mnt/ojc/workplace1/reconviagen_demo/logs/segment_white.exit
exit "$code"
