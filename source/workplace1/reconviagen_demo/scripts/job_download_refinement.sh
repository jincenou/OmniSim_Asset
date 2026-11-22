#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
python scripts/download_refinement.py > logs/download_refinement.log 2>&1
code=$?
printf "%s\n" "$code" > logs/download_refinement.exit
exit "$code"
