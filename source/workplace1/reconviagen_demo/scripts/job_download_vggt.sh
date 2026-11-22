#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
python scripts/download_parallel.py > logs/download_vggt.log 2>&1
code=$?
printf "%s\n" "$code" > logs/download_vggt.exit
exit "$code"
