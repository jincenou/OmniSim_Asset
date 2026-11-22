#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
(python /mnt/ojc/workplace1/reconviagen_demo/scripts/download_parallel.py) > /mnt/ojc/workplace1/reconviagen_demo/logs/download_resume.log 2>&1
code=$?
printf "%s\n" "$code" > /mnt/ojc/workplace1/reconviagen_demo/logs/download_resume.exit
exit "$code"
