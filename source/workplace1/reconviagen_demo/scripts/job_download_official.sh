#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
RECON_DOWNLOAD_PROXY=http://127.0.0.1:7897 RECON_DOWNLOAD_HOST=huggingface.co python scripts/download_parallel.py > logs/download_official.log 2>&1
code=$?
printf "%s\n" "$code" > logs/download_official.exit
exit "$code"
