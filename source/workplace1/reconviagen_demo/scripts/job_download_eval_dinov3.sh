#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 ALL_PROXY=http://127.0.0.1:7897
export RECON_DOWNLOAD_PROXY=http://127.0.0.1:7897 RECON_DOWNLOAD_HOST=huggingface.co
export HF_HUB_DISABLE_XET=1
/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python -u scripts/download_eval_dinov3.py > logs/download_eval_dinov3.log 2>&1
code=$?
echo "$code" > logs/download_eval_dinov3.exit
exit "$code"
