#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
curl -L --fail --connect-timeout 20 --max-time 300 'https://hf-mirror.com/spaces/JeffreyXiang/TRELLIS/resolve/main/wheels/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl' -o cache/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl.part > logs/download_raster.log 2>&1
code=$?
if [ "$code" -ne 0 ]; then echo "$code" > logs/download_raster.exit; exit "$code"; fi
mv cache/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl.part cache/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl
env/bin/pip install --no-deps --no-cache-dir cache/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl >> logs/download_raster.log 2>&1
code=$?
echo "$code" > logs/download_raster.exit
exit "$code"
