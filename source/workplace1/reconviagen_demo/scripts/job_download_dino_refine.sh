#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
python - <<'INNER' > logs/download_dino_refine.log 2>&1
import sys
sys.path.insert(0,'scripts')
from download_refinement import fetch
fetch('dino_vitbase16_pretrain.pth','https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth')
INNER
code=$?
printf "%s\n" "$code" > logs/download_dino_refine.exit
exit "$code"
