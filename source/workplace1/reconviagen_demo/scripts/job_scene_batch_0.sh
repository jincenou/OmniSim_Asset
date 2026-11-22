#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export CUDA_VISIBLE_DEVICES=0 PATH=/mnt/ojc/workplace1/reconviagen_demo/env_splatad/bin:/mnt/ojc/miniconda3/envs/splatad/bin:$PATH MPLCONFIGDIR=/mnt/ojc/workplace1/reconviagen_demo/cache/matplotlib TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions_splatad MAX_JOBS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_HOME=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_splatad HF_HOME=/mnt/ojc/workplace1/reconviagen_demo/cache/hf XDG_CACHE_HOME=/mnt/ojc/workplace1/reconviagen_demo/cache/xdg PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/mnt/ojc/workplace1/reconviagen_demo/env_splatad/bin/python scripts/run_scene_validation_batch.py --scene scene-0103 > scene_validation/logs/batch_scene-0103.log 2>&1
