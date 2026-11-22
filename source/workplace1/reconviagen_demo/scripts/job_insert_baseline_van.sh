#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export PATH=/mnt/ojc/workplace1/reconviagen_demo/env_splatad/bin:/mnt/ojc/miniconda3/envs/splatad/bin:$PATH
export CUDA_VISIBLE_DEVICES=1
export MPLCONFIGDIR=/mnt/ojc/workplace1/reconviagen_demo/cache/matplotlib
export TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions_splatad
export MAX_JOBS=4
export PYTHONUNBUFFERED=1
/mnt/ojc/workplace1/reconviagen_demo/env_splatad/bin/python scripts/insert_scene.py --config /mnt/ojc/splatad_repro/outputs/full_scene0012_retrain_clean/unnamed/splatad/2026-07-13_111639/config.yml --traj_npz /mnt/wmj/workspace4/pipeline/traj_pipeline/output/insert_6cam/scene-0012_V5_merge_in_traj.npz --asset_ply /mnt/ojc/workplace1/reconviagen_demo/results/service_van/B0_seed42/asset/canonical.ply --out_dir /mnt/ojc/workplace1/reconviagen_demo/results/service_van/B0_seed42/scene0012 --target_len 5.690076076823228 --start_frame 30 --max_frames 42 > logs/insert_baseline_van.log 2>&1
code=$?
printf "%s\n" "$code" > logs/insert_baseline_van.exit
exit "$code"
