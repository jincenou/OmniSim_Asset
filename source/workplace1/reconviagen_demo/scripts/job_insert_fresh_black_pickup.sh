#!/bin/bash
cd /mnt/ojc/workplace1/reconviagen_demo
export PATH=/mnt/ojc/workplace1/reconviagen_demo/env_splatad/bin:/mnt/ojc/miniconda3/envs/splatad/bin:$PATH
export CUDA_VISIBLE_DEVICES=5
export MPLCONFIGDIR=/mnt/ojc/workplace1/reconviagen_demo/cache/matplotlib
export TORCH_EXTENSIONS_DIR=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_extensions_splatad
export MAX_JOBS=4
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TORCH_HOME=/mnt/ojc/workplace1/reconviagen_demo/cache/torch_splatad
export HF_HOME=/mnt/ojc/workplace1/reconviagen_demo/cache/hf
export XDG_CACHE_HOME=/mnt/ojc/workplace1/reconviagen_demo/cache/xdg
export PYTHONUNBUFFERED=1
/mnt/ojc/workplace1/reconviagen_demo/env/bin/python scripts/finalize_heading.py --case fresh_black_pickup --variant B0 --flip > logs/heading_fresh_black_pickup_B0.log 2>&1
/mnt/ojc/workplace1/reconviagen_demo/env_splatad/bin/python scripts/insert_scene.py --config /mnt/ojc/splatad_repro/outputs/full_scene0104/unnamed/splatad/2026-07-12_151931/config.yml --traj_npz /mnt/ojc/workplace1/reconviagen_demo/inputs/scene-0104_fresh_traj.npz --asset_ply /mnt/ojc/workplace1/reconviagen_demo/results/fresh_black_pickup/R_seed42/asset/canonical.ply --out_dir /mnt/ojc/workplace1/reconviagen_demo/results/fresh_black_pickup/R_seed42/scene0104 --target_len 5.244319285120627 --start_frame 50 --max_frames 40 > logs/insert_fresh_black_pickup_R.log 2>&1
code=$?
echo "$code" > logs/insert_fresh_black_pickup_R.exit
if [ "$code" -ne 0 ]; then exit "$code"; fi
/mnt/ojc/workplace1/reconviagen_demo/env_splatad/bin/python scripts/insert_scene.py --config /mnt/ojc/splatad_repro/outputs/full_scene0104/unnamed/splatad/2026-07-12_151931/config.yml --traj_npz /mnt/ojc/workplace1/reconviagen_demo/inputs/scene-0104_fresh_traj.npz --asset_ply /mnt/ojc/workplace1/reconviagen_demo/results/fresh_black_pickup/B0_seed42/asset/canonical.ply --out_dir /mnt/ojc/workplace1/reconviagen_demo/results/fresh_black_pickup/B0_seed42/scene0104 --target_len 5.244319285120627 --start_frame 50 --max_frames 40 > logs/insert_fresh_black_pickup_B0.log 2>&1
code=$?
echo "$code" > logs/insert_fresh_black_pickup_B0.exit
if [ "$code" -ne 0 ]; then exit "$code"; fi
