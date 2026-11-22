#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
R=/mnt/ojc/workplace1/4danyone/waymo/reconstruction
B=/mnt/ojc/workplace1/4danyone/waymo/bridge
export CUDA_VISIBLE_DEVICES=7
export CUDA_HOME=/mnt/ojc/miniconda3/envs/asset_harvester_2026
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$R/deps/torch_extensions" TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
trap 'code=$?; printf "%s\n" "$code" > "$R/logs/probe.exit"' EXIT
cd "$B/backend"
if [ -e "$R/outputs/probe" ]; then
  printf '%s\n' 'Probe output already exists; choose a new result directory.' >&2
  exit 1
fi
/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python -u src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe \
 --data-dir "$B/waymo_sample" --init-npz-path "$B/waymo_sample/init.npz" \
 --result-dir "$R/outputs/probe" --start-frame 0 --end-frame 3 \
 --data-factor 1 --test-every 8 --batch-size 1 --init-scale 1.0 \
 --alpha-mode transparent --random-bkgd --lambda-alpha 0.1 \
 --max-steps 1000 --save-steps 1000 --eval-steps 1000 --eval-sample-every 1 \
 --render-traj-n-frames 60 --render-traj-time-frames 60 --render-traj-fps 25 \
 --export-ply --export-ply-format ftgs --disable-viewer
