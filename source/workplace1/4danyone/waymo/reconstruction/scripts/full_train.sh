#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
R=/mnt/ojc/workplace1/4danyone/waymo/reconstruction
B=/mnt/ojc/workplace1/4danyone/waymo/bridge
export CUDA_VISIBLE_DEVICES=7
export CUDA_HOME=/mnt/ojc/miniconda3/envs/asset_harvester_2026
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$R/deps/torch_extensions" TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
trap 'code=$?; printf "%s\n" "$code" > "$R/logs/full-train.exit"' EXIT
cd "$B/backend"
if [ -e "$R/outputs/phone_full" ]; then
  printf '%s\n' 'Completed result exists; use an explicit checkpoint and a new result directory.' >&2
  exit 1
fi
/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python -u src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe \
 --data-dir "$R/data/phone_full" --init-npz-path "$R/data/phone_full/init.npz" \
 --result-dir "$R/outputs/phone_full" --start-frame 0 --end-frame 121 \
 --data-factor 1 --test-every 8 --batch-size 1 --init-scale 1.0 \
 --alpha-mode transparent --random-bkgd --lambda-alpha 0.1 \
 --max-steps 5000 --save-steps 1000 3000 5000 --eval-steps 1000 3000 5000 --eval-sample-every 30 \
 --render-traj-n-frames 121 --render-traj-time-frames 121 --render-traj-fps 25 \
 --export-ply --export-ply-format ftgs --disable-viewer
