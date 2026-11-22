#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
R=/mnt/ojc/workplace1/4danyone/waymo/reconstruction
P=/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python
export CUDA_HOME=/mnt/ojc/miniconda3/envs/asset_harvester_2026
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
export TORCH_EXTENSIONS_DIR="$R/deps/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$R/logs/setup.exit"' EXIT
cd "$R/deps"
"$P" -m pip install 'numpy==1.26.4' 'pycolmap==4.2.1' 'scipy==1.15.3' 'torchmetrics==1.4.3' 'torch==2.7.1' 'torchvision==0.22.1' 'opencv-python-headless==4.11.0.86' tyro scikit-learn matplotlib tensorboard 'imageio[ffmpeg]' plyfile tqdm natsort ninja pyyaml setuptools wheel rich
git -c http.proxy=http://127.0.0.1:17904 clone https://github.com/nerfstudio-project/gsplat.git gsplat
git -C gsplat checkout b60e917c95afc449c5be33a634f1f457e116ff5e
git -c http.proxy=http://127.0.0.1:17904 clone https://github.com/rahul-goel/fused-ssim.git fused-ssim
git -C fused-ssim checkout 1272e21a282342e89537159e4bad508b19b34157
"$P" -m pip install --no-build-isolation --no-deps ./gsplat ./fused-ssim
"$P" -m pip freeze > "$R/research/requirements.txt"
"$P" /mnt/ojc/workplace1/4danyone/waymo/bridge/backend/src/simple_trainer_freetime_4d_pure_relocation.py --help > "$R/research/trainer-help.txt"
