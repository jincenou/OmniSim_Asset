#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/waymo/backend_study/scripts/env.sh
mkdir -p "$STUDY_ROOT/cache/pip" "$STUDY_ROOT/cache/torch" "$STUDY_ROOT/cache/extensions" "$STUDY_ROOT/cache/matplotlib" "$STUDY_ROOT/cache/xdg" "$STUDY_ROOT/tmp"
trap 'code=$?; printf "%s\n" "$code" > "$STUDY_ROOT/logs/setup.exit"' EXIT
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
if [[ ! -e "$STUDY_ENV" ]]; then
    "$CUDA_HOME/bin/python" -m venv --system-site-packages "$STUDY_ENV"
fi
"$STUDY_ENV/bin/python" -c 'import sys,torch; assert sys.prefix=="/mnt/ojc/miniconda3/envs/cem4dgs"; assert torch.__version__=="2.3.1+cu118"; print(sys.prefix, torch.__version__)'
"$STUDY_ENV/bin/python" -m pip install --no-deps kornia==0.6.12 joblib==1.4.2 natsort==8.4.0 scikit-learn==1.5.2 threadpoolctl==3.5.0 faiss-gpu-cu11==1.9.0.post1
"$STUDY_ENV/bin/python" -m pip install --no-build-isolation --no-deps "$STUDY_ROOT/sources/cem4dgs/submodules/diff_gaussian_rasterization_df"
"$STUDY_ENV/bin/python" -m pip check
"$STUDY_ENV/bin/python" -m pip freeze > "$STUDY_ROOT/research/environment.lock.txt"
printf '%s\n' dependencies_installed_cuda_validation_pending > "$STUDY_ROOT/logs/setup.state"
