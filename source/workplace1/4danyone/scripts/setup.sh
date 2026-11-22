#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
cd /mnt/ojc/workplace1/4danyone/repo
ENV_PREFIX=/mnt/ojc/miniconda3/envs/4danyone
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    /mnt/ojc/miniconda3/bin/conda create --prefix "$ENV_PREFIX" python=3.11 pip -y
fi
"$ENV_PREFIX/bin/python" -m pip install --no-cache-dir /mnt/ojc/workplace1/4danyone/cache/wheels/*.whl -r requirements.txt
"$ENV_PREFIX/bin/python" -m pip check
"$ENV_PREFIX/bin/python" -m pip freeze > ../research/requirements.lock.txt
"$ENV_PREFIX/bin/python" inference.py --help > ../research/inference-help.txt 2>&1
touch ../logs/setup.complete
