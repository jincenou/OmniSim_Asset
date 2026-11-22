#!/bin/bash
set -euo pipefail
export HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human_tail_20261024
export HUMAN_ASSET_ROOT=/mnt/ojc/workplace1/asset_library/human_tail_validation
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
/mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/frontend.py subset H0066 --source-root /mnt/ojc/workplace1/asset_work/human300_20261024 --source-identity H0066 --end-frame 53
/mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/frontend.py subset H0081 --source-root /mnt/ojc/workplace1/asset_work/human300_20261024 --source-identity H0081 --end-frame 53
/mnt/ojc/miniconda3/envs/4danyone/bin/python -u - <<'WAIT'
import subprocess,time
while True:
    free=[int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) for gpu in (0,2)]
    if min(free)>=24000:break
    print('Waiting for GPU0/2 memory',free,flush=True)
    time.sleep(30)
WAIT
exec /mnt/ojc/miniconda3/envs/4danyone/bin/python -u /mnt/ojc/workplace1/human/run.py batch pilot --cases H0066 H0081 --gpus 0 2
