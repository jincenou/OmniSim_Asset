#!/bin/bash
set -euo pipefail
export HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human300_20261024
exec /mnt/ojc/miniconda3/envs/4danyone/bin/python -u - <<'PY'
import sys,asyncio
sys.path.insert(0,'/mnt/ojc/workplace1/human')
from common import worker
asyncio.run(worker('prepare',['accessory','H0001'],0,library='4danyone-sam2',log_name='H0001_accessory.log'))
PY
