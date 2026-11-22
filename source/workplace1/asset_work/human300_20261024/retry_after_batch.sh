#!/bin/bash
set -euo pipefail
export HUMAN_RUN_ROOT=/mnt/ojc/workplace1/asset_work/human300_20261024
export HUMAN_ASSET_ROOT=/mnt/ojc/workplace1/asset_library/human
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
/mnt/ojc/miniconda3/envs/4danyone/bin/python - <<'PY'
from pathlib import Path
import time, os, json, sys
p = Path('/proc/3921075/cmdline')
expected = b'/mnt/ojc/workplace1/human/run.py\x00batch\x00run\x00'
while p.exists():
    try:
        if expected not in p.read_bytes():
            break
    except FileNotFoundError:
        break
    time.sleep(30)
ids = json.loads(Path('/mnt/ojc/workplace1/asset_work/human300_20261024/RETRY_INPUTS.json').read_text())['ids']
os.execv(sys.executable, [sys.executable, '-u', '/mnt/ojc/workplace1/human/run.py', 'batch', 'run', '--cases', *ids, '--gpus', *map(str, range(8))])
PY
