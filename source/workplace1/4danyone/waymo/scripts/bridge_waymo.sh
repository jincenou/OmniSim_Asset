#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FDANYONE_BRIDGE_LABEL=waymo_sample
export FDANYONE_BRIDGE_SOURCE=/mnt/ojc/workplace1/4danyone/waymo/outputs/turbo
cd /mnt/ojc/workplace1/4danyone/waymo
trap 'code=$?; printf "%s\n" "$code" > logs/bridge-waymo.exit' EXIT
/mnt/ojc/miniconda3/envs/4danyone/bin/python - <<'PY'
import json
from pathlib import Path
review=json.loads(Path('outputs/turbo/preview/final_validation.json').read_text())
assert review['state']=='validated'
assert review['visual_review']=='accepted_for_bridge_probe_with_limitations'
if Path('bridge/waymo_sample').exists():
    raise FileExistsError('Preserve existing bridge/waymo_sample; review it before any explicit rerun.')
PY
/mnt/ojc/miniconda3/envs/4danyone/bin/python scripts/bridge_masks.py
/mnt/ojc/miniconda3/envs/4danyone/bin/python scripts/bridge_export.py
/mnt/ojc/miniconda3/envs/4danyone-bridge/bin/python scripts/bridge_check.py
/mnt/ojc/miniconda3/envs/4danyone/bin/python scripts/bridge_preview.py
