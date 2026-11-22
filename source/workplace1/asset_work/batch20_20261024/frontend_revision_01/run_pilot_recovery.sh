#!/bin/bash
set -euo pipefail
cd /mnt/ojc/pre_flux_pipeline
export ASSET_BATCH_ROOT=/mnt/ojc/workplace1/asset_work/batch20_20261024/frontend_revision_01
export ASSET_RUN_SUFFIX=_pilot ASSET_CONFIG_SUFFIX=_pilot
export ASSET_GENERATION_GPUS=0,2,6
export VLM_RESPONSE_FORMAT=json_object OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec /mnt/ojc/miniconda3/envs/vllm_vlm/bin/python -u /mnt/ojc/workplace1/asset_work/batch20_20261024/scripts/run_batch.py --recover >> /mnt/ojc/workplace1/asset_work/batch20_20261024/logs/pilot_recovery_01.log 2>&1
