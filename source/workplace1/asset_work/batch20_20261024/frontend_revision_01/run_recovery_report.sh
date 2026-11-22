#!/bin/bash
set -euo pipefail
cd /mnt/ojc/pre_flux_pipeline
export ASSET_BATCH_ROOT=/mnt/ojc/workplace1/asset_work/batch20_20261024/frontend_revision_01
export ASSET_RUN_SUFFIX=_pilot ASSET_CONFIG_SUFFIX=_pilot
export ASSET_GENERATION_GPUS=0,2,6
export VLM_RESPONSE_FORMAT=json_object OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export ASSET_METRIC_GPUS=0,2,6 ASSET_PREVIEW_GPUS=0,2,6
export ASSET_METRIC_REGISTRY=/mnt/ojc/workplace1/asset_work/batch20_20261024/frontend_revision_01/evaluation/metric_registry_recovery_01.json
export ASSET_PILOT_PREVIEW_DIR=kept_previews_recovery_01
export MPLCONFIGDIR=/tmp/asset_report_matplotlib
exec /mnt/ojc/miniconda3/envs/vllm_vlm/bin/python -u /mnt/ojc/workplace1/asset_work/batch20_20261024/scripts/pilot_report.py --watch >> /mnt/ojc/workplace1/asset_work/batch20_20261024/logs/pilot_recovery_report_01.log 2>&1
