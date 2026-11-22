#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ojc/pre_flux_pipeline
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VLM_RESPONSE_FORMAT=json_schema
export CONFIG=/mnt/ojc/workplace1/asset_work/batch20_20261024/frontend_revision_01/configs/nuscenes_pilot.yaml OUTPUT_ROOT=/mnt/ojc/workplace1/asset_work/batch20_20261024/frontend_revision_01 RUN_NAME=nuscenes_pilot GPUS=3
exec bash scripts/run_nuscenes.sh --handoff-only
