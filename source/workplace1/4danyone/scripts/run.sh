#!/usr/bin/env bash
set -euo pipefail
source /mnt/ojc/workplace1/4danyone/scripts/env.sh
cd /mnt/ojc/workplace1/4danyone/repo
test -f ../logs/setup.complete
test -f ../logs/download.complete
PYTHON=/mnt/ojc/miniconda3/envs/4danyone/bin/python
MODE=${1:?Specify smoke, turbo, or base}
case "$MODE" in
    smoke) VIEWS=6; GPUS='[0]'; TURBO=True; export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" ;;
    turbo) VIEWS=24; GPUS='[0,1,2,3]'; TURBO=True ;;
    base) VIEWS=24; GPUS='[0,1,2,3]'; TURBO=False ;;
    *) printf '%s\n' 'Unknown mode' >&2; exit 2 ;;
esac
OUTPUT=/mnt/ojc/workplace1/4danyone/outputs/$MODE
test ! -e "$OUTPUT"
"$PYTHON" -c 'import shutil; assert shutil.disk_usage(".").free > 10 * 1024**3, "Need more than 10 GiB free before inference"'
"$PYTHON" -c 'import os, subprocess, torch; ids=os.environ["CUDA_VISIBLE_DEVICES"]; assert set(ids.split(",")).issubset({"4","5","6","7"}); print(subprocess.check_output(["nvidia-smi", "--id="+ids,"--query-gpu=index,uuid,memory.free","--format=csv,noheader"],text=True)); assert torch.cuda.is_available(); assert torch.cuda.device_count() == len(ids.split(",")); x=torch.ones(8,device="cuda"); assert x.sum().item()==8; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$PYTHON" scripts/download_smplx.py --archive_path /mnt/ojc/lhmplus/models/Damo_XR_Lab/LHMPP-Prior/human_model_files/smplx/SMPLX_NEUTRAL.npz
exec "$PYTHON" inference.py \
    --video_path data/source/pexels/2785536-uhd_2160_3840_25fps.mp4 \
    --output_dir "$OUTPUT" \
    --views_per_layer "$VIEWS" \
    --gpu_ids "$GPUS" \
    --attention_backend sdpa \
    --enable_turbo "$TURBO" \
    --seed 42
