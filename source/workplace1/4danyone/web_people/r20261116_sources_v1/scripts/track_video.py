"""Read-only source tracking; preserve decoded presentation timestamps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'research/ultralytics')
os.environ['YOLO_AUTOINSTALL'] = 'false'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_id')
    parser.add_argument('video')
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO
    from source_io import read_source, source_identity
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1
    uuid = str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    assert uuid == os.environ['FD_EXPECTED_GPU_UUID'].removeprefix('GPU-')
    weights = Path('/mnt/ojc/workplace1/4danyone/repo/models/gvhmr/yolov8x.pt')
    tracker = ROOT/'scripts/bytetrack.yaml'
    assert weights.is_file()
    output = ROOT/'sources'/f'{args.source_id}_tracks.json'
    assert not output.exists()
    model = YOLO(str(weights))
    histories, timestamps = {}, []
    started = time.time()
    for index, timestamp, rgb in read_source(args.video):
        assert not timestamps or timestamp > timestamps[-1]
        timestamps.append(timestamp)
        result = model.track(rgb[:,:,::-1].copy(), persist=True,
            tracker=str(tracker), device=0, conf=.25, imgsz=960, classes=[0],
            batch=1, verbose=False, save=False, save_txt=False, save_crop=False,
            plots=False, project=str(ROOT/'research/ultralytics_runs'))[0]
        if result.boxes.id is not None:
            for ident, box, score in zip(result.boxes.id.int().cpu().tolist(),
                    result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), strict=True):
                histories.setdefault(str(ident), []).append(dict(frame=index,
                    timestamp=timestamp, bbox=box, confidence=score))
        if index % 100 == 0:
            print(json.dumps(dict(frame=index, elapsed=time.time()-started)), flush=True)
    data = dict(source_id=args.source_id, video=args.video,
        source_identity=source_identity(args.video),
        tracks=histories, timestamps=timestamps, frame_count=len(timestamps),
        gpu_uuid=uuid, seconds=time.time()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        tracker_sha256=hashlib.sha256(tracker.read_bytes()).hexdigest(),
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        identity_scope='source-local predicted tracks; requires manual continuity review',
        selected_people=[], quality_reviewed=False)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, allow_nan=False))
    temporary.replace(output)
    print(json.dumps({k:v for k,v in data.items() if k not in ('tracks','timestamps')}), flush=True)


if __name__ == '__main__':
    main()
