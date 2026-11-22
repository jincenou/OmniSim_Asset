"""Independent-track check of initial time width; a bounded matched 15k pair."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import yaml

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / 'backend/src'), str(R / 'backend')]
from simple_trainer_freetime_4d_pure_relocation import Config, FreeTime4DRunner

def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigma', type=int, choices=[8, 15], required=True)
    args = ap.parse_args()
    root = R / 'diagnostics/m02_duration'
    root.mkdir(parents=True, exist_ok=True)
    out = root / f'sigma{args.sigma}'
    out.mkdir(exist_ok=False)
    base = R / 'reconstruction/outputs/m02_short61_24'
    init = R / 'reconstruction/data/m02_short61_24/init.npz'
    with np.load(init) as data:
        values = {key: data[key].copy() for key in data.files}
    assert values['positions'].shape == (8930, 3)
    assert not values['velocities'].any()
    assert np.all(values['durations'] == .25)
    values['durations'][:] = args.sigma / 60
    initial = out / 'init.npz'
    np.savez_compressed(initial, **values)
    original, restored = np.load(init), np.load(initial)
    changed = [key for key in original.files if not np.array_equal(original[key], restored[key])]
    assert changed == (['durations'] if args.sigma == 8 else [])
    cfg_values = yaml.unsafe_load((base / 'cfg.yml').read_text())
    cfg_values.update(result_dir=str(out), init_npz_path=str(initial), max_steps=15000,
                      lr_schedule_steps=15000, save_steps=[5000, 15000], eval_steps=[15000], ckpt_path=None)
    cfg = Config(**cfg_values)
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert gpu in (4, 5, 6, 7)
    uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip().removeprefix('GPU-')
    torch.cuda.init()
    assert torch.cuda.device_count() == 1 and str(torch.cuda.get_device_properties(0).uuid) == uuid
    save(out / 'experiment.json', {'config': asdict(cfg), 'sigma_initial_frames': args.sigma,
         'changed_npz_fields': changed, 'points': 8930, 'physical_gpu': gpu, 'cuda_uuid': uuid,
         'pid': os.getpid(), 'source_init_sha256': hashlib.sha256(init.read_bytes()).hexdigest(),
         'scope': 'Independent M02 check of R07 initial-sigma hypothesis. Same 15k schedule, seed, data, split, point count and loss; only initial durations differ. Old5k is a separate schedule reference. No body velocity prior.'})
    runner = FreeTime4DRunner(0, 0, 1, cfg)
    assert len(runner.trainset) == 21 * 61 and len(runner.valset) == 3 * 61
    assert len(runner.splats['means']) == 8930
    save(out / 'initial_parameters.json', {key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                                         for key, value in runner.splats.items()})
    runner.train()
    assert runner.sampling_ledger.steps == 15000
    save(out / 'complete.json', {'state': 'trained_metrics_pending', 'steps': 15000,
                                  'sampling': runner.sampling_ledger.state()})
    runner.writer.close()

if __name__ == '__main__':
    main()
