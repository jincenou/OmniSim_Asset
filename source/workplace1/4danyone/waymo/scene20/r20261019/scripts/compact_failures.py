"""Retain diagnostic evidence for terminal failures, not disposable decoded frames."""
import hashlib
import json
import shutil
from pathlib import Path

R = Path(__file__).resolve().parents[1]

def main():
    jobs = {}
    for name in ('STATUS.json', 'RESERVE_STATUS.json'):
        jobs.update(json.loads((R / name).read_text())['jobs'])
    archive = R / 'research/failures'
    archive.mkdir(exist_ok=True)
    records = []
    for key, job in jobs.items():
        if job['state'] != 'failed':
            continue
        matches = list((R / 'scenes').glob(f'*/{key}'))
        if not matches:
            continue
        assert len(matches) == 1
        b = matches[0]
        assert not (b / 'manifest.json').exists(), 'Never compact a delivered result'
        w = b / '.work'
        if w.exists():
            assert job.get('stage') in ('conditioning', 'motion', 'motion_gate'), job
            receipt = json.loads((R / f'research/{key}/{job["stage"]}.json').read_text())
            assert receipt['exit_code'] != 0
            mappings = {
                'source/native_contact.jpg': 'input/native_contact.jpg',
                'source/native.json': 'input/native.json',
                'source/native_cameras.npz': 'input/native_cameras.npz',
                'source/inputs/main.mp4': 'input/model_input.mp4',
                'source/inputs/cameras.npz': 'input/source_cameras.npz',
                'source/inputs/frame_mapping.json': 'input/frame_mapping.json',
                'source/segmentation/target_masks.npz': 'input/target_masks.npz',
                'source/research/batch_gate_mask.json': 'input/mask_gate.json',
                'source/research/motion_gate.json': 'motion/gate.json',
                'source/motion_pair/projection_comparison.mp4': 'motion/source_fit.mp4',
                'source/motion_pair/comparison.json': 'motion/comparison.json',
                'generation/preflight/root_motion.npz': 'motion/root_motion.npz',
            }
            motion = w / 'source/motion_pair/calibrated_moving'
            if motion.exists():
                mappings.update({str(p.relative_to(w)): f'motion/{p.name}' for p in motion.iterdir() if p.is_file()})
            retained = {}
            for src, dst in mappings.items():
                p, q = w / src, b / dst
                if not p.is_file():
                    continue
                q.parent.mkdir(exist_ok=True)
                shutil.copy2(p, q)
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                assert hashlib.sha256(q.read_bytes()).hexdigest() == digest
                retained[dst] = digest
            scratch_bytes = sum(p.stat().st_size for p in w.rglob('*') if p.is_file())
            (b / 'failure.json').write_text(json.dumps({
                'job': job, 'retained': retained, 'no_generated_result_claimed': True,
                'removed_scratch_bytes': scratch_bytes,
                'discarded': 'private decoded crop/SAM frames, duplicate intermediate videos, pose feature caches',
                'original_data_modified': False,
            }, indent=2))
            shutil.rmtree(w)
        assert (b / 'failure.json').exists()
        destination = archive / key
        assert not destination.exists()
        source = str(b.relative_to(R))
        b.rename(destination)
        if not any(b.parent.iterdir()):
            b.parent.rmdir()
        records.append({'sample': key, 'from': source, 'to': str(destination.relative_to(R))})
    history = R / 'research/failure_cleanup.json'
    old = json.loads(history.read_text()) if history.exists() else []
    history.write_text(json.dumps(old + records, indent=2))
    print(json.dumps({'archived_terminal_failures': len(records)}))

if __name__ == '__main__':
    main()
