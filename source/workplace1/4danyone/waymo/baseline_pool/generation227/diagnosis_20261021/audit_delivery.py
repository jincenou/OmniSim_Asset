"""Read-only archive verification and preview decode; never assigns quality labels."""
import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify_one(index):
    import av
    classification = json.loads((D / 'classification.json').read_text())
    row = next(r for r in classification['rows'] if r['index'] == index)
    inventory = json.loads((D / 'move_inventory.json').read_text())
    original = next(r for r in inventory['objects'] if r['id'] == row['id'])
    root = Path(row['output'])
    errors = []
    count = 0
    byte_count = 0
    for item in original['files']:
        path = root / item['path']
        try:
            stat = path.stat()
            if (stat.st_ino, stat.st_dev, stat.st_size) != (item['inode'], item['device'], item['size']):
                errors.append({'file': item['path'], 'error': 'inode_device_size_changed'})
            if sha(path) != item['sha256']:
                errors.append({'file': item['path'], 'error': 'sha256_changed'})
            count += 1
            byte_count += stat.st_size
        except OSError as exc:
            errors.append({'file': item['path'], 'error': str(exc)})
    if root.is_symlink() or Path(row['old_output']).resolve() != root.resolve():
        errors.append({'error': 'archive_or_legacy_path_mismatch'})
    frames = 0
    dimensions = set()
    try:
        with av.open(str(root / 'preview.mp4')) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            for frame in container.decode(stream):
                frames += 1
                dimensions.add((frame.width, frame.height))
        if frames != row['frames'] or len(dimensions) != 1:
            errors.append({'error': 'preview_frames_or_dimensions', 'frames': frames, 'expected': row['frames']})
    except Exception as exc:
        errors.append({'error': 'preview_decode_failed', 'message': str(exc)})
    return dict(index=index, id=row['id'], category=row['category'], output=str(root),
                files=count, bytes=byte_count, decoded_preview_frames=frames,
                dimensions=sorted(dimensions), errors=errors,
                state='verified' if not errors else 'failed', visual_quality_approved=False)


async def main():
    status_path = D / 'delivery_check_status.json'
    result_path = D / 'delivery_check.json'
    if result_path.exists():
        raise RuntimeError('Existing check receipt must not be overwritten')
    started = time.time()
    digest = sha(D / 'classification.json')
    classification = json.loads((D / 'classification.json').read_text())
    semaphore = asyncio.Semaphore(4)
    rows = []
    def status(state):
        save(status_path, dict(state=state, pid=os.getpid(), started_at=started,
                              updated_at=time.time(), completed=len(rows), total=len(classification['rows']),
                              failures=sum(r['state'] != 'verified' for r in rows),
                              purpose='Archive hashes/inodes and full preview decoding; no visual approval'))
    status('running')
    async def one(row):
        async with semaphore:
            proc = await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).resolve()),
                '--one', str(row['index']), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode:
                result = dict(index=row['index'], id=row['id'], state='failed',
                              errors=[{'error': 'worker_failed', 'returncode': proc.returncode,
                                       'stderr': stderr.decode(errors='replace')[-2000:]}])
            else:
                result = json.loads(stdout)
            rows.append(result)
            status('running')
    await asyncio.gather(*(one(row) for row in classification['rows']))
    rows.sort(key=lambda r: r['index'])
    unchanged = digest == sha(D / 'classification.json')
    failed = sum(r['state'] != 'verified' for r in rows)
    summary = dict(at=time.time(), seconds=time.time() - started, objects=len(rows),
                   failed_objects=failed, regular_files=sum(r.get('files', 0) for r in rows),
                   bytes=sum(r.get('bytes', 0) for r in rows),
                   preview_frames=sum(r.get('decoded_preview_frames', 0) for r in rows),
                   classification_sha256=digest, classification_unchanged=unchanged,
                   counts=classification['counts'], copied_video_bytes=0,
                   state='verified' if not failed and unchanged else 'failed_or_stale_snapshot',
                   scope='All original archived regular-file hashes/inodes/device/size, legacy aliases and complete previews. Not a visual quality verdict; not a fresh decode of all 24 original views.',
                   rows=rows)
    save(result_path, summary)
    with (D / 'delivery_check.csv').open('w', newline='') as stream:
        fields = ['index', 'id', 'state', 'category', 'files', 'bytes', 'decoded_preview_frames', 'output']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    status(summary['state'])
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}), flush=True)
    if summary['state'] != 'verified':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--one', type=int)
    args = parser.parse_args()
    if args.one is not None:
        print(json.dumps(verify_one(args.one), ensure_ascii=False, allow_nan=False))
    else:
        asyncio.run(main())
