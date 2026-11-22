"""Download pinned official checkpoints with size and SHA256 verification."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import time

ROOT = Path(__file__).resolve().parents[1]

def fetch(job):
    repo, rev, item = job
    name = item['rfilename']
    dest = ROOT / 'cache' / repo.replace('/', '--') / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get('lfs', {}).get('sha256')
    def valid(path):
        if not path.is_file() or path.stat().st_size != item['size']:
            return False
        if expected:
            h = hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
            return h.hexdigest() == expected
        return True
    if valid(dest): return str(dest)
    part = dest.with_name(dest.name + '.partial')
    url = f'https://huggingface.co/{repo}/resolve/{rev}/{name}'
    for attempt in range(5):
        try:
            offset = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(req, timeout=120) as response:
                mode = 'ab' if offset and response.status == 206 else 'wb'
                with part.open(mode) as out:
                    while True:
                        chunk = response.read(4 * 1024**2)
                        if not chunk: break
                        out.write(chunk)
            break
        except Exception:
            if attempt == 4: raise
            time.sleep(2 ** attempt)
    if not valid(part): raise ValueError(f'Checkpoint integrity failure: {part}')
    part.replace(dest)
    print('VERIFIED', dest, flush=True)
    return str(dest)

def main():
    jobs = []
    for repo in ['Stable-X/trellis-vggt-v0-2', 'Stable-X/vggt-object-v0-1', 'ZhengPeng7/BiRefNet']:
        meta = json.loads((ROOT/'docs'/(repo.replace('/', '--')+'.json')).read_text())
        for item in meta['siblings']:
            name = item['rfilename']
            if name.endswith(('.json', '.safetensors', '.py')) and name != 'handler.py':
                jobs.append((repo, meta['sha'], item))
    required = sum(x[2]['size'] for x in jobs if not (ROOT/'cache'/x[0].replace('/', '--')/x[2]['rfilename']).exists())
    free = shutil.disk_usage(ROOT).free
    if free < required + 2 * 1024**3:
        raise RuntimeError(f'Insufficient space: need {required}+2GiB, free {free}')
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, jobs))
    (ROOT/'docs/weights_downloaded.json').write_text(json.dumps(results, indent=2))

if __name__ == '__main__': main()
