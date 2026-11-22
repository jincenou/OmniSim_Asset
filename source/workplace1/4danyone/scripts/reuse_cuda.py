"""Copy RECORD-verified, Python-independent CUDA libraries into the new prefix."""
import base64
import csv
import hashlib
import json
import shutil
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/mnt/ojc/miniconda3/envs/vllm_vlm/lib/python3.10/site-packages')
TARGET = Path('/mnt/ojc/miniconda3/envs/4danyone/lib/python3.11/site-packages')
metadata = Parser().parsestr((SOURCE / 'torch-2.8.0.dist-info/METADATA').read_text())
plan = {}
packages = []

for requirement in metadata.get_all('Requires-Dist'):
    if not requirement.startswith('nvidia-'):
        continue
    name, tail = requirement.split('==', 1)
    version = tail.split(';', 1)[0].strip()
    distribution = SOURCE / (name.replace('-', '_') + '-' + version + '.dist-info')
    tags = [line for line in (distribution / 'WHEEL').read_text().splitlines() if line.startswith('Tag:')]
    if not tags or not all('py3-none-' in tag for tag in tags):
        raise RuntimeError(f'Package is not Python-independent: {name}')
    packages.append({'name': name, 'version': version, 'tags': tags})
    for relative, encoded_hash, size in csv.reader((distribution / 'RECORD').open()):
        source = (SOURCE / relative).resolve()
        destination = (TARGET / relative).resolve()
        if not source.is_relative_to(SOURCE) or not destination.is_relative_to(TARGET):
            raise RuntimeError(f'Unexpected path in RECORD: {relative}')
        if not source.is_file():
            raise RuntimeError(f'Missing source file: {source}')
        if destination.exists():
            raise RuntimeError(f'Target already occupied: {destination}')
        if size and source.stat().st_size != int(size):
            raise RuntimeError(f'Source size mismatch: {relative}')
        if encoded_hash:
            algorithm, expected = encoded_hash.split('=', 1)
            with source.open('rb') as stream:
                actual = base64.urlsafe_b64encode(hashlib.file_digest(stream, algorithm).digest()).rstrip(b'=').decode()
            if actual != expected:
                raise RuntimeError(f'Source RECORD hash mismatch: {relative}')
        plan[relative] = (source, destination)

for source, destination in plan.values():
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if source.stat().st_ino == destination.stat().st_ino:
        raise RuntimeError('Unexpected hard link; isolation requires independent files')

result = {'source_environment_unchanged': True, 'source': str(SOURCE), 'target': str(TARGET), 'packages': packages, 'file_count': len(plan), 'bytes': sum(source.stat().st_size for source, _ in plan.values()), 'validation': 'all available RECORD digests and sizes checked before copying; independent file copies'}
(ROOT / 'research/cuda-reuse.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
