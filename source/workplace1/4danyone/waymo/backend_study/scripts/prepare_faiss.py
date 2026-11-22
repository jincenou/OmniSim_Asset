"""Fetch a pinned CUDA11.8 GPU FAISS wheel into an isolated trial directory."""
import hashlib,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VERSION='1.9.0.post1'
metadata=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/faiss-gpu-cu11/{VERSION}/json'))
wheel=next(x for x in metadata['urls'] if 'cp310-cp310-manylinux' in x['filename'] and 'x86_64' in x['filename'])
path=ROOT/'cache/wheels'/wheel['filename'];path.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps({'state':'downloading','version':VERSION,'bytes':wheel['size'],'filename':wheel['filename']}),flush=True)
assert not path.exists()
urllib.request.urlretrieve(wheel['url'],path)
assert path.stat().st_size==wheel['size']
assert hashlib.sha256(path.read_bytes()).hexdigest()==wheel['digests']['sha256']
target=ROOT/'tmp/faiss_candidate';target.mkdir(exist_ok=False)
with zipfile.ZipFile(path) as archive:
    for name in archive.namelist():assert not name.startswith('/') and '..' not in Path(name).parts
    archive.extractall(target)
    text=archive.read(next(x for x in archive.namelist() if x.endswith('.dist-info/METADATA'))).decode()
record={'state':'downloaded_sha256_verified_for_trial_only','distribution':'faiss-gpu-cu11','version':VERSION,
    'wheel':str(path),'url':wheel['url'],'sha256':wheel['digests']['sha256'],'size':wheel['size'],
    'metadata_url':f'https://pypi.org/pypi/faiss-gpu-cu11/{VERSION}/json',
    'dependencies':[line for line in text.splitlines() if line.startswith('Requires-Dist:')],
    'trial_directory':str(target),'installed':False,'packaging':'community GPU wheel, dynamically linked CUDA11.8'}
(ROOT/'research/faiss_candidate.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
