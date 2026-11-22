"""Copy selected immutable package files, checking source/destination hashes. No hardlinks."""
from pathlib import Path
import importlib.metadata as im,hashlib,json,shutil
from packaging.requirements import Requirement
R=Path('/mnt/ojc/workplace1/4danyone/eff4d');src=Path('/mnt/ojc/miniconda3/envs/idol/lib/python3.10/site-packages');dst=Path('/mnt/ojc/miniconda3/envs/eff4d/lib/python3.10/site-packages')
def norm(x):return x.lower().replace('_','-').replace('.','-')
dists={norm(d.metadata['Name']):d for d in im.distributions(path=[str(src)])}
names=['torch','torchvision','numpy','scipy','pillow','plyfile','tqdm','ninja','imageio','imageio-ffmpeg','kornia','imagesize','omegaconf','opencv-python','lpips','pytorch-msssim']
queue=names[:];chosen={};missing=[]
while queue:
 n=norm(queue.pop())
 if n in chosen:continue
 if n not in dists:missing.append(n);continue
 d=dists[n];chosen[n]=d
 for raw in d.requires or []:
  req=Requirement(raw)
  if req.marker is None or req.marker.evaluate({'python_version':'3.10','python_full_version':'3.10.22','extra':''}):queue.append(req.name)
report={'packages':{},'missing':sorted(set(missing)),'files':[]}
for n,d in chosen.items():
 if n in ('pip','setuptools','wheel','packaging'):continue
 report['packages'][n]=d.version
 for f in d.files or []:
  rel=Path(str(f))
  if '..' in rel.parts or rel.suffix=='.pyc':continue
  s=src/rel;t=dst/rel
  if not s.is_file():raise FileNotFoundError(s)
  before=s.stat();h=hashlib.sha256(s.read_bytes()).hexdigest();t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(s,t);shutil.copymode(s,t)
  assert hashlib.sha256(t.read_bytes()).hexdigest()==h
  after=s.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  assert s.stat().st_ino!=t.stat().st_ino
  report['files'].append({'source':str(s),'target':str(t),'sha256':h})
 print(n,d.version,flush=True)
# Private CUDA toolkit copied by explicit tree; no Python packages or old extension binaries.
toolsrc=Path('/mnt/ojc/miniconda3/envs/idol');tool=R/'cache/cuda118';tool.mkdir(exist_ok=True)
paths=['include','nvvm','bin/nvcc','bin/ptxas','bin/fatbinary','bin/nvlink','bin/cudafe++','bin/bin2c','bin/nvcc.profile']
paths += [str(p.relative_to(toolsrc)) for p in (toolsrc/'lib').glob('*cudart*')]
for rel in paths:
 s=toolsrc/rel
 if not s.exists():raise FileNotFoundError(s)
 files=list(s.rglob('*')) if s.is_dir() else [s]
 for f in files:
  if not f.is_file():continue
  t=tool/f.relative_to(toolsrc);t.parent.mkdir(parents=True,exist_ok=True)
  h=hashlib.sha256(f.read_bytes()).hexdigest();shutil.copyfile(f,t);shutil.copymode(f,t);assert hashlib.sha256(t.read_bytes()).hexdigest()==h
  report['files'].append({'source':str(f),'target':str(t),'sha256':h})
(R/'research/runtime_copy.json').write_text(json.dumps(report,indent=2));print('COMPLETE',report['missing'],flush=True)
