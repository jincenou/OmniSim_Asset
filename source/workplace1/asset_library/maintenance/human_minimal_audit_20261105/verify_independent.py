from pathlib import Path
from unittest.mock import patch
from collections import Counter
import builtins,io,json,shutil,tempfile,hashlib
import numpy as np
from hasim.shared.assets.library_v2 import AssetLibrary,load_asset
from hasim.shared.assets.human_cycle import CycleProfile,CyclicRuntimeAsset
ROOT=Path('/mnt/ojc/workplace1/asset_library')
OUT=ROOT/'maintenance/human_minimal_audit_20261105'
REPO=Path('/mnt/ojc/workplace1/Hasim_trajectory_assets')
source=AssetLibrary(ROOT);report={'samples':[]}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
for location in [Path('/mnt/ojc/workplace1/asset_minimal_examples/20261029/02_portable_human'),REPO/'local/human_cycles/minimal_H1132']:
 portable=AssetLibrary(location).assets[0];base=load_asset(source.get(portable.id))
 before={str(f):digest(f) for f in base.files_used}
 denied=[ROOT/'human',Path('/mnt/ojc/workplace1/asset_work'),location]
 raw_open=builtins.open;raw_io=io.open;blocked=[]
 def check(file):
  if isinstance(file,(str,bytes,Path)):
   path=Path(file.decode() if isinstance(file,bytes) else file).resolve()
   if any(path.is_relative_to(p) for p in denied):
    blocked.append(str(path));raise PermissionError('Source access blocked by independent audit')
 def guarded(file,*a,**kw):check(file);return raw_open(file,*a,**kw)
 def guarded_io(file,*a,**kw):check(file);return raw_io(file,*a,**kw)
 with tempfile.TemporaryDirectory(prefix='human-independent-',dir='/tmp') as temp:
  dest=Path(temp)/'relocated';shutil.copytree(location,dest)
  with patch('builtins.open',guarded),patch('io.open',guarded_io):
   entry=AssetLibrary(dest).assets[0];runtime=load_asset(entry)
   assert all(np.array_equal(v,base.arrays[k]) for k,v in runtime.arrays.items())
   for t in [0,.11,entry.duration_s/2,entry.duration_s]:
    a,b=runtime.sample(t),base.sample(t)
    assert all(np.array_equal(getattr(a,k),getattr(b,k)) for k in ['means','quaternions','scales','opacities','sh','local_from_appearance'])
   cycle_times=[]
   if entry.meta.get('cycle_profile'):
    p=CycleProfile.from_dict(entry.meta['cycle_profile']);x=CyclicRuntimeAsset(runtime,p);y=CyclicRuntimeAsset(base,p)
    cycle_times=[0,.11,20.5,100.52]
    for t in cycle_times:
     a,b=x.sample(t),y.sample(t)
     assert all(np.array_equal(getattr(a,k),getattr(b,k)) for k in ['means','quaternions','scales','opacities','sh','local_from_appearance'])
   assert all(f.is_relative_to(dest) for f in runtime.files_used)
  assert not blocked
 after={str(f):digest(f) for f in base.files_used};assert before==after
 original=ROOT/'human'/portable.id.rsplit('_',1)[-1]
 report['samples'].append({'id':portable.id,'portable_path':str(location),'bytes':sum(p.stat().st_size for p in location.iterdir() if p.is_file()),'source_runtime_hashes_unchanged':before==after,'source_hashes':before,'relocated_with_source_reads_blocked':True,'runtime_arrays_exact':True,'cycle_sample_times':cycle_times,'runtime_files':[f.name for f in runtime.files_used], 'source_package_file_count':sum(f.is_file() for f in original.rglob('*'))})
rows=json.loads((ROOT/'human/drivelab_asset.json').read_text())['assets']
report['inventory']={'manifest_entries':len(rows),'usable_candidates':len(source.candidates('human')),'excluded':source.excluded,'quality_states':dict(Counter(x.get('quality_state','unknown') for x in rows))}
report['status']='passed'
(OUT/'independent_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'samples':len(report['samples']),'inventory':report['inventory']},ensure_ascii=False))
