import json,os,time,hashlib,collections,sys
from pathlib import Path
O=Path(__file__).parent;L=Path('/mnt/ojc/workplace1/asset_library');R=[O.parent/n for n in ['scene_production_20261031','web_vehicles_20261103']]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def sig(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink]
def strings(x):
 if isinstance(x,str):yield x
 elif isinstance(x,dict):
  for v in x.values():yield from strings(v)
 elif isinstance(x,list):
  for v in x:yield from strings(v)
def write(n,d):(O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def protected():
 refs=set();manifests=[L/'manifests/asset_library_rigid.json',L/'releases/rigid/drivelab_asset.json',L/'human/drivelab_asset.json']
 for p in manifests:
  for s in strings(json.loads(p.read_text())):
   if s.endswith('.ply'):
    t=Path(s)
    if t.is_absolute():refs.add(str(t.resolve()))
    else:
     refs.add(str((L/t).resolve()));refs.add(str((p.parent/t).resolve()))
 return refs,{str(p):sha(p) for p in manifests}
def plan():
 refs,manifests=protected();candidates={};blocked=set();states={};cps=[]
 for root in R:
  for cp in (root/'chunks').glob('*/completed.json'):
   if time.time()-cp.stat().st_mtime<7200:continue
   cps.append(cp);states[str(cp)]=sha(cp)
   for sp in cp.parent.rglob('curation.json'):
    states[str(sp)]=sha(sp)
    for oid,a in json.loads(sp.read_text()).get('assets',{}).items():
     e=a.get('entry',{});paths=[e.get('source_path'),e.get('canonical',{}).get('canonical_path')];canon=e.get('canonical',{}).get('canonical_path')
     if canon:paths.append(str(Path(canon).parent/'axis.ply'))
     for name in paths:
      if not name:continue
      p=Path(name)
      if not p.is_absolute():continue
      if a.get('status')!='reject':blocked.add(str(p.resolve()));continue
      if p.name not in ['raw.ply','canonical.ply','axis.ply'] or p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(cp.parent.resolve()):continue
      if p.stat().st_nlink!=1:continue
      candidates[str(p)]={'path':str(p),'signature':sig(p),'allocated':p.stat().st_blocks*512,'chunk':str(cp.parent),'oid':oid,'reason':'rejected_model','state':str(sp),'state_sha256':states[str(sp)]}
 files=[v for k,v in candidates.items() if str(Path(k).resolve()) not in refs|blocked]
 for cp in cps:
  for p in cp.parent.rglob('*'):
   if p.is_symlink() or not p.is_file():continue
   if p.suffix in ['.nbc','.nbi'] and 'cache/numba/' in str(p) and p.stat().st_nlink==1:
    files.append({'path':str(p),'signature':sig(p),'allocated':p.stat().st_blocks*512,'chunk':str(cp.parent),'reason':'rebuildable_numba_cache'})
 touched={v['chunk'] for v in files};remove={v['path'] for v in files};preserved=[]
 for ch in touched:
  for p in Path(ch).rglob('*'):
   if p.is_file() and str(p) not in remove:preserved.append([str(p),sig(p)])
 write('rejected_preserved.json',preserved)
 d={'files':files,'manifests':manifests,'states':{k:v for k,v in states.items() if any(Path(k).is_relative_to(ch) for ch in touched)},'reason_counts':dict(collections.Counter(v['reason'] for v in files)),'GiB':sum(v['allocated'] for v in files)/2**30,'preserved_count':len(preserved)}
 write('rejected_plan.json',d);print(json.dumps({k:v for k,v in d.items() if k not in ['files','manifests','states']}),flush=True)
def apply():
 d=json.loads((O/'rejected_plan.json').read_text());refs,manifests=protected();assert manifests==d['manifests'],'Library changed; replan'
 for k,v in d['states'].items():assert sha(Path(k))==v,('state_changed',k)
 before=os.statvfs('/mnt');deleted=[];skip=[]
 with (O/'rejected_deleted.jsonl').open('a') as log:
  for v in d['files']:
   p=Path(v['path'])
   if p.is_symlink() or not p.is_file() or sig(p)!=v['signature'] or str(p.resolve()) in refs:skip.append(str(p));continue
   assert p.resolve().is_relative_to(Path(v['chunk']).resolve())
   log.write(json.dumps({'event':'intent',**v})+'\n');log.flush();p.unlink();deleted.append(v)
   log.write(json.dumps({'event':'deleted','path':str(p),'bytes':v['allocated']})+'\n');log.flush()
 changed=[]
 for n,s in json.loads((O/'rejected_preserved.json').read_text()):
  p=Path(n)
  if not p.is_file() or sig(p)!=s:changed.append(n)
 after=os.statvfs('/mnt');r={'deleted':len(deleted),'counts':dict(collections.Counter(v['reason'] for v in deleted)),'allocated_GiB':sum(v['allocated'] for v in deleted)/2**30,'free_before_GiB':before.f_bavail*before.f_frsize/2**30,'free_after_GiB':after.f_bavail*after.f_frsize/2**30,'skipped':skip,'preserved_count':d['preserved_count'],'preserved_changed':changed,'library_unchanged':protected()[1]==manifests}
 write('rejected_result.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':plan() if sys.argv[1]=='plan' else apply()
