import json, os, hashlib, time, stat, collections, concurrent.futures, argparse
from pathlib import Path
OUT=Path(__file__).parent
ROOTS=[Path('/mnt/ojc/workplace1/asset_work')/n for n in ['scene_production_20261031','web_vehicles_20261103']]
LIB=Path('/mnt/ojc/workplace1/asset_library')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def sig(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def strings(x):
 if isinstance(x,str):yield x
 elif isinstance(x,dict):
  for v in x.values():yield from strings(v)
 elif isinstance(x,list):
  for v in x:yield from strings(v)
def protect_library():
 refs=set()
 for mp in [LIB/'manifests/asset_library_rigid.json',LIB/'releases/rigid/drivelab_asset.json']:
  for s in strings(json.loads(mp.read_text())):
   if s.endswith('.ply'):
    p=Path(s)
    if p.is_absolute():refs.add(str(p.resolve()))
    else:
     refs.add(str((LIB/p).resolve()));refs.add(str((mp.parent/p).resolve()))
 return refs
def plan():
 protected=protect_library();candidates={};skips=collections.Counter();completed=[]
 for root in ROOTS:
  for cp in (root/'chunks').glob('*/completed.json'):
   if time.time()-cp.stat().st_mtime<3600:continue
   completed.append(cp)
   for sp in cp.parent.rglob('curation.json'):
    d=json.loads(sp.read_text())
    for oid,a in d.get('assets',{}).items():
     e=a.get('entry',{});cn=e.get('canonical',{});cs=cn.get('canonical_path');rs=e.get('source_path')
     if not cs:continue
     protected.add(str(Path(cs).resolve()))
     if rs:protected.add(str(Path(rs).resolve()))
     canon=Path(cs);axis=canon.parent/'axis.ply'
     if canon.name!='canonical.ply' or not axis.is_file() or not rs:continue
     raw=Path(rs)
     if not canon.is_file() or not raw.is_file():skips['missing_canonical_or_raw']+=1;continue
     if any(p.is_symlink() or not p.resolve().is_relative_to(cp.parent.resolve()) for p in [axis,canon,raw]):skips['link_or_outside_chunk']+=1;continue
     if axis.stat().st_nlink!=1:skips['hardlinked_axis']+=1;continue
     expected=cn.get('sha256') or a.get('canonical_sha256')
     if not expected:skips['missing_hash']+=1;continue
     candidates[str(axis)]=dict(path=str(axis),bytes=axis.stat().st_size,allocated_bytes=axis.stat().st_blocks*512,signature=sig(axis),canonical=str(canon),canonical_sha256=expected,canonical_signature=sig(canon),raw=str(raw),raw_signature=sig(raw),state=str(sp),state_sha256=digest(sp),completed=str(cp),completed_sha256=digest(cp),obj_id=oid,status=a.get('status'),reason='Superseded axis-normalized intermediate; final canonical and original raw retained')
 candidates=[v for k,v in candidates.items() if str(Path(k).resolve()) not in protected]
 def verify(v):
  if digest(Path(v['canonical']))!=v['canonical_sha256']:return None
  return v
 verified=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
  for i,v in enumerate(ex.map(verify,candidates),1):
   if v:verified.append(v)
   else:skips['canonical_hash_mismatch']+=1
   if i%500==0:print('verified',i,'/',len(candidates),flush=True)
 # Record every preserved file in the frozen completed chunks, including input images and renders.
 preserved=[];remove={v['path'] for v in verified}
 for cp in completed:
  for p in cp.parent.rglob('*'):
   if p.is_file() and str(p) not in remove:preserved.append([str(p),sig(p)])
 write(OUT/'preserved_files.json',preserved)
 result=dict(created=time.time(),authorization='User authorized deleting unnecessary intermediate files and iterations on 2026-11-05; retain original inputs and final assets',files=verified,skipped=dict(skips),completed_chunks=len(completed),preserved_files=len(preserved),candidate_count=len(verified),allocated_GiB=sum(v['allocated_bytes'] for v in verified)/2**30)
 write(OUT/'plan.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False),flush=True)
def apply():
 d=json.loads((OUT/'plan.json').read_text());protected=protect_library();deleted=[];skipped=[];before=os.statvfs('/mnt');free_before=before.f_bavail*before.f_frsize
 log=(OUT/'deleted.jsonl').open('a')
 for v in d['files']:
  p=Path(v['path']);canon=Path(v['canonical']);raw=Path(v['raw']);sp=Path(v['state']);cp=Path(v['completed'])
  if not p.exists():skipped.append([str(p),'already absent']);continue
  if str(p.resolve()) in protected or p.is_symlink() or p.name!='axis.ply' or sig(p)!=v['signature'] or sig(canon)!=v['canonical_signature'] or sig(raw)!=v['raw_signature'] or digest(sp)!=v['state_sha256'] or digest(cp)!=v['completed_sha256']:
   skipped.append([str(p),'state or file changed']);continue
  # All mutations restricted to explicitly planned, verified axis intermediates.
  log.write(json.dumps(dict(event='delete_intent',**v),ensure_ascii=False)+'\n');log.flush()
  p.unlink()
  log.write(json.dumps(dict(event='deleted',path=str(p),allocated_bytes=v['allocated_bytes'],time=time.time()))+'\n');log.flush();deleted.append(v)
 log.close()
 changed=[];missing=[]
 for path,expected in json.loads((OUT/'preserved_files.json').read_text()):
  p=Path(path)
  if not p.is_file():missing.append(path)
  elif sig(p)!=expected:changed.append(path)
 after=os.statvfs('/mnt');free_after=after.f_bavail*after.f_frsize
 result=dict(finished=time.time(),deleted_files=len(deleted),unlinked_allocated_GiB=sum(x['allocated_bytes'] for x in deleted)/2**30,free_before_GiB=free_before/2**30,free_after_GiB=free_after/2**30,skipped=skipped,preserved_file_count=d['preserved_files'],preserved_missing=missing,preserved_changed=changed,production_note='Supervisors remain running; free space may be consumed concurrently',scope='Only superseded axis.ply in committed chunks; raw, canonical, FLUX, sources, audit images, metadata and all library/release/upload files retained')
 write(OUT/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','apply']);args=ap.parse_args();plan() if args.action=='plan' else apply()
