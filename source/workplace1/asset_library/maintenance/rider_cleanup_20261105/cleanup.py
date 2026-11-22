from pathlib import Path
import hashlib,json,os,shutil,collections
root=Path('/mnt/ojc/workplace1/asset_library')
out=root/'maintenance/rider_cleanup_20261105'
archives=[root/'maintenance'/n/'process_archive' for n in ['rider_vehicle_20261030','rider_vehicle_20261102']]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def disk():
 seen=set();total=0
 for a in archives:
  for p in a.rglob('*'):
   if p.is_file() and not p.is_symlink():
    s=p.stat();k=(s.st_dev,s.st_ino)
    if k not in seen:total+=s.st_blocks*512;seen.add(k)
 return total
before=disk();groups=collections.defaultdict(list)
for a in archives:
 for p in a.rglob('*'):
  if p.is_file() and not p.is_symlink() and p.suffix.lower() in {'.ply','.png','.jpg','.jpeg','.mp4','.webp','.npz'} and p.stat().st_size>=65536:
   s=p.stat();groups[(s.st_dev,s.st_size,s.st_mode&0o777)].append(p)
plan=[]
for group in groups.values():
 if len(group)<2:continue
 hashed=collections.defaultdict(list)
 for p in group:hashed[sha(p)].append(p)
 for digest,ps in hashed.items():
  if len(ps)<2:continue
  ps.sort(key=lambda p:(len(str(p)),str(p)));keep=ps[0]
  for p in ps[1:]:
   if p.stat().st_ino!=keep.stat().st_ino:plan.append(dict(path=str(p),keep=str(keep),sha256=digest,bytes=p.stat().st_size))
(out/'plan.json').write_text(json.dumps(plan,indent=2))
print('PLAN',len(plan),'duplicate paths',sum(r['bytes'] for r in plan),'logical bytes',flush=True)
for r in plan:
 p=Path(r['path']);keep=Path(r['keep']);assert sha(p)==sha(keep)==r['sha256']
 tmp=p.with_name(p.name+'.dedup-tmp');os.link(keep,tmp);os.replace(tmp,p)
removed=[]
for a in archives:
 for p in list(a.rglob('__pycache__')):
  if p.is_dir() and not p.is_symlink():removed.append(str(p));shutil.rmtree(p)
for r in plan:assert sha(Path(r['path']))==r['sha256']
after=disk();report=dict(before_allocated_bytes=before,after_allocated_bytes=after,reclaimed_bytes=before-after,deduplicated_paths=len(plan),removed_cache_directories=removed,verification='All deduplicated files verified SHA256 before and after; original paths retained with hard links. Final asset directories unchanged.',scope=[str(a) for a in archives])
(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
