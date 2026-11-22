"""Scoped, hashed cleanup of this task only. No traversal through vendor symlinks."""
from pathlib import Path
import os,json,hashlib,collections,gzip,re,sys,datetime
L=Path('/mnt/ojc/workplace1/asset_library');W=Path('/mnt/ojc/workplace1/rider_vehicle_asset');M=L/'maintenance/rider_vehicle_20261030';A=M/'process_archive';C=Path(__file__).resolve().parent
PACKAGES=['H_rider_nuscenes_b2710614','B0_rider_niu_blue','H_rider_niu_black']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def files(root):
 for folder,dirs,names in os.walk(root,followlinks=False):
  dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
  for n in names:
   p=Path(folder)/n
   if not p.is_symlink():yield p
if '--apply' not in sys.argv:
 assert W.is_dir() and not W.is_symlink() and not A.exists()
 protected={};duplicates={}
 for name in PACKAGES:
  folder=L/'rigid/motorcycle'/name
  receipt=json.loads((folder/'package.json').read_text())
  for n,h in receipt['files'].items():assert sha(folder/n)==h
  for p in files(folder):
   h=sha(p);protected[str(p)]=h;duplicates.setdefault((p.stat().st_size,h),p)
 for p in [L/'manifests/asset_library_rigid.json',L/'rigid/asset_library_rigid.json',L/'manifests/global_asset_manifest.jsonl',L/'rigid/INDEX.csv',L/'rigid_index.html',L/'previews/rider_vehicle_demos/index.html']:
  protected[str(p)]=sha(p)
 # Keep native selected source frames and +/-2 neighboring frames for pose evidence.
 keep_native=set()
 for p in (W/'packets').glob('*/packet.json'):
  for row in json.loads(p.read_text())['observations']:
   f=Path(row['image_path'])
   if 'niu_mqi_native' in f.parts:
    match=re.match(r'(.*_)(\d{3})\.png$',f.name)
    for i in range(max(0,int(match[2])-2),int(match[2])+3):keep_native.add(str(f.with_name(match[1]+f'{i:03}.png')))
 plan=[];total=0
 for p in files(W):
  rel=p.relative_to(W);parts=rel.parts;size=p.stat().st_size;h=sha(p);total+=size;action='keep';reason='input/code/config/log/review or useful comparison evidence';target=None
  if any(x in parts for x in ['__pycache__','.pytest_cache']) or ('runs'==parts[0] and 'cache' in parts):action='delete';reason='rebuildable task-local cache'
  elif rel==Path('src/yt-dlp'):action='delete';reason='downloaded tool executable, not source data'
  elif parts[0]=='runs' and ('frames' in parts or 'diagnostics' in parts) and p.suffix in ['.png','.jpg']:action='delete';reason='per-frame rasterization; retain video/contact sheets/diagnostic JSON'
  elif parts[0]=='runs' and p.name=='ground_grid.npz':action='delete';reason='derived scene ground grid, recreated by existing renderer'
  elif parts[0]=='sources' and 'niu_mqi_native' in parts and p.suffix=='.png' and str(p) not in keep_native:action='delete';reason='unused native extracted frame; selected frame/neighbors/video retained'
  elif parts[0]=='sources' and p.name=='niu_black_8p85_9p8.mp4':action='delete';reason='reencoded clipped unused shot; full source preview and native selected frames retained'
  elif parts[:2]==('sources','web') and (re.match(r'(black|blue)_\d+\.jpg$',p.name) or (any(x.endswith('_frames') for x in parts) and re.match(r'\d+\.jpg$',p.name))):action='delete';reason='redundant video preview frame; contact sheet and source video retained'
  elif (size,h) in duplicates and size>=1024:
   action='link_final';reason='byte-identical finalized asset file';target=str(duplicates[(size,h)])
  elif parts[0]=='runs' and p.suffix=='.ply':action='delete';reason='unselected reconstruction or intermediate axis-aligned PLY; comparison images/video/recipe retained'
  item=dict(path=str(rel),size=size,sha256=h,action=action,reason=reason)
  if target:item['target']=target
  plan.append(item)
 backup=[dict(path=str(p.relative_to(M)),size=p.stat().st_size,sha256=sha(p)) for p in files(M/'before')]
 result=dict(old_work=str(W),archive=str(A),before_work_bytes=total,protected=protected,files=plan,compress_backups=backup,keep_native_frames=sorted(keep_native),authorization='2026-10-31 user explicitly requested deleting unused intermediates and consolidating retained process inputs')
 (C/'plan.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 counts=collections.Counter();sizes=collections.Counter()
 for r in plan:counts[r['action']]+=1;sizes[r['action']]+=r['size']
 print(json.dumps(dict(counts=counts,bytes=sizes,backup_bytes=sum(x['size'] for x in backup)),indent=2));sys.exit()
plan=json.loads((C/'plan.json').read_text());assert not A.exists() and not W.is_symlink()
# Validate the entire input snapshot and final artifacts before mutation.
for p,h in plan['protected'].items():assert sha(Path(p))==h,p
for r in plan['files']:assert sha(W/r['path'])==r['sha256'],r['path']
for r in plan['compress_backups']:assert sha(M/r['path'])==r['sha256']
W.rename(A);W.symlink_to(A,target_is_directory=True)
freed=0;counts=collections.Counter()
with (C/'operations.jsonl').open('a') as journal:
 for r in plan['files']:
  if r['action']=='keep':continue
  p=A/r['path'];assert not p.is_symlink() and sha(p)==r['sha256']
  if r['action']=='link_final':assert sha(Path(r['target']))==r['sha256']
  journal.write(json.dumps(dict(event='planned',**r))+'\n');journal.flush()
  p.unlink()
  if r['action']=='link_final':p.symlink_to(os.path.relpath(r['target'],p.parent))
  freed+=r['size'];counts[r['action']]+=1
  journal.write(json.dumps(dict(event='completed',**r))+'\n');journal.flush()
 for r in plan['compress_backups']:
  p=M/r['path'];dest=p.with_name(p.name+'.gz');assert not dest.exists()
  with p.open('rb') as fi,gzip.open(dest,'wb',compresslevel=9) as fo:
   for b in iter(lambda:fi.read(1024*1024),b''):fo.write(b)
  with gzip.open(dest,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==r['sha256']
  p.unlink();delta=r['size']-dest.stat().st_size;freed+=delta
  journal.write(json.dumps(dict(event='compressed_backup',compressed=str(dest),saved_bytes=delta,**r))+'\n');journal.flush()
# Remove empty task directories, never symlink targets.
for folder,dirs,names in os.walk(A,topdown=False,followlinks=False):
 p=Path(folder)
 if p!=A and not p.is_symlink() and not any(p.iterdir()):p.rmdir()
for p,h in plan['protected'].items():assert sha(Path(p))==h,p
summary=dict(status='cleaned',archive=str(A),compatibility_symlink=str(W),actions=counts,bytes_reclaimed=freed,remaining_work_regular_file_bytes=sum(p.stat().st_size for p in files(A)),protected_files_verified=len(plan['protected']),final_packages_unchanged=True)
(C/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
