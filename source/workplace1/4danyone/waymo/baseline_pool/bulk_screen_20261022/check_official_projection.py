"""Audit each selected window against its labelled auxiliary root; no merging."""
import asyncio,collections,csv,json,time,hashlib
from pathlib import Path
import cv2,numpy as np
cv2.setNumThreads(1)
B=Path(__file__).resolve().parent;P=B.parent;DATA=Path('/mnt/public_data/Waymo-3DSkelMo/waymo_open_dataset_3dskelmo/10fps');AX=np.eye(4);AX[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]
def audit(rr):
 out=[];cache={}
 for r in rr:
  c=json.loads((Path(r['package'])/'READY.json').read_text())['candidate'];obs=c['observations'];tid=c['associated_lidar_track_id'];seg=c['segment'].removeprefix('segment-').removesuffix('_with_camera_labels');npz=DATA/('training' if r['split']=='train' else 'validation')/seg/(tid+'.npz');entry={k:r[k] for k in ['split','scene','review_id','identity_key','camera','package']};entry.update(observations=len(obs),auxiliary_path=str(npz),quality_approved=False,physical_identity_verified=False)
  if not npz.exists():entry.update(state='auxiliary_missing',exact_time_fraction=None,inside_fraction=None,median_normalized_distance=None);out.append(entry);continue
  with np.load(npz,allow_pickle=False) as z:lookup={int(t):v for t,v in zip(z['waymo_timestamps'],z['trans'])}
  scene=Path('/mnt/public_data/waymo')/r['split']/r['scene'];cam=int(r['camera']);ck=('K',cam)
  if ck not in cache:
   v=np.loadtxt(scene/'intrinsics'/f'{cam}.txt').reshape(-1);cache[ck]=(v,np.array([[v[0],0,v[2]],[0,v[1],v[3]],[0,0,1]]))
  v,K=cache[ck];m=[]
  for o in obs:
   key=(o['frame_index'],cam)
   if key not in cache:cache[key]=np.linalg.inv(np.loadtxt(scene/'cam_to_world'/f"{o['frame_index']:03d}_{cam}.txt")@AX)
   m.append(cache[key])
  m=np.array(m);times=[int(o['timestamp_micros']) for o in obs];pos=np.array([lookup.get(t,[np.nan]*3) for t in times]);xyz=np.einsum('fij,fj->fi',m[:,:3,:3],pos)+m[:,:3,3];q,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,v[4:]);q=q.reshape(-1,2);valid=np.isfinite(q).all(1)&(xyz[:,2]>.1);boxes=np.array([o['bbox_xyxy'] for o in obs]);wh=boxes[:,2:]-boxes[:,:2];dist=np.linalg.norm((q-(boxes[:,:2]+wh*[.5,.6]))/np.maximum(wh,1),axis=1);inside=valid&(q>=boxes[:,:2]).all(1)&(q<=boxes[:,2:]).all(1);full=all(t in lookup for t in times);support=inside.mean()>=.8 and valid.sum()>=20 and np.median(np.where(valid,dist,1e6))<=.6
  entry.update(state='incomplete_time_support' if not full else 'root_projection_supported' if support else 'root_projection_discordant',exact_time_fraction=sum(t in lookup for t in times)/len(times),inside_fraction=float(inside.mean()),median_normalized_distance=float(np.median(dist[valid])) if valid.any() else None,auxiliary_sha256=hashlib.sha256(npz.read_bytes()).hexdigest());out.append(entry)
 return out
async def main():
 start=time.time();rows=list(csv.DictReader((B/'official_identity_candidates.csv').open()));groups=collections.defaultdict(list)
 for r in rows:groups[(r['split'],r['scene'])].append(r)
 sem=asyncio.Semaphore(2)
 async def task(rr):
  async with sem:return await asyncio.to_thread(audit,rr)
 results=await asyncio.gather(*(task(rr) for rr in groups.values()));out=[r for rr in results for r in rr];fields=list(dict.fromkeys(k for r in out for k in r))
 with (B/'official_projection_support.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
 summary=dict(at=time.time(),seconds=time.time()-start,windows=len(out),scenes=len(groups),states=dict(collections.Counter(r['state'] for r in out)),threshold_source='Existing IDENTITY_PROTOCOL.json inside>=0.8, median<=0.6, valid>=20; no threshold search; runner-up not checked',scope='Selected window exact timestamps and official auxiliary root only. Root containment support is not visual identity validation or generated quality. No associations updated.',automatic_merge=False,quality_people_count=None)
 (B/'OFFICIAL_PROJECTION_SUPPORT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':asyncio.run(main())
