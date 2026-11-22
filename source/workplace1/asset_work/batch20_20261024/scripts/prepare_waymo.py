"""Frozen Waymo TFRecords -> rectified RGB frame indexes; labels remain evaluation-only."""
import argparse,asyncio,hashlib,json,os,struct,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def worker(scene):
 import cv2,numpy as np
 sys.path.insert(0,'/mnt/ojc/data/waymo_track_bbox_10/python_pkgs')
 from waymo_open_dataset import dataset_pb2
 cv2.setNumThreads(1)
 row=next(r for r in json.loads((ROOT/'registry.json').read_text())['waymo'] if r['scene_id']==scene)
 raw=Path(row['raw_tfrecord']);images=ROOT/'inputs/waymo_images'/scene;images.mkdir(parents=True,exist_ok=True)
 gtpath=ROOT/'evaluation/waymo_native'/f'{scene}.jsonl';gtpath.parent.mkdir(parents=True,exist_ok=True)
 names={1:'front',2:'front_left',3:'front_right',4:'side_left',5:'side_right'}
 native_from_cv=np.eye(4);native_from_cv[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]
 maps={};streams={c:[] for c in names};frames=[];started=time.monotonic();fi=0
 with raw.open('rb') as source,gtpath.open('w') as labels:
  while head:=source.read(12):
   if len(head)!=12:raise ValueError('Truncated TFRecord header')
   n=struct.unpack('<Q',head[:8])[0];payload=source.read(n)
   if len(payload)!=n or len(source.read(4))!=4:raise ValueError('Truncated TFRecord payload')
   fr=dataset_pb2.Frame();fr.ParseFromString(payload)
   assert {im.name for im in fr.images}==set(names),('missing camera',scene,fi)
   cals={c.name:c for c in fr.context.camera_calibrations}
   saved={};frame_rows={}
   for im in fr.images:
    c=cals[im.name];intr=np.asarray(c.intrinsic);k=np.array([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1.]])
    if im.name not in maps:
     maps[im.name]=cv2.initUndistortRectifyMap(k,intr[4:],None,k,(c.width,c.height),cv2.CV_32FC1)
    rgb=cv2.imdecode(np.frombuffer(im.image,np.uint8),cv2.IMREAD_COLOR)
    assert rgb.shape[:2]==(c.height,c.width)
    rect=cv2.remap(rgb,*maps[im.name],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    path=images/f'f{fi:04d}_c{im.name}.jpg'
    if not cv2.imwrite(str(path),rect,[cv2.IMWRITE_JPEG_QUALITY,95]):raise IOError(path)
    streams[im.name].append(str(path));saved[im.name]=str(path)
    cam_world=np.array(im.pose.transform).reshape(4,4)@np.array(c.extrinsic.transform).reshape(4,4)@native_from_cv
    if fi%5==0:
     item=dict(path=str(path),camera_name=names[im.name],frame_id=fi//5,native_frame_index=fi,
       timestamp_us=round(im.pose_timestamp*1e6),camera_intrinsic=k.tolist(),camera_to_world=cam_world.tolist(),
       motion_paths=streams[im.name][max(0,len(streams[im.name])-5):-1] if fi else [])
     frames.append(item);frame_rows[im.name]=item
   laser=[dict(id=l.id,type=int(l.type),box=[l.box.center_x,l.box.center_y,l.box.center_z,l.box.length,l.box.width,l.box.height,l.box.heading]) for l in fr.laser_labels]
   gt=dict(native_frame_index=fi,timestamp_us=fr.timestamp_micros,ego_to_world=list(fr.pose.transform),laser_labels=laser)
   if fi%5==0:
    cameras=[]
    for group in fr.camera_labels:
     cal=cals[group.name];intr=np.asarray(cal.intrinsic);k=np.array(frame_rows[group.name]['camera_intrinsic']);annotations=[]
     for l in group.labels:
      b=l.box;x0,x1=b.center_x-b.length/2,b.center_x+b.length/2;y0,y1=b.center_y-b.width/2,b.center_y+b.width/2
      t=np.linspace(0,1,9);points=np.concatenate([np.c_[x0+(x1-x0)*t,np.full(9,y0)],np.c_[x0+(x1-x0)*t,np.full(9,y1)],np.c_[np.full(9,x0),y0+(y1-y0)*t],np.c_[np.full(9,x1),y0+(y1-y0)*t]])
      points=cv2.undistortPoints(points.reshape(-1,1,2),k,intr[4:],P=k).reshape(-1,2)
      bbox=np.clip([*points.min(0),*points.max(0)],0,[cal.width,cal.height,cal.width,cal.height]).tolist()
      annotations.append(dict(camera_track_id=l.id,laser_object_id=l.association.laser_object_id or None,type=int(l.type),raw_bbox_xyxy=[x0,y0,x1,y1],bbox_xyxy=bbox))
     cameras.append(dict(camera_name=names[group.name],frame_index=fi//5,path=saved[group.name],image_wh=[cal.width,cal.height],annotations=annotations))
    gt['camera_annotations']=cameras
   labels.write(json.dumps(gt)+'\n');fi+=1
 if fi<2 or any(len(s)!=fi for s in streams.values()):raise ValueError('Incomplete scene')
 index=dict(scene_id=scene,dataset='waymo',source_tfrecord=str(raw),source_bytes=raw.stat().st_size,
   source_mtime_ns=raw.stat().st_mtime_ns,native_frames=fi,frames=frames,rectified=True,
   calibration='CameraImage.pose ego-to-world at image time @ camera-to-ego @ native-from-OpenCV; standard pinhole rectification',
   detector_rate_hz=2,native_rate_hz=10,annotation_use='Evaluation file only, not an input to detection/association/generation')
 write(ROOT/'inputs/waymo_indexes'/f'{scene}.json',index)
 result=dict(scene_id=scene,status='prepared',native_frames=fi,detector_images=len(frames),all_rgb_images=fi*5,seconds=time.monotonic()-started)
 write(ROOT/'inputs/waymo_preparation'/f'{scene}.json',result);print(json.dumps(result),flush=True)

async def main():
 rows=json.loads((ROOT/'registry.json').read_text())['waymo'];slots=asyncio.Semaphore(3)
 async def one(row):
  scene=row['scene_id'];status=ROOT/'inputs/waymo_preparation'/f'{scene}.json';index=ROOT/'inputs/waymo_indexes'/f'{scene}.json'
  if status.exists() and index.exists():
   saved=json.loads(index.read_text());raw=Path(row['raw_tfrecord']);done=json.loads(status.read_text())
   if (saved['source_bytes']==raw.stat().st_size and saved['source_mtime_ns']==raw.stat().st_mtime_ns and len(saved['frames'])==done['detector_images'] and all(Path(f['path']).is_file() and all(Path(x).is_file() for x in f['motion_paths']) for f in saved['frames'])):
    print(scene,'verified existing completion',flush=True)
    return dict(scene_id=scene,exit_code=0,reused=True)
   raise ValueError('Existing prepared inputs changed: '+scene)
  async with slots:
   log=ROOT/'logs'/f'prepare_{row["scene_id"]}.log'
   with log.open('wb') as f:
    p=await asyncio.create_subprocess_exec(sys.executable,__file__,'--scene',row['scene_id'],env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=asyncio.subprocess.STDOUT)
    code=await p.wait()
   print(row['scene_id'],code,flush=True)
   return dict(scene_id=row['scene_id'],exit_code=code,log=str(log))
 results=await asyncio.gather(*(one(row) for row in rows));write(ROOT/'inputs/waymo_preparation/jobs.json',results)
 if any(r['exit_code'] for r in results):raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene');a=p.parse_args()
 if a.scene:worker(a.scene)
 else:asyncio.run(main())
