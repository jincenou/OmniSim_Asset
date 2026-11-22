"""Bounded, resumable read-only native label audit for the 100-track queue."""
import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=Path('/mnt/public_data/waymo')
RAW=Path('/mnt/public_data/waymo_raw/validation')


def scan(job):
    sys.path.insert(0,'/mnt/ojc/data/waymo_track_bbox_10/python_pkgs')
    from waymo_open_dataset import dataset_pb2
    import numpy as np
    tracks=defaultdict(list);poses=[];timestamps=[]
    with Path(job['raw']).open('rb') as stream:
        i=0
        while True:
            header=stream.read(12)
            if not header:break
            if len(header)!=12:raise EOFError('Truncated TFRecord header')
            size=struct.unpack('<Q',header[:8])[0];payload=stream.read(size);crc=stream.read(4)
            if len(payload)!=size or len(crc)!=4:raise EOFError('Truncated TFRecord payload')
            frame=dataset_pb2.Frame();frame.ParseFromString(payload)
            assert frame.context.name==job['segment'].removeprefix('segment-').removesuffix('_with_camera_labels')
            poses.append(np.array(frame.pose.transform).reshape(4,4));timestamps.append(frame.timestamp_micros)
            for labels in frame.camera_labels:
                if labels.name not in (1,2,3):continue
                for label in labels.labels:
                    if label.type!=2:continue
                    b=label.box;x0=b.center_x-b.length/2;y0=b.center_y-b.width/2
                    assoc=label.association.laser_object_id if label.HasField('association') else ''
                    tracks[(labels.name-1,label.id,assoc)].append({'frame_index':i,'timestamp_micros':frame.timestamp_micros,
                        'bbox_xyxy':[x0,y0,x0+b.length,y0+b.width],'difficulty':int(label.detection_difficulty_level)})
            i+=1
    candidates=[];poses=np.stack(poses)
    source_pose=np.loadtxt(PUBLIC/'val'/job['scene']/'ego_to_world/000.txt')
    assert np.allclose(poses[0],source_pose,atol=1e-5),'Raw and processed scene pose mismatch'
    for (cam,track,assoc),obs in tracks.items():
        windows=[]
        for start in range(len(obs)-48):
            part=obs[start:start+49]
            if part[-1]['frame_index']-part[0]['frame_index']!=48:continue
            boxes=np.array([x['bbox_xyxy'] for x in part]);heights=boxes[:,3]-boxes[:,1]
            if heights.min()<80 or np.median(heights)<120:continue
            if boxes[:,0].min()<8 or boxes[:,1].min()<8 or boxes[:,2].max()>1912 or boxes[:,3].max()>1272:continue
            a=part[0]['frame_index'];b=part[-1]['frame_index']
            pos=poses[a:b+1,:3,3];path=float(np.linalg.norm(np.diff(pos,axis=0),axis=1).sum())
            rotation=poses[a:b+1,:3,:3]@poses[a,:3,:3].T
            angle=float(np.degrees(np.arccos(np.clip((np.trace(rotation,axis1=1,axis2=2)-1)/2,-1,1))).max())
            score=float(np.median(heights))/(1+10*path+angle)
            windows.append((score,part,path,angle,float(np.median(heights)),float(heights.min())))
        if not windows:continue
        score,part,path,angle,median,minimum=max(windows,key=lambda x:x[0])
        candidates.append({'split':'val','scene':job['scene'],'segment':job['segment'],'camera_index':cam,'track_id':track,
            'associated_lidar_track_id':assoc or None,'identity_key':job['segment']+':'+(assoc if assoc else f'cam{cam}:{track}'),
            'observations':part,'height_median_px':median,'height_min_px':minimum,'ego_path_m':path,'ego_rotation_deg':angle,
            'static_camera_candidate':path<=.25 and angle<=.5,'score':score,'source':'native camera_labels; no cross-camera identity guessed'})
    result={**job,'frames':i,'pedestrian_camera_tracks':len(tracks),'candidates':candidates}
    (ROOT/'track_scan'/f"{job['scene']}.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'scene':job['scene'],'candidates':len(candidates),'static':sum(x['static_camera_candidate'] for x in candidates)}),flush=True)


async def main(limit):
    directory=ROOT/'track_scan';directory.mkdir(exist_ok=True)
    jobs=[]
    for p in sorted((PUBLIC/'annotations/val').glob('*.json')):
        raw=RAW/(p.stem+'.tfrecord')
        if not raw.exists():continue
        d=json.loads(p.read_text());scene=f"{d['scene_id']:03d}"
        if not (PUBLIC/'val'/scene/'images/000_0.jpg').exists():continue
        jobs.append({'scene':scene,'segment':p.stem,'raw':str(raw)})
    jobs=sorted(jobs,key=lambda x:int(x['scene']))[:limit]
    (directory/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    sem=asyncio.Semaphore(2)
    async def run(job):
        async with sem:
            if (directory/f"{job['scene']}.json").exists():return
            p=await asyncio.create_subprocess_exec(sys.executable,__file__,'--job',json.dumps(job))
            if await p.wait():raise RuntimeError('Native label extraction failed: '+job['scene'])
    await asyncio.gather(*map(run,jobs))
    candidates=[c for j in jobs for c in json.loads((directory/f"{j['scene']}.json").read_text())['candidates']]
    unique={}
    for c in sorted(candidates,key=lambda c:c['score'],reverse=True):unique.setdefault(c['identity_key'],c)
    ranked=list(unique.values());selected=[c for c in ranked if c['static_camera_candidate']][:100]
    (ROOT/'candidate_queue.json').write_text(json.dumps({'target':100,'scenes_scanned':len(jobs),'unique_candidates':len(ranked),'static_candidates':sum(c['static_camera_candidate'] for c in ranked),'selected':selected,'ranked':ranked,'status':'candidate_audit_only_not_quality_acceptance'},indent=2)+'\n')
    print(json.dumps({'scenes':len(jobs),'unique':len(ranked),'selected':len(selected)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job');p.add_argument('--limit',type=int,default=40);a=p.parse_args()
    if a.job:scan(json.loads(a.job))
    else:asyncio.run(main(a.limit))
