"""Read-only native track cache and alternative-window search for batch100."""
import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path
import struct
import sys

import numpy as np

R=Path(__file__).resolve().parents[1]
OUT=R/'track_reserve'
PUBLIC=Path('/mnt/public_data/waymo')


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,separators=(',',':'))+'\n');temporary.replace(path)


def portrait_size(boxes):
    low=np.floor(boxes[:,:2].min(0)-24);high=np.ceil(boxes[:,2:].max(0)+24)
    size=((high-low+15)//16)*16
    unit=int(np.ceil(max(size[0]/11,size[1]/20)/2)*2)
    return np.array([11*unit,20*unit])


def extract(job):
    sys.path.insert(0,'/mnt/ojc/data/waymo_track_bbox_10/python_pkgs')
    from waymo_open_dataset import dataset_pb2
    raw=Path(job['raw']);stat=raw.stat();fingerprint={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    cache=OUT/'native'/f"{job['scene']}.json"
    if cache.exists():
        result=json.loads(cache.read_text())
        assert result['raw_stat']==fingerprint,'Source changed since cached scan'
        return result
    tracks=defaultdict(list);poses=[];timestamps=[]
    with raw.open('rb') as stream:
        while header:=stream.read(12):
            if len(header)!=12:raise EOFError('Truncated header')
            size=struct.unpack('<Q',header[:8])[0];payload=stream.read(size);crc=stream.read(4)
            if len(payload)!=size or len(crc)!=4:raise EOFError('Truncated TFRecord')
            frame=dataset_pb2.Frame();frame.ParseFromString(payload)
            assert frame.context.name==job['segment'].removeprefix('segment-').removesuffix('_with_camera_labels')
            index=len(poses);poses.append(np.array(frame.pose.transform).reshape(4,4).tolist());timestamps.append(frame.timestamp_micros)
            for camera in frame.camera_labels:
                if camera.name not in (1,2,3):continue
                for label in camera.labels:
                    if label.type!=2:continue
                    b=label.box;x=b.center_x-b.length/2;y=b.center_y-b.width/2
                    assoc=label.association.laser_object_id if label.HasField('association') else ''
                    tracks[(camera.name-1,label.id,assoc)].append({'frame_index':index,'timestamp_micros':frame.timestamp_micros,
                        'bbox_xyxy':[x,y,x+b.length,y+b.width],'difficulty':int(label.detection_difficulty_level)})
    assert np.allclose(poses[0],np.loadtxt(PUBLIC/'val'/job['scene']/'ego_to_world/000.txt'),atol=1e-5)
    assert np.all(np.diff(timestamps)>0)
    result={**job,'raw_stat':fingerprint,'poses':poses,'timestamps':timestamps,
            'tracks':[{'camera_index':cam,'track_id':track,'associated_lidar_track_id':assoc or None,'observations':obs}
                      for (cam,track,assoc),obs in tracks.items()]}
    save(cache,result);return result


def windows(job):
    raw=extract(job);poses=np.asarray(raw['poses']);options=[]
    for track in raw['tracks']:
        obs=track['observations'];eligible=[]
        for start in range(len(obs)-48):
            part=obs[start:start+49]
            if part[-1]['frame_index']-part[0]['frame_index']!=48:continue
            if not np.all(np.abs(np.diff([o['timestamp_micros'] for o in part])/1e6-.1)<.01):continue
            boxes=np.array([o['bbox_xyxy'] for o in part]);heights=boxes[:,3]-boxes[:,1]
            if heights.min()<80 or np.median(heights)<120:continue
            if boxes[:,0].min()<8 or boxes[:,1].min()<8 or boxes[:,2].max()>1912 or boxes[:,3].max()>1272:continue
            size=portrait_size(boxes)
            if np.any(size>np.array([1920,1280])):continue
            a=part[0]['frame_index'];b=part[-1]['frame_index']
            path=float(np.linalg.norm(np.diff(poses[a:b+1,:3,3],axis=0),axis=1).sum())
            rot=poses[a:b+1,:3,:3]@poses[a,:3,:3].T
            angle=float(np.degrees(np.arccos(np.clip((np.trace(rot,axis1=1,axis2=2)-1)/2,-1,1))).max())
            if path>.1 or angle>.5:continue
            center=(boxes[:,:2]+boxes[:,2:])/2
            relative_motion=float(np.ptp(center,axis=0).max()/np.median(heights))
            variation=float(np.ptp(heights)/np.median(heights))
            score=float(np.median(heights)/(1+10*path+angle+4*relative_motion+2*variation))
            identity=job['segment']+':'+(track['associated_lidar_track_id'] or f"cam{track['camera_index']}:{track['track_id']}")
            eligible.append({'split':'val','scene':job['scene'],'segment':job['segment'],**{k:v for k,v in track.items() if k!='observations'},
                'identity_key':identity,'observations':part,'height_median_px':float(np.median(heights)),
                'height_min_px':float(heights.min()),'ego_path_m':path,'ego_rotation_deg':angle,'static_camera_candidate':True,
                'score':score,'bbox_relative_motion':relative_motion,'height_relative_variation':variation,
                'portrait_dimensions':size.tolist(),'source':'native camera_labels; alternative 49-frame window, unchanged batch gates'})
        selected=[]
        for option in sorted(eligible,key=lambda r:r['score'],reverse=True):
            start=option['observations'][0]['frame_index']
            if any(abs(start-r['observations'][0]['frame_index'])<25 for r in selected):continue
            selected.append(option)
            if len(selected)==3:break
        options.extend(selected)
    save(OUT/'scenes'/f"{job['scene']}.json",{'scene':job['scene'],'options':options})
    print(json.dumps({'scene':job['scene'],'options':len(options)}),flush=True)


async def main():
    (OUT/'native').mkdir(parents=True,exist_ok=True);(OUT/'scenes').mkdir(exist_ok=True)
    jobs=json.loads((R/'track_scan/jobs.json').read_text());sem=asyncio.Semaphore(2)
    async def run(job):
        async with sem:
            if (OUT/'scenes'/f"{job['scene']}.json").exists():return
            p=await asyncio.create_subprocess_exec(sys.executable,__file__,'--job',json.dumps(job))
            if await p.wait():raise RuntimeError('Alternative window scan failed: '+job['scene'])
    outcomes=await asyncio.gather(*map(run,jobs),return_exceptions=True)
    failures=[str(r) for r in outcomes if isinstance(r,BaseException)]
    if failures:raise RuntimeError('; '.join(failures))
    options=[o for job in jobs for o in json.loads((OUT/'scenes'/f"{job['scene']}.json").read_text())['options']]
    identities=defaultdict(list)
    for option in sorted(options,key=lambda r:r['score'],reverse=True):identities[option['identity_key']].append(option)
    report={'status':'candidate_windows_only_not_assets','scenes':len(jobs),'distinct_identities':len(identities),
            'window_count':len(options),'identity_options':dict(identities),
            'scope':'Existing local validation data only; does not alter running batch or lower acceptance thresholds'}
    save(OUT/'reserve.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='identity_options'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job');args=p.parse_args()
    if args.job:windows(json.loads(args.job))
    else:asyncio.run(main())
