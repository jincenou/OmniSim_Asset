"""Population short-context RGB tracking diagnostic; never write source assets."""
import argparse
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import motion_experiment as experiment  # installs published-asset write guard
from common import read, save, sha, environment, PYTHONS, gpu_identity
from placement import transform_points

ROOT=experiment.LAB


def policy():
    assert sha(ROOT/'TRACKING_PLAN.json')==(ROOT/'TRACKING_PLAN.sha256').read_text().split()[0]
    return read(ROOT/'TRACKING_PLAN.json')


def starts_for(n,p):
    starts=list(range(0,n-p['minimum_window_frames']+1,p['stride_frames']))
    tail=max(0,n-p['window_frames'])
    if tail not in starts and (not starts or starts[-1]+p['window_frames']<n):starts.append(tail)
    return sorted(starts)


def seed_at(frame, point, p):
    """At most six-pixel shift onto foreground, never force an off-image query."""
    h,w=frame.shape[:2];x,y=point
    if not np.isfinite(point).all() or not (0<=x<w and 0<=y<h):return None
    radius=p['seed_foreground_radius_px'];cx,cy=int(round(x)),int(round(y))
    y0,y1=max(0,cy-radius),min(h,cy+radius+1);x0,x1=max(0,cx-radius),min(w,cx+radius+1)
    background=np.median(np.concatenate([frame[:8].reshape(-1,3),frame[-8:].reshape(-1,3)]),axis=0)
    foreground=np.linalg.norm(frame[y0:y1,x0:x1].astype(float)-background,axis=-1)>p['foreground_rgb_distance']
    yy,xx=np.where(foreground)
    if not len(xx):return None
    xy=np.stack([xx+x0,yy+y0],1);distance=np.linalg.norm(xy-point,axis=1);best=np.argmin(distance)
    return xy[best].astype(float) if distance[best]<=radius else None


def run_one(identity,view):
    import cv2
    import torch
    p=policy();package=experiment.ASSETS/identity;before=experiment.inventory(package)
    destination=ROOT/'reports/foot_windows';destination.mkdir(parents=True,exist_ok=True)
    output=destination/f'{identity}_v{view:02d}'
    begin=time.monotonic();device=gpu_identity()
    checkpoint=Path(p['checkpoint']);assert sha(checkpoint)==p['checkpoint_sha256']
    source=Path(__file__).parent/'vendor/cotracker';pin=read(source/'VENDORED_SOURCE.json')
    assert all(sha(source/name)==value for name,value in pin['files'].items())
    sys.path.insert(0,str(source));from cotracker.predictor import CoTrackerPredictor
    with np.load(package/'placement_motion.npz',allow_pickle=False) as z:T=z['standard_model_meters_to_asset']
    with np.load(ROOT/'reports/foot_correspondence'/f'{identity}.npz',allow_pickle=False) as z:
        feet=np.stack([z[side+'_dynamic_proxy'] for side in ('left','right')],1)
    cameras=read(package/'cameras.json');cam=next(x for x in cameras['cameras'] if x['view']==view)
    xyz=transform_points(transform_points(feet,T),np.linalg.inv(cam['camtoworld']))
    projection=xyz@np.asarray(cam['K']).T;projection=projection[:,:,:2]/projection[:,:,2:]
    frames=[];capture=cv2.VideoCapture(str(package/'demo.mp4'))
    while True:
        ok,frame=capture.read()
        if not ok:break
        assert frame.shape==(672,1760,3)
        column=3 if view==0 else 4
        frames.append(cv2.cvtColor(frame[32:,column*352:(column+1)*352],cv2.COLOR_BGR2RGB))
    capture.release();assert len(frames)==len(feet)
    frames=np.stack(frames);n=len(frames);starts=starts_for(n,p);queries=[];mapping=[];rejected=[]
    for start in starts:
        end=min(n,start+p['window_frames'])
        separated=np.linalg.norm(projection[start,0]-projection[start,1])>=p['minimum_projected_foot_separation_px']
        for side in range(2):
            seed=seed_at(frames[start],projection[start,side],p) if separated and xyz[start,side,2]>0 else None
            if seed is None:
                rejected.append(dict(start=start,side=side,reason='off_image_background_or_projected_feet_overlap'));continue
            queries.append([start,*seed]);mapping.append((start,end,side))
    common=dict(identity=identity,view=view,frames=n,policy_sha256=sha(ROOT/'TRACKING_PLAN.json'),
                demo_sha256=sha(package/'demo.mp4'),source_commit=pin['commit'],checkpoint_sha256=p['checkpoint_sha256'],
                attempted_seed_count=2*len(starts),accepted_seed_count=len(queries),rejected_seeds=rejected,
                device=device,scope='Context ablation, not semantic tracking/contact truth')
    if not queries:
        assert experiment.inventory(package)==before
        save(output.with_suffix('.json'),dict(**common,state='unavailable_no_safe_query',source_unchanged=True));return
    model=CoTrackerPredictor(checkpoint=str(checkpoint),offline=True,v2=False).cuda().eval()
    tensor=torch.from_numpy(frames).permute(0,3,1,2)[None].float().cuda()
    q=torch.as_tensor(np.asarray(queries,np.float32)[None],device='cuda')
    torch.cuda.reset_peak_memory_stats()
    arrays={};records=[]
    with torch.inference_mode():
        long,lv=model(tensor,queries=q,backward_tracking=False)
        reverse_queries=torch.cat([torch.zeros_like(long[:,-1,:,:1]),long[:,-1]],-1)
        lr,lrv=model(tensor.flip(1),queries=reverse_queries,backward_tracking=False)
        long=long[0].cpu().numpy();lv=lv[0].cpu().numpy();lr=lr[0].flip(0).cpu().numpy();lrv=lrv[0].flip(0).cpu().numpy()
        arrays.update(long_tracks=long,long_visible=lv,long_reverse=lr,long_reverse_visible=lrv,queries=np.asarray(queries),geometry_projection=projection)
        for start in starts:
            ids=[j for j,item in enumerate(mapping) if item[0]==start]
            if not ids:continue
            end=mapping[ids[0]][1];clip=tensor[:,start:end];local=q[:,ids].clone();local[:,:,0]=0
            short,sv=model(clip,queries=local,backward_tracking=False)
            reverse_queries=torch.cat([torch.zeros_like(short[:,-1,:,:1]),short[:,-1]],-1)
            sr,srv=model(clip.flip(1),queries=reverse_queries,backward_tracking=False)
            short=short[0].cpu().numpy();sv=sv[0].cpu().numpy();sr=sr[0].flip(0).cpu().numpy();srv=srv[0].flip(0).cpu().numpy()
            arrays[f'w{start}_tracks']=short;arrays[f'w{start}_reverse']=sr
            arrays[f'w{start}_visible']=sv;arrays[f'w{start}_reverse_visible']=srv;arrays[f'w{start}_query_ids']=np.array(ids)
            for k,j in enumerate(ids):
                side=mapping[j][2]
                # Same internal slots; both query and reverse endpoint excluded.
                for frame in range(start+1,end-1):
                    local_index=frame-start
                    for branch,xy,back,visible in (
                        ('long',long[frame,j],lr[frame,j],bool(lv[frame,j] and lrv[frame,j])),
                        ('short',short[local_index,k],sr[local_index,k],bool(sv[local_index,k] and srv[local_index,k]))):
                        cycle=float(np.linalg.norm(xy-back));own=float(np.linalg.norm(xy-projection[frame,side]));other=float(np.linalg.norm(xy-projection[frame,1-side]))
                        risk=other+p['other_foot_advantage_px']<own and own>p['own_foot_distance_px']
                        records.append(dict(branch=branch,start=start,frame=frame,side=side,visible=visible,cycle_px=cycle,
                            geometry_error_px=own,other_foot_risk=bool(risk),usable_diagnostic=bool(visible and cycle<=p['cycle_threshold_px']
                            and own<=p['near_geometry_threshold_px'] and not risk)))
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    summary={}
    for branch in ('long','short'):
        selected=[x for x in records if x['branch']==branch];visible=[x for x in selected if x['visible']]
        summary[branch]=dict(observations=len(selected),bidirectional_visible_fraction=sum(x['visible'] for x in selected)/len(selected),
            usable_diagnostic_fraction=sum(x['usable_diagnostic'] for x in selected)/len(selected),
            cycle_p95_px=float(np.quantile([x['cycle_px'] for x in visible],.95)) if visible else None,
            geometry_error_p95_px=float(np.quantile([x['geometry_error_px'] for x in visible],.95)) if visible else None,
            other_foot_risk_fraction=sum(x['other_foot_risk'] for x in visible)/len(visible) if visible else None)
    assert experiment.inventory(package)==before
    save(output.with_suffix('.json'),dict(**common,state='tracking_context_comparison_complete',summary=summary,
        observations=records,source_unchanged=True,seconds=time.monotonic()-begin,peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,
        limits=['Foreground seed is not depth-tested visibility','Both contexts share a geometry-derived initial query',
                'Dynamic geometry is condition-seeded, not independent semantic truth','Window-local coverage may double-count overlapping slots; denominators are reported observations',
                'No triangulation or speed/contact accepted, no new videos or asset updates']))


async def run_all():
    p=policy();queue=asyncio.Queue();reserved=set();lock=asyncio.Lock()
    for item in experiment.frozen_plan()['cohort']:
        for view in p['views']:queue.put_nowait((item['id'],view))
    async def worker():
        while not queue.empty():
            try:identity,view=queue.get_nowait()
            except asyncio.QueueEmpty:return
            gpu=None
            while gpu is None:
                async with lock:
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                    choices=[tuple(map(int,line.split(','))) for line in raw.strip().splitlines()]
                    choices=sorted((x for x in choices if x[0] not in reserved and x[1]>=p['gpu_min_free_mib']),key=lambda x:-x[1])
                    if choices:gpu=choices[0][0];reserved.add(gpu)
                if gpu is None:await asyncio.sleep(30)
            key=f'{identity}_v{view:02d}'
            try:
                with (ROOT/'logs'/f'{key}_windows.log').open('w') as log:
                    process=await asyncio.create_subprocess_exec(str(PYTHONS['4danyone']),__file__,'one',identity,str(view),
                        env=environment(gpu,'4danyone'),stdout=log,stderr=asyncio.subprocess.STDOUT)
                    save(ROOT/'results'/f'{key}_windows.json',dict(state='running',pid=process.pid,gpu=gpu,updated_at=time.time()))
                    code=await process.wait()
                save(ROOT/'results'/f'{key}_windows.json',dict(state='complete' if code==0 else 'failed',exit_code=code,gpu=gpu,updated_at=time.time()))
            finally:reserved.remove(gpu);queue.task_done()
    await asyncio.gather(*[worker() for _ in range(p['concurrency'])])
    save(ROOT/'reports/foot_windows_queue.json',dict(state='finished',jobs=48,updated_at=time.time(),source_assets_modified=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['one','run']);parser.add_argument('identity',nargs='?');parser.add_argument('view',nargs='?',type=int);args=parser.parse_args()
    if args.command=='one':run_one(args.identity,args.view)
    else:asyncio.run(run_all())
