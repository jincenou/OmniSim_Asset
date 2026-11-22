"""Population contact experiment; no production writes or implicit adoption."""
import argparse
import asyncio
import collections
import json
import os
from pathlib import Path
import sys
import time
import subprocess

import numpy as np
from scipy.optimize import lsq_linear, minimize_scalar

from motion_experiment import LAB, ASSETS, inventory, frozen_plan
from common import read, save, sha
from playback import compose
from placement import transform_points, vertical_sigma

OUT = LAB / 'action_split'
RULES = dict(version=1, stage='Action-specific constant-ground and fixed-clock root correction',
    geometry='Dynamic 32-neighbor region; even ranks fit, odd ranks evaluate. Shared source-conditioned region is not independent semantic truth.',
    contact_height_band_model_m=.04, contact_vertical_speed_model_mps=.20,
    maximum_region_distance_model_m=.12, minimum_contact_edges=12,
    ground_bound_model_m=.10, penetration_weight=4., root_axis_bound_model_m=.10,
    root_norm_limit_model_m=.15, velocity_scale_model_mps=.30,
    displacement_scale_model_m=.05, displacement_weight=.02,
    acceleration_scale_model_mps2=2., acceleration_weight=.5,
    fixed_endpoints=True, unchanged_clock=True, loopable=False,
    fairness='Same native-speed straight reference path and elapsed times. Candidate may shift locally by bounded translation; not exact path following.',
    scope='New engineering choices frozen before this stage; proxy metrics only, no production adoption.')


def freeze():
    p=OUT/'CONTACT_PLAN.json'
    if p.exists():
        assert read(p)==RULES
        assert sha(p)==(OUT/'CONTACT_PLAN.sha256').read_text().strip()
    else:
        save(p,RULES);(OUT/'CONTACT_PLAN.sha256').write_text(sha(p)+'\n')
    save(OUT/'CONTACT_CODE.json',dict(code_sha256=sha(Path(__file__)),plan_sha256=sha(p)))


def load_arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def root_delta(times, feet, support):
    """Least-squares translation on fixed elapsed clock, zero at both endpoints."""
    n=len(times);dt=np.diff(times)
    edges=support[:-1]&support[1:]
    if edges.sum()<RULES['minimum_contact_edges']:
        raise ValueError('Insufficient consecutive image-supported contact hypotheses')
    D=np.diff(np.eye(n),axis=0)/dt[:,None]
    D2=np.diff(D,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None]
    velocity=np.diff(feet[:,:,:2],axis=0)/dt[:,None,None]
    rows,side=np.nonzero(edges)
    A=np.vstack([D[rows]/RULES['velocity_scale_model_mps'],
        np.eye(n)*np.sqrt(RULES['displacement_weight'])/RULES['displacement_scale_model_m'],
        D2*np.sqrt(RULES['acceleration_weight'])/RULES['acceleration_scale_model_mps2']])[:,1:-1]
    b=np.vstack([-velocity[rows,side]/RULES['velocity_scale_model_mps'],np.zeros((n+n-2,2))])
    solutions=[]
    for axis in (0,1):
        fit=lsq_linear(A,b[:,axis],bounds=(-RULES['root_axis_bound_model_m'],RULES['root_axis_bound_model_m']),tol=1e-9,max_iter=200)
        if not fit.success:raise ValueError('Bounded root fit did not converge')
        solutions.append(np.r_[0.,fit.x,0.])
    return np.stack(solutions,axis=1),edges


def stats(times,points,height,support):
    edges=support[:-1]&support[1:]
    speed=np.linalg.norm(np.diff(points[:,:,:2],axis=0)/np.diff(times)[:,None,None],axis=-1)
    return dict(contact_frames=int(support.sum()),contact_edges=int(edges.sum()),
        region_slip_p95_model_mps=float(np.quantile(speed[edges],.95)) if edges.any() else None,
        penetration_p95_model_m=float(np.quantile(np.maximum(-height[support],0),.95)) if support.any() else None,
        floating_p95_model_m=float(np.quantile(np.maximum(height[support],0),.95)) if support.any() else None)


def solve(identity):
    assert read(OUT/'CONTACT_PLAN.json')==RULES
    assert sha(OUT/'CONTACT_PLAN.json')==(OUT/'CONTACT_PLAN.sha256').read_text().strip()
    spec=next(s for s in frozen_plan()['cohort'] if s['id']==identity)
    package=ASSETS/identity;before=inventory(package);start=time.monotonic()
    result=dict(identity=identity,stratum=spec['stratum'],state='unavailable',production_adoption=False,
        contact_verified=False,loopable=False,plan_sha256=sha(OUT/'CONTACT_PLAN.json'))
    try:
        p=load_arrays(package/'placement_playback.npz');t=p['frame_timestamps'];n=len(t)
        walking=spec['stratum']=='walking';speed=float(p['native_speed_model_mps']) if walking else 0.
        if walking and (not bool(p['speed_available']) or speed<=.02):raise ValueError('No usable reference trajectory speed')
        world=np.zeros((n,3));world[:,0]=speed*(t-t[0])
        baseline=compose(t,p['root_xy'],p['root_yaw'],np.eye(4),p['ground_offset'],t,world,np.zeros(n))[0]
        candidates=load_arrays(LAB/'reports/foot_correspondence'/f'{identity}.npz')
        # Only read required payload arrays; no duplicate payload saved.
        with np.load(package/'asset_4dgs.npz',allow_pickle=False) as z:
            xyz=z['frame_xyz'];quat=z['frame_rotation'];scales=z['frame_scaling']
        fit_points=[];eval_points=[];fit_h=[];eval_h=[];distance=[]
        for side in ('left','right'):
            ids=candidates[side+'_dynamic_indices'];w=candidates[side+'_dynamic_weights']
            raw=xyz[np.arange(n)[:,None],ids]
            points=transform_points(raw,p['asset_to_standard'])
            lower=points[:,:,2]-2*vertical_sigma(quat[np.arange(n)[:,None],ids],scales[np.arange(n)[:,None],ids],p['asset_to_standard'][2,:3])
            groups=[]
            for parity in (0,1):
                weights=w[:,parity::2];weights=weights/weights.sum(1,keepdims=True)
                centroid=np.sum(points[:,parity::2]*weights[:,:,None],axis=1)
                # Lower envelope proxy avoids treating a tracked shoe upper as a sole.
                groups.append((centroid,np.quantile(lower[:,parity::2],.2,axis=1)))
            fit_points.append(groups[0][0]);fit_h.append(groups[0][1])
            eval_points.append(groups[1][0]);eval_h.append(groups[1][1])
            distance.append(np.linalg.norm(groups[0][0]-candidates[side+'_target'],axis=1))
        del xyz,quat,scales
        fp=np.stack(fit_points,1);ep=np.stack(eval_points,1)
        fh=np.stack(fit_h,1);eh=np.stack(eval_h,1)
        def place(x):return np.einsum('fij,fsj->fsi',baseline[:,:3,:3],x)+baseline[:,None,:3,3]
        fw=place(fp);ew=place(ep);ground=float(p['ground_offset'])
        evidence=load_arrays(OUT/f'{identity}_walking_evidence.npz')['any_view_image_agreement'] if walking else np.ones((n,2),bool)
        support=(evidence & (np.stack(distance,1)<=RULES['maximum_region_distance_model_m'])
            & (fh<=np.quantile(fh,.2,axis=0)[None]+RULES['contact_height_band_model_m'])
            & (np.abs(np.gradient(fp[:,:,2],t,axis=0))<=RULES['contact_vertical_speed_model_mps']))
        if support.sum()<8:raise ValueError('Fewer than eight geometry/image-supported contact samples')
        observed=fh[support]+ground
        def cost(d):
            h=observed+d;return np.mean(np.maximum(h,0)**2+RULES['penetration_weight']*np.minimum(h,0)**2)
        g=minimize_scalar(cost,bounds=(-RULES['ground_bound_model_m'],RULES['ground_bound_model_m']),method='bounded')
        if not g.success:raise ValueError('Ground fit failed')
        delta,edges=root_delta(t,fw,support) if walking else (np.zeros((n,2)),support[:-1]&support[1:])
        candidate=baseline.copy();candidate[:,:2,3]+=delta;candidate[:,2,3]+=g.x
        ewc=ew.copy();ewc[:,:,:2]+=delta[:,None];ewc[:,:,2]+=g.x
        before_metrics=stats(t,ew,eh+ground,support)
        after_metrics=stats(t,ewc,eh+ground+g.x,support)
        np.savez_compressed(OUT/f'{identity}_contact_candidate.npz',times=t,sample_frames=np.arange(n,dtype=float),
            baseline_world_from_standard=baseline,candidate_world_from_standard=candidate,
            contact_hypothesis=support,evaluation_region=ep,evaluation_lower_envelope=eh,
            fitted_root_delta=delta,ground_delta=np.array(g.x))
        result.update(state='proxy_evaluated_runtime_pending',baseline=before_metrics,candidate=after_metrics,
            ground_delta_model_m=float(g.x),root_delta_norm_max_model_m=float(np.linalg.norm(delta,axis=1).max()),
            root_bounds_pass=bool(np.linalg.norm(delta,axis=1).max()<=RULES['root_norm_limit_model_m']),
            unchanged_clock=True,contact_sample_fraction=float(support.mean()),
            scope='Disjoint neighbor ranks but shared source-guided region; region velocity is not tracked material velocity. No independent semantic/contact or visual acceptance.',
            asset_sha256=read(package/'package_complete.json')['asset_sha256'])
        assert np.array_equal(candidate[:,:3,:3],baseline[:,:3,:3])
        if not walking:assert np.array_equal(candidate[:,:2,3],baseline[:,:2,3])
    except Exception as e:result['reason']=f'{type(e).__name__}: {e}'
    finally:
        result['source_inventory_unchanged']=inventory(package)==before
        result['seconds']=time.monotonic()-start
        save(OUT/f'{identity}_contact_result.json',result)
        if not result['source_inventory_unchanged']:raise RuntimeError('Published source changed')
    return result


async def run():
    freeze();limit=asyncio.Semaphore(2)
    async def one(spec):
        async with limit:
            identity=spec['id'];log=LAB/'logs'/f'{identity}_contact.log'
            with log.open('w') as stream:
                proc=await asyncio.create_subprocess_exec(sys.executable,__file__,'solve',identity,stdout=stream,stderr=asyncio.subprocess.STDOUT)
                code=await proc.wait()
            return dict(identity=identity,exit_code=code)
    jobs=await asyncio.gather(*(one(s) for s in frozen_plan()['cohort']))
    results=[read(OUT/f'{s["id"]}_contact_result.json') for s in frozen_plan()['cohort']]
    save(OUT/'CONTACT_RESULTS.json',dict(updated_at=time.time(),jobs=jobs,results=results,
        states=dict(collections.Counter(r['state'] for r in results)),production_adoption=False))


def replay(identity):
    import torch
    from PIL import Image, ImageDraw
    from assets import load_asset, raw_render, camera, write_video, quaternion_to_matrix, gpu_identity
    from placement import camera_in_native_frame
    package=ASSETS/identity;before=inventory(package)
    result=read(OUT/f'{identity}_contact_result.json');assert result['state']=='proxy_evaluated_runtime_pending'
    assert sha(package/'asset_4dgs.npz')==result['asset_sha256']
    a=load_arrays(OUT/f'{identity}_contact_candidate.npz');p=load_arrays(package/'placement_playback.npz')
    with np.load(package/'asset_4dgs.npz',allow_pickle=False) as z:
        native_to_asset=z['asset_from_native'];baked=z['frame_xyz']
    t=a['times'];n=len(t);T=p['asset_to_standard']
    model,deform,pipe,_=load_asset(package/'runtime',53000)
    speed=float(p['native_speed_model_mps']) if result['stratum']=='walking' else 0.
    length=speed*(t[-1]-t[0]);center=length*.5;distance=max(4.,length*.85+2.)
    width=384;height=384;K=np.array([[380.,0,191.5],[0,380.,191.5],[0,0,1.]])
    target=np.array([center,0,.85]);views=[];floors=[]
    yy,xx=np.mgrid[:height,:width];rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],axis=-1)
    for eye in (np.array([center,-distance,2.]),np.array([center+distance*.6,-distance,2.2])):
        forward=target-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        c2w=np.eye(4);c2w[:3,:3]=np.stack([right,np.cross(forward,right),forward],1);c2w[:3,3]=eye;views.append(c2w)
        rays_world=rays@c2w[:3,:3].T;hit=-eye[2]/rays_world[:,:,2];xy=eye[:2]+rays_world[:,:,:2]*hit[:,:,None]
        floor=np.repeat(np.where(((np.floor(xy[:,:,0]/.25)+np.floor(xy[:,:,1]/.25))%2)[:,:,None]>0,.66,.44),3,2)
        floor[hit<=0]=.82;floors.append(floor)
    errors=[];begin=time.monotonic()
    @torch.no_grad()
    def frames():
        for frame in range(n):
            warp=deform.step(model.get_xyz.detach(),model.get_xyz.new_tensor([[frame/(n-1)]]).expand(len(model.get_xyz),-1))
            xyz=((quaternion_to_matrix(warp['d_xyz'][:,:4])@model.get_xyz[...,None]).squeeze(-1)+warp['d_xyz'][:,4:]).cpu().numpy()
            error=float(np.max(np.abs(transform_points(xyz,native_to_asset)-baked[frame])))
            errors.append(error)
            if error>1e-4:raise ValueError('Packaged runtime differs from baked frame positions')
            rows=[]
            for branch in ('baseline','candidate'):
                W=a[branch+'_world_from_standard'][frame];panels=[]
                for v,c2w in enumerate(views):
                    native_cam,_=camera_in_native_frame(c2w,W,T,native_to_asset)
                    c=dict(width=width,height=height,frame_count=n,time_denominator=n-1,cameras=[dict(view=v,K=K.tolist(),camtoworld=native_cam.tolist())])
                    rgb,alpha=raw_render(model,deform,pipe,camera(c,v,float(frame)))
                    rgb=rgb.permute(1,2,0).cpu().numpy();alpha=alpha.permute(1,2,0).cpu().numpy()
                    assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
                    panels.append(np.clip((rgb+floors[v]*(1-alpha))*255,0,255).astype(np.uint8))
                im=Image.fromarray(np.concatenate(panels,1));ImageDraw.Draw(im).text((8,8),f'{identity} {branch} frame {frame} | experimental',fill='white',stroke_width=1,stroke_fill='black');rows.append(np.asarray(im))
            yield np.concatenate(rows,0)
    video=OUT/f'{identity}_comparison.mp4'
    write_video(video,frames(),fps=1/float(np.median(np.diff(t))))
    import cv2
    cap=cv2.VideoCapture(str(video));decoded=0
    while True:
        ok,decoded_frame=cap.read()
        if not ok:break
        assert decoded_frame.shape==(height*2,width*2,3)
        decoded+=1
    cap.release()
    assert decoded==n
    assert inventory(package)==before and sha(package/'asset_4dgs.npz')==result['asset_sha256']
    save(OUT/f'{identity}_replay.json',dict(identity=identity,frames=decoded,runtime_baked_xyz_max_error=max(errors),
        device=gpu_identity(),seconds=time.monotonic()-begin,source_inventory_unchanged=True,
        scope='Fixed-clock packaged runtime, two views. Floor is background only, not a depth occluder. Visual quality still unreviewed.',production_adoption=False))


async def replay_all():
    from common import environment, PYTHONS
    queue=asyncio.Queue();reserved=set();lock=asyncio.Lock();jobs=[]
    for s in frozen_plan()['cohort']:
        if read(OUT/f'{s["id"]}_contact_result.json')['state']=='proxy_evaluated_runtime_pending' and not (OUT/f'{s["id"]}_replay.json').exists():queue.put_nowait(s['id'])
    async def worker():
        while not queue.empty():
            try:identity=queue.get_nowait()
            except asyncio.QueueEmpty:return
            gpu=None
            while gpu is None:
                async with lock:
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                    options=[tuple(map(int,l.split(','))) for l in raw.strip().splitlines()]
                    options=sorted((x for x in options if x[0] not in reserved and x[1]>=10240),key=lambda x:-x[1])
                    if options:gpu=options[0][0];reserved.add(gpu)
                if gpu is None:await asyncio.sleep(30)
            try:
                logpath=LAB/'logs'/f'{identity}_contact_replay.log'
                jobpath=OUT/f'{identity}_replay_job.json'
                if jobpath.exists():
                    previous=read(jobpath)
                    if previous['state']=='running':raise RuntimeError('Refusing duplicate replay with unresolved running state')
                    save(LAB/'logs'/f'{identity}_contact_replay_previous_{time.time_ns()}.json',previous)
                if logpath.exists():logpath.rename(logpath.with_name(logpath.stem+f'_previous_{time.time_ns()}.log'))
                with logpath.open('w') as stream:
                    proc=await asyncio.create_subprocess_exec(str(PYTHONS['grid4d']),__file__,'replay',identity,env=environment(gpu,'grid4d'),stdout=stream,stderr=asyncio.subprocess.STDOUT)
                    save(OUT/f'{identity}_replay_job.json',dict(state='running',pid=proc.pid,gpu=gpu))
                    code=await proc.wait()
                state=dict(identity=identity,state='complete' if code==0 else 'failed',exit_code=code,gpu=gpu)
                save(OUT/f'{identity}_replay_job.json',state);jobs.append(state)
            finally:reserved.remove(gpu);queue.task_done()
    await asyncio.gather(worker(),worker())
    save(OUT/'REPLAY_QUEUE.json',dict(jobs=jobs,finished_at=time.time()))


def material_audit(identity):
    """Post-hoc diagnosis on identical, non-fitted Gaussian IDs across an edge."""
    package=ASSETS/identity;before=inventory(package)
    a=load_arrays(OUT/f'{identity}_contact_candidate.npz');p=load_arrays(package/'placement_playback.npz')
    neighbors=load_arrays(LAB/'reports/foot_correspondence'/f'{identity}.npz')
    with np.load(package/'asset_4dgs.npz',allow_pickle=False) as z:xyz=z['frame_xyz']
    points=transform_points(xyz,p['asset_to_standard']);t=a['times'];records={}
    for foot,side in enumerate(('left','right')):
        ids=neighbors[side+'_dynamic_indices'];mask=a['contact_hypothesis'][:,foot]
        edges=np.flatnonzero(mask[:-1]&mask[1:]);samples={'baseline':[],'candidate':[]};counts=[];accepted=[]
        for f in edges:
            # Odd ranks in BOTH frames, never used to fit either end of this edge.
            common=np.intersect1d(ids[f,1::2],ids[f+1,1::2])
            if len(common)<4:continue
            accepted.append(int(f));counts.append(len(common))
            for branch in samples:
                W=a[branch+'_world_from_standard'];start=transform_points(points[f,common],W[f]);end=transform_points(points[f+1,common],W[f+1])
                samples[branch].extend(np.linalg.norm((end-start)[:,:2],axis=1)/(t[f+1]-t[f]))
        records[side]=dict(hypothesis_edges=len(edges),evaluated_edges=len(accepted),point_edge_count=sum(counts),
            edge_ids=accepted,baseline_p95_model_mps=float(np.quantile(samples['baseline'],.95)) if counts else None,
            candidate_p95_model_mps=float(np.quantile(samples['candidate'],.95)) if counts else None)
    assert inventory(package)==before
    result=dict(identity=identity,feet=records,source_inventory_unchanged=True,
        scope='Post-hoc identical-ID edge velocity, held out from both endpoint fits; minimum four shared odd-rank points. Still no independent semantic foot/contact truth. No candidate chosen using this audit.')
    save(OUT/f'{identity}_material_audit.json',result)
    return result


async def audit_all():
    save(OUT/'MATERIAL_AUDIT_PLAN.json',dict(stage='Post-hoc correspondence-confound diagnosis, no optimization',
        minimum_shared_points=4,selection='Intersection of odd ranks at both consecutive contact frames',
        denominator='All candidate people; missing edges explicit',production_adoption=False))
    limiter=asyncio.Semaphore(2)
    async def one(s):
        async with limiter:
            with (LAB/'logs'/f'{s["id"]}_material.log').open('w') as stream:
                proc=await asyncio.create_subprocess_exec(sys.executable,__file__,'material',s['id'],stdout=stream,stderr=asyncio.subprocess.STDOUT)
                return dict(identity=s['id'],exit_code=await proc.wait())
    specs=[s for s in frozen_plan()['cohort'] if (OUT/f'{s["id"]}_contact_candidate.npz').exists() and not (OUT/f'{s["id"]}_material_audit.json').exists()]
    jobs=await asyncio.gather(*(one(s) for s in specs))
    save(OUT/'MATERIAL_AUDIT_QUEUE.json',dict(jobs=jobs,finished_at=time.time()))


def speed_diagnostic(identity):
    """Estimate a constant native translation from final-Gaussian stance hypotheses."""
    package=ASSETS/identity;before=inventory(package)
    a=load_arrays(OUT/f'{identity}_contact_candidate.npz');p=load_arrays(package/'placement_playback.npz')
    neighbors=load_arrays(LAB/'reports/foot_correspondence'/f'{identity}.npz')
    t=a['times'];n=len(t)
    W=compose(t,p['root_xy'],p['root_yaw'],np.eye(4),0.,t,np.zeros((n,3)),np.zeros(n))[0]
    with np.load(package/'asset_4dgs.npz',allow_pickle=False) as z:points=transform_points(z['frame_xyz'],p['asset_to_standard'])
    side_fit={};evaluation={};edge_counts={}
    for foot,side in enumerate(('left','right')):
        ids=neighbors[side+'_dynamic_indices'];mask=a['contact_hypothesis'][:,foot];edges=np.flatnonzero(mask[:-1]&mask[1:])
        parity_values=[];parity_edges=[]
        for parity in (0,1):
            values=[];count=0
            for f in edges:
                common=np.intersect1d(ids[f,parity::2],ids[f+1,parity::2])
                if len(common)<4:continue
                start=transform_points(points[f,common],W[f]);end=transform_points(points[f+1,common],W[f+1])
                # Equal weight per time edge, not proportional to surviving point count.
                values.append(np.median((end-start)[:,:2]/(t[f+1]-t[f]),axis=0));count+=1
            parity_values.append(np.asarray(values).reshape(-1,2));parity_edges.append(count)
        edge_counts[side]=dict(fit=parity_edges[0],evaluation=parity_edges[1])
        if len(parity_values[0]):side_fit[side]=-np.median(parity_values[0],axis=0)
        evaluation[side]=parity_values[1]
    if not side_fit:raise ValueError('No final-Gaussian stance edges for speed estimation')
    velocity=np.mean(list(side_fit.values()),axis=0)
    baseline=np.array([float(p['native_speed_model_mps']),0.]);metrics={}
    for side,values in evaluation.items():
        metrics[side]=dict(edges=len(values),baseline_p95_model_mps=float(np.quantile(np.linalg.norm(values+baseline,axis=1),.95)) if len(values) else None,
            calibrated_p95_model_mps=float(np.quantile(np.linalg.norm(values+velocity,axis=1),.95)) if len(values) else None)
    assert inventory(package)==before
    result=dict(identity=identity,baseline_velocity_model_mps=baseline.tolist(),estimated_velocity_model_mps=velocity.tolist(),
        estimated_speed_model_mps=float(np.linalg.norm(velocity)),side_estimates={k:v.tolist()for k,v in side_fit.items()},edge_counts=edge_counts,
        side_disagreement_model_mps=float(np.linalg.norm(side_fit['left']-side_fit['right'])) if len(side_fit)==2 else None,
        metrics=metrics,source_inventory_unchanged=True,production_adoption=False,
        scope='Post-hoc constant native-speed calibration, original clock and original ground. Changes path length/direction; not a same-planned-path comparison. Uses final Gaussian material IDs but contact and foot semantics remain hypotheses; no reliable speed or anti-slip certification.')
    save(OUT/f'{identity}_speed_diagnostic.json',result)
    return result


async def speed_all():
    save(OUT/'SPEED_DIAGNOSTIC_PLAN.json',dict(stage='Post-hoc native-speed diagnosis, not a new accepted recipe',
        fit='Even-rank shared Gaussian IDs on consecutive supported frames, median per edge then per foot, average available foot estimates',
        evaluate='Odd-rank shared IDs; same contact hypotheses, no independent semantic truth',minimum_common_points=4,
        changes='Constant planned translation vector only; no ground, heading, action-time, geometry or source-file changes',
        compare='Native stride calibration with different traveled distance/direction; not fixed-path anti-slip performance',production_adoption=False))
    limiter=asyncio.Semaphore(2)
    async def one(s):
        async with limiter:
            with (LAB/'logs'/f'{s["id"]}_speed_diagnostic.log').open('w') as stream:
                proc=await asyncio.create_subprocess_exec(sys.executable,__file__,'speed',s['id'],stdout=stream,stderr=asyncio.subprocess.STDOUT)
                return dict(identity=s['id'],exit_code=await proc.wait())
    specs=[s for s in frozen_plan()['cohort'] if s['stratum']=='walking' and (OUT/f'{s["id"]}_contact_candidate.npz').exists()]
    jobs=await asyncio.gather(*(one(s) for s in specs))
    save(OUT/'SPEED_DIAGNOSTIC_QUEUE.json',dict(jobs=jobs,finished_at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','solve','replay','replay_all','material','audit_all','speed','speed_all']);parser.add_argument('identity',nargs='?');args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':asyncio.run(run())
    elif args.action=='replay':replay(args.identity)
    elif args.action=='replay_all':asyncio.run(replay_all())
    elif args.action=='material':print(json.dumps(material_audit(args.identity)))
    elif args.action=='audit_all':asyncio.run(audit_all())
    elif args.action=='speed':print(json.dumps(speed_diagnostic(args.identity)))
    elif args.action=='speed_all':asyncio.run(speed_all())
    else:print(json.dumps(solve(args.identity),ensure_ascii=False))
