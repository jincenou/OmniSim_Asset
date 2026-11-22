"""Isolated population experiment. Published assets are read-only inputs."""
import argparse
import asyncio
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
import subprocess

LAB = Path(__file__).resolve().parent.parent
os.environ['HUMAN_RUN_ROOT'] = str(LAB)
ASSETS = Path('/mnt/ojc/workplace1/asset_library/human')
os.environ['HUMAN_ASSET_ROOT'] = str(ASSETS)
from common import read, save, sha
from playback import compose, validate_estimate
from placement import contact_playback_curve


def readonly_assets(event, args):
    """Deny Python-level mutation of any path resolving inside the asset library."""
    paths = []
    if event == 'open':
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+'))
        writing |= isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing:
            paths = [path]
    elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.truncate', 'os.utime'):
        paths = args[:1]
    elif event in ('os.rename', 'os.link', 'os.symlink'):
        paths = args[:2]
    for path in paths:
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).resolve()
            if resolved == ASSETS or ASSETS in resolved.parents:
                raise PermissionError(f'Experiment cannot mutate published asset: {resolved}')


sys.addaudithook(readonly_assets)


def frozen_plan():
    assert sha(LAB / 'PLAN.json') == (LAB / 'PLAN.sha256').read_text().split()[0]
    return read(LAB / 'PLAN.json')


def inventory(package):
    # Content hashes for small metadata; immutable payloads are hashed by renderer.
    return {str(p.relative_to(package)): dict(bytes=p.stat().st_size,
            mtime_ns=p.stat().st_mtime_ns,
            sha256=sha(p) if p.stat().st_size < 2**20 else None)
            for p in sorted(package.rglob('*')) if p.is_file()}


def regularized_clock(times, feet, support, speed, policy):
    """Fit one smooth positive speed curve, then invert its cumulative distance."""
    valid = support.any(axis=1)
    if valid.sum() < 8:
        raise ValueError('Fewer than eight contact-hypothesis frames')
    gaps = np.split(np.flatnonzero(~valid), np.flatnonzero(np.diff(np.flatnonzero(~valid)) > 1) + 1)
    dt = float(np.median(np.diff(times)))
    gap = max((len(g) * dt for g in gaps), default=0.)
    if gap > policy['maximum_interpolated_gap_seconds']:
        raise ValueError('Unsupported contact interval exceeds frozen limit')
    velocity = np.gradient(feet, times, axis=0)
    target = -(velocity[:, :, 0] * support).sum(1) / np.maximum(support.sum(1), 1)
    if np.median(target[valid]) <= .02:
        raise ValueError('No supported positive forward gait; do not force backward/turning clip')
    weights = np.sqrt(support.sum(1) / 2.)
    initial = np.clip(np.interp(times, times[valid], target[valid]) / speed, .5, 2.)

    def residual(value):
        return np.r_[weights * (value - target / speed),
                     np.sqrt(policy['velocity_smooth_weight']) * np.diff(value),
                     np.sqrt(policy['velocity_curvature_weight']) * np.diff(value, 2),
                     20. * (np.sum((value[:-1]+value[1:])*.5*np.diff(times)) / (times[-1]-times[0]) - 1.)]

    fit = least_squares(residual, initial, bounds=(.5, 2.), loss='soft_l1',
                        f_scale=policy['velocity_huber_scale_model_mps'] / speed,
                        max_nfev=100)
    if not fit.success:
        raise ValueError('Frozen bounded optimizer did not converge')
    v = fit.x * speed
    distance = np.r_[0., np.cumsum((v[:-1]+v[1:])*.5*np.diff(times))]
    query = np.interp(np.linspace(0., distance[-1], len(times)), distance, times)
    rate = np.gradient(query, times)
    accel = np.gradient(rate, times)
    return query, dict(rate_min=float(rate.min()), rate_max=float(rate.max()),
                       rate_acceleration_max=float(np.abs(accel).max()),
                       support_frames=int(valid.sum()), gap_seconds=gap,
                       optimizer_evaluations=int(fit.nfev),
                       bounds_pass=bool(rate.min() >= .5-1e-6 and rate.max() <= 2.+1e-6
                           and np.abs(accel).max() <= policy['maximum_retime_rate_acceleration_per_s']))


def solve(identity):
    begin = time.monotonic()
    plan = frozen_plan(); spec = next(x for x in plan['cohort'] if x['id'] == identity)
    policy = plan['policy']; package = ASSETS / identity
    destination = LAB / 'reports/placement_trials' / identity
    destination.mkdir(parents=True, exist_ok=True)
    before = inventory(package)
    save(destination/'source_inventory.json', before)
    result = dict(identity=identity, stratum=spec['stratum'], state='unavailable',
                  asset_unchanged=False, quality_approved=False, plan_sha256=sha(LAB/'PLAN.json'))
    try:
        validate_estimate(package)
        with np.load(package/'placement_motion.npz', allow_pickle=False) as z:
            motion = {k: z[k] for k in z.files}
        with np.load(package/'placement_playback.npz', allow_pickle=False) as z:
            practical = {k: z[k] for k in z.files}
        times = practical['frame_timestamps']; count = len(times)
        required = ['gaussian_inplace_foot_positions', 'support_foot_candidate',
                    'left_foot_2sigma_support_height', 'right_foot_2sigma_support_height']
        missing = [k for k in required if k not in motion]
        if missing:
            raise ValueError('Foot evidence unavailable: ' + ', '.join(missing))
        feet = motion['gaussian_inplace_foot_positions']
        support = motion['support_foot_candidate'].astype(bool)
        height = np.stack([motion[s+'_foot_2sigma_support_height'] for s in ('left','right')], 1)
        speed = float(practical['native_speed_model_mps']) if spec['stratum']=='walking' else 0.
        if spec['stratum']=='walking' and (not bool(practical['speed_available']) or speed <= .02):
            raise ValueError('No available positive planned gait speed')
        world = np.zeros((count, 3)); world[:, 0] = speed * (times-times[0])
        def matrices(query, ground):
            return compose(times, practical['root_xy'], practical['root_yaw'], np.eye(4),
                           ground, query, world, np.zeros(count))[0]
        baseline = matrices(times, float(practical['ground_offset']))
        if spec['stratum']=='walking':
            query, diagnostics = regularized_clock(times, feet, support, speed, policy)
        else:
            query = times.copy(); diagnostics = dict(bounds_pass=True, rate_min=1., rate_max=1., rate_acceleration_max=0.)
        if support.sum() < 8:
            raise ValueError('Insufficient contact support for ground correction')
        observed = height[support] + float(practical['ground_offset'])
        def ground_cost(delta):
            h = observed + delta
            return np.mean(np.maximum(h,0)**2 + policy['penetration_cost_weight']*np.minimum(h,0)**2)
        limit = policy['maximum_ground_delta_model_m']
        fit = minimize_scalar(ground_cost, bounds=(-limit,limit), method='bounded')
        assert fit.success
        ground = float(practical['ground_offset']) + float(fit.x)
        arrays = dict(times=times, baseline_frames=np.arange(count,dtype=float),
                      baseline_world_from_standard=baseline,
                      regularized_frames=np.interp(query,times,np.arange(count)),
                      regularized_world_from_standard=matrices(query,ground))
        legacy_state = 'available'
        try:
            if spec['stratum']=='walking':
                curve=contact_playback_curve(times,feet,support,policy)
                old_query=curve['sample_times']; lateral=curve['root_xy'][:,1]
                old_yaw=curve['alignment_yaw']
            else:
                old_query=times; lateral=np.zeros(count); old_yaw=0.
            old_world=world.copy();old_world[:,1]=lateral
            old_ground=-float(np.median(height.min(1)))
            arrays.update(legacy_frames=np.interp(old_query,times,np.arange(count)),
                legacy_world_from_standard=compose(times,practical['root_xy'],practical['root_yaw'],
                    np.eye(4),old_ground,old_query,old_world,np.full(count,old_yaw))[0])
        except ValueError as e:
            legacy_state='unavailable: '+str(e)
        np.savez_compressed(destination/'candidate.npz',**arrays)
        result.update(state='candidate_ready_for_runtime_evaluation', branches=[x for x in ['baseline','legacy','regularized'] if x+'_frames' in arrays],
             asset_sha256=read(package/'package_complete.json')['asset_sha256'],placement_sha256=sha(package/'placement.json'),
             candidate_sha256=sha(destination/'candidate.npz'),policy_sha256=sha(Path(__file__).parent/'config/placement_trial.json'),
             native_forward_speed_model_mps=speed,ground_offset_model_m=ground,
             ground_delta_model_m=float(fit.x),clock_diagnostics=diagnostics,legacy_state=legacy_state,
             bounded=diagnostics['bounds_pass'], loopable=False,
             scope='Fixed practical-speed straight trajectory; legacy retains lateral correction and alignment. Contact proxies fit the curve, not independent ground truth.')
    except Exception as e:
        result['reason']=f'{type(e).__name__}: {e}'
    finally:
        result['asset_unchanged']=inventory(package)==before
        result['seconds']=time.monotonic()-begin
        save(destination/'candidate.json',result)
        if not result['asset_unchanged']:
            raise RuntimeError('Source package changed during isolated experiment')
    return result


def aggregate():
    plan=frozen_plan();rows=[]
    for item in plan['cohort']:
        path=LAB/'reports/placement_trials'/item['id']/'candidate.json'
        row=read(path) if path.exists() else dict(identity=item['id'],state='pending',stratum=item['stratum'])
        render=path.parent/'render_validation.json'
        if render.exists(): row['runtime_evaluation']=read(render)
        rows.append(row)
    report=dict(updated_at=time.time(),cohort_size=len(rows),
        states=dict(collections.Counter(r['state'] for r in rows)),results=rows,
        production_adoption=False, scope='Fixed cohort; unavailable remains in denominator; no proxy-only quality acceptance')
    save(LAB/'REPORT.json',report)
    return report


def render(identity):
    package=ASSETS/identity;before=inventory(package)
    from assets import render_placement_trial
    render_placement_trial(identity)
    assert inventory(package)==before, 'Published source modified during runtime evaluation'
    report=LAB/'reports/placement_trials'/identity/'render_validation.json'
    data=read(report);data['source_inventory_unchanged']=True
    save(report,data)


async def run_all():
    await prepare_all()
    from common import environment, PYTHONS
    queue=asyncio.Queue()
    for item in frozen_plan()['cohort']:
        report=read(LAB/'reports/placement_trials'/item['id']/'candidate.json')
        if report['state']=='candidate_ready_for_runtime_evaluation':queue.put_nowait(item['id'])
    reserved=set();allocation=asyncio.Lock()
    async def worker():
        while not queue.empty():
            try:identity=queue.get_nowait()
            except asyncio.QueueEmpty:return
            gpu=None
            while gpu is None:
                async with allocation:
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                    choices=[tuple(map(int,line.split(','))) for line in raw.strip().splitlines()]
                    choices=sorted((x for x in choices if x[0] not in reserved and x[1]>=10240),key=lambda x:-x[1])
                    if choices:gpu=choices[0][0];reserved.add(gpu)
                if gpu is None:await asyncio.sleep(30)
            try:
                env=environment(gpu,'grid4d')
                with (LAB/'logs'/f'{identity}_render.log').open('w') as log:
                    process=await asyncio.create_subprocess_exec(str(PYTHONS['grid4d']),__file__,'render',identity,
                        env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
                    save(LAB/'results'/f'{identity}.json',dict(state='render_running',pid=process.pid,gpu=gpu,updated_at=time.time()))
                    code=await process.wait()
                save(LAB/'results'/f'{identity}.json',dict(state='render_complete' if code==0 else 'render_failed',exit_code=code,gpu=gpu,updated_at=time.time()))
            finally:
                reserved.remove(gpu);queue.task_done()
            aggregate()
    await asyncio.gather(worker(),worker())
    report=aggregate()
    save(LAB/'STATUS.json',dict(state='numeric_and_runtime_queue_finished_visual_review_pending',
        updated_at=time.time(),counts=report['states'],production_assets_modified=False))


async def prepare_all():
    semaphore=asyncio.Semaphore(3)
    async def one(item):
        async with semaphore:
            log=LAB/'logs'/f"{item['id']}_solve.log"
            with log.open('w') as stream:
                process=await asyncio.create_subprocess_exec(sys.executable,__file__,'solve',item['id'],stdout=stream,stderr=asyncio.subprocess.STDOUT)
                await process.wait()
    await asyncio.gather(*map(one,frozen_plan()['cohort']))
    report=aggregate();print(json.dumps(report['states']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['solve','prepare','summary','render','run']);parser.add_argument('identity',nargs='?');args=parser.parse_args()
    if args.command=='solve':print(json.dumps(solve(args.identity)),flush=True)
    elif args.command=='prepare':asyncio.run(prepare_all())
    elif args.command=='render':render(args.identity)
    elif args.command=='run':asyncio.run(run_all())
    else: print(json.dumps(aggregate()['states']),flush=True)
