"""Reviewed R01 M0/M2 root-local conditioning, with authentic per-variant motion."""
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from root_local_geometry import separate_horizontal_root,source_camera_maps

ROUND=Path(__file__).resolve().parents[1];W=ROUND.parent;REPO=ROUND/'moving61_runtime'
ROOT=Path(os.environ['FDANYONE_WAYMO_ROOT']);ORIGINAL=Path(os.environ['FDANYONE_SOURCE_EXPERIMENT']).resolve()
VIEWS=int(os.environ.get('FDANYONE_ROOT_VIEWS','6'))
assert VIEWS in (6,24)
sys.path.insert(0,str(REPO))

def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    import torch
    from fdanyone.skeleton import pipeline as skeleton
    from fdanyone.motion.result import MotionResult
    from fdanyone.video import decode_canonical_clip
    from fdanyone.assets import resolve_regressor,resolve_foreground_model
    from fdanyone.views import resolve_view_plan
    physical=int(os.environ['CUDA_VISIBLE_DEVICES']);assert physical in (4,5,6,7)
    expected=subprocess.check_output(['nvidia-smi',f'--id={physical}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();actual=str(torch.cuda.get_device_properties(0).uuid);assert actual==expected
    variant=os.environ['FDANYONE_MOTION_VARIANT']
    assert variant in ('static_estimated','calibrated_moving') and VIEWS==24
    review=json.loads((ORIGINAL/'motion_pair/review.json').read_text())
    assert review['input_sha256']==sha(ORIGINAL/'inputs/main.mp4')
    assert review['variants'][variant]['allow_exploratory_generation']
    assert json.loads((ORIGINAL/'research/batch_gate_mask.json').read_text())['passed']
    motion_dir=ORIGINAL/'motion_pair'/variant
    assert review['variants'][variant]['motion_sha256']==sha(motion_dir/'motion.safetensors')
    ROOT.mkdir(parents=True,exist_ok=False);(ROOT/'inputs').mkdir();(ROOT/'preflight').mkdir();(ROOT/'research').mkdir()
    (ROOT/'inputs/main.mp4').symlink_to(ORIGINAL/'inputs/main.mp4')
    (ROOT/'inputs/frame_mapping.json').symlink_to(ORIGINAL/'inputs/frame_mapping.json')
    (ROOT/'candidate.json').symlink_to(ORIGINAL/'candidate.json')
    (ROOT/'preflight/motion').symlink_to(motion_dir,target_is_directory=True)
    save(ROOT/'preflight/device.json',{'physical_gpu':physical,'actual_cuda_uuid':actual,'pid':os.getpid()})
    clip=decode_canonical_clip(ROOT/'inputs/main.mp4',num_frames=61,start_time=0,fps=Fraction(25))
    motion=MotionResult.load(ROOT/'preflight/motion');motion.validate_against_clip(clip)
    original_geometry=skeleton._body_geometry;original_framing=skeleton.solve_clip_framing
    original_masks=skeleton.predict_foreground_masks;record={};geometry_cache={}
    def geometry(*args,**kwargs):
        g=original_geometry(*args,**kwargs);local,maps,checks=separate_horizontal_root(g)
        source_map,error=source_camera_maps(g,local,maps['local_to_canonical'])
        maps['local_to_smpl_incam']=source_map
        np.savez_compressed(ROOT/'preflight/root_motion.npz',**maps,frame_times=np.asarray(motion.frame_timestamps_sec))
        geometry_cache.update(original=g,local=local)
        record.update(root_checks=checks,source_rigid_recomposition_error=error)
        return local
    def masks(*args,**kwargs):
        result=original_masks(*args,**kwargs);np.savez_compressed(ROOT/'preflight/source_masks.npz',masks=result);return result
    def framing(points,names,profile,camera_factory,aspect_ratio):
        base=original_framing(points,names,profile,camera_factory,aspect_ratio)
        def inside(radius):
            for c in camera_factory(radius,base.target_height):
                crop=skeleton.center_crop(c.image_height,c.image_width,1280,704)
                k=np.asarray(c.K).copy();k[0,0]=k[1,1]=base.focal_normalized*c.image_height
                c=skeleton._cropped_camera(replace(c,K=tuple(map(tuple,k))),crop)
                w=np.asarray(c.world_to_camera);p=points@w[:3,:3].T+w[:3,3]
                q=p@np.asarray(c.K).T;uv=q[...,:2]/q[...,2:]
                # 1e-5 interior guard prevents the binary-search boundary from
                # rounding outside the independently checked 6% margin.
                if not ((p[...,2]>0)&(uv[...,0]>=.06001*c.image_width)&(uv[...,0]<=.93999*c.image_width)&(uv[...,1]>=.06001*c.image_height)&(uv[...,1]<=.93999*c.image_height)).all():return False
            return True
        low=.1*base.radius;high=base.radius
        while not inside(high):
            high*=1.15
            if high>2*base.radius:raise RuntimeError('Root-local branch still needs excessive camera radius')
        for _ in range(24):
            mid=(low+high)/2
            if inside(mid):high=mid
            else:low=mid
        record['framing']={'initial_radius':base.radius,'local_all_time_radius':high,'margin_fraction':.06}
        return replace(base,radius=high,radius_solve=replace(base.radius_solve,radius=high,bound=None,limiting_constraint='root_local_all_time_joint_bounds'))
    skeleton._body_geometry=geometry;skeleton.solve_clip_framing=framing;skeleton.predict_foreground_masks=masks
    target=ROOT/'preflight/conditioning'
    skeleton.build_skeleton_conditioning(motion=motion,clip=clip,regressor_path=resolve_regressor(None,model_dir=REPO/'models'),
        foreground_model_path=resolve_foreground_model(REPO/'models'),gvhmr_root=REPO/'third_party/GVHMR',output_dir=target,device='cuda:0',
        view_plan=resolve_view_plan(views_per_layer=VIEWS,layer_pitches=[15],start_yaw=0,yaw_span=360,enable_rcp=VIEWS==24,enable_tcr=True))
    payload=json.loads((target/'cameras.json').read_text());meta=json.loads((target/'metadata.json').read_text())
    payload['original_motion_world_to_canonical_world']=payload.pop('motion_world_to_canonical_world')
    payload['world_frame']={**payload['world_frame'],'name':'root_local_human_world','root_translation':'per-frame horizontal displacement stored separately'}
    payload['motion_world_to_local_world_file']='../root_motion.npz';save(target/'cameras.json',payload)
    points=geometry_cache['local'].keypoints_world;coverage=[]
    for c in payload['cameras']:
        w=np.asarray(c['world_to_camera']);p=points@w[:3,:3].T+w[:3,3];q=p@np.asarray(c['K']).T;uv=q[...,:2]/q[...,2:]
        good=(p[...,2]>0)&(uv[...,0]>=.06*c['image_width'])&(uv[...,0]<=.94*c['image_width'])&(uv[...,1]>=.06*c['image_height'])&(uv[...,1]<=.94*c['image_height'])
        coverage.append({'view':c['camera_id'],'all_joints_all_frames_inside':bool(good.all()),'median_joint_height_fraction':float(np.median(np.ptp(uv[...,1],axis=1))/c['image_height'])})
    crop=meta['source_crop'];source_masks=np.load(ROOT/'preflight/source_masks.npz')['masks']>12.75
    retained=source_masks[:,crop['top']:crop['top']+crop['height'],crop['left']:crop['left']+crop['width']].sum((1,2))/np.maximum(source_masks.sum((1,2)),1)
    assert all(c['all_joints_all_frames_inside'] for c in coverage) and retained.min()>=.95,(coverage,retained.min())
    record.update(state='geometry_and_source_retention_passed_experimental_generation_allowed',source_experiment=str(ORIGINAL),
        source_video_sha256=sha(ORIGINAL/'inputs/main.mp4'),root_motion_sha256=sha(ROOT/'preflight/root_motion.npz'),coverage=coverage,
        source_motion_hashes={name:sha(motion_dir/name) for name in ('motion.json','motion.safetensors')},
        source_mask_retained_min=float(retained.min()),source_motion_reused_without_modification=True,motion_variant=variant,motion_source=str(motion_dir),world_frame='root_local_human_world',
        limitations=['Experimental transformation; generated appearance/motion still need review','Camera translation is not consumed; root trajectory is estimated and not Waymo world registered','Not arbitrary SMPL-X retargeting'])
    save(ROOT/'research/root_local.json',record)
    save(ROOT/'preflight/review.json',{'allow_generation':True,'scope':f'{VIEWS}-view experimental generation; numerical geometry and original input gates pass','visual_generation_review':'pending','evidence':'../research/root_local.json'})
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
