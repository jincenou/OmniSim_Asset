"""Explicit experiment: fit all-time joint positions, not only per-frame body spans."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import numpy as np

BASE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ['FDANYONE_WAYMO_ROOT']).resolve()
sys.path.insert(0,str(BASE.parent/'repo'))


def main():
    import torch
    from fdanyone.skeleton import pipeline as skeleton
    from fdanyone.motion.result import MotionResult
    from fdanyone.video import decode_canonical_clip
    from fdanyone.assets import resolve_regressor,resolve_foreground_model
    from fdanyone.views import resolve_view_plan
    from fractions import Fraction
    torch.cuda.init();assert os.environ['CUDA_VISIBLE_DEVICES'] in ('4','5','6','7')
    original=skeleton.solve_clip_framing
    evidence={}
    def fit(points,names,profile,camera_factory,aspect_ratio):
        base=original(points,names,profile,camera_factory,aspect_ratio)
        def inside(radius):
            for c in camera_factory(radius,base.target_height):
                w=np.array(c.world_to_camera);p=points@w[:3,:3].T+w[:3,3]
                K=np.array(c.K);K[0,0]=K[1,1]=base.focal_normalized*c.image_height
                q=p@K.T;uv=q[...,:2]/q[...,2:]
                if not ((p[...,2]>0)&(uv[...,0]>=.06*c.image_width)&(uv[...,0]<.94*c.image_width)&(uv[...,1]>=.06*c.image_height)&(uv[...,1]<.94*c.image_height)).all():return False
            return True
        low=base.radius;high=low
        while not inside(high):
            high*=1.15
            if high>100:raise RuntimeError('Cannot frame the estimated trajectory within bounded camera radius')
        for _ in range(20):
            mid=(low+high)/2
            if inside(mid):high=mid
            else:low=mid
        limit=float(os.environ.get('FDANYONE_MAX_RADIUS_RATIO','100'))
        if high/base.radius>limit:
            raise RuntimeError(f'Full-trajectory framing requires radius ratio {high/base.radius:.3f}, exceeds declared limit {limit}')
        evidence.update(original_radius=base.radius,corrected_radius=high,margin_fraction=.06,
            test='Every estimated joint at every time in every safety-ring view must lie inside image margins',
            tradeoff='Retain global root motion; wider framing reduces person pixel size. No claim of improved unobserved identity.')
        return replace(base,radius=high,radius_solve=replace(base.radius_solve,radius=high,bound=None,limiting_constraint='all_time_absolute_joint_bounds'))
    skeleton.solve_clip_framing=fit
    clip=decode_canonical_clip(ROOT/'inputs/main.mp4',num_frames=121,start_time=0,fps=Fraction(25))
    model=BASE.parent/'repo/models';repo=BASE.parent/'repo'
    skeleton.build_skeleton_conditioning(motion=MotionResult.load(ROOT/'preflight/motion'),clip=clip,
        regressor_path=resolve_regressor(None,model_dir=model),foreground_model_path=resolve_foreground_model(model),
        gvhmr_root=repo/'third_party/GVHMR',output_dir=ROOT/'preflight/conditioning_fit',device='cuda:0',
        view_plan=resolve_view_plan(views_per_layer=int(os.environ.get('FDANYONE_CONDITIONING_VIEWS','6')),layer_pitches=[15],start_yaw=0,yaw_span=360,enable_rcp=True,enable_tcr=True))
    (ROOT/'preflight/conditioning').rename(ROOT/'preflight/conditioning_original_radius')
    (ROOT/'preflight/conditioning_fit').rename(ROOT/'preflight/conditioning')
    (ROOT/'research/framing_correction.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence),flush=True)


if __name__=='__main__':main()
