"""CPU-only real cached motion audit; no new video generation or GPU use."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from root_local_geometry import separate_horizontal_root,source_camera_maps

ROOT=Path(__file__).resolve().parents[1];W=ROOT.parent;REPO=W.parent/'repo'
sys.path.insert(0,str(REPO))
from fdanyone.motion.result import MotionResult
from fdanyone.skeleton.pipeline import _body_geometry
from fdanyone.assets import resolve_regressor


def main():
    assert not torch.cuda.is_available(),'CPU audit requires CUDA_VISIBLE_DEVICES empty'
    label='pilot_063';experiment=W/'experiments'/label;out=ROOT/'root_local'/label
    out.mkdir(parents=True,exist_ok=False)
    motion_dir=experiment/'preflight/motion';motion=MotionResult.load(motion_dir)
    assert torch.allclose(motion.smpl_params_global['body_pose'],motion.smpl_params_incam['body_pose'],atol=1e-5,rtol=0)
    assert torch.allclose(motion.smpl_params_global['betas'],motion.smpl_params_incam['betas'],atol=1e-5,rtol=0)
    geometry=_body_geometry(motion,resolve_regressor(None,model_dir=REPO/'models'),REPO/'third_party/GVHMR','cpu',include_mesh=True)
    local,maps,checks=separate_horizontal_root(geometry)
    source_map,error=source_camera_maps(geometry,local,maps['local_to_canonical'])
    maps['local_to_smpl_incam']=source_map
    incam=np.einsum('tij,tkj->tki',source_map[:,:3,:3],local.keypoints_world)+source_map[:,None,:3,3]
    k=motion.K_fullimg.numpy();qa=np.einsum('tij,tkj->tki',k,incam);qb=np.einsum('tij,tkj->tki',k,geometry.keypoints_incam)
    uv_error=float(np.max(np.abs(qa[...,:2]/qa[...,2:]-qb[...,:2]/qb[...,2:])))
    assert uv_error<.02,uv_error
    np.savez_compressed(out/'geometry.npz',vertices_original=geometry.vertices_world,vertices_local=local.vertices_world,
        joints_original=geometry.joints_world,joints_local=local.joints_world,keypoints_original=geometry.keypoints_world,
        keypoints_local=local.keypoints_world,keypoints_source=geometry.keypoints_incam,
        mesh_original=geometry.mesh_world,mesh_local=local.mesh_world,mesh_faces=geometry.mesh_faces,
        frame_times=np.asarray(motion.frame_timestamps_sec),**maps)
    cameras=json.loads((experiment/'preflight/conditioning/cameras.json').read_text())['cameras']
    metadata=json.loads((experiment/'preflight/conditioning/metadata.json').read_text())
    original_center=geometry.vertices_world.mean(axis=(0,1));original_center[1]=metadata['framing']['target_height']
    local_center=local.vertices_world.mean(axis=(0,1));local_center[1]=original_center[1]
    def framing(points,center):
        def evaluate(scale):
            bounds=[];inside=[]
            for camera in cameras:
                c=np.asarray(camera['camera_to_world']).copy();c[:3,3]=center+scale*(c[:3,3]-original_center)
                w=np.linalg.inv(c);p=points@w[:3,:3].T+w[:3,3];q=p@np.asarray(camera['K']).T;uv=q[...,:2]/q[...,2:]
                height=camera['image_height'];width=camera['image_width']
                inside.append(bool(((p[...,2]>0)&(uv[...,0]>=.06*width)&(uv[...,0]<=.94*width)&(uv[...,1]>=.06*height)&(uv[...,1]<=.94*height)).all()))
                bounds.extend((np.ptp(uv[...,1],axis=1)/height).tolist())
            return all(inside),float(np.median(bounds))
        low=.1;high=1.
        while not evaluate(high)[0]:
            high*=1.2
            if high>30:raise RuntimeError('Bounded framing failed')
        for _ in range(24):
            mid=(low+high)/2
            if evaluate(mid)[0]:high=mid
            else:low=mid
        return {'camera_count':len(cameras),'minimum_radius_scale':high,'median_joint_height_fraction':evaluate(high)[1],'all_joints_all_frames_in_6percent_margin':evaluate(high)[0]}
    report={'state':'cpu_geometry_and_projection_audit_passed_generation_not_started','label':label,'frames':motion.num_frames,
       'source_motion_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in motion_dir.iterdir() if p.is_file()},
       'definition':'Remove only root x/z displacement relative to first frame in canonical Y-up coordinates; preserve vertical motion and all rotations',
       'checks':checks,'source_rigid_recomposition_error':error,'source_projection_max_pixel_error':uv_error,
       'framing_original':framing(geometry.keypoints_world,original_center),'framing_local':framing(local.keypoints_world,local_center),
       'source_motion_and_incam_parameters_modified':False,'gpu_used':False,
       'limitations':['Uses estimated GVHMR motion, not verified real Waymo source geometry','Only cached target camera ring checked; new 24-view conditioning still requires full preflight','No generated root-local video or asset yet','Stored original motion_to_canonical is provenance; consumers must use explicit per-frame motion_to_local','Local-to-incam comes from identical SMPL pose representations, not measured Waymo extrinsics']}
    (out/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
