"""Verify actual HUST CUDA camera matrices against shared source/FTGS K and poses."""
from pathlib import Path
import sys,json,os,hashlib
import numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source/4dgaussians'))
from scene.waymo_generated import GeneratedCameras
from gpu_guard import validate_gpu

def main():
 label=sys.argv[1];device=validate_gpu();manifest=json.loads((R/'four_dg/data'/label/'fdanyone.json').read_text());source=R/'samples'/label;rawcameras=json.loads((source/'generation/outputs/turbo/cameras.json').read_text())['cameras'];z=np.load(source/'data/init.npz');at=np.isclose(z['times'],0);raw=z['positions'][at];A=np.asarray(manifest['source_to_normalized']);canonical=(np.column_stack([raw,np.ones(len(raw))])@A.T)[:,:3];canonical_saved=np.load(R/'four_dg/data'/label/'canonical.npz')['points'];assert abs(canonical-canonical_saved).max()<2e-6;pick=np.rint(np.linspace(0,len(raw)-1,min(100,len(raw)))).astype(int);raw=raw[pick];canonical=canonical[pick];points=torch.as_tensor(np.column_stack([canonical,np.ones(len(canonical))]),device='cuda',dtype=torch.float32);dataset=GeneratedCameras(manifest,list(range(24)));frames=manifest['frame_count'];rows=[]
 for view in range(24):
  cam=manifest['cameras'][view];K=np.asarray(cam['K']);w=np.linalg.inv(cam['camtoworld']);cpu=canonical@w[:3,:3].T+w[:3,3];uv=cpu@K.T;uv=uv[:,:2]/uv[:,2:];old=rawcameras[view];world=np.linalg.inv(old['camera_to_world']);q=raw@world[:3,:3].T+world[:3,3];k=np.asarray(old['K']).copy();k[:2]*=.5;u=q@k.T;u=u[:,:2]/u[:,2:];normalized_error=float(abs(u-uv).max());assert normalized_error<1e-3
  for frame in [0,(frames-1)//2,frames-1]:
   camera=dataset[view*frames+frame];assert abs(camera.time-frame/(frames-1))<1e-8;ndc=points@camera.full_proj_transform;xy=ndc[:,:2]/ndc[:,3:];pixels=((xy+1)*torch.tensor([352,640],device='cuda')-1)/2;error=float(abs(pixels.cpu().numpy()-uv).max());assert error<1e-3,(view,frame,error);rows.append({'view':view,'frame':frame,'time':camera.time,'cuda_vs_K_pose_max_error_px':error,'normalization_projection_error_px':normalized_error})
 result={'state':'new_actual_data_all24_cameras_three_times_cuda_projection_passed','label':label,'device':device,'threshold_px':.001,'max_error_px':max(x['cuda_vs_K_pose_max_error_px'] for x in rows),'max_normalization_error_px':max(x['normalization_projection_error_px'] for x in rows),'canonical_point_alignment_max_abs':float(abs((np.column_stack([z['positions'][at],np.ones(int(at.sum()))])@A.T)[:,:3]-canonical_saved).max()),'scope':'Numerical physical-camera/pixel-center and common canonical mapping; not RGB quality or realWaymo world calibration','samples':rows};p=R/'research'/f'{label}_camera_projection.json';p.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='samples'},flush=True)
if __name__=='__main__':main()
