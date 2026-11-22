"""Check the native CEC projection and backprojection against the existing OpenCV cameras."""
import numpy as np
import torch
from common import ROOT,REPO,manifest,camera,save,device,stamp,sha
from utils.camera_utils import ray2pix,pix2ray

@torch.inference_mode()
def main():
    gpu=device();rows=[]
    for label in ['N04','N08','N11']:
        data,path=manifest(label)
        with np.load(path.parent/data['canonical_npz']) as z:xyz=z['points'][:128]
        for view in range(24):
            cam=camera(data,view,0);record=data['cameras'][view]
            K=np.asarray(record['K']);w=np.linalg.inv(np.asarray(record['camtoworld']))
            points=np.c_[xyz,np.ones(len(xyz))]@w.T
            expected=points[:,:3]@K.T;expected=expected[:,:2]/expected[:,2:]
            r=torch.tensor(points,dtype=torch.float32,device='cuda');P=cam.projection_matrix.T
            u,v=ray2pix(r,640,352,P);actual=torch.stack([v,u],-1).cpu().numpy()
            rays=pix2ray(u,v,640,352,P.inverse())
            # Compare rays up to projective scale in camera coordinates.
            ray_error=float(abs((rays[:,:3]/rays[:,2:3]-r[:,:3]/r[:,2:3])).max())
            error=float(abs(actual-expected).max())
            assert error<.002 and ray_error<1e-4,(label,view,error,ray_error)
            rows.append({'label':label,'view':view,'projection_max_pixels':error,'normalized_ray_max_error':ray_error})
    result={'state':'passed','checked_at':stamp(),'device':gpu,'rows':rows,
        'projection_max_pixels':max(x['projection_max_pixels'] for x in rows),
        'normalized_ray_max_error':max(x['normalized_ray_max_error'] for x in rows),
        'source_sha256':sha(REPO/'utils/camera_utils.py'),
        'change':'Native CEC ray2pix homogeneous divisor corrected from clip_z to clip_w; rasterizer unchanged',
        'scope':'Same 72 cameras and first128 canonical points per person; no quality metric tuning'}
    save(ROOT/'research/cec_projection_after.json',result)
    print({k:v for k,v in result.items() if k!='rows'},flush=True)

if __name__=='__main__':main()
