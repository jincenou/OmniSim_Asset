"""Actual SMPL-X vertices and synthetic camera projection, no GPU required."""
from pathlib import Path
import json,sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from camera_geometry import transform_smpl_root,transform_points,project

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]/'repo'
sys.path.insert(0,str(REPO))
from fdanyone.skeleton.pipeline import _gvhmr_geometry_context

def main():
    torch.set_num_threads(2)
    rng=np.random.default_rng(20261020)
    n=4
    with _gvhmr_geometry_context(REPO/'third_party/GVHMR'):
        from hmr4d.utils.smplx_utils import make_smplx
        model=make_smplx('supermotion').double().eval()
        pose=torch.tensor(rng.normal(0,.1,(n,63)),dtype=torch.float64)
        betas=torch.tensor(rng.normal(0,.2,(n,10)),dtype=torch.float64)
        orient=rng.normal(0,.2,(n,3));trans=np.tile([.2,.1,8.],(n,1))
        with torch.no_grad():
            zero=torch.zeros(n,3,dtype=torch.float64)
            rest=model(body_pose=pose,betas=betas,global_orient=zero,transl=zero)
            pelvis=rest.joints[:,0].numpy()
            before=model(body_pose=pose,betas=betas,global_orient=torch.from_numpy(orient),transl=torch.from_numpy(trans)).vertices.numpy()
            c=np.repeat(np.eye(4)[None],n,axis=0)
            c[:,:3,:3]=Rotation.from_rotvec(rng.normal(0,.3,(n,3))).as_matrix()
            c[:,:3,3]=rng.normal(0,4,(n,3))
            a,t=transform_smpl_root(orient,trans,pelvis,c)
            after=model(body_pose=pose,betas=betas,global_orient=torch.from_numpy(a),transl=torch.from_numpy(t)).vertices.numpy()
    expected=transform_points(before,c)
    error=float(np.max(np.abs(after-expected)))
    K=np.repeat(np.array([[900.,0,352],[0,900,640],[0,0,1]])[None],n,axis=0)
    back=transform_points(after,np.linalg.inv(c))
    px=float(np.max(np.abs(project(before,K)-project(back,K))))
    # Stationary world points observed by translating and rotating cameras.
    stationary=np.repeat(before[[0]],n,axis=0)
    recovered=transform_points(transform_points(stationary,np.linalg.inv(c)),c)
    stationary_error=float(np.max(np.abs(stationary-recovered)))
    assert error<1e-6 and px<=1e-3 and stationary_error<1e-9,(error,px,stationary_error)
    report=dict(passed=True,vertices_per_body=10475,frames=n,max_world_vertex_error_m=error,max_projection_roundtrip_px=px,stationary_world_roundtrip_m=stationary_error,
      scope='Algebra/actual body-model equivalence only, not true Waymo human depth or motion accuracy; no translation added to native GVHMR global outputs.')
    (ROOT/'camera_geometry_test.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
