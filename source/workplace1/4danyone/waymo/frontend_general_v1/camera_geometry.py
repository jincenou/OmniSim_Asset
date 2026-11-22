"""SE(3) utilities. SMPL root rotations pivot around its shaped rest pelvis."""
import numpy as np
from scipy.spatial.transform import Rotation

def transform_smpl_root(global_orient, transl, rest_pelvis, camera_to_world):
    a=np.asarray(global_orient,dtype=np.float64)
    t=np.asarray(transl,dtype=np.float64)
    j=np.asarray(rest_pelvis,dtype=np.float64)
    c=np.asarray(camera_to_world,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=3 or t.shape!=a.shape or j.shape!=a.shape or c.shape!=(len(a),4,4):
        raise ValueError('Expected matched frame arrays, pelvis and SE(3)')
    if not all(np.isfinite(x).all() for x in (a,t,j,c)):
        raise ValueError('Nonfinite geometry')
    r=c[:,:3,:3]
    if not np.allclose(r@r.transpose(0,2,1),np.eye(3),atol=1e-6) or not np.allclose(np.linalg.det(r),1,atol=1e-6) or not np.allclose(c[:,3],[0,0,0,1]):
        raise ValueError('Invalid SE(3)')
    root=Rotation.from_matrix(r@Rotation.from_rotvec(a).as_matrix()).as_rotvec()
    translation=np.einsum('tij,tj->ti',r,t+j)+c[:,:3,3]-j
    return root,translation

def transform_points(points,transform):
    return np.einsum('tij,tnj->tni',transform[:,:3,:3],points)+transform[:,None,:3,3]

def project(points,K):
    if np.any(points[...,2]<=0):raise ValueError('Behind-camera points')
    h=np.einsum('tij,tnj->tni',K,points)
    return h[...,:2]/h[...,2:]
