"""Explicit horizontal-root coordinate change before skeleton projection."""
from dataclasses import replace
import numpy as np

def separate_horizontal_root(geometry):
    root=np.asarray(geometry.joints_world[:,0],dtype=np.float64)
    delta=root-root[:1];delta[:,1]=0
    transform=np.repeat(np.eye(4)[None],len(root),axis=0);transform[:,:3,3]=delta
    def local(points):return (np.asarray(points,dtype=np.float64)-delta[:,None,:]).astype(np.float32)
    updated=replace(geometry,vertices_world=local(geometry.vertices_world),joints_world=local(geometry.joints_world),
        keypoints_world=local(geometry.keypoints_world),mesh_world=None if geometry.mesh_world is None else local(geometry.mesh_world))
    # This original constant transform is provenance, not the new local map.
    # local_from_motion(t) is stored explicitly and must be used by consumers.
    inverse=np.linalg.inv(transform)
    maps={'local_to_canonical':transform,'canonical_to_local':inverse,
          'motion_to_local':inverse@np.asarray(geometry.motion_world_to_canonical_world)[None]}
    checks={}
    for name in ('vertices_world','joints_world','keypoints_world','mesh_world'):
        original=getattr(geometry,name);modified=getattr(updated,name)
        if original is None:continue
        checks[name+'_recomposition_max_abs']=float(np.max(np.abs(modified+delta[:,None,:]-original)))
        checks[name+'_vertical_max_abs']=float(np.max(np.abs(modified[...,1]-original[...,1])))
    vectors=geometry.joints_world-geometry.joints_world[:,:1]
    local_vectors=updated.joints_world-updated.joints_world[:,:1]
    checks['root_relative_joint_vectors_max_abs']=float(np.max(np.abs(vectors-local_vectors)))
    checks['horizontal_root_residual_max_abs']=float(np.max(np.abs(updated.joints_world[:,0][:,[0,2]]-updated.joints_world[0,0,[0,2]])))
    checks['horizontal_root_path_removed']=float(np.linalg.norm(np.diff(delta,axis=0),axis=1).sum())
    assert max(v for k,v in checks.items() if k.endswith('_max_abs'))<2e-5,checks
    assert np.array_equal(updated.keypoints_incam,geometry.keypoints_incam)
    return updated,maps,checks


def source_camera_maps(original,local,local_to_canonical):
    # Infer the rigid relation between the SAME SMPL body in two stored spaces.
    # This consistency check does not recover real Waymo metric calibration.
    a=np.asarray(original.keypoints_world,dtype=np.float64);b=np.asarray(original.keypoints_incam,dtype=np.float64)
    ac=a.mean(1,keepdims=True);bc=b.mean(1,keepdims=True)
    u,_,vt=np.linalg.svd(np.swapaxes(a-ac,1,2)@(b-bc))
    fix=np.repeat(np.eye(3)[None],len(a),axis=0);fix[:,2,2]=np.linalg.det(np.swapaxes(vt,1,2)@np.swapaxes(u,1,2))
    rot=np.swapaxes(vt,1,2)@fix@np.swapaxes(u,1,2)
    trans=bc[:,0]-np.einsum('tij,tj->ti',rot,ac[:,0])
    canonical_to_incam=np.repeat(np.eye(4)[None],len(a),axis=0);canonical_to_incam[:,:3,:3]=rot;canonical_to_incam[:,:3,3]=trans
    local_to_incam=canonical_to_incam@local_to_canonical
    projected=np.einsum('tij,tkj->tki',local_to_incam[:,:3,:3],local.keypoints_world)+local_to_incam[:,None,:3,3]
    error=float(np.max(np.abs(projected-b)))
    if error>=1e-4:raise ValueError(f'Global/incam SMPL geometry is not a consistent rigid pair: max error {error}')
    return local_to_incam,error
