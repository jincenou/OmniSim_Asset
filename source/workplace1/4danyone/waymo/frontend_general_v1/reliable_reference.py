"""One fixed canonical reference from a reliable temporal window; never per-frame de-yaw."""
from dataclasses import replace
import numpy as np

def reliable_reference(geometry,reliable,min_consecutive=5):
    evidence=np.asarray(reliable,dtype=bool)
    if evidence.ndim!=2 or evidence.shape[1]!=17 or len(evidence)!=len(geometry.joints_world):raise ValueError('Expected COCO17 evidence for each frame')
    trusted=evidence[:,[5,6,11,12]].all(1)
    eligible=np.convolve(trusted.astype(int),np.ones(min_consecutive,dtype=int),mode='valid')==min_consecutive
    starts=np.flatnonzero(eligible)
    if not len(starts):raise ValueError('No consecutive supported shoulders/hips; canonical reference unknown')
    start=int(starts[0]);slots=np.arange(start,start+min_consecutive)
    joints=geometry.joints_world[slots]
    lateral=joints[:,1][:,[0,2]]-joints[:,2][:,[0,2]]+joints[:,16][:,[0,2]]-joints[:,17][:,[0,2]]
    norms=np.linalg.norm(lateral,axis=1)
    if np.any(norms<1e-6):raise ValueError('Degenerate reference anatomy')
    lateral=lateral/norms[:,None];direction=np.stack([lateral[:,1],-lateral[:,0]],1);mean=direction.mean(0);concentration=float(np.linalg.norm(mean))
    if concentration<.95:raise ValueError('Reference window has ambiguous facing; no forced orientation')
    mean/=concentration;angle=float(np.pi-np.arctan2(mean[0],mean[1]));c,s=np.cos(angle),np.sin(angle)
    R=np.array([[c,0,s],[0,1,0],[-s,0,c]]);M=np.eye(4);M[:3,:3]=R
    def apply(points):return None if points is None else (np.asarray(points,dtype=np.float64)@R.T).astype(np.float32)
    result=replace(geometry,vertices_world=apply(geometry.vertices_world),joints_world=apply(geometry.joints_world),keypoints_world=apply(geometry.keypoints_world),mesh_world=apply(geometry.mesh_world),motion_world_to_canonical_world=M@geometry.motion_world_to_canonical_world)
    assert np.array_equal(result.keypoints_incam,geometry.keypoints_incam)
    return result,{'selected_frames':slots.tolist(),'rotation_original_canonical_to_reliable':M.tolist(),'circular_concentration':concentration,'single_constant_reference':True,'source_camera_keypoints_unchanged':True,
      'selection':'earliest5 consecutive frames with supported shoulders and hips; no output selection','body_facing_not_velocity':True,'ambiguity_policy':'no reference if missing/degenerate/incoherent; do not silently use first frame'}
