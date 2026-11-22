import numpy as np
from motion_evidence import uncertain_intervals,resample_measured_camera,points_to_world

def main():
 # A genuinely stationary world point observed from a translating/rotating camera
 # must remain stationary after full extrinsic reconstruction.
 from scipy.spatial.transform import Rotation
 t=np.array([0.,1.,2.]);c=np.repeat(np.eye(4)[None],3,axis=0)
 rotvec=np.zeros((3,3));rotvec[:,2]=np.deg2rad([0,20,40])
 c[:,:3,:3]=Rotation.from_rotvec(rotvec).as_matrix();c[:,0,3]=[0,4,8]
 q=np.linspace(0,2,51);r=resample_measured_camera(t,c,q);world=np.repeat(np.array([[[2.,3.,8.]]]),len(q),axis=0)
 camera=np.einsum('tji,tkj->tki',r[:,:3,:3],world-r[:,None,:3,3])
 np.testing.assert_allclose(points_to_world(camera,r),world,atol=1e-12)
 try:resample_measured_camera(t,c,[-.1])
 except ValueError:pass
 else:raise AssertionError('Extrapolation was accepted')
 k=np.zeros((51,17,3));k[:,:,2]=.9;k[10:20,15,2]=.1
 a=uncertain_intervals(k,q);assert len(a['intervals'])==1 and a['intervals'][0]['bounded_by_observations']
 assert a['intervals'][0]['start_slot']==10 and a['intervals'][0]['end_slot_inclusive']==19
 k[:5,16,2]=.1;a=uncertain_intervals(k,q)
 assert any(not row['bounded_by_observations'] for row in a['intervals'])
 assert not uncertain_intervals(np.dstack([np.zeros((51,17)),np.zeros((51,17)),np.ones((51,17))]),q)['requires_completion_review']
 print('Passed: moving-camera invariance, out-of-support rejection, internal/end visibility gaps, clear sequence.')
if __name__=='__main__':main()
