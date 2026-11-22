from pathlib import Path
from dataclasses import dataclass
import numpy as np,json
from scipy.spatial.transform import Rotation
from reliable_reference import reliable_reference
@dataclass
class Geometry:
 vertices_world:np.ndarray
 joints_world:np.ndarray
 keypoints_world:np.ndarray
 keypoints_incam:np.ndarray
 motion_world_to_canonical_world:np.ndarray
 mesh_world:np.ndarray=None

def main():
 n=30;t=np.arange(n);template=np.zeros((24,3));template[1]=[.15,.8,0];template[2]=[-.15,.8,0];template[16]=[.2,1.4,0];template[17]=[-.2,1.4,0];template[0]=[0,.8,0];template[15]=[0,1.7,.1]
 yaw=np.r_[np.linspace(-1,.1,8),np.full(5,.4),np.linspace(.4,1.8,17)];R=Rotation.from_euler('y',yaw[:,None]).as_matrix();points=np.einsum('tij,nj->tni',R,template)
 # Deliberate backward root motion independent of heading, retained by canonical map.
 points+=np.stack([t*0,t*0,-t*.02],1)[:,None,:];g=Geometry(points,points,points,points.copy(),np.eye(4),points.copy());e=np.ones((n,17),bool);e[:8,[5,6,11,12]]=False
 result,report=reliable_reference(g,e);assert report['selected_frames']==[8,9,10,11,12]
 M=np.array(report['rotation_original_canonical_to_reliable']);restored=result.joints_world@M[:3,:3];assert np.max(np.abs(restored-g.joints_world))<1e-6
 # A constant rotation preserves actual turn angle differences, pair distances and trajectory.
 before=np.linalg.norm(np.diff(points,axis=0),axis=-1);after=np.linalg.norm(np.diff(result.joints_world,axis=0),axis=-1);assert np.max(np.abs(before-after))<1e-6
 lateral=result.joints_world[:,1]-result.joints_world[:,2];head=np.unwrap(np.arctan2(lateral[:,2],lateral[:,0]));assert np.max(np.abs(np.diff(head)-np.diff(np.unwrap(-yaw))))<1e-6
 rejected=False
 try:reliable_reference(g,np.zeros_like(e))
 except ValueError:rejected=True
 assert rejected
 out=Path(__file__).resolve().parent/'reliable_reference_test.json';out.write_text(json.dumps({'passed':True,'first8_unreliable_frames_skipped':True,'preserved_temporal_turns':True,'preserved_backward_trajectory':True,'unknown_reference_explicitly_rejected':True,'report':report},indent=2)+'\n')
 print('fixed reference tests passed')
if __name__=='__main__':main()
