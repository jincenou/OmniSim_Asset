"""Map existing SMPL-X motion to released RoHM representation, with roundtrip."""
from pathlib import Path
import argparse,json,pickle,sys,hashlib
import numpy as np
import torch,smplx
from scipy.spatial.transform import Rotation,Slerp
from safetensors.torch import load_file
from camera_geometry import transform_smpl_root,transform_points
from optimize_motion import supported_points
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent/'frontend_quality_20261020';REPO=ROOT.parents[1]/'repo'
sys.path.insert(0,str(PRIOR/'vendor/RoHM'))
from utils.other_utils import REPR_LIST,REPR_DIM_DICT
from data_loaders.common.quaternion import rot6d_to_rotmat
from data_loaders.motion_representation import cano_seq_smplx,get_repr_smplx,recover_from_repr_smpl

# Explicit names: SMPL-X's first22 main body joints; COCO only has selected observations.
SMPL_NAMES=('pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3','left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist')
COCO_NAMES=('nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('sample');parser.add_argument('--case-config',type=Path);args=parser.parse_args();torch.set_num_threads(2)
    protocol=ROOT/'rohm_protocol_v1.json' if args.case_config is None else ROOT/'occlusion_v1/ROHM_PROTOCOL.json'
    config=json.loads(protocol.read_text());assert args.sample in config['subjects']
    cohort=json.loads((ROOT/'cohort_v1.json').read_text());case=next(x for x in cohort['dev6'] if x['sample_id']==args.sample)
    source=Path(case['manifest']).parent
    if args.case_config is not None:
        if not args.case_config.resolve().is_relative_to(ROOT/'occlusion_v1') or args.case_config.parent.name!='rgb':raise ValueError('Only frozen RGB counterfactuals')
        source=args.case_config.resolve().parent
        if json.loads(args.case_config.read_text())['sample']!=args.sample:raise ValueError('Case identity mismatch')
        assert protocol.with_suffix('.sha256').read_text().split()[0]==hashlib.sha256(protocol.read_bytes()).hexdigest()
    state=load_file(str(source/'motion/motion.safetensors'))
    params={k:state['smpl_params_global.'+k].numpy() for k in ['body_pose','global_orient','transl','betas']}
    n=len(params['transl']);times=np.arange(n)/25;query=np.arange(round(times[-1]*30)+1)/30
    assert query[-1]<=times[-1]+1e-9
    params['body_pose']=np.stack([Slerp(times,Rotation.from_rotvec(params['body_pose'].reshape(n,21,3)[:,j]))(query).as_rotvec() for j in range(21)],axis=1).reshape(len(query),63)
    params['global_orient']=Slerp(times,Rotation.from_rotvec(params['global_orient']))(query).as_rotvec()
    params['transl']=np.column_stack([np.interp(query,times,params['transl'][:,j]) for j in range(3)])
    params['betas']=np.repeat(params['betas'].mean(0,keepdims=True),len(query),axis=0)
    model=smplx.create(str(REPO/'models/body_models'),model_type='smplx',gender='neutral',flat_hand_mean=True,use_pca=False,batch_size=len(query)).eval()
    to_tensor=lambda x:{k:torch.as_tensor(v,dtype=torch.float32) for k,v in x.items()}
    with torch.inference_mode():body=model(**to_tensor(params));joints=body.joints[:,:22].numpy()
    # GVHMR global y-up to right-handed z-up; no Waymo camera translation added.
    c=np.repeat(np.eye(4)[None],len(query),axis=0);c[:,:3,:3]=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    pelvis=joints[:,0]-params['transl'];a,t=transform_smpl_root(params['global_orient'],params['transl'],pelvis,c)
    zparams=dict(params,global_orient=a,transl=t);zjoints=transform_points(joints,c)
    canonical,canonical_params,transform=cano_seq_smplx(zjoints,zparams,return_transf_mat=True)
    with torch.inference_mode():actual=model(**to_tensor(canonical_params)).joints[:,:22].numpy()
    vertex_scope_error=float(np.max(np.abs(actual-canonical)))
    assert vertex_scope_error<1e-4,vertex_scope_error
    representation=get_repr_smplx(canonical,canonical_params)
    packed=np.concatenate([representation[k] for k in REPR_LIST],axis=-1)
    assert packed.shape==(len(query)-1,294) and np.isfinite(packed).all()
    # Verify rotation layout is RoHM's column layout, not another library's row convention.
    restored_rotation=rot6d_to_rotmat(torch.tensor(representation['smplx_body_pose_6d'].reshape(-1,6),dtype=torch.float32)).numpy()
    target_rotation=Rotation.from_rotvec(canonical_params['body_pose'][:-1].reshape(-1,3)).as_matrix()
    rotation_error=float(np.max(np.abs(restored_rotation-target_rotation)));assert rotation_error<1e-5
    decoded=recover_from_repr_smpl({k:torch.tensor(v[None],dtype=torch.float32) for k,v in representation.items()},recover_mode='joint_abs_traj').numpy()[0]
    joint_error=float(np.max(np.abs(decoded-canonical[:-1])));assert joint_error<1e-4,joint_error
    stats=PRIOR/'models/rohm/checkpoints/posenet_checkpoint'
    mean_dict=pickle.loads((stats/'AMASS_mean.pkl').read_bytes());std_dict=pickle.loads((stats/'AMASS_std.pkl').read_bytes())
    mean=np.concatenate([mean_dict[k] for k in REPR_LIST]);std=np.concatenate([std_dict[k] for k in REPR_LIST]);assert (std>0).all()
    normalized=(packed-mean)/std
    kp=state['observed_keypoints_2d'].numpy();mask=np.load(source/'input/target_masks.npz')['masks']
    evidence,_=supported_points(kp,mask,json.loads((ROOT/'protocol_v1.json').read_text())['visibility'])
    right=np.searchsorted(times,query,side='left').clip(0,n-1);left=np.maximum(right-1,0)
    exact=np.isclose(times[right],query,atol=1e-8);left[exact]=right[exact]
    obs=evidence[left]&evidence[right]
    reliable=np.zeros((len(query),22),dtype=bool)
    for j,name in enumerate(SMPL_NAMES):
        if name in COCO_NAMES:reliable[:,j]=obs[:,COCO_NAMES.index(name)]
    # Root trajectory is estimated conditioning, not an observed 3D joint.
    slices={};start=0
    for k in REPR_LIST:slices[k]=slice(start,start+REPR_DIM_DICT[k]);start+=REPR_DIM_DICT[k]
    condition=normalized.copy();position=condition[:,slices['local_positions']].reshape(-1,22,3);position[~reliable[:-1]]=0
    velocity=condition[:,slices['local_vel']].reshape(-1,22,3);velocity[~(reliable[:-1]&reliable[1:])]=0
    rotation_reliable=reliable[:,1:].copy()
    limb_map=json.loads((ROOT/'protocol_v1.json').read_text())['O1']['changed_joint_indices_body_pose_to_coco_child']
    for j,child in limb_map.items():rotation_reliable[:,int(j)] &= obs[:,child]
    angles=condition[:,slices['smplx_body_pose_6d']].reshape(-1,21,6);angles[~rotation_reliable[:-1]]=0
    condition[:,slices['foot_contact']]=0
    assert np.isfinite(condition).all()
    out=ROOT/'rohm_proposals'/args.sample if args.case_config is None else source/'rohm_proposal'
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'conditioning.npz',condition=condition.astype(np.float32),raw_representation=packed.astype(np.float32),mean=mean,std=std,
      canonical_transform=transform,gvhmr_y_up_to_z_up=c[0],times_30fps=query[:-1],original_times=times,reliable_smpl22=reliable[:-1],rotation_reliable=rotation_reliable[:-1],
      original_body_pose=state['smpl_params_global.body_pose'].numpy(),original_betas=state['smpl_params_global.betas'].numpy())
    report={'state':'real_motion_repr_roundtrip_passed_not_sampled','sample':args.sample,'source':str(source),'source_frames':n,'resampled_frames':len(query),'representation_times':len(query)-1,
      'canonical_joint_forward_error_m':vertex_scope_error,'joint_repr_roundtrip_max_error_m':joint_error,'pose6d_matrix_roundtrip_error':rotation_error,
      'joint_names':SMPL_NAMES,'observed_names':COCO_NAMES,'unmapped_joint_evidence':'unknown, never inferred visible','condition_shape':list(condition.shape),
      'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
      'scope':'PoseNet-only proposal; estimated native GVHMR trajectory conditioning. Actual source camera is not passed as static camera. No missing frame duplicated as observation.'}
    (out/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
