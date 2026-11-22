"""Released PoseNet actual 1000-step sampling; gap proposal, no hidden truth claim."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,sys,time,hashlib,os
import numpy as np
import torch
from scipy.spatial.transform import Rotation,Slerp
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent/'frontend_quality_20261020';REPO=ROOT.parents[1]/'repo'
sys.path.insert(0,str(PRIOR/'vendor/RoHM'))
from utils.other_utils import REPR_LIST,REPR_DIM_DICT
from model.posenet import PoseNet
from diffusion import gaussian_diffusion_posenet as gd
from diffusion.respace import SpacedDiffusionPoseNet
from utils.model_util import create_gaussian_diffusion
from data_loaders.common.quaternion import rot6d_to_rotmat

def main():
    parser=argparse.ArgumentParser();parser.add_argument('sample');parser.add_argument('--case-config',type=Path);args=parser.parse_args();torch.set_num_threads(2)
    prior_protocol=ROOT/'rohm_protocol_v1.json' if args.case_config is None else ROOT/'occlusion_v1/ROHM_PROTOCOL.json'
    cfg=json.loads(prior_protocol.read_text());assert args.sample in cfg['subjects']
    expected=os.environ['FD_EXPECTED_GPU_UUID'].removeprefix('GPU-')
    assert torch.cuda.device_count()==1 and str(torch.cuda.get_device_properties(0).uuid)==expected
    device=torch.device('cuda:0');torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);started=time.time()
    out=ROOT/'rohm_proposals'/args.sample
    if args.case_config is not None:
        if not args.case_config.resolve().is_relative_to(ROOT/'occlusion_v1') or args.case_config.parent.name!='rgb':raise ValueError('Only frozen RGB counterfactuals')
        assert prior_protocol.with_suffix('.sha256').read_text().split()[0]==hashlib.sha256(prior_protocol.read_bytes()).hexdigest()
        out=args.case_config.resolve().parent/'rohm_proposal'
    if (out/'sample_report.json').exists():raise FileExistsError(out/'sample_report.json')
    prep=json.loads((out/'preparation.json').read_text());data=np.load(out/'conditioning.npz')
    condition=torch.tensor(data['condition'].T[None,:,None,:],device=device)
    dataset=SimpleNamespace(body_feat_dim=294,pose_feat_dim=272,traj_feat_dim=22)
    model=PoseNet(dataset=dataset,body_feat_dim=294,latent_dim=512,ff_size=1024,num_layers=8,num_heads=4,dropout=.1,activation='gelu',
      body_model_path=str(REPO/'models/body_models'),device=device,traj_feat_dim=22).to(device)
    checkpoint=PRIOR/'models/rohm/checkpoints/posenet_checkpoint/model000200000.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval()
    diffusion=create_gaussian_diffusion(SimpleNamespace(noise_schedule=cfg['noise_schedule'],sigma_small=cfg['sigma_small']),gd,SpacedDiffusionPoseNet,num_diffusion_timesteps=cfg['diffusion_steps'],device=device)
    batch={'cond':condition}
    output=diffusion.p_sample_loop(model,batch,condition.shape,device=device,progress=True,clip_denoised=False,cond_fn_with_grad=False,early_stop=False)
    assert torch.isfinite(output).all()
    representation=output[0,:,0].T.detach().cpu().numpy()*data['std']+data['mean']
    start=sum(REPR_DIM_DICT[k] for k in REPR_LIST[:REPR_LIST.index('smplx_body_pose_6d')])
    rotation6d=representation[:,start:start+126]
    matrices=rot6d_to_rotmat(torch.tensor(rotation6d.reshape(-1,6),dtype=torch.float32)).numpy()
    raw_pose=Rotation.from_matrix(matrices).as_rotvec().reshape(-1,21,3)
    times=data['times_30fps'];target=data['original_times'];available=(target>=times[0])&(target<=times[-1])
    original=data['original_body_pose'].reshape(-1,21,3);resampled=original.copy()
    for j in range(21):resampled[available,j]=Slerp(times,Rotation.from_rotvec(raw_pose[:,j]))(target[available]).as_rotvec()
    # Protect every trusted original limb rotation; do not replace torso/head/hands.
    state=load_file(str(Path(prep['source'])/'motion/motion.safetensors'))
    from optimize_motion import supported_points
    protocol=json.loads((ROOT/'protocol_v1.json').read_text())
    visible,_=supported_points(state['observed_keypoints_2d'].numpy(),np.load(Path(prep['source'])/'input/target_masks.npz')['masks'],protocol['visibility'])
    fusion=original.copy();changed=np.zeros(original.shape[:2],dtype=bool)
    for joint,child in protocol['O1']['changed_joint_indices_body_pose_to_coco_child'].items():
        j=int(joint);use=(~visible[:,child])&available;fusion[use,j]=resampled[use,j];changed[use,j]=True
    np.testing.assert_array_equal(fusion[~changed],original[~changed])
    save_file({'raw_pose_30fps':torch.from_numpy(raw_pose.astype(np.float32)),'raw_proposal_25fps':torch.from_numpy(resampled.reshape(-1,63).astype(np.float32)),
      'gap_proposal_25fps':torch.from_numpy(fusion.reshape(-1,63).astype(np.float32)),'changed_joint_mask':torch.from_numpy(changed),
      'proposal_available':torch.from_numpy(available),'times_30fps':torch.from_numpy(times),'times_25fps':torch.from_numpy(target),
      'original_betas':torch.from_numpy(data['original_betas']),'observed_keypoints_2d':state['observed_keypoints_2d']},str(out/'proposal.safetensors'))
    angle_change=(Rotation.from_rotvec(fusion.reshape(-1,3))*Rotation.from_rotvec(original.reshape(-1,3)).inv()).magnitude().reshape(len(target),21)
    report={'state':'real_pretrained_pose_prior_sampled_needs_camera_alignment','sample':args.sample,'gpu_uuid':expected,'sampling_steps':cfg['diffusion_steps'],'strict_loaded':True,
      'proposal_scope':'PoseNet only, not full RoHM trajectory refinement or source-camera-guided sampling','protocol_sha256':hashlib.sha256(prior_protocol.read_bytes()).hexdigest(),
      'weights_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'condition_sha256':hashlib.sha256((out/'conditioning.npz').read_bytes()).hexdigest(),
      'changed_joint_frames':int(changed.sum()),'unsupported_output_slots':np.flatnonzero(~available).tolist(),'trusted_and_unselected_rotations_unchanged':True,
      'max_proposed_rotation_change_deg':float(np.rad2deg(angle_change.max())),'elapsed_seconds':time.time()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
      'quality_passed':False,'limitations':['No trusted world trajectory replaced.','Unobserved endpoint remains original estimate and explicitly unavailable.','Per-frame measured camera alignment and source-motion review still required.','Plausible sampled pose is not unique hidden ground truth.']}
    (out/'sample_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
