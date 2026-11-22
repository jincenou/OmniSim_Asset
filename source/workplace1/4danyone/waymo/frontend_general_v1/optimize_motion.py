"""Frozen-protocol O1/O2 test-time motion proposals; never modify old assets.

The camera-world initialization is independent of GVHMR's native global rollout.
Only supported detections constrain reprojection. Hidden motion remains unknown.
"""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.spatial.transform import Rotation
import torch
from safetensors.torch import load_file,save_file
from camera_geometry import transform_smpl_root
from gap_proposal import bounded_rotation_gaps
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]/'repo'
PRIOR=ROOT.parent/'frontend_quality_20261020'
sys.path.insert(0,str(REPO))
from fdanyone.skeleton.pipeline import _gvhmr_geometry_context

def supported_points(kp,masks,config):
    radius=config['mask_support_radius_px'];height,width=masks.shape[1:]
    smoothed=uniform_filter((masks>=config['mask_alpha_min']).astype(float),size=(1,2*radius+1,2*radius+1),mode='constant')
    x=np.rint(kp[...,0]).astype(int);y=np.rint(kp[...,1]).astype(int)
    inside=(x>=0)&(x<width)&(y>=0)&(y<height)
    support=smoothed[np.arange(len(kp))[:,None],np.clip(y,0,height-1),np.clip(x,0,width-1)]
    reliable=inside&(kp[...,2]>=config['detector_confidence_min'])&(support>=config['mask_support_fraction_min'])
    return reliable,support

def main():
    parser=argparse.ArgumentParser();parser.add_argument('sample');parser.add_argument('--device',default='cpu');parser.add_argument('--variant',choices=['v1','root_only','local_temporal','rohm','condmdi'],default='v1');parser.add_argument('--case-config',type=Path);parser.add_argument('--extend-dev6',action='store_true');parser.add_argument('--condmdi-branch',choices=['no_text','text']);parser.add_argument('--condmdi-real',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);device=torch.device(args.device);started=time.time()
    protocol=ROOT/('protocol_v1.json' if args.variant=='v1' else 'protocol_v2.json');digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert protocol.with_suffix('.sha256').read_text().split()[0]==digest
    cfg=json.loads(protocol.read_text());p=cfg['O2'];weights=p['weights']
    cohort=json.loads((ROOT/'cohort_v1.json').read_text())
    selected=[x for x in cohort['dev6']+cohort['auxiliary_seen'] if x['sample_id']==args.sample]
    if len(selected)!=1:raise ValueError('Only development/previously seen subjects permitted before pipeline freeze')
    source=Path(selected[0]['manifest']).parent
    if args.condmdi_real:
        from condmdi_paths import frozen_protocol, case_paths
        _, real_cfg=frozen_protocol('real')
        if args.variant!='condmdi' or args.case_config is not None or args.sample not in real_cfg['cases'] or args.condmdi_branch is None:
            raise ValueError('Unregistered real text-motion case')
        real_source,real_directory=case_paths('real',args.sample)
        assert source==real_source
    if args.extend_dev6:
        extension=ROOT/'dev6_geometry_protocol.json'
        assert hashlib.sha256(extension.read_bytes()).hexdigest()==extension.with_suffix('.sha256').read_text().split()[0]
        if args.variant!='local_temporal' or args.case_config is not None or args.sample not in json.loads(extension.read_text())['subjects']:raise ValueError('Not a fixed Dev6 extension')
    if args.variant!='v1' and args.sample not in cfg['pilot_subjects'] and args.case_config is None and not args.extend_dev6 and not args.condmdi_real:raise ValueError('Outside frozen pilot')
    out=ROOT/('motion_candidates' if args.variant=='v1' else 'motion_candidates_v2')/args.sample
    if args.variant!='v1':out=out/args.variant
    case_cfg=None
    if args.case_config is not None:
        case_path=args.case_config.resolve()
        if not case_path.is_relative_to(ROOT/'occlusion_v1'):raise ValueError('Counterfactual config outside experiment')
        frozen=ROOT/'occlusion_v1/PROTOCOL.json'
        if frozen.with_suffix('.sha256').read_text().split()[0]!=hashlib.sha256(frozen.read_bytes()).hexdigest():raise ValueError('Changed counterfactual protocol')
        case_cfg=json.loads(case_path.read_text())
        if args.sample not in json.loads(frozen.read_text())['subjects'] or case_cfg['sample']!=args.sample or args.variant not in ['local_temporal','rohm','condmdi']:raise ValueError('Unregistered counterfactual')
        source=case_path.parent
        out=source/('optimized_rohm' if args.variant=='rohm' else 'optimized')
    if args.variant=='condmdi':
        frozen=ROOT/('text_motion/real/RETARGET_PROTOCOL.json' if args.condmdi_real else 'text_motion/RETARGET_PROTOCOL.json')
        assert hashlib.sha256(frozen.read_bytes()).hexdigest()==frozen.with_suffix('.sha256').read_text().strip()
        if args.condmdi_real:
            proposal_directory=real_directory/args.condmdi_branch
        else:
            if args.case_config is None or args.condmdi_branch is None or source.name!='rgb' or source.parent.name not in json.loads(frozen.read_text())['cases']:
                raise ValueError('Unregistered text-motion alignment')
            proposal_directory=ROOT/'text_motion/cases'/source.parent.name/args.condmdi_branch
        out=proposal_directory/'aligned'
    elif args.condmdi_branch is not None:
        raise ValueError('Text branch only valid for CondMDI')
    out.mkdir(parents=True,exist_ok=False)
    state=load_file(str(source/'motion/motion.safetensors'))
    original={k:state['smpl_params_incam.'+k].numpy() for k in ['body_pose','betas','global_orient','transl']}
    kp=state['observed_keypoints_2d'].numpy();n=len(kp)
    cameras=np.load(PRIOR/'motion_unmasked'/args.sample/'measured_camera_maps.npz')
    c=cameras['camera_to_waymo_world'].copy();times=cameras['relative_exposure_seconds'];origin=c[0,:3,3].copy();c[:,:3,3]-=origin
    masks=np.load(source/'input/target_masks.npz')['masks']
    reliable,support=supported_points(kp,masks,cfg['visibility'])
    mapping=json.loads((source/'input/frame_mapping.json').read_text())['frames']
    height=np.array([f['bbox_input_xyxy'][3]-f['bbox_input_xyxy'][1] for f in mapping])
    if reliable.sum()<cfg['gates']['visibility_min_evaluable_points']:raise ValueError('Insufficient reliable evidence for candidate fitting')
    body=original['body_pose'].reshape(n,21,3)
    pose_evidence=np.repeat(reliable[:,5:7].all(1)[:,None],21,axis=1)
    o1_reliable=np.ones((n,21),dtype=bool)
    for joint,child in cfg['O1']['changed_joint_indices_body_pose_to_coco_child'].items():
        pose_evidence[:,int(joint)]=reliable[:,child];o1_reliable[:,int(joint)]=reliable[:,child]
    proposed,changed,gaps=bounded_rotation_gaps(body,times,o1_reliable,cfg['O1']['max_anchor_interval_seconds'])
    if args.variant in ['rohm','condmdi']:
        if args.variant=='condmdi':
            prior_path=proposal_directory/'retargeted.safetensors'
        else:
            prior_path=ROOT/'rohm_proposals'/args.sample/'proposal.safetensors' if args.case_config is None else source/'rohm_proposal/proposal.safetensors'
        prior=load_file(str(prior_path))
        proposed=prior['gap_proposal_25fps'].numpy().reshape(n,21,3)
        changed=prior['changed_joint_mask'].numpy()
        np.testing.assert_array_equal(proposed[~changed],body[~changed])
        gaps=[]
    # Own memory: float32 NumPy prior proposals must not alias trainable Parameters.
    tensor=lambda a:torch.tensor(a,dtype=torch.float32,device=device)
    with _gvhmr_geometry_context(REPO/'third_party/GVHMR'):
        from hmr4d.utils.smplx_utils import make_smplx
        from pytorch3d.transforms import axis_angle_to_matrix
        model=make_smplx('supermotion_coco17').to(device).eval()
        model.requires_grad_(False)
        betas=tensor(original['betas'].mean(0,keepdims=True)).expand(n,-1).contiguous()
        rest=model.get_skeleton(betas).detach()
        scale=float((rest[0,:,1].max()-rest[0,:,1].min()).cpu())
        if not .5<scale<3.:raise ValueError('Invalid fixed body scale')
        root_a,root_t=transform_smpl_root(original['global_orient'],original['transl'],rest[:,0].cpu().numpy(),c)
        root=torch.nn.Parameter(tensor(root_a));position=torch.nn.Parameter(tensor(root_t/scale));pose=torch.nn.Parameter(tensor(proposed.reshape(n,63)))
        initial_root=root.detach().clone();initial_position=position.detach().clone()
        original_pose=tensor(original['body_pose']);proposal=pose.detach().clone()
        initial_rot=axis_angle_to_matrix(initial_root);original_rot=axis_angle_to_matrix(original_pose.reshape(n,21,3))
        R=tensor(c[:,:3,:3]);T=tensor(c[:,:3,3]);K=state['K_fullimg'].to(device)
        dt=tensor(np.diff(times));h=tensor(height);target=tensor(kp[...,:2])
        evidence=tensor(reliable)*tensor(kp[...,2]).square();pose_conf=tensor(pose_evidence)
        mse_rot=lambda a,b:(a-b).square().sum((-1,-2))/2
        def geometry():
            world=model(body_pose=pose,betas=betas,global_orient=root,transl=position*scale)
            cam=torch.einsum('tji,tnj->tni',R,world-T[:,None,:])
            if not torch.isfinite(cam).all() or (cam[...,2]<=.1).any():raise ValueError('Nonfinite or behind-camera candidate')
            homogeneous=torch.einsum('tij,tnj->tni',K,cam)
            uv=homogeneous[...,:2]/homogeneous[...,2:]
            return world,uv
        def diagnostics():
            with torch.no_grad():
                world,uv=geometry();error=torch.linalg.vector_norm(uv-target,dim=-1)/h[:,None]
                valid=error[tensor(reliable).bool()]
                return {'detector_nme_mean':float(valid.mean()),'detector_nme_p95':float(torch.quantile(valid,.95)),
                    'world_root_span_m':torch.ptp(position*scale,dim=0).tolist() if hasattr(torch,'ptp') else ((position.max(0).values-position.min(0).values)*scale).tolist()}
        before=diagnostics();trace=[]
        joint_steps=0 if args.variant=='root_only' else p['joint_steps']
        for phase,steps in [('root',p['root_steps']),('joint',joint_steps)]:
            pose.requires_grad_(phase=='joint')
            params=[{'params':[root],'lr':p['lr_root_orientation']},{'params':[position],'lr':p['lr_root_translation_bodyheight_units']}]
            if phase=='joint':params.append({'params':[pose],'lr':p['lr_pose']})
            optimizer=torch.optim.Adam(params)
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True);world,uv=geometry()
                residual=(uv-target)/h[:,None,None]
                reprojection=(torch.nn.functional.huber_loss(residual,torch.zeros_like(residual),delta=p['reprojection_huber_delta_bbox_height'],reduction='none').sum(-1)*evidence).sum()/evidence.sum()
                rotation=axis_angle_to_matrix(pose.reshape(n,21,3))
                anchor=(mse_rot(rotation,original_rot)*(pose_conf+(1-pose_conf)*p['unknown_pose_anchor_fraction'])).mean()
                root_anchor=torch.nn.functional.huber_loss(position,initial_position,delta=p['root_anchor_huber_delta_bodyheights'])+mse_rot(axis_angle_to_matrix(root),initial_rot).mean()
                velocity=(world[1:]-world[:-1])/dt[:,None,None]
                acceleration=(velocity[1:]-velocity[:-1])/((dt[1:]+dt[:-1])/2)[:,None,None]
                temporal=(acceleration/(scale*p['acceleration_normalizer_bodyheights_per_second2'])).square().mean()
                proposal_anchor=(mse_rot(rotation,axis_angle_to_matrix(proposal.reshape(n,21,3)))*(1-pose_conf)).mean()
                loss=weights['reprojection']*reprojection+weights['trusted_pose_anchor']*anchor+weights['root_anchor']*root_anchor+weights['world_joint_acceleration']*temporal+weights['proposal_anchor']*proposal_anchor
                if args.variant in ['local_temporal','rohm','condmdi']:
                    root_rotation=axis_angle_to_matrix(root)
                    local=torch.einsum('tji,tnj->tni',root_rotation,world-(position*scale)[:,None,:])
                    local=local-local[:,11:13].mean(1,keepdim=True)
                    lv=(local[1:]-local[:-1])/dt[:,None,None]
                    la=(lv[1:]-lv[:-1])/((dt[1:]+dt[:-1])/2)[:,None,None]
                    local_term=(la/(scale*p['acceleration_normalizer_bodyheights_per_second2'])).square().mean()
                    correction=rotation@original_rot.transpose(-1,-2)
                    cv=(correction[1:]-correction[:-1])/dt[:,None,None,None]/p['correction_velocity_normalizer_per_second']
                    cv_weight=pose_conf[1:]*pose_conf[:-1]
                    cv_weight=cv_weight+(1-cv_weight)*p['temporal_correction_unknown_weight']
                    correction_term=(cv.square().sum((-1,-2))/2*cv_weight).mean()
                    root_correction=root_rotation@initial_rot.transpose(-1,-2)
                    root_cv=(root_correction[1:]-root_correction[:-1])/dt[:,None,None]/p['correction_velocity_normalizer_per_second']
                    root_term=(root_cv.square().sum((-1,-2))/2).mean()
                    loss=loss+weights['local_joint_acceleration']*local_term+weights['pose_correction_velocity']*correction_term+weights['root_correction_velocity']*root_term
                if not torch.isfinite(loss):raise ValueError('Nonfinite optimization loss')
                loss.backward();optimizer.step()
                if step in [0,steps-1]:trace.append({'phase':phase,'step':step+1,'loss':float(loss.detach()),'reprojection':float(reprojection.detach()),'temporal':float(temporal.detach())})
        after=diagnostics()
        # Convert candidate world root back to camera space with the same rest pelvis.
        world_a=root.detach().cpu().numpy();world_t=(position*scale).detach().cpu().numpy()
        cam_a,cam_t=transform_smpl_root(world_a,world_t,rest[:,0].cpu().numpy(),np.linalg.inv(c))
        final={
            'world.global_orient':root.detach().cpu(),'world.transl_relative_origin':(position*scale).detach().cpu(),
            'incam.global_orient':torch.from_numpy(cam_a).float(),'incam.transl':torch.from_numpy(cam_t).float(),
            'body_pose':pose.detach().cpu(),'betas':betas.cpu(),'observed_keypoints_2d':state['observed_keypoints_2d'],
            'proposal_initial.body_pose':torch.from_numpy(proposed.reshape(n,63)).float(),
            'evidence_supported_visible_candidate':torch.from_numpy(reliable),'O1_changed_joint_mask':torch.from_numpy(changed),
            'camera_to_world_relative_origin':torch.from_numpy(c),'world_origin':torch.from_numpy(origin),
            'times_seconds':torch.from_numpy(times)}
        save_file({k:v.contiguous() for k,v in final.items()},str(out/'motion_candidates.safetensors'))
    report={'state':'O1_O2_parameters_created_needs_projection_and_motion_review','sample':args.sample,'variant':args.variant,'source':str(source),'device':str(device),'protocol_sha256':digest,
      'counterfactual_config':str(args.case_config) if args.case_config else None,
      'iterations':p['root_steps']+joint_steps,'weights_trained':False,'test_time_pose_optimization':True,
      'reliable_detector_points':int(reliable.sum()),'unknown_detector_points':int(reliable.size-reliable.sum()),
      'independently_verified_contact_frames':0,'contact_loss_active':False,'contact_status':'NA; no reliable independently verified support labels, no unconditional foot lock',
      'initial_proposal_kind':('CondMDI_'+args.condmdi_branch if args.variant=='condmdi' else 'RoHM_PoseNet_gap' if args.variant=='rohm' else 'O1_bounded_rotation'),'proposal_changed_joint_frames':int(changed.sum()),'O1_gaps':gaps,'before_O2':before,'after_O2':after,'loss_trace':trace,'body_scale_m':scale,
      'original_betas_max_temporal_deviation':float(np.abs(original['betas']-original['betas'].mean(0)).max()),
      'elapsed_seconds':time.time()-started,'quality_passed':False,
      'limitations':['Detector reprojection is not true pose error.','No proof of hidden motion, metric depth or stationary world trajectory.','No repaired skeleton or 24-view generation published until geometry/visibility/motion review.','Camera-origin shift only for float precision; absolute world origin preserved.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
