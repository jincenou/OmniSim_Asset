"""Explicit3D-box-assisted root trajectory and initialization experiment."""
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from optimize_motion import supported_points, transform_smpl_root, _gvhmr_geometry_context, REPO
G=Path(__file__).resolve().parent
E=G/'text_motion/bilateral_context/repair_v1'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('branch',choices=['no_text','text'])
    parser.add_argument('arm',choices=['box_assisted'])
    cli=parser.parse_args()
    torch.set_num_threads(2)
    protocol=E/'BOX_PROTOCOL.json'
    digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert digest==protocol.with_suffix('.sha256').read_text().strip()
    cfg=json.loads(protocol.read_text())
    for path,checksum in cfg['code_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==checksum,path
    device=torch.device('cuda');started=time.time()
    import os
    assert torch.cuda.device_count()==1
    assert str(torch.cuda.get_device_properties(0).uuid)==os.environ['FD_EXPECTED_GPU_UUID'].removeprefix('GPU-')
    p=dict(cfg['O2']);p.update(cfg['arms'][cli.arm]);weights=p['weights']
    source=E/'source';out=E/'cases/P18'/cli.branch/cli.arm
    out.mkdir(exist_ok=False)
    state=load_file(str(source/'motion/motion.safetensors'))
    original={k:state['smpl_params_incam.'+k].numpy() for k in ['body_pose','betas','global_orient','transl']}
    kp=state['observed_keypoints_2d'].numpy();n=len(kp)
    cameras=np.load(source/'input/measured_cameras.npz')
    c=cameras['camera_to_waymo_world'].copy();times=cameras['relative_exposure_seconds'];origin=c[0,:3,3].copy();c[:,:3,3]-=origin
    masks=np.load(source/'input/target_masks.npz')['masks']
    reliable,support=supported_points(kp,masks,cfg['visibility'])
    mapping=json.loads((source/'input/frame_mapping.json').read_text())['frames']
    height=np.array([f['bbox_input_xyxy'][3]-f['bbox_input_xyxy'][1] for f in mapping])
    body=original['body_pose'].reshape(n,21,3)
    pose_evidence=np.repeat(reliable[:,5:7].all(1)[:,None],21,axis=1)
    for joint,child in cfg['O1']['changed_joint_indices_body_pose_to_coco_child'].items():
        pose_evidence[:,int(joint)]=reliable[:,child]
    prior=load_file(str(E/'cases/P18'/cli.branch/'retargeted.safetensors'))
    proposed=prior['gap_proposal_25fps'].numpy().reshape(n,21,3);changed=prior['changed_joint_mask'].numpy()
    np.testing.assert_array_equal(proposed[~changed],body[~changed])
    args=SimpleNamespace(variant='condmdi');gaps=[]
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
        box_data=np.load(E/'box_prior.npz')
        np.testing.assert_allclose(box_data['times_seconds'],times,atol=1e-8,rtol=0)
        box_world=box_data['box_center_world']-origin
        trusted=reliable[:,[5,6,11,12]].all(1)&box_data['supported']
        if trusted.sum()<6:raise ValueError('Insufficient trusted root observations for constant offset')
        offset=np.median(root_t[trusted]-box_world[trusted],axis=0)
        root_t=box_world+offset
        box_target=tensor((box_world-box_world.mean(0))/scale)
        box_weight=tensor(box_data['supported'])

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
                centered=position-position.mean(0,keepdim=True)
                box_term=(torch.nn.functional.huber_loss(centered,box_target,
                    delta=cfg['box_huber_delta_bodyheights'],reduction='none').sum(-1)*box_weight).sum()/box_weight.sum()
                loss=loss+cfg['box_relative_position_weight']*box_term
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
    report=dict(state='aligned_needs_motion_review',branch=cli.branch,arm=cli.arm,source=str(source),
                protocol_sha256=digest,iterations=p['root_steps']+p['joint_steps'],
                before=before,after=after,trace=trace,body_scale_m=scale,
                supported_points=int(reliable.sum()),unknown_points=int(reliable.size-reliable.sum()),
                unknown_pose_anchor_fraction=p['unknown_pose_anchor_fraction'],
                temporal_correction_unknown_weight=p['temporal_correction_unknown_weight'],
                independently_verified_contact_frames=0,contact_loss_active=False,
                elapsed_seconds=time.time()-started,quality_approved=False,
                box_supervision_used=True,box_relative_position_weight=cfg['box_relative_position_weight'],
                limits='3D-box-assisted branch. Box agreement is now training/optimization fit, not independent validation; no pose/heading/contact truth.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
