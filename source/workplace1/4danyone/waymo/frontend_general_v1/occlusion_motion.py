"""Known-target GVHMR pair: static estimated camera versus calibrated motion.

The supplied camera rotations bypass visual odometry. Global predictions remain
GVHMR estimates, not metric registration to the Waymo world frame.
"""
from fractions import Fraction
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
import torch
sys.path.insert(0, '/mnt/ojc/workplace1/4danyone/waymo/scene20/r20261019/scripts')
from moving_prepare import R,save,sha
import os
ROOT=Path(os.environ["FDANYONE_MOVING_ROOT"])
from moving_interpolate import device_check

REPO=Path(os.environ['FDANYONE_RUNTIME']);COUNT=int(os.environ['FDANYONE_FRAMES']);sys.path[:0]=[str(REPO),'/mnt/ojc/workplace1/4danyone/waymo/scripts']
from fdanyone.motion import gvhmr
from fdanyone.motion.result import MotionResult,SMPL_PARAMETER_NAMES
from fdanyone.video import decode_canonical_clip,write_gvhmr_video
from fdanyone.skeleton.pipeline import _body_geometry
from fdanyone.skeleton.keypoints import KEYPOINT_NAMES,LINKS
from fdanyone.assets import resolve_regressor
from prepare import encode

COCO_NAMES=('nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder',
    'left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle')


def main():
    torch.set_num_threads(4);device=device_check()
    root,revision=gvhmr.validate_gvhmr(REPO/'third_party/GVHMR')
    mapping=json.loads((ROOT/'inputs/frame_mapping.json').read_text())
    if sha(ROOT/'inputs/main.mp4')!=mapping['video_sha256'] or sha(ROOT/'inputs/cameras.npz')!=mapping['camera_npz_sha256']:
        raise ValueError('Input or camera data changed')
    out=ROOT/'motion_pair';out.mkdir(exist_ok=False);save(out/'device.json',device)
    clip=decode_canonical_clip(ROOT/'inputs/main.mp4',num_frames=COUNT,start_time=0,fps=Fraction(25))
    clip.write_metadata(out/'canonical_clip.json');video=write_gvhmr_video(clip,out/'canonical_clip.mp4')
    camera=np.load(ROOT/'inputs/cameras.npz');boxes=torch.tensor([x['bbox_input_xyxy'] for x in mapping['frames']],dtype=torch.float32)
    results={}
    with gvhmr.gvhmr_imports(root),gvhmr._legacy_checkpoint_loading():
        gvhmr.install_pytorch3d_compat()
        # Register optional imports before importing preproc.__init__. We supply
        # measured camera rotations; SimpleVO remains unavailable and is not called.
        gvhmr._register_inference_store()
        import hydra
        from hydra import compose,initialize_config_module
        from omegaconf import open_dict
        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy,estimate_K
        from hmr4d.utils.geo_transform import compute_cam_angvel
        from hmr4d.utils.preproc.vitpose import VitPoseExtractor
        from hmr4d.utils.preproc.vitfeat_extractor import Extractor
        from hmr4d.utils.net_utils import detach_to_cpu
        from hmr4d.utils.video_io_utils import get_video_reader
        with initialize_config_module(version_base='1.3',config_module='hmr4d.configs'):
            cfg=compose(config_name='demo',overrides=[gvhmr.hydra_override('video_name','calibrated'),
                'static_cam=false','verbose=false','use_dpvo=false',gvhmr.hydra_override('output_root',out/'runtime')])
        with open_dict(cfg):cfg.video_path=str(video)
        Path(cfg.preprocess_dir).mkdir(parents=True,exist_ok=False)
        gvhmr._verify_gvhmr_decode(clip,video,get_video_reader)
        bbx_xys=get_bbx_xys_from_xyxy(boxes,base_enlarge=1.2).float()
        torch.save({'bbx_xyxy':boxes,'bbx_xys':bbx_xys},cfg.paths.bbx)
        # Dataset identity boxes are explicit inputs; no tracker substitution.
        branch=ROOT.name
        case=json.loads((ROOT.parent/'case.json').read_text())
        if branch=='kp_only':
            cached=Path(case['original_source'])/'motion_pair/runtime/calibrated/preprocess'
            keypoints=torch.load(cached/'vitpose.pt',map_location='cpu',weights_only=True).clone()
            features=torch.load(cached/'vit_features.pt',map_location='cpu',weights_only=True)
            feature_source=str(cached/'vit_features.pt')
        elif branch=='rgb':
            extractor=VitPoseExtractor();keypoints=extractor.extract(str(video),bbx_xys);del extractor
            extractor=Extractor();features=extractor.extract_video_features(str(video),bbx_xys);del extractor
            feature_source='fresh occluded canonical_clip.mp4; no original feature load'
        else:raise ValueError('Unknown controlled branch')
        raw_keypoints=keypoints.clone();torch.save(raw_keypoints,out/'detector_before_known_occluder.pt')
        occ=np.load(ROOT.parent/'evidence_masks.npz')['occluders']
        xy=np.rint(keypoints[...,:2].cpu().numpy()).astype(int);hh,ww=occ.shape[1:]
        inside=(xy[...,0]>=0)&(xy[...,0]<ww)&(xy[...,1]>=0)&(xy[...,1]<hh)
        hidden=inside&(occ[np.arange(len(occ))[:,None],xy[...,1].clip(0,hh-1),xy[...,0].clip(0,ww-1)]>0)
        keypoints[torch.from_numpy(hidden).to(keypoints.device)]=0
        torch.save(keypoints,cfg.paths.vitpose);torch.save(features,cfg.paths.vit_features);torch.cuda.empty_cache()
        save(out/'feature_provenance.json',{'branch':branch,'feature_source':feature_source,
            'input_video_sha256':sha(video),'keypoints_sha256':sha(Path(cfg.paths.vitpose)),
            'features_sha256':sha(Path(cfg.paths.vit_features)),'suppressed_keypoints':int(hidden.sum()),
            'known_occluder_is_oracle_test_condition':True,
            'original_unoccluded_visual_features_consumed':branch=='kp_only',
            'RGB_branch_recomputes_all_features':branch=='rgb'})
        if tuple(keypoints.shape)!=(COUNT,17,3):raise ValueError('Unexpected detector keypoint format')
        model=hydra.utils.instantiate(cfg.model,_recursive_=False)
        checkpoint=torch.load(cfg.ckpt_path,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
        model=model.eval().cuda()
        shared={'length':torch.tensor(COUNT),'bbx_xys':bbx_xys,'kp2d':keypoints,'f_imgseq':features}
        variants={'calibrated_moving':{'K_fullimg':torch.from_numpy(camera['K_input']).float(),'cam_angvel':compute_cam_angvel(torch.from_numpy(camera['R_world_to_cv_camera']).float())}}
        for name,calibration in variants.items():
            if not all(torch.isfinite(v).all() for v in calibration.values()):raise ValueError('Invalid camera tensors')
            with torch.inference_mode():prediction=detach_to_cpu(model.predict({**shared,**calibration},static_cam=name!='calibrated_moving'))
            motion=MotionResult(gvhmr_revision=revision,fps=clip.fps,
                frame_timestamps_sec=tuple(float(f.canonical_timestamp) for f in clip.frames),
                source_frame_indices=tuple(f.source_index for f in clip.frames),source_pts=tuple(f.source_pts for f in clip.frames),
                source_size_bytes=clip.source_size_bytes,source_mtime_ns=clip.source_mtime_ns,
                image_height=clip.height,image_width=clip.width,
                smpl_params_global={k:prediction['smpl_params_global'][k] for k in SMPL_PARAMETER_NAMES},
                smpl_params_incam={k:prediction['smpl_params_incam'][k] for k in SMPL_PARAMETER_NAMES},
                K_fullimg=prediction['K_fullimg'],observed_keypoints_2d=keypoints.detach().cpu())
            motion.validate_against_clip(clip);motion.save(out/name);results[name]=motion
            save(out/name/'calibration.json',{'static_cam':name!='calibrated_moving','strict_checkpoint_load':True,
                'motion_coordinate_world':'gvhmr_gravity_aligned_y_up; not registered Waymo world',
                'camera_translation_consumed':False,'camera_npz':str(ROOT/'inputs/cameras.npz') if name!='static_estimated' else None,
                'source_video_sha256':mapping['video_sha256'],'known_target_boxes':True,'tracker_used':False})
        del model,features,shared;torch.cuda.empty_cache()
    projected={};metrics={};observed=keypoints.detach().cpu().numpy();coco_ids=[KEYPOINT_NAMES.index(n) for n in COCO_NAMES]
    for name,motion in results.items():
        geometry=_body_geometry(motion,resolve_regressor(None,model_dir=REPO/'models'),root,'cuda:0')
        points=geometry.keypoints_incam;K=motion.K_fullimg.numpy();xyz=np.einsum('tij,tnj->tni',K,points)
        if not np.isfinite(xyz).all() or (points[...,2]<=0).any():raise ValueError('Nonfinite or behind-camera body projection')
        uv=xyz[...,:2]/xyz[...,2:];projected[name]=uv
        distance=np.linalg.norm(uv[:,coco_ids]-observed[...,:2],axis=-1)
        trusted=observed[...,2]>.3;heights=boxes.numpy()[:,3]-boxes.numpy()[:,1]
        if not trusted.any():raise ValueError('No confident source detections')
        normalized=distance/heights[:,None]
        metrics[name]={'median_detector_reprojection_px':float(np.median(distance[trusted])),
            'p95_detector_reprojection_px':float(np.percentile(distance[trusted],95)),
            'median_detector_reprojection_over_bbox_height':float(np.median(normalized[trusted])),
            'p95_detector_reprojection_over_bbox_height':float(np.percentile(normalized[trusted],95)),
            'confident_points_per_frame_min_median':[int(trusted.sum(1).min()),float(np.median(trusted.sum(1)))],
            'global_root_span_xyz':np.ptp(motion.smpl_params_global['transl'].numpy(),axis=0).tolist(),
            'diagnostic_only':'Error versus detected COCO17 points, not ground-truth body pose; manual overlay required'}
        np.savez_compressed(out/name/'projection.npz',keypoints_uv=uv,observed_coco17=observed,
            confidence_mask=trusted,detector_distance_px=distance,source_boxes=boxes.numpy())
    panels=[];contacts=[]
    for frame in range(COUNT):
        images=[]
        for name in ('source','calibrated_moving'):
            im=Image.fromarray(clip.frames[frame].rgb.copy());draw=ImageDraw.Draw(im)
            if name!='source':
                uv=projected[name][frame]
                for _,first,second,_,major in LINKS:
                    if major:draw.line([tuple(uv[first]),tuple(uv[second])],fill='lime',width=2)
                for u,v,confidence in observed[frame]:
                    if confidence>.3:draw.ellipse((u-2,v-2,u+2,v+2),fill='red')
            draw.rectangle((0,0,clip.width,22),fill='black');draw.text((4,4),f'{name} f{frame}',fill='white');images.append(im)
        sheet=Image.new('RGB',(clip.width*2,clip.height))
        for col,im in enumerate(images):sheet.paste(im,(col*clip.width,0))
        panels.append(sheet)
        if frame in np.rint(np.linspace(0,COUNT-1,5)).astype(int):contacts.append(sheet)
    encode(out/'projection_comparison.mp4',panels,25)
    import av
    with av.open(str(out/'projection_comparison.mp4')) as container:
        decoded=list(container.decode(video=0))
        if len(decoded)!=COUNT or any(f.width!=clip.width*2 or f.height!=clip.height for f in decoded):raise ValueError('Incomplete projection video')
    contact=Image.new('RGB',(clip.width*2,clip.height*len(contacts)))
    for i,im in enumerate(contacts):contact.paste(im,(0,clip.height*i))
    contact.save(out/'projection_contact.jpg',quality=95)
    save(out/'comparison.json',{'state':'synthetic_occlusion_motion_inferred_review_pending','device':device,
        'gvhmr_revision':revision,'input_sha256':mapping['video_sha256'],'metrics':metrics,
        'changes':['synthetic RGB occlusion or keypoint-only masking as recorded in feature_provenance.json'],
        'same':['target boxes','camera calibration','checkpoint','timestamps'],
        'video_sha256':sha(out/'projection_comparison.mp4'),'video_frames_decoded':COUNT,
        'color_legend':{'green':'regressed SMPL-X skeleton projection','red':'ViTPose detections'},
        'generation_started':False,'source_projection_approved':False,
        'limitations':['Controlled synthetic occlusion; known occluder mask; kp_only retains original image evidence; not a real-world occlusion detector',
            'Camera translation is not consumed by GVHMR and world metric trajectory is not verified',
            'Single exposure-center pose approximates rolling shutter; human fitting accuracy remains to be inspected']})
    print(json.dumps({'state':'motion_pair_ready','metrics':metrics}),flush=True)


if __name__=='__main__':main()
