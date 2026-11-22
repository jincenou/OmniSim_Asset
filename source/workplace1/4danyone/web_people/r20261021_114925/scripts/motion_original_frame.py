"""Input adapter v2: native-frame known boxes/K; crop only after body inference.

Keeps the existing static GVHMR model and G0 normalization. No camera motion
estimator, trajectory repair, pose smoothing or new motion prior is added.
"""
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
import torch
from prepare_person import ROOT,PROJECT,selected,save,sha,encode
from source_io import read_source,source_identity
sys.path.insert(0,str(PROJECT/'repo'))
from fdanyone.motion import gvhmr
from fdanyone.motion.result import MotionResult,SMPL_PARAMETER_NAMES
from fdanyone.video import decode_canonical_clip

def transform_points(A,uv):
    h=np.concatenate([uv,np.ones((*uv.shape[:-1],1),dtype=uv.dtype)],axis=-1)
    out=np.einsum('tij,tnj->tni',A,h)
    return out[...,:2]/out[...,2:]

def run(ident):
    choice=selected(ident)
    assert choice['camera_class'] in ('static_verified','approximately_static'), 'moving/unknown source stays pending; no implicit static processing'
    source=Path(choice['source_path']);assert source_identity(source)==choice['source_identity']
    torch.set_num_threads(4)
    assert torch.cuda.device_count()==1
    actual=str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    assert actual==os.environ['FD_EXPECTED_GPU_UUID'].removeprefix('GPU-')
    dest=ROOT/'inputs'/ident
    assert json.loads((dest/'research/batch_gate_mask.json').read_text())['passed']
    mapping=json.loads((dest/'inputs/frame_mapping.json').read_text());rows=mapping['frames']
    assert sha(dest/'inputs/main.mp4')==mapping['video_sha256']
    clip=decode_canonical_clip(dest/'inputs/main.mp4',num_frames=121,start_time=0,fps=Fraction(25))
    work=dest/'motion_runtime';work.mkdir(exist_ok=False)
    wanted={r['source_frame_index']:i for i,r in enumerate(rows)};images=[None]*121
    for index,timestamp,rgb in read_source(source):
        if index in wanted:
            slot=wanted[index];assert abs(timestamp-rows[slot]['source_timestamp_s'])<1e-7
            images[slot]=Image.fromarray(rgb)
        if index>=max(wanted):break
    assert all(im is not None for im in images)
    width,height=images[0].size;assert all(im.size==(width,height) for im in images)
    original=work/'original_full_frame.mp4';encode(original,images,lossless=True)
    full_clip=decode_canonical_clip(original,num_frames=121,start_time=0,fps=Fraction(25))
    assert all(np.array_equal(frame.rgb,np.asarray(im)) for frame,im in zip(full_clip.frames,images,strict=True))
    boxes=torch.tensor([r['bbox_source_xyxy'] for r in rows],dtype=torch.float32)
    A=np.asarray([r['source_to_crop'] for r in rows],dtype=np.float64)
    crop_boxes=np.asarray([r['bbox_input_xyxy'] for r in rows])
    projected_boxes=transform_points(A,boxes.numpy().reshape(121,2,2)).reshape(121,4)
    assert np.abs(projected_boxes-crop_boxes).max()<1e-3
    root,revision=gvhmr.validate_gvhmr(PROJECT/'repo/third_party/GVHMR')
    with gvhmr.gvhmr_imports(root),gvhmr._legacy_checkpoint_loading():
        gvhmr.install_pytorch3d_compat();gvhmr._register_inference_store()
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
            cfg=compose(config_name='demo',overrides=[gvhmr.hydra_override('video_name',ident),
                'static_cam=true','verbose=false','use_dpvo=false',gvhmr.hydra_override('output_root',work/'runtime')])
        with open_dict(cfg):cfg.video_path=str(original)
        Path(cfg.preprocess_dir).mkdir(parents=True,exist_ok=False)
        gvhmr._verify_gvhmr_decode(full_clip,original,get_video_reader)
        bbx=get_bbx_xys_from_xyxy(boxes,base_enlarge=1.2).float()
        extractor=VitPoseExtractor();kp=extractor.extract(str(original),bbx);del extractor
        extractor=Extractor();features=extractor.extract_video_features(str(original),bbx);del extractor
        K=estimate_K(width,height).float().repeat(121,1,1)
        rotation=torch.eye(3).repeat(121,1,1);angular=compute_cam_angvel(rotation)
        data={'length':torch.tensor(121),'bbx_xys':bbx,'kp2d':kp,'f_imgseq':features,'K_fullimg':K,'cam_angvel':angular}
        assert kp.shape==(121,17,3) and all(bool(torch.isfinite(t).all()) for t in data.values())
        # Exact objects passed to predict, not reconstructed camera metadata.
        torch.save(detach_to_cpu(data),work/'actual_predict_inputs.pt')
        model=hydra.utils.instantiate(cfg.model,_recursive_=False)
        checkpoint=torch.load(cfg.ckpt_path,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
        model=model.eval().cuda()
        with torch.inference_mode():prediction=detach_to_cpu(model.predict(data,static_cam=True))
        assert torch.equal(prediction['K_fullimg'],K)
        torch.save(prediction,work/'original_prediction.pt')
        del model,features,data;torch.cuda.empty_cache()
    K_original=K.numpy();K_crop=(A@K_original).astype(np.float32)
    observed=kp.detach().cpu().numpy();observed_crop=observed.copy()
    observed_crop[...,:2]=transform_points(A,observed[...,:2])
    result=MotionResult(gvhmr_revision=revision,fps=clip.fps,
        frame_timestamps_sec=tuple(float(f.canonical_timestamp) for f in clip.frames),
        source_frame_indices=tuple(f.source_index for f in clip.frames),source_pts=tuple(f.source_pts for f in clip.frames),
        source_size_bytes=clip.source_size_bytes,source_mtime_ns=clip.source_mtime_ns,image_height=clip.height,image_width=clip.width,
        smpl_params_global={k:prediction['smpl_params_global'][k] for k in SMPL_PARAMETER_NAMES},
        smpl_params_incam={k:prediction['smpl_params_incam'][k] for k in SMPL_PARAMETER_NAMES},
        K_fullimg=torch.from_numpy(K_crop),observed_keypoints_2d=torch.from_numpy(observed_crop))
    result.validate_against_clip(clip);target=dest/'motion_pair/static_estimated';result.save(target)
    from fdanyone.skeleton.pipeline import _body_geometry
    from fdanyone.skeleton.keypoints import KEYPOINT_NAMES,LINKS
    from fdanyone.assets import resolve_regressor
    sys.path.insert(0,str(ROOT/'research/baseline_evidence'))
    from geometry_precision import build
    geometry=build(_body_geometry)(result,resolve_regressor(None,model_dir=PROJECT/'repo/models'),root,'cuda:0')
    points=geometry.keypoints_incam
    assert np.isfinite(points).all() and (points[...,2]>0).all()
    original_projected=np.einsum('tij,tnj->tni',K_original,points)
    original_uv=original_projected[...,:2]/original_projected[...,2:]
    crop_projected=np.einsum('tij,tnj->tni',K_crop,points)
    crop_uv=crop_projected[...,:2]/crop_projected[...,2:]
    crop_error=float(np.abs(transform_points(A,original_uv)-crop_uv).max())
    original_error=float(np.abs(transform_points(np.linalg.inv(A),crop_uv)-original_uv).max())
    assert max(crop_error,original_error)<.01
    coco=('nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle')
    distance=np.linalg.norm(original_uv[:,[KEYPOINT_NAMES.index(n) for n in coco]]-observed[...,:2],axis=-1)
    reliable=observed[...,2]>.3;assert reliable.any()
    np.savez_compressed(dest/'research/projection.npz',A_t=A,K_original=K_original,K_crop=K_crop,
        skeleton_original_uv=original_uv,skeleton_crop_uv=crop_uv,observed_original=observed,
        observed_crop=observed_crop,source_boxes=boxes.numpy(),crop_boxes=crop_boxes)
    panels=[];contacts=[]
    for frame,image in enumerate(images):
        overlay=image.copy();draw=ImageDraw.Draw(overlay)
        for _,first,second,_,major in LINKS:
            if major:draw.line([tuple(original_uv[frame,first]),tuple(original_uv[frame,second])],fill='lime',width=3)
        for u,v,confidence in observed[frame]:
            if confidence>.3:draw.ellipse((u-3,v-3,u+3,v+3),fill='red')
        draw.text((12,12),f'{ident} ESTIMATED POSE on REAL SOURCE f={frame}; green=SMPL, red=detector',fill='white',stroke_width=2,stroke_fill='black')
        preview=overlay.resize((960,int(round(height*960/width/2)*2)));panels.append(preview)
        if frame in (0,30,60,90,120):contacts.append(preview)
    encode(dest/'pose_preview.mp4',panels)
    sheet=Image.new('RGB',(960,panels[0].height*5))
    for i,im in enumerate(contacts):sheet.paste(im,(0,i*im.height))
    sheet.save(dest/'research/pose_source_contact.jpg',quality=94)
    record=dict(state='coordinate_checks_passed_visual_gate_pending',version='motion_original_frame_v2',
        camera_class=choice['camera_class'],static_cam=True,strict_checkpoint_load=True,gvhmr_input='original full frame + tracked target boxes',
        estimated_K=K_original[0].tolist(),original_size=[width,height],appearance_size=[clip.width,clip.height],
        actual_predict_inputs_sha256=sha(work/'actual_predict_inputs.pt'),original_video_sha256=sha(original),
        crop_projection_equivalence_max_abs_px=crop_error,inverse_crop_projection_max_abs_px=original_error,
        original_detector_reprojection_median_px=float(np.median(distance[reliable])),
        original_detector_reprojection_p95_px=float(np.percentile(distance[reliable],95)),
        source_camera_translation_used=False,source_camera_rotation_estimated=False,
        crop_handling='Model inference uses original K/boxes/keypoints. After inference only K and 2D observations are mapped by A_t into appearance crop; 3D parameters are unchanged.',
        limitations=['Estimated focal length, no metric camera calibration','Static assumption is approximate for approximately_static sources','Detector residual is not pose ground truth','No Waymo world extrinsics or repeated global transform'],
        visual_review='pending',generation_allowed=False)
    save(dest/'research/camera_interface.json',record)
    save(dest/'research/motion_gate.json',dict(input_sha256=sha(dest/'inputs/main.mp4'),
        kind='coordinate_interface_only_visual_review_pending',visual_review='pending',
        variants={'static_estimated':dict(allow_exploratory_generation=False,motion_sha256=sha(target/'motion.safetensors'))},
        camera_interface='camera_interface.json'))
    print(ident,'motion coordinates checked; original-video overlay must be reviewed before generation',flush=True)
