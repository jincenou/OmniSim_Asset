"""Run the released motion/conditioning stages with retained audit artifacts."""
import json
import os
import subprocess
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('FDANYONE_WAYMO_ROOT',str(BASE))).resolve()
REPO = BASE / 'short_tracks/repo'
sys.path.insert(0, str(REPO))
from fdanyone.video import decode_canonical_clip, write_gvhmr_video
from fdanyone.motion import gvhmr
from fdanyone.skeleton import pipeline as skeleton
from fdanyone.views import resolve_view_plan
from fdanyone.assets import resolve_foreground_model, resolve_regressor


def main():
    import torch
    visible=os.environ['CUDA_VISIBLE_DEVICES']
    physical=int(visible)
    assert physical in (4,5,6,7),visible
    expected_uuid=subprocess.check_output(['nvidia-smi',f'--id={physical}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    # Initialize the CUDA runtime before Ultralytics can reinterpret logical 0
    # as a new physical visibility mask. Verify the actual device, not the label.
    torch.cuda.init()
    actual_uuid=str(torch.cuda.get_device_properties(0).uuid)
    assert actual_uuid==expected_uuid,(actual_uuid,expected_uuid)
    out = ROOT / 'preflight'
    out.mkdir(exist_ok=False)
    (out/'device.json').write_text(json.dumps({'physical_gpu':physical,'cuda_visible_devices_at_start':visible,'actual_cuda_uuid':actual_uuid,'pid':os.getpid()},indent=2)+'\n')
    clip = decode_canonical_clip(ROOT/'inputs/main.mp4', num_frames=21, start_time=0, fps=Fraction(25))
    clip.write_metadata(out/'canonical_clip.json')
    video = write_gvhmr_video(clip, out/'canonical_clip.mp4')
    # Explicit experiment policy: tracking failure must abort instead of using a full-frame box.
    gvhmr._is_empty_tracker_error = lambda error: False
    try:
        motion = gvhmr.run_gvhmr(clip=clip, working_video=video, output_dir=out/'runtime', gvhmr_root=REPO/'third_party/GVHMR', device='cuda:0')
    finally:
        os.environ['CUDA_VISIBLE_DEVICES']=visible
    assert str(torch.cuda.get_device_properties(0).uuid)==expected_uuid
    motion.save(out/'motion')
    bbpaths = list((out/'runtime').rglob('bbx.pt'))
    if len(bbpaths)!=1:
        raise RuntimeError(f'Expected one bbox file, found {bbpaths}')
    boxes = torch.load(bbpaths[0], weights_only=True)['bbx_xyxy'].numpy()
    ref = np.array([x['bbox_input_xyxy'] for x in json.loads((ROOT/'inputs/frame_mapping.json').read_text())['frames']])
    inter = np.maximum(0, np.minimum(boxes[:,2:],ref[:,2:])-np.maximum(boxes[:,:2],ref[:,:2])).prod(1)
    union = (boxes[:,2:]-boxes[:,:2]).prod(1)+(ref[:,2:]-ref[:,:2]).prod(1)-inter
    iou = inter/union
    kp = motion.observed_keypoints_2d.numpy()
    checks = {'bbox_iou_min':float(iou.min()),'bbox_iou_median':float(np.median(iou)),
              'bbox_iou_below_0_5_frames':int((iou<.5).sum()),'keypoint_confidence_median':float(np.median(kp[...,2])),
              'physical_gpu':physical,'actual_cuda_uuid':actual_uuid,'frames':len(clip.frames),'fps':str(clip.fps),'full_frame_substitution':'disabled explicitly',
              'motion_camera_mode':'official static, estimated intrinsics; real Waymo K recorded but not injected'}
    (out/'tracking_check.json').write_text(json.dumps(checks,indent=2)+'\n')
    if np.median(iou)<.65 or (iou<.5).sum()>1:
        raise RuntimeError(f'Target identity/box overlap gate failed: {checks}')
    original_masks = skeleton.predict_foreground_masks
    def retain_masks(frames, model_path, device):
        masks = original_masks(frames,model_path,device)
        np.savez_compressed(out/'source_masks.npz',masks=masks)
        return masks
    skeleton.predict_foreground_masks = retain_masks
    views = resolve_view_plan(views_per_layer=6,layer_pitches=[15],start_yaw=0,yaw_span=360,enable_rcp=True,enable_tcr=True)
    skeleton.build_skeleton_conditioning(motion=motion,clip=clip,regressor_path=resolve_regressor(None,model_dir=REPO/'models'),
        foreground_model_path=resolve_foreground_model(REPO/'models'),gvhmr_root=REPO/'third_party/GVHMR',
        output_dir=out/'conditioning',device='cuda:0',view_plan=views)
    masks = np.load(out/'source_masks.npz')['masks']
    width,height=clip.width,clip.height
    sheet = Image.new('RGB',(width*5,height*2),(24,24,24))
    for col,idx in enumerate([0,5,10,15,20]):
        im=Image.fromarray(clip.frames[idx].rgb)
        draw=ImageDraw.Draw(im);draw.rectangle(tuple(ref[idx]),outline='green',width=2);draw.rectangle(tuple(boxes[idx]),outline='yellow',width=2)
        for x,y,c in kp[idx]:
            if c>.3:draw.ellipse((x-2,y-2,x+2,y+2),fill='red')
        draw.text((4,4),f'Frame {idx} IoU {iou[idx]:.2f}',fill='white')
        sheet.paste(im,(col*width,0));sheet.paste(Image.fromarray(masks[idx]).convert('RGB'),(col*width,height))
    sheet.save(ROOT/'previews/motion_masks.jpg')
    checks['state']='motion_and_conditioning_completed_visual_review_required'
    (out/'preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
