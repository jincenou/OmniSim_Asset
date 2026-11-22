"""Prompt the selected pedestrian, exclude the touching pipe, propagate one ID."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

BASE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('FDANYONE_WAYMO_ROOT',str(BASE))).resolve()
sys.path.insert(0,str(BASE/'segmentation'))
from sam2.build_sam import build_sam2_video_predictor


def main():
    assert torch.cuda.is_available()
    mapping=json.loads((ROOT/'inputs_rife/frame_mapping.json').read_text())
    points=np.array([[135,96],[125,180],[111,286],[112,390],[201,205],[201,260],[30,230],[240,350]],np.float32)
    labels=np.array([1,1,1,1,0,0,0,0],np.int32)
    prompt_path=ROOT/'segmentation/prompt.json'
    prompt=json.loads(prompt_path.read_text()) if prompt_path.exists() else None
    if prompt is not None:
        points=np.array(prompt['points'],np.float32);labels=np.array(prompt['labels'],np.int32)
    box=np.array(mapping['frames'][0]['bbox_input_xyxy'],np.float32)
    crop=mapping['frames'][0]['crop_xyxy'];width,height=crop[2]-crop[0],crop[3]-crop[1]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(BASE/'segmentation/sam2.1_hiera_large.pt'),device='cuda',apply_postprocessing=False)
        state=predictor.init_state(video_path=str(ROOT/'segmentation/frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
        predictor.add_new_points_or_box(state,frame_idx=0,obj_id=1,points=points,labels=labels,box=box)
        extras=[] if prompt is None else prompt.get('extra_objects',[])
        for obj in extras:
            predictor.add_new_points_or_box(state,frame_idx=0,obj_id=obj['id'],points=np.array(obj['points'],np.float32),
                labels=np.ones(len(obj['points']),np.int32),box=np.array(obj['box'],np.float32))
        masks=np.empty((121,height,width),np.uint8);seen=[]
        for frame_idx,object_ids,logits in predictor.propagate_in_video(state):
            assert object_ids==[1]+[obj['id'] for obj in extras]
            masks[frame_idx]=logits[:,0].sigmoid().amax(0).float().cpu().numpy().__mul__(255).round().astype(np.uint8)
            seen.append(frame_idx)
    assert sorted(seen)==list(range(121))
    np.savez_compressed(ROOT/'segmentation/target_masks.npz',masks=masks)
    result={'state':'segmented_visual_review_pending','physical_gpu':int(os.environ.get('CUDA_VISIBLE_DEVICES','5').split(',')[0]),'model':'SAM2.1 Hiera Large, reused local checkpoint',
            'track_id':mapping['track_id'],'object_id':1,'seed_frame':0,'points':points.tolist(),'labels':labels.tolist(),'box':box.tolist(),
            'extra_objects':extras,'combination':'per-pixel max of explicitly prompted person and carried-object probabilities',
            'prompt_rationale':prompt['rationale'] if prompt else 'positive visible head/torso/legs; negative brown pipe/background, based on reviewed first frame',
            'frames':len(seen),'postprocessing':False,'new_detection_model':False,'temporal_propagation':True,
            'mask_resolution':[width,height],'foreground_pixels_min_max':[int((masks>127).sum((1,2)).min()),int((masks>127).sum((1,2)).max())]}
    (ROOT/'segmentation/manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
