"""Separate carried-object mask, unioned with unchanged body mask; no recoloring."""
import json
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import av
import torch
from prepare_person import ROOT, PROJECT, encode, sha, save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--person',choices=['W02','W04'],default='W02')
    person=parser.parse_args().person
    out=ROOT/f'diagnostics/accessory_{person}_v1';plan=json.loads((out/'PLAN.json').read_text())
    assert plan['person']==person
    source=ROOT/'inputs'/person
    assert sha(source/'segmentation/target_masks.npz')==plan['original_mask_sha256']
    assert not (out/'corrected_input.mp4').exists()
    assert torch.cuda.device_count()==1
    uuid=str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    assert uuid==os.environ['FD_EXPECTED_GPU_UUID'].removeprefix('GPU-')
    sys.path.insert(0,str(PROJECT/'waymo/segmentation'))
    from sam2.build_sam import build_sam2_video_predictor
    from sam_transfer_guard import install_transfer_guard
    start=time.time()
    original=np.load(source/'segmentation/target_masks.npz')['masks']
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',
            str(PROJECT/'waymo/segmentation/sam2.1_hiera_large.pt'),device='cuda',apply_postprocessing=False)
        install_transfer_guard(predictor)
        state=predictor.init_state(video_path=str(source/'segmentation/frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
        predictor.add_new_points_or_box(state,frame_idx=0,obj_id=2,
            points=np.asarray(plan['bag_points_xy'],np.float32),labels=np.asarray(plan['bag_point_labels'],np.int32),
            box=np.asarray(plan['bag_box_xyxy'],np.float32))
        masks=[];seen=[]
        for frame,ids,logits in predictor.propagate_in_video(state):
            assert ids==[2] and torch.isfinite(logits).all()
            masks.append(np.rint(logits[0,0].sigmoid().float().cpu().numpy()*255).astype(np.uint8));seen.append(frame)
    assert seen==list(range(121))
    bag=np.asarray(masks);assert bag.shape==original.shape
    combined=np.maximum(original,bag)
    assert np.all(combined>=original)
    np.savez_compressed(out/'masks.npz',bag=bag,combined=combined)
    pages={0:Image.new('RGB',(1040,1476),(128,128,128)),1:Image.new('RGB',(1040,1476),(128,128,128)),
           2:Image.new('RGB',(1040,1476),(128,128,128)),3:Image.new('RGB',(1040,1476),(128,128,128)),
           4:Image.new('RGB',(1040,492),(128,128,128))}
    probes=plan['review_frames'];pixel_checks=[]
    def output():
        with av.open(str(source/'crop_unsegmented.mp4')) as con:
            con.streams.video[0].thread_count=2
            for i,frame in enumerate(con.decode(video=0)):
                rgb=frame.to_ndarray(format='rgb24')
                before=original[i].astype(np.float32)[...,None]/255
                after=combined[i].astype(np.float32)[...,None]/255
                old=np.rint(rgb*before+255*(1-before)).clip(0,255).astype(np.uint8)
                new=np.rint(rgb*after+255*(1-after)).clip(0,255).astype(np.uint8)
                # A directly observed bag core, not a guessed segmentation score.
                if i==0:
                    x0,y0,x1,y1=plan.get('diagnostic_roi_xyxy',[475,485,510,520])
                    core=(slice(y0,y1),slice(x0,x1))
                    pixel_checks.append(dict(frame=0,roi_xyxy=[x0,y0,x1,y1],
                        original_alpha_mean=float(before[core].mean()),corrected_alpha_mean=float(after[core].mean()),
                        source_RGB_mean=float(rgb[core].mean()),old_input_RGB_mean=float(old[core].mean()),
                        corrected_input_RGB_mean=float(new[core].mean())))
                if i in probes:
                    j=probes.index(i);page=pages[j//3];row=j%3
                    overlay=rgb.copy();add=bag[i]>127
                    overlay[add]=(overlay[add].astype(float)*.5+np.array([255,0,255])*.5).astype(np.uint8)
                    for col,(arr,label) in enumerate([(rgb,'SOURCE'),(old,'OLD INPUT'),(new,'BODY + BAG'),(overlay,'BAG MASK')]):
                        im=ImageOps.contain(Image.fromarray(arr),(260,464));page.paste(im,(col*260,row*492+28))
                        ImageDraw.Draw(page).text((col*260+5,row*492+5),f'{label} f={i}',fill='white')
                yield Image.fromarray(new)
    encode(out/'corrected_input.mp4',output(),lossless=True)
    for page,im in pages.items():im.save(out/f'review_{page}.jpg',quality=94)
    areas=(bag>127).sum((1,2))
    save(out/'RESULT.json',dict(state='input_trial_completed_visual_review_pending',seconds=time.time()-start,
        gpu_uuid=uuid,frames=121,original_human_mask_decrease_pixels=int((combined<original).sum()),
        bag_area_min=int(areas.min()),bag_area_max=int(areas.max()),bag_area_median=float(np.median(areas)),
        original_and_final_input_sha256=[sha(source/'source_crop.mp4'),sha(out/'corrected_input.mp4')],
        source_preserved=sha(source/'segmentation/target_masks.npz')==plan['original_mask_sha256'],
        diagnostic_ROI= pixel_checks,quality_approved=False,baseline_modified=False))
    print(json.dumps(json.loads((out/'RESULT.json').read_text())),flush=True)


if __name__=='__main__':main()
