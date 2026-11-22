"""Re-evaluate failed single-person alpha with the pipeline's pinned BiRefNet.

Only observed source RGB is composited. Preserve the failed SAM2 outputs and
require the same numeric input gate plus a separate visual review afterwards.
"""
from pathlib import Path
import sys,json,shutil
import av,numpy as np,torch
from PIL import Image
from prepare_person import ROOT,PROJECT,selected,save,sha,encode
sys.path.insert(0,str(PROJECT/'repo'))
from fdanyone.foreground import predict_foreground_masks
from fdanyone.assets import resolve_foreground_model

ident=sys.argv[1];selected(ident);dest=ROOT/'inputs'/ident
assert not json.loads((dest/'research/batch_gate_mask.json').read_text())['passed']
archive=dest/'diagnostics/sam2_bbox_v1';archive.mkdir(parents=True,exist_ok=False)
for name in ['segmentation/target_masks.npz','segmentation/manifest.json','research/batch_gate_mask.json','source_crop.mp4']:
 p=dest/name
 if p.exists():shutil.move(str(p),archive/p.name)
with av.open(str(dest/'crop_unsegmented.mp4')) as c:frames=tuple(f.to_ndarray(format='rgb24') for f in c.decode(video=0))
assert len(frames)==121
weights=resolve_foreground_model(PROJECT/'repo/models')
masks=predict_foreground_masks(frames,weights,'cuda:0',batch_size=2)
np.savez_compressed(dest/'segmentation/target_masks.npz',masks=masks)
save(dest/'segmentation/manifest.json',dict(method='pinned_pipeline_BiRefNet_per_frame',weights=str(weights),frames=121,
  original_failed_method='SAM2 bbox plus three fixed points',original_failure_archive=str(archive),
  source_RGB_unchanged=True,visual_review='pending',reason='Initial SAM2 alpha removed observed clothing; independent foreground prediction before any generation'))
images=[]
for rgb,mask in zip(frames,masks,strict=True):
 a=mask.astype(np.float32)[...,None]/255
 images.append(Image.fromarray(np.rint(rgb*a+255*(1-a)).clip(0,255).astype(np.uint8)))
encode(dest/'source_crop.mp4',images,lossless=True)
link=dest/'inputs/main.mp4'
if not link.is_symlink():link.symlink_to('../source_crop.mp4')
mapping=json.loads((dest/'inputs/frame_mapping.json').read_text())
binary=masks>127;boxes=np.asarray([x['bbox_input_xyxy'] for x in mapping['frames']])
areas=binary.sum((1,2))/((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]));core=[]
for mask,b in zip(binary,boxes,strict=True):
 xa,ya,xb,yb=b;w=xb-xa;h=yb-ya;x0,y0,x1,y1=np.rint([xa+.3*w,ya+.2*h,xa+.7*w,ya+.6*h]).astype(int)
 patch=mask[max(y0,0):min(y1,mask.shape[0]),max(x0,0):min(x1,mask.shape[1])];core.append(float(patch.mean()) if patch.size else 0.)
passed=bool(areas.min()>.15 and np.median(areas)>.3 and areas.min()/np.median(areas)>.55 and min(core)>.3 and np.quantile(core,.1)>.55)
save(dest/'research/batch_gate_mask.json',dict(passed=passed,mask_area_min=float(areas.min()),mask_area_median=float(np.median(areas)),torso_min=min(core),torso_p10=float(np.quantile(core,.1)),scope='Same G0 foreground validity thresholds; visual acceptance remains pending'))
assert passed,'Foreground gate failed; keep both failures and reject input'
mapping.update(video_sha256=sha(dest/'source_crop.mp4'),background_compositing='straight alpha over white',alpha_method='pinned_BiRefNet_repair_of_failed_SAM2')
save(dest/'inputs/frame_mapping.json',mapping)
print(ident,'repaired alpha numeric gate passed; visual review pending')
