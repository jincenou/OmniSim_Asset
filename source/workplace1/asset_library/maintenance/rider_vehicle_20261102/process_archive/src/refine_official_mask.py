"""Repair segmentation only; RGB pixels always come unchanged from source."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing,binary_fill_holes,label
W=Path(__file__).resolve().parents[1]
for name in sys.argv[1:]:
 p=W/'packets'/name; d=json.loads((p/'packet_joint_v3.json').read_text());r=d['observations'][0]
 rgb=np.array(Image.open(r['image_path']).convert('RGB'));m=np.array(Image.open(p/'anchor_joint_v3/union_mask.png'))>0
 # Close narrow alpha cracks, then repair enclosed small dropout regions.
 closed=binary_closing(m,structure=np.ones((11,11),bool))
 holes=binary_fill_holes(closed)&~closed;lab,count=label(holes);sizes=np.bincount(lab.ravel());ids=np.where((sizes<18000)&(sizes>0))[0];ids=ids[ids!=0]
 repaired=closed|np.isin(lab,ids)
 lab,count=label(repaired);sizes=np.bincount(lab.ravel());sizes[0]=0;repaired=np.isin(lab,np.where(sizes>1500)[0])
 out=p/'anchor_refined';out.mkdir(exist_ok=False)
 Image.fromarray((repaired*255).astype('uint8')).save(out/'union_mask.png')
 ys,xs=np.where(repaired);box=[max(0,int(xs.min())-20),max(0,int(ys.min())-20),min(rgb.shape[1],int(xs.max())+21),min(rgb.shape[0],int(ys.max())+21)]
 Image.fromarray(np.dstack([rgb,(repaired*255).astype('uint8')])).crop(box).save(out/'rgba.png')
 white=rgb.copy();white[~repaired]=255;Image.fromarray(white).crop(box).save(out/'anchor.png')
 r.update(id='anchor_refined',anchor_path=str(out/'anchor.png'),rgba_path=str(out/'rgba.png'),crop_box=box,sha256=hashlib.sha256((out/'rgba.png').read_bytes()).hexdigest())
 r['mask_refinement']={'base':'anchor_joint_v3/union_mask.png','closing_kernel':11,'max_enclosed_hole_pixels':18000,'min_component_pixels':1500,'rgb_synthesis':False,'added_pixels':int((repaired&~m).sum()),'removed_pixels':int((m&~repaired).sum()),'visual_review_required':True}
 (p/'segmentation_refined.json').write_text(json.dumps(d,indent=2));print(name,r['mask_refinement'])
