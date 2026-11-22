import json,sys,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
sys.path.insert(0,'/mnt/ojc/third_party/Grounded-SAM-2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
packet=Path(sys.argv[1]);rows=json.loads(packet.read_text());predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/mnt/ojc/third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt',device='cuda'))
for row in rows:
 out=Path(row['output']);out.mkdir(parents=True,exist_ok=True);rgb=np.array(Image.open(row['image_path']).convert('RGB'))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  predictor.set_image(rgb);mask=np.zeros(rgb.shape[:2],bool);scores=[]
  for part in row['parts']:
   m,s,_=predictor.predict(box=np.array(part['box']),point_coords=np.array(part['points']) if part.get('points') else None,point_labels=np.array(part['labels']) if part.get('labels') else None,multimask_output=False)
   mask|=m[0].astype(bool);scores.append(float(s[0]))
 # No largest component, fill_holes or erosion. Any manual corrections are explicit polygons.
 original_mask=mask.copy()
 for poly in row.get('exclude_polygons',[]): cv2.fillPoly(mask.view('uint8'),[np.array(poly,np.int32)],0)
 box=row['crop_box'];rgba=np.dstack((rgb,mask.astype('uint8')*255));Image.fromarray(rgb).save(out/'source.png');Image.fromarray(original_mask.astype('uint8')*255).save(out/'sam_mask.png');Image.fromarray(mask.astype('uint8')*255).save(out/'mask.png');Image.fromarray(rgba).crop(box).save(out/'rgba.png')
 white=rgb.copy();white[~mask]=255;Image.fromarray(white).crop(box).save(out/'anchor.png')
 contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);over=rgb.copy();cv2.drawContours(over,contours,-1,(0,255,90),2);Image.fromarray(over).crop(box).save(out/'contour.png')
 cut=Image.fromarray(rgba).crop(box);cw,ch=cut.size;sheet=Image.new('RGB',(cw*3,ch))
 for k,bg in enumerate([(255,255,255),(25,25,25),None]):
  if bg is None:
   yy,xx=np.indices((ch,cw));grid=np.where(((xx//20+yy//20)%2)[...,None],180,230).astype('uint8');back=Image.fromarray(np.repeat(grid,3,axis=2))
  else:back=Image.new('RGB',cut.size,bg)
  back.paste(cut,mask=cut.getchannel('A'));sheet.paste(back,(k*cw,0))
 sheet.save(out/'mask_review.jpg');row.update(sam_scores=scores,mask_pixels=int(mask.sum()),rgba_sha256=hashlib.sha256((out/'rgba.png').read_bytes()).hexdigest());(out/'segmentation.json').write_text(json.dumps(row,indent=2));print(row['output'],scores,flush=True)
