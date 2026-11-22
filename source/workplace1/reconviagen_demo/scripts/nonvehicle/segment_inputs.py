"""Independent SAM2 masks from native boxes, preserving originals and alpha."""
from pathlib import Path
import json,sys,os,argparse
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
SHARED=Path(__file__).resolve().parents[2]
ROOT=SHARED/"nonvehicle10"
sys.path.insert(0,'/mnt/ojc/workplace1/4danyone/waymo/segmentation')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args()
out=ROOT/'inputs'/a.case;case=json.loads((out/'manifest.json').read_text())
model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/mnt/ojc/models/grounded_sam2/sam2.1_hiera_large.pt',device='cuda',apply_postprocessing=False)
predictor=SAM2ImagePredictor(model)
tiles=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for rec in case['images']:
    image=Image.open(rec['path']).convert('RGB');arr=np.array(image);w,h=image.size
    box=np.array(rec.get('box_crop_xyxy',[w*.04,h*.04,w*.96,h*.96]),np.float32)
    predictor.set_image(arr)
    extra={}
    if a.case=='nonvehicle_03_stop_sign':extra=dict(point_coords=np.array([[(box[0]+box[2])/2,(box[1]+box[3])/2]],np.float32),point_labels=np.array([1]))
    masks,scores,logits=predictor.predict(box=box,multimask_output=False,return_logits=False,**extra)
    mask=masks[0].astype(np.uint8)*255
    if mask.sum()==0:raise ValueError(f'Empty mask: {rec["id"]}')
    mp=out/(rec['id']+'_mask.png');rp=out/(rec['id']+'_rgba.png')
    Image.fromarray(mask).save(mp);rgba=image.convert('RGBA');rgba.putalpha(Image.fromarray(mask));rgba.save(rp)
    rec.update(mask_path=str(mp),rgba_path=str(rp),mask_model='SAM2.1 Hiera Large',
        mask_prompt_box=box.tolist(),mask_positive_center_point=bool(extra),mask_score=float(scores[0]),mask_foreground_fraction=float((mask>0).mean()))
    preview=Image.new('RGBA',image.size,(80,80,80,255));preview.alpha_composite(rgba)
    tile=Image.new('RGB',(320,290),'#ddd');im=ImageOps.contain(preview.convert('RGB'),(320,250));tile.paste(im,(0,25))
    ImageDraw.Draw(tile).text((3,3),f"{rec['id']} / {rec['split']}",fill='black');tiles.append(tile)
sheet=Image.new('RGB',(1280,290*((len(tiles)+3)//4)),'#ddd')
for i,tile in enumerate(tiles):sheet.paste(tile,(i%4*320,i//4*290))
sheet.save(out/'segmented_inputs.jpg')
(out/'segmented_manifest.json').write_text(json.dumps(case,indent=2))
print(a.case,'segmented',len(case['images']),flush=True)
