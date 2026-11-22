from pathlib import Path
import json,numpy as np,cv2
from PIL import Image,ImageOps,ImageDraw
R=Path(__file__).resolve().parents[2]/'nonvehicle10';out=R/'inputs/nonvehicle_03_stop_sign';p=out/'segmented_manifest.json';case=json.loads(p.read_text());source=case['images'][0];mask=np.array(Image.open(source['mask_path']));T0=np.eye(3);T0[:2,2]=source['crop_box'][:2]
for row in case['images'][1:3]:
 M=np.eye(3);M[:2]=row['association']['affine'];T1=np.eye(3);T1[:2,2]=-np.array(row['crop_box'][:2]);W=T1@M@T0;im=Image.open(row['path']).convert('RGBA');alpha=cv2.warpAffine(mask,W[:2],im.size,flags=cv2.INTER_NEAREST,borderValue=0);Image.fromarray(alpha).save(row['mask_path']);im.putalpha(Image.fromarray(alpha));im.save(row['rgba_path']);row.update(mask_method='Verified source SAM2 mask propagated by recorded same-camera KLT affine; independent SAM2 mask failed visual QA',mask_model='SAM2 source + KLT affine propagation',mask_foreground_fraction=float((alpha>0).mean()),mask_warp=W.tolist())
p.write_text(json.dumps(case,indent=2));sheet=Image.new('RGB',(1280,290),'#ddd')
for i,row in enumerate(case['images']):
 im=Image.open(row['rgba_path']);bg=Image.new('RGBA',im.size,'#555');bg.alpha_composite(im);tile=ImageOps.contain(bg.convert('RGB'),(315,260));sheet.paste(tile,(i*320,25))
sheet.save(out/'segmented_inputs.jpg')
