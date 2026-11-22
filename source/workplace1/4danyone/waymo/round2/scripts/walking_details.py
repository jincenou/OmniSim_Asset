"""Detail previews from existing unfiltered videos, no GPU or enhancement."""
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from render_demo import details

R=Path(__file__).resolve().parents[1];LABEL='walk063_rootlocal24';ASSET=R/'reconstruction/outputs'/LABEL
DATA=R/'reconstruction/data'/LABEL;SOURCE=ASSET/'demo/fixed_views.mp4';OUT=ASSET/'demo/details'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False);records=[];n=0
 with imageio.get_reader(str(SOURCE),input_params=['-threads','1']) as reader:
  meta=reader.get_meta_data();assert meta['fps']==25 and tuple(meta['size'])==(1056,640)
  for frame,array in enumerate(reader):
   n+=1
   if frame not in (0,30,60,90,120):continue
   for i,view in enumerate((0,6,12)):
    image=DATA/'images'/f'cam{view:02d}'/f'{frame:06d}.png';rgba=np.asarray(Image.open(image).convert('RGBA'),dtype=np.float32)/255
    target=rgba[...,:3]*rgba[...,3:]+1-rgba[...,3:]
    gs=array[:,i*352:(i+1)*352].astype(np.float32)/255
    sheet,row=details(target,gs,rgba[...,3]>.5,view,frame)
    file=OUT/f'view{view:02d}_frame{frame:03d}.jpg';sheet.save(file,quality=95)
    records.append({**row,'path':str(file),'target_sha256':sha(image),'preview_sha256':sha(file)})
 if n!=121:raise ValueError('Incomplete existing action video')
 (OUT/'validation.json').write_text(json.dumps({'state':'detail_previews_written','decoded_frames':n,'views':[0,6,12],'frames':[0,30,60,90,120],'source_video':str(SOURCE),'source_sha256':sha(SOURCE),'method':'Decode existing unfiltered GS preview video; compare to saved generated RGBA on white. GS preview has video compression. Diagnostic only; quantitative metrics use direct float renders. Nearest pixel enlargement, no enhancement.','records':records},indent=2)+'\n')
 print(json.dumps({'detail_sheets':len(records),'gpu_used':False}))
if __name__=='__main__':main()
