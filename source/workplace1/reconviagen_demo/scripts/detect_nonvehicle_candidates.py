from pathlib import Path
import sys,os,json,io,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import torch
R=Path(__file__).resolve().parents[1];O=R/'nonvehicle10'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['XDG_CACHE_HOME']=str(R/'cache/xdg')
sys.path.insert(0,str(R/'tools/detector_deps'))
sys.path.insert(0,'/mnt/ojc/third_party/Grounded-SAM-2')
from grounding_dino.groundingdino.util.slconfig import SLConfig
from grounding_dino.groundingdino.models import build_model
from grounding_dino.groundingdino.util.misc import clean_state_dict
from grounding_dino.groundingdino.util.utils import get_phrases_from_posmap
import grounding_dino.groundingdino.datasets.transforms as T
from torchvision.ops import box_convert,batched_nms
from native_candidates import frames
cfg=SLConfig.fromfile('/mnt/ojc/third_party/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py');cfg.device='cuda';cfg.text_encoder_type='/mnt/ojc/models/bert-base-uncased'
net=build_model(cfg);state=torch.load('/mnt/ojc/third_party/Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth',map_location='cpu');result=net.load_state_dict(clean_state_dict(state['model']),strict=False);print('LOAD',result,flush=True);net.eval().cuda()
transform=T.Compose([T.RandomResize([800],max_size=1333),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])]);caption='traffic light. traffic sign. traffic cone. road barrier. bollard. trash can. fire hydrant.'
raws=[Path('/mnt/ojc/asset_coverage_experiment/runs/waymo_002_022_full_v1/ah/inputs/raw_tfrecords/segment-10444454289801298640_4360_000_4380_000_with_camera_labels.tfrecord')]
cs=json.loads((R/'inputs/random10_cases.json').read_text());raws+=sorted({Path(c['images'][0]['raw_tfrecord']) for c in cs})[:2]
records=[]
for ri,raw in enumerate(raws):
 for fi,frame in frames(raw):
  if fi not in [20,60,100,140,180]:continue
  for cam in [1,2,3]:
   data=next(x for x in frame.images if x.name==cam);im=Image.open(io.BytesIO(data.image)).convert('RGB');arr=np.array(im);x,_=transform(im,None)
   with torch.inference_mode():pred=net(x[None].cuda(),captions=[caption])
   logits=pred['pred_logits'].sigmoid().cpu()[0];boxes=pred['pred_boxes'].cpu()[0];keep=logits.max(1).values>.22;logits=logits[keep];boxes=boxes[keep];tokens=net.tokenizer(caption)
   phrases=[get_phrases_from_posmap(l>.18,tokens,net.tokenizer).replace('.','').strip() for l in logits];b=box_convert(boxes*torch.tensor([im.width,im.height,im.width,im.height]),'cxcywh','xyxy');scores=logits.max(1).values
   cats=[]
   for p in phrases:
    cats.append(next((cat for terms,cat in [(['light'],'traffic_light'),(['sign'],'traffic_sign'),(['cone'],'cone'),(['barrier'],'barrier'),(['bollard'],'bollard'),(['trash','can'],'trash_can'),(['hydrant'],'fire_hydrant')] if any(t in p for t in terms)),None))
   category_ids=torch.tensor([['traffic_light','traffic_sign','cone','barrier','bollard','trash_can','fire_hydrant'].index(c) if c else 99 for c in cats]);idxs=batched_nms(b,scores,category_ids,.45)
   for j in idxs:
    j=int(j);cat=cats[j];box=b[j].numpy();box[[0,2]]=np.clip(box[[0,2]],0,im.width);box[[1,3]]=np.clip(box[[1,3]],0,im.height);w,h=box[2]-box[0],box[3]-box[1]
    if cat is None or min(w,h)<28 or max(w,h)<65 or w*h>im.width*im.height*.45 or min(box[0],box[1],im.width-box[2],im.height-box[3])<4:continue
    idx=len(records);tag=f'candidate_{idx:03}';pad=.12*max(w,h);crop=[max(0,int(box[0]-pad)),max(0,int(box[1]-pad)),min(im.width,int(box[2]+pad)),min(im.height,int(box[3]+pad))];dest=O/'inputs'/f'{tag}.png';im.crop(crop).save(dest)
    calib=next(c for c in frame.context.camera_calibrations if c.name==cam)
    rec=dict(candidate_id=tag,category=cat,phrase=phrases[j],score=float(scores[j]),raw_tfrecord=str(raw),context=frame.context.name,frame_index=fi,camera=cam,box_xyxy=box.tolist(),crop_box=crop,box_crop_xyxy=[float(box[0]-crop[0]),float(box[1]-crop[1]),float(box[2]-crop[0]),float(box[3]-crop[1])],path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),image_timestamp_s=data.pose_timestamp,ego_to_world=list(frame.pose.transform),camera_pose_to_world=list(data.pose.transform),camera_to_ego=list(calib.extrinsic.transform),intrinsic=list(calib.intrinsic),image_size=[im.width,im.height]);records.append(rec)
   print(ri,fi,cam,'candidates',len(records),flush=True)
  (O/'inputs/candidates.json').write_text(json.dumps(records,indent=2))
  if fi>=180:break
 for page in range((len(records)+29)//30):
  subset=records[page*30:(page+1)*30];sheet=Image.new('RGB',(1500,300*((len(subset)+5)//6)),'#ddd');d=ImageDraw.Draw(sheet)
  for k,rec in enumerate(subset):
   xx=k%6*250;yy=k//6*300;small=ImageOps.contain(Image.open(rec['path']),(245,260));sheet.paste(small,(xx,yy+35));d.text((xx+3,yy+3),rec['candidate_id']+' '+rec['category'],fill='black');d.text((xx+3,yy+18),f"f{rec['frame_index']} c{rec['camera']} {rec['score']:.2f}",fill='black')
  sheet.save(O/'docs'/f'candidates_{page:02}.jpg',quality=90)
print('CANDIDATES COMPLETE',len(records),flush=True)
