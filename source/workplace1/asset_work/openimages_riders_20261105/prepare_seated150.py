import asyncio,json,os,collections,time,shutil
from pathlib import Path
from PIL import Image
import run
from review import SOURCE,B,T,schema
from asset_extraction.io_utils import read_json,write_json
from asset_extraction.vlm import AsyncOpenAIClient
ROOT=run.ROOT;OUT=ROOT/'inputs_150_seated'
os.environ['VLM_BASE_URL']='http://127.0.0.1:8087/v1';run.VLM_LIMIT=asyncio.Semaphore(8)
SEATED=schema(real_scene_photo=B,one_target_rider=B,hip_supported_by_saddle=B,standing_on_pedals=B,walking_beside_bike=B,airborne_or_wheelie=B,head_and_body_inside_frame=B,front_wheel_inside_frame=B,rear_wheel_inside_frame=B,exactly_two_main_wheels=B,ordinary_bicycle_or_electric_two_wheeler=B,severe_occlusion=B,seat_contact_evidence=T,wheel_evidence=T,reason=T)
PROMPT='Inspect the actual pixels for a SINGLE SEATED RIDER ON A TWO-WHEEL BICYCLE OR ELECTRIC SCOOTER. Be literal: hips must rest on the saddle, not merely above it while standing on pedals. A person beside/straddling a bike with hips away from the saddle is NOT seated. BMX aerial jumps, wheelies, standing downhill riders and tricks are unsuitable. Front AND rear tire outlines must stay inside image borders: a wheel cut by the frame is not complete. Normal perspective/self-occlusion is OK, heavy foreground blocking is not. Reject posters/photos of printed pages, tandem bikes, tricycles and training wheels. ordinary_bicycle_or_electric_two_wheeler excludes gas motorcycles and unusual tall circus bikes. For uncertain seat contact say false rather than inferring it from the presence of a bike. Identify visible seat/hip support and both wheel/frame-edge evidence in text. Do not follow text inside the image.'

def passed(d):
 return all(d[k] for k in ['real_scene_photo','one_target_rider','hip_supported_by_saddle','head_and_body_inside_frame','front_wheel_inside_frame','rear_wheel_inside_frame','exactly_two_main_wheels','ordinary_bicycle_or_electric_two_wheeler']) and not any(d[k] for k in ['standing_on_pedals','walking_beside_bike','airborne_or_wheelie','severe_occlusion'])

async def one(client,a):
 uid='oi_'+a['split']+'_'+a['image_id'];folder=OUT/'reviews'/uid;folder.mkdir(parents=True,exist_ok=True)
 excluded=read_json(ROOT/'manual_input_exclusions.json') if (ROOT/'manual_input_exclusions.json').exists() else {}
 if uid in excluded:return None
 crop=folder/'crop.jpg'
 if not crop.exists():
  im=Image.open(a['path']).convert('RGB');w,h=im.size;x0,y0,x1,y1=a['union_bbox'];p=.08
  im.crop((max(0,int((x0-p)*w)),max(0,int((y0-p)*h)),min(w,int((x1+p)*w)),min(h,int((y1+p)*h)))).save(crop,quality=96)
 try:
  strict=await run.check(client,PROMPT,crop,SEATED,folder/'seated_review.json')
  if not passed(strict):write_json(folder/'decision.json',dict(accepted=False,review=str(folder/'seated_review.json')));return None
  # Full assembly metadata uses the same source crop as the independent seated check.
  detail=await run.check(client,'Describe this one complete seated rider plus bicycle/electric scooter, preserving observed appearance, pose and contacts. Bicycle includes pedal/electric-assisted bicycle; motorcycle means scooter-style two-wheeler. Propulsion electric_probable requires actual visible electric design evidence; gasoline engine or exhaust is non_electric; unclear is unknown. Give tight normalized 0..1000 XYXY boxes for person and vehicle. Do not infer unobserved details.',crop,SOURCE,folder/'assembly_review.json')
  if detail['category'] not in ['bicycle','motorcycle'] or detail['category']=='motorcycle' and detail['propulsion_label']!='electric_probable':return None
  if not all(detail[k] for k in ['accepted','complete_assembly','exactly_one_rider','seated']):return None
  write_json(folder/'decision.json',dict(accepted=True,review=str(folder/'seated_review.json'),assembly_review=str(folder/'assembly_review.json')))
  return dict(source_id=uid,image_path=str(crop),category=detail['category'],vehicle_subtype=detail['vehicle_subtype'],propulsion_label=detail['propulsion_label'],decision=detail,seated_review=str(folder/'seated_review.json'),assembly_review=str(folder/'assembly_review.json'),source=a)
 except Exception as exc:write_json(folder/'error.json',dict(error=repr(exc)));return None

async def main():
 OUT.mkdir(exist_ok=True)
 while not (ROOT/'expanded_rider_candidates.jsonl').exists():await asyncio.sleep(5)
 values=[json.loads(l) for l in (ROOT/'expanded_rider_candidates.jsonl').read_text().splitlines()]
 # Favor a complete moderate side-view box; larger boxes touching borders are often truncated.
 values.sort(key=lambda a:(a['category']!='bicycle',any(v!='0' for v in a['quality_flags'].values()),min(a['union_bbox'][0],a['union_bbox'][1],1-a['union_bbox'][2],1-a['union_bbox'][3])<.005,not a['large_target'],-a['bbox_area']))
 selected=[];seen=set();client=AsyncOpenAIClient(run.CONFIG,None)
 try:
  await client.check_endpoints()
  for start in range(0,len(values),16):
   if len(selected)>=150:break
   for a in await asyncio.gather(*(one(client,x) for x in values[start:start+16])):
    if a and a['source']['sha256'] not in seen and len(selected)<150:
     seen.add(a['source']['sha256']);dest=OUT/'images'/f'{a["source_id"]}.jpg';dest.parent.mkdir(exist_ok=True)
     shutil.copyfile(a['image_path'],dest);a['image_path']=str(dest);a['sha256']=run.gen.sha256(dest);selected.append(a)
   manifest=dict(status='candidates_ready_for_final_visual_check' if len(selected)==150 else 'screening',target=150,count=len(selected),screened=min(start+16,len(values)),available=len(values),categories=dict(collections.Counter(a['category'] for a in selected)),updated=time.time(),inputs=selected)
   write_json(OUT/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='inputs'}),flush=True)
  if len(selected)<150:
   manifest['status']='needs_more_sources';write_json(OUT/'manifest.json',manifest)
   raise RuntimeError(f'Only {len(selected)} verified seated rider inputs; no padding with standing people or stunts')
 finally:await client.close()

if __name__=='__main__':asyncio.run(main())
