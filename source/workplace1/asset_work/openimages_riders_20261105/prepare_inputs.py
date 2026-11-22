"""Collect exactly 100 reviewed whole rider+bicycle/electric inputs, preserving source evidence."""
import asyncio,collections,json,os,shutil,sys,time
from pathlib import Path
import run
from review import SOURCE
from asset_extraction.io_utils import read_json,write_json
from asset_extraction.vlm import AsyncOpenAIClient
from PIL import Image
ROOT=run.ROOT
OUT=ROOT/'inputs_100'
os.environ['VLM_BASE_URL']='http://127.0.0.1:8087/v1'
run.VLM_LIMIT=asyncio.Semaphore(8)
PROMPT='Select ONE complete seated rider with a bicycle or an electric scooter/bike. Both wheels, vehicle body, rider head, torso and main limbs must be inside the image. Accept natural self-occlusion and minor blur. Reject empty bikes, people merely standing beside bikes, multiple occupants on one vehicle, severe frame truncation, and out-of-saddle stunts. Category bicycle includes pedal and electric-assisted bicycles. Category motorcycle is scooter-style vehicle. electric_probable requires visible design evidence; a combustion engine/exhaust is non_electric, unclear propulsion is unknown. Describe observed pose and contact without inventing details. Give separate normalized 0..1000 XYXY rider and vehicle boxes.'

def eligible(d):
 return all(d.get(k) is True for k in ['accepted','complete_assembly','exactly_one_rider','seated']) and (d['category']=='bicycle' or d['category']=='motorcycle' and d['propulsion_label']=='electric_probable')

def candidates():
 values=run.inventory();seen={a['sha256'] for a in values}
 for cat in ['bicycle','motorcycle']:
  for line in (run.INPUT/f'{cat}_selected.jsonl').read_text().splitlines():
   a=json.loads(line)
   if a.get('rider_present') and a['sha256'] not in seen:values.append(a);seen.add(a['sha256'])
 # Current source decisions are cheap to reuse, including raw inputs with imperfect downstream masks.
 values.sort(key=lambda a:not (ROOT/'sources'/('oi_'+a['split']+'_'+a['image_id'])/'source_review.json').exists())
 return values

async def one(client,a):
 uid='oi_'+a['split']+'_'+a['image_id'];folder=OUT/'reviews'/uid;folder.mkdir(parents=True,exist_ok=True)
 original_review=ROOT/'sources'/uid/'source_review.json'
 crop=folder/'crop.jpg'
 if not crop.exists():
  im=Image.open(a['path']).convert('RGB');w,h=im.size;x0,y0,x1,y1=a['union_bbox'];p=.05
  im.crop((max(0,int((x0-p)*w)),max(0,int((y0-p)*h)),min(w,int((x1+p)*w)),min(h,int((y1+p)*h)))).save(crop,quality=96)
 try:
  if original_review.exists():
   saved=read_json(original_review);d=saved['decision'];evidence=str(original_review)
  else:
   d=await run.check(client,PROMPT,crop,SOURCE,folder/'review.json');evidence=str(folder/'review.json')
  record=dict(source_id=uid,accepted=eligible(d),decision=d,review=evidence,original=a,crop=str(crop))
 except Exception as exc:record=dict(source_id=uid,accepted=False,error=repr(exc),original=a)
 write_json(folder/'decision.json',record);return record

async def main():
 OUT.mkdir(exist_ok=True);items=candidates();selected=[];seen=set()
 if (OUT/'manifest.json').exists():selected=read_json(OUT/'manifest.json')['inputs'];seen={a['source_id'] for a in selected}
 client=AsyncOpenAIClient(run.CONFIG,None)
 try:
  await client.check_endpoints()
  for offset in range(0,len(items),8):
   if len(selected)>=100:break
   batch=items[offset:offset+8]
   records=await asyncio.gather(*(one(client,a) for a in batch))
   for r in records:
    if not r['accepted'] or r['source_id'] in seen or len(selected)>=100:continue
    dest=OUT/'images'/f'{r["source_id"]}.jpg';dest.parent.mkdir(exist_ok=True)
    shutil.copyfile(r['crop'],dest)
    selected.append(dict(source_id=r['source_id'],image_path=str(dest),sha256=run.gen.sha256(dest),category=r['decision']['category'],propulsion_label=r['decision']['propulsion_label'],vehicle_subtype=r['decision']['vehicle_subtype'],review=r['review'],decision=r['decision'],source=r['original']))
    seen.add(r['source_id'])
   counts=collections.Counter(a['category'] for a in selected)
   manifest=dict(status='ready' if len(selected)==100 else 'preparing',target=100,count=len(selected),category_counts=dict(counts),updated=time.time(),definition='100 source photos with visually reviewed whole seated rider and bicycle/electric vehicle; not 100 reconstructed or accepted 3D assets',inputs=selected)
   write_json(OUT/'manifest.json',manifest)
   print(json.dumps({k:v for k,v in manifest.items() if k!='inputs'}),flush=True)
  if len(selected)<100:
   manifest['status']='insufficient_candidates';write_json(OUT/'manifest.json',manifest)
   raise RuntimeError(f'Only {len(selected)} eligible inputs; do not pad with empty bikes or gasoline motorcycles')
  import csv
  with (OUT/'INDEX.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['source_id','category','vehicle_subtype','propulsion_label','image_path','source_page_url'])
   for a in selected:w.writerow([a['source_id'],a['category'],a['vehicle_subtype'],a['propulsion_label'],a['image_path'],a['source']['source_page_url']])
  (OUT/'README.txt').write_text('100张带骑手的骑行车辆输入。images/ 为完整主体裁剪图，manifest.json 和 INDEX.csv 记录类型与来源。电动车仅凭视觉判断者标为 electric_probable；不能当已核验车型。所有图均保留原始来源、授权信息和审核证据。这是输入准备完成，不代表100个三维资产已经生成或合格。\n')
 finally:await client.close()

if __name__=='__main__':asyncio.run(main())
