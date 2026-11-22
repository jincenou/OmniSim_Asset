import asyncio,json,time,collections,shutil,os
from pathlib import Path
import prepare_seated150 as s
from asset_extraction.vlm import AsyncOpenAIClient
from asset_extraction.io_utils import read_json,write_json
ROOT=s.ROOT;OUT=s.OUT
os.environ['VLM_BASE_URL']='http://127.0.0.1:8082/v1'
s.run.VLM_LIMIT=asyncio.Semaphore(4)
from sync_generation_inputs import sync
async def main():
 base={json.loads(l)['sha256'] for l in (ROOT/'expanded_rider_candidates.jsonl').read_text().splitlines()}
 selected=read_json(OUT/'additional_manifest.json')['inputs'] if (OUT/'additional_manifest.json').exists() else [];attempted={a['source']['sha256'] for a in selected};client=AsyncOpenAIClient(s.run.CONFIG,None)
 try:
  for retry in range(120):
   try:await client.check_endpoints();break
   except Exception:await asyncio.sleep(5)
  else:raise RuntimeError('Qwen unavailable')
  while True:
   b=read_json(OUT/'manifest.json');n=b['count']+len(selected)
   if n>=150:break
   values=[]
   for c in ['bicycle','motorcycle']:
    for p in (s.run.INPUT/'records'/c).glob('*.json'):
     a=read_json(p)
     if a.get('status')=='downloaded' and a.get('rider_present') and a['sha256'] not in base and a['sha256'] not in attempted:values.append(a)
   values.sort(key=lambda a:(a['category']!='bicycle',not a.get('large_target',False),-a['bbox_area']))
   if not values:
    write_json(OUT/'additional_status.json',dict(status='waiting_for_more_downloads',accepted=len(selected),base_count=b['count'],total=n,updated=time.time()));await asyncio.sleep(10);continue
   for a in values[:8]:attempted.add(a['sha256'])
   for a in await asyncio.gather(*(s.one(client,a) for a in values[:8])):
    if a:
     dest=OUT/'images'/f'{a["source_id"]}.jpg';dest.parent.mkdir(exist_ok=True);shutil.copyfile(a['image_path'],dest);a['image_path']=str(dest);a['sha256']=s.run.gen.sha256(dest);selected.append(a)
   write_json(OUT/'additional_manifest.json',dict(count=len(selected),inputs=selected));sync()
   state=dict(status='screening',accepted=len(selected),base_count=b['count'],total=b['count']+len(selected),attempted=len(attempted),updated=time.time());write_json(OUT/'additional_status.json',state);print(json.dumps(state),flush=True)
  write_json(OUT/'additional_status.json',dict(status='enough_combined_candidates',total=n,updated=time.time()))
 finally:await client.close()
if __name__=='__main__':asyncio.run(main())
