import sys,json,concurrent.futures,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'openimages_vehicle_wave_20261105'))
import acquire_people as acquire
choices=acquire.candidates();todo=[];seen=set()
for cat in ['bicycle','motorcycle']:
 for a in choices[cat]:
  key=(a['split'],a['image_id'])
  if not a['rider_present'] or key in seen or a['bbox_area']<.018 or a['quality_flags']['IsGroupOf']=='1':continue
  seen.add(key);todo.append(a)
print('download candidates',len(todo),dict(collections.Counter(a['category'] for a in todo)),flush=True)
rows=[];hashes=set()
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
 for a in pool.map(acquire.download,todo):
  if a['status']=='downloaded' and a['sha256'] not in hashes:rows.append(a);hashes.add(a['sha256'])
  if len(rows)%50==0:print('downloaded',len(rows),flush=True)
p=ROOT/'all_people_candidates.jsonl';p.write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in rows))
(ROOT/'all_people_acquisition.json').write_text(json.dumps(dict(status='completed',count=len(rows),categories=dict(collections.Counter(a['category'] for a in rows)),updated=time.time()),indent=2))
print('DONE',len(rows),flush=True)
