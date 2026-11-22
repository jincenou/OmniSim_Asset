"""Select and download 300 distinct Open Images V7 photographs per target class."""
import csv,json,hashlib,concurrent.futures,time,urllib.request,collections,os,io
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent
MID={'bus':'/m/01bjv','truck':'/m/07r04','van':'/m/0h2r6','motorcycle':'/m/04_sv','bicycle':'/m/0199g'}
PERSON='/m/01g317'
SPLITS=['test','validation']
TARGET=300
UA='DriveLabOpenImagesVehicleResearch/1.0'

def write(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.part');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temp,p)
def box(row):return [float(row[k]) for k in ['XMin','YMin','XMax','YMax']]
def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def rider(person,vehicle):
 h=max(0,min(person[2],vehicle[2])-max(person[0],vehicle[0]))
 return h/max(.001,vehicle[2]-vehicle[0])>=.25 and person[3]>=vehicle[1] and (person[1]+person[3])/2<vehicle[3]
def candidates():
 groups=collections.defaultdict(lambda:{'vehicles':[],'people':[]})
 for split in SPLITS:
  with (ROOT/'metadata'/(split+'-boxes.csv')).open(newline='') as f:
   for row in csv.DictReader(f):
    key=(split,row['ImageID']);label=row['LabelName']
    if label not in MID.values() and label!=PERSON:continue
    b=box(row)
    if label==PERSON:groups[key]['people'].append(b);continue
    if row['IsDepiction']=='1':continue
    category=next(c for c,m in MID.items() if m==label)
    groups[key]['vehicles'].append(dict(category=category,bbox=b,area=area(b),occluded=row['IsOccluded']=='1',quality_flags={k:row[k] for k in ['IsTruncated','IsGroupOf','IsDepiction','IsInside']},annotation_source=row['Source']))
 metadata={}
 for split in SPLITS:
  with (ROOT/'metadata'/(split+'-images.csv')).open(newline='') as f:
   for row in csv.DictReader(f):
    if (split,row['ImageID']) in groups:metadata[split,row['ImageID']]=row
 choices={c:[] for c in MID}
 for key,data in groups.items():
  if key not in metadata:continue
  for v in data['vehicles']:
   c=v['category'];persons=[p for p in data['people'] if rider(p,v['bbox'])];p=max(persons,key=area) if persons else None
   q=dict(split=key[0],image_id=key[1],category=c,bbox=v['bbox'],bbox_area=v['area'],occluded=v['occluded'],quality_flags=v['quality_flags'],rider_present=bool(p),rider_bbox=p,union_bbox=[min(p[0],v['bbox'][0]),min(p[1],v['bbox'][1]),max(p[2],v['bbox'][2]),max(p[3],v['bbox'][3])] if p else None,annotation_source=v['annotation_source'],license=metadata[key].get('License'),author=metadata[key].get('Author'),source_page_url=metadata[key].get('OriginalLandingURL'),original_url=metadata[key].get('OriginalURL'),title=metadata[key].get('Title'),image_url=f'https://open-images-dataset.s3.amazonaws.com/{key[0]}/{key[1]}.jpg')
   choices[c].append(q)
 for c in MID:
  choices[c].sort(key=lambda x:(all(v=='0' for v in x['quality_flags'].values()),(x['rider_present'] if c in ['bicycle','motorcycle'] else False),not x['occluded'],min(x['bbox_area'],.5),x['split']=='test'),reverse=True)
 return choices

def download(q):
 p=ROOT/'images'/q['split']/(q['image_id']+'.jpg');record=ROOT/'records'/q['category']/(q['split']+'_'+q['image_id']+'.json')
 if record.exists():return json.loads(record.read_text())
 p.parent.mkdir(parents=True,exist_ok=True)
 try:
  if not p.exists():
   req=urllib.request.Request(q['image_url'],headers={'User-Agent':UA})
   with urllib.request.urlopen(req,timeout=35) as resp:
    raw=resp.read(20*1024*1024+1)
    if len(raw)>20*1024*1024:raise ValueError('Image exceeds 20 MiB')
   Image.open(io.BytesIO(raw)).verify()
   temp=p.with_name(p.name+'.part');temp.write_bytes(raw);os.replace(temp,p)
  with Image.open(p) as im:width,height=im.size
  b=q['union_bbox'] if q['union_bbox'] else q['bbox'];px=[round(b[0]*width),round(b[1]*height),round(b[2]*width),round(b[3]*height)]
  q=dict(q,large_target=max(px[2]-px[0],px[3]-px[1])>=350 and min(px[2]-px[0],px[3]-px[1])>=100,status='downloaded',path=str(p),width=width,height=height,target_bbox_px=px,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),retrieved=time.time())
 except Exception as exc:q=dict(q,status='failed',error=repr(exc),retrieved=time.time())
 write(record,q);return q

def main():
 choices=candidates();write(ROOT/'candidate_counts.json',{c:dict(eligible_images=len({(x['split'],x['image_id']) for x in v}),rider_images=sum(x['rider_present'] for x in v)) for c,v in choices.items()})
 used=set();todo={}
 for c in MID:
  rows=[]
  for q in choices[c]:
   id=(q['split'],q['image_id'])
   if id in used:continue
   used.add(id);rows.append(q)
  todo[c]=rows
  if len(rows)<TARGET:raise RuntimeError(f'Only {len(rows)} unique candidates for {c}')
 write(ROOT/'selection.json',dict(target_per_category=TARGET,classes=list(MID),candidates={c:len(v) for c,v in todo.items()},sources=['Open Images V7 official class mapping and V5 box annotations (same box IDs)','Official Open Images image metadata and CVDF S3 images'],selection_time=time.time()))
 print('candidates', {c:len(v) for c,v in todo.items()},flush=True)
 selected={c:[] for c in MID};failed=collections.Counter();attempted=collections.Counter();hashes=set()
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
  # Small pages let the target stop at exactly 300 successes per class.
  while any(len(v)<TARGET for v in selected.values()):
   batch=[]
   for c in MID:
    missing=TARGET-len(selected[c])
    if missing<=0:continue
    take=min(max(missing+5,10),60)
    chunk=todo[c][attempted[c]:attempted[c]+take];attempted[c]+=len(chunk);batch.extend(chunk)
   if not batch:break
   for r in pool.map(download,batch):
    c=r['category']
    if r['status']=='downloaded' and r['sha256'] not in hashes and len(selected[c])<TARGET:
     hashes.add(r['sha256']);selected[c].append(r)
    elif r['status']!='downloaded':failed[c]+=1
   write(ROOT/'status.json',dict(status='downloading',target_per_category=TARGET,downloaded={c:len(v) for c,v in selected.items()},attempted=dict(attempted),failed=dict(failed),updated=time.time()))
   print('progress',{c:len(v) for c,v in selected.items()},flush=True)
 for c,rows in selected.items():
  with (ROOT/(c+'_selected.jsonl')).open('w') as f:
   for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 write(ROOT/'status.json',dict(status='completed' if all(len(v)==TARGET for v in selected.values()) else 'incomplete',target_per_category=TARGET,downloaded={c:len(v) for c,v in selected.items()},attempted=dict(attempted),failed=dict(failed),completed=time.time()))
 print('FINAL',{c:len(v) for c,v in selected.items()},flush=True)
if __name__=='__main__':main()
