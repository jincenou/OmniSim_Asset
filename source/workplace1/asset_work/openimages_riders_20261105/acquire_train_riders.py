import csv,io,json,sys,time,urllib.request,collections,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'openimages_vehicle_wave_20261105'))
import acquire
HUMAN={'/m/01g317','/m/04yx4','/m/03bt1vf','/m/01bl7v','/m/05r655'}
VEH={'/m/0199g':'bicycle','/m/04_sv':'motorcycle'}
URL='https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-vrd.csv'
META='https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
selection=ROOT/'train_relation_selection.json'
if selection.exists():selected=json.loads(selection.read_text())
else:
 selected={};relations=collections.Counter()
 with urllib.request.urlopen(URL,timeout=120) as response:
  for number,row in enumerate(csv.DictReader(io.TextIOWrapper(response))):
   if row['LabelName1'] in HUMAN and row['LabelName2'] in VEH:pidx,vidx=1,2
   elif row['LabelName2'] in HUMAN and row['LabelName1'] in VEH:pidx,vidx=2,1
   else:continue
   rel=row['RelationshipLabel'];relations[rel]+=1
   if rel not in {'ride','rides','on'}:continue
   p=[float(row[k+str(pidx)]) for k in ['XMin','YMin','XMax','YMax']];v=[float(row[k+str(vidx)]) for k in ['XMin','YMin','XMax','YMax']]
   if not acquire.rider(p,v) or acquire.area(v)<.025:continue
   u=[min(p[0],v[0]),min(p[1],v[1]),max(p[2],v[2]),max(p[3],v[3])]
   if min(u[0],u[1],1-u[2],1-u[3])<.003:continue
   imageid=row['ImageID'];category=VEH[row['LabelName'+str(vidx)]]
   q=dict(split='train',image_id=imageid,category=category,bbox=v,bbox_area=acquire.area(v),occluded=False,quality_flags={k:'unknown' for k in ['IsTruncated','IsGroupOf','IsDepiction','IsInside']},rider_present=True,rider_bbox=p,union_bbox=u,annotation_source=URL,relationship=rel,image_url=f'https://open-images-dataset.s3.amazonaws.com/train/{imageid}.jpg')
   if imageid not in selected or q['bbox_area']>selected[imageid]['bbox_area']:selected[imageid]=q
   if len(selected)%100==0:print('relation candidates',len(selected),dict(relations),flush=True)
   if len(selected)>=1400:break
 selection.write_text(json.dumps(selected,indent=2));print('RELATIONS',len(selected),dict(relations),flush=True)
metadata_path=ROOT/'train_selected_metadata.json'
if metadata_path.exists():metadata=json.loads(metadata_path.read_text())
else:
 metadata={}
 with urllib.request.urlopen(META,timeout=120) as response:
  for i,row in enumerate(csv.DictReader(io.TextIOWrapper(response))):
   if row['ImageID'] in selected:metadata[row['ImageID']]=row
   if i%200000==0:print('metadata scan',i,'matched',len(metadata),flush=True)
   if len(metadata)==len(selected):break
 metadata_path.write_text(json.dumps(metadata,indent=2))
rows=[]
for imageid,a in selected.items():
 if imageid not in metadata:continue
 m=metadata[imageid];a.update(license=m.get('License'),author=m.get('Author'),source_page_url=m.get('OriginalLandingURL'),original_url=m.get('OriginalURL'),title=m.get('Title'));rows.append(a)
output=[];seen=set()
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
 for a in pool.map(acquire.download,rows):
  if a['status']=='downloaded' and a['sha256'] not in seen:output.append(a);seen.add(a['sha256'])
  if len(output)%50==0:print('downloaded',len(output),flush=True)
  # Atomic rolling candidates permit screening while later photos download.
  if len(output)%50==0:
   p=ROOT/'train_rider_candidates.jsonl';tmp=p.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(x)+'\n' for x in output));tmp.replace(p)
p=ROOT/'train_rider_candidates.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in output))
(ROOT/'train_acquisition.json').write_text(json.dumps(dict(status='completed',count=len(output),updated=time.time()),indent=2));print('DONE',len(output),flush=True)
