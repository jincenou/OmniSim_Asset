import json
from pathlib import Path
from PIL import Image,ImageDraw
root=Path('/mnt/ojc/workplace1/asset_work/vehicle1000_20261025/evaluation/waymo_native');out=Path(__file__).resolve().parents[1]/'sources/waymo';out.mkdir(exist_ok=True)
best={};counts={4:set(),1:set()};missing=0
files=sorted(root.glob('*.jsonl'))[:30]
for p in files:
 for line in p.open():
  f=json.loads(line);lidar={r['id']:r for r in f['laser_labels']}
  for cam in f.get('camera_annotations',[]):
   if not Path(cam['path']).is_file():missing+=1;continue
   for r in cam['annotations']:
    if r['type'] not in (1,4):continue
    key=p.stem+':'+cam['camera_name']+':'+r['camera_track_id'];counts[r['type']].add(key)
    if r['type']==1:
     l=lidar.get(r.get('laser_object_id'))
     if not l or not (1.2<l['box'][3]<3.0 and l['box'][4]<1.5):continue
    x1,y1,x2,y2=r['bbox_xyxy'];w,h=cam['image_wh']
    if x1<8 or y1<8 or x2>w-8 or y2>h-8 or min(x2-x1,y2-y1)<65:continue
    score=(x2-x1)*(y2-y1)
    if score>best.get(key,{}).get('score',0):best[key]=dict(identity=key,**r,score=score,image_path=cam['path'],native_frame_index=f['native_frame_index'],timestamp_us=f['timestamp_us'])
rows=sorted(best.values(),key=lambda r:r['score'],reverse=True);(out/'candidates.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(dict(scene_files=len(files),camera_scoped_vehicle_identities=len(counts[1]),camera_scoped_cyclist_identities=len(counts[4]),missing_camera_rows=missing,ranked_candidates=len(rows),vehicle_filter='linked lidar physical length 1.2..3.0m,width<1.5m; conservative subset not exhaustive motorcycle detector'),indent=2))
for start in range(0,min(48,len(rows)),24):
 sheet=Image.new('RGB',(6*250,4*235),'#ddd');draw=ImageDraw.Draw(sheet)
 for j,r in enumerate(rows[start:start+24]):
  im=Image.open(r['image_path']);b=r['bbox_xyxy'];im=im.crop((b[0]-8,b[1]-8,b[2]+8,b[3]+8));im.thumbnail((245,200));xx=j%6*250;yy=j//6*235;sheet.paste(im,(xx+(250-im.width)//2,yy));draw.text((xx+3,yy+202),f"{start+j} type={r['type']} {r['identity'].split(':')[0]}",fill='black')
 sheet.save(out/f'sheet_{start//24}.jpg')
print('Waymo',len(files),'scenes',len(rows),'candidates')
