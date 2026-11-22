import json
from pathlib import Path
from PIL import Image,ImageDraw
W=Path(__file__).resolve().parents[1];out=W/'sources/bicycle_dataset';old=W.parent.parent/'rider_vehicle_20261030/process_archive';rank=json.loads((out/'candidates.json').read_text());allrows=json.loads((old/'sources/nuscenes/observations.json').read_text());records=[]
for idx in [1,2,4,5,6,8,10,18,20,23]:
 r=rank[idx];rows=sorted([x for x in allrows if x['instance']==r['instance'] and x['camera']==r['camera'] and Path(x['image_path']).exists()],key=lambda x:x['timestamp_us']);nearest=sorted(rows,key=lambda x:abs(x['timestamp_us']-r['timestamp_us']))[:7];nearest.sort(key=lambda x:x['timestamp_us']);sheet=Image.new('RGB',(280*len(nearest),270),'white');draw=ImageDraw.Draw(sheet)
 for j,x in enumerate(nearest):
  im=Image.open(x['image_path']).crop(x['bbox']);im.thumbnail((275,235));sheet.paste(im,(j*280,0));draw.text((j*280,238),str(x['timestamp_us'])+'\n'+x['camera'],fill='black')
 p=out/f'track_{idx:02}.jpg';sheet.save(p);records.append(dict(candidate_index=idx,instance=r['instance'],observations=nearest,pose_review='pending',sheet=str(p)))
(out/'track_review_packets.json').write_text(json.dumps(records,indent=2))
print(len(records),'track review packets prepared')
