"""Input-only random candidate stream, frozen before any generated outputs."""
from pathlib import Path
from collections import defaultdict
import random,math,json,io,hashlib,argparse
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from native_candidates import ROOT,frames

p=argparse.ArgumentParser();p.add_argument('--output',default='random10_selection');p.add_argument('--min-span',type=float,default=45);p.add_argument('--limit',type=int,default=16);p.add_argument('--exclude-proposed',action='store_true');args=p.parse_args()
out=ROOT/'inputs'/args.output;out.mkdir(exist_ok=True)
if (out/'candidates.json').exists():raise FileExistsError(out/'candidates.json')
excluded=set()
for name in ['service_van','fresh_blue_suv','fresh_black_pickup','fresh_silver_suv','silver_sedan_03','white_sedan']:
    m=json.loads((ROOT/'inputs'/name/'manifest.json').read_text())
    excluded.add(m.get('identity',{}).get('laser_object_id',m['original_obj_id']))
    excluded.update(r['projected_laser_object_id'] for r in m['images'] if r.get('projected_laser_object_id'))
if args.exclude_proposed:
    excluded.update(c['laser_object_id'] for c in json.loads((ROOT/'inputs/random10_selection/candidates.json').read_text())['candidates'])
pool=[]
for scene in ['service_van','silver_sedan_03','white_sedan']:
    data=json.loads((ROOT/'inputs'/scene/'associated_inventory.json').read_text());tracks=defaultdict(list)
    for raw in data['observations']:
        row=dict(raw);lid=row.get('projected_laser_object_id');b=row['box_xyxy'];w,h=row['image_size']
        if not lid or lid in excluded:continue
        if row.get('projected_best_iou',0)<.4 or row.get('projected_iou_margin',0)<.15:continue
        if min(b[0],b[1],w-b[2],h-b[3])<5 or min(b[2]-b[0],b[3]-b[1])<75:continue
        box=row['projected_laser_box'];cam=np.array(row['camera_to_ego']).reshape(4,4)[:3,3]
        a=math.atan2(cam[1]-box[1],cam[0]-box[0])-box[6]
        row['view_azimuth_rad']=math.atan2(math.sin(a),math.cos(a));row['bbox_area']=(b[2]-b[0])*(b[3]-b[1])
        tracks[lid].append(row)
    for lid,rows in sorted(tracks.items()):
        ends=[r for r in rows if abs(math.cos(r['view_azimuth_rad']))>.25]
        if not ends:continue
        selected=[max(ends,key=lambda r:r['bbox_area'])]
        while len(selected)<6:
            available=[r for r in rows if all(abs(r['frame_index']-s['frame_index'])>=5 for s in selected)]
            if not available:break
            def score(r):
                gap=min(abs(math.atan2(math.sin(r['view_azimuth_rad']-s['view_azimuth_rad']),math.cos(r['view_azimuth_rad']-s['view_azimuth_rad']))) for s in selected)
                return (.05+gap)*math.sqrt(r['bbox_area'])
            selected.append(max(available,key=score))
        if len(selected)!=6:continue
        spread=max(abs(math.atan2(math.sin(a['view_azimuth_rad']-b['view_azimuth_rad']),math.cos(a['view_azimuth_rad']-b['view_azimuth_rad']))) for a in selected for b in selected)
        if spread<math.radians(args.min_span):continue
        pool.append(dict(scene=scene,context=data['context'],laser_object_id=lid,raw_tfrecord=data['raw_tfrecord'],views=selected,spread_degrees=math.degrees(spread)))
pool.sort(key=lambda x:(x['context'],x['laser_object_id']))
random.Random(20261023).shuffle(pool)
if not pool:raise ValueError('No eligible identities')
selected=pool[:min(args.limit,len(pool))]
for rank,c in enumerate(selected):c['candidate_id']=f'random_candidate_{rank:02}'
registry=dict(seed=20261023,eligible_pool_size=len(pool),population='three locally available audited Waymo scenes, not the paper 128-scene population',
    criteria=f'native projected 3D-track match IoU>=.4 margin>=.15, image border>=5px, min bbox side>=75px, six views with frame gaps>=5, azimuth span>={args.min_span}deg; exclude previous demo identities',
    amendment_reason='Input-only extension to lower azimuth span; original proposals contain severe persistent occlusion. No reconstruction outputs generated.' if args.exclude_proposed else None,
    query_selection='First ten candidates passing explicit input-only visual review; no generated output inspected; rejected inputs recorded before query freeze',
    candidate_order=[dict(context=c['context'],laser_object_id=c['laser_object_id']) for c in pool],candidates=selected)
(out/'sampling_plan.json').write_text(json.dumps(registry,indent=2))
for raw_path in sorted({c['raw_tfrecord'] for c in selected}):
    byframe=defaultdict(list)
    for c in selected:
        if c['raw_tfrecord']!=raw_path:continue
        for j,row in enumerate(c['views']):byframe[row['frame_index']].append((c,j,row))
    for fi,frame in frames(Path(raw_path)):
        if fi not in byframe:continue
        cameras={}
        for c,j,row in byframe[fi]:
            cam=row['camera']
            if cam not in cameras:cameras[cam]=Image.open(io.BytesIO(next(x.image for x in frame.images if x.name==cam))).convert('RGB')
            im=cameras[cam];b=row['box_xyxy'];pad=max(b[2]-b[0],b[3]-b[1])*.12
            crop=[max(0,int(b[0]-pad)),max(0,int(b[1]-pad)),min(im.width,int(b[2]+pad)),min(im.height,int(b[3]+pad))]
            dest=out/f'{c["candidate_id"]}_view{j}.png';im.crop(crop).save(dest)
            row.update(path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),crop_box=crop,
                box_crop_xyxy=[b[0]-crop[0],b[1]-crop[1],b[2]-crop[0],b[3]-crop[1]])
for c in selected:
    sheet=Image.new('RGB',(1200,600),'#ddd');d=ImageDraw.Draw(sheet)
    for j,row in enumerate(c['views']):
        x=j%3*400;y=j//3*300;im=ImageOps.contain(Image.open(row['path']),(400,260));sheet.paste(im,(x,y+35))
        d.text((x+4,y+7),f'{c["candidate_id"]} view{j} f{row["frame_index"]} c{row["camera"]}',fill='black')
    sheet.save(out/f'{c["candidate_id"]}.jpg',quality=92)
(out/'candidates.json').write_text(json.dumps(registry,indent=2))
print('Prepared',len(selected),'input candidates from eligible pool',len(pool),flush=True)
