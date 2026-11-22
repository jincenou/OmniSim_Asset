"""Screen actual short windows for sustained ego motion before visual review."""
import asyncio
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps

R=Path(__file__).resolve().parents[1]
OUT=R/'moving_candidates_v2'
def scan(path):
    scene=json.loads(path.read_text());poses=np.asarray(scene['poses']);rows=[]
    for track in scene['tracks']:
        obs=track['observations']
        groups=np.split(np.arange(len(obs)),np.flatnonzero(np.diff([o['frame_index'] for o in obs])!=1)+1)
        for group in groups:
            if len(group)<25:continue
            for start in range(0,len(group)-24,8):
                part=[obs[int(i)] for i in group[start:start+25]]
                frames=[x['frame_index'] for x in part]
                intervals=np.diff([x['timestamp_micros'] for x in part])/1e6
                if np.max(np.abs(intervals-.1))>.005:continue
                boxes=np.array([x['bbox_xyxy'] for x in part]);h=boxes[:,3]-boxes[:,1]
                if h.min()<150 or np.median(h)<190:continue
                if boxes[:,0].min()<16 or boxes[:,1].min()<16 or boxes[:,2].max()>1904 or boxes[:,3].max()>1264:continue
                p=poses[frames];dist=np.linalg.norm(np.diff(p[:,:3,3],axis=0),axis=1)
                if dist.sum()<1 or (dist>.02).sum()<18:continue
                identity=scene['segment']+':'+(track['associated_lidar_track_id'] or f"cam{track['camera_index']}:{track['track_id']}")
                rotation=p[:,:3,:3]@p[0,:3,:3].T
                angle=float(np.rad2deg(np.arccos(np.clip((np.trace(rotation,axis1=1,axis2=2)-1)/2,-1,1))).max())
                rows.append({**{k:v for k,v in track.items() if k!='observations'},'identity_key':identity,
                    'scene':scene['scene'],'split':'val','segment':scene['segment'],'observations':part,
                    'raw_tfrecord':scene['raw'],'cache':str(path),'native_frame_count':25,
                    'duration_s':(part[-1]['timestamp_micros']-part[0]['timestamp_micros'])/1e6,
                    'height_min_px':float(h.min()),'height_median_px':float(np.median(h)),
                    'ego_path_m':float(dist.sum()),'ego_rotation_deg':angle,
                    'moving_intervals_over_2cm':int((dist>.02).sum()),
                    'category':'sustained_moving_camera_short','human_action':'unreviewed',
                    'selection_score':float(np.median(h))})
    return rows

def preview(row,index):
    label=f'M{index:02d}';canvas=Image.new('RGB',(720,390),'white');draw=ImageDraw.Draw(canvas)
    draw.text((5,5),f'{label} scene{row["scene"]} ego{row["ego_path_m"]:.2f}m /2.4s | action UNREVIEWED',fill='black')
    for slot,i in enumerate((0,8,16,24)):
        o=row['observations'][i];path=Path('/mnt/public_data/waymo/val')/row['scene']/'images'/f'{o["frame_index"]:03d}_{row["camera_index"]}.jpg'
        x0,y0,x1,y1=o['bbox_xyxy'];margin=.12*(y1-y0)
        with Image.open(path) as im:
            tile=im.crop((max(0,int(x0-margin)),max(0,int(y0-margin)),min(im.width,int(x1+margin)),min(im.height,int(y1+margin))))
            tile=ImageOps.contain(tile,(176,340));canvas.paste(tile,(slot*180+(180-tile.width)//2,40+(340-tile.height)//2))
        draw.text((slot*180+5,23),f'frame{o["frame_index"]}',fill='black')
    path=OUT/f'{label}.jpg';canvas.save(path,quality=94)
    return {**row,'review_id':label,'review_image':str(path)}

async def main():
    OUT.mkdir(exist_ok=False)
    sem=asyncio.Semaphore(2)
    async def one(p):
        async with sem:return await asyncio.to_thread(scan,p)
    results=await asyncio.gather(*map(one,sorted((R.parent/'reconstruction/track_reserve/native').glob('*.json'))))
    known={x['identity_key'] for x in json.loads((R/'sample_manifest.json').read_text())['candidates']}
    candidates=sorted((x for group in results for x in group),key=lambda x:x['selection_score'],reverse=True)
    selected=[];seen=set(known);scenes={}
    for row in candidates:
        if row['identity_key'] in seen or scenes.get(row['scene'],0)>=2:continue
        selected.append(row);seen.add(row['identity_key']);scenes[row['scene']]=scenes.get(row['scene'],0)+1
        if len(selected)==12:break
    reviews=await asyncio.gather(*(asyncio.to_thread(preview,row,i) for i,row in enumerate(selected)))
    data={'state':'sustained_ego_motion_screened_action_and_occlusion_review_pending',
        'criteria':{'native_frames':25,'min_height_px':150,'median_height_px':190,'ego_path_m':1,'moving_intervals_over_2cm_min':18},
        'candidates':reviews,'scope':'Processed ego poses only; raw calibration and actual camera exposure poses must be audited before inference. Different tracks, not guaranteed different natural people.'}
    (OUT/'manifest.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'screened_windows':len(candidates),'selected_tracks':len(reviews)}),flush=True)

if __name__=='__main__':asyncio.run(main())
