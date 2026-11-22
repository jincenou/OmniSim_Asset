"""CPU-only review candidates; ego motion is measured, human motion remains unlabeled."""
import asyncio
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
W=ROOT.parent
PUBLIC=Path('/mnt/public_data/waymo/val')

def scan(path):
    scene=json.loads(path.read_text());poses=np.asarray(scene['poses']);rows=[]
    for track in scene['tracks']:
        obs=track['observations']
        groups=np.split(np.arange(len(obs)),np.flatnonzero(np.diff([o['frame_index'] for o in obs])!=1)+1)
        eligible=[g for g in groups if len(g)>=9]
        if not eligible:continue
        group=max(eligible,key=len)
        length=min(len(group),49)
        starts=range(0,len(group)-length+1,max(1,length//2))
        for start in starts:
            part=[obs[int(i)] for i in group[start:start+length]]
            boxes=np.asarray([o['bbox_xyxy'] for o in part]);h=boxes[:,3]-boxes[:,1]
            if np.median(h)<100 or h.min()<60:continue
            a,b=part[0]['frame_index'],part[-1]['frame_index']
            ego=float(np.linalg.norm(np.diff(poses[a:b+1,:3,3],axis=0),axis=1).sum())
            rotations=poses[a:b+1,:3,:3]@poses[a,:3,:3].T
            angle=float(np.degrees(np.arccos(np.clip((np.trace(rotations,axis1=1,axis2=2)-1)/2,-1,1))).max())
            center=(boxes[:,:2]+boxes[:,2:])/2
            boxmotion=float(np.ptp(center,axis=0).max()/np.median(h))
            difficult=bool(h.min()<100 or boxes[:,0].min()<8 or boxes[:,1].min()<8 or boxes[:,2].max()>1912 or boxes[:,3].max()>1272)
            category='short' if length<49 else ('difficult' if difficult else ('moving_camera' if ego>1 or angle>2 else 'near_static_camera'))
            if category=='near_static_camera' and (ego>.1 or angle>.5):continue
            key=scene['segment']+':'+(track['associated_lidar_track_id'] or f"cam{track['camera_index']}:{track['track_id']}")
            rows.append({'identity_key':key,'scene':scene['scene'],'split':'val','segment':scene['segment'],
                **{k:v for k,v in track.items() if k!='observations'},'observations':part,'category':category,
                'native_frame_count':length,'duration_s':(part[-1]['timestamp_micros']-part[0]['timestamp_micros'])/1e6,
                'height_median_px':float(np.median(h)),'height_min_px':float(h.min()),'ego_path_m':ego,
                'ego_rotation_deg':angle,'bbox_relative_motion':boxmotion,'human_action':'unreviewed',
                'human_action_caution':'Image displacement includes ego motion; it is not evidence of walking',
                'camera_calibration_status':'Processed calibration exists; raw calibration and camera axes need verification',
                'raw_tfrecord':scene['raw'],'cache':str(path),'selection_score':float(np.median(h))/(1+max(0,boxmotion-2))})
    return rows

def preview(row,index):
    obs=row['observations'];chosen=[obs[0],obs[len(obs)//2],obs[-1]]
    canvas=Image.new('RGB',(540,350),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,5),f"R{index:02d} scene {row['scene']} {row['category']} | ego {row['ego_path_m']:.2f}m",fill='black')
    draw.text((8,23),f"Native {len(obs)} frames / {row['duration_s']:.2f}s | action UNREVIEWED",fill='black')
    inputs=[]
    for slot,o in enumerate(chosen):
        image=PUBLIC/row['scene']/'images'/f"{o['frame_index']:03d}_{row['camera_index']}.jpg"
        with Image.open(image) as im:
            x0,y0,x1,y1=o['bbox_xyxy'];margin=.12*(y1-y0)
            crop=(max(0,int(x0-margin)),max(0,int(y0-margin)),min(im.width,int(x1+margin)),min(im.height,int(y1+margin)))
            tile=im.crop(crop);tile.thumbnail((176,280));canvas.paste(tile,(slot*180+(180-tile.width)//2,50+(280-tile.height)//2))
        inputs.append({'image':str(image),'frame_index':o['frame_index'],'timestamp_micros':o['timestamp_micros'],'review_crop_xyxy':crop})
    target=ROOT/'review'/f'R{index:02d}.jpg';target.parent.mkdir(exist_ok=True);canvas.save(target,quality=90)
    return {**row,'review_id':f'R{index:02d}','review_image':str(target),'preview_sources':inputs}

async def main():
    sem=asyncio.Semaphore(2)
    async def one(path):
        async with sem:return await asyncio.to_thread(scan,path)
    results=await asyncio.gather(*map(one,sorted((W/'reconstruction/track_reserve/native').glob('*.json'))))
    rows=sorted([row for group in results for row in group],key=lambda r:r['selection_score'],reverse=True)
    selected=[];seen=set();scenes={}
    for category,count in [('moving_camera',8),('near_static_camera',6),('short',4),('difficult',2)]:
        group=[]
        for row in rows:
            if row['category']!=category or row['identity_key'] in seen or scenes.get(row['scene'],0)>=2:continue
            selected.append(row);group.append(row);seen.add(row['identity_key']);scenes[row['scene']]=scenes.get(row['scene'],0)+1
            if len(group)==count:break
    reviews=await asyncio.gather(*map(lambda item:asyncio.to_thread(preview,item[1],item[0]),enumerate(selected)))
    report={'state':'review_candidates_only_not_generation_queue','count':len(reviews),'gpu_work_started':False,
        'human_action_labels_pending':True,'category_counts':{k:sum(r['category']==k for r in reviews) for k in ['moving_camera','near_static_camera','short','difficult']},'candidates':reviews}
    (ROOT/'sample_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='candidates'}),flush=True)

if __name__=='__main__':asyncio.run(main())
