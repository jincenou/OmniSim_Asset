"""Prepare independently identified, fixed-crop Waymo experiments."""
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image,ImageDraw,ImageOps

BASE=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ['FDANYONE_WAYMO_ROOT']).resolve()
DATA=Path('/mnt/public_data/waymo/train')


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def native():
    config=json.loads((ROOT/'config.json').read_text())
    if 'candidate_record' in config:
        record=json.loads(Path(config['candidate_record']).read_text())
        row={**record,'candidate_id':config['candidate'],'camera':['FRONT','FRONT_LEFT','FRONT_RIGHT'][record['camera_index']],
             'track':record['track_id'],'observations':[{**o,'bbox_2d':dict(zip(('xmin','ymin','xmax','ymax'),o['bbox_xyxy']))} for o in record['observations']]}
        data=Path('/mnt/public_data/waymo')/record['split']
    else:
        row=next(x for x in json.loads((BASE/'candidate_manifest.json').read_text()) if x['candidate_id']==config['candidate'])
        data=DATA
    observations=[x for x in row['observations'] if config['start_frame']<=x['frame_index']<=config['start_frame']+8]
    assert len(observations)==9
    boxes=np.array([[o['bbox_2d'][k] for k in ('xmin','ymin','xmax','ymax')] for o in observations])
    lo=np.floor(boxes[:,:2].min(0)-24).astype(int);hi=np.ceil(boxes[:,2:].max(0)+24).astype(int)
    size=((hi-lo+15)//16)*16
    if config.get('portrait_crop'):
        # A fixed exact model aspect ratio preserves the trajectory without a
        # hidden center crop later; no per-frame recentering or image resizing.
        unit=int(np.ceil(max(size[0]/11,size[1]/20)/2)*2)
        size=np.array([11*unit,20*unit])
        if np.any(size>np.array([1920,1280])):
            raise RuntimeError('Fixed portrait crop exceeds source bounds; explicit padding branch required')
        lo=np.floor((boxes[:,:2].min(0)+boxes[:,2:].max(0)-size)/2).astype(int)
    lo=np.maximum(0,np.minimum(lo,np.array([1920,1280])-size));hi=lo+size
    crop=[int(x) for x in np.r_[lo,hi]]
    camera=row['camera_index'];scene=data/row['scene'];k=np.loadtxt(scene/'intrinsics'/f'{camera}.txt')
    K=np.array([[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1]])
    A=np.eye(3);A[:2,2]=-lo
    records=[];poses=[]
    for i,o in enumerate(observations):
        frame=o['frame_index'];pose=np.loadtxt(scene/'ego_to_world'/f'{frame:03d}.txt');poses.append(pose)
        b=boxes[i]-np.tile(lo,2)
        records.append({'input_frame':i,'model_slot':i,'frame_index':frame,'source_path':str(scene/'images'/f'{frame:03d}_{camera}.jpg'),
           'timestamp_micros':o['timestamp_micros'],'relative_seconds':(o['timestamp_micros']-observations[0]['timestamp_micros'])/1e6,
           'video_seconds':i/10,'crop_xyxy':crop,'source_to_input':A.tolist(),'K_input':(A@K).tolist(),'bbox_input_xyxy':b.tolist(),
           'ego_to_world':pose.tolist(),'source_camera_to_world_file':str(scene/'cam_to_world'/f'{frame:03d}_{camera}.txt'),
           'interpolated':False,'enhanced':False})
    poses=np.stack(poses);rot=poses[:,:3,:3]@poses[0,:3,:3].T
    angles=np.degrees(np.arccos(np.clip((np.trace(rot,axis1=1,axis2=2)-1)/2,-1,1)))
    audit={'candidate_id':row['candidate_id'],'scene':row['scene'],'camera':row['camera'],'camera_index':camera,'track_id':row['track'],
       'source_split':config.get('source_split','train'),'identity_key':row.get('identity_key'),
       'source_intrinsic':K.tolist(),'source_distortion':k[4:].tolist(),'undistorted':False,'true_calibration_used_by_gvhmr':False,
       'frames':records,'ego_translation_path_m':float(np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1).sum()),
       'ego_net_displacement_m':float(np.linalg.norm(poses[-1,:3,3]-poses[0,:3,3])),
       'ego_rotation_max_deg':float(angles.max()),'body_height_range_px':[float((boxes[:,3]-boxes[:,1]).min()),float((boxes[:,3]-boxes[:,1]).max())]}
    if audit['ego_translation_path_m']>.1 or audit['ego_rotation_max_deg']>.5:
        raise RuntimeError('Moving camera requires separate adaptation, not this static-camera baseline')
    save(ROOT/'inputs_native/frame_mapping.json',audit)
    b=np.array(records[0]['bbox_input_xyxy']);points=np.array(config['positive_points_box_fraction'])*(b[2:]-b[:2])+b[:2]
    save(ROOT/'segmentation/prompt.json',{'points':points.tolist(),'labels':[1]*len(points),'rationale':config['prompt_rationale']})
    canvas=Image.new('RGB',(240*9,400),'#222222');draw=ImageDraw.Draw(canvas)
    for col,record in enumerate(records[::6]):
        im=Image.open(record['source_path']).convert('RGB').crop(crop)
        d=ImageDraw.Draw(im);d.rectangle(record['bbox_input_xyxy'],outline='lime',width=2)
        if col==0:
            for x,y in points:d.ellipse((x-3,y-3,x+3,y+3),fill='red')
        im=ImageOps.contain(im,(236,370));canvas.paste(im,(col*240+(240-im.width)//2,28))
        draw.text((col*240+4,5),f"{row['candidate_id']} f{record['frame_index']}",fill='white')
    canvas.save(ROOT/'previews/source.jpg',quality=93)
    print(json.dumps({k:v for k,v in audit.items() if k!='frames'}),flush=True)


def frames():
    (ROOT/'inputs').rename(ROOT/'inputs_rife');(ROOT/'inputs').mkdir()
    with av.open(str(ROOT/'inputs_rife/main.mp4')) as con:
        con.streams.video[0].thread_count=1
        for i,frame in enumerate(con.decode(video=0)):
            frame.to_image().save(ROOT/'segmentation/frames'/f'{i:05d}.jpg',quality=100,subsampling=0)
    assert i==20


if __name__=='__main__':
    {'native':native,'frames':frames}[sys.argv[1]]()
