"""Build real-view selection sheets before any model inference."""
from native_candidates import ROOT,frames
from pathlib import Path
import numpy as np
import json,math,cv2
from PIL import Image,ImageDraw,ImageOps

for name in ['service_van','silver_sedan_03','white_sedan']:
    out=ROOT/'inputs'/name;d=json.loads((out/'associated_inventory.json').read_text())
    votes=d['projection_audit']['projected_3d_votes'];lid=max(votes,key=votes.get)
    candidates=[]
    for rec in d['observations']:
        if rec.get('projected_laser_object_id')!=lid:continue
        b=rec['box_xyxy'];width,height=rec['image_size']
        if b[0]<3 or b[1]<3 or b[2]>width-3 or b[3]>height-3:continue
        box=rec['projected_laser_box'];cam=np.array(rec['camera_to_ego']).reshape(4,4)[:3,3]
        delta=cam[:2]-np.array(box[:2]);angle=math.atan2(delta[1],delta[0])-box[6]
        rec['view_azimuth_rad']=math.atan2(math.sin(angle),math.cos(angle))
        rec['bbox_area']=(b[2]-b[0])*(b[3]-b[1]);candidates.append(rec)
    candidates.sort(key=lambda r:r['view_azimuth_rad'])
    selected=[]
    # Angle farthest-point selection; area breaks ties. Input-only inspection.
    if candidates:selected=[max(candidates,key=lambda r:r['bbox_area'])]
    while len(selected)<min(12,len(candidates)):
        available=[r for r in candidates if all(abs(r['frame_index']-s['frame_index'])>=3 for s in selected)]
        if not available:break
        def score(r):
            dist=min(abs(math.atan2(math.sin(r['view_azimuth_rad']-s['view_azimuth_rad']),math.cos(r['view_azimuth_rad']-s['view_azimuth_rad']))) for s in selected)
            return dist*np.sqrt(r['bbox_area'])
        selected.append(max(available,key=score))
    selected.sort(key=lambda r:r['frame_index'])
    byframe={r['frame_index']:r for r in selected}
    for fi,frame in frames(Path(d['raw_tfrecord'])):
        if fi not in byframe:continue
        r=byframe[fi];native=next(x for x in frame.images if x.name==r['camera'])
        image=Image.open(__import__('io').BytesIO(native.image)).convert('RGB')
        b=r['box_xyxy'];pad=max(b[2]-b[0],b[3]-b[1])*.15
        crop=[max(0,int(b[0]-pad)),max(0,int(b[1]-pad)),min(image.width,int(b[2]+pad)),min(image.height,int(b[3]+pad))]
        dest=out/f'real_f{fi:03}_c{r["camera"]}.png';image.crop(crop).save(dest)
        r.update(path=str(dest),crop_box=crop,source_type='real',raw_tfrecord=d['raw_tfrecord'],context=d['context'],
            laser_object_id=lid,association_method='native projected laser box IoU >0.4 and margin >0.15; visually checked seed',
            box_crop_xyxy=[b[0]-crop[0],b[1]-crop[1],b[2]-crop[0],b[3]-crop[1]],
            setup='metadata-assisted',status='selection_candidate')
    (out/'view_candidates.json').write_text(json.dumps(selected,indent=2))
    sheet=Image.new('RGB',(1200,900),'#ddd');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(selected):
        im=ImageOps.contain(Image.open(r['path']),(300,240));x=i%4*300;y=i//4*300
        sheet.paste(im,(x,y+35));draw.text((x+3,y+3),f"{i}: f{r['frame_index']} c{r['camera']} az {math.degrees(r['view_azimuth_rad']):.1f}",fill='black')
    sheet.save(out/'real_view_candidates.jpg')
    print(name,[(i,r['frame_index'],r['camera'],round(math.degrees(r['view_azimuth_rad']),1)) for i,r in enumerate(selected)],flush=True)
