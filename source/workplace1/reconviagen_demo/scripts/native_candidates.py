"""Read native Waymo labels; identify archived crops against labelled vehicles."""
from pathlib import Path
import sys, struct, json, io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/mnt/ojc/data/waymo_track_bbox_10/python_pkgs')
from waymo_open_dataset import dataset_pb2
cv2.setNumThreads(1)

def frames(path):
    with path.open('rb') as f:
        idx=0
        while head:=f.read(12):
            if len(head)!=12: raise ValueError('truncated TFRecord header')
            n=struct.unpack('<Q',head[:8])[0];data=f.read(n)
            if len(data)!=n or len(f.read(4))!=4:raise ValueError('truncated record')
            frame=dataset_pb2.Frame();frame.ParseFromString(data)
            yield idx, frame
            idx+=1

def run(row):
    scene=int(row['obj_id'].split('_')[2])
    ann=None
    for p in Path('/mnt/public_data/waymo/annotations/val').glob('*.json'):
        data=json.loads(p.read_text())
        if data['scene_id']==scene:ann=data;break
    if ann is None:raise ValueError(scene)
    raw=Path('/mnt/public_data/waymo_raw/validation')/(ann['scene_name']+'.tfrecord')
    source=cv2.imread(str(Path(row['archive_root'])/row['source_copy']))
    sift=cv2.SIFT_create(nfeatures=1000)
    kp,des=sift.detectAndCompute(cv2.cvtColor(source,cv2.COLOR_BGR2GRAY),None)
    out=ROOT/'inputs'/row['name'];out.mkdir(exist_ok=True)
    obs=[]; matches=[]
    for fi,frame in frames(raw):
        lasers={l.id:l for l in frame.laser_labels}
        images={x.name:x for x in frame.images}
        cals={x.name:x for x in frame.context.camera_calibrations}
        for cam in frame.camera_labels:
            im=None
            for label in cam.labels:
                if label.type!=1:continue
                b=label.box
                if b.length<50 or b.width<35:continue
                laser=lasers.get(label.association.laser_object_id)
                rec=dict(frame_index=fi,camera=int(cam.name),camera_2d_id=label.id,
                    laser_object_id=label.association.laser_object_id or None,
                    box_xyxy=[b.center_x-b.length/2,b.center_y-b.width/2,b.center_x+b.length/2,b.center_y+b.width/2],
                    timestamp_micros=frame.timestamp_micros,
                    image_timestamp_s=images[cam.name].pose_timestamp,
                    ego_to_world=list(frame.pose.transform),camera_pose_to_world=list(images[cam.name].pose.transform),
                    camera_to_ego=list(cals[cam.name].extrinsic.transform),intrinsic=list(cals[cam.name].intrinsic),
                    image_size=[cals[cam.name].width,cals[cam.name].height])
                if laser:
                    lb=laser.box
                    rec['laser_box']=[lb.center_x,lb.center_y,lb.center_z,lb.length,lb.width,lb.height,lb.heading]
                    rec['speed_mps']=[laser.metadata.speed_x,laser.metadata.speed_y]
                obs.append(rec)
                if fi%5:continue
                if im is None:im=cv2.imdecode(np.frombuffer(images[cam.name].image,np.uint8),cv2.IMREAD_COLOR)
                x0,y0,x1,y1=rec['box_xyxy'];pad=max(x1-x0,y1-y0)*.15
                box=[max(0,int(x0-pad)),max(0,int(y0-pad)),min(im.shape[1],int(x1+pad)),min(im.shape[0],int(y1+pad))]
                crop=im[box[1]:box[3],box[0]:box[2]]
                k2,d2=sift.detectAndCompute(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),None)
                if d2 is None:continue
                good=[a for pair in cv2.BFMatcher().knnMatch(des,d2,k=2) if len(pair)==2 for a,b in [pair] if a.distance<.8*b.distance]
                if len(good)<4:continue
                a=np.float32([kp[m.queryIdx].pt for m in good]);bb=np.float32([k2[m.trainIdx].pt for m in good])
                mat,inliers=cv2.estimateAffinePartial2D(a,bb,method=cv2.RANSAC,ransacReprojThreshold=3)
                if mat is None or inliers.sum()<4:continue
                scale=np.linalg.norm(mat[:,0]);angle=np.degrees(np.arctan2(mat[1,0],mat[0,0]))
                if not .25<scale<6 or abs(angle)>25:continue
                path=out/f'candidate_f{fi:03}_c{cam.name}_{len(matches):03}.jpg'
                cv2.imwrite(str(path),crop)
                matches.append(dict(**rec,inliers=int(inliers.sum()),feature_matches=len(good),
                                    affine=mat.tolist(),crop_box=box,candidate_crop=str(path)))
        if fi%50==0:print(row['name'],fi,len(matches),flush=True)
    matches.sort(key=lambda x:x['inliers'],reverse=True)
    (out/'native_inventory.json').write_text(json.dumps(dict(raw_tfrecord=str(raw),context=ann['scene_name'],
        scene_id=scene,source=row,observations=obs,candidates=matches[:30]),indent=2))
    sheet=Image.new('RGB',(1200,900),'#dddddd');draw=ImageDraw.Draw(sheet)
    for i,m in enumerate(matches[:12]):
        im=ImageOps.contain(Image.open(m['candidate_crop']),(300,195));x=i%4*300;y=i//4*300
        sheet.paste(im,(x,y+35));draw.text((x+3,y+3),f"f{m['frame_index']} c{m['camera']} n={m['inliers']}",fill='black')
        draw.text((x+3,y+235),(m['laser_object_id'] or m['camera_2d_id'])[:28],fill='black')
    sheet.save(out/'native_match_candidates.jpg')
    print(row['name'],'DONE',[(m['frame_index'],m['camera'],m['inliers']) for m in matches[:3]],flush=True)

if __name__=='__main__':
    rows=json.loads((ROOT/'docs/candidate_sources.json').read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,rows))
