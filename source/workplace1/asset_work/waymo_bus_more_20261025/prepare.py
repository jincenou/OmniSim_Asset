"""Targeted bus demo: metadata locates identities; production DINO/SAM extracts evidence."""
import json, sys, struct, hashlib
from pathlib import Path
from types import SimpleNamespace
import cv2, numpy as np
from PIL import Image
P=Path('/mnt/ojc/pre_flux_pipeline'); B=Path(__file__).resolve().parent
sys.path.insert(0,str(P));sys.path.insert(0,'/mnt/ojc/data/waymo_track_bbox_10/python_pkgs')
from waymo_open_dataset import dataset_pb2
from asset_extraction.vision import OfficialGroundedSAM2Adapter,_bbox_iou,_pre_score,_clean_mask,_mask_quality,_crop_box,_export_white_jpg
from asset_extraction.io_utils import write_json
cv2.setNumThreads(1)
NAMES={1:'front',2:'front_left',3:'front_right',4:'side_left',5:'side_right'}
QC=SimpleNamespace(min_mask_area_ratio=.0005,max_mask_area_ratio=.8,min_largest_component_ratio=.6,max_edge_touch_ratio=.35)

def observations(candidate):
    dest=B/'sources'/candidate['key'];dest.mkdir(parents=True,exist_ok=True)
    rows=[];maps={};native_from_cv=np.eye(4);native_from_cv[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]
    with open(candidate['tfrecord'],'rb') as f:
        fi=0
        while head:=f.read(12):
            n=struct.unpack('<Q',head[:8])[0];payload=f.read(n);assert len(f.read(4))==4
            if fi%10==0:
                fr=dataset_pb2.Frame();fr.ParseFromString(payload)
                cals={c.name:c for c in fr.context.camera_calibrations}
                for group in fr.projected_lidar_labels:
                    matches=[l for l in group.labels if l.id.startswith(candidate['track_id']+'_')]
                    if not matches:continue
                    b=matches[0].box;c=cals[group.name]
                    if b.length<160 or b.width<90:continue
                    box=np.array([b.center_x-b.length/2,b.center_y-b.width/2,b.center_x+b.length/2,b.center_y+b.width/2])
                    if box[0]<8 or box[1]<8 or box[2]>c.width-8 or box[3]>c.height-8:continue
                    im=next(i for i in fr.images if i.name==group.name);intr=np.asarray(c.intrinsic)
                    k=np.array([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1.]])
                    if group.name not in maps:maps[group.name]=cv2.initUndistortRectifyMap(k,intr[4:],None,k,(c.width,c.height),cv2.CV_32FC1)
                    rect=cv2.remap(cv2.imdecode(np.frombuffer(im.image,np.uint8),cv2.IMREAD_COLOR),*maps[group.name],cv2.INTER_LINEAR)
                    path=dest/f'f{fi:04}_c{group.name}.jpg';cv2.imwrite(str(path),rect,[cv2.IMWRITE_JPEG_QUALITY,95])
                    corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[0],box[3]],[box[2],box[3]]])
                    corners=cv2.undistortPoints(corners[:,None,:],k,intr[4:],P=k)[:,0,:]
                    rows.append(dict(source_image=str(path),frame_index=fi,camera_name=NAMES[group.name],
                        search_bbox=[*corners.min(0).tolist(),*corners.max(0).tolist()],area=float(b.length*b.width),
                        timestamp_us=round(im.pose_timestamp*1e6),camera_intrinsic=k.tolist(),
                        camera_to_world=(np.array(im.pose.transform).reshape(4,4)@np.array(c.extrinsic.transform).reshape(4,4)@native_from_cv).tolist()))
            fi+=1
    # At most six temporally distinct observations, before any generation output is inspected.
    selected=[]
    for r in sorted(rows,key=lambda x:-x['area']):
        if all(abs(r['frame_index']-s['frame_index'])>=10 for s in selected):selected.append(r)
        if len(selected)==6:break
    return selected

def main():
    candidates=json.loads((B/'search/candidates.json').read_text())
    chosen=[dict(candidates[i],search_index=i) for i in [3,5,13,23,33]]
    write_json(B/'selection.json',dict(candidates=chosen,protocol='metadata-assisted identity search and same-track observation association; no sensor dimensions used in reconstruction or scale',generation='unchanged DINO/SAM2/source audit/FLUX/H/strict curation'))
    adapter=OfficialGroundedSAM2Adapter(Path('/mnt/ojc/third_party/Grounded-SAM-2'),device='cuda:0')
    assets=[]
    for c in chosen:
        result=B/'sources'/c['key']/'segmented.json'
        if result.exists():assets.append(json.loads(result.read_text()));continue
        refs=[]
        for obs in observations(c):
            rgb,tensor=adapter.load_detection_image(obs['source_image'])
            detections=adapter.detect_loaded(rgb,tensor,'bus . coach .')
            if not detections:continue
            d=max(detections,key=lambda d:_bbox_iou(d.bbox_xyxy,obs['search_bbox']))
            overlap=_bbox_iou(d.bbox_xyxy,obs['search_bbox'])
            if overlap<.45:continue
            seg=adapter.segment_loaded_image(rgb,[d.bbox_xyxy])[0]
            mask,components,ratio=_clean_mask(seg.mask,d.bbox_xyxy)
            row=dict(bbox_xyxy=d.bbox_xyxy,category='bus',score=d.score)
            row['pre_score']=_pre_score(row,rgb.shape)
            passed,quality,reason=_mask_quality(row,mask,ratio,rgb.shape,QC)
            if not passed:continue
            box=_crop_box(mask,d.bbox_xyxy,rgb.shape[1],rgb.shape[0]);src=Path(obs['source_image'])
            rgba=Image.fromarray(np.dstack([rgb,mask.astype(np.uint8)*255])).crop(box)
            rgba.thumbnail((768,768),Image.Resampling.LANCZOS);rp=src.with_suffix('.rgba.png');rgba.save(rp)
            white=src.with_suffix('.foreground.jpg');_export_white_jpg(rgb,mask,d.bbox_xyxy,white,768,92)
            refs.append(dict(**obs,bbox_xyxy=d.bbox_xyxy,rgba_path=str(rp),image_path=str(white),
                sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),crop_box=list(box),quality=quality,
                sam_score=seg.sam_score,detector_score=d.score,search_detection_iou=overlap,
                real_view_eligible=row['real_view_eligible'],source_frame_edge=row['source_frame_edge'],native_foreground=row['native_foreground']))
        asset=dict(candidate=c,obj_id='waymo_bus_'+c['key'],scene_id='waymo_'+hashlib.sha256(c['context'].encode()).hexdigest()[:12],observations=refs)
        write_json(result,asset);assets.append(asset);print(c['search_index'],len(refs),'segmented views',flush=True)
    write_json(B/'prepared.json',assets)

if __name__=='__main__':main()
