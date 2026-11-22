"""Calibrated undistortion and a recorded moving crop; no synthetic images."""
import hashlib
import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

R=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ['FDANYONE_SAMPLE_ROOT'])


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(2)
    audit_path=R/'camera_audit'/os.environ['FDANYONE_SAMPLE_ID']/'audit.json';audit=json.loads(audit_path.read_text())
    if audit['state']!='processed_camera_composition_checked_exposure_unknown':raise ValueError('Processed camera composition not checked')
    cameras=np.load(audit_path.parent/'native_cameras.npz');K=cameras['K'];dist=cameras['distortion']
    ROOT.mkdir(parents=True,exist_ok=False);native=ROOT/'native';native.mkdir()
    records=audit['records'];boxes=[]
    for row in records:
        x0,y0,x1,y1=row['bbox_xyxy_distorted']
        x,y=np.meshgrid(np.linspace(x0,x1,17),np.linspace(y0,y1,17))
        pixels=np.stack([x,y],-1).reshape(-1,1,2)
        u=cv2.undistortPointsIter(pixels,K,dist,None,K,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,1e-10)).reshape(-1,2)
        boxes.append(np.r_[u.min(0),u.max(0)])
    boxes=np.asarray(boxes)
    # Constant crop dimensions: preserve scale changes; only translate the crop.
    extent=(boxes[:,2:]-boxes[:,:2]).max(0)*1.3
    unit=int(np.ceil(max(extent[0]/11,extent[1]/20)/2)*2)
    width,height=11*unit,20*unit
    if width>1920 or height>1280:raise ValueError('Required crop exceeds source; no implicit resize')
    frames=[];affines=[];checks=[];previews=[]
    # Remap directly into the calibrated crop; an arbitrary full rectified canvas must not cut observed source pixels.
    for index,(row,box) in enumerate(zip(records,boxes,strict=True)):
        path=Path(row['source_path'])
        if sha(path)!=row['processed_jpeg_sha256']:raise ValueError('Image changed since camera audit')
        pixels=np.asarray(Image.open(path).convert('RGB'))
        lo=np.rint((box[:2]+box[2:]-[width,height])/2).astype(int)
        centered=lo.copy();lo=np.maximum(0,np.minimum(lo,[1920-width,1280-height]))
        box_old=box-np.tile(lo,2)
        if not (box_old[0]>=0 and box_old[1]>=0 and box_old[2]<=width and box_old[3]<=height):lo=centered
        hi=lo+[width,height]
        affine=np.eye(3);affine[:2,2]=-lo
        maps=cv2.initUndistortRectifyMap(K,dist,np.eye(3),affine@K,(width,height),cv2.CV_32FC1)
        crop=cv2.remap(pixels,*maps,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);b=box-np.tile(lo,2)
        if not (b[0]>=0 and b[1]>=0 and b[2]<=width and b[3]<=height):raise ValueError('Annotated person clipped by crop')
        target=native/f'{index:05d}.png';Image.fromarray(crop).save(target)
        # Test K' = A K against full-image rays, including off-center principal point.
        rays=np.array([[-.2,-.1,1],[0,0,1],[.3,.2,1.]])
        full=rays@K.T;full/=full[:,2:];direct=rays@(affine@K).T;direct/=direct[:,2:]
        checks.append(float(np.abs(direct-full@affine.T).max()));affines.append(affine)
        frames.append({'native_index':index,'frame_index':row['frame_index'],'raw_source_path':str(path),
            'raw_source_sha256':sha(path),'path':str(target),'sha256':sha(target),
            'frame_timestamp_micros':row['frame_timestamp_micros'],
            'sample_timestamp_s':row['sample_timestamp_s'],
            'crop_xyxy':[*lo.tolist(),*hi.tolist()],'undistorted_to_input':affine.tolist(),
            'K_input':(affine@K).tolist(),'bbox_input_xyxy':b.tolist(),
            'interpolated':False,'enhanced':False,'undistorted':True,'rectification':'direct remap to A@K; no full-canvas clipping'})
        if index in np.rint(np.linspace(0,len(records)-1,5)).astype(int):
            panel=Image.fromarray(crop);d=ImageDraw.Draw(panel);d.rectangle(tuple(b),outline='lime',width=2)
            d.text((4,4),f'Native f{row["frame_index"]}',fill='white');previews.append(panel)
    if max(checks)>1e-9:raise ValueError('Crop projection check failed')
    canvas=Image.new('RGB',(width*len(previews),height))
    for i,p in enumerate(previews):canvas.paste(p,(width*i,0))
    canvas.save(ROOT/'native_contact.jpg',quality=95)
    np.savez_compressed(ROOT/'native_cameras.npz',K_input=np.asarray(affines)@K,
        R_world_to_cv_camera=cameras['R_world_to_cv_camera'],
        frame_cv_camera_to_world=cameras['frame_cv_camera_to_world'],
        undistorted_to_input=affines,sample_timestamps_s=cameras['sample_timestamps_s'])
    save(ROOT/'native.json',{'preprocessing_protocol':'random8-frontend-v1.2','state':'native_undistorted_moving_crop_ready','review_id':os.environ['FDANYONE_SAMPLE_ID'],
        'scene':audit['scene'],'track_id':audit['track_id'],'camera_index':audit['camera_index'],
        'size':[width,height],'frames':frames,'crop_projection_max_abs_px':max(checks),
        'source_camera_audit':str(audit_path),'source_camera_audit_sha256':sha(audit_path),
        'camera_npz_sha256':sha(ROOT/'native_cameras.npz'),
        'camera_timing':'nominal frame time; exposure unavailable','raw_pixel_equality':'unverified without raw source','scope':'Motion diagnostic on unsegmented real images with explicit target boxes. Moving crop translations recorded, no resizing, enhancement or global world-motion acceptance.'})
    print(json.dumps({'state':'prepared','frames':len(frames),'size':[width,height],'projection_error':max(checks)}),flush=True)


if __name__=='__main__':main()
