"""Ordinary-video loader with the existing G0 crop and instance-mask recipe."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('/mnt/ojc/workplace1/4danyone')
sys.path.insert(0,str(PROJECT/'repo'))
from source_io import read_source, source_identity, sha


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def selected(ident):
    path=ROOT/'SELECTED_TRACKS.json'
    assert sha(path)==path.with_suffix('.sha256').read_text().strip()
    data=json.loads(path.read_text())
    assert len(data['tracks'])==8 and len({row['id'] for row in data['tracks']})==8
    return next(row for row in data['tracks'] if row['id']==ident)


def encode(path,images,lossless=False):
    import av
    assert not path.exists()
    temporary=path.with_name(path.stem+'.writing.mp4')
    assert not temporary.exists()
    with av.open(str(temporary),'w') as output:
        stream=None
        for index,image in enumerate(images):
            if stream is None:
                stream=output.add_stream('libx264rgb' if lossless else 'libx264',rate=25)
                stream.width=image.width;stream.height=image.height
                stream.pix_fmt='rgb24' if lossless else 'yuv420p'
                stream.options={'crf':'0' if lossless else '18','preset':'fast','threads':'2'}
            frame=av.VideoFrame.from_image(image);frame.pts=index;frame.time_base=Fraction(1,25)
            for packet in stream.encode(frame):output.mux(packet)
        assert stream is not None
        for packet in stream.encode():output.mux(packet)
    with av.open(str(temporary)) as con:
        frames=list(con.decode(video=0));assert len(frames)==121
        assert all(abs(float(f.pts*f.time_base)-i/25)<1e-8 for i,f in enumerate(frames))
    temporary.replace(path)


def prepare(ident):
    import numpy as np
    from PIL import Image,ImageDraw
    choice=selected(ident);source=Path(choice['source_path'])
    assert source_identity(source)==choice['source_identity']
    tracks=json.loads((ROOT/'sources'/f'{choice["source_id"]}_tracks.json').read_text())
    observations={r['frame']:r for r in tracks['tracks'][str(choice['track_id'])]}
    native=np.asarray(tracks['timestamps'],float)
    clock=choice['start_time']+np.arange(121)/25
    picked=np.abs(native[None,:]-clock[:,None]).argmin(1)
    assert len(set(picked))==121 and np.all(np.diff(picked)>0)
    assert picked[0]>=choice['start_frame'] and picked[-1]<=choice['end_frame']
    assert all(int(i) in observations for i in picked),'Broken target track; no fabricated boxes'
    assert np.abs(native[picked]-clock).max()<=np.diff(native).max()/2+1e-6
    boxes=np.asarray([observations[int(i)]['bbox'] for i in picked])
    extent=(boxes[:,2:]-boxes[:,:2]).max(0)*1.3
    unit=int(np.ceil(max(extent[0]/11,extent[1]/20)/2)*2);width,height=11*unit,20*unit
    dest=ROOT/'inputs'/ident;dest.mkdir(parents=True,exist_ok=False)
    (dest/'inputs').mkdir();(dest/'research').mkdir();(dest/'segmentation/frames').mkdir(parents=True)
    save(dest/'candidate.json',choice)
    frames={int(i):None for i in picked}
    for index,timestamp,rgb in read_source(source):
        if index in frames:frames[index]=(timestamp,rgb)
        if index>=int(picked[-1]):break
    assert all(value is not None for value in frames.values())
    crops=[];observed=[];mapping=[]
    for slot,(index,box) in enumerate(zip(picked,boxes,strict=True)):
        timestamp,rgb=frames[int(index)];sh,sw=rgb.shape[:2]
        assert width<=sw and height<=sh,'Required crop exceeds original; no implicit scale reduction'
        lo=np.rint((box[:2]+box[2:]-[width,height])/2).astype(int)
        lo=np.maximum(0,np.minimum(lo,[sw-width,sh-height]));hi=lo+[width,height]
        b=box-np.tile(lo,2)
        assert b[0]>=0 and b[1]>=0 and b[2]<=width and b[3]<=height
        image=Image.fromarray(rgb);crop=image.crop((*lo,*hi));crops.append(crop)
        crop.save(dest/'segmentation/frames'/f'{slot:05d}.jpg',quality=95,subsampling=0)
        draw=ImageDraw.Draw(image);draw.rectangle(tuple(box),outline='lime',width=4)
        draw.text((12,12),f'{ident} REAL SOURCE; t={timestamp:.3f}',fill='white',stroke_width=2,stroke_fill='black')
        preview_h=int(round(sh*960/sw/2)*2);observed.append(image.resize((960,preview_h)))
        mapping.append(dict(model_frame=slot,model_timestamp_s=slot/25,target_source_time_s=float(clock[slot]),
            source_width=sw,source_height=sh,
            source_frame_index=int(index),source_timestamp_s=timestamp,
            nearest_frame_error_s=float(timestamp-clock[slot]),bbox_source_xyxy=box.tolist(),
            crop_xyxy=[*lo.tolist(),*hi.tolist()],bbox_input_xyxy=b.tolist(),
            source_to_crop=[[1,0,int(-lo[0])],[0,1,int(-lo[1])],[0,0,1]],
            enhanced=False,interpolated=False,undistorted=False))
    encode(dest/'source_clip.mp4',observed)
    encode(dest/'crop_unsegmented.mp4',crops,lossless=True)
    save(dest/'inputs/frame_mapping.json',dict(frames=mapping,output_fps=25,output_frames=121,
        size=[width,height],source_identity=choice['source_identity'],start_time=choice['start_time'],
        timestamp_origin='First target time is the selected original source timestamp; no change in playback speed',
        camera='motion_original_frame_v2: estimated original-frame K, known target boxes; source motion class must pass; no calibrated camera',
        crop_recipe='G0 constant dimensions: max bbox extent*1.3, 11:20; translate per frame',
        source_preview='source_clip.mp4 is annotated/downscaled for viewing; original pixels used for crop'))
    print(ident,'crop prepared',width,height,flush=True)


def segment(ident):
    import av
    import numpy as np
    import torch
    from PIL import Image
    selected(ident)
    assert torch.cuda.device_count()==1
    uuid=str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    assert uuid==os.environ['FD_EXPECTED_GPU_UUID'].removeprefix('GPU-')
    dest=ROOT/'inputs'/ident;mapping=json.loads((dest/'inputs/frame_mapping.json').read_text())
    sys.path.insert(0,str(PROJECT/'waymo/segmentation'))
    from sam2.build_sam import build_sam2_video_predictor
    from sam_transfer_guard import install_transfer_guard
    box=np.asarray(mapping['frames'][0]['bbox_input_xyxy'],np.float32);x0,y0,x1,y1=box;bw,bh=x1-x0,y1-y0
    points=np.asarray([[x0+.5*bw,y0+.15*bh],[x0+.5*bw,y0+.45*bh],[x0+.45*bw,y0+.8*bh]],np.float32)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',
            str(PROJECT/'waymo/segmentation/sam2.1_hiera_large.pt'),device='cuda',apply_postprocessing=False)
        install_transfer_guard(predictor)
        state=predictor.init_state(video_path=str(dest/'segmentation/frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
        predictor.add_new_points_or_box(state,frame_idx=0,obj_id=1,points=points,labels=np.ones(3,np.int32),box=box)
        masks=[];seen=[]
        for index,ids,logits in predictor.propagate_in_video(state):
            assert ids==[1] and bool(torch.isfinite(logits).all())
            masks.append((logits[0,0].sigmoid().float().cpu().numpy()*255).round().astype(np.uint8));seen.append(index)
    assert seen==list(range(121));masks=np.asarray(masks)
    np.savez_compressed(dest/'segmentation/target_masks.npz',masks=masks)
    save(dest/'segmentation/manifest.json',dict(points=points.tolist(),box=box.tolist(),frames=121,gpu_uuid=uuid,
        recipe='Old G0 bbox + fixed face/torso/leg points; no manual output-based prompt change',transfer_guard='existing checked sigmoid transfer',visual_review='pending'))
    images=[]
    with av.open(str(dest/'crop_unsegmented.mp4')) as con:
        for i,frame in enumerate(con.decode(video=0)):
            rgb=frame.to_ndarray(format='rgb24');alpha=masks[i].astype(np.float32)[...,None]/255
            images.append(Image.fromarray(np.rint(rgb*alpha+255*(1-alpha)).clip(0,255).astype(np.uint8)))
    encode(dest/'source_crop.mp4',images,lossless=True)
    (dest/'inputs/main.mp4').symlink_to('../source_crop.mp4')
    binary=masks>127;boxes=np.asarray([x['bbox_input_xyxy'] for x in mapping['frames']]);areas=binary.sum((1,2))/((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]));core=[]
    for mask,b in zip(binary,boxes,strict=True):
        xa,ya,xb,yb=b;w=xb-xa;h=yb-ya;x0,y0,x1,y1=np.rint([xa+.3*w,ya+.2*h,xa+.7*w,ya+.6*h]).astype(int)
        patch=mask[max(y0,0):min(y1,mask.shape[0]),max(x0,0):min(x1,mask.shape[1])];core.append(float(patch.mean()) if patch.size else 0.)
    passed=bool(areas.min()>.15 and np.median(areas)>.3 and areas.min()/np.median(areas)>.55 and min(core)>.3 and np.quantile(core,.1)>.55)
    save(dest/'research/batch_gate_mask.json',dict(passed=passed,mask_area_min=float(areas.min()),
        mask_area_median=float(np.median(areas)),torso_min=min(core),torso_p10=float(np.quantile(core,.1)),
        scope='Unchanged G0 numeric foreground validity gate; not appearance or identity quality approval'))
    if not passed:raise RuntimeError('G0 instance mask validity failed; retain failure without replacing person')
    mapping['video_sha256']=sha(dest/'source_crop.mp4');mapping['background_compositing']='straight alpha over white'
    save(dest/'inputs/frame_mapping.json',mapping)
    print(ident,'foreground validity passed; visual review pending',flush=True)


def motion(ident):
    from motion_original_frame import run
    run(ident)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','segment','motion']);p.add_argument('id');a=p.parse_args()
    {'prepare':prepare,'segment':segment,'motion':motion}[a.stage](a.id)
