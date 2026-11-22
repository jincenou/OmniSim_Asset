"""Traceable, unfiltered four-stage demos from existing validated dynamic assets."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation,Slerp
import torch
from evaluate_asset61 import Renderer,save,sha
from render61_helpers import render,composite,uint8,pane,decoded
from video_io import MP4Writer

ROUND=Path(__file__).resolve().parents[1];W=ROUND.parent


def details(target,predicted,mask,view,frame):
    ys,xs=np.where(mask)
    if not len(xs):raise ValueError('Empty reference mask')
    x0,x1=int(xs.min()),int(xs.max())+1;y0,y1=int(ys.min()),int(ys.max())+1
    height=y1-y0
    boxes={'upper_body_hands_accessories':(max(0,x0-10),max(0,int(y0+.18*height)),min(352,x1+10),min(640,int(y0+.77*height))),
           'feet':(max(0,x0-10),max(0,int(y0+.72*height)),min(352,x1+10),min(640,y1+10))}
    sheet=Image.new('RGB',(1024,560),(40,40,40));draw=ImageDraw.Draw(sheet)
    for row,(name,box) in enumerate(boxes.items()):
        for col,array in enumerate((target,predicted)):
            im=Image.fromarray(uint8(array)).crop(box)
            im=ImageOps.contain(im,(500,252),Image.Resampling.NEAREST)
            sheet.paste(im,(col*512+(512-im.width)//2,row*280+24))
            draw.text((col*512+5,row*280+5),f'{name}: '+('generated target' if col==0 else 'GS render'),fill='white')
    return sheet,{'view':view,'frame':frame,'boxes':boxes,'method':'Reference foreground-relative regions; no anatomical detector. Nearest pixel enlargement, no enhancement.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
    job=json.loads(args.job.read_text());source=Path(job['source']);data=Path(job['data']);checkpoint=Path(job['checkpoint'])
    output=Path(job['output']);physical=int(os.environ['CUDA_VISIBLE_DEVICES']);assert physical in (4,5,6,7)
    expected=subprocess.check_output(['nvidia-smi',f'--id={physical}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();actual=str(torch.cuda.get_device_properties(0).uuid)
    if torch.cuda.device_count()!=1 or expected!=actual:raise ValueError('Unauthorized CUDA device')
    mapping_path=Path(job['mapping']);native_path=Path(job['native_mapping'])
    mapping=json.loads(mapping_path.read_text());native=json.loads(native_path.read_text())
    if mapping['track_id']!=native['track_id']:raise ValueError('Source identity mismatch')
    frames=mapping['frames'];assert len(frames)==61
    native_frames={row['frame_index']:row for row in native['frames']}
    if any(row['left_source_frame'] not in native_frames for row in frames):raise ValueError('Source timeline not covered')
    if not np.allclose([row['video_seconds'] for row in frames],np.arange(61)/25,atol=1e-8):raise ValueError('Input timeline changed')
    renderer=Renderer(checkpoint,data);assert len(renderer.parser.camtoworlds)==24
    poses=renderer.parser.camtoworlds;views=[0,6,12];indices=[renderer.indices[v] for v in views]
    Ks=np.stack([renderer.parser.Ks_dict[renderer.parser.camera_ids[i]] for i in indices])
    if any(renderer.parser.imsize_dict[renderer.parser.camera_ids[i]]!=(352,640) for i in indices):raise ValueError('This demo protocol requires half-resolution asset cameras')
    allposes=poses[[renderer.indices[v] for v in range(24)]]
    closed=np.concatenate([allposes,allposes[:1]]);at=np.linspace(0,24,72,endpoint=False)
    orbit=np.repeat(np.eye(4)[None],72,0);orbit[:,:3,:3]=Slerp(np.arange(25),Rotation.from_matrix(closed[:,:3,:3]))(at).as_matrix()
    orbit[:,:3,3]=np.stack([np.interp(at,np.arange(25),closed[:,axis,3]) for axis in range(3)],-1)
    output.mkdir(parents=True,exist_ok=False)
    inputpath=Path(job['input']);generated=source/'videos/00.mp4'
    inputs=decoded(inputpath);targets=decoded(generated);raw_cache={};source_hashes={};timeline=[];minimum=10**9;regions=[]
    contact=Image.new('RGB',(704,7*340),'#333333');draw=ImageDraw.Draw(contact)
    with ExitStack() as stack:
        writers={}
        for name,width,height in [('four_panel',1408,664),('fixed_views',1056,640),('backgrounds',1056,640),('orbit_fixed_time',352,640)]:
            writer=MP4Writer(str(output/f'{name}.mp4'),25,width,height);writers[name]=writer;stack.callback(writer.close)
        for frame in range(61):
            rgb,alpha=render(renderer,poses[indices],Ks,frame,np.zeros(3));images=composite(rgb,alpha)
            counts=(alpha>.1).sum((1,2,3));minimum=min(minimum,int(counts.min()))
            if counts.min()<100:raise ValueError(f'Empty rendered view frame{frame}')
            writers['fixed_views'].append_data(uint8(np.concatenate(images,axis=1)))
            backgrounds=np.concatenate([composite(rgb[:1],alpha[:1],bg)[0] for bg in (0.,.5,1.)],axis=1)
            writers['backgrounds'].append_data(uint8(backgrounds))
            slot=frames[frame];original=native_frames[slot['left_source_frame']];path=Path(original['raw_source_path'])
            if str(path) not in raw_cache:
                source_hashes[str(path)]=sha(path)
                with Image.open(path) as im:
                    raw=im.convert('RGB');crop=Image.open(original['path']).convert('RGB')
                    ImageDraw.Draw(crop).rectangle(original['bbox_input_xyxy'],outline='lime',width=2)
                    inset=ImageOps.contain(raw,(132,88));crop.paste(inset,(0,0));raw_cache[str(path)]=crop
            raw=raw_cache[str(path)];inp=next(inputs);gen=next(targets)
            panels=[pane(raw,'Waymo undistorted crop; native held frame'),pane(inp,'Actual input; RIFE + target mask'),pane(gen,'4DAnyone generated view0'),pane(Image.fromarray(uint8(images[0])),'GS same generated camera, t')]
            array=np.concatenate(panels,axis=1);writers['four_panel'].append_data(array)
            if frame%10==0:
                row=frame//10;small=Image.fromarray(array).resize((704,332),Image.Resampling.LANCZOS);contact.paste(small,(0,row*340));draw.text((3,row*340+22),f't={frame/25:.2f}s',fill='orange')
            if frame in (0,30,60):
                reference,ra=renderer.render(0,frame)
                if max(np.abs(reference-rgb[0]).max(),np.abs(ra-alpha[0,...,0]).max())>1e-5:raise ValueError('Demo and metric renderer mismatch')
                Image.fromarray(uint8(backgrounds)).save(output/f'backgrounds_{frame:03d}.jpg',quality=95)
                for i,view in enumerate(views):
                    with Image.open(data/'images'/f'cam{view:02d}'/f'{frame:06d}.png') as im:rgba=np.asarray(im.convert('RGBA'),dtype=np.float32)/255
                    target=rgba[...,:3]*rgba[...,3:]+(1-rgba[...,3:])*.5
                    sheet,region=details(target,composite(rgb[i],alpha[i],.5),rgba[...,3]>.5,view,frame)
                    sheet.save(output/f'details_view{view:02d}_{frame:03d}.jpg',quality=95);regions.append(region)
            timeline.append({'output_frame':frame,'seconds':frame/25,'source_frame':original['frame_index'],'timestamp_micros':original['frame_timestamp_micros'],'source_path':str(path),'interpolated_input':slot['interpolated']})
        orbit_min=10**9
        for pose in orbit:
            rgb,alpha=render(renderer,pose[None],Ks[:1],30,np.zeros(3));count=int((alpha>.1).sum());orbit_min=min(orbit_min,count)
            if count<100:raise ValueError('Empty orbit view')
            writers['orbit_fixed_time'].append_data(uint8(composite(rgb,alpha)[0]))
    contact.save(output/'review_contact.jpg',quality=95)
    videos={}
    for path in output.glob('*.mp4'):
        with imageio.get_reader(str(path),input_params=['-threads','1']) as reader:
            meta=reader.get_meta_data();count=sum(1 for _ in reader)
            expected_count=72 if path.stem=='orbit_fixed_time' else 61
            assert count==expected_count and meta['fps']==25
            videos[path.name]={'frames':count,'fps':25,'size':meta['size'],'sha256':sha(path)}
    np.savez_compressed(output/'cameras.npz',fixed_c2w=poses[indices],fixed_view_ids=views,K=Ks,orbit_c2w=orbit,
        orbit_model_time=np.full(72,.5),frame_seconds=np.arange(61)/25,canonical_to_asset=renderer.parser.transform)
    save(output/'validation.json',{'state':'rendered_decoded_visual_review_pending','job':job,'gpu':physical,'cuda_uuid':actual,
        'checkpoint_sha256':sha(checkpoint),'input_sha256':sha(inputpath),'generated_view0_sha256':sha(generated),
        'mapping_sha256':sha(mapping_path),'native_mapping_sha256':sha(native_path),'videos':videos,'raw_source_hashes':source_hashes,
        'source_timeline':timeline,'minimum_fixed_view_foreground_pixels':minimum,'minimum_orbit_foreground_pixels':orbit_min,
        'detail_regions':regions,'source_camera_registration':False,'point_filtering':False,
        'scope':'Only generated and GS panels share camera. Native Waymo panel holds left observation at each interpolated slot. Source/context crop and view-region enlargements are display-only; full unfiltered models preserved.'})
    print(json.dumps({'label':job['label'],'state':'rendered_decoded_visual_review_pending','videos':list(videos)}),flush=True)


if __name__=='__main__':main()
