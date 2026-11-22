"""Validate an animated PLY and render fixed-camera, full-time evidence."""
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import shutil

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
import torch

R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(Path('/mnt/ojc/workplace1/4danyone/waymo/round2/backend/src')),str(Path('/mnt/ojc/workplace1/4danyone/waymo/round2/backend'))]
from datasets.FreeTime_dataset import FreeTimeParser
from freetime_ops import positions_at_time, temporal_opacity, validate_vanilla_checkpoint
from gsplat import rasterization
from video_io import MP4Writer


@torch.inference_mode()
def main():
    label=os.environ.get('FDANYONE_ASSET_NAME','phone_full')
    assert label.replace('_','').isalnum()
    out=R/'outputs'/label; ckpt=out/'ckpts/ckpt_4999.pt'
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('4','5','6','7')
    torch.cuda.init()
    expected=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    assert str(torch.cuda.get_device_properties(0).uuid)==expected
    state=torch.load(ckpt,map_location='cuda',weights_only=False)
    validate_vanilla_checkpoint(state)
    assert state['n_frames']==121 and state['step']==4999 and state['use_velocity']
    s=state['splats']; ply=out/'ckpt_4999.ftgs.ply'; vertices=PlyData.read(ply)['vertex'].data
    assert len(vertices)==len(s['means'])
    for prefix,key in [('velocity','velocities'),('scale','scales'),('rot','quats')]:
        decoded=np.stack([vertices[f'{prefix}_{i}'] for i in range(s[key].shape[1])],-1)
        assert np.allclose(decoded,s[key].cpu().numpy(),atol=1e-6)
    assert np.allclose(vertices['time'],s['times'].cpu().numpy().ravel())
    assert np.allclose(vertices['log_duration'],s['durations'].cpu().numpy().ravel())
    parser=FreeTimeParser(str(R/'data'/label),normalize=True,start_frame=0,end_frame=121)
    views=len(parser.camtoworlds);assert views in (6,24)
    selected=[0,4,8,12,16,20] if views==24 else list(range(6))
    poses=torch.as_tensor(parser.camtoworlds[selected],device='cuda',dtype=torch.float32)
    Ks=torch.as_tensor(np.stack([parser.Ks_dict[i+1] for i in selected]),device='cuda',dtype=torch.float32)
    preview=out/'preview';preview.mkdir(exist_ok=True)
    writer=MP4Writer(str(preview/'fixed_views.mp4'),25,1056,1280)
    sheet=Image.new('RGB',(1056,640*3),'white'); draw=ImageDraw.Draw(sheet)
    hashes=[];changes=[];previous=None
    try:
        for frame in range(121):
            t=frame/120
            means=positions_at_time(s['means'],s['velocities'],s['times'],t,True)
            opacity=(s['opacities'].sigmoid()*temporal_opacity(s['times'],s['durations'],t)).clamp_min(1e-4)
            rgb,alpha,_=rasterization(means=means,quats=s['quats'],scales=s['scales'].exp(),opacities=opacity,
                colors=torch.cat([s['sh0'],s['shN']],1),viewmats=torch.linalg.inv(poses),Ks=Ks,width=352,height=640,
                sh_degree=3,packed=True,near_plane=.01,far_plane=1e10,rasterize_mode='classic')
            images=((rgb+1-alpha).clamp(0,1)*255).round().byte().cpu().numpy()
            hashes.append(hashlib.sha256(images[0].tobytes()).hexdigest())
            if previous is not None:changes.append(float(np.abs(images[0].astype(float)-previous).mean()))
            previous=images[0].astype(float)
            tiled=np.concatenate([np.concatenate(images[:3],axis=1),np.concatenate(images[3:],axis=1)],axis=0)
            writer.append_data(tiled)
            if frame in (0,60,120):
                # Three fixed cameras at three times; cameras never move.
                row=(0,60,120).index(frame)
                for col,index in enumerate((0,2,4)):
                    sheet.paste(Image.fromarray(images[index]),(col*352,row*640))
                    draw.text((col*352+8,row*640+8),f'view {selected[index]} t={frame/25:.2f}s',fill='black')
    finally:
        writer.close()
    sheet.save(preview/'time_views.jpg',quality=94)
    videos={}
    for path in (preview/'fixed_views.mp4',out/'videos/traj_4d_step4999.mp4'):
        reader=imageio.get_reader(path);meta=reader.get_meta_data();count=0
        for image in reader:count+=1
        reader.close();assert count==121 and meta['fps']==25
        videos[str(path.relative_to(out))]={'decoded_frames':count,'fps':meta['fps'],'size':meta['size']}
    assert len(set(hashes))>100
    speed=s['velocities'].norm(dim=1)
    result={'status':'trained_rendered_exported_validated','optimization_steps':5000,'gaussians':len(vertices),
        'checkpoint':str(ckpt),'animated_asset':str(ply),'asset_bytes':ply.stat().st_size,'sha256':hashlib.sha256(ply.read_bytes()).hexdigest(),
        'physical_gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),'cuda_uuid':str(torch.cuda.get_device_properties(0).uuid),'frames':121,'sample_span_seconds':4.8,
        'observation_views':views,'fixed_preview_views':selected,
        'videos':videos,'ply_parameter_roundtrip':True,'fixed_view_unique_frames':len(set(hashes)),
        'fixed_view_mean_adjacent_pixel_change':float(np.mean(changes)),
        'velocity_norm_median_normalized_scene_units':float(speed.median()),
        'velocity_norm_max_normalized_scene_units':float(speed.max()),
        'scope':'Animated reconstruction of generated multiview observations, not measured Waymo novel-view truth. Perceptual quality requires separate review.',
        'world_transform':parser.transform.tolist()}
    (preview/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    source=Path(os.environ.get('FDANYONE_ASSET_SOURCE',str(R.parent/'outputs/turbo')))
    mapping=json.loads((source.parents[1]/'inputs/frame_mapping.json').read_text())
    asset={'format':'FreeTimeGS Vanilla animated PLY','asset':ply.name,'checkpoint':str(ckpt.relative_to(out)),
        'source_scene':mapping['scene'],'source_track':mapping['track_id'],'source_split':mapping.get('source_split','train'),
        'frames':121,'fps':25,'sample_span_seconds':4.8,'encoded_duration_seconds':4.84,'time_units':'normalized [0,1], seconds=4.8*t',
        'duration_activation':'exp(log_duration)','velocity_units':'normalized scene distance per normalized time',
        'root_local_to_asset':parser.transform.tolist(),'asset_to_root_local':np.linalg.inv(parser.transform).tolist(),
        'world_frame':'root_local_human_world','root_motion':'root_motion.npz','original_motion':'gvhmr_gravity_aligned_y_up (provenance; not directly asset coordinates)',
        'waymo_world_placement':'not calibrated','observation_views':views,'asset_optimization_steps':5000,
        'model_retraining':False,'backend':'Independent FreeTimeGsVanilla with full-time rendering patch',
        'quality_status':'technical_validation_passed_visual_review_pending'}
    shutil.copy2(source/'root_motion.npz',out/'root_motion.npz')
    shutil.copy2(source/'root_local.json',out/'root_local.json')
    asset['root_motion_sha256']=hashlib.sha256((out/'root_motion.npz').read_bytes()).hexdigest()
    (out/'asset.json').write_text(json.dumps(asset,indent=2)+'\n')


if __name__=='__main__':main()
